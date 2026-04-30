#!/usr/bin/env python3
"""
fetch_streetview_tiles.py
--------------------------
Downloads full equirectangular panoramas using the Street View Tiles API.
Unlike the Static API (which gives cropped perspective crops), this gives
COLMAP the full 360° image with consistent spherical geometry — much better
for photogrammetry.

Workflow:
  1. POST to createSession to get a session token
  2. For each coordinate, get the panoId + real camera position
  3. Download all panorama tiles at zoom level 3 (good resolution/size tradeoff)
  4. Stitch tiles into a single equirectangular image
  5. Save metadata (real lat/lng, heading) for COLMAP pose priors

Usage:
  pip install requests pillow
  python fetch_streetview_tiles.py
"""

import os
import json
import math
import requests
from pathlib import Path
from dotenv import load_dotenv

try:
    from PIL import Image
    import io
except ImportError:
    print("[ERROR] Pillow not installed. Run: pip install pillow")
    exit(1)

load_dotenv()

API_KEY = os.getenv("GOOGLE_MAPS_API_KEY")
if not API_KEY:
    raise RuntimeError("GOOGLE_MAPS_API_KEY not found in .env")

# =============================================================================
# CONFIG
# =============================================================================
OUTPUT_FOLDER = "street_images"
METADATA_FILE = "street_images/panorama_metadata.json"

# Zoom level 3 = 4x2 tiles = good balance of resolution and download size
# Each tile is 512x512px, so full panorama = 2048x1024px
# Zoom level 4 = 8x4 tiles = 4096x2048px (higher quality but 4x more requests)
ZOOM_LEVEL = 3

PATH_COORDINATES = [
    (33.76433, -84.38206),
    (33.76435, -84.38206),
    (33.76437, -84.38206),
    (33.76439, -84.38206),
    (33.76441, -84.38206),
    (33.76443, -84.38206),
    (33.76445, -84.38206),
    (33.76447, -84.38206),
    (33.76449, -84.38206),
    (33.76451, -84.38206),
    (33.76453, -84.38206),
    (33.76455, -84.38206),
]

# =============================================================================
# STEP 1: Get session token
# =============================================================================
def get_session_token(api_key: str) -> str:
    print("Getting session token...")
    url = "https://tile.googleapis.com/v1/createSession"
    payload = {
        "mapType": "streetview",
        "language": "en-US",
        "region": "US"
    }
    r = requests.post(f"{url}?key={api_key}", json=payload)
    r.raise_for_status()
    token = r.json()["session"]
    print(f"[OK] Session token obtained (valid ~2 weeks)")
    return token


# =============================================================================
# STEP 2: Get panoId + real camera position for a coordinate
# =============================================================================
def get_pano_info(api_key: str, session: str, lat: float, lng: float):
    url = "https://tile.googleapis.com/v1/streetview/panoIds"
    payload = {
        "locations": [{"lat": lat, "lng": lng}],
        "radius": 50  # meters search radius
    }
    r = requests.post(f"{url}?session={session}&key={api_key}", json=payload)
    r.raise_for_status()
    data = r.json()

    if not data.get("panoIds"):
        return None, None, None

    pano_id = data["panoIds"][0]

    # Get metadata for this panoId (real position, heading, dimensions)
    meta_url = f"https://tile.googleapis.com/v1/streetview/metadata"
    meta_r = requests.get(
        meta_url,
        params={"session": session, "key": api_key, "panoId": pano_id}
    )
    meta_r.raise_for_status()
    meta = meta_r.json()

    real_lat = meta["location"]["lat"]
    real_lng = meta["location"]["lng"]
    heading  = meta.get("heading", 0)  # compass heading camera is facing

    return pano_id, real_lat, real_lng, heading, meta


# =============================================================================
# STEP 3 & 4: Download tiles and stitch into equirectangular panorama
# =============================================================================
def download_and_stitch(api_key: str, session: str, pano_id: str, zoom: int) -> Image.Image:
    """
    Downloads all tiles for a panorama at the given zoom level and stitches
    them into a single equirectangular image.

    At zoom level Z:
      - num_x_tiles = 2^Z
      - num_y_tiles = 2^(Z-1)
    Each tile is 512x512px.
    """
    num_x = 2 ** zoom
    num_y = 2 ** (zoom - 1)
    tile_size = 512

    full_width  = num_x * tile_size
    full_height = num_y * tile_size
    panorama = Image.new("RGB", (full_width, full_height))

    print(f"    Downloading {num_x}x{num_y} tiles ({full_width}x{full_height}px)...")

    for y in range(num_y):
        for x in range(num_x):
            tile_url = (
                f"https://tile.googleapis.com/v1/streetview/tiles"
                f"/{zoom}/{x}/{y}"
                f"?session={session}&key={api_key}&panoId={pano_id}"
            )
            r = requests.get(tile_url)
            if r.status_code != 200:
                print(f"    [WARN] Tile {x},{y} failed: {r.status_code}")
                continue

            tile_img = Image.open(io.BytesIO(r.content))
            panorama.paste(tile_img, (x * tile_size, y * tile_size))

    return panorama


# =============================================================================
# MAIN
# =============================================================================
def main():
    Path(OUTPUT_FOLDER).mkdir(exist_ok=True)

    session = get_session_token(API_KEY)

    seen_panos = set()
    image_index = 0
    all_metadata = []

    print(f"\nDownloading panoramas for {len(PATH_COORDINATES)} coordinates...")

    for lat, lng in PATH_COORDINATES:
        result = get_pano_info(API_KEY, session, lat, lng)
        if result[0] is None:
            print(f"  No panorama found at ({lat}, {lng}), skipping.")
            continue

        pano_id, real_lat, real_lng, heading, meta = result

        if pano_id in seen_panos:
            print(f"  Skipping duplicate panorama at ({lat}, {lng})")
            continue

        seen_panos.add(pano_id)
        fname = f"facade_{image_index:03d}.jpg"
        fpath = os.path.join(OUTPUT_FOLDER, fname)

        print(f"\n  [{image_index}] panoId={pano_id[:12]}... @ ({real_lat:.5f}, {real_lng:.5f})")

        panorama = download_and_stitch(API_KEY, session, pano_id, ZOOM_LEVEL)
        panorama.save(fpath, "JPEG", quality=95)
        print(f"    Saved: {fname}")

        # Store metadata for COLMAP pose priors
        all_metadata.append({
            "image_name": fname,
            "image_index": image_index,
            "pano_id": pano_id,
            "lat": real_lat,
            "lng": real_lng,
            "heading": heading,
            "image_width": panorama.width,
            "image_height": panorama.height,
        })

        image_index += 1

    # Save metadata JSON for generate_colmap_poses.py to use
    with open(METADATA_FILE, "w") as f:
        json.dump(all_metadata, f, indent=2)

    print(f"\n{'='*50}")
    print(f"  Done! Downloaded {image_index} unique panoramas.")
    print(f"  Metadata saved to: {METADATA_FILE}")
    print(f"\n  Next steps:")
    print(f"    python generate_colmap_poses.py   # uses metadata for pose priors")
    print(f"    ./run_colmap.sh")
    print(f"{'='*50}")


if __name__ == "__main__":
    main()