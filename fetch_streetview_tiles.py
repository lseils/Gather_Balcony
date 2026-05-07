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
METADATA_FILE = "panorama_metadata.json"

# Zoom level 3 = 4x2 tiles = good balance of resolution and download size
# Each tile is 512x512px, so full panorama = 2048x1024px
# Zoom level 4 = 8x4 tiles = 4096x2048px (higher quality but 4x more requests)
ZOOM_LEVEL = 3

PATH_COORDINATES = [
    (33.76433, -84.38210), # Start
    (33.76450, -84.38210), # ~18 meters away
    (33.76470, -84.38210), # ~22 meters away
    (33.76490, -84.38210), # ~22 meters away
    (33.76510, -84.38210), # Final point
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
        "radius": 200
    }
    
    # Step 1: Get the Pano ID
    r = requests.post(f"{url}?session={session}&key={api_key}", json=payload)
    r.raise_for_status()
    data = r.json()

    if not data.get("panoIds") or data["panoIds"][0] == "":
        return None

    pano_id = data["panoIds"][0]

    # Step 2: Get metadata for this panoId
    meta_url = "https://tile.googleapis.com/v1/streetview/metadata"
    # We append parameters to the URL string to ensure the API parses them correctly
    full_meta_url = f"{meta_url}?session={session}&key={api_key}&panoId={pano_id}"

    meta_r = requests.get(full_meta_url)
    meta_r.raise_for_status()
    meta = meta_r.json()

    # Step 3: Extract flattened keys (lat, lng, heading) directly from the meta dict
    if "lat" not in meta or "lng" not in meta:
        print(f"DEBUG: Pano ID {pano_id} found but missing coordinate keys.")
        return None
    
    real_lat = meta.get("lat")
    real_lng = meta.get("lng")
    heading  = meta.get("heading", 0)

    # Return the data so the main loop can start download_and_stitch
    return pano_id, real_lat, real_lng, heading, meta

# =============================================================================
# STEP 3 & 4: Download tiles and DO NOT STITCH — instead, save them separately as perspective crops
# =============================================================================
def download_tiles_separately(api_key: str, session: str, pano_id: str, 
                               zoom: int, output_folder: str, image_index: int):
    """
    Downloads tiles individually as perspective crops instead of stitching
    into equirectangular. Each tile is a valid gnomonic (perspective) image
    that COLMAP can process natively with a pinhole camera model.
    
    At zoom 3: 8x4 = 32 tiles per panorama
    At zoom 4: 16x8 = 128 tiles per panorama (overkill)
    
    We only save the tiles facing the facade (center columns) to avoid
    giving COLMAP sky, ground, and behind-camera tiles.
    """
    num_x = 2 ** zoom
    num_y = 2 ** (zoom - 1)

    saved = []

    # Only grab horizontally centered tiles facing the facade heading
    # For a facade-facing shot, the center 3 columns out of num_x are enough
    # and vertically skip the top/bottom rows (sky and ground)
    x_center = num_x // 2
    x_range = range(max(0, x_center - 1), min(num_x, x_center + 2))  # 3 columns
    y_range = range(1, num_y - 1)  # skip top and bottom rows

    for y in y_range:
        for x in x_range:
            tile_url = (
                f"https://tile.googleapis.com/v1/streetview/tiles"
                f"/{zoom}/{x}/{y}"
                f"?session={session}&key={api_key}&panoId={pano_id}"
            )
            r = requests.get(tile_url)
            if r.status_code != 200:
                print(f"    [WARN] Tile {x},{y} failed: {r.status_code}")
                continue

            fname = f"facade_{image_index:03d}_tile_{x}_{y}.jpg"
            fpath = os.path.join(output_folder, fname)
            with open(fpath, "wb") as f:
                f.write(r.content)

            saved.append({
                "image_name": fname,
                "pano_index": image_index,
                "tile_x": x,
                "tile_y": y,
                "zoom": zoom,
            })

    return saved


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

        if result is None:
            print(f"  No panorama found at ({lat}, {lng}), skipping.")
            continue

        pano_id, real_lat, real_lng, heading, meta = result

        if pano_id in seen_panos:
            print(f"  Skipping duplicate panorama at ({lat}, {lng})")
            continue

        seen_panos.add(pano_id)

        print(f"\n  [{image_index}] panoId={pano_id[:12]}... @ ({real_lat:.5f}, {real_lng:.5f})")

        # FIX 1: pass output_folder and image_index
        tiles = download_tiles_separately(
            API_KEY, session, pano_id, ZOOM_LEVEL, OUTPUT_FOLDER, image_index
        )
        print(f"    Saved {len(tiles)} tiles")

        # FIX 2: metadata per tile, not per panorama
        for tile in tiles:
            all_metadata.append({
                "image_name":  tile["image_name"],
                "pano_index":  tile["pano_index"],
                "tile_x":      tile["tile_x"],
                "tile_y":      tile["tile_y"],
                "zoom":        tile["zoom"],
                "pano_id":     pano_id,
                "lat":         real_lat,
                "lng":         real_lng,
                "heading":     heading,
            })

        image_index += 1

    with open(METADATA_FILE, "w") as f:
        json.dump(all_metadata, f, indent=2)

    total_tiles = len(all_metadata)
    print(f"\n{'='*50}")
    print(f"  Done! {image_index} panoramas → {total_tiles} tiles saved.")
    print(f"  Metadata saved to: {METADATA_FILE}")
    print(f"\n  Next steps:")
    print(f"    ./run_colmap.sh")
    print(f"{'='*50}")

if __name__ == "__main__":
    main()