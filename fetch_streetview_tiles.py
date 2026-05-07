#!/usr/bin/env python3
"""
fetch_streetview_tiles.py
--------------------------
Downloads Google Street View tiles along a dense 4-meter path to ensure 
high overlap for COLMAP Structure from Motion (SfM).
"""

import os
import json
import math
import requests
from pathlib import Path
from dotenv import load_dotenv
import py360convert
import numpy as np


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

ZOOM_LEVEL = 3

# We will only use the FIRST and LAST coordinate to draw our line.
# The script will automatically drop pins every 4 meters between them.
PATH_COORDINATES = [
    (33.76433, -84.38210), # Start (South end of building)
    (33.76510, -84.38210), # Final point (North end of building)
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
    print(f"[OK] Session token obtained")
    return token

# =============================================================================
# STEP 2: Interpolate coordinates to get dense Pano IDs
# =============================================================================
def get_dense_pano_ids(api_key, start_coord, end_coord, step_meters=4.0):
    print(f"\nCalculating dense path every {step_meters}m...")
    lat1, lon1 = start_coord
    lat2, lon2 = end_coord
    
    avg_lat = math.radians((lat1 + lat2) / 2.0)
    lat_diff_meters = (lat2 - lat1) * 111139.0
    lon_diff_meters = (lon2 - lon1) * (111139.0 * math.cos(avg_lat))
    total_distance = math.sqrt(lat_diff_meters**2 + lon_diff_meters**2)
    
    num_steps = max(int(total_distance / step_meters), 1)
    print(f"Total distance: ~{total_distance:.1f}m. Sampling {num_steps} points...")

    unique_panos = []
    
    for i in range(num_steps + 1):
        fraction = i / float(num_steps)
        current_lat = lat1 + (lat2 - lat1) * fraction
        current_lon = lon1 + (lon2 - lon1) * fraction
        
        url = f"https://maps.googleapis.com/maps/api/streetview/metadata"
        params = {
            "location": f"{current_lat},{current_lon}",
            "key": api_key,
            "radius": 15
        }
        
        response = requests.get(url, params=params)
        data = response.json()
        
        if data.get("status") == "OK":
            pano_id = data["pano_id"]
            if pano_id not in unique_panos:
                unique_panos.append(pano_id)
                print(f"  Found new Pano ID: {pano_id} (Step {i})")
        else:
            print(f"  No pano found near step {i} ({current_lat:.5f}, {current_lon:.5f})")

    print(f"[OK] Found {len(unique_panos)} tightly spaced panoramas.")
    return unique_panos

# =============================================================================
# STEP 3: Get real metadata for a specific Pano ID
# =============================================================================
def get_pano_metadata_by_id(api_key: str, session: str, pano_id: str):
    """Fetches the exact real-world lat, lng, and heading for a given pano_id"""
    meta_url = f"https://tile.googleapis.com/v1/streetview/metadata?session={session}&key={api_key}&panoId={pano_id}"
    meta_r = requests.get(meta_url)
    if meta_r.status_code != 200:
        return None
    
    meta = meta_r.json()
    if "lat" not in meta or "lng" not in meta:
        return None
    
    return meta.get("lat"), meta.get("lng"), meta.get("heading", 0)

# =============================================================================
# STEP 4: Download, stitch, and crop Street View tiles
# =============================================================================
def download_stitch_and_crop(api_key: str, session: str, pano_id: str, 
                             output_folder: str, image_index: int, 
                             car_heading: float):
    """
    Downloads all zoom 3 tiles, stitches them into a full panorama, 
    and extracts a perfect 90-degree rectilinear perspective crop looking directly at the facade.
    """
    zoom = 3
    num_x = 2 ** zoom # 8
    num_y = 2 ** (zoom - 1) # 4
    
    # 1. Create a blank image for the full panorama (2048 x 1024 for Zoom 3)
    pano_img = Image.new('RGB', (num_x * 512, num_y * 512))
    
    print(f"    Downloading and stitching {num_x * num_y} tiles...")
    
    # 2. Download and paste all tiles
    for y in range(num_y):
        for x in range(num_x):
            tile_url = (
                f"https://tile.googleapis.com/v1/streetview/tiles"
                f"/{zoom}/{x}/{y}"
                f"?session={session}&key={api_key}&panoId={pano_id}"
            )
            r = requests.get(tile_url)
            if r.status_code == 200:
                tile_img = Image.open(io.BytesIO(r.content))
                pano_img.paste(tile_img, (x * 512, y * 512))

    print(f"Tile size: {tile_img.size}")
    
    # 3. Calculate the correct view angle
    # TARGET_BEARING is the compass direction of the building from the street. 
    # West = 270. East = 90. North = 0. South = 180.
    TARGET_BEARING = 270.0   #West - my building
    heading_deg = TARGET_BEARING - current_heading
    heading_deg = (heading_deg + 180) % 360 - 180  # Normalize to [-180, 180]
    
    # The center of the panorama is the direction the car was facing (car_heading).
    # We need to calculate how many degrees to turn the camera to look at the building.
    yaw_deg = TARGET_BEARING - car_heading
    
    # Normalize yaw to be between -180 and 180
    yaw_deg = (yaw_deg + 180) % 360 - 180
    
    # Set pitch to look slightly up at the balconies (0 is horizon, positive is up)
    pitch_deg = 0.0 

    # 4. Extract the perspective crop using py360convert
    print(f"    Extracting perspective crop (Yaw: {yaw_deg:.1f}°, Pitch: {pitch_deg:.1f}°)")
    pano_np = np.array(pano_img)
    
    # We use a 90 degree FOV to ensure massive overlap between 4-meter steps
    crop_np = py360convert.e2p(
        pano_np, 
        fov_deg=90, 
        u_deg=yaw_deg, 
        v_deg=pitch_deg, 
        out_hw=(1024, 1024), # Output resolution
        in_rot_deg=0
    )
    
    crop_img = Image.fromarray(crop_np)
    
    # 5. Save the final perspective image
    fname = f"facade_perspective_{image_index:03d}.jpg"
    fpath = os.path.join(output_folder, fname)
    crop_img.save(fpath, quality=95)
    
    return fname

def download_raw_tiles(api_key, session, pano_id, image_index, heading):
    # Focusing on the building to the West (Bearing 270)
    # At Zoom 3: Tile X=6 is roughly 270 degrees
    # We take a 3x2 grid of tiles to cover the facade
    x_range = [5, 6, 7] 
    y_range = [1, 2] # Middle-upper rows (avoiding the car/road)

    saved_files = []
    for y in y_range:
        for x in x_range:
            tile_url = f"https://tile.googleapis.com/v1/streetview/tiles/3/{x}/{y}?session={session}&key={api_key}&panoId={pano_id}"
            r = requests.get(tile_url)
            if r.status_code == 200:
                fname = f"pano_{image_index:03d}_tile_{x}_{y}.jpg"
                with open(os.path.join(OUTPUT_FOLDER, fname), "wb") as f:
                    f.write(r.content)
                
                # We save the X and Y so we can calculate the offset later
                saved_files.append({
                    "filename": fname,
                    "x": x,
                    "y": y,
                    "pano_heading": heading
                })
    return saved_files

# =============================================================================
# STEP 4: Download the tiles
# =============================================================================
def download_tiles_separately(api_key: str, session: str, pano_id: str,
                               zoom: int, output_folder: str, image_index: int,
                               heading: float):
    
    x_range = [3, 4, 5, 6, 7]   # West-facing columns
    y_range = range(1, 5)       # Skip sky/ground extremes

    saved = []
    for y in y_range:
        for x in x_range:
            tile_url = (
                f"https://tile.googleapis.com/v1/streetview/tiles"
                f"/{zoom}/{x}/{y}"
                f"?session={session}&key={api_key}&panoId={pano_id}"
            )
            r = requests.get(tile_url)
            if r.status_code != 200:
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
                "heading": heading,
            })

    return saved

# =============================================================================
# MAIN LOGIC
# =============================================================================
def main():
    Path(OUTPUT_FOLDER).mkdir(exist_ok=True)
    session = get_session_token(API_KEY)

    # 1. Grab start and end points from config
    start_coord = PATH_COORDINATES[0]
    end_coord = PATH_COORDINATES[-1]

    # 2. Generate the dense list of pano IDs (~4 meters apart)
    pano_list = get_dense_pano_ids(API_KEY, start_coord, end_coord, step_meters=4.0)

    all_metadata = []
    print(f"\nDownloading and cropping {len(pano_list)} panoramas...")

    # 3. Loop over the dense list of panoramas
    for image_index, pano_id in enumerate(pano_list):
        
        # Get the actual GPS data for this pano
        meta_result = get_pano_metadata_by_id(API_KEY, session, pano_id)
        if meta_result is None:
            print(f"  [{image_index}] Skipping {pano_id[:12]} (No metadata found)")
            continue
            
        real_lat, real_lng, heading = meta_result
        print(f"  [{image_index}] Processing {pano_id[:12]}... @ ({real_lat:.5f}, {real_lng:.5f})")

        # Call the stitch and crop function (NO raw tiles!)
        saved_filename = download_stitch_and_crop(
            API_KEY, session, pano_id, OUTPUT_FOLDER, image_index, heading
        )

        # Save metadata (just one entry per location)
        all_metadata.append({
            "image_name":  saved_filename,
            "pano_index":  image_index,
            "pano_id":     pano_id,
            "lat":         real_lat,
            "lng":         real_lng,
            "car_heading": heading,
        })

    # Save to JSON
    with open(METADATA_FILE, "w") as f:
        json.dump(all_metadata, f, indent=2)

    print(f"\n{'='*50}")
    print(f"  Done! {len(pano_list)} panoramas successfully cropped.")
    print(f"  Metadata saved to: {METADATA_FILE}")
    print(f"{'='*50}")


if __name__ == "__main__":
    main()