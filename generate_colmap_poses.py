#!/usr/bin/env python3
"""
generate_colmap_poses.py
------------------------
Reads panorama_metadata.json (written by fetch_streetview_tiles.py) and
generates COLMAP-compatible cameras.txt / images.txt with accurate camera
positions and a SPHERICAL camera model for equirectangular panoramas.

Usage:
    python generate_colmap_poses.py
    (run AFTER fetch_streetview_tiles.py, BEFORE run_colmap.sh)
"""

import os
import json
import math
from pathlib import Path

# =============================================================================
# CONFIG
# =============================================================================
METADATA_FILE = "street_images/panorama_metadata.json"
COLMAP_DIR    = "colmap_workspace"
SPARSE_DIR    = os.path.join(COLMAP_DIR, "sparse", "0")

# =============================================================================
# HELPERS
# =============================================================================

def heading_to_quaternion(heading_deg: float):
    """
    Convert panorama compass heading to COLMAP quaternion (world-to-camera).
    Heading: degrees clockwise from North.
    """
    yaw = math.radians(-heading_deg + 90)  # convert to math convention
    qw = math.cos(yaw / 2)
    qx = 0.0
    qy = 0.0
    qz = math.sin(yaw / 2)
    return qw, qx, qy, qz


def latlon_to_xyz(lat, lng, ref_lat, ref_lng):
    """Convert lat/lng to local XYZ in meters relative to reference point."""
    R = 6371000
    x = math.radians(lng - ref_lng) * R * math.cos(math.radians(ref_lat))
    y = math.radians(lat - ref_lat) * R
    z = 0.0
    return x, y, z


# =============================================================================
# MAIN
# =============================================================================
def main():
    if not Path(METADATA_FILE).exists():
        print(f"[ERROR] {METADATA_FILE} not found.")
        print("  Run fetch_streetview_tiles.py first.")
        exit(1)

    with open(METADATA_FILE) as f:
        metadata = json.load(f)

    if not metadata:
        print("[ERROR] No panoramas in metadata file.")
        exit(1)

    Path(SPARSE_DIR).mkdir(parents=True, exist_ok=True)

    ref_lat = metadata[0]["lat"]
    ref_lng = metadata[0]["lng"]

    print(f"Generating COLMAP pose priors for {len(metadata)} panoramas...")
    print(f"Reference point: ({ref_lat}, {ref_lng})")
    print(f"Camera model: SPHERICAL (equirectangular panorama)")

    # -------------------------------------------------------------------------
    # cameras.txt — SPHERICAL model for equirectangular panoramas
    # COLMAP's SPHERICAL model has no parameters beyond image dimensions
    # -------------------------------------------------------------------------
    cameras_path = os.path.join(SPARSE_DIR, "cameras.txt")
    img_w = metadata[0]["image_width"]
    img_h = metadata[0]["image_height"]

    with open(cameras_path, "w") as f:
        f.write("# Camera list with one line of data per camera:\n")
        f.write("#   CAMERA_ID, MODEL, WIDTH, HEIGHT, PARAMS[]\n")
        # SPHERICAL model has no distortion params — just width and height
        f.write(f"1 SPHERICAL {img_w} {img_h}\n")

    print(f"[OK] cameras.txt — SPHERICAL {img_w}x{img_h}")

    # -------------------------------------------------------------------------
    # images.txt — one entry per panorama with real GPS-derived position
    # -------------------------------------------------------------------------
    images_path = os.path.join(SPARSE_DIR, "images.txt")

    with open(images_path, "w") as f:
        f.write("# Image list with two lines of data per image:\n")
        f.write("#   IMAGE_ID, QW, QX, QY, QZ, TX, TY, TZ, CAMERA_ID, NAME\n")
        f.write("#   POINTS2D[] as (X, Y, POINT3D_ID)\n")

        for i, pano in enumerate(metadata):
            tx, ty, tz = latlon_to_xyz(pano["lat"], pano["lng"], ref_lat, ref_lng)
            qw, qx, qy, qz = heading_to_quaternion(pano["heading"])

            f.write(
                f"{i+1} {qw:.8f} {qx:.8f} {qy:.8f} {qz:.8f} "
                f"{tx:.4f} {ty:.4f} {tz:.4f} 1 {pano['image_name']}\n"
            )
            f.write("\n")

            print(f"  [{i}] {pano['image_name']} — pos=({tx:.1f}m, {ty:.1f}m) heading={pano['heading']:.1f}°")

    print(f"[OK] images.txt — {len(metadata)} images")

    # -------------------------------------------------------------------------
    # points3D.txt — empty, COLMAP fills this
    # -------------------------------------------------------------------------
    with open(os.path.join(SPARSE_DIR, "points3D.txt"), "w") as f:
        f.write("# 3D point list — COLMAP will populate this\n")

    print(f"[OK] points3D.txt — empty")
    print(f"\n  Now run: ./run_colmap.sh")


if __name__ == "__main__":
    main()