#!/usr/bin/env python3
"""
generate_colmap_poses.py
------------------------
Reads panorama_metadata.json (written by fetch_streetview_tiles.py) and
generates COLMAP-compatible cameras.txt / images.txt with accurate camera
positions and a PINHOLE camera model for perspective tiles.

Usage:
    python generate_colmap_poses.py
    (run AFTER fetch_streetview_tiles.py, BEFORE run_colmap.sh)
"""

import os
import json
import math
from pathlib import Path
import numpy as np

# =============================================================================
# CONFIG
# =============================================================================
METADATA_FILE = "panorama_metadata.json"
COLMAP_DIR    = "colmap_workspace"
SPARSE_DIR    = os.path.join(COLMAP_DIR, "sparse", "0")

TILE_SIZE     = 512   # Street View tiles are always 512x512px

# =============================================================================
# HELPERS
# =============================================================================

def tile_focal_length(zoom: int) -> float:
    """
    Compute focal length in pixels for a Street View tile.
    
    Street View tiles are gnomonic projections. At zoom level Z,
    the full 360° equirectangular is divided into 2^Z columns.
    Each tile covers 360 / 2^Z degrees horizontally.
    
    focal_length = (TILE_SIZE / 2) / tan(hfov / 2)
    """
    num_x_tiles = 2 ** zoom
    hfov_deg = 360.0 / num_x_tiles          # horizontal FOV per tile
    hfov_rad = math.radians(hfov_deg)
    focal = (TILE_SIZE / 2.0) / math.tan(hfov_rad / 2.0)
    return focal


def tile_heading(base_heading: float, tile_x: int, zoom: int) -> float:
    """
    Compute the compass heading a specific tile column is facing.
    tile_x=0 is due North (0°), increasing clockwise.
    """
    num_x_tiles = 2 ** zoom
    deg_per_tile = 360.0 / num_x_tiles
    tile_heading = (base_heading + (tile_x - num_x_tiles / 2) * deg_per_tile) % 360
    return tile_heading


def heading_pitch_to_quaternion(heading_deg: float, pitch_deg: float = 0.0):
    """
    Convert heading + pitch to COLMAP world-to-camera quaternion.
    Heading: degrees clockwise from North.
    Pitch: degrees up from horizontal (usually 0 for facade tiles).
    """
    # Convert to math convention: heading clockwise from North → CCW from East
    yaw = math.radians(-heading_deg + 90)
    pitch = math.radians(pitch_deg)

    # Yaw rotation (around Z)
    qw_y = math.cos(yaw / 2)
    qz_y = math.sin(yaw / 2)

    # Pitch rotation (around X)  
    qw_p = math.cos(pitch / 2)
    qx_p = math.sin(pitch / 2)

    # Combine: q = q_yaw * q_pitch
    qw = qw_y * qw_p
    qx = qw_y * qx_p
    qy = qz_y * qx_p  # cross terms
    qz = qz_y * qw_p

    # Normalize
    norm = math.sqrt(qw**2 + qx**2 + qy**2 + qz**2)
    return qw/norm, qx/norm, qy/norm, qz/norm


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
        print("[ERROR] No tiles in metadata file.")
        exit(1)

    Path(SPARSE_DIR).mkdir(parents=True, exist_ok=True)

    ref_lat = metadata[0]["lat"]
    ref_lng = metadata[0]["lng"]

    # Compute camera intrinsics from zoom level
    zoom = metadata[0]["zoom"]
    focal = tile_focal_length(zoom)
    cx = TILE_SIZE / 2.0
    cy = TILE_SIZE / 2.0

    print(f"Generating COLMAP pose priors for {len(metadata)} tiles...")
    print(f"Reference point: ({ref_lat:.5f}, {ref_lng:.5f})")
    print(f"Camera model: PINHOLE — focal={focal:.1f}px, {TILE_SIZE}x{TILE_SIZE}")

    # -------------------------------------------------------------------------
    # cameras.txt — single PINHOLE camera (all tiles same intrinsics)
    # PINHOLE params: fx fy cx cy
    # -------------------------------------------------------------------------
    cameras_path = os.path.join(SPARSE_DIR, "cameras.txt")
    with open(cameras_path, "w") as f:
        f.write("# Camera list with one line of data per camera:\n")
        f.write("#   CAMERA_ID, MODEL, WIDTH, HEIGHT, PARAMS[]\n")
        f.write(f"1 PINHOLE {TILE_SIZE} {TILE_SIZE} {focal:.4f} {focal:.4f} {cx:.4f} {cy:.4f}\n")

    print(f"[OK] cameras.txt — PINHOLE {TILE_SIZE}x{TILE_SIZE} f={focal:.1f}")

    # -------------------------------------------------------------------------
    # images.txt — one entry per tile with pose derived from GPS + tile position
    # -------------------------------------------------------------------------
    images_path = os.path.join(SPARSE_DIR, "images.txt")

    with open(images_path, "w") as f:
        f.write("# Image list with two lines of data per image:\n")
        f.write("#   IMAGE_ID, QW, QX, QY, QZ, TX, TY, TZ, CAMERA_ID, NAME\n")
        f.write("#   POINTS2D[] as (X, Y, POINT3D_ID)\n")

        for i, tile in enumerate(metadata):
            # Camera position from GPS
            tx_world, ty_world, tz_world = latlon_to_xyz(
                tile["lat"], tile["lng"], ref_lat, ref_lng
            )

            # Tile-specific heading (each column faces a different direction)
            th = tile_heading(tile["heading"], tile["tile_x"], tile["zoom"])
            qw, qx, qy, qz = heading_pitch_to_quaternion(th)

            # T = -R * C (COLMAP convention)
            R = np.array([
                [1 - 2*qy**2 - 2*qz**2, 2*qx*qy - 2*qz*qw, 2*qx*qz + 2*qy*qw],
                [2*qx*qy + 2*qz*qw,     1 - 2*qx**2 - 2*qz**2, 2*qy*qz - 2*qx*qw],
                [2*qx*qz - 2*qy*qw,     2*qy*qz + 2*qx*qw, 1 - 2*qx**2 - 2*qy**2]
            ])
            C = np.array([tx_world, ty_world, tz_world])
            T = -R.dot(C)

            f.write(
                f"{i+1} {qw:.8f} {qx:.8f} {qy:.8f} {qz:.8f} "
                f"{T[0]:.4f} {T[1]:.4f} {T[2]:.4f} 1 {tile['image_name']}\n"
            )
            f.write("\n")  # empty points2D line — required by COLMAP format

            print(f"  [{i:02d}] {tile['image_name']} — "
                  f"pos=({tx_world:.1f}m, {ty_world:.1f}m) "
                  f"heading={th:.1f}°")

    print(f"[OK] images.txt — {len(metadata)} tiles")

    # points3D.txt — empty, COLMAP fills this during mapping
    with open(os.path.join(SPARSE_DIR, "points3D.txt"), "w") as f:
        f.write("# 3D point list — COLMAP will populate this\n")

    print(f"[OK] points3D.txt — empty")
    print(f"\n  Now run: ./run_colmap.sh")


if __name__ == "__main__":
    main()