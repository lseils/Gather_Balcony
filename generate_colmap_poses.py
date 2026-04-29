#!/usr/bin/env python3
"""
generate_colmap_poses.py
------------------------
Street View images lack EXIF GPS data, so COLMAP can't determine camera
positions automatically. This script creates a COLMAP-compatible text model
with known camera poses derived from the coordinates and headings used in
fetch_streetview.py, giving COLMAP a strong prior to work from.

Usage:
    python generate_colmap_poses.py
    (run BEFORE run_colmap.sh)
"""

import os
import math
import struct
import sqlite3

# =============================================================================
# CONFIG — must match fetch_streetview.py exactly
# =============================================================================
OUTPUT_FOLDER   = "street_images"
COLMAP_DIR      = "colmap_workspace"
SPARSE_DIR      = os.path.join(COLMAP_DIR, "sparse", "0")

PATH_COORDINATES = [
    (33.76433, -84.38209),
    (33.76443, -84.38209),
    (33.76453, -84.38209),
    (33.76463, -84.38209),
]

FACADE_HEADINGS = [230, 245, 260, 275, 295]

OPTIMAL_PITCH = 20   # degrees
OPTIMAL_FOV   = 60   # degrees
IMAGE_SIZE    = 640  # pixels (square)

# =============================================================================
# HELPERS
# =============================================================================

def heading_pitch_to_quaternion(heading_deg, pitch_deg):
    """
    Convert Street View heading + pitch to a rotation quaternion (qw, qx, qy, qz).
    Heading: degrees clockwise from North (0=N, 90=E, 180=S, 270=W)
    Pitch:   degrees up from horizontal
    Returns quaternion in COLMAP convention (world-to-camera).
    """
    # Convert to radians
    yaw   = math.radians(-heading_deg + 90)  # Convert to math convention (CCW from East)
    pitch = math.radians(pitch_deg)

    # Rotation around Z (yaw) then X (pitch)
    cy, sy = math.cos(yaw / 2),   math.sin(yaw / 2)
    cp, sp = math.cos(pitch / 2), math.sin(pitch / 2)

    # Yaw quaternion (around Z axis)
    qw_y, qx_y, qy_y, qz_y = cy, 0, 0, sy
    # Pitch quaternion (around X axis)
    qw_p, qx_p, qy_p, qz_p = cp, sp, 0, 0

    # Combined: pitch * yaw
    qw = qw_y * qw_p - qx_y * qx_p - qy_y * qy_p - qz_y * qz_p
    qx = qw_y * qx_p + qx_y * qw_p + qy_y * qz_p - qz_y * qy_p
    qy = qw_y * qy_p - qx_y * qz_p + qy_y * qw_p + qz_y * qx_p
    qz = qw_y * qz_p + qx_y * qy_p - qy_y * qx_p + qz_y * qw_p

    return qw, qx, qy, qz


def latlon_to_xyz(lat, lng, ref_lat, ref_lng):
    """
    Convert lat/lng to local XYZ in meters relative to a reference point.
    Uses simple flat-earth approximation (fine for small areas like a city block).
    """
    R = 6371000  # Earth radius in meters
    x = math.radians(lng - ref_lng) * R * math.cos(math.radians(ref_lat))
    y = math.radians(lat - ref_lat) * R
    z = 0.0  # Street View is roughly at ground level
    return x, y, z


def fov_to_focal(fov_deg, image_size):
    """Convert field-of-view to focal length in pixels."""
    return (image_size / 2) / math.tan(math.radians(fov_deg / 2))


# =============================================================================
# MAIN
# =============================================================================

def main():
    os.makedirs(SPARSE_DIR, exist_ok=True)

    ref_lat, ref_lng = PATH_COORDINATES[0]
    focal = fov_to_focal(OPTIMAL_FOV, IMAGE_SIZE)
    cx = cy = IMAGE_SIZE / 2.0

    print(f"Reference point: ({ref_lat}, {ref_lng})")
    print(f"Focal length:    {focal:.1f}px  (from FOV={OPTIMAL_FOV}°)")
    print(f"Principal point: ({cx}, {cy})")
    print()

    # -------------------------------------------------------------------------
    # cameras.txt — one shared camera model for all images (SIMPLE_PINHOLE)
    # -------------------------------------------------------------------------
    cameras_path = os.path.join(SPARSE_DIR, "cameras.txt")
    with open(cameras_path, "w") as f:
        f.write("# Camera list with one line of data per camera:\n")
        f.write("#   CAMERA_ID, MODEL, WIDTH, HEIGHT, PARAMS[]\n")
        # SIMPLE_PINHOLE: f, cx, cy
        f.write(f"1 SIMPLE_PINHOLE {IMAGE_SIZE} {IMAGE_SIZE} {focal:.4f} {cx:.4f} {cy:.4f}\n")
    print(f"[OK] cameras.txt written")

    # -------------------------------------------------------------------------
    # images.txt — one entry per image with pose (qw qx qy qz tx ty tz)
    # -------------------------------------------------------------------------
    images_path = os.path.join(SPARSE_DIR, "images.txt")
    image_index = 0
    entries = []

    for lat, lng in PATH_COORDINATES:
        tx, ty, tz = latlon_to_xyz(lat, lng, ref_lat, ref_lng)
        for heading in FACADE_HEADINGS:
            fname = f"facade_{image_index:03d}.jpg"
            qw, qx, qy, qz = heading_pitch_to_quaternion(heading, OPTIMAL_PITCH)

            # Camera translation in world coords
            # COLMAP uses world-to-camera transform: t = -R * T_world
            entries.append((image_index + 1, qw, qx, qy, qz, tx, ty, tz, 1, fname))
            image_index += 1

    with open(images_path, "w") as f:
        f.write("# Image list with two lines of data per image:\n")
        f.write("#   IMAGE_ID, QW, QX, QY, QZ, TX, TY, TZ, CAMERA_ID, NAME\n")
        f.write("#   POINTS2D[] as (X, Y, POINT3D_ID)\n")
        for entry in entries:
            img_id, qw, qx, qy, qz, tx, ty, tz, cam_id, fname = entry
            f.write(f"{img_id} {qw:.8f} {qx:.8f} {qy:.8f} {qz:.8f} "
                    f"{tx:.4f} {ty:.4f} {tz:.4f} {cam_id} {fname}\n")
            f.write("\n")  # Empty line = no 2D points yet (COLMAP will fill these)

    print(f"[OK] images.txt written ({image_index} images)")

    # -------------------------------------------------------------------------
    # points3D.txt — empty (COLMAP will populate this during reconstruction)
    # -------------------------------------------------------------------------
    points_path = os.path.join(SPARSE_DIR, "points3D.txt")
    with open(points_path, "w") as f:
        f.write("# 3D point list with one line of data per point:\n")
        f.write("#   POINT3D_ID, X, Y, Z, R, G, B, ERROR, TRACK[] as (IMAGE_ID, POINT2D_IDX)\n")
    print(f"[OK] points3D.txt written (empty — COLMAP will fill this)")

    print()
    print("=" * 50)
    print("  Pose priors ready! Now run:")
    print()
    print("  ./run_colmap.sh")
    print()
    print("  The mapper will use these poses as initialization,")
    print("  which should register far more images.")
    print("=" * 50)


if __name__ == "__main__":
    main()