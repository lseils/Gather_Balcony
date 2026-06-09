#!/usr/bin/env python3
"""
crop_perspective.py
--------------------
Reads raw equirectangular panoramas saved by fetch_streetview_tiles.py and
extracts three rectilinear perspective crops per panorama:
  - center: looking directly at the building facade
  - left:   30° left of the facade
  - right:  30° right of the facade

Correct rotation order: yaw (heading) first, then pitch.
The crop heading is TARGET_BEARING - car_heading, NOT car_heading directly.

Outputs:
  perspective_crops/pano_000_center.jpg  ← facing facade directly
  perspective_crops/pano_000_left.jpg    ← 30° left of facade
  perspective_crops/pano_000_right.jpg   ← 30° right of facade
"""

import os
import json
import numpy as np
import cv2
from pathlib import Path

# =============================================================================
# CONFIG — tune these without re-downloading anything
# =============================================================================
INPUT_DIR   = "street_images"           # raw equirectangular panoramas
OUTPUT_DIR  = "perspective_crops"       # COLMAP input folder
META_FILE   = "panorama_metadata.json"

# Compass bearing the building faces FROM the street.
# West = 270, East = 90, North = 0, South = 180.
TARGET_BEARING = 250.0

# Offsets (degrees) from TARGET_BEARING for the three crops.
# 0° = center, -30° = left, +30° = right.
CROP_OFFSETS = {
    "left":   -30.0,
    "center":   0.0,
    "right":  +30.0,
}

FOV_DEG   = 90     # horizontal FOV in degrees. 90° gives strong overlap at 4m spacing.
PITCH_DEG = 16.0   # degrees UP from horizon. Positive = tilt up toward upper floors.
OUT_W     = 2048   # output width  (pixels)
OUT_H     = 1536   # output height (pixels)


# =============================================================================
# Core projection: equirectangular → rectilinear perspective
# =============================================================================
def equirect_to_perspective(pano: np.ndarray,
                             fov_deg: float,
                             yaw_deg: float,
                             pitch_deg: float,
                             out_w: int,
                             out_h: int) -> np.ndarray:
    """
    Extract a rectilinear perspective crop from an equirectangular panorama.

    Args:
        pano:       H×W×3 equirectangular image (full 360°×180°).
        fov_deg:    Horizontal field of view of the output crop (degrees).
        yaw_deg:    Horizontal rotation from the panorama's 0° (straight ahead).
                    Positive = rotate right. Range: -180 to 180.
        pitch_deg:  Vertical tilt. Positive = tilt UP. Range: -90 to 90.
        out_w:      Output image width in pixels.
        out_h:      Output image height in pixels.

    Returns:
        out_h × out_w × 3 perspective image.
    """
    h, w = pano.shape[:2]

    fov   = np.radians(fov_deg)
    yaw   = np.radians(yaw_deg)
    pitch = np.radians(pitch_deg)

    # Focal length in pixels for a sensor of width out_w
    f  = (out_w / 2.0) / np.tan(fov / 2.0)
    cx = out_w / 2.0
    cy = out_h / 2.0

    # Build a grid of output pixel coordinates
    xs = np.linspace(0, out_w - 1, out_w)
    ys = np.linspace(0, out_h - 1, out_h)
    xv, yv = np.meshgrid(xs, ys)

    # Unproject pixels to unit rays in camera space
    # Camera looks along +Z; X is right; Y is up.
    rx =  (xv - cx) / f
    ry = -(yv - cy) / f   # flip so positive Y is up
    rz =  np.ones_like(rx)

    # --- Rotation 1: Yaw around the Y axis ---
    rx2 =  rx * np.cos(yaw) + rz * np.sin(yaw)
    ry2 =  ry
    rz2 = -rx * np.sin(yaw) + rz * np.cos(yaw)

    # --- Rotation 2: Pitch around the X axis ---
    rx3 =  rx2
    ry3 =  ry2 * np.cos(pitch) - rz2 * np.sin(pitch)
    rz3 =  ry2 * np.sin(pitch) + rz2 * np.cos(pitch)

    # Normalize to unit sphere
    norm = np.sqrt(rx3**2 + ry3**2 + rz3**2)
    rx3 /= norm
    ry3 /= norm
    rz3 /= norm

    # Convert unit vector → spherical coordinates
    lon = np.arctan2(rx3, rz3)            # horizontal angle: -π to π
    lat = np.arcsin(np.clip(ry3, -1, 1))  # vertical angle:   -π/2 to π/2

    # Map spherical coords → equirectangular pixel coordinates
    map_x = ((lon / np.pi + 1.0) / 2.0 * w).astype(np.float32)
    map_y = ((0.5 - lat / np.pi) * h).astype(np.float32)

    # Sample the panorama (BORDER_WRAP handles the 360° seam correctly)
    crop = cv2.remap(pano, map_x, map_y,
                     interpolation=cv2.INTER_LINEAR,
                     borderMode=cv2.BORDER_WRAP)
    return crop


# =============================================================================
# MAIN
# =============================================================================
def main():
    Path(OUTPUT_DIR).mkdir(exist_ok=True)

    with open(META_FILE) as f:
        metas = json.load(f)

    print(f"Target bearing (building direction): {TARGET_BEARING}°")
    print(f"Crop offsets: { {k: f'{TARGET_BEARING + v}°' for k, v in CROP_OFFSETS.items()} }")
    print(f"FOV: {FOV_DEG}°  |  Pitch: {PITCH_DEG}°  |  Output: {OUT_W}×{OUT_H}\n")

    ok_count = 0
    for meta in metas:
        img_path = os.path.join(INPUT_DIR, meta["image_name"])
        pano = cv2.imread(img_path)

        if pano is None:
            print(f"[SKIP] {img_path} — file not found or unreadable")
            continue

        pano_h, pano_w = pano.shape[:2]
        print(f"[{meta['pano_index']:03d}] {meta['image_name']}  "
              f"({pano_w}×{pano_h})  car_heading={meta['car_heading']:.1f}°")

        for crop_name, bearing_offset in CROP_OFFSETS.items():
            # Rotate from pano's forward direction to face TARGET_BEARING + offset
            yaw_deg = (TARGET_BEARING + bearing_offset) - meta["car_heading"]

            # Normalize to [-180, 180]
            yaw_deg = (yaw_deg + 180.0) % 360.0 - 180.0

            print(f"    [{crop_name:6s}] bearing={TARGET_BEARING + bearing_offset:.1f}°  "
                  f"yaw={yaw_deg:.1f}°")

            crop = equirect_to_perspective(
                pano,
                fov_deg=FOV_DEG,
                yaw_deg=yaw_deg,
                pitch_deg=PITCH_DEG,
                out_w=OUT_W,
                out_h=OUT_H,
            )

            base_name = meta["image_name"].replace(".jpg", "")
            out_name  = f"{base_name}_{crop_name}.jpg"
            out_path  = os.path.join(OUTPUT_DIR, out_name)
            cv2.imwrite(out_path, crop, [cv2.IMWRITE_JPEG_QUALITY, 95])
            ok_count += 1

        print()

    total_panos = len(metas)
    print(f"{'='*52}")
    print(f"  Done! {ok_count} crops from {total_panos} panoramas saved to '{OUTPUT_DIR}/'")
    print(f"  ({total_panos} × {len(CROP_OFFSETS)} crops = {total_panos * len(CROP_OFFSETS)} total)")
    print(f"  Next step: point run_colmap.sh at '{OUTPUT_DIR}/'")
    print(f"{'='*52}")


if __name__ == "__main__":
    main()