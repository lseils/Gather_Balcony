#!/usr/bin/env python3
"""
panda_crop_to_mesh.py
----------------------
Takes PanDA's ERP depth maps + raw panoramas, crops both to a
perspective view facing the building facade, then builds a relief mesh.

Uses the same equirect_to_perspective projection as crop_perspective.py
so the color crop and depth crop are perfectly aligned.

Usage:
  python panda_crop_to_mesh.py
  python panda_crop_to_mesh.py --pano street_images/pano_001.jpg --depth_scale 500
"""

import argparse
import os
from pathlib import Path

import cv2
import numpy as np

# =============================================================================
# CONFIG — matches your crop_perspective.py settings
# =============================================================================
STREET_IMAGES_DIR = "street_images"
PANDA_DEPTH_DIR   = "panda_depth"
OUTPUT_DIR        = "panda_meshes"

TARGET_BEARING    = 250.0   # compass bearing the building faces from street
FOV_DEG           = 90.0    # horizontal FOV
PITCH_DEG         = 16.0    # tilt up toward upper floors
OUT_W             = 2048    # output width
OUT_H             = 1536    # output height

DEPTH_SCALE       = 500.0   # Z exaggeration for mesh relief
DISCONTINUITY     = 0.03    # depth jump threshold for triangle cutting
STRIDE            = 3       # pixel subsample for mesh density


# =============================================================================
# EQUIRECT -> PERSPECTIVE PROJECTION
# (same math as crop_perspective.py)
# =============================================================================
def equirect_to_perspective(pano: np.ndarray,
                             fov_deg: float,
                             yaw_deg: float,
                             pitch_deg: float,
                             out_w: int,
                             out_h: int) -> np.ndarray:
    h, w = pano.shape[:2]

    fov   = np.radians(fov_deg)
    yaw   = np.radians(yaw_deg)
    pitch = np.radians(pitch_deg)

    f  = (out_w / 2.0) / np.tan(fov / 2.0)
    cx = out_w / 2.0
    cy = out_h / 2.0

    xs = np.linspace(0, out_w - 1, out_w)
    ys = np.linspace(0, out_h - 1, out_h)
    xv, yv = np.meshgrid(xs, ys)

    rx =  (xv - cx) / f
    ry = -(yv - cy) / f
    rz =  np.ones_like(rx)

    rx2 =  rx * np.cos(yaw) + rz * np.sin(yaw)
    ry2 =  ry
    rz2 = -rx * np.sin(yaw) + rz * np.cos(yaw)

    rx3 =  rx2
    ry3 =  ry2 * np.cos(pitch) - rz2 * np.sin(pitch)
    rz3 =  ry2 * np.sin(pitch) + rz2 * np.cos(pitch)

    norm = np.sqrt(rx3**2 + ry3**2 + rz3**2)
    rx3 /= norm; ry3 /= norm; rz3 /= norm

    lon = np.arctan2(rx3, rz3)
    lat = np.arcsin(np.clip(ry3, -1, 1))

    map_x = ((lon / np.pi + 1.0) / 2.0 * w).astype(np.float32)
    map_y = ((0.5 - lat / np.pi) * h).astype(np.float32)

    interp = cv2.INTER_LINEAR if pano.dtype == np.uint8 else cv2.INTER_LINEAR
    crop = cv2.remap(pano, map_x, map_y,
                     interpolation=interp,
                     borderMode=cv2.BORDER_WRAP)
    return crop


# =============================================================================
# MESH BUILDER
# =============================================================================
def build_mesh(depth: np.ndarray,
               rgb: np.ndarray,
               stride: int,
               depth_scale: float,
               discontinuity: float):
    h, w = depth.shape
    ys = np.arange(0, h, stride)
    xs = np.arange(0, w, stride)
    grid_h, grid_w = len(ys), len(xs)

    xv, yv = np.meshgrid(xs, ys)
    dv = depth[yv, xv]

    vertices = np.stack([
        xv.astype(np.float32),
        (h - yv).astype(np.float32),
        (dv * depth_scale).astype(np.float32)
    ], axis=-1).reshape(-1, 3)

    rgb_small = cv2.resize(rgb, (w, h), interpolation=cv2.INTER_AREA)
    colors = rgb_small[yv, xv][:, ::-1].reshape(-1, 3)  # BGR -> RGB

    faces = []
    for r in range(grid_h - 1):
        for c in range(grid_w - 1):
            i00 = r * grid_w + c
            i01 = r * grid_w + (c + 1)
            i10 = (r + 1) * grid_w + c
            i11 = (r + 1) * grid_w + (c + 1)

            d00 = dv[r, c]; d01 = dv[r, c + 1]
            d10 = dv[r + 1, c]; d11 = dv[r + 1, c + 1]

            tri1 = (d00, d10, d11)
            if (max(tri1) - min(tri1)) < discontinuity:
                faces.append((i00, i10, i11))

            tri2 = (d00, d11, d01)
            if (max(tri2) - min(tri2)) < discontinuity:
                faces.append((i00, i11, i01))

    return vertices, colors, np.array(faces, dtype=np.int64)


# =============================================================================
# PLY / OBJ WRITERS
# =============================================================================
def write_ply(path, vertices, colors, faces):
    with open(path, "w") as f:
        f.write("ply\nformat ascii 1.0\n")
        f.write(f"element vertex {len(vertices)}\n")
        f.write("property float x\nproperty float y\nproperty float z\n")
        f.write("property uchar red\nproperty uchar green\nproperty uchar blue\n")
        f.write(f"element face {len(faces)}\n")
        f.write("property list uchar int vertex_indices\nend_header\n")
        for v, c in zip(vertices, colors):
            f.write(f"{v[0]} {v[1]} {v[2]} {int(c[0])} {int(c[1])} {int(c[2])}\n")
        for face in faces:
            f.write(f"3 {face[0]} {face[1]} {face[2]}\n")


def write_obj(path, vertices, faces):
    with open(path, "w") as f:
        for v in vertices:
            f.write(f"v {v[0]} {v[1]} {v[2]}\n")
        for face in faces:
            f.write(f"f {face[0]+1} {face[1]+1} {face[2]+1}\n")


# =============================================================================
# MAIN
# =============================================================================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pano",         default=None,
                        help="Single pano to process (e.g. street_images/pano_001.jpg). "
                             "If omitted, processes all panos.")
    parser.add_argument("--bearing",      type=float, default=TARGET_BEARING)
    parser.add_argument("--fov",          type=float, default=FOV_DEG)
    parser.add_argument("--pitch",        type=float, default=PITCH_DEG)
    parser.add_argument("--depth_scale",  type=float, default=DEPTH_SCALE)
    parser.add_argument("--discontinuity",type=float, default=DISCONTINUITY)
    parser.add_argument("--stride",       type=int,   default=STRIDE)
    parser.add_argument("--output_dir",   default=OUTPUT_DIR)
    args = parser.parse_args()

    Path(args.output_dir).mkdir(exist_ok=True)

    # Gather panos to process
    if args.pano:
        pano_paths = [args.pano]
    else:
        pano_paths = sorted([
            os.path.join(STREET_IMAGES_DIR, f)
            for f in os.listdir(STREET_IMAGES_DIR)
            if f.endswith(".jpg") or f.endswith(".png")
        ])

    for pano_path in pano_paths:
        pano_name = Path(pano_path).stem  # e.g. "pano_001"
        depth_path = os.path.join(PANDA_DEPTH_DIR, f"{pano_name}.png")

        if not os.path.exists(depth_path):
            print(f"[SKIP] No depth map for {pano_name} at {depth_path}")
            continue

        print(f"\nProcessing {pano_name}...")

        # Load pano RGB and PanDA depth
        pano_rgb   = cv2.imread(pano_path)
        pano_depth = cv2.imread(depth_path, cv2.IMREAD_GRAYSCALE)

        if pano_rgb is None or pano_depth is None:
            print(f"  [SKIP] Could not load files")
            continue

        print(f"  Pano RGB:   {pano_rgb.shape[1]}x{pano_rgb.shape[0]}")
        print(f"  Pano depth: {pano_depth.shape[1]}x{pano_depth.shape[0]}")

        # Compute yaw from car heading — we don't have metadata here so use
        # TARGET_BEARING directly as absolute yaw from pano's 0° forward.
        # NOTE: if your pano's forward direction != 0°, adjust bearing here.
        # For Google Street View panos, forward is typically the driving direction.
        # You can load panorama_metadata.json to get car_heading per pano.
        import json
        meta_path = "panorama_metadata.json"
        car_heading = 0.0
        if os.path.exists(meta_path):
            with open(meta_path) as mf:
                metas = json.load(mf)
            for m in metas:
                if m["image_name"].replace(".jpg", "") == pano_name:
                    car_heading = m["car_heading"]
                    break

        yaw_deg = args.bearing - car_heading
        yaw_deg = (yaw_deg + 180.0) % 360.0 - 180.0
        print(f"  Car heading: {car_heading:.1f}°  →  yaw to facade: {yaw_deg:.1f}°")

        # Crop RGB (using pano resolution)
        color_crop = equirect_to_perspective(
            pano_rgb, args.fov, yaw_deg, args.pitch, OUT_W, OUT_H
        )

        # Crop depth — resize pano_depth to pano_rgb size first so they share
        # the same equirectangular grid, then project
        depth_fullsize = cv2.resize(
            pano_depth,
            (pano_rgb.shape[1], pano_rgb.shape[0]),
            interpolation=cv2.INTER_LINEAR
        )
        depth_crop = equirect_to_perspective(
            depth_fullsize, args.fov, yaw_deg, args.pitch, OUT_W, OUT_H
        )

        # Normalize depth to 0-1
        depth_norm = depth_crop.astype(np.float32) / 255.0

        # Save crops for inspection
        cv2.imwrite(os.path.join(args.output_dir, f"{pano_name}_color_crop.jpg"), color_crop)
        depth_vis = cv2.applyColorMap(depth_crop, cv2.COLORMAP_INFERNO)
        cv2.imwrite(os.path.join(args.output_dir, f"{pano_name}_depth_crop.png"), depth_vis)

        # Build mesh
        print(f"  Building mesh (depth_scale={args.depth_scale}, "
              f"discontinuity={args.discontinuity}, stride={args.stride})...")
        vertices, colors, faces = build_mesh(
            depth_norm, color_crop, args.stride, args.depth_scale, args.discontinuity
        )
        print(f"  {len(vertices):,} vertices, {len(faces):,} faces")

        stem = f"{pano_name}_panda"
        write_ply(os.path.join(args.output_dir, f"{stem}_mesh.ply"), vertices, colors, faces)
        write_obj(os.path.join(args.output_dir, f"{stem}_mesh.obj"), vertices, faces)
        print(f"  Saved → {args.output_dir}/{stem}_mesh.ply")

    print(f"\n{'='*52}")
    print(f"  Done! Meshes saved to '{OUTPUT_DIR}/'")
    print(f"\n  Tuning tips:")
    print(f"    Flat balconies?  increase --depth_scale (try 800)")
    print(f"    Spiky edges?     decrease --discontinuity (try 0.015)")
    print(f"    Too dense?       increase --stride (try 5)")
    print(f"{'='*52}")


if __name__ == "__main__":
    main()