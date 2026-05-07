# crop_perspective.py
import cv2, numpy as np, os, json

def equirect_to_perspective(pano, fov_deg, heading_deg, pitch_deg, out_w, out_h):
    """Extract a perspective crop from an equirectangular panorama."""
    h, w = pano.shape[:2]
    fov = np.radians(fov_deg)
    heading = np.radians(heading_deg)
    pitch = np.radians(pitch_deg)

    f = (out_w / 2) / np.tan(fov / 2)
    cx, cy = out_w / 2, out_h / 2

    # Build pixel grid
    x = np.linspace(0, out_w - 1, out_w)
    y = np.linspace(0, out_h - 1, out_h)
    xv, yv = np.meshgrid(x, y)

    # Ray directions in camera space
    dx = (xv - cx) / f
    dy = -(yv - cy) / f
    dz = np.ones_like(dx)

    # 1. Heading (yaw) — rotate around Y axis
    dx2 =  dx * np.cos(heading) + dz * np.sin(heading)
    dz2 = -dx * np.sin(heading) + dz * np.cos(heading)

    # 2. Pitch — rotate around X axis  
    dy3 =  dy * np.cos(pitch) - dz2 * np.sin(pitch)
    dz3 =  dy * np.sin(pitch) + dz2 * np.cos(pitch)
    # dx2 unchanged

    # Normalize
    norm = np.sqrt(dx2**2 + dy3**2 + dz3**2)
    dx2 /= norm;  dy3 /= norm;  dz3 /= norm

    lon = np.arctan2(dx2, dz3)
    lat = np.arcsin(np.clip(dy3, -1, 1))

    # To pixel coords in equirectangular
    map_x = ((lon / np.pi + 1) / 2 * w).astype(np.float32)
    map_y = ((0.5 - lat / np.pi) * h).astype(np.float32)

    return cv2.remap(pano, map_x, map_y, cv2.INTER_LINEAR, borderMode=cv2.BORDER_WRAP)

# --- Config ---
INPUT_DIR  = "street_images"
OUTPUT_DIR = "perspective_crops"
META_FILE  = "panorama_metadata.json"
FOV        = 90       # degrees — wider = more overlap, lower quality
PITCH      = 0        # degrees down from horizon (try -5 to aim slightly up at facade)
OUT_W      = 2048
OUT_H      = 1536

os.makedirs(OUTPUT_DIR, exist_ok=True)

with open(META_FILE) as f:
    metas = json.load(f)

for meta in metas:
    img_path = os.path.join(INPUT_DIR, meta["image_name"])
    pano = cv2.imread(img_path)
    if pano is None:
        print(f"[SKIP] {img_path} not found")
        continue

    current_heading = meta.get("car_heading", meta.get("heading", 0))
    
    crop = equirect_to_perspective(
        pano,
        fov_deg=FOV,
        heading_deg=current_heading,
        pitch_deg=PITCH,
        out_w=OUT_W,
        out_h=OUT_H,
    )

    out_name = meta["image_name"].replace(".jpg", f"_persp.jpg")
    out_path = os.path.join(OUTPUT_DIR, out_name)
    cv2.imwrite(out_path, crop, [cv2.IMWRITE_JPEG_QUALITY, 95])
    print(f"[OK] {out_path}")

print("Done. Now point run_colmap.sh at 'perspective_crops/'")