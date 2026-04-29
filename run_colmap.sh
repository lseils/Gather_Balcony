#!/bin/bash
# =============================================================================
# COLMAP Pipeline: Street View Images → Dense Point Cloud → Mesh
# For use on remote HPC/server (CLI only, no GUI)
# =============================================================================

set -e  # Exit on any error

# -----------------------------------------------------------------------------
# CONFIG — edit these to match your environment
# -----------------------------------------------------------------------------
IMAGE_DIR="street_images"           # Output folder from fetch_streetview.py
COLMAP_DIR="colmap_workspace"       # Where all COLMAP outputs will live
NUM_THREADS=$(nproc)                # Use all available CPU cores
GPU_INDEX=0                         # Set to -1 if no GPU available

# -----------------------------------------------------------------------------
# DERIVED PATHS (don't edit)
# -----------------------------------------------------------------------------
DB="$COLMAP_DIR/database.db"
SPARSE_DIR="$COLMAP_DIR/sparse"
DENSE_DIR="$COLMAP_DIR/dense"
MESH_DIR="$COLMAP_DIR/mesh"

# =============================================================================
# STEP 0: Preflight checks
# =============================================================================
echo "============================================"
echo "  COLMAP Street View Pipeline"
echo "============================================"

# PACE HPC: COLMAP runs inside a Singularity container
colmap() {
    apptainer exec --nv /usr/local/pace-apps/manual/packages/singularity_images/colmap-3.10.sif colmap "$@"
}
export -f colmap

echo "[OK] COLMAP loaded successfully"

IMAGE_COUNT=$(ls "$IMAGE_DIR"/*.jpg "$IMAGE_DIR"/*.png 2>/dev/null | wc -l)
if [ "$IMAGE_COUNT" -lt 3 ]; then
    echo "[ERROR] Found only $IMAGE_COUNT image(s) in '$IMAGE_DIR'. Need at least 3."
    exit 1
fi
echo "[OK] Found $IMAGE_COUNT images in '$IMAGE_DIR'"

mkdir -p "$SPARSE_DIR" "$DENSE_DIR" "$MESH_DIR"

# =============================================================================
# STEP 1: Feature Extraction
# =============================================================================
echo ""
echo "[1/6] Extracting features..."
colmap feature_extractor \
    --database_path "$DB" \
    --image_path "$IMAGE_DIR" \
    --ImageReader.single_camera 1 \
    --ImageReader.camera_model OPENCV \
    --SiftExtraction.use_gpu 1 \
    --SiftExtraction.gpu_index $GPU_INDEX \
    --SiftExtraction.num_threads $NUM_THREADS \
    --SiftExtraction.max_image_size 3200 \
    --SiftExtraction.max_num_features 8192

# =============================================================================
# STEP 2: Feature Matching
# =============================================================================
echo ""
echo "[2/6] Matching features (exhaustive — best for small sets)..."
# Exhaustive matching compares every image against every other.
# With only 20 images this is fast, and much more robust than sequential
# matching for Street View images which have large viewpoint changes.
colmap exhaustive_matcher \
    --database_path "$DB" \
    --SiftMatching.use_gpu 1 \
    --SiftMatching.gpu_index $GPU_INDEX \
    --SiftMatching.num_threads $NUM_THREADS \
    --SiftMatching.guided_matching 1

# =============================================================================
# STEP 3: Sparse Reconstruction (Structure from Motion)
# Uses pose priors from generate_colmap_poses.py if available
# =============================================================================
echo ""
echo "[3/6] Running sparse reconstruction (SfM)..."

if [ -f "$SPARSE_DIR/0/cameras.txt" ]; then
    echo "  Found pose priors — using known camera positions as initialization..."
    colmap point_triangulator \
        --database_path "$DB" \
        --image_path "$IMAGE_DIR" \
        --input_path "$SPARSE_DIR/0" \
        --output_path "$SPARSE_DIR/0" \
        --Mapper.num_threads $NUM_THREADS \
        --Mapper.tri_min_angle 4
else
    echo "  No pose priors found — running full SfM mapper..."
    echo "  Tip: Run generate_colmap_poses.py first for better results"
    colmap mapper \
    --database_path "$DB" \
    --image_path "$IMAGE_DIR" \
    --output_path "$SPARSE_DIR" \
    --Mapper.num_threads $NUM_THREADS \
    --Mapper.init_min_tri_angle 0.5 \
    --Mapper.tri_min_angle 0.5 \
    --Mapper.tri_complete_max_reproj_error 4 \
    --Mapper.multiple_models 0 \
    --Mapper.extract_colors 1 \
    --Mapper.init_num_trials 500
fi

echo "[OK] Sparse model ready"

# Convert txt model to bin if needed (required for dense steps)
if [ -f "$SPARSE_DIR/0/cameras.txt" ] && [ ! -f "$SPARSE_DIR/0/cameras.bin" ]; then
    echo "  Converting text model to binary..."
    colmap model_converter \
        --input_path "$SPARSE_DIR/0" \
        --output_path "$SPARSE_DIR/0" \
        --output_type BIN
fi

# =============================================================================
# STEP 4: Dense Reconstruction (MVS)
# =============================================================================
echo ""
echo "[4/6] Undistorting images for dense reconstruction..."
colmap image_undistorter \
    --image_path "$IMAGE_DIR" \
    --input_path "$SPARSE_DIR/0" \
    --output_path "$DENSE_DIR" \
    --output_type COLMAP \
    --max_image_size 2000

echo ""
echo "[5/6] Running PatchMatch stereo (this is the slow step)..."
colmap patch_match_stereo \
    --workspace_path "$DENSE_DIR" \
    --workspace_format COLMAP \
    --PatchMatchStereo.gpu_index $GPU_INDEX \
    --PatchMatchStereo.window_radius 5 \
    --PatchMatchStereo.num_samples 15 \
    --PatchMatchStereo.num_iterations 5 \
    --PatchMatchStereo.geom_consistency true

# =============================================================================
# STEP 5: Stereo Fusion → Dense Point Cloud
# =============================================================================
echo ""
echo "[6/6] Fusing depth maps into dense point cloud..."
colmap stereo_fusion \
    --workspace_path "$DENSE_DIR" \
    --workspace_format COLMAP \
    --input_type geometric \
    --output_path "$DENSE_DIR/fused.ply" \
    --StereoFusion.num_threads $NUM_THREADS \
    --StereoFusion.min_num_pixels 3

echo "[OK] Dense point cloud: $DENSE_DIR/fused.ply"

# =============================================================================
# STEP 6: Meshing → CAD/BIM-ready mesh
# Poisson mesher has a known segfault in COLMAP 3.10 container,
# so we use Delaunay only here. open3d_postprocess.py handles Poisson.
# =============================================================================
echo ""
echo "[+] Running Delaunay surface reconstruction (facade-friendly)..."
colmap delaunay_mesher \
    --input_path "$DENSE_DIR" \
    --input_type dense \
    --output_path "$MESH_DIR/mesh_delaunay.ply"

echo "[OK] Delaunay mesh: $MESH_DIR/mesh_delaunay.ply"

# =============================================================================
# DONE
# =============================================================================
echo ""
echo "============================================"
echo "  Pipeline complete!"
echo "============================================"
echo ""
echo "  Outputs:"
echo "    Sparse model:     $SPARSE_DIR/0/"
echo "    Dense cloud:      $DENSE_DIR/fused.ply"
echo "    Delaunay mesh:    $MESH_DIR/mesh_delaunay.ply"
echo ""
echo "  Next step — clean up and export for CAD/BIM:"
echo "    source venv/bin/activate"
echo "    pip install open3d"
echo "    python open3d_postprocess.py"
echo ""