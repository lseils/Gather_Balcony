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
VOCAB_TREE=""                       # Optional: path to vocab_tree_flickr100K_words256K.bin
                                    # Download from: https://demuc.de/colmap/#download
                                    # Leave empty to use exhaustive matching (slower but fine for <100 images)
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

COLMAP_VERSION=$(colmap --version 2>&1 | head -1)
echo "[OK] $COLMAP_VERSION"

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
    --ImageReader.single_camera 0 \
    --ImageReader.camera_model SIMPLE_RADIAL \
    --SiftExtraction.use_gpu 1 \
    --SiftExtraction.gpu_index $GPU_INDEX \
    --SiftExtraction.num_threads $NUM_THREADS \
    --SiftExtraction.max_image_size 3200 \
    --SiftExtraction.max_num_features 8192

# =============================================================================
# STEP 2: Feature Matching
# =============================================================================
echo ""
echo "[2/6] Matching features..."

if [ -n "$VOCAB_TREE" ] && [ -f "$VOCAB_TREE" ]; then
    echo "  Using vocabulary tree matching (fast)..."
    colmap vocab_tree_matcher \
        --database_path "$DB" \
        --VocabTreeMatching.vocab_tree_path "$VOCAB_TREE" \
        --VocabTreeMatching.num_images 50 \
        --SiftMatching.use_gpu 1 \
        --SiftMatching.gpu_index $GPU_INDEX \
        --SiftMatching.num_threads $NUM_THREADS
else
    echo "  Using sequential matching (good for ordered street-level captures)..."
    # Sequential matching is ideal here: images are taken along a walking path,
    # so adjacent images overlap heavily. overlap=10 matches each image to its
    # 10 nearest neighbors in the sequence.
    colmap sequential_matcher \
        --database_path "$DB" \
        --SequentialMatching.overlap 10 \
        --SequentialMatching.quadratic_overlap 1 \
        --SiftMatching.use_gpu 1 \
        --SiftMatching.gpu_index $GPU_INDEX \
        --SiftMatching.num_threads $NUM_THREADS
fi

# =============================================================================
# STEP 3: Sparse Reconstruction (Structure from Motion)
# =============================================================================
echo ""
echo "[3/6] Running sparse reconstruction (SfM)..."
colmap mapper \
    --database_path "$DB" \
    --image_path "$IMAGE_DIR" \
    --output_path "$SPARSE_DIR" \
    --Mapper.num_threads $NUM_THREADS \
    --Mapper.init_min_tri_angle 4 \
    --Mapper.multiple_models 0 \
    --Mapper.extract_colors 1

# Check SfM succeeded
if [ ! -d "$SPARSE_DIR/0" ]; then
    echo "[ERROR] SfM failed — no model in $SPARSE_DIR/0"
    echo "  Tip: Try exhaustive_matcher instead of sequential_matcher if images"
    echo "       don't have enough overlap."
    exit 1
fi

REGISTERED=$(colmap model_analyzer --path "$SPARSE_DIR/0" 2>&1 | grep "Registered images" | awk '{print $NF}')
echo "[OK] Registered $REGISTERED / $IMAGE_COUNT images"

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
    --StereoFusion.min_num_pixels 5

echo "[OK] Dense point cloud: $DENSE_DIR/fused.ply"

# =============================================================================
# STEP 6: Meshing (Poisson) → CAD/BIM-ready mesh
# =============================================================================
echo ""
echo "[+] Running Poisson surface reconstruction..."
colmap poisson_mesher \
    --input_path "$DENSE_DIR/fused.ply" \
    --output_path "$MESH_DIR/mesh_poisson.ply" \
    --PoissonMeshing.trim 10

echo "[OK] Poisson mesh: $MESH_DIR/mesh_poisson.ply"

# Also run Delaunay mesher as an alternative (often cleaner for facades)
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
echo "    Poisson mesh:     $MESH_DIR/mesh_poisson.ply"
echo "    Delaunay mesh:    $MESH_DIR/mesh_delaunay.ply"
echo ""
echo "  Next steps for CAD/BIM:"
echo "    1. Open .ply in CloudCompare or MeshLab to clean/decimate"
echo "    2. Export as .obj or .fbx for Revit/AutoCAD"
echo "    3. Or use open3d_postprocess.py (included) to automate cleanup"
echo ""