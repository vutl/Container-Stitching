#!/bin/bash
# Batch process all containers with both feather and seam blending

set -e

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
LOG_DIR="$SCRIPT_DIR/batch_logs"
mkdir -p "$LOG_DIR"

# Find all container folders
CONTAINERS=$(find downloads/gdrive_1sASbG/imgs_stit/right downloads/gdrive_1sASbG/imgs_stit/left -maxdepth 1 -type d | grep -E '_[0-9]+_[0-9]+_' | sort)

TOTAL=$(echo "$CONTAINERS" | wc -l)
CURRENT=0

echo "==============================================="
echo "BATCH PROCESSING: $TOTAL containers found"
echo "Processing with BOTH feather and seam blending"
echo "==============================================="
echo ""

for CONTAINER_DIR in $CONTAINERS; do
    CURRENT=$((CURRENT + 1))
    CONTAINER_NAME=$(basename "$CONTAINER_DIR")
    SIDE=$(echo "$CONTAINER_DIR" | grep -o -E '(left|right)')
    
    echo ""
    echo "[$CURRENT/$TOTAL] Processing: $SIDE/$CONTAINER_NAME"
    echo "=================================================="
    
    # Count images
    IMG_COUNT=$(ls "$CONTAINER_DIR"/*.jpg 2>/dev/null | wc -l)
    echo "  Images found: $IMG_COUNT"
    
    if [ $IMG_COUNT -lt 10 ]; then
        echo "  ⚠️  SKIP: Too few images (<10)"
        continue
    fi
    
    # Determine image suffix
    IMG_SUFFIX=".jpg"
    if ls "$CONTAINER_DIR"/*_no_warp.jpg >/dev/null 2>&1; then
        IMG_SUFFIX="_no_warp.jpg"
    fi
    
    LOG_BASE="$LOG_DIR/${SIDE}_${CONTAINER_NAME}"
    
    # ========== FEATHER BLEND ==========
    echo ""
    echo "  [1/2] Running with FEATHER blend..."
    LOG_FEATHER="${LOG_BASE}_feather.log"
    
    if python "$SCRIPT_DIR/run_full_pipeline.py" \
        --img-dir "$CONTAINER_DIR" \
        --img-suffix "$IMG_SUFFIX" \
        --conf 0.05 \
        --lock-dy \
        --stride 3 \
        --blend feather \
        --seam-width 1 \
        > "$LOG_FEATHER" 2>&1; then
        
        # Rename panorama folder to indicate feather
        if [ -d "$CONTAINER_DIR/panorama_c1" ]; then
            mv "$CONTAINER_DIR/panorama_c1" "$CONTAINER_DIR/panorama_c1_feather" 2>/dev/null || true
        fi
        if [ -d "$CONTAINER_DIR/panorama_c2" ]; then
            mv "$CONTAINER_DIR/panorama_c2" "$CONTAINER_DIR/panorama_c2_feather" 2>/dev/null || true
        fi
        echo "  ✓ FEATHER completed"
    else
        echo "  ✗ FEATHER failed - check log: $LOG_FEATHER"
    fi
    
    # ========== SEAM BLEND ==========
    echo "  [2/2] Running with SEAM blend..."
    LOG_SEAM="${LOG_BASE}_seam.log"
    
    # Clean intermediate outputs to force fresh processing with seam blend
    rm -rf "$CONTAINER_DIR/panorama_c1" "$CONTAINER_DIR/panorama_c2" 2>/dev/null || true
    
    if python "$SCRIPT_DIR/run_full_pipeline.py" \
        --img-dir "$CONTAINER_DIR" \
        --img-suffix "$IMG_SUFFIX" \
        --conf 0.5 \
        --lock-dy \
        --stride 3 \
        --blend seam \
        --seam-width 8 \
        > "$LOG_SEAM" 2>&1; then
        
        # Rename panorama folder to indicate seam
        if [ -d "$CONTAINER_DIR/panorama_c1" ]; then
            mv "$CONTAINER_DIR/panorama_c1" "$CONTAINER_DIR/panorama_c1_seam" 2>/dev/null || true
        fi
        if [ -d "$CONTAINER_DIR/panorama_c2" ]; then
            mv "$CONTAINER_DIR/panorama_c2" "$CONTAINER_DIR/panorama_c2_seam" 2>/dev/null || true
        fi
        echo "  ✓ SEAM completed"
    else
        echo "  ✗ SEAM failed - check log: $LOG_SEAM"
    fi
    
    echo "  Done: $CONTAINER_NAME"
done

echo ""
echo "==============================================="
echo "BATCH PROCESSING COMPLETED!"
echo "Total containers processed: $TOTAL"
echo "Logs saved to: $LOG_DIR/"
echo "==============================================="
