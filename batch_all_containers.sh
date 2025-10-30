#!/bin/bash
# Robust batch processor for all containers in left/ and right/
# Behaviors added:
# - Do not abort on single failure; continue to next container
# - Try multiple stitch configs (feather/seam and seam widths)
# - If all stitch attempts fail, build a simple horizontal composite fallback
# - Produce detailed logs in batch_logs/

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
LOG_DIR="$SCRIPT_DIR/batch_logs"
mkdir -p "$LOG_DIR"

# Find all container folders (pattern contains timestamp-like parts)
CONTAINERS_LEFT=$(find downloads/gdrive_1sASbG/imgs_stit/left -maxdepth 1 -type d | grep -E '_[0-9]+_[0-9]+_' | sort || true)
CONTAINERS_RIGHT=$(find downloads/gdrive_1sASbG/imgs_stit/right -maxdepth 1 -type d | grep -E '_[0-9]+_[0-9]+_' | sort || true)
CONTAINERS=$(echo -e "$CONTAINERS_LEFT\n$CONTAINERS_RIGHT" | sed '/^$/d')

TOTAL=$(echo "$CONTAINERS" | wc -l)
CURRENT=0
SUCCESS_COUNT=0
FAIL_COUNT=0

echo "==============================================="
echo "BATCH PROCESSING: $TOTAL containers found"
echo "Robust mode: retries + fallback composite if no panorama"
echo "Logs saved to: $LOG_DIR/"
echo "==============================================="
echo ""

run_pipeline(){
    # args: container_dir img_suffix conf blend seam_width out_log
    local cdir="$1"; shift
    local img_suffix="$1"; shift
    local conf="$1"; shift
    local blend="$1"; shift
    local seam_width="$1"; shift
    local logf="$1"; shift

    python "$SCRIPT_DIR/run_full_pipeline.py" \
        --img-dir "$cdir" \
        --img-suffix "$img_suffix" \
        --conf "$conf" \
        --lock-dy \
        --stride 3 \
        --blend "$blend" \
        --seam-width "$seam_width" \
        > "$logf" 2>&1
    return $?
}

composite_fallback(){
    # Create a guaranteed fallback panorama by horizontally concatenating rectified or aligned images
    local cdir="$1"
    local outdir="$cdir/panorama_c1_fallback"
    mkdir -p "$outdir"
     # pass cdir and outdir as argv to the embedded python so sys.argv is available
     python - "$cdir" "$outdir" <<'PY'
import sys, pathlib, cv2, numpy as np
cdir=pathlib.Path(sys.argv[1])
outdir=pathlib.Path(sys.argv[2])
imgs=[]
for p in sorted((cdir/'rectified_c1').glob('*_aligned.jpg')):
    imgs.append(str(p))
if not imgs:
    for p in sorted((cdir/'aligned_c1').glob('*_aligned.jpg')):
        imgs.append(str(p))
if not imgs:
    print('NO_IMAGES')
    sys.exit(2)
arrs=[cv2.imread(p) for p in imgs]
arrs=[a for a in arrs if a is not None]
if not arrs:
    print('NO_READABLE')
    sys.exit(3)
# Resize heights to max height
H=max(a.shape[0] for a in arrs)
norm=[]
for a in arrs:
    if a.shape[0]!=H:
        scale=H/float(a.shape[0])
        w=int(a.shape[1]*scale)
        a=cv2.resize(a,(w,H),interpolation=cv2.INTER_LINEAR)
    norm.append(a)
pan=np.hstack(norm)
first=0; last=len(norm)-1
out=outdir/f'panorama_{first}_{last}_fallback.jpg'
cv2.imwrite(str(out),pan)
print(str(out))
sys.exit(0)
PY
    return $?
}

for CONTAINER_DIR in $CONTAINERS; do
    CURRENT=$((CURRENT + 1))
    CONTAINER_NAME=$(basename "$CONTAINER_DIR")
    # skip empty lines
    if [ -z "$CONTAINER_NAME" ]; then
        continue
    fi
    SIDE=$(echo "$CONTAINER_DIR" | grep -o -E '(left|right)' || true)

    echo ""
    echo "[$CURRENT/$TOTAL] Processing: ${SIDE}/${CONTAINER_NAME}"
    echo "--------------------------------------------------"

    IMG_COUNT=$(ls "$CONTAINER_DIR"/*.jpg 2>/dev/null | wc -l || true)
    echo "  Images found: $IMG_COUNT"
    if [ -z "$IMG_COUNT" ] || [ "$IMG_COUNT" -lt 6 ]; then
        echo "  ⚠️  SKIP: Too few images (<6); continuing"
        continue
    fi

    IMG_SUFFIX=".jpg"
    if ls "$CONTAINER_DIR"/*_no_warp.jpg >/dev/null 2>&1; then
        IMG_SUFFIX="_no_warp.jpg"
    fi

    LOG_BASE="$LOG_DIR/${SIDE}_${CONTAINER_NAME}"
    CONTAINER_SUCCESS=false

    # Candidate configurations in order (conf,blend,seam_width)
    CANDIDATES=(
        "0.005|feather|3"
        "0.005|seam|3"
        "0.005|seam|8"
        "0.005|feather|8"
        "0.001|seam|8"
        "0.001|feather|8"
    )

    for spec in "${CANDIDATES[@]}"; do
        conf=$(echo "$spec" | cut -d'|' -f1)
        blend=$(echo "$spec" | cut -d'|' -f2)
        sw=$(echo "$spec" | cut -d'|' -f3)
        LOGF="${LOG_BASE}_${blend}_sw${sw}_conf${conf}.log"

        echo "  Trying: blend=$blend seam-width=$sw conf=$conf (log: $(basename "$LOGF"))"
        # clean previous panorama outputs
        rm -rf "$CONTAINER_DIR/panorama_c1" "$CONTAINER_DIR/panorama_c2" 2>/dev/null || true

        if run_pipeline "$CONTAINER_DIR" "$IMG_SUFFIX" "$conf" "$blend" "$sw" "$LOGF"; then
            PANO_COUNT=$(ls "$CONTAINER_DIR"/panorama_c1/*.jpg 2>/dev/null | wc -l || true)
            if [ "$PANO_COUNT" -gt 0 ]; then
                # mark success and rename folder to include config
                mv "$CONTAINER_DIR/panorama_c1" "$CONTAINER_DIR/panorama_c1_${blend}_sw${sw}" 2>/dev/null || true
                if [ -d "$CONTAINER_DIR/panorama_c2" ]; then
                    mv "$CONTAINER_DIR/panorama_c2" "$CONTAINER_DIR/panorama_c2_${blend}_sw${sw}" 2>/dev/null || true
                fi
                echo "  ✓ Success with blend=$blend seam-width=$sw"
                CONTAINER_SUCCESS=true
                SUCCESS_COUNT=$((SUCCESS_COUNT+1))
                break
            else
                echo "  ✗ Run completed but no panorama created (blend=$blend sw=$sw)"
            fi
        else
            echo "  ✗ Run failed (blend=$blend sw=$sw) - see $LOGF"
        fi
    done

    if ! $CONTAINER_SUCCESS ; then
        echo "  ⚠️ All stitch attempts failed for $CONTAINER_NAME — creating composite fallback"
        composite_fallback "$CONTAINER_DIR" || echo "  ✗ Fallback composite also failed"
        # If composite created, mark success
        if [ -d "$CONTAINER_DIR/panorama_c1_fallback" ]; then
            echo "  ✓ Fallback panorama created: $CONTAINER_DIR/panorama_c1_fallback/"
            SUCCESS_COUNT=$((SUCCESS_COUNT+1))
        else
            echo "  ❌ No panorama produced for $CONTAINER_NAME"
            FAIL_COUNT=$((FAIL_COUNT+1))
        fi
    fi
done

echo ""
echo "==============================================="
echo "BATCH PROCESSING COMPLETED!"
echo "Total containers processed: $TOTAL"
echo "  ✅ Success (including fallbacks): $SUCCESS_COUNT"
echo "  ❌ Failed (no panorama): $FAIL_COUNT"
echo "Logs saved to: $LOG_DIR/"
echo "==============================================="
