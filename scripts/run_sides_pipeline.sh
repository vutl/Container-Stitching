#!/usr/bin/env bash
set -euo pipefail

# Full side-face pipeline runner: test_sides -> trim -> warp_align_sides -> rectify_to_rectangle_sides -> test3_sides
# Works on one or more dataset image folders (typically .../<DATASET>/img2)
#
# Usage:
#   scripts/run_sides_pipeline.sh \
#     --model path/to/yolo_model.pt \
#     [--stride 2] [--indices A-B] [--raw-suffix _no_warp.jpg] \
#     [--detector kaze] [--seam-width 5] [--lock-dy] \
#     [--center] [--top 20] [--bottom 20] [--vertical-margin 20] \
#     [--conf 0.05] [--iou 0.6] \
#     /abs/path/to/DATASET1/img2 [/abs/path/to/DATASET2/img2 ...]
#
# Notes:
# - If --indices is omitted, the script auto-detects [min-max] from file names in each img2 dir
#   by parsing the leading integer (e.g., 222_no_warp.jpg -> 222).
# - Default matching/stitching uses translation + seam + (optional) lock-dy with detector=kaze.
# - To speed up, increase --stride (e.g., 2 or 3). Higher stride means fewer frames used.
# - Outputs are created alongside each dataset folder.

print_usage() {
  sed -n '1,200p' "$0" | sed -n '1,60p' | sed 's/^# \{0,1\}//'
}

# Defaults
MODEL=""
STRIDE=2
INDICES=""
RAW_SUFFIX="_no_warp.jpg"
DETECTOR="kaze"
SEAM_WIDTH=5
LOCK_DY=1
CENTER=0
TOP=20
BOTTOM=20
VERTICAL_MARGIN=20
CONF=0.05
IOU=0.6
DATASETS=()

# Parse args
while [[ $# -gt 0 ]]; do
  case "$1" in
    --model) MODEL="${2:-}"; shift 2;;
    --stride) STRIDE="${2:-}"; shift 2;;
    --indices) INDICES="${2:-}"; shift 2;;
    --raw-suffix) RAW_SUFFIX="${2:-}"; shift 2;;
    --detector) DETECTOR="${2:-}"; shift 2;;
    --seam-width) SEAM_WIDTH="${2:-}"; shift 2;;
    --lock-dy) LOCK_DY=1; shift;;
    --no-lock-dy) LOCK_DY=0; shift;;
    --center) CENTER=1; shift;;
    --top) TOP="${2:-}"; shift 2;;
    --bottom) BOTTOM="${2:-}"; shift 2;;
    --vertical-margin) VERTICAL_MARGIN="${2:-}"; shift 2;;
    --conf) CONF="${2:-}"; shift 2;;
    --iou) IOU="${2:-}"; shift 2;;
    -h|--help) print_usage; exit 0;;
    --) shift; break;;
    -*) echo "Unknown option: $1" >&2; print_usage; exit 1;;
    *) DATASETS+=("$1"); shift;;
  esac
done

if [[ -z "$MODEL" ]]; then
  echo "ERROR: --model path/to/yolo_model.pt is required" >&2
  print_usage
  exit 1
fi

if [[ ${#DATASETS[@]} -eq 0 ]]; then
  echo "ERROR: Provide at least one dataset img2 directory" >&2
  print_usage
  exit 1
fi

# Small helpers
join_by() { local IFS="$1"; shift; echo "$*"; }

for IMG_DIR in "${DATASETS[@]}"; do
  if [[ ! -d "$IMG_DIR" ]]; then
    echo "ERROR: Not a directory: $IMG_DIR" >&2
    exit 1
  fi

  # Dataset root (assumes IMG_DIR ends with /img2)
  ROOT_DIR="$(dirname "$IMG_DIR")"
  DS_NAME="$(basename "$ROOT_DIR")"

  echo "\n=== Processing dataset: $DS_NAME ==="
  echo "Image directory: $IMG_DIR"

  # Auto-detect indices if not provided
  USE_INDICES="$INDICES"
  if [[ -z "$USE_INDICES" ]]; then
    # Extract leading integers from filenames matching RAW_SUFFIX
    mapfile -t IDX_LIST < <(find "$IMG_DIR" -maxdepth 1 -type f -name "*${RAW_SUFFIX}" -printf "%f\n" \
      | sed -nE "s/^([0-9]+).*$(printf %q "$RAW_SUFFIX")$/\1/p" | sort -n)
    if [[ ${#IDX_LIST[@]} -eq 0 ]]; then
      echo "ERROR: No files matching *${RAW_SUFFIX} found in $IMG_DIR (cannot auto-detect indices)" >&2
      exit 1
    fi
    MIN_IDX="${IDX_LIST[0]}"
    MAX_IDX="${IDX_LIST[-1]}"
    USE_INDICES="${MIN_IDX}-${MAX_IDX}"
  fi
  echo "Indices: $USE_INDICES"

  # Derive suffix for trimmed images, e.g., _no_warp.jpg -> _no_warp_trim.jpg
  RAW_BASE="${RAW_SUFFIX%.*}"
  RAW_EXT="${RAW_SUFFIX##*.}"
  TRIM_SUFFIX="${RAW_BASE}_trim.${RAW_EXT}"

  OUT_ANN="${ROOT_DIR}/out_annot_sides"
  OUT_TRIM="${ROOT_DIR}/out_trimmed_sides"
  OUT_TRIM_CORNERS="${ROOT_DIR}/out_trimmed_sides_corners"
  OUT_TRIM_VIS="${ROOT_DIR}/out_trimmed_sides_vis"
  ALIGNED="${ROOT_DIR}/aligned_sides"
  RECTIFIED="${ROOT_DIR}/aligned_rectified_sides"

  mkdir -p "$OUT_ANN" "$OUT_TRIM" "$OUT_TRIM_CORNERS" "$OUT_TRIM_VIS" "$ALIGNED" "$RECTIFIED"

  echo "[1/5] Detecting corners with test_sides.py"
  python3 test_sides.py \
    --model "$MODEL" \
    --dir "$IMG_DIR" \
    --indices "$USE_INDICES" \
    --suffix "$RAW_SUFFIX" \
    --out "$OUT_ANN" \
    --conf "$CONF" \
    --iou "$IOU" \
    --max-per-quad 1 \
    --corner-center-thr-alpha 0.4  # Base config: reduce center pull

  echo "[2/5] Trimming borders and updating corners"
  python3 trim_black_update_corners.py \
    --dirs "$IMG_DIR" \
    --suffix "$RAW_SUFFIX" \
    --corners-suffix _corners.txt \
    --corners-dir "$OUT_ANN" \
    --corners-out-dir "$OUT_TRIM_CORNERS" \
    --out-dir "$OUT_TRIM" \
    --vis-dir "$OUT_TRIM_VIS" \
    --thresh 8 \
    --min-row-frac 0.02 \
    --min-col-frac 0.02

  echo "[3/5] Warp-align to common canvas (sides)"
  WARP_CMD=(python3 warp_align_sides.py \
    --dir "$OUT_TRIM" \
    --indices "$USE_INDICES" \
    --img-suffix "$TRIM_SUFFIX" \
    --corners-dir "$OUT_TRIM_CORNERS" \
    --corners-suffix _corners.txt \
    --out "$ALIGNED" \
    --top "$TOP" \
    --bottom "$BOTTOM")
  if [[ "$CENTER" -eq 1 ]]; then
    WARP_CMD+=(--center-container)
  fi
  "${WARP_CMD[@]}"

  echo "[4/5] Rectifying to rectangles and writing masks"
  python3 rectify_to_rectangle_sides.py \
    --dir "$ALIGNED" \
    --indices "$USE_INDICES" \
    --image-suffix _aligned.jpg \
    --corners-suffix _aligned_corners.txt \
    --out "$RECTIFIED" \
    --vertical-margin "$VERTICAL_MARGIN" \
    --write-mask

  echo "[5/5] Stitching into panorama (translation + seam)"
  OUT_PANO="${ROOT_DIR}/out_panorama_${DETECTOR}_translation_seam_stride${STRIDE}"
  if [[ "$LOCK_DY" -eq 1 ]]; then OUT_PANO+="_dylock"; fi
  if [[ "$CENTER" -eq 1 ]]; then OUT_PANO+="_centered"; fi
  mkdir -p "$OUT_PANO"

  STITCH_CMD=(python3 test3_sides.py \
    --aligned-dir "$RECTIFIED" \
    --indices "$USE_INDICES" \
    --corners-dir "$RECTIFIED" \
    --transform translation \
    --detector "$DETECTOR" \
    --blend seam \
    --seam-width "$SEAM_WIDTH" \
    --stride "$STRIDE" \
    --out "$OUT_PANO")
  if [[ "$LOCK_DY" -eq 1 ]]; then STITCH_CMD+=(--lock-dy); fi
  "${STITCH_CMD[@]}"

  echo "Done: ${OUT_PANO}"
  echo "(Check panorama_*_bbox.jpg inside)"

done

