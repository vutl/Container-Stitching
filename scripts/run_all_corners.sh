#!/usr/bin/env bash
set -euo pipefail

# Batch runner for test_sides.py across all datasets under ROOT.
# It discovers img2 folders and runs three configs (A/B/C) to compare
# regular-corner behaviors after refinement updates.
#
# Usage:
#   scripts/run_all_corners.sh [ROOT_DIR]
#
# ROOT_DIR defaults to the shared unzip folder.

ROOT_DIR="${1:-/home/atin-tts-1/Container-Stitching/drive-download-20251013T040727Z-1-001_unzipped}"
MODEL="/home/atin-tts-1/Container-Stitching/last11scor_3class_31_07.pt"
SUFFIX="_no_warp.jpg"

find_datasets() {
  find "$ROOT_DIR" -maxdepth 2 -type d -name img2 | sort
}

index_range() {
  local dir="$1"
  shopt -s nullglob
  local files=("$dir"/*"$SUFFIX")
  if (( ${#files[@]} == 0 )); then
    echo ""; return 0
  fi
  local stems
  stems=$(printf "%s\n" "${files[@]##*/}" | sed -E "s/${SUFFIX//./\.}//" | sed -E "s/_no_warp\.jpg$//")
  local min max
  min=$(printf "%s\n" $stems | sort -n | head -n1)
  max=$(printf "%s\n" $stems | sort -n | tail -n1)
  echo "${min}-${max}"
}

run_config() {
  local cfg="$1"; shift
  local extra_args=("$@")
  while IFS= read -r img2; do
    local rng outdir dsroot
    rng=$(index_range "$img2")
    if [[ -z "$rng" ]]; then
      echo "[SKIP] No ${SUFFIX} files in $img2" >&2
      continue
    fi
    dsroot="${img2%/img2}"
    outdir="$dsroot/out_annot_sides_${cfg}"
    mkdir -p "$outdir"
    echo "[CFG ${cfg}] $(basename "$dsroot") -> indices [$rng]"
    python3 test_sides.py \
      --model "$MODEL" \
      --dir "$img2" \
      --indices "$rng" \
      --suffix "$SUFFIX" \
      --out "$outdir" \
      --conf 0.05 \
      --iou 0.6 \
      "${extra_args[@]}"
  done < <(find_datasets)
}

# Configs
# A: baseline tuned (less center pull)
CFG_A=(--corner-center-thr-alpha 0.4)
# B: more edge-hugging
CFG_B=(--corner-side-margin-frac 0.14 --corner-center-near-px 6 --corner-refine-search-h 10 --corner-refine-search-w 10 --corner-center-thr-alpha 0.35 --corner-perc-alpha 0.40 --corner-std-mult 0.90)
# C: conservative with wider search and slightly larger center window (fallback only)
CFG_C=(--corner-center-window-frac 0.50 --corner-center-near-px 8 --corner-refine-search-h 12 --corner-refine-search-w 12 --corner-center-thr-alpha 0.45)

# Run sequence
run_config A "${CFG_A[@]}"
run_config B "${CFG_B[@]}"
run_config C "${CFG_C[@]}"

echo "All configs finished."