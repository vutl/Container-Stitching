#!/usr/bin/env bash
# Run the container batch processing under nohup for BOTH blend modes: feather and seam.
# Creates per-container logs: batch_logs/<side>_<name>_feather.log and _seam.log
# Master log: batch_logs/batch_master_both.log

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LOG_DIR="$SCRIPT_DIR/batch_logs"
mkdir -p "$LOG_DIR"
ROOTS=("downloads/gdrive_1sASbG/imgs_stit/right" "downloads/gdrive_1sASbG/imgs_stit/left")
MASTER_LOG="$LOG_DIR/batch_master_both.log"

echo "Batch (both blends) started at $(date)" >> "$MASTER_LOG"

for ROOT in "${ROOTS[@]}"; do
  if [ ! -d "$ROOT" ]; then
    echo "Root not found, skipping: $ROOT" | tee -a "$MASTER_LOG"
    continue
  fi
  echo "Scanning: $ROOT" | tee -a "$MASTER_LOG"
  while IFS= read -r d; do
    if [ -z "$d" ] || [ "$d" = "$ROOT" ]; then
      continue
    fi
    NAME=$(basename "$d")
    SIDE=$(echo "$d" | grep -o -E '(left|right)' || true)

    # Run find_direction once per container
    LOG_BASE="$LOG_DIR/${SIDE}_${NAME}"
    LOG_FIND="$LOG_BASE"_find.log
    echo "==== Processing: $d (both blends) ====" | tee -a "$MASTER_LOG" "$LOG_FIND"
    echo "Starting find_direction.py for $d at $(date)" | tee -a "$MASTER_LOG" "$LOG_FIND"
    if ! python3 "$SCRIPT_DIR/find_direction.py" --dir "$d" --suffix ".jpg" --model "$SCRIPT_DIR/last11scor_3class_31_07.pt" --conf 0.05 --iou 0.6 >> "$LOG_FIND" 2>&1; then
      echo "find_direction FAILED for $d" | tee -a "$MASTER_LOG" "$LOG_FIND"
    fi

    # Candidate 1: feather blend
    LOG_FEATHER="$LOG_BASE"_feather.log
    echo "Starting run_full_pipeline.py (feather) for $d at $(date)" | tee -a "$MASTER_LOG" "$LOG_FEATHER"
    if ! python3 "$SCRIPT_DIR/run_full_pipeline.py" \
         --img-dir "$d" \
         --img-suffix ".jpg" \
         --conf 0.05 \
         --iou 0.6 \
         --seam-consec 2 \
         --min-gu-conf 0.20 \
         --blend feather \
         --seam-width 5 \
         --lock-dy \
         --stride 1 >> "$LOG_FEATHER" 2>&1; then
      echo "run_full_pipeline (feather) FAILED for $d" | tee -a "$MASTER_LOG" "$LOG_FEATHER"
    fi

    # Candidate 2: seam blend
    LOG_SEAM="$LOG_BASE"_seam.log
    echo "Starting run_full_pipeline.py (seam) for $d at $(date)" | tee -a "$MASTER_LOG" "$LOG_SEAM"
    if ! python3 "$SCRIPT_DIR/run_full_pipeline.py" \
         --img-dir "$d" \
         --img-suffix ".jpg" \
         --conf 0.05 \
         --iou 0.6 \
         --seam-consec 2 \
         --min-gu-conf 0.20 \
         --blend seam \
         --seam-width 5 \
         --lock-dy \
         --stride 1 >> "$LOG_SEAM" 2>&1; then
      echo "run_full_pipeline (seam) FAILED for $d" | tee -a "$MASTER_LOG" "$LOG_SEAM"
    fi

    echo "==== Finished: $d at $(date) ====" | tee -a "$MASTER_LOG"
  done < <(find "$ROOT" -maxdepth 1 -type d | grep -E '_[0-9]+_[0-9]+_' | sort)
done

echo "Batch (both blends) finished at $(date)" >> "$MASTER_LOG"
exit 0
