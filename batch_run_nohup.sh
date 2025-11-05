#!/usr/bin/env bash
# Run the container batch processing under nohup so it continues after the shell/VSCode exits.
# Writes a master log at batch_logs/batch_master.log and per-container logs at batch_logs/<side>_<name>.log

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LOG_DIR="$SCRIPT_DIR/batch_logs"
mkdir -p "$LOG_DIR"
ROOTS=("downloads/gdrive_1sASbG/imgs_stit/right" "downloads/gdrive_1sASbG/imgs_stit/left")
MASTER_LOG="$LOG_DIR/batch_master.log"

echo "Batch run started at $(date)" >> "$MASTER_LOG"

for ROOT in "${ROOTS[@]}"; do
  if [ ! -d "$ROOT" ]; then
    echo "Root not found, skipping: $ROOT" | tee -a "$MASTER_LOG"
    continue
  fi
  echo "Scanning: $ROOT" | tee -a "$MASTER_LOG"
  # find candidate container dirs under root
  while IFS= read -r d; do
    # skip the root dir itself
    if [ -z "$d" ] || [ "$d" = "$ROOT" ]; then
      continue
    fi
    NAME=$(basename "$d")
    SIDE=$(echo "$d" | grep -o -E '(left|right)' || true)
    LOG="$LOG_DIR/${SIDE}_${NAME}.log"

    echo "==== Processing: $d -> $LOG ====" | tee -a "$MASTER_LOG" "$LOG"
    echo "Starting find_direction.py for $d at $(date)" | tee -a "$MASTER_LOG" "$LOG"
    # run find_direction (don't fail the whole batch if one container fails)
    if ! python3 "$SCRIPT_DIR/find_direction.py" --dir "$d" --suffix ".jpg" --model "$SCRIPT_DIR/last11scor_3class_31_07.pt" --conf 0.05 --iou 0.6 >> "$LOG" 2>&1; then
      echo "find_direction FAILED for $d" | tee -a "$MASTER_LOG" "$LOG"
    fi

    echo "Starting run_full_pipeline.py for $d at $(date)" | tee -a "$MASTER_LOG" "$LOG"
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
         --stride 1 >> "$LOG" 2>&1; then
      echo "run_full_pipeline FAILED for $d" | tee -a "$MASTER_LOG" "$LOG"
    fi

    echo "==== Finished: $d at $(date) ====" | tee -a "$MASTER_LOG" "$LOG"
  done < <(find "$ROOT" -maxdepth 1 -type d | grep -E '_[0-9]+_[0-9]+_' | sort)
done

echo "Batch run finished at $(date)" >> "$MASTER_LOG"

# Keep the script exit code as success (0) even if some containers failed; inspect per-container logs for failures
exit 0
