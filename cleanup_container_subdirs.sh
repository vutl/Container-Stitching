#!/usr/bin/env bash
# Safe cleanup script: remove subdirectories inside each container folder under
# the two roots and remove only the `direction.txt` file. Do NOT remove image files
# that live at the top of the container folder.
#
# Usage:
#   # dry-run (default)
#   ./cleanup_container_subdirs.sh --dry-run
#
#   # actually delete after confirming
#   ./cleanup_container_subdirs.sh --yes
#
#   # target different roots (space separated)
#   ./cleanup_container_subdirs.sh --yes --roots "path1" "path2"

set -euo pipefail
ROOTS=("downloads/gdrive_1sASbG/imgs_stit/left" "downloads/gdrive_1sASbG/imgs_stit/right")
DRY_RUN=true
CONFIRM=false

# parse args
while [[ $# -gt 0 ]]; do
  case "$1" in
    --dry-run)
      DRY_RUN=true; shift;;
    --yes|-y)
      DRY_RUN=false; CONFIRM=true; shift;;
    --roots)
      shift
      # collect remaining args until next flag or end
      NEW=()
      while [[ $# -gt 0 && "$1" != --* ]]; do
        NEW+=("$1"); shift
      done
      if [ ${#NEW[@]} -gt 0 ]; then
        ROOTS=(${NEW[@]})
      fi
      ;;
    --help|-h)
      sed -n '1,200p' "$0"
      exit 0;;
    *)
      echo "Unknown arg: $1"; echo "Use --help"; exit 2;;
  esac
done

# Confirm when not dry-run and not explicitly confirmed
if ! $DRY_RUN && ! $CONFIRM; then
  echo "Warning: you are about to delete directories. Re-run with --yes to proceed." >&2
  exit 3
fi

echo "Roots: ${ROOTS[*]}"
echo "Mode: $( $DRY_RUN && echo DRY-RUN || echo PERFORM )"

for R in "${ROOTS[@]}"; do
  if [ ! -d "$R" ]; then
    echo "Root not found, skipping: $R"
    continue
  fi
  # iterate each immediate child dir (container folders)
  echo "\nScanning containers under: $R"
  while IFS= read -r C; do
    # ensure it's a directory
    if [ -z "$C" ] || [ ! -d "$C" ]; then
      continue
    fi
    echo "\nContainer: $C"
    # safety check: require at least one image file inside container (jpg/png)
    IMG_COUNT=$(find "$C" -maxdepth 1 -type f \( -iname '*.jpg' -o -iname '*.jpeg' -o -iname '*.png' \) | wc -l || true)
    if [ "$IMG_COUNT" -eq 0 ]; then
      echo "  -> No top-level images found in $C; skipping for safety"
      continue
    fi

    # list subdirectories to remove (immediate children only)
    SUBDIRS=()
    while IFS= read -r d; do
      SUBDIRS+=("$d")
    done < <(find "$C" -mindepth 1 -maxdepth 1 -type d -print)

    if [ ${#SUBDIRS[@]} -eq 0 ]; then
      echo "  -> No subdirectories to remove"
    else
      echo "  -> Subdirectories to remove:"
      for s in "${SUBDIRS[@]}"; do
        echo "     $s"
      done
      if $DRY_RUN; then
        echo "  (dry-run: not removing)"
      else
        for s in "${SUBDIRS[@]}"; do
          echo "  Removing: $s"
          rm -rf -- "$s"
        done
      fi
    fi

    # delete direction.txt if present at container root
    if [ -f "$C/direction.txt" ]; then
      echo "  -> Found direction.txt"
      if $DRY_RUN; then
        echo "     (dry-run: not removing $C/direction.txt)"
      else
        rm -f -- "$C/direction.txt" && echo "     removed direction.txt"
      fi
    else
      echo "  -> No direction.txt found"
    fi

  done < <(find "$R" -mindepth 1 -maxdepth 1 -type d -print | sort)
done

echo "\nDone."

exit 0
