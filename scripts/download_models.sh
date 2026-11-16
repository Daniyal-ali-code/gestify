#!/usr/bin/env bash
# Download model files that are not tracked in git.
set -e
cd "$(dirname "$0")/.."

URL="https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task"
DEST="assets/hand_landmarker.task"

mkdir -p assets
if [ -f "$DEST" ]; then
  echo "Already present: $DEST"
else
  echo "Downloading hand_landmarker.task ..."
  curl -L --fail -o "$DEST" "$URL"
  echo "Saved to $DEST"
fi
