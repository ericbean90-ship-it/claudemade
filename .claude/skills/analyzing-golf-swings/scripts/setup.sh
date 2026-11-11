#!/usr/bin/env bash
# One-time (per container) setup for the golf swing pipeline. Idempotent.
set -euo pipefail
if ! ldconfig -p | grep -q libEGL.so.1 || ! fc-list | grep -qi nanum; then
  (apt-get install -y libegl1 libgles2 fonts-nanum >/dev/null 2>&1) || \
  (apt-get update -qq >/dev/null 2>&1 && apt-get install -y libegl1 libgles2 fonts-nanum >/dev/null 2>&1)
fi
python3 -c "import mediapipe, cv2, scipy, PIL" 2>/dev/null || \
  pip install -q -r "$(dirname "$0")/requirements.txt" 2>&1 | grep -v -i warning || true
M="$HOME/.cache/golf-swing/pose_landmarker_heavy.task"
if [ ! -s "$M" ]; then
  mkdir -p "$(dirname "$M")"
  curl -sS -o "$M" https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_heavy/float16/latest/pose_landmarker_heavy.task
fi
python3 -c "import mediapipe, cv2, scipy, PIL; print('setup ok')"
