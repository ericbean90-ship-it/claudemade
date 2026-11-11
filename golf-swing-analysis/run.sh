#!/usr/bin/env bash
# Golf swing "pro ghost" pipeline.
# Usage: ./run.sh <front.mp4> <dtl.mp4> <workdir>
# Before step 4, edit <workdir>/keyframes.json (P1..P10 frame numbers of YOUR swing) and the
# ball pixel coordinates below if your camera setup differs.
set -euo pipefail
FRONT=$1; DTL=$2; W=$3
T="$(cd "$(dirname "$0")" && pwd)/tools"
mkdir -p "$W/vid" "$W/pose" "$W/models" "$W/out" "$W/frames"
cp "$FRONT" "$W/vid/front.mp4"; cp "$DTL" "$W/vid/back.mp4"

MODEL="$W/models/pose_heavy.task"
[ -f "$MODEL" ] || curl -sS -o "$MODEL" \
  https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_heavy/float16/latest/pose_landmarker_heavy.task

# 1) 2D pose per frame (both views)
python3 "$T/extract_pose.py" "$W/vid/front.mp4" "$MODEL" "$W/pose/front.npz" || true
python3 "$T/extract_pose.py" "$W/vid/back.mp4"  "$MODEL" "$W/pose/back.npz"  || true
# 2) joint 2-camera calibration from the address pose
python3 "$T/calib.py" "$W"
# 3) ball position (pixel coords: front u v, dtl u v)
python3 "$T/ball.py" "$W" ${BALL_FRONT_U:-95} ${BALL_FRONT_V:-609} ${BALL_DTL_U:-331} ${BALL_DTL_V:-462}
[ -f "$W/keyframes.json" ] || cp "$(dirname "$T")/keyframes.example.json" "$W/keyframes.json"
# 4) user 3D metrics, keyframe preview, ghost render
python3 "$T/user3d.py" "$W"
python3 "$T/preview.py" "$W"
python3 "$T/render.py" "$W" "$W/out"
echo "outputs in $W/out"
