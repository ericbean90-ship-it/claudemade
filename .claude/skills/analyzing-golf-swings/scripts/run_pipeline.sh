#!/usr/bin/env bash
# Usage: bash run_pipeline.sh <front.mp4> <back(dtl).mp4> <workdir> [stage]
#   stage: all (default) | analyze (pose+calib+keyframes+metrics, no videos) | render (videos only)
# After 'analyze', LOOK at <workdir>/out/keyframes_sheet.jpg; fix <workdir>/keyframes.json if needed,
# then rerun with stage 'metrics' and 'render'.
set -euo pipefail
FRONT=$1; DTL=$2; W=$3; STAGE=${4:-all}
T="$(cd "$(dirname "$0")" && pwd)"
mkdir -p "$W/vid" "$W/pose" "$W/models" "$W/out" "$W/frames"
[ -f "$W/vid/front.mp4" ] || cp "$FRONT" "$W/vid/front.mp4"
[ -f "$W/vid/back.mp4" ]  || cp "$DTL"   "$W/vid/back.mp4"
[ -e "$W/models/pose_heavy.task" ] || ln -s "$HOME/.cache/golf-swing/pose_landmarker_heavy.task" "$W/models/pose_heavy.task"
q() { grep -v -E "^(W0|I0|INFO|WARNING|  File|    |Traceback|TypeError|Exception ignored|E0000)" || true; }

if [ "$STAGE" = all ] || [ "$STAGE" = analyze ]; then
  [ -f "$W/pose/front.npz" ] || python3 "$T/extract_pose.py" "$W/vid/front.mp4" "$W/models/pose_heavy.task" "$W/pose/front.npz" 2>&1 | q
  [ -f "$W/pose/back.npz" ]  || python3 "$T/extract_pose.py" "$W/vid/back.mp4"  "$W/models/pose_heavy.task" "$W/pose/back.npz"  2>&1 | q
  python3 "$T/calib.py" "$W" | tail -4
  python3 "$T/ball.py" "$W" ${BALL_FRONT_U:-95} ${BALL_FRONT_V:-609} ${BALL_DTL_U:-331} ${BALL_DTL_V:-462}
  [ -f "$W/keyframes.json" ] || python3 "$T/detect_keyframes.py" "$W"
fi
if [ "$STAGE" = all ] || [ "$STAGE" = analyze ] || [ "$STAGE" = metrics ]; then
  python3 "$T/user3d.py" "$W" > "$W/out/user3d.txt"
  python3 "$T/metrics.py" "$W" "$FRONT" "$DTL"
fi
if [ "$STAGE" = all ] || [ "$STAGE" = render ]; then
  python3 "$T/render.py" "$W" "$W/out" | tail -1
fi
echo "done: $W/out"
