"""Prepare the Modelgolf 'model pro' render (YouTube XlyQ92c8I-w, 4:09) for overlay.

Usage: python ref_prepare.py <workdir> <modelgolf_video.mp4> [start_sec=181.0] [dur_sec=9]
Extracts the 9-iron continuous swing (default 3:01-3:10), keeps only frames where the pose
changes (the render updates ~7.5x/s), separates the two mannequins and runs pose on them.
Then: python ref_overlay.py <workdir> <workdir>/out
"""
import os
import sys
import json
import glob
import subprocess
import numpy as np
import cv2

S, VID = sys.argv[1], sys.argv[2]
START = float(sys.argv[3]) if len(sys.argv) > 3 else 181.0
DUR = float(sys.argv[4]) if len(sys.argv) > 4 else 9.0
T = os.path.dirname(os.path.abspath(__file__))
R = f"{S}/ref"

dur = float(subprocess.check_output(["ffprobe", "-v", "error", "-show_entries", "format=duration",
                                     "-of", "csv=p=0", VID]).decode().strip())
if abs(dur - 249.4) > 2 and len(sys.argv) <= 3:
    sys.exit(f"영상 길이 {dur:.1f}s: Modelgolf 원본(249.4s)과 다릅니다. 스윙 구간 시작초/길이를 인자로 주세요.")

os.makedirs(f"{R}/fr", exist_ok=True)
for f in glob.glob(f"{R}/fr/*.png"):
    os.remove(f)
subprocess.run(["ffmpeg", "-v", "error", "-y", "-ss", str(START), "-t", str(DUR), "-i", VID,
                "-vsync", "0", f"{R}/fr/%04d.png"], check=True)
fs = sorted(glob.glob(f"{R}/fr/*.png"))
prev, uniq = None, []
for k, f in enumerate(fs):
    g = cv2.imread(f, 0).astype(np.int16)
    g[318:] = 0
    if prev is None or np.abs(g - prev).mean() >= 0.05:
        uniq.append(k)
    prev = g
json.dump(uniq, open(f"{R}/uniq.json", "w"))
print("frames", len(fs), "unique poses", len(uniq))
subprocess.run([sys.executable, f"{T}/ref_split.py", S], check=True)
subprocess.run(f"{sys.executable} {T}/ref_pose.py {S} 2>&1 | grep detected", shell=True, check=False)
