"""Strip of numbered frames (front | back) to pick a keyframe by eye.

Usage: python frames_strip.py <workdir> <start> <end> [step=4]  -> out/strip_<start>_<end>.jpg
"""
import sys
import numpy as np
import cv2

S, a, b = sys.argv[1], int(sys.argv[2]), int(sys.argv[3])
step = int(sys.argv[4]) if len(sys.argv) > 4 else 4
caps = {v: cv2.VideoCapture(f"{S}/vid/{v}.mp4") for v in ("front", "back")}
tiles = []
for i in range(a, b + 1, step):
    row = []
    for v, c in caps.items():
        c.set(cv2.CAP_PROP_POS_FRAMES, i)
        ok, f = c.read()
        row.append(cv2.resize(f, None, fx=0.5, fy=0.5))
    t = np.hstack(row)
    cv2.rectangle(t, (0, 0), (t.shape[1], 22), (0, 0, 0), -1)
    cv2.putText(t, f"f{i}", (6, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 1, cv2.LINE_AA)
    tiles.append(t)
while len(tiles) % 4:
    tiles.append(np.zeros_like(tiles[0]))
out = np.vstack([np.hstack(tiles[r:r + 4]) for r in range(0, len(tiles), 4)])
cv2.imwrite(f"{S}/out/strip_{a}_{b}.jpg", out, [cv2.IMWRITE_JPEG_QUALITY, 85])
print(f"{S}/out/strip_{a}_{b}.jpg")
