"""Zoomed, gridded crops around where the ball should be, to read its pixel position by eye.

Usage: python ball_grid.py <workdir> [frame]
Needs calib.npz (run after the analyze stage). Writes out/ball_front.jpg and out/ball_dtl.jpg with
pixel coordinates on the grid; then rerun:  python ball.py <workdir> <fu> <fv> <bu> <bv>
"""
import os
import sys
import numpy as np
import cv2

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from geom import load_cams

S = sys.argv[1]
cf, cb, _ = load_cams(S)
guess = np.array([0.0, 0.02, 0.78])          # ~0.8 m in front of the ankle line, mid-stance
fr_i = int(sys.argv[2]) if len(sys.argv) > 2 else 60
for name, cam, vid in (("front", cf, "front"), ("dtl", cb, "back")):
    c = cv2.VideoCapture(f"{S}/vid/{vid}.mp4")
    c.set(cv2.CAP_PROP_POS_FRAMES, fr_i)
    ok, f = c.read()
    u, v = cam.project(guess)[0][0]
    H, W = f.shape[:2]
    x0, x1 = int(np.clip(u - 110, 0, W - 220)), int(np.clip(u + 110, 220, W))
    y0, y1 = int(np.clip(v - 90, 0, H - 180)), int(np.clip(v + 90, 180, H))
    k = 4
    crop = cv2.resize(f[y0:y1, x0:x1], None, fx=k, fy=k, interpolation=cv2.INTER_CUBIC)
    for gx in range(x0 - x0 % 10, x1, 10):
        cv2.line(crop, ((gx - x0) * k, 0), ((gx - x0) * k, crop.shape[0]), (0, 255, 255) if gx % 50 == 0 else (90, 90, 90), 1)
        if gx % 20 == 0:
            cv2.putText(crop, str(gx), ((gx - x0) * k + 2, 12), 0, 0.4, (0, 255, 255), 1)
    for gy in range(y0 - y0 % 10, y1, 10):
        cv2.line(crop, (0, (gy - y0) * k), (crop.shape[1], (gy - y0) * k), (0, 255, 255) if gy % 50 == 0 else (90, 90, 90), 1)
        if gy % 20 == 0:
            cv2.putText(crop, str(gy), (2, (gy - y0) * k - 2), 0, 0.4, (0, 255, 255), 1)
    cv2.imwrite(f"{S}/out/ball_{name}.jpg", crop)
    print(f"{name}: crop x{x0}-{x1} y{y0}-{y1} -> {S}/out/ball_{name}.jpg")
