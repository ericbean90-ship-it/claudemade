"""Split the reference render (two mannequins on black) into DTL (left) and FRONT (right) layers.

For each unique pose frame: luminance matte -> connected components -> the two largest blobs are
the bodies; every other blob (club pieces, ball) goes to the nearer body. Writes RGBA PNGs.
"""
import sys
import json
import glob
import numpy as np
import cv2

S = sys.argv[1]
R = f"{S}/ref"
fs = sorted(glob.glob(f"{R}/fr/*.png"))
uniq = json.load(open(f"{R}/uniq.json"))
import os
os.makedirs(f"{R}/dtl", exist_ok=True)
os.makedirs(f"{R}/front", exist_ok=True)

for k in uniq:
    im = cv2.imread(fs[k])
    g = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    alpha = np.clip((g.astype(np.float32) - 10) / 50.0, 0, 1)
    m = (g > 18).astype(np.uint8)
    md = cv2.dilate(m, np.ones((3, 3), np.uint8))
    n, lab, st, cen = cv2.connectedComponentsWithStats(md, 8)
    order = np.argsort(-st[1:, cv2.CC_STAT_AREA]) + 1
    bodies = sorted(order[:2], key=lambda i: cen[i][0])   # left = DTL, right = FRONT
    dist = []
    for b in bodies:
        dist.append(cv2.distanceTransform((lab != b).astype(np.uint8), cv2.DIST_L2, 3))
    own = np.zeros(g.shape, np.int8) - 1
    for i in range(1, n):
        sel = lab == i
        if i == bodies[0]:
            own[sel] = 0
        elif i == bodies[1]:
            own[sel] = 1
        elif st[i, cv2.CC_STAT_AREA] < 25:
            continue                                   # flying ball / specks
        elif st[i, cv2.CC_STAT_TOP] >= 312:
            continue                                   # subtitle glyphs live entirely in the bottom band
        else:
            own[sel] = 0 if dist[0][sel].mean() < dist[1][sel].mean() else 1
    for side, name in ((0, "dtl"), (1, "front")):
        a = alpha * (own == side)
        rgba = np.dstack([im, (a * 255).astype(np.uint8)])
        cv2.imwrite(f"{R}/{name}/{k:04d}.png", rgba)
print("split", len(uniq))
