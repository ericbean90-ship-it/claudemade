"""Triangulate the user's swing (both views) into the golfer frame and print metrics."""
import os
import sys
import json
import numpy as np
from scipy.ndimage import median_filter, gaussian_filter1d
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from geom import load_cams, triangulate

S = sys.argv[1]
cf, cb, XG = load_cams(S)
F = np.load(f"{S}/pose/front.npz")["img"]
B = np.load(f"{S}/pose/back.npz")["img"]
F = median_filter(F, size=(5, 1, 1), mode="nearest")
B = median_filter(B, size=(5, 1, 1), mode="nearest")
N = F.shape[0]

groups = {"head": [0, 7, 8], "sho": [11, 12], "pel": [23, 24], "hands": [15, 16]}
out = {}
for k, js in groups.items():
    pts, gaps = [], []
    for i in range(N):
        uf = F[i, js, :2].mean(0)
        ub = B[i, js, :2].mean(0)
        p, g = triangulate(cf, uf, cb, ub)
        pts.append(p)
        gaps.append(g)
    P = gaussian_filter1d(np.array(pts), 1.5, axis=0)
    out[k] = P
    print(f"{k:6s} median ray gap {np.median(gaps)*100:.1f} cm")

KF = json.load(open(f"{S}/keyframes.json"))
a = KF["P1"]
print("\nphase   frame | head dX dY dZ (cm)  | pelvis dX dY dZ (cm) | spine fwd  face-on tilt | hands X Y Z (m)")
for name, i in KF.items():
    h = (out["head"][i] - out["head"][a]) * 100
    p = (out["pel"][i] - out["pel"][a]) * 100
    sp = out["sho"][i] - out["pel"][i]
    fwd = np.degrees(np.arctan2(sp[2], sp[1]))
    lat = np.degrees(np.arctan2(-sp[0], sp[1]))  # + = upper body leans away from target
    hd = out["hands"][i]
    print(f"{name:5s} {i:5d} | {h[0]:6.1f} {h[1]:6.1f} {h[2]:6.1f} | {p[0]:6.1f} {p[1]:6.1f} {p[2]:6.1f} | {fwd:6.1f} {lat:7.1f} | {hd.round(3)}")

np.savez(f"{S}/user3d.npz", **out)
