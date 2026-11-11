"""Triangulate the ball (address) from its pixel position in each view.

Usage: python ball.py <workdir> <front_u> <front_v> <dtl_u> <dtl_v>
If the ball is hidden in the front view, pass the point where the shaft line meets the
bottom of the frame; the DTL ray dominates the result anyway.
Writes <workdir>/ball.npy (golfer frame, metres).
"""
import os
import sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from geom import load_cams, triangulate

S = sys.argv[1]
fu, fv, bu, bv = map(float, sys.argv[2:6])
cf, cb, _ = load_cams(S)
p, gap = triangulate(cf, (fu, fv), cb, (bu, bv))
p[1] = 0.021  # ball radius above the floor
np.save(f"{S}/ball.npy", p)
print("ball (G frame, m):", p.round(3), "ray gap %.1f cm" % (gap * 100))
