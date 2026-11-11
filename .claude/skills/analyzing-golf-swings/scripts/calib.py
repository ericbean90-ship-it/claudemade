"""Joint calibration of the two cameras (front / down-the-line) from the address pose.

Unknowns: 3D address skeleton X (in MediaPipe-world metric units) + per-camera
(rvec, tvec, f, k1).  Residuals: reprojection error in both views (visibility
weighted) + weak prior toward the MediaPipe world skeleton + L/R bone symmetry.

Then a golfer frame G is defined:
  origin = mid-ankle point dropped to the floor plane
  +Y up (floor normal), +X toward target (golfer's left), +Z toward the ball.
Saves calib.npz with everything needed to project G-frame points into each view.
"""
import sys
import json
import numpy as np
import cv2
from scipy.optimize import least_squares

S = sys.argv[1]  # scratch dir
F = np.load(f"{S}/pose/front.npz")
B = np.load(f"{S}/pose/back.npz")
W, H = int(F["w"]), int(F["h"])
A0, A1 = 120, 196  # address window (frames)

J = [0, 7, 8, 11, 12, 13, 14, 15, 16, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32]
Xmp = np.nanmedian(F["world"][A0:A1], axis=0)[J]          # (n,3) front-cam aligned
xf = np.nanmedian(F["img"][A0:A1, :, :2], axis=0)[J]
xb = np.nanmedian(B["img"][A0:A1, :, :2], axis=0)[J]
vf = np.clip(np.nanmedian(F["img"][A0:A1, :, 3], axis=0)[J], 0.15, 1)
vb = np.clip(np.nanmedian(B["img"][A0:A1, :, 3], axis=0)[J], 0.15, 1)
idx = {j: i for i, j in enumerate(J)}
cx, cy = W / 2, H / 2

def project(X, rvec, tvec, f, k1):
    R, _ = cv2.Rodrigues(rvec)
    P = X @ R.T + tvec
    x = P[:, 0] / P[:, 2]
    y = P[:, 1] / P[:, 2]
    r2 = x * x + y * y
    d = 1 + k1 * r2
    return np.stack([f * x * d + cx, f * y * d + cy], 1), P[:, 2]

def init_cam(X, x2d, w):
    best = None
    for f in np.linspace(250, 1400, 47):
        K = np.array([[f, 0, cx], [0, f, cy], [0, 0, 1.0]])
        ok, rv, tv = cv2.solvePnP(X.astype(np.float64), x2d.astype(np.float64), K, None,
                                  flags=cv2.SOLVEPNP_EPNP)
        if not ok:
            continue
        ok, rv, tv = cv2.solvePnP(X, x2d, K, None, rv, tv, True, flags=cv2.SOLVEPNP_ITERATIVE)
        p, z = project(X, rv.ravel(), tv.ravel(), f, 0)
        if np.any(z <= 0):
            continue
        e = np.sqrt(np.average(np.sum((p - x2d) ** 2, 1), weights=w))
        if best is None or e < best[0]:
            best = (e, rv.ravel(), tv.ravel(), f)
    return best

bf = init_cam(Xmp, xf, vf)
bb = init_cam(Xmp, xb, vb)
print("init front rms %.2f f=%.0f | back rms %.2f f=%.0f" % (bf[0], bf[3], bb[0], bb[3]))

bones = [(11, 13), (13, 15), (12, 14), (14, 16), (23, 25), (25, 27), (24, 26), (26, 28),
         (11, 23), (12, 24), (27, 29), (28, 30), (27, 31), (28, 32)]
pairs = [((11, 13), (12, 14)), ((13, 15), (14, 16)), ((23, 25), (24, 26)), ((25, 27), (26, 28)),
         ((11, 23), (12, 24))]
blen0 = {b: np.linalg.norm(Xmp[idx[b[0]]] - Xmp[idx[b[1]]]) for b in bones}

n = len(J)
def unpack(p):
    X = p[:3 * n].reshape(n, 3)
    cf = p[3 * n:3 * n + 8]
    cb = p[3 * n + 8:3 * n + 16]
    return X, cf, cb

def resid(p):
    X, cf, cb = unpack(p)
    pf, _ = project(X, cf[:3], cf[3:6], cf[6], cf[7])
    pb, _ = project(X, cb[:3], cb[3:6], cb[6], cb[7])
    r = [((pf - xf) * np.sqrt(vf)[:, None]).ravel(),
         ((pb - xb) * np.sqrt(vb)[:, None]).ravel(),
         60.0 * (X - Xmp).ravel()]
    for b in bones:
        r.append([300.0 * (np.linalg.norm(X[idx[b[0]]] - X[idx[b[1]]]) - blen0[b])])
    for a, b in pairs:
        la = np.linalg.norm(X[idx[a[0]]] - X[idx[a[1]]])
        lb = np.linalg.norm(X[idx[b[0]]] - X[idx[b[1]]])
        r.append([600.0 * (la - lb)])
    r.append([20.0 * cf[7], 20.0 * cb[7]])
    return np.concatenate([np.ravel(x) for x in r])

p0 = np.concatenate([Xmp.ravel(), bf[1], bf[2], [bf[3], 0.0], bb[1], bb[2], [bb[3], 0.0]])
sol = least_squares(resid, p0, loss="soft_l1", f_scale=4.0, max_nfev=4000)
X, cf, cb = unpack(sol.x)
pf, _ = project(X, cf[:3], cf[3:6], cf[6], cf[7])
pb, _ = project(X, cb[:3], cb[3:6], cb[6], cb[7])
ef = np.linalg.norm(pf - xf, axis=1)
eb = np.linalg.norm(pb - xb, axis=1)
print("BA front rms %.2f px (median %.2f) f=%.0f k1=%.3f" % (np.sqrt(np.mean(ef ** 2)), np.median(ef), cf[6], cf[7]))
print("BA back  rms %.2f px (median %.2f) f=%.0f k1=%.3f" % (np.sqrt(np.mean(eb ** 2)), np.median(eb), cb[6], cb[7]))
for j, a, b in zip(J, ef, eb):
    print(f"  j{j:2d} front {a:5.1f}  back {b:5.1f}")

# ---------------- golfer frame ----------------
g = lambda j: X[idx[j]]
feet = np.stack([g(29), g(30), g(31), g(32)])
c = feet.mean(0)
_, _, vt = np.linalg.svd(feet - c)
up = vt[2]
hipc = (g(23) + g(24)) / 2
if np.dot(hipc - c, up) < 0:
    up = -up
# blend floor normal with (camera -y) for robustness (feet are only 4 noisy pts)
cam_up = np.array([0, -1.0, 0])
print("floor-normal vs cam-up angle %.1f deg" % np.degrees(np.arccos(np.clip(up @ cam_up, -1, 1))))
up = up + cam_up
up /= np.linalg.norm(up)
tgt = g(27) - g(28)                       # right ankle -> left ankle (toward target)
tgt = tgt - up * (tgt @ up)
tgt /= np.linalg.norm(tgt)
fwd = np.cross(tgt, up)                    # X cross Y = Z (toward ball)
handc = (g(15) + g(16)) / 2
if np.dot(handc - hipc, fwd) < 0:
    raise SystemExit("frame handedness check failed")
mid_ank = (g(27) + g(28)) / 2
heel_h = np.mean([(g(29) - c) @ up, (g(30) - c) @ up])
origin = mid_ank - up * ((mid_ank - c) @ up)  # drop to floor plane through feet
Rg = np.stack([tgt, up, fwd])               # rows: G axes expressed in mp coords

def to_G(P):
    return (P - origin) @ Rg.T

XG = to_G(X)
names = {0: "nose", 7: "Lear", 8: "Rear", 11: "Lsh", 12: "Rsh", 13: "Lel", 14: "Rel", 15: "Lwr",
         16: "Rwr", 23: "Lhip", 24: "Rhip", 25: "Lkn", 26: "Rkn", 27: "Lank", 28: "Rank",
         29: "Lheel", 30: "Rheel", 31: "Ltoe", 32: "Rtoe"}
print("address skeleton in golfer frame (m): X=target, Y=up, Z=ball")
for j in J:
    print(f"  {names[j]:6s} {XG[idx[j]].round(3)}")

np.savez(f"{S}/calib.npz", J=np.array(J), X_mp=X, XG=XG, origin=origin, Rg=Rg,
         cam_front=cf, cam_back=cb, W=W, H=H)
print("saved")
