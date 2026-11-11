"""Shared camera / frame helpers.  Golfer frame G: X=target, Y=up, Z=toward ball (metres)."""
import numpy as np
import cv2


class Cam:
    def __init__(self, params, Rg, origin, W, H):
        self.rv, self.tv, self.f, self.k1 = params[:3], params[3:6], params[6], params[7]
        self.R, _ = cv2.Rodrigues(self.rv)
        self.Rg, self.org, self.W, self.H = Rg, origin, W, H

    def to_cam(self, PG):
        Pm = np.asarray(PG) @ self.Rg + self.org
        return Pm @ self.R.T + self.tv

    def project(self, PG):
        P = self.to_cam(np.atleast_2d(PG))
        x = P[:, 0] / P[:, 2]
        y = P[:, 1] / P[:, 2]
        d = 1 + self.k1 * (x * x + y * y)
        return np.stack([self.f * x * d + self.W / 2, self.f * y * d + self.H / 2], 1), P[:, 2]

    def ray(self, uv):
        xd = (uv[0] - self.W / 2) / self.f
        yd = (uv[1] - self.H / 2) / self.f
        x, y = xd, yd
        for _ in range(30):
            d = 1 + self.k1 * (x * x + y * y)
            x, y = xd / d, yd / d
        c = -self.R.T @ self.tv
        dm = self.R.T @ np.array([x, y, 1.0])
        o = (c - self.org) @ self.Rg.T
        dG = dm @ self.Rg.T
        return o, dG / np.linalg.norm(dG)


def load_cams(S):
    C = np.load(f"{S}/calib.npz")
    W, H = int(C["W"]), int(C["H"])
    cf = Cam(C["cam_front"], C["Rg"], C["origin"], W, H)
    cb = Cam(C["cam_back"], C["Rg"], C["origin"], W, H)
    XG = {int(j): C["XG"][i] for i, j in enumerate(C["J"])}
    return cf, cb, XG


def triangulate(c1, uv1, c2, uv2):
    o1, d1 = c1.ray(uv1)
    o2, d2 = c2.ray(uv2)
    w = o1 - o2
    a, b, c = d1 @ d1, d1 @ d2, d2 @ d2
    d, e = d1 @ w, d2 @ w
    den = a * c - b * b
    s = (b * e - c * d) / den
    t = (a * e - b * d) / den
    p1, p2 = o1 + s * d1, o2 + t * d2
    return (p1 + p2) / 2, np.linalg.norm(p1 - p2)


def rot(axis, ang):
    axis = np.asarray(axis, float)
    axis = axis / np.linalg.norm(axis)
    R, _ = cv2.Rodrigues(axis * ang)
    return R


def slerp(a, b, t):
    a = a / np.linalg.norm(a)
    b = b / np.linalg.norm(b)
    d = np.clip(a @ b, -1, 1)
    om = np.arccos(d)
    if om < 1e-6:
        return a
    return (np.sin((1 - t) * om) * a + np.sin(t * om) * b) / np.sin(om)
