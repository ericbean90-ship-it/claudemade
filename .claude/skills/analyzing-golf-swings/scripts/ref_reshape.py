"""Reshape one model-pro mannequin layer to the user's build, keeping every joint angle.

  * head      uniform scale about the neck base                (body-shape.json "head")
  * shoulders shoulder joints pulled toward the neck; the arms ride along unchanged ("shoulder")
              -> skeleton-driven 2D warp (soft nearest-bone blending, backward mapping)
  * girth     silhouette edges trimmed by (1 - factor) x the local half-thickness of the nearest
              bone ("torso_girth", "limb_girth"). Trimming only the outline never mixes limbs that
              overlap in the picture (a crossing-leg warp did).
  * camera    the head also gets the user's camera perspective (m_head): a close face-on camera draws
              a head over the ball 20-40 % larger than the hips, and the mannequin render has no
              perspective. Only the head: the body-size scale already holds the body's average
              perspective, and re-projecting the whole body widened the shoulders again.
Bone directions never change in the body-shape steps, so the swing angles are the mannequin's own.
The club is cut out (thin dark line), the body is reshaped, and the club is pasted back with the hands.
"""
import numpy as np
import cv2

LIMBS = [(11, 13), (13, 15), (12, 14), (14, 16), (23, 25), (25, 27), (24, 26), (26, 28)]
FEET = [(27, 31), (28, 32)]
HANDS = [(15, 19), (16, 20)]
RADIUS = {"torso": 30.0, "clav": 11.0, "hip": 14.0, "head": 13.0, "upper": 8.0, "lower": 7.0,
          "thigh": 12.0, "shin": 8.5, "foot": 6.0, "hand": 6.0}       # mannequin px (render is 640x360)


def split_club(lay):
    """-> body RGBA, club RGBA (float32). Club = thin dark structure + the parts of it inside the body."""
    a = lay[..., 3]
    lum = cv2.cvtColor(np.clip(lay[..., :3], 0, 255).astype(np.uint8), cv2.COLOR_BGR2GRAY)
    solid = (a > 128).astype(np.uint8)
    core = cv2.morphologyEx(solid, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)))
    n, lab, st, _ = cv2.connectedComponentsWithStats(core)
    body = (lab == 1 + np.argmax(st[1:, cv2.CC_STAT_AREA])) if n > 1 else core.astype(bool)
    body = cv2.dilate(body.astype(np.uint8), np.ones((3, 3), np.uint8)).astype(bool)
    outside = (a > 40) & ~body
    dark = ((lum < 100) & (a > 40)).astype(np.uint8)
    n, lab = cv2.connectedComponents(dark)
    touch = np.unique(lab[outside & (dark > 0)])
    club = outside | np.isin(lab, touch[touch > 0])
    club = cv2.dilate(club.astype(np.uint8), np.ones((3, 3), np.uint8)).astype(bool) & (a > 10)
    body_l = lay.copy()
    inside = club & body                                   # shaft drawn across the body: fill with body
    body_l[club & ~body] = 0
    if inside.any():
        rgb = np.clip(body_l[..., :3], 0, 255).astype(np.uint8)
        rgb = cv2.inpaint(rgb, inside.astype(np.uint8), 4, cv2.INPAINT_TELEA)
        body_l[..., :3] = rgb.astype(np.float32)
        body_l[inside, 3] = 255.0
    club_l = np.zeros_like(lay)
    club_l[club] = lay[club]
    return body_l, club_l


def frame_joints(P):
    """named 2D joints of one mannequin pose (33x2+, px)."""
    J = {i: P[i, :2].astype(np.float64) for i in range(33)}
    J["ms"] = (J[11] + J[12]) / 2
    J["mh"] = (J[23] + J[24]) / 2
    up = J["ms"] - J["mh"]
    up /= max(np.linalg.norm(up), 1e-6)
    J["nb"] = J["ms"] + up * 0.12 * np.linalg.norm(J["ms"] - J["mh"])
    hc = (J[0] + J[7] + J[8]) / 3
    d = hc - J["nb"]
    hd = d / np.linalg.norm(d) if np.linalg.norm(d) > 0.5 * RADIUS["head"] else up
    J["ht"] = J["nb"] + hd * 2.2 * RADIUS["head"]
    return J


def target_joints(J, shape):
    """shoulders pulled toward the neck (arms follow unchanged); head grows about the neck base."""
    T = {k: v.copy() for k, v in J.items()}
    ksh = shape.get("shoulder", 1.0)
    for side, chain in ((11, (13, 15, 17, 19, 21)), (12, (14, 16, 18, 20, 22))):
        new = J["ms"] + (J[side] - J["ms"]) * ksh
        dlt = new - J[side]
        T[side] = new
        for j in chain:
            T[j] = J[j] + dlt
    T["ht"] = T["nb"] + (J["ht"] - J["nb"]) * shape.get("head", 1.0)
    return T


def bone_list(shape):
    """(joint a, joint b, trim factor, blend radius)"""
    kt, kl = shape.get("torso_girth", 1.0), shape.get("limb_girth", 1.0)
    B = [("mh", "ms", kt, RADIUS["torso"]),
         ("ms", 11, kl, RADIUS["clav"]), ("ms", 12, kl, RADIUS["clav"]),
         ("mh", 23, kt, RADIUS["hip"]), ("mh", 24, kt, RADIUS["hip"])]
    for (a, b), r in zip(LIMBS, ["upper", "lower", "upper", "lower", "thigh", "shin", "thigh", "shin"]):
        B.append((a, b, kl, RADIUS[r]))
    for a, b in FEET + HANDS:
        B.append((a, b, 1.0, RADIUS["foot"]))
    return B


def _seg(X, A, B):
    d = B - A
    L2 = max(d @ d, 1e-6)
    t = np.clip(((X - A) @ d) / L2, 0, 1)[..., None]
    return t, A + t * d


def warp_body(body, J, T, shape, sharp=6.0):
    H, W = body.shape[:2]
    ys, xs = np.where(body[..., 3] > 0)
    if len(xs) == 0:
        return body
    pad = 40
    x0, x1 = max(xs.min() - pad, 0), min(xs.max() + pad, W - 1)
    y0, y1 = max(ys.min() - pad, 0), min(ys.max() + pad, H - 1)
    gx, gy = np.meshgrid(np.arange(x0, x1 + 1, dtype=np.float64), np.arange(y0, y1 + 1, dtype=np.float64))
    X = np.stack([gx, gy], -1)
    bones = bone_list(shape)
    srcs, ds = [], []
    for a, b, _, r in bones:
        t, Q2 = _seg(X, T[a], T[b])
        srcs.append(J[a] + t * (J[b] - J[a]) + (X - Q2))       # same direction: move with the bone
        ds.append(np.linalg.norm(X - Q2, axis=-1) / r)
    hs = shape.get("head", 1.0)
    _, Q2 = _seg(X, T["nb"], T["ht"])
    ds.append(np.linalg.norm(X - Q2, axis=-1) / (RADIUS["head"] * hs))
    srcs.append(J["nb"] + (X - T["nb"]) / hs)
    D = np.stack(ds)
    w = np.exp(-sharp * (D - D.min(0)))
    w /= w.sum(0)
    S = np.einsum("bhw,bhwc->hwc", w, np.stack(srcs)).astype(np.float32)
    pm = body.copy()
    pm[..., :3] *= pm[..., 3:4] / 255.0                    # premultiplied, so edges stay clean
    out = np.zeros_like(body)
    out[y0:y1 + 1, x0:x1 + 1] = cv2.remap(pm, S[..., 0], S[..., 1], cv2.INTER_LINEAR,
                                          borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    # girth: trim the outline by (1 - k) x the half-thickness found on the nearest bone's axis
    solid = (out[..., 3] > 128).astype(np.uint8)
    dist = cv2.distanceTransform(solid, cv2.DIST_L2, 5)
    near = np.argmin(D, axis=0)
    trim = np.zeros(near.shape, np.float32)
    for i, (a, b, k, _) in enumerate(bones):
        if k >= 0.999:
            continue
        sel = near == i
        if not sel.any():
            continue
        _, Q2 = _seg(X[sel], T[a], T[b])
        qx = np.clip(np.round(Q2[:, 0]).astype(int), 0, W - 1)
        qy = np.clip(np.round(Q2[:, 1]).astype(int), 0, H - 1)
        trim[sel] = (1 - k) * dist[qy, qx]
    keep = np.clip(dist[y0:y1 + 1, x0:x1 + 1] - trim + 0.5, 0, 1)
    out[y0:y1 + 1, x0:x1 + 1, 3] *= keep
    al = out[..., 3:4] / 255.0
    out[..., :3] = np.where(al > 1e-3, out[..., :3] / np.maximum(al, 1e-3), 0)
    return out


def reshape(lay, P, shape, m_head=1.0):
    """lay: RGBA float32 mannequin layer; P: its pose (33x2+); shape: body-shape factors;
    m_head: how much the user's camera enlarges a head at this depth relative to the body."""
    body, club = split_club(lay)
    J = frame_joints(P)
    shape = dict(shape, head=shape.get("head", 1.0) * m_head)
    T = target_joints(J, shape)
    out = warp_body(body, J, T, shape)
    dh = ((T[15] - J[15]) + (T[16] - J[16])) / 2           # the club rides with the hands
    M = np.float32([[1, 0, dh[0]], [0, 1, dh[1]]])
    cl = cv2.warpAffine(club, M, (lay.shape[1], lay.shape[0]), flags=cv2.INTER_LINEAR, borderValue=0)
    ca = cl[..., 3:4] / 255.0
    oa = out[..., 3:4] / 255.0
    rgb = cl[..., :3] * ca + out[..., :3] * oa * (1 - ca)
    al = ca + oa * (1 - ca)
    out[..., :3] = np.where(al > 1e-3, rgb / np.maximum(al, 1e-3), 0)
    out[..., 3:4] = al * 255.0
    return out
