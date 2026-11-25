"""Composite the reference render (separated DTL / FRONT mannequins) onto the user's two videos.

Outputs (2 videos + sheet):
  model_front_nextbay.mp4, model_dtl_nextbay.mp4   - model swings beside the user (same scale)
  model_keyframes.jpg                              - next-bay key positions, both views
Overlay videos (model on top of the user) are NOT produced: user decision 2026-11-25.
The overlay mode survives only in the `check` images, to verify the address scale fit.
Timing: piecewise-linear warp of the model's P1..P10 onto the user's P1..P10.
Scale: the user's BODY SIZE (body_profile.json: thigh + shin + torso, pooled over sessions) projected
with this session's camera, divided by the mannequin's body length (ref/model.json). Posture does not
enter, so the model is never inflated because the user stands more upright. Feet share the user's
floor line. Without those two files: least-squares fit of address landmarks (old behaviour).
Build: body-shape.json (head / shoulder / girth factors) reshapes every mannequin pose with
ref_reshape.py, and the head also gets this session's camera perspective (build_persp).
"""
import os
import sys
import json
import subprocess
import numpy as np
import cv2
from PIL import Image, ImageDraw, ImageFont
from scipy.ndimage import median_filter
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from geom import load_cams
from body_profile import address_joints
from ref_reshape import reshape as reshape_layer

S, OUT = sys.argv[1], sys.argv[2]
R = f"{S}/ref"
UKF = json.load(open(f"{S}/keyframes.json"))
U_P = np.array([UKF[f"P{i}"] for i in range(1, 11)], float)
# model P1..P10 (frame numbers inside ref/fr), optional override in ref/keyframes.json
_mk = json.load(open(f"{R}/keyframes.json")) if os.path.exists(f"{R}/keyframes.json") else {}
M_P = np.array(_mk.get("P", [46, 66, 81, 117, 133, 142, 146, 158, 165, 192]), float)
M_ADDR, M_END = int(_mk.get("address", 46)), int(_mk.get("end", 204))
uniq = [u for u in json.load(open(f"{R}/uniq.json")) if u >= 42]

FONT_B = "/usr/share/fonts/truetype/nanum/NanumSquareB.ttf"
FONT_R = "/usr/share/fonts/truetype/nanum/NanumSquareR.ttf"
fB = {s: ImageFont.truetype(FONT_B, s) for s in (14, 15, 17)}
fR = {s: ImageFont.truetype(FONT_R, s) for s in (12, 13)}
EDGE = np.array([255, 230, 60], np.float32)      # BGR cyan outline


def load_frames(v):
    c = cv2.VideoCapture(f"{S}/vid/{v}.mp4")
    out = []
    while True:
        ok, f = c.read()
        if not ok:
            break
        out.append(f)
    return out


FR = {"front": load_frames("front"), "dtl": load_frames("back")}
N = len(FR["front"])
H, W = FR["front"][0].shape[:2]
UP = {"front": np.load(f"{S}/pose/front.npz")["img"], "dtl": np.load(f"{S}/pose/back.npz")["img"]}
MP = {v: {int(k): np.array(x) for k, x in json.load(open(f"{R}/pose_{v}.json")).items()} for v in ("front", "dtl")}
LAY = {v: {k: cv2.imread(f"{R}/{v}/{k:04d}.png", cv2.IMREAD_UNCHANGED).astype(np.float32) for k in uniq}
       for v in ("front", "dtl")}


# ------------------------------------------------------------------ alignment
def keypts(P, view):
    mid = lambda a, b: (P[a, :2] + P[b, :2]) / 2
    pts = [mid(7, 8), mid(11, 12), mid(23, 24), mid(25, 26), mid(27, 28)]
    if view == "front":   # left/right are reliable face-on
        pts += [P[11, :2], P[12, :2], P[23, :2], P[24, :2], P[27, :2], P[28, :2]]
    return np.array(pts)


BODY = None
if os.path.exists(f"{S}/body_profile.json") and os.path.exists(f"{R}/model.json"):
    _cf, _cb, _ = load_cams(S)
    _J = address_joints(S)
    BODY = {"seg_m": json.load(open(f"{S}/body_profile.json"))["median_m"],
            "model_px": json.load(open(f"{R}/model.json"))["body_len_px"],
            "cam": {"front": _cf, "dtl": _cb},
            "joint": {j: np.median(x, axis=0) for j, x in _J.items()}}


def body_scale(view):
    """model px -> user px that gives the mannequin the user's body length (None if unavailable).

    Each segment of the profile (thighs, shins, torso) is stood upright at the place it occupies in
    this session's address pose and projected with the real camera, so perspective and lens
    distortion at the feet, hips and shoulders are included (a single f/depth was off by up to 6 %)."""
    if BODY is None:
        return None
    cam, Jm, L = BODY["cam"][view], BODY["joint"], BODY["seg_m"]
    pt = lambda js: np.mean([Jm[j] for j in js], axis=0)
    px = 0.0
    for a, b, seg, w in (((23,), (25,), "thigh", .5), ((24,), (26,), "thigh", .5),
                         ((25,), (27,), "shin", .5), ((26,), (28,), "shin", .5), ((11, 12), (23, 24), "torso", 1.)):
        M, h = (pt(a) + pt(b)) / 2, np.array([0, L[seg] / 2, 0])
        px += w * np.linalg.norm(cam.project(M + h)[0][0] - cam.project(M - h)[0][0])
    return px / BODY["model_px"][view]


def foot_y(P):
    return np.mean(P[[27, 28, 29, 30], 1])


def similarity(view):
    p1 = UKF["P1"]
    u = median_filter(UP[view][max(p1 - 40, 0):p1 + 1], size=(5, 1, 1)).mean(0)
    m = np.mean([MP[view][k] for k in (42, 46)], axis=0)
    U_, M_ = keypts(u, view), keypts(m, view)
    mu, mm = U_.mean(0), M_.mean(0)
    # one uniform scale: keeps the model's angles (spine tilt, shaft plane) undistorted
    g_fit = np.sum((U_ - mu) * (M_ - mm)) / np.sum((M_ - mm) ** 2)
    g_body = body_scale(view)
    if g_body is None:
        g, t = g_fit, mu - g_fit * mm
        how = "address-posture fit (no body profile)"
    else:
        g = g_body
        t = np.array([mu[0] - g * mm[0], foot_y(u) - g * foot_y(m)])   # same floor line as the user
        how = f"body size (posture fit would be {g_fit:.3f}, {100 * (g_fit / g - 1):+.0f}%)"
    sc = np.array([g, g])
    err = np.linalg.norm(sc * M_ + t - U_, axis=1)
    print(f"{view}: scale {g:.3f} by {how}; address residual mean {err.mean():.1f}px max {err.max():.1f}px")
    return sc, t


SIM = {v: similarity(v) for v in ("front", "dtl")}


# ------------------------------------------------------------------ the user's build + camera perspective
SHAPE = json.load(open(f"{S}/body-shape.json")) if os.path.exists(f"{S}/body-shape.json") else {}
SHAPE = {k: float(SHAPE[k]) for k in ("head", "shoulder", "torso_girth", "limb_girth") if k in SHAPE}


def nearest_pose(view, k):
    if k in MP[view]:
        return MP[view][k]
    ks = np.array(sorted(MP[view]))
    return MP[view][int(ks[np.argmin(np.abs(ks - k))])]


GROUPS = {"head": [0, 7, 8], "sho": [11, 12], "hip": [23, 24], "knee": [25, 26], "ank": [27, 28]}


def build_persp():
    """{view: {pose: head magnification}} -- how much nearer the lens the mannequin's head is than its
    body, seen from the user's camera, with the mannequin's two-view 3D placed at the user's spot."""
    if BODY is None:
        return None
    geo = json.load(open(f"{R}/model.json"))
    kf = BODY["seg_m"]["body_len"] / geo["body_len_px"]["front"]       # metres per front px
    s = geo["front_per_dtl_px"]
    cf, cb = BODY["cam"]["front"], BODY["cam"]["dtl"]
    hipU = (BODY["joint"][23] + BODY["joint"][24]) / 2
    sx = np.sign(cf.project(hipU + [0.1, 0, 0])[0][0, 0] - cf.project(hipU)[0][0, 0])
    sz = np.sign(cb.project(hipU + [0, 0, 0.1])[0][0, 0] - cb.project(hipU)[0][0, 0])
    f0, d0 = nearest_pose("front", M_ADDR), nearest_pose("dtl", M_ADDR)
    hf0, hd0 = (f0[23, :2] + f0[24, :2]) / 2, (d0[23, :2] + d0[24, :2]) / 2

    def G(k):
        f, d = nearest_pose("front", k), nearest_pose("dtl", k)
        return hipU + np.stack([sx * (f[:, 0] - hf0[0]) * kf, -(f[:, 1] - hf0[1]) * kf,
                                sz * (d[:, 0] - hd0[0]) * s * kf], 1)

    out = {}
    for v, cam in (("front", cf), ("dtl", cb)):
        per = {}
        for k in uniq:
            z = cam.to_cam(G(k))[:, 2]
            body = np.mean([np.mean(z[GROUPS[g_]]) for g_ in ("sho", "hip", "knee", "ank")])
            per[k] = float(np.clip(body / np.mean(z[GROUPS["head"]]), 0.85, 1.5))
        a = per[min(per, key=lambda q: abs(q - M_ADDR))]
        print(f"{v}: camera perspective on the head at address x{a:.2f} "
              f"(range {min(per.values()):.2f}-{max(per.values()):.2f})")
        out[v] = per
    return out


PERSP = build_persp()
if SHAPE or PERSP:
    print("model build:", SHAPE or "mannequin's own", "| perspective:", "on" if PERSP else "off")
    for v in ("front", "dtl"):
        for k in uniq:
            LAY[v][k] = reshape_layer(LAY[v][k], nearest_pose(v, k), SHAPE, PERSP[v][k] if PERSP else 1.0)


def model_body_x(view):
    """horizontal extent of the model *body* (no club) at address, in user pixels."""
    m = np.mean([MP[view][k] for k in (42, 46)], axis=0)
    sc, t = SIM[view]
    xs = m[[0, 7, 8, 11, 12, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32], 0] * sc[0] + t[0]
    return xs.min(), xs.max()


# next-bay horizontal offsets (pixels in the user frame): same depth, same scale, beside the user.
# The model goes to whichever side of the user has more free floor, centred in that gap.
def nextbay_dx(view):
    x0, x1 = model_body_x(view)
    p1 = UKF["P1"]
    u = UP[view][max(p1 - 40, 0):p1 + 1]
    pts = [11, 12, 23, 24, 27, 28, 31, 32] if view == "front" else [0, 7, 8, 15, 16, 23, 24, 29, 30]
    uL, uR = np.nanmin(u[:, pts, 0]), np.nanmax(u[:, pts, 0])
    margin = 30
    left_gap, right_gap = uL - margin, (W - uR) - margin
    half = (x1 - x0) / 2
    if right_gap >= left_gap:
        target_c = uR + margin + max(right_gap, 2 * half) / 2
    else:
        target_c = uL - margin - max(left_gap, 2 * half) / 2
    return target_c - (x0 + x1) / 2


DX = {v: nextbay_dx(v) for v in ("front", "dtl")}
print("next-bay dx", {k: round(float(v), 1) for k, v in DX.items()})


# ------------------------------------------------------------------ timing
def model_time(i):
    if i <= U_P[0]:
        return M_ADDR
    if i >= U_P[-1]:
        # continue into the model's finish hold proportionally, then stay
        return min(M_P[-1] + (i - U_P[-1]) * (M_P[-1] - M_P[-2]) / (U_P[-1] - U_P[-2]), M_END)
    return float(np.interp(i, U_P, M_P))


def model_layer(view, tau):
    us = uniq
    j = int(np.clip(np.searchsorted(us, tau, side="right") - 1, 0, len(us) - 2))
    a, b = us[j], us[j + 1]
    w = float(np.clip((tau - a) / (b - a), 0, 1))
    # hold each rendered pose, cross-fade only in the last 30% (avoids a doubled club)
    w = float(np.clip((w - 0.7) / 0.3, 0, 1))
    A, B = LAY[view][a], LAY[view][b]
    pa, pb = A[..., 3:4] / 255.0, B[..., 3:4] / 255.0
    pm = (1 - w) * A[..., :3] * pa + w * B[..., :3] * pb
    al = (1 - w) * pa + w * pb
    rgb = np.where(al > 1e-3, pm / np.maximum(al, 1e-3), 0)
    return np.dstack([rgb, al[..., 0] * 255.0])


def warp(layer, view, dx=0.0):
    sc, t = SIM[view]
    Mx = np.array([[sc[0], 0, t[0] + dx], [0, sc[1], t[1]]], np.float32)
    out = cv2.warpAffine(layer, Mx, (W, H), flags=cv2.INTER_CUBIC, borderValue=(0, 0, 0, 0))
    out[..., 3] = np.clip(out[..., 3], 0, 255)
    return out


def composite(frame, lay, body_op, edge_op):
    a = (lay[..., 3:4] / 255.0)
    rgb = np.clip(lay[..., :3], 0, 255)
    out = frame.astype(np.float32) * (1 - body_op * a) + rgb * (body_op * a)
    mb = (a[..., 0] > 0.45).astype(np.uint8)
    edge = cv2.morphologyEx(mb, cv2.MORPH_GRADIENT, np.ones((3, 3), np.uint8)).astype(np.float32)
    edge = cv2.GaussianBlur(edge, (3, 3), 0)[..., None]
    out = out * (1 - edge_op * edge) + EDGE * (edge_op * edge)
    return np.clip(out, 0, 255).astype(np.uint8)


def phase_name(i):
    k = UKF
    if i < k["P1"]:
        return "어드레스"
    if i < k["P2"]:
        return "테이크어웨이"
    if i < k["P4"] - 8:
        return "백스윙"
    if i <= k["P4"] + 8:
        return "탑"
    if i < k["P7"] - 2:
        return "다운스윙"
    if i <= k["P7"] + 3:
        return "임팩트"
    if i < k["P10"]:
        return "팔로스루"
    return "피니시"


def annotate(img, i, title, slow=False):
    pil = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    d = ImageDraw.Draw(pil, "RGBA")
    w = pil.width
    d.rectangle([0, 0, w, 30], fill=(0, 0, 0, 150))
    d.text((8, 6), title, font=fB[15], fill=(255, 255, 255, 255))
    ph = phase_name(i)
    tw = d.textlength(ph, font=fB[17])
    d.text((w - tw - 10, 5), ph, font=fB[17], fill=(255, 235, 80, 255))
    d.rectangle([6, 36, 196, 58], fill=(0, 0, 0, 130))
    d.rectangle([12, 42, 26, 52], fill=(235, 235, 235, 230), outline=(60, 230, 255, 255))
    d.text((32, 40), "모델프로 9번 아이언 · @modelgolf", font=fR[12], fill=(255, 255, 255, 255))
    if slow:
        d.rectangle([w - 92, 36, w - 6, 58], fill=(200, 40, 40, 200))
        d.text((w - 84, 39), "슬로우 x0.4", font=fB[14], fill=(255, 255, 255, 255))
    return cv2.cvtColor(np.array(pil), cv2.COLOR_RGB2BGR)


TITLES = {("front", "nextbay"): "정면 · 옆 타석 모델프로", ("dtl", "nextbay"): "후면 · 옆 타석 모델프로",
          ("front", "overlay"): "정면 · 나 + 모델프로 겹치기", ("dtl", "overlay"): "후면 · 나 + 모델프로 겹치기"}


def render(view, mode, i):
    lay = warp(model_layer(view, model_time(i)), view, DX[view] if mode == "nextbay" else 0.0)
    if mode == "nextbay":
        img = composite(FR[view][i], lay, 0.92, 0.55)
    else:
        img = composite(FR[view][i], lay, 0.30, 0.60)
    return img


def writer(path, w, h, fps=70):
    return subprocess.Popen(["ffmpeg", "-v", "error", "-y", "-f", "rawvideo", "-pix_fmt", "bgr24", "-s", f"{w}x{h}",
                             "-r", str(fps), "-i", "-", "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "20",
                             "-preset", "medium", "-movflags", "+faststart", path], stdin=subprocess.PIPE)


def sequence():
    """(frame index, slow flag) list: full pass, short hold, then 0.4x replay of P3..P9."""
    seq = [(i, False) for i in range(140, N)] + [(N - 1, False)] * 35
    for i in range(UKF["P3"], UKF["P9"] + 1):
        seq += [(i, True)] * (2 if i < UKF["P4"] - 20 else 3)
    return seq


if __name__ == "__main__":
    os.makedirs(OUT, exist_ok=True)
    if len(sys.argv) > 3 and sys.argv[3] == "check":
        for i in (190, UKF["P4"], UKF["P7"], UKF["P10"]):
            row = [annotate(render(v, m, i), i, TITLES[(v, m)]) for v in ("front", "dtl") for m in ("nextbay", "overlay")]
            cv2.imwrite(f"{OUT}/check_{i}.jpg", np.hstack(row))
        sys.exit(0)
    seq = sequence()
    cache = {}
    for v in ("front", "dtl"):
        for i in sorted(set(k for k, _ in seq)):
            cache[(v, i)] = render(v, "nextbay", i)
    for v in ("front", "dtl"):
        wtr = writer(f"{OUT}/model_{v}_nextbay.mp4", W, H)
        for i, slow in seq:
            wtr.stdin.write(annotate(cache[(v, i)], i, TITLES[(v, "nextbay")], slow).tobytes())
        wtr.stdin.close()
        wtr.wait()
    # key-position sheet (next-bay, both views)
    rows = [np.hstack([annotate(cache[("front", UKF[k])], UKF[k], k),
                       annotate(cache[("dtl", UKF[k])], UKF[k], k)]) for k in ("P1", "P2", "P4", "P6", "P7", "P10")]
    cv2.imwrite(f"{OUT}/model_keyframes.jpg", np.vstack([np.hstack(rows[j:j + 2]) for j in range(0, 6, 2)]),
                [cv2.IMWRITE_JPEG_QUALITY, 88])
    print("done")
