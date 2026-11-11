"""Composite the reference render (separated DTL / FRONT mannequins) onto the user's two videos.

Outputs (5 videos):
  model_front_nextbay.mp4, model_dtl_nextbay.mp4   - model swings beside the user (same scale)
  model_front_overlay.mp4, model_dtl_overlay.mp4   - model overlaid on the user, high transparency
  model_both_overlay.mp4                           - front + DTL overlays side by side
Timing: piecewise-linear warp of the model's P1..P10 onto the user's P1..P10.
Scale/position: least-squares similarity fit of address landmarks (model -> user).
"""
import sys
import json
import subprocess
import numpy as np
import cv2
from PIL import Image, ImageDraw, ImageFont
from scipy.ndimage import median_filter

S, OUT = sys.argv[1], sys.argv[2]
R = f"{S}/ref"
UKF = json.load(open(f"{S}/keyframes.json"))
U_P = np.array([UKF[f"P{i}"] for i in range(1, 11)], float)
# model P1..P10 (frame numbers inside ref/fr), optional override in ref/keyframes.json
_mk = json.load(open(f"{R}/keyframes.json")) if __import__("os").path.exists(f"{R}/keyframes.json") else {}
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


def similarity(view):
    u = median_filter(UP[view][150:196], size=(5, 1, 1)).mean(0)
    m = np.mean([MP[view][k] for k in (42, 46)], axis=0)
    U_, M_ = keypts(u, view), keypts(m, view)
    mu, mm = U_.mean(0), M_.mean(0)
    # one uniform scale: keeps the model's angles (spine tilt, shaft plane) undistorted
    g = np.sum((U_ - mu) * (M_ - mm)) / np.sum((M_ - mm) ** 2)
    sc = np.array([g, g])
    t = mu - sc * mm
    err = np.linalg.norm(sc * M_ + t - U_, axis=1)
    print(f"{view}: scale x {sc[0]:.3f} y {sc[1]:.3f}, residual mean {err.mean():.1f}px max {err.max():.1f}px")
    return sc, t


SIM = {v: similarity(v) for v in ("front", "dtl")}


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
    u = UP[view][150:196]
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
    import os
    os.makedirs(OUT, exist_ok=True)
    if len(sys.argv) > 3 and sys.argv[3] == "check":
        for i in (190, UKF["P4"], UKF["P7"], UKF["P10"]):
            row = [annotate(render(v, m, i), i, TITLES[(v, m)]) for v in ("front", "dtl") for m in ("nextbay", "overlay")]
            cv2.imwrite(f"{OUT}/check_{i}.jpg", np.hstack(row))
        sys.exit(0)
    seq = sequence()
    cache = {}
    for v in ("front", "dtl"):
        for m in ("nextbay", "overlay"):
            for i in sorted(set(k for k, _ in seq)):
                cache[(v, m, i)] = render(v, m, i)
    files = {}
    for v in ("front", "dtl"):
        for m in ("nextbay", "overlay"):
            p = f"{OUT}/model_{v}_{m}.mp4"
            wtr = writer(p, W, H)
            for i, slow in seq:
                wtr.stdin.write(annotate(cache[(v, m, i)], i, TITLES[(v, m)], slow).tobytes())
            wtr.stdin.close()
            wtr.wait()
            files[(v, m)] = p
    p = f"{OUT}/model_both_overlay.mp4"
    wtr = writer(p, W * 2, H)
    for i, slow in seq:
        fr = np.hstack([annotate(cache[("front", "overlay", i)], i, "정면 · 나 + 모델프로", slow),
                        annotate(cache[("dtl", "overlay", i)], i, "후면 · 나 + 모델프로", slow)])
        wtr.stdin.write(fr.tobytes())
    wtr.stdin.close()
    wtr.wait()
    # key-position sheet (overlay, both views)
    rows = [np.hstack([annotate(cache[("front", "overlay", UKF[k])], UKF[k], k),
                       annotate(cache[("dtl", "overlay", UKF[k])], UKF[k], k)]) for k in ("P1", "P2", "P4", "P6", "P7", "P10")]
    cv2.imwrite(f"{OUT}/model_keyframes.jpg", np.vstack([np.hstack(rows[j:j + 2]) for j in range(0, 6, 2)]),
                [cv2.IMWRITE_JPEG_QUALITY, 88])
    print("done")
