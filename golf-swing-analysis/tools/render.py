"""Render the semi-transparent 'pro ghost' (user's own body + setup, pro motion) over both views."""
import os
import sys
import json
import subprocess
import numpy as np
import cv2
from PIL import Image, ImageDraw, ImageFont

S = sys.argv[1]
OUT = sys.argv[2]
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from geom import load_cams
from pro_model import ProGolfer

cf, cb, XG = load_cams(S)
ball = np.load(f"{S}/ball.npy")
KF = json.load(open(f"{S}/keyframes.json"))
pro = ProGolfer(XG, ball, KF)
U = np.load(f"{S}/user3d.npz")
SC = pro.scale  # model metres -> real metres divisor

FONT_B = "/usr/share/fonts/truetype/nanum/NanumSquareB.ttf"
FONT_R = "/usr/share/fonts/truetype/nanum/NanumSquareR.ttf"
fB = {s: ImageFont.truetype(FONT_B, s) for s in (13, 15, 17, 20)}
fR = {s: ImageFont.truetype(FONT_R, s) for s in (12, 13, 14)}

GHOST = (255, 214, 40)      # BGR  (azure)
EDGE = (255, 255, 200)
CLUBC = (60, 255, 255)      # BGR yellow
REF = (200, 200, 200)


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
CAM = {"front": cf, "dtl": cb}

# ---------------------------------------------------------------- user metrics (real cm)
a = KF["P1"]
def ucm(key, i, ax):
    return (U[key][i][ax] - U[key][a][ax]) * 100 / SC

J0 = pro.pose(KF["P1"])
def pcm(key, i, ax):
    return (pro.pose(i)[key][ax] - J0[key][ax]) * 100 / SC

def spine_fwd(sho, pel):
    v = sho - pel
    return np.degrees(np.arctan2(v[2], v[1]))

M = {
    "u_head_top": -ucm("head", KF["P4"], 0), "p_head_top": -pcm("head", KF["P4"], 0),
    "u_pel_top": -ucm("pel", KF["P4"], 0), "p_pel_top": -pcm("pel", KF["P4"], 0),
    "u_pel_imp": ucm("pel", KF["P7"], 0), "p_pel_imp": pcm("pel", KF["P7"], 0),
    "u_spine_loss": spine_fwd(U["sho"][a], U["pel"][a]) - spine_fwd(U["sho"][KF["P7"]], U["pel"][KF["P7"]]),
    "u_hands_out": (U["hands"][KF["P7"]][2] - U["hands"][a][2]) * 100 / SC,
    "u_pel_ball": ucm("pel", KF["P7"], 2),
}
Jp7 = pro.pose(KF["P7"])
M["p_spine_loss"] = spine_fwd((J0[11] + J0[12]) / 2, J0["pel"]) - spine_fwd((Jp7[11] + Jp7[12]) / 2, Jp7["pel"])
M["p_hands_out"] = (Jp7["grip"][2] - J0["grip"][2]) * 100 / SC
json.dump({k: round(float(v), 1) for k, v in M.items()}, open(f"{OUT}/metrics.json", "w"), indent=1, ensure_ascii=False)
print({k: round(float(v), 1) for k, v in M.items()})

# ---------------------------------------------------------------- drawing helpers
SS = 2  # supersampling


def capsule(mask, p1, p2, r1, r2):
    p1 = np.asarray(p1, float) * SS
    p2 = np.asarray(p2, float) * SS
    r1, r2 = max(r1 * SS, 1), max(r2 * SS, 1)
    cv2.circle(mask, tuple(np.round(p1).astype(int)), int(round(r1)), 255, -1, cv2.LINE_AA)
    cv2.circle(mask, tuple(np.round(p2).astype(int)), int(round(r2)), 255, -1, cv2.LINE_AA)
    d = p2 - p1
    L = np.linalg.norm(d)
    if L < 1e-3:
        return
    n = np.array([-d[1], d[0]]) / L
    poly = np.array([p1 + n * r1, p2 + n * r2, p2 - n * r2, p1 - n * r1])
    cv2.fillConvexPoly(mask, np.round(poly).astype(np.int32), 255, cv2.LINE_AA)


R_SEG = [  # (a, b, radius_a, radius_b) metres
    (11, 13, 0.050, 0.044), (13, 15, 0.042, 0.032), (12, 14, 0.050, 0.044), (14, 16, 0.042, 0.032),
    (23, 25, 0.075, 0.055), (25, 27, 0.052, 0.040), (24, 26, 0.075, 0.055), (26, 28, 0.052, 0.040),
    (29, 31, 0.035, 0.035), (27, 29, 0.035, 0.035), (28, 30, 0.035, 0.035), (30, 32, 0.035, 0.035),
    ("neck", "head", 0.050, 0.050),
]


def ghost_layer(view, J):
    cam = CAM[view]
    H, W = cam.H, cam.W
    keys = list(J.keys())
    pts, z = cam.project(np.array([J[k] for k in keys]))
    P = {k: pts[i] for i, k in enumerate(keys)}
    Z = {k: z[i] for i, k in enumerate(keys)}
    rad = lambda k, r: cam.f * r / Z[k]
    m = np.zeros((H * SS, W * SS), np.uint8)
    # torso: shoulders + hips polygon, rounded with capsules
    torso = np.array([P[11], P[12], P[24], P[23]]) * SS
    cv2.fillConvexPoly(m, np.round(cv2.convexHull(torso.astype(np.float32))).astype(np.int32).reshape(-1, 2), 255, cv2.LINE_AA)
    capsule(m, P[11], P[23], rad(11, 0.065), rad(23, 0.085))
    capsule(m, P[12], P[24], rad(12, 0.065), rad(24, 0.085))
    capsule(m, P[11], P[12], rad(11, 0.065), rad(12, 0.065))
    capsule(m, P[23], P[24], rad(23, 0.09), rad(24, 0.09))
    # trunk depth (chest / belly / glutes) so the side view is not paper-thin
    capsule(m, P["neck"], P["pel"], rad("neck", 0.10), rad("pel", 0.115))
    for a_, b_, ra, rb in R_SEG:
        capsule(m, P[a_], P[b_], rad(a_, ra), rad(b_, rb))
    hr = rad("head", 0.105)
    cv2.ellipse(m, tuple(np.round(P["head"] * SS).astype(int)), (int(hr * SS * 0.92), int(hr * SS * 1.05)), 0, 0, 360, 255, -1, cv2.LINE_AA)
    cv2.circle(m, tuple(np.round(P["grip"] * SS).astype(int)), int(rad("grip", 0.045) * SS), 255, -1, cv2.LINE_AA)
    m = cv2.resize(m, (W, H), interpolation=cv2.INTER_AREA).astype(np.float32) / 255.0
    # club
    c = np.zeros((H * SS, W * SS), np.uint8)
    cv2.line(c, tuple(np.round(P["butt"] * SS).astype(int)), tuple(np.round(P["clubhead"] * SS).astype(int)), 255,
             5, cv2.LINE_AA)
    chr_ = min(rad("clubhead", 0.025), 7)
    cv2.circle(c, tuple(np.round(P["clubhead"] * SS).astype(int)), max(4, int(chr_ * SS)), 255, -1, cv2.LINE_AA)
    c = cv2.resize(c, (W, H), interpolation=cv2.INTER_AREA).astype(np.float32) / 255.0
    return m, c, P


def composite(img, m, c, alpha=0.38):
    out = img.astype(np.float32)
    g = np.array(GHOST, np.float32)
    out = out * (1 - alpha * m[..., None]) + g * (alpha * m[..., None])
    # outline
    mb = (m > 0.5).astype(np.uint8)
    edge = cv2.morphologyEx(mb, cv2.MORPH_GRADIENT, np.ones((3, 3), np.uint8)).astype(np.float32)
    edge = cv2.GaussianBlur(edge, (3, 3), 0)
    out = out * (1 - 0.85 * edge[..., None]) + np.array(EDGE, np.float32) * (0.85 * edge[..., None])
    out = out * (1 - 0.9 * c[..., None]) + np.array(CLUBC, np.float32) * (0.9 * c[..., None])
    return np.clip(out, 0, 255).astype(np.uint8)


def dashed(img, p1, p2, color, th=1, dash=8):
    p1, p2 = np.asarray(p1, float), np.asarray(p2, float)
    L = np.linalg.norm(p2 - p1)
    n = int(L // dash)
    for k in range(0, n, 2):
        a_ = p1 + (p2 - p1) * k / n
        b_ = p1 + (p2 - p1) * min(k + 1, n) / n
        cv2.line(img, tuple(np.round(a_).astype(int)), tuple(np.round(b_).astype(int)), color, th, cv2.LINE_AA)


def phase_name(i):
    k = KF
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


def info_lines(view, i):
    k = KF
    L = []
    if view == "front" and k["P2"] <= i <= k["P5"] + 10:
        L = [("백스윙 머리 옆이동", f"나 {M['u_head_top']:.0f}cm", f"프로 {M['p_head_top']:.0f}cm"),
             ("백스윙 골반 밀림", f"나 {M['u_pel_top']:.0f}cm", f"프로 {M['p_pel_top']:.0f}cm")]
    if view == "front" and k["P5"] + 10 < i <= k["P9"]:
        L = [("임팩트 골반 타깃쪽 이동", f"나 {M['u_pel_imp']:.0f}cm", f"프로 {M['p_pel_imp']:.0f}cm")]
    if view == "dtl" and k["P5"] - 15 <= i <= k["P9"]:
        L = [("임팩트 척추각 손실", f"나 {M['u_spine_loss']:.0f}°", f"프로 {M['p_spine_loss']:.0f}°"),
             ("임팩트 손이 몸에서 멀어짐", f"나 {M['u_hands_out']:.0f}cm", f"프로 {M['p_hands_out']:.0f}cm")]
    return L


def annotate(img, view, i, title):
    pil = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    d = ImageDraw.Draw(pil, "RGBA")
    W = pil.width
    d.rectangle([0, 0, W, 30], fill=(0, 0, 0, 150))
    d.text((8, 6), title, font=fB[15], fill=(255, 255, 255, 255))
    ph = phase_name(i)
    tw = d.textlength(ph, font=fB[17])
    d.text((W - tw - 10, 5), ph, font=fB[17], fill=(255, 235, 80, 255))
    # legend
    y0 = 36
    d.rectangle([6, y0, 150, y0 + 22], fill=(0, 0, 0, 120))
    d.rectangle([12, y0 + 6, 26, y0 + 16], fill=(40, 214, 255, 200))
    d.text((32, y0 + 4), "프로(내 체형)", font=fR[13], fill=(255, 255, 255, 255))
    lines = info_lines(view, i)
    if lines:
        h = 10 + 38 * len(lines)
        if view == "front":
            xb, yb, xe = W - 214, 36, W - 6
        else:
            xb, yb, xe = 6, pil.height - 28 - h, W - 6
        d.rectangle([xb, yb, xe, yb + h], fill=(0, 0, 0, 170))
        for n_, (lab, me, pr) in enumerate(lines):
            yy = yb + 6 + 38 * n_
            d.text((xb + 8, yy), lab, font=fR[14], fill=(220, 220, 220, 255))
            d.text((xb + 8, yy + 17), me, font=fB[17], fill=(255, 140, 90, 255))
            d.text((xb + 104, yy + 17), pr, font=fB[17], fill=(40, 214, 255, 255))
    return cv2.cvtColor(np.array(pil), cv2.COLOR_RGB2BGR)


# reference lines from the address pose
addr = pro.pose(0)
ref = {}
pf, _ = cf.project(np.array([addr["head"], addr["pel"]]))
ref["front_head_x"] = pf[0][0]
ref["front_pel_x"] = pf[1][0]
pb, _ = cb.project(np.array([ball, addr["grip"], ball + (addr["grip"] - ball) * 2.2]))
ref["dtl_plane"] = (pb[0], pb[2])


def render_frame(i, view):
    img = FR[view][i].copy()
    J = pro.pose(i)
    m, c, P = ghost_layer(view, J)
    if view == "front":
        x = ref["front_head_x"]
        dashed(img, (x, 40), (x, 300), REF, 1)
    else:
        dashed(img, ref["dtl_plane"][0], ref["dtl_plane"][1], (120, 200, 255), 1)
    img = composite(img, m, c)
    title = "정면 (Face-on)" if view == "front" else "후방 (Down-the-line)"
    return annotate(img, view, i, title)


def writer(path, w, h, fps):
    return subprocess.Popen(["ffmpeg", "-v", "error", "-y", "-f", "rawvideo", "-pix_fmt", "bgr24", "-s", f"{w}x{h}",
                             "-r", str(fps), "-i", "-", "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "20",
                             "-preset", "medium", "-movflags", "+faststart", path], stdin=subprocess.PIPE)


if __name__ == "__main__":
    cache = {}
    for i in range(N):
        cache[i] = (render_frame(i, "front"), render_frame(i, "dtl"))
    W, H = cf.W, cf.H
    for view, idx in (("front", 0), ("dtl", 1)):
        w = writer(f"{OUT}/ghost_{view}.mp4", W, H, 70)
        for i in range(N):
            w.stdin.write(cache[i][idx].tobytes())
        w.stdin.close()
        w.wait()
    # side by side: full pass, then slow replay of P3..P9
    w = writer(f"{OUT}/ghost_side_by_side.mp4", W * 2, H, 70)
    start = 140
    for i in range(start, N):
        w.stdin.write(np.hstack(cache[i]).tobytes())
    for _ in range(35):
        w.stdin.write(np.hstack(cache[N - 1]).tobytes())
    for i in range(KF["P3"], KF["P9"] + 1):
        fr = np.hstack(cache[i])
        pil = Image.fromarray(cv2.cvtColor(fr, cv2.COLOR_BGR2RGB))
        d = ImageDraw.Draw(pil, "RGBA")
        d.rectangle([W - 70, 34, W + 70, 58], fill=(200, 40, 40, 200))
        d.text((W - 54, 37), "슬로우 x0.4", font=fB[15], fill=(255, 255, 255, 255))
        fr = cv2.cvtColor(np.array(pil), cv2.COLOR_RGB2BGR)
        for _ in range(2 if i < KF["P4"] - 20 else 3):
            w.stdin.write(fr.tobytes())
    w.stdin.close()
    w.wait()
    # key-frame comparison sheet
    rows = []
    for name in ("P1", "P2", "P4", "P6", "P7", "P10"):
        i = KF[name]
        rows.append(np.hstack(cache[i]))
    sheet = np.vstack([np.hstack(rows[k:k + 2]) for k in range(0, 6, 2)])
    cv2.imwrite(f"{OUT}/ghost_keyframes.jpg", sheet, [cv2.IMWRITE_JPEG_QUALITY, 88])
    for name in ("P4", "P7"):
        cv2.imwrite(f"{OUT}/ghost_{name}.jpg", np.hstack(cache[KF[name]]), [cv2.IMWRITE_JPEG_QUALITY, 90])
    print("done")
