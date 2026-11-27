"""Put the user's lesson pro, cut out of his own videos (pro_cutout.py), in the next bay of the user's videos.

Usage: python pro_overlay.py <user workdir> <pro dir> <out dir> [check]
  pro dir: vid/{front,back}.mp4, cut/{front,back}_alpha.npy, cut/{front,back}_plate.png, cut/club.json,
           pose/{front,back}.npz, keyframes.json   (a pro workdir, or pro_asset.py unpack)
Outputs: pro_front_nextbay.mp4, pro_dtl_nextbay.mp4, pro_keyframes.jpg   (check: check_pro_<frame>.jpg)

Placement: the pro was filmed in the user's own bay, so the backgrounds are matched directly (SIFT on
the empty background, similarity fit) instead of trusting two separate calibrations: same camera ->
scale 1, the pro stands exactly on the user's spot; a moved camera -> the fitted scale/shift. He is
then slid sideways into the next bay (the side with more free floor). Real person: drawn opaque,
no reshaping. Timing: the pro's P1..P10 warped onto the user's P1..P10, nearest real frame.
"""
import os
import sys
import json
import subprocess
import numpy as np
import cv2
from PIL import Image, ImageDraw, ImageFont

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from pro_cutout import read, render_club

S, PRO, OUT = sys.argv[1], sys.argv[2], sys.argv[3]
CHECK = len(sys.argv) > 4 and sys.argv[4] == "check"
VIEWS = {"front": ("front", "front"), "dtl": ("back", "back")}   # out name -> (user video, pro file key)
UKF = json.load(open(f"{S}/keyframes.json"))
PKF = json.load(open(f"{PRO}/keyframes.json"))
U_P = np.array([UKF[f"P{i}"] for i in range(1, 11)], float)
M_P = np.array([PKF[f"P{i}"] for i in range(1, 11)], float)
CLUB = json.load(open(f"{PRO}/cut/club.json"))

FONT_B = "/usr/share/fonts/truetype/nanum/NanumSquareB.ttf"
FONT_R = "/usr/share/fonts/truetype/nanum/NanumSquareR.ttf"
fB = {s: ImageFont.truetype(FONT_B, s) for s in (14, 15, 17)}
fR = {s: ImageFont.truetype(FONT_R, s) for s in (12, 13)}

FR = {v: read(f"{S}/vid/{u}.mp4") for v, (u, _) in VIEWS.items()}
N = min(len(f) for f in FR.values())
H, W = FR["front"][0].shape[:2]
UP = {v: np.load(f"{S}/pose/{u}.npz")["img"] for v, (u, _) in VIEWS.items()}
PF = {v: read(f"{PRO}/vid/{p}.mp4") for v, (_, p) in VIEWS.items()}
PA = {v: np.load(f"{PRO}/cut/{p}_alpha.npy") for v, (_, p) in VIEWS.items()}
PL = {v: np.load(f"{PRO}/pose/{p}.npz")["img"] for v, (_, p) in VIEWS.items()}
PN = min(len(PF["front"]), len(PF["dtl"]))


# ------------------------------------------------------------------ placement (background registration)
def user_plate(v):
    F = FR[v][::6]
    return np.median(F, axis=0).astype(np.uint8)


def bg_mask(shape, people):
    """features only on the empty set: no people, no burned-in logo / time stamp."""
    m = np.full(shape[:2], 255, np.uint8)
    for P in people:
        x0, y0 = np.nanmin(P[..., 0]) - 40, np.nanmin(P[..., 1]) - 60
        x1, y1 = np.nanmax(P[..., 0]) + 40, np.nanmax(P[..., 1]) + 30
        m[max(int(y0), 0):int(y1), max(int(x0), 0):int(x1)] = 0
    m[:110, shape[1] // 2:] = 0        # rangex logo
    m[-45:, :] = 0                     # date / time stamp
    return m


def register(v):
    """similarity pro image -> user image, from the backgrounds."""
    up = user_plate(v)
    pp = cv2.imread(f"{PRO}/cut/{VIEWS[v][1]}_plate.png")
    sift = cv2.SIFT_create(4000)
    ku, du = sift.detectAndCompute(cv2.cvtColor(up, cv2.COLOR_BGR2GRAY), bg_mask(up.shape, [UP[v]]))
    kp, dp = sift.detectAndCompute(cv2.cvtColor(pp, cv2.COLOR_BGR2GRAY), bg_mask(pp.shape, [PL[v]]))
    if du is None or dp is None:
        return None, 0
    good = []
    for pair in cv2.BFMatcher().knnMatch(dp, du, k=2):
        if len(pair) == 2 and pair[0].distance < 0.75 * pair[1].distance:
            good.append(pair[0])
    if len(good) < 12:
        return None, len(good)
    a = np.float32([kp[g.queryIdx].pt for g in good])
    b = np.float32([ku[g.trainIdx].pt for g in good])
    M, inl = cv2.estimateAffinePartial2D(a, b, method=cv2.RANSAC, ransacReprojThreshold=2.5)
    return M, int(inl.sum()) if inl is not None else 0


def foot_y(P):
    return np.mean(P[[27, 28, 29, 30], 1])


def placement(v):
    M, inl = register(v)
    p1 = UKF["P1"]
    u = np.median(UP[v][max(p1 - 40, 0):p1 + 1], axis=0)
    m = PL[v][PKF["P1"]]
    sane = M is not None and 0.75 < np.hypot(M[0, 0], M[1, 0]) < 1.33 and abs(np.degrees(np.arctan2(M[1, 0], M[0, 0]))) < 4
    if not sane or inl < 15:
        # different bay: scale by body length, feet on the user's floor line
        seg = lambda P: (np.linalg.norm(P[23, :2] - P[25, :2]) + np.linalg.norm(P[25, :2] - P[27, :2])
                         + np.linalg.norm(P[24, :2] - P[26, :2]) + np.linalg.norm(P[26, :2] - P[28, :2])) / 2
        g = seg(u) / seg(m)
        M = np.float32([[g, 0, 0], [0, g, 0]])
        M[0, 2] = np.mean(u[[23, 24], 0]) - g * np.mean(m[[23, 24], 0])
        M[1, 2] = foot_y(u) - g * foot_y(m)
        print(f"{v}: background match failed ({inl} inliers) -> scaled by leg length x{g:.3f}. "
              f"Different bay? The pro's real size relative to the user is approximate.")
    else:
        g = float(np.hypot(M[0, 0], M[1, 0]))
        rot = float(np.degrees(np.arctan2(M[1, 0], M[0, 0])))
        print(f"{v}: background match {inl} inliers -> scale {g:.3f}, rotation {rot:+.1f} deg, "
              f"shift ({M[0, 2]:+.0f}, {M[1, 2]:+.0f}) px")
    # next bay: slide sideways to the side with more free floor
    mp = (M[:, :2] @ m[:, :2].T).T + M[:, 2]
    pts = [0, 7, 8, 11, 12, 13, 14, 15, 16, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32]
    x0, x1 = mp[pts, 0].min(), mp[pts, 0].max()
    uu = UP[v][max(p1 - 40, 0):p1 + 1]
    upts = [11, 12, 23, 24, 27, 28, 31, 32] if v == "front" else [0, 7, 8, 15, 16, 23, 24, 29, 30]
    uL, uR = np.nanmin(uu[:, upts, 0]), np.nanmax(uu[:, upts, 0])
    margin = 30
    left_gap, right_gap = uL - margin, (W - uR) - margin
    half = (x1 - x0) / 2
    if right_gap >= left_gap:
        target_c = uR + margin + max(right_gap, 2 * half) / 2
    else:
        target_c = uL - margin - max(left_gap, 2 * half) / 2
    dx = target_c - (x0 + x1) / 2
    return M, float(dx)


PLACE = {v: placement(v) for v in VIEWS}
print("next-bay dx", {v: round(PLACE[v][1], 1) for v in PLACE})


# ------------------------------------------------------------------ timing
def pro_frame(i):
    if i <= U_P[0]:
        t = M_P[0] - (U_P[0] - i)
    elif i >= U_P[-1]:
        t = M_P[-1] + (i - U_P[-1])
    else:
        t = float(np.interp(i, U_P, M_P))
    return int(np.clip(round(t), 0, PN - 1))


def pro_layer(v, k, shift=True):
    lay = np.dstack([PF[v][k].astype(np.float32), PA[v][k].astype(np.float32)])
    c = CLUB[VIEWS[v][1]]
    hand = (PL[v][k, 15, :2] + PL[v][k, 16, :2]) / 2
    render_club(lay, hand, c["angle"][k], c["length"][k], behind=bool(c["behind"][k]))
    M, dx = PLACE[v]
    M2 = M.copy()
    if shift:
        M2[0, 2] += dx
    out = cv2.warpAffine(lay, M2, (W, H), flags=cv2.INTER_LINEAR, borderValue=(0, 0, 0, 0))
    out[..., 3] = np.clip(out[..., 3], 0, 255)
    return out


def composite(frame, lay, op=1.0):
    a = lay[..., 3:4] / 255.0 * op
    return np.clip(frame.astype(np.float32) * (1 - a) + lay[..., :3] * a, 0, 255).astype(np.uint8)


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
    d.rectangle([6, 36, 176, 58], fill=(0, 0, 0, 130))
    d.text((14, 40), "레슨 프로 · 같은 타석 촬영", font=fR[12], fill=(255, 255, 255, 255))
    if slow:
        d.rectangle([w - 92, 36, w - 6, 58], fill=(200, 40, 40, 200))
        d.text((w - 84, 39), "슬로우 x0.4", font=fB[14], fill=(255, 255, 255, 255))
    return cv2.cvtColor(np.array(pil), cv2.COLOR_RGB2BGR)


TITLES = {"front": "정면 · 옆 타석 레슨 프로", "dtl": "후면 · 옆 타석 레슨 프로"}


def render(v, i):
    return composite(FR[v][i], pro_layer(v, pro_frame(i)))


def writer(path, w, h, fps=70):
    return subprocess.Popen(["ffmpeg", "-v", "error", "-y", "-f", "rawvideo", "-pix_fmt", "bgr24", "-s", f"{w}x{h}",
                             "-r", str(fps), "-i", "-", "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "20",
                             "-preset", "medium", "-movflags", "+faststart", path], stdin=subprocess.PIPE)


def sequence():
    """(frame, slow) list: full pass from 2 s before address, short hold, then 0.4x replay of P3..P9."""
    seq = [(i, False) for i in range(max(0, UKF["P1"] - 140), N)] + [(N - 1, False)] * 35
    for i in range(UKF["P3"], UKF["P9"] + 1):
        seq += [(i, True)] * (2 if i < UKF["P4"] - 20 else 3)
    return seq


if __name__ == "__main__":
    os.makedirs(OUT, exist_ok=True)
    if CHECK:
        for k in ("P1", "P4", "P7", "P10"):
            i = UKF[k]
            row = []
            for v in VIEWS:
                row.append(annotate(render(v, i), i, TITLES[v]))
                on_spot = composite(FR[v][i], pro_layer(v, pro_frame(i), shift=False), 0.5)
                row.append(annotate(on_spot, i, "정합 확인: 같은 자리 50%"))
            cv2.imwrite(f"{OUT}/check_pro_{i}.jpg", np.hstack(row))
        sys.exit(0)
    seq = sequence()
    for v in VIEWS:
        cache = {i: render(v, i) for i in sorted(set(k for k, _ in seq))}
        wtr = writer(f"{OUT}/pro_{v}_nextbay.mp4", W, H)
        for i, slow in seq:
            wtr.stdin.write(annotate(cache[i], i, TITLES[v], slow).tobytes())
        wtr.stdin.close()
        wtr.wait()
        if v == "front":
            sheet_f = {k: annotate(cache[UKF[k]], UKF[k], k) for k in ("P1", "P2", "P4", "P6", "P7", "P10")}
        else:
            sheet_d = {k: annotate(cache[UKF[k]], UKF[k], k) for k in ("P1", "P2", "P4", "P6", "P7", "P10")}
        del cache
    rows = [np.hstack([sheet_f[k], sheet_d[k]]) for k in ("P1", "P2", "P4", "P6", "P7", "P10")]
    cv2.imwrite(f"{OUT}/pro_keyframes.jpg", np.vstack([np.hstack(rows[j:j + 2]) for j in range(0, 6, 2)]),
                [cv2.IMWRITE_JPEG_QUALITY, 88])
    print("done")
