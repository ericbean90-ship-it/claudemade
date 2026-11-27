"""Cut the lesson pro out of both videos (누끼) and rebuild the club.

Usage: python pro_cutout.py <pro workdir> <ball_front_u> <ball_front_v> <ball_dtl_u> <ball_dtl_v>
Needs <workdir>/vid/{front,back}.mp4, pose/{front,back}.npz and keyframes.json
(run_pipeline.sh ... analyze on the pro's videos). Writes <workdir>/cut/{front,back}_alpha.npy (uint8)
and <workdir>/cut/club.json.

  * body   Robust Video Matting (rvm_mobilenetv3_fp32.onnx, recurrent -> no flicker, ~0.05 s/frame)
  * club   a person matting model drops the thin, motion-blurred shaft. The cameras are fixed, so a
           background plate (median of the frames where the pro does not cover a pixel, after the
           club starts moving) shows where the club is. Per frame the shaft angle is fitted from
           those pixels as a ray from the hands; gaps are filled by interpolating the angle, and the
           club is drawn back as a clean shaft + head (render_club) instead of noisy blurred pixels.
"""
import os
import sys
import json
import numpy as np
import cv2

MODEL = os.path.expanduser("~/.cache/golf-swing/rvm_mobilenetv3_fp32.onnx")
URL = "https://github.com/PeterL1n/RobustVideoMatting/releases/download/v1.0.0/rvm_mobilenetv3_fp32.onnx"


def read(path):
    cap, F = cv2.VideoCapture(path), []
    while True:
        ok, fr = cap.read()
        if not ok:
            return np.array(F)
        F.append(fr)


def rvm(F, ratio=0.5):
    import onnxruntime as ort
    so = ort.SessionOptions()
    so.intra_op_num_threads = os.cpu_count() or 4
    s = ort.InferenceSession(MODEL, so, providers=["CPUExecutionProvider"])
    rec = [np.zeros((1, 1, 1, 1), np.float32)] * 4
    out = []
    for _ in range(2):                         # first pass only warms the recurrent state
        out = []
        for fr in (F if _ else F[:30]):
            x = (cv2.cvtColor(fr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255).transpose(2, 0, 1)[None]
            _f, pha, *rec = s.run(None, {"src": x, "r1i": rec[0], "r2i": rec[1], "r3i": rec[2], "r4i": rec[3],
                                         "downsample_ratio": np.array([ratio], np.float32)})
            out.append((pha[0, 0] * 255).astype(np.uint8))
    return np.array(out)


def _masked_median(F, A, idx):
    free = np.stack([cv2.dilate((A[i] > 25).astype(np.uint8), np.ones((9, 9), np.uint8)) == 0 for i in idx])
    blk = F[idx].astype(np.float32)
    blk[~free] = np.nan
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return np.nanmedian(blk, axis=0)


def plate(F, A, start):
    """global background: median of the frames (after the club starts moving) where the pro is not."""
    idx = np.arange(start, len(F), 2)
    P = np.zeros(F.shape[1:], np.uint8)
    for y0 in range(0, F.shape[1], 64):
        med = _masked_median(F[:, y0:y0 + 64], A[:, y0:y0 + 64], idx)
        P[y0:y0 + 64] = np.where(np.isnan(med), np.median(F[::4, y0:y0 + 64], axis=0), med).astype(np.uint8)
    return P


def local_plates(F, A, PL, step=8, half=48):
    """background every `step` frames from +-`half` frames around it: the screen behind the DTL view
    changes over time (shot replays) and would otherwise look like a club."""
    out = {}
    for c in range(0, len(F), step):
        idx = np.arange(max(0, c - half), min(len(F), c + half + 1), 4)
        med = _masked_median(F, A, idx)
        out[c] = np.where(np.isnan(med), PL, med).astype(np.uint8)
    return out


def club_track(F, A, L, PL, ball, K, LP=None):
    """per frame: (angle, length) of the club as a ray from the hands, NaN where not seen."""
    n, H, W = A.shape
    yy, xx = np.mgrid[0:H, 0:W]
    a0 = L[K["P1"]]
    reach = 0.75 * np.linalg.norm((a0[27, :2] + a0[28, :2]) / 2 - a0[0, :2])
    hand = (L[:, 15, :2] + L[:, 16, :2]) / 2
    Lc = np.linalg.norm(np.asarray(ball, float) - hand[K["P1"]])          # hands -> ball at address
    ang, length, sup = np.full(n, np.nan), np.full(n, np.nan), np.zeros(n)
    for i in range(n):
        bgp = LP[min(LP, key=lambda c: abs(c - i))] if LP else PL
        d = cv2.medianBlur(np.abs(F[i].astype(np.int16) - bgp.astype(np.int16)).max(2).astype(np.uint8), 3)
        h = hand[i]
        r2 = (xx - h[0]) ** 2 + (yy - h[1]) ** 2
        m = (d > 28) & (r2 < reach ** 2) & (r2 > 6 ** 2)
        m &= ((xx - ball[0]) ** 2 + (yy - ball[1]) ** 2) > 14 ** 2
        body = cv2.dilate((A[i] > 90).astype(np.uint8), np.ones((5, 5), np.uint8)) > 0
        m &= ~body
        ys, xs = np.where(m)
        if len(xs) < 6:
            continue
        v = np.stack([xs - h[0], ys - h[1]], 1).astype(np.float64)
        nb = int(reach / 3) + 2
        steps = (np.arange(nb) + 0.5) * 3
        # score = length of the unbroken run of pixels along a ray from the hands. Stretches hidden by
        # the body (shaft in front of the legs) neither break nor score; scattered compression noise
        # in dark areas never forms a run.
        best = (0, None, 0.0)
        for t in np.linspace(-np.pi, np.pi, 360, endpoint=False):
            u = np.array([np.cos(t), np.sin(t)])
            along = v @ u
            ok = (along > 0) & (np.abs(v @ np.array([-u[1], u[0]])) < 2.5)
            if ok.sum() < 4:
                continue
            bins = np.zeros(nb, bool)
            bins[np.minimum((along[ok] / 3).astype(int), nb - 1)] = True
            px = np.clip(np.round(h[0] + steps * u[0]).astype(int), 0, W - 1)
            py = np.clip(np.round(h[1] + steps * u[1]).astype(int), 0, H - 1)
            hidden = body[py, px]
            filled = np.where(bins)[0]
            if (~hidden[:filled[0]]).sum() > 10:      # open floor between hands and first pixel
                continue
            end, gap, cnt = filled[0], 0, 0
            for b in range(filled[0], nb):
                if bins[b]:
                    end, gap, cnt = b, 0, cnt + 1
                elif hidden[b]:
                    continue
                else:
                    gap += 1
                    if gap > 2:
                        break
            score = cnt + 0.25 * len(filled)
            if score > best[0]:
                best = (score, t, (end + 1) * 3.0)
        if best[0] < 6:
            continue
        ang[i], length[i], sup[i] = best[1], np.clip(best[2], 0.15 * Lc, 1.1 * Lc), best[0]
    # drop jumps: a seen angle more than 30 deg off the median of its +-4 seen neighbours
    seen = np.where(~np.isnan(ang))[0]
    bad = []
    for j, i in enumerate(seen):
        nb = [ang[seen[q]] for q in range(max(0, j - 4), min(len(seen), j + 5)) if q != j]
        if len(nb) >= 3:
            ref = np.angle(np.median(np.cos(nb)) + 1j * np.median(np.sin(nb)))
            if abs(np.angle(np.exp(1j * (ang[i] - ref)))) > np.radians(30):
                bad.append(i)
    ang[bad], length[bad] = np.nan, np.nan
    return {"angle": ang.tolist(), "support": sup.tolist()}


def club_3d(S, raw, ball, K, gapmax=20, gapseen=120, pair=2):
    """Both views' 2D shaft angles -> 3D shaft direction (two planes through the cameras intersect),
    a fixed 3D club length from the address (hands -> ball), then projected back into each view.
    Foreshortening (a short club at the top seen from behind) comes out right by itself, and a frame
    seen in one view only gets the neighbours' direction pushed onto that view's plane."""
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from geom import load_cams, triangulate
    cf, cb, _ = load_cams(S)
    cams = {"front": cf, "back": cb}
    Lm = {v: np.load(f"{S}/pose/{v}.npz")["img"] for v in cams}
    n = len(Lm["front"])
    hand2 = {v: (Lm[v][:, 15, :2] + Lm[v][:, 16, :2]) / 2 for v in cams}
    H3 = np.array([triangulate(cf, hand2["front"][i], cb, hand2["back"][i])[0] for i in range(n)])
    B3 = triangulate(cf, np.asarray(ball["front"], float), cb, np.asarray(ball["back"], float))[0]

    def plane_normal(v, i, t):
        c = cams[v]
        h = hand2[v][i]
        _, d1 = c.ray(h)
        _, d2 = c.ray(h + 40 * np.array([np.cos(t), np.sin(t)]))
        nn = np.cross(d1, d2)
        return nn / np.linalg.norm(nn)

    def towards(v, i, D, t):
        """flip D so it points the way the 2D angle t points in view v"""
        a, _ = cams[v].project(H3[i])
        b, _ = cams[v].project(H3[i] + 0.1 * D)
        return D if (b[0] - a[0]) @ np.array([np.cos(t), np.sin(t)]) >= 0 else -D

    ang = {v: np.array(raw[v]["angle"], float) for v in cams}
    D = np.full((n, 3), np.nan)
    for i in range(n):
        if not np.isfinite(ang["front"][i]):
            continue
        js = [j for j in sorted(range(max(0, i - pair), min(n, i + pair + 1)), key=lambda j: abs(j - i))
              if np.isfinite(ang["back"][j])]
        if not js:
            continue
        d = np.cross(plane_normal("front", i, ang["front"][i]), plane_normal("back", js[0], ang["back"][js[0]]))
        if np.linalg.norm(d) > 1e-6:
            D[i] = towards("front", i, d / np.linalg.norm(d), ang["front"][i])
    for i in list(range(0, K["P1"] + 1)) + [K["P7"]]:        # the club head sits at the ball
        d = B3 - H3[i]
        D[i] = d / np.linalg.norm(d)
    L3 = float(np.linalg.norm(B3 - H3[K["P1"]]))
    # fill short gaps by interpolating the direction, then honour a single view if it saw the club
    ok = np.isfinite(D[:, 0])
    idx = np.arange(n)
    known = idx[ok]
    for i in idx[~ok]:
        lo, hi = known[known < i], known[known > i]
        seen1 = np.isfinite(ang["front"][i]) or np.isfinite(ang["back"][i])
        if not len(lo) or not len(hi) or hi[0] - lo[-1] > (gapseen if seen1 else gapmax):
            continue
        w = (i - lo[-1]) / (hi[0] - lo[-1])
        a0, a1 = D[lo[-1]], D[hi[0]]
        om = np.arccos(np.clip(a0 @ a1, -1, 1))
        d = a0 if om < 1e-4 else (np.sin((1 - w) * om) * a0 + np.sin(w * om) * a1) / np.sin(om)
        for v in cams:
            if np.isfinite(ang[v][i]):
                nn = plane_normal(v, i, ang[v][i])
                d = d - (d @ nn) * nn
        if np.linalg.norm(d) > 1e-6:
            D[i] = d / np.linalg.norm(d)
    # drop single-frame flips: >30 deg off the arc through the frames 3 before and 3 after
    for i in range(3, n - 3):
        a0, a1, d = D[i - 3], D[i + 3], D[i]
        if np.isfinite(a0).all() and np.isfinite(a1).all() and np.isfinite(d).all():
            mid = a0 + a1
            if np.linalg.norm(mid) > 1e-6 and np.degrees(np.arccos(np.clip(d @ (mid / np.linalg.norm(mid)), -1, 1))) > 30 \
                    and np.degrees(np.arccos(np.clip(a0 @ a1, -1, 1))) < 40:
                D[i] = mid / np.linalg.norm(mid)
    # gentle smoothing (3-frame) of the direction
    Ds = D.copy()
    for i in range(1, n - 1):
        blk = D[i - 1:i + 2]
        if np.isfinite(blk).all():
            m = blk.mean(0)
            Ds[i] = m / np.linalg.norm(m)
    torso = np.array([triangulate(cf, Lm["front"][i, [11, 12, 23, 24], :2].mean(0), cb,
                                  Lm["back"][i, [11, 12, 23, 24], :2].mean(0))[0] for i in range(n)])
    head = np.array([triangulate(cf, Lm["front"][i, [0, 7, 8], :2].mean(0), cb,
                                 Lm["back"][i, [0, 7, 8], :2].mean(0))[0] for i in range(n)])
    out = {"club_m": L3, "dir3d": np.where(np.isfinite(Ds), Ds, np.nan).tolist()}
    for v, c in cams.items():
        a2, l2, bh = np.full(n, np.nan), np.full(n, np.nan), np.zeros(n, bool)
        for i in range(n):
            if np.isfinite(Ds[i, 0]):
                p0, _ = c.project(H3[i])
                p1, z1 = c.project(H3[i] + L3 * Ds[i])
                dv = p1[0] - p0[0]
                a2[i], l2[i] = np.arctan2(dv[1], dv[0]), np.linalg.norm(dv)
                zm = c.to_cam((H3[i] + 0.5 * L3 * Ds[i])[None])[0, 2]
                near = min(c.to_cam(torso[i][None])[0, 2], c.to_cam(head[i][None])[0, 2])
                bh[i] = zm > near + 0.03          # shaft middle behind the body part nearest the lens
        out[v] = {"angle": a2.tolist(), "length": l2.tolist(), "behind": bh.tolist(),
                  "seen2d": np.isfinite(ang[v]).tolist()}
    return out


def solid_alpha(a):
    """The pro is a real person: draw him fully opaque. Dark trousers on a dark floor and motion blur
    leave see-through patches (alpha 0.5-0.9) inside the body; fill the inside of the silhouette and
    keep only a 1-2 px soft rim."""
    a = a.astype(np.float32) / 255
    m = (a > 0.35).astype(np.uint8)
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)))
    n, lab, st, _ = cv2.connectedComponentsWithStats(1 - m)        # fill enclosed holes
    for i in range(1, n):
        x, y, w, h, area = st[i]
        if x > 0 and y > 0 and x + w < m.shape[1] and y + h < m.shape[0] and area < 400:
            m[lab == i] = 1
    core = cv2.erode(m, np.ones((3, 3), np.uint8))
    rim = np.clip((a - 0.15) / 0.45, 0, 1) * m
    out = np.where(core > 0, 1.0, np.maximum(rim, cv2.GaussianBlur(m.astype(np.float32), (3, 3), 0) * (a > 0.1)))
    return (np.clip(out, 0, 1) * 255).astype(np.uint8)


def render_club(img_rgba, hand, ang, length, scale=1.0, behind=False):
    """draw a clean shaft + head on an RGBA float layer (in place). hand/length in layer px."""
    if ang is None or not np.isfinite(ang) or not np.isfinite(length):
        return img_rgba
    u = np.array([np.cos(ang), np.sin(ang)])
    a = np.asarray(hand, float) - u * 8 * scale
    b = np.asarray(hand, float) + u * length
    H, W = img_rgba.shape[:2]
    sh = np.zeros((H, W), np.uint8)
    cv2.line(sh, (int(round(a[0] * 4)), int(round(a[1] * 4))), (int(round(b[0] * 4)), int(round(b[1] * 4))),
             255, max(1, int(round(2 * scale))), cv2.LINE_AA, shift=2)
    hd = np.zeros((H, W), np.uint8)
    c = b + u * 3 * scale
    cv2.ellipse(hd, (int(c[0]), int(c[1])), (max(2, int(5 * scale)), max(1, int(3 * scale))),
                float(np.degrees(ang)) + 90, 0, 360, 255, -1, cv2.LINE_AA)
    body_a = img_rgba[..., 3:4].copy() / 255
    for m, col in ((sh, (205, 205, 205)), (hd, (70, 70, 75))):
        al = (m.astype(np.float32) / 255)[..., None]
        if behind:                                # the body covers the club where they overlap
            al = al * (1 - body_a)
        oa = img_rgba[..., 3:4] / 255
        rgb = np.array(col, np.float32) * al + img_rgba[..., :3] * oa * (1 - al)
        na = al + oa * (1 - al)
        img_rgba[..., :3] = np.where(na > 1e-3, rgb / np.maximum(na, 1e-3), 0)
        img_rgba[..., 3:4] = na * 255
    return img_rgba


if __name__ == "__main__":
    S = sys.argv[1]
    BALL = {"front": (float(sys.argv[2]), float(sys.argv[3])), "back": (float(sys.argv[4]), float(sys.argv[5]))}
    if not os.path.exists(MODEL):
        import urllib.request
        os.makedirs(os.path.dirname(MODEL), exist_ok=True)
        urllib.request.urlretrieve(URL, MODEL)
    K = json.load(open(f"{S}/keyframes.json"))
    os.makedirs(f"{S}/cut", exist_ok=True)
    raw = {}
    for v in ("front", "back"):
        F = read(f"{S}/vid/{v}.mp4")
        A = rvm(F)
        PL = plate(F, A, K["P2"])
        L = np.load(f"{S}/pose/{v}.npz")["img"]
        raw[v] = club_track(F, A, L, PL, BALL[v], K, local_plates(F, A, PL))
        np.save(f"{S}/cut/{v}_alpha.npy", A)
        cv2.imwrite(f"{S}/cut/{v}_plate.png", PL)
    json.dump(raw, open(f"{S}/cut/club_raw.json", "w"))
    club = club_3d(S, raw, BALL, K)
    sw = slice(K["P1"], K["P10"] + 1)
    for v in ("front", "back"):
        seen = np.array(club[v]["seen2d"])[sw].mean()
        drawn = np.isfinite(np.array(club[v]["angle"], float))[sw].mean()
        print(f"{v}: club seen in {seen * 100:.0f}% of P1-P10 frames, drawn in {drawn * 100:.0f}%")
    print(f"club length {club['club_m']:.2f} (calibrated units)")
    club["ball"] = BALL
    json.dump(club, open(f"{S}/cut/club.json", "w"))
