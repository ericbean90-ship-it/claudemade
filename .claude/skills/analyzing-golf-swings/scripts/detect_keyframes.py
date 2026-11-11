"""Auto-detect P1..P10 frame numbers from the pose tracks, and write a sheet for visual checking.

P1 takeaway start, P4 top and P7 impact are found from the hand tracks; P2,P3,P5,P6,P8,P9,P10
come from typical timing ratios between those three.  ALWAYS look at out/keyframes_sheet.jpg
and edit keyframes.json by hand if a frame is off (P4 and P7 matter most).

Usage: python detect_keyframes.py <workdir>
"""
import sys
import json
import numpy as np
import cv2
from scipy.ndimage import median_filter, gaussian_filter1d

S = sys.argv[1]
F = median_filter(np.load(f"{S}/pose/front.npz")["img"], size=(9, 1, 1), mode="nearest")
B = median_filter(np.load(f"{S}/pose/back.npz")["img"], size=(9, 1, 1), mode="nearest")
N = len(F)

wf = gaussian_filter1d(F[:, [15, 16], :2].mean(1), 1.0, axis=0)      # front hands (x, y)
wb_y = gaussian_filter1d(B[:, [15, 16], 1].mean(1), 1.0)             # DTL hand height (y down)
shw = np.nanmedian(np.linalg.norm(F[:40, 11, :2] - F[:40, 12, :2], axis=1))
x0, y0 = np.nanmedian(wf[:40], axis=0)

# P7 impact: between the last "deep backswing" frame (hands far to the trail side) and the first
# "deep follow-through" frame (hands far to the target side), where the hands cross address x.
side = wf[:, 0] - x0
deep = 0.8 * shw
bs_side = np.sign(np.nanmedian(side[np.abs(side) > deep][:20])) if np.any(np.abs(side) > deep) else -1.0
follow = np.where(side * bs_side < -deep)[0]
if len(follow) == 0:
    sys.exit("임팩트를 찾지 못함: 정면 영상에서 손이 양쪽으로 크게 움직이지 않습니다 (스윙 영상이 맞는지 확인)")
f0 = int(follow[0])
back = np.where(side[:f0] * bs_side > deep)[0]
b0 = int(back[-1]) if len(back) else max(f0 - 40, 1)
p7 = b0
for t in range(b0 + 1, f0 + 1):
    if side[t] * bs_side <= 0:
        p7 = t
        break

# P4 top: hands highest (DTL view) before impact; take the LAST frame near the minimum
lo = max(p7 - 260, 0)
win = wb_y[lo:p7 - 10]
ymin = np.nanmin(win)
tol = 0.03 * shw
cand = [lo + i for i, v in enumerate(win) if v <= ymin + tol]
p4 = int(cand[-1])

# P1 takeaway start: last still frame before the hands leave the address position
d = np.linalg.norm(wf - np.array([x0, y0]), axis=1)
p1 = 0
for t in range(p4, 0, -1):
    if d[t] < 0.045 * shw:
        p1 = t
        break

bs, ds = p4 - p1, p7 - p4
kf = {
    "P1": p1, "P2": round(p1 + 0.35 * bs), "P3": round(p1 + 0.57 * bs), "P4": p4,
    "P5": round(p4 + 0.55 * ds), "P6": round(p4 + 0.79 * ds), "P7": p7,
    "P8": round(p7 + 0.29 * ds), "P9": round(p7 + 0.53 * ds), "P10": min(round(p7 + 1.6 * ds), N - 1),
}
kf = {k: int(v) for k, v in kf.items()}
json.dump(kf, open(f"{S}/keyframes.json", "w"))
print("keyframes", kf, f"(tempo backswing:downswing = {bs / ds:.1f}:1)")


def frames(v):
    c = cv2.VideoCapture(f"{S}/vid/{v}.mp4")
    out = []
    while True:
        ok, f = c.read()
        if not ok:
            break
        out.append(f)
    return out


FV, BV = frames("front"), frames("back")
tiles = []
for k, i in kf.items():
    t = np.hstack([FV[i], BV[i]])
    t = cv2.resize(t, None, fx=0.5, fy=0.5)
    cv2.rectangle(t, (0, 0), (t.shape[1], 22), (0, 0, 0), -1)
    cv2.putText(t, f"{k}  f{i}", (6, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 1, cv2.LINE_AA)
    tiles.append(t)
rows = [np.hstack(tiles[r:r + 5]) for r in (0, 5)]
cv2.imwrite(f"{S}/out/keyframes_sheet.jpg", np.vstack(rows), [cv2.IMWRITE_JPEG_QUALITY, 85])
print("sheet:", f"{S}/out/keyframes_sheet.jpg")
