"""On-screen date/time of the shot (what the user sees in the video) -> <workdir>/stamp.json

Usage: python stamp.py <workdir> <front_video_path> <back_video_path>   (run BEFORE the analyze stage)
rangex burns 'YYYY-MM-DD HH:MM' (local time) into the bottom-left of each clip; the file name holds
the same moment in UTC.  We propose filename + STAMP_OFFSET_H (default 9, KST) and save a zoomed crop
of the burned-in stamp to out/stamp.png: LOOK at it and fix stamp.json by hand if it differs.
"""
import os
import re
import sys
import json
import datetime
import numpy as np
import cv2

S, FRONT, BACK = sys.argv[1], sys.argv[2], sys.argv[3]
off = float(os.environ.get("STAMP_OFFSET_H", "9"))
m = re.search(r"(20\d{2})(\d{2})(\d{2})_(\d{2})(\d{2})(\d{2})", os.path.basename(FRONT))
if not m:
    sys.exit("파일명에서 촬영 시각을 찾지 못함: out/stamp.png를 보고 stamp.json을 직접 작성하세요")
t = datetime.datetime(*map(int, m.groups())) + datetime.timedelta(hours=off)
rec = {"screen_time": t.strftime("%Y-%m-%d %H:%M"), "prefix": t.strftime("%Y-%m-%d_%H%M"), "offset_h": off}
json.dump(rec, open(f"{S}/stamp.json", "w"), ensure_ascii=False)
rows = []
for v in (FRONT, BACK):
    c = cv2.VideoCapture(v)
    ok, f = c.read()
    if ok:
        H = f.shape[0]
        rows.append(cv2.resize(f[H - 40:H, 0:240], None, fx=3, fy=3, interpolation=cv2.INTER_CUBIC))
os.makedirs(f"{S}/out", exist_ok=True)
cv2.imwrite(f"{S}/out/stamp.png", np.vstack(rows))
print(rec, "| 확인용 크롭:", f"{S}/out/stamp.png")
