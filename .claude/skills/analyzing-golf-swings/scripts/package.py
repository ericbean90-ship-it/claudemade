"""Copy this session's deliverables into <workdir>/upload/ with Korean, date-prefixed names, and print
what create_file needs for each (title, mime, and the sha256 file reference for binary uploads).

Usage: python package.py <workdir>
Prefix = stamp.json 'prefix' (the date burned into the video, e.g. 2026-11-11_1916).
"""
import os
import sys
import glob
import json
import shutil
import hashlib

S = sys.argv[1]
pre = json.load(open(f"{S}/stamp.json"))["prefix"]
U = f"{S}/upload"
os.makedirs(U, exist_ok=True)
MAP = [  # source (in out/), Korean name suffix, mime
    (f"{pre}_리포트.md", "리포트.md", "text/markdown"),
    ("before_after.jpg", "전후비교.jpg", "image/jpeg"),
    ("ghost_side_by_side.mp4", "프로고스트_합본.mp4", "video/mp4"),
    ("model_both_overlay.mp4", "모델프로_합본.mp4", "video/mp4"),
    ("model_front_nextbay.mp4", "모델프로_정면_옆타석.mp4", "video/mp4"),
    ("model_dtl_nextbay.mp4", "모델프로_후면_옆타석.mp4", "video/mp4"),
    ("model_front_overlay.mp4", "모델프로_정면_겹치기.mp4", "video/mp4"),
    ("model_dtl_overlay.mp4", "모델프로_후면_겹치기.mp4", "video/mp4"),
]
rows = []
for src, suf, mime in MAP:
    p = f"{S}/out/{src}"
    if not os.path.exists(p):
        continue
    dst = f"{U}/{pre}_{suf}"
    shutil.copyfile(p, dst)
    h = hashlib.sha256(open(dst, "rb").read()).hexdigest()
    rows.append({"title": os.path.basename(dst), "mime": mime, "bytes": os.path.getsize(dst),
                 "file_ref": f"sha256:{h}:{os.path.abspath(dst)}"})
json.dump(rows, open(f"{U}/manifest.json", "w"), ensure_ascii=False, indent=1)
for r in rows:
    print(f"{r['title']}\t{r['bytes']}\t{r['mime']}\n  {r['file_ref']}")
