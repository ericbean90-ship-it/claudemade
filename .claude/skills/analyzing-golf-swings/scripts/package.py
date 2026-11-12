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
import subprocess

MAX_BYTES = 2_900_000   # Drive connector carries ~3.1 MB of file per call; keep a margin


def fit_video(src, dst, max_bytes=MAX_BYTES):
    """Re-encode (2-pass x264, same size/fps) so the file fits the connector limit."""
    dur = float(subprocess.check_output(["ffprobe", "-v", "error", "-show_entries", "format=duration",
                                         "-of", "csv=p=0", src]).decode())
    kbps = int(max_bytes * 8 / dur / 1000 * 0.92)
    for _ in range(6):
        log = dst + ".2pass"
        base = ["ffmpeg", "-v", "error", "-y", "-i", src, "-c:v", "libx264", "-preset", "slow",
                "-b:v", f"{kbps}k", "-pix_fmt", "yuv420p", "-an", "-passlogfile", log]
        subprocess.run(base + ["-pass", "1", "-f", "mp4", os.devnull], check=True)
        subprocess.run(base + ["-pass", "2", "-movflags", "+faststart", dst], check=True)
        for f in glob.glob(log + "*"):
            os.remove(f)
        if os.path.getsize(dst) <= max_bytes:
            return kbps
        kbps = int(kbps * 0.88)
    raise SystemExit(f"{dst}: {max_bytes} 바이트 이하로 줄이지 못함")

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
    if os.path.getsize(p) > MAX_BYTES and mime == "video/mp4":
        k = fit_video(p, dst)
        print(f"  (용량 맞춤: {os.path.getsize(p)/1e6:.1f}MB -> {os.path.getsize(dst)/1e6:.2f}MB, {k}kbps)")
    else:
        shutil.copyfile(p, dst)
    if os.path.getsize(dst) > MAX_BYTES:
        print(f"  경고: {os.path.basename(dst)} 가 {MAX_BYTES} 바이트를 넘어 드라이브에 올릴 수 없음")
    h = hashlib.sha256(open(dst, "rb").read()).hexdigest()
    rows.append({"title": os.path.basename(dst), "mime": mime, "bytes": os.path.getsize(dst),
                 "file_ref": f"sha256:{h}:{os.path.abspath(dst)}"})
json.dump(rows, open(f"{U}/manifest.json", "w"), ensure_ascii=False, indent=1)
for r in rows:
    print(f"{r['title']}\t{r['bytes']}\t{r['mime']}\n  {r['file_ref']}")
