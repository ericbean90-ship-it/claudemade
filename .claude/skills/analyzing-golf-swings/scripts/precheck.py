"""Before analysing: is this pair of videos already in the history?  (cheap MD5 check)

Usage: python precheck.py <front.mp4> <back.mp4> <workdir>
Reads every file in <workdir>/history/ (any extension) as a JSON record.
Exit 0 = new videos, 3 = identical to a past record (prints which), 2 = front/back timestamps differ.
"""
import os
import re
import sys
import glob
import json
import hashlib


def md5(p):
    h = hashlib.md5()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


front, back, W = sys.argv[1:4]
ts = [re.search(r"(20\d{6}_\d{6})", os.path.basename(p)) for p in (front, back)]
if ts[0] and ts[1] and ts[0][1] != ts[1][1]:
    print(f"경고: 정면({ts[0][1]})과 후면({ts[1][1]}) 촬영 시각이 다릅니다. 같은 샷인지 사용자에게 확인하세요.")
    sys.exit(2)
cur = {"front": md5(front), "dtl": md5(back)}
n = 0
for p in sorted(glob.glob(f"{W}/history/*")):
    try:
        r = json.load(open(p))
    except Exception:  # noqa: BLE001
        print("기록 파일을 JSON으로 읽지 못함:", p)
        continue
    n += 1
    if r.get("video_md5") == cur:
        print(f"같은 영상: {r.get('shot_time') or r.get('analyzed')} 기록과 파일이 완전히 같습니다.")
        print("그때 과제:", r.get("notes", "-"))
        sys.exit(3)
print(f"새 영상 (비교할 과거 기록 {n}개)", cur)
