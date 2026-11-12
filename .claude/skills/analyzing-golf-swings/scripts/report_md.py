"""Write the Korean session report skeleton -> <workdir>/out/report-<YYYY-MM-DD_HHMMSS>.md

Usage: python report_md.py <workdir>
Fills the title and the comparison table (from out/compare.md); the narrative sections are left as
<!-- 작성: ... --> placeholders that Claude MUST replace before uploading (check: grep -c '<!-- 작성').
"""
import re
import sys
import glob
import json

S = sys.argv[1]
cur = json.load(open(f"{S}/out/metrics.json"))
hist = []
for p in glob.glob(f"{S}/history/*"):
    try:
        r = json.load(open(p))
    except Exception:  # noqa: BLE001
        continue
    if r.get("video_md5") != cur.get("video_md5"):
        hist.append(r)
hist.sort(key=lambda r: r.get("shot_time") or "")

shot = cur.get("shot_time") or cur.get("analyzed")
stamp = re.sub(r"[: ]", lambda m: "_" if m.group(0) == " " else "", shot or "unknown")
md_day = lambda s: f"{int(s[5:7])}/{int(s[8:10])}" if s and len(s) >= 10 else "?"

if hist:
    title_cmp = f"## 지난 회차 대비 변화 ({md_day(hist[-1].get('shot_time'))} → {md_day(shot)})"
else:
    title_cmp = "## 지난 회차 대비 변화 (첫 기록: 기준선)"
compare = open(f"{S}/out/compare.md").read().strip()

md = f"""# 골프 스윙 리포트 — {shot} 촬영

## 스윙 분석 요약
<!-- 작성: 첫 줄은 큰 흐름 한 문장 요약(무엇이 좋아졌고 다음 과제는 무엇인지). 이어서 이번 근본 과제 하나를 비유와 함께 2~4줄. -->

{title_cmp}
{compare}

<!-- 작성: 표 아래에 좋아진 것 / 나빠진 것 / 오차 범위(±2cm, ±2°)라 변화 없음 을 2~4줄로 해석. 카메라 변경 경고가 있으면 그 영향도 한 줄. -->

## 핵심 느낌
<!-- 작성: 이번 과제를 위한 몸의 느낌 3가지를 번호 목록으로. -->

## 드릴
<!-- 작성: 드릴 3~4개(방법 한 줄씩) + 마지막에 '다음 회차 확인 기준'(숫자 목표 2~4개). -->
"""
out = f"{S}/out/report-{stamp}.md"
open(out, "w").write(md)
print(out)
