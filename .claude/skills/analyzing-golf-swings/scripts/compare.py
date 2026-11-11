"""Compare this session's metrics.json with past records -> markdown table (Korean).

Usage: python compare.py <workdir>
Reads <workdir>/history/*.json (past records, any order) and <workdir>/out/metrics.json.
Writes <workdir>/out/compare.md and prints it. Exit code 3 if the videos are identical to a past record.
"""
import sys
import json
import glob

S = sys.argv[1]
cur = json.load(open(f"{S}/out/metrics.json"))
hist = []
for p in glob.glob(f"{S}/history/*"):
    try:
        hist.append(json.load(open(p)))
    except Exception as e:  # noqa: BLE001
        print("skip", p, e)
hist.sort(key=lambda r: (r.get("shot_time") or "", r.get("analyzed") or ""))

# key, label, better ('low' | 'high' | 'target' | None), noise threshold
ROWS = [
    ("head_sway_top_cm", "백스윙 머리 옆이동 (cm)", "low", 2.0),
    ("pelvis_sway_top_cm", "백스윙 골반 밀림 (cm)", "low", 2.0),
    ("pelvis_shift_impact_cm", "임팩트 골반 타깃쪽 이동 (cm)", "high", 2.0),
    ("spine_loss_impact_deg", "임팩트 척추각 손실 (°)", "low", 2.0),
    ("pelvis_toward_ball_impact_cm", "임팩트 골반 공쪽 이동 (cm)", "low", 2.0),
    ("head_impact_cm", "임팩트 머리 타깃쪽 이동 (cm)", "target", 2.0),
    ("hands_out_downswing_cm", "다운스윙 후반 손이 몸에서 멀어짐 (cm)", "low", 2.0),
    ("hands_outside_p6_vs_p2_cm", "다운스윙 손 경로 바깥 (cm)", "low", 2.0),
    ("pelvis_shift_finish_cm", "피니시 골반 타깃쪽 (cm)", "high", 2.0),
    ("spine_fwd_finish_deg", "피니시 척추 앞숙임 (°)", None, 2.0),
    ("tempo_ratio", "템포 백:다운 (:1)", "target", 0.3),
    ("stance_to_shoulder_ratio", "스탠스/어깨 폭 비율", "target", 0.08),
]

dup = [h for h in hist if h.get("video_md5") == cur.get("video_md5")]
lines = []
if dup:
    lines.append(f"**같은 영상**: 이전 기록({dup[0].get('shot_time') or dup[0].get('analyzed')})과 파일이 완전히 같습니다. 변화 판단 불가.")
prev = [h for h in hist if h.get("video_md5") != cur.get("video_md5")][-4:]
cg = cur.get("camera_geom")
if prev and isinstance(cg, dict):
    pg = prev[-1].get("camera_geom")
    import math
    moved = (not isinstance(pg, dict)) or any(
        math.dist(cg[k + "_pos"], pg[k + "_pos"]) > 0.3 or max(cg[k + "_f"], pg[k + "_f"]) / min(cg[k + "_f"], pg[k + "_f"]) > 1.25
        for k in ("front", "dtl"))
    if moved:
        lines.append("**카메라 위치/화각이 직전 회차와 다릅니다**(또는 직전 기록에 카메라 정보 없음): "
                     "척추 앞숙임 같은 절대 각도는 비교에서 빼고, 어드레스 대비 변화량 지표만 비교하세요.")

head = ["지표"] + [(h.get("shot_time") or h.get("analyzed") or "?")[:10] for h in prev] + ["이번", "직전 대비", "프로 목표"]
lines.append("| " + " | ".join(head) + " |")
lines.append("|" + "---|" * len(head))
tgt = cur.get("pro_targets", {})
for key, label, better, thr in ROWS:
    v = cur["metrics"].get(key)
    if v is None:
        continue
    row = [label] + [f"{h['metrics'].get(key, '-')}" for h in prev] + [f"**{v}**"]
    if prev and key in prev[-1]["metrics"]:
        d = v - prev[-1]["metrics"][key]
        if abs(d) < thr:
            verdict = "변화 없음(오차 범위)"
        elif better == "low":
            verdict = "좋아짐" if d < 0 else "나빠짐"
        elif better == "high":
            verdict = "좋아짐" if d > 0 else "나빠짐"
        elif better == "target" and key in tgt:
            verdict = "좋아짐" if abs(v - tgt[key]) < abs(prev[-1]["metrics"][key] - tgt[key]) else "나빠짐"
        else:
            verdict = "변화"
        row.append(f"{d:+.1f} {verdict}")
    else:
        row.append("첫 기록")
    row.append(f"{tgt[key]}" if key in tgt else "-")
    lines.append("| " + " | ".join(row) + " |")

md = "\n".join(lines)
open(f"{S}/out/compare.md", "w").write(md + "\n")
print(md)
sys.exit(3 if dup else 0)
