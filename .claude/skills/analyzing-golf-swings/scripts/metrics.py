"""Standard swing metrics for one session -> <workdir>/out/metrics.json (the record kept in history).

Usage: python metrics.py <workdir> <front_video_path> <dtl_video_path>
Units: real-world cm / degrees (model metres divided by the body-scale factor of pro_model).
Sign conventions: sway/shift toward the TARGET is positive unless the key says otherwise.
"""
import os
import re
import sys
import json
import hashlib
import datetime
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from geom import load_cams
from pro_model import ProGolfer

S, FRONT, DTL = sys.argv[1:4]
cf, cb, XG = load_cams(S)
KF = json.load(open(f"{S}/keyframes.json"))
pro = ProGolfer(XG, np.load(f"{S}/ball.npy"), KF)
SC = pro.scale
U = np.load(f"{S}/user3d.npz")
F = np.load(f"{S}/pose/front.npz")["img"]
P = {k: KF[k] for k in KF}
a = P["P1"]


def cm(key, i, ax):
    return float((U[key][i][ax] - U[key][a][ax]) * 100 / SC)


def spine_fwd(i):
    v = U["sho"][i] - U["pel"][i]
    return float(np.degrees(np.arctan2(v[2], v[1])))


def md5(p):
    h = hashlib.md5()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


m = re.search(r"(20\d{2})(\d{2})(\d{2})_(\d{2})(\d{2})(\d{2})", os.path.basename(FRONT))
shot = (f"{m[1]}-{m[2]}-{m[3]} {m[4]}:{m[5]}:{m[6]}" if m else None)

addr = slice(max(a - 40, 0), a)
shw = np.nanmedian(np.linalg.norm(F[addr, 11, :2] - F[addr, 12, :2], axis=1))
stance = np.nanmedian(np.abs(F[addr, 27, 0] - F[addr, 28, 0]))

metrics = {
    "head_sway_top_cm": -cm("head", P["P4"], 0),          # + = moved away from target
    "pelvis_sway_top_cm": -cm("pel", P["P4"], 0),         # + = moved away from target
    "pelvis_shift_impact_cm": cm("pel", P["P7"], 0),      # + = toward target
    "head_impact_cm": cm("head", P["P7"], 0),             # + = toward target
    "pelvis_toward_ball_impact_cm": cm("pel", P["P7"], 2),
    "spine_fwd_address_deg": spine_fwd(a),
    "spine_loss_impact_deg": spine_fwd(a) - spine_fwd(P["P7"]),
    "spine_fwd_finish_deg": spine_fwd(P["P10"]),
    # hands farther from the body than at address during the late downswing (P6..just before impact);
    # a single impact frame is too noisy because the hands sweep inward fast through the ball
    "hands_out_downswing_cm": float((np.mean(U["hands"][P["P6"]:P["P7"] - 1, 2]) - U["hands"][a][2]) * 100 / SC),
    "hands_outside_p6_vs_p2_cm": float((U["hands"][P["P6"]][2] - U["hands"][P["P2"]][2]) * 100 / SC),
    "pelvis_shift_finish_cm": cm("pel", P["P10"], 0),
    "tempo_ratio": float((P["P4"] - P["P1"]) / max(P["P7"] - P["P4"], 1)),
    "stance_to_shoulder_ratio": float(stance / shw),
}

# the pro model's own numbers on the same scale = targets
J0 = pro.pose(a)


def pcm(key, i, ax):
    return float((pro.pose(i)[key][ax] - J0[key][ax]) * 100 / SC)


def pspine(i):
    J = pro.pose(i)
    v = (J[11] + J[12]) / 2 - J["pel"]
    return float(np.degrees(np.arctan2(v[2], v[1])))


targets = {
    "head_sway_top_cm": -pcm("head", P["P4"], 0),
    "pelvis_sway_top_cm": -pcm("pel", P["P4"], 0),
    "pelvis_shift_impact_cm": pcm("pel", P["P7"], 0),
    "spine_loss_impact_deg": pspine(a) - pspine(P["P7"]),
    "hands_out_downswing_cm": float((np.mean([pro.pose(i)["grip"][2] for i in range(P["P6"], P["P7"] - 1)])
                                     - J0["grip"][2]) * 100 / SC),
    "tempo_ratio": 3.0,
    "stance_to_shoulder_ratio": 1.1,
}

rec = {
    "schema": 1,
    "shot_time": shot,
    "analyzed": datetime.datetime.now().strftime("%Y-%m-%d %H:%M"),
    "video_md5": {"front": md5(FRONT), "dtl": md5(DTL)},
    "keyframes": P,
    "metrics": {k: round(v, 1) for k, v in metrics.items()},
    "pro_targets": {k: round(v, 1) for k, v in targets.items()},
    "notes": "",
}
json.dump(rec, open(f"{S}/out/metrics.json", "w"), ensure_ascii=False, indent=1)
print(json.dumps(rec["metrics"], ensure_ascii=False))
