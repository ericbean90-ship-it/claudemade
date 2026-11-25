"""The user's body size (segment lengths), measured every session and pooled across sessions.

The model-pro mannequin is drawn at THIS body size (ref_overlay.py), not fitted to the address
posture: a posture fit makes the model 5-20 % too big whenever the user stands more upright.

Units: metres in the calibrated (MediaPipe-world) scale. They are consistent between sessions,
which is all the model scaling needs (body length x f / depth -> pixels).

Usage:
  python body_profile.py <workdir> [<seed.json>]
      pool = seed sessions + every history/* record with a "body" field + this session
      -> <workdir>/body_profile.json (median per segment) and a Korean summary
  python body_profile.py --make-seed <out.json> <key>=<workdir> [<key>=<workdir> ...]
      one-off: build a seed file from old work dirs (key = record name, e.g. swing-2026-11-10_101026)
"""
import os
import sys
import glob
import json
import numpy as np
from scipy.ndimage import median_filter

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from geom import load_cams, triangulate

JOINTS = (0, 7, 8, 11, 12, 13, 14, 15, 16, 23, 24, 25, 26, 27, 28)
NAMES = {"shoulder_w": "어깨 너비", "hip_w": "골반 너비", "upper_arm": "위팔", "forearm": "아래팔",
         "thigh": "허벅지", "shin": "정강이", "torso": "몸통", "neck": "목(어깨-귀)", "body_len": "몸 길이(다리+몸통)"}


def address_joints(S):
    """Triangulated joints (golfer frame, m) for every 2nd frame of the 40 frames before P1."""
    cf, cb, _ = load_cams(S)
    F = median_filter(np.load(f"{S}/pose/front.npz")["img"], size=(5, 1, 1), mode="nearest")
    B = median_filter(np.load(f"{S}/pose/back.npz")["img"], size=(5, 1, 1), mode="nearest")
    p1 = json.load(open(f"{S}/keyframes.json"))["P1"]
    fr = range(max(p1 - 40, 0), p1 + 1, 2)
    return {j: np.array([triangulate(cf, F[i, j, :2], cb, B[i, j, :2])[0] for i in fr]) for j in JOINTS}


def measure(S):
    J = address_joints(S)
    L = lambda a, b: float(np.median(np.linalg.norm(J[a] - J[b], axis=1)))
    ms, mh, me = (J[11] + J[12]) / 2, (J[23] + J[24]) / 2, (J[7] + J[8]) / 2
    b = {
        "shoulder_w": L(11, 12), "hip_w": L(23, 24),
        "upper_arm": (L(11, 13) + L(12, 14)) / 2, "forearm": (L(13, 15) + L(14, 16)) / 2,
        "thigh": (L(23, 25) + L(24, 26)) / 2, "shin": (L(25, 27) + L(26, 28)) / 2,
        "torso": float(np.median(np.linalg.norm(ms - mh, axis=1))),
        "neck": float(np.median(np.linalg.norm(me - ms, axis=1))),
    }
    b["body_len"] = b["thigh"] + b["shin"] + b["torso"]
    return {k: round(v, 4) for k, v in b.items()}


def pool(S, seed=None):
    sess = {}
    if seed and os.path.exists(seed):
        sess.update(json.load(open(seed))["sessions"])
    for p in glob.glob(f"{S}/history/*"):
        try:
            r = json.load(open(p))
        except Exception:  # noqa: BLE001
            continue
        if isinstance(r, dict) and r.get("body"):
            sess[os.path.splitext(os.path.basename(p))[0]] = r["body"]
    cur = None
    mp = f"{S}/out/metrics.json"
    if os.path.exists(mp):
        cur = json.load(open(mp)).get("body")
    sess[os.path.basename(os.path.normpath(S))] = cur or measure(S)
    keys = sorted(sess)
    med = {k: round(float(np.median([sess[s][k] for s in keys])), 4) for k in NAMES}
    return {"n_sessions": len(keys), "sessions": keys, "median_m": med,
            "this_session_m": sess[os.path.basename(os.path.normpath(S))]}


if __name__ == "__main__":
    if sys.argv[1] == "--make-seed":
        out = sys.argv[2]
        sess = {}
        for kv in sys.argv[3:]:
            k, w = kv.split("=", 1)
            sess[k] = measure(w)
            print(k, sess[k])
        json.dump({"note": "past sessions without a 'body' field in their record", "sessions": sess},
                  open(out, "w"), ensure_ascii=False, indent=1)
        sys.exit(0)
    S = sys.argv[1]
    prof = pool(S, sys.argv[2] if len(sys.argv) > 2 else None)
    json.dump(prof, open(f"{S}/body_profile.json", "w"), ensure_ascii=False, indent=1)
    med, cur = prof["median_m"], prof["this_session_m"]
    print(f"신체 프로필: {prof['n_sessions']}회차 중앙값 (이번 회차와의 차이)")
    for k, nm in NAMES.items():
        print(f"  {nm:12s} {med[k]*100:6.1f}  ({(cur[k]/med[k]-1)*100:+5.1f}%)")
