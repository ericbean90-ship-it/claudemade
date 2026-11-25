"""Keep the prepared model-pro mannequin as one zip, so later sessions need no Modelgolf upload.

  python ref_model.py pack <workdir> <out.zip>
      needs <workdir>/ref from ref_prepare.py. Measures the mannequin's body length in each view
      (ref/model.json), blanks colour outside each mannequin's matte, and zips what ref_overlay.py reads.
  python ref_model.py unpack <zip or Drive blob file> <workdir>
      -> <workdir>/ref (a Drive download may be base64 text; both forms are accepted)

The zip holds frames derived from someone else's video: keep it in the user's private Drive only.
"""
import os
import io
import sys
import glob
import json
import base64
import zipfile
import numpy as np
import cv2


def mid(P, a, b):
    return (P[a, :2] + P[b, :2]) / 2


def model_geometry(R):
    """Mannequin segment lengths in pixels of each view (two-view estimate: x from front, z from DTL)."""
    MF = {int(k): np.array(v) for k, v in json.load(open(f"{R}/pose_front.json")).items()}
    MD = {int(k): np.array(v) for k, v in json.load(open(f"{R}/pose_dtl.json")).items()}
    rows, ratio = [], []
    for k in sorted(set(MF) & set(MD)):
        f, d = MF[k], MD[k]
        s = (mid(f, 27, 28)[1] - mid(f, 11, 12)[1]) / (mid(d, 27, 28)[1] - mid(d, 11, 12)[1])
        P3 = lambda j: np.array([f[j, 0], (f[j, 1] + s * d[j, 1]) / 2, s * d[j, 0]])
        L = lambda a, b: np.linalg.norm(P3(a) - P3(b))
        torso = np.linalg.norm((P3(11) + P3(12)) / 2 - (P3(23) + P3(24)) / 2)
        rows.append([(L(23, 25) + L(24, 26)) / 2, (L(25, 27) + L(26, 28)) / 2, torso,
                     (L(11, 13) + L(12, 14)) / 2, (L(13, 15) + L(14, 16)) / 2])
        ratio.append(s)
    med = np.median(np.array(rows), axis=0)
    s = float(np.median(ratio))           # front px per DTL px
    seg = dict(zip(["thigh", "shin", "torso", "upper_arm", "forearm"], [float(x) for x in med]))
    body = seg["thigh"] + seg["shin"] + seg["torso"]
    return {"segments_front_px": seg, "front_per_dtl_px": s,
            "body_len_px": {"front": body, "dtl": body / s}, "frames_used": len(rows)}


def pack(S, out):
    R = f"{S}/ref"
    geo = model_geometry(R)
    json.dump(geo, open(f"{R}/model.json", "w"), indent=1)
    with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED) as z:
        for v in ("front", "dtl"):
            for p in sorted(glob.glob(f"{R}/{v}/*.png")):
                im = cv2.imread(p, cv2.IMREAD_UNCHANGED)
                im[im[..., 3] == 0, :3] = 0            # colour of the other mannequin / background is never used
                ok, buf = cv2.imencode(".png", im, [cv2.IMWRITE_PNG_COMPRESSION, 9])
                z.writestr(f"{v}/{os.path.basename(p)}", buf.tobytes())
        for n in ("pose_front.json", "pose_dtl.json", "uniq.json", "model.json", "keyframes.json"):
            if os.path.exists(f"{R}/{n}"):
                z.write(f"{R}/{n}", n)
    print(f"{out}: {os.path.getsize(out)/1e6:.2f} MB, body length px {geo['body_len_px']}")


def unpack(src, S):
    data = open(src, "rb").read()
    if not data.startswith(b"PK"):
        data = base64.b64decode(data)
    R = f"{S}/ref"
    os.makedirs(R, exist_ok=True)
    with zipfile.ZipFile(io.BytesIO(data)) as z:
        z.extractall(R)
    n = len(glob.glob(f"{R}/front/*.png"))
    print(f"{R}: {n} poses per view, model.json {'ok' if os.path.exists(f'{R}/model.json') else 'MISSING'}")


if __name__ == "__main__":
    if sys.argv[1] == "pack":
        pack(sys.argv[2], sys.argv[3])
    elif sys.argv[1] == "unpack":
        unpack(sys.argv[2], sys.argv[3])
    else:
        sys.exit(__doc__)
