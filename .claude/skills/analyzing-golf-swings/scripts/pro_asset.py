"""Keep the lesson pro's cut-out as two small zips (one per camera) for the Drive folder, and preview it.

  python pro_asset.py pack <pro workdir> <out dir>
      -> pro-asset-front.zip, pro-asset-back.zip (each < 2.9 MB: colour video + alpha video + pose,
         club, keyframes, background plate, stamp, the pro's own metrics)
  python pro_asset.py unpack <out pro dir> <zip> [<zip> ...]
      -> vid/{front,back}.mp4, cut/{front,back}_alpha.npy, cut/club.json, pose/*.npz, ... (pro_overlay.py input)
  python pro_asset.py preview <pro dir> <out prefix>
      -> <prefix>_누끼_정면후면.mp4 (pro alone, both cameras, grey background) and
         <prefix>_누끼_정면.webm / _후면.webm (transparent VP9: drop onto any background)

The cut-out shows a real person (the user's lesson pro): keep it in the user's private Drive only.
"""
import os
import io
import sys
import json
import glob
import shutil
import zipfile
import subprocess
import tempfile
import numpy as np
import cv2

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from pro_cutout import read, render_club

MAX = 2_900_000


def enc(frames, path, crf, gray=False, fps=70):
    h, w = frames[0].shape[:2]
    p = subprocess.Popen(["ffmpeg", "-v", "error", "-y", "-f", "rawvideo", "-pix_fmt", "gray" if gray else "bgr24",
                          "-s", f"{w}x{h}", "-r", str(fps), "-i", "-", "-c:v", "libx264", "-preset", "slow",
                          "-crf", str(crf), "-pix_fmt", "yuv420p", path], stdin=subprocess.PIPE)
    for f in frames:
        p.stdin.write(np.ascontiguousarray(f).tobytes())
    p.stdin.close()
    p.wait()


def pack(S, out):
    os.makedirs(out, exist_ok=True)
    club = json.load(open(f"{S}/cut/club.json"))
    shared = {n: open(f"{S}/{n}", "rb").read() for n in ("keyframes.json", "stamp.json") if os.path.exists(f"{S}/{n}")}
    if os.path.exists(f"{S}/out/metrics.json"):
        shared["pro_metrics.json"] = open(f"{S}/out/metrics.json", "rb").read()
    shared["cut/club.json"] = json.dumps(club).encode()
    for v in ("front", "back"):
        F = read(f"{S}/vid/{v}.mp4")
        A = np.load(f"{S}/cut/{v}_alpha.npy")
        tmp = tempfile.mkdtemp()
        enc(A, f"{tmp}/alpha.mp4", 18, gray=True)
        for crf in (22, 24, 26, 28, 30, 32):
            enc(F, f"{tmp}/rgb.mp4", crf)
            zp = f"{out}/pro-asset-{v}.zip"
            with zipfile.ZipFile(zp, "w", zipfile.ZIP_STORED) as z:
                z.write(f"{tmp}/rgb.mp4", f"vid/{v}.mp4")
                z.write(f"{tmp}/alpha.mp4", f"cut/{v}_alpha.mp4")
                z.write(f"{S}/cut/{v}_plate.png", f"cut/{v}_plate.png")
                buf = io.BytesIO()
                np.savez_compressed(buf, img=np.load(f"{S}/pose/{v}.npz")["img"].astype(np.float32))
                z.writestr(f"pose/{v}.npz", buf.getvalue())
                for n, b in shared.items():
                    z.writestr(n, b)
            if os.path.getsize(zp) <= MAX:
                break
        shutil.rmtree(tmp)
        print(f"{zp}: {os.path.getsize(zp) / 1e6:.2f} MB (colour crf {crf})")


def unpack(D, zips):
    os.makedirs(D, exist_ok=True)
    for zp in zips:
        data = open(zp, "rb").read()
        if not data.startswith(b"PK"):
            import base64
            data = base64.b64decode(data)
        with zipfile.ZipFile(io.BytesIO(data)) as z:
            z.extractall(D)
    for p in glob.glob(f"{D}/cut/*_alpha.mp4"):
        cap, A = cv2.VideoCapture(p), []
        while True:
            ok, f = cap.read()
            if not ok:
                break
            A.append(f[..., 0])
        np.save(p.replace(".mp4", ".npy"), np.array(A, np.uint8))
    print(D, "views:", sorted(os.path.basename(p)[:-4] for p in glob.glob(f"{D}/vid/*.mp4")))


def preview(D, prefix):
    club = json.load(open(f"{D}/cut/club.json"))
    K = json.load(open(f"{D}/keyframes.json"))
    lay = {}
    for v in ("front", "back"):
        F = read(f"{D}/vid/{v}.mp4")
        A = np.load(f"{D}/cut/{v}_alpha.npy")
        L = np.load(f"{D}/pose/{v}.npz")["img"]
        out = []
        for k in range(min(len(F), len(A))):
            x = np.dstack([F[k].astype(np.float32), A[k].astype(np.float32)])
            render_club(x, (L[k, 15, :2] + L[k, 16, :2]) / 2, club[v]["angle"][k], club[v]["length"][k],
                        behind=bool(club[v]["behind"][k]))
            out.append(np.clip(x, 0, 255).astype(np.uint8))
        lay[v] = out
    n = min(len(lay["front"]), len(lay["back"]))
    rng = range(max(0, K["P1"] - 70), n)
    tmp = tempfile.mkdtemp()
    grey = []
    for k in rng:
        row = []
        for v in ("front", "back"):
            x = lay[v][k].astype(np.float32)
            a = x[..., 3:4] / 255
            row.append((x[..., :3] * a + 95 * (1 - a)).astype(np.uint8))
        grey.append(np.hstack(row))
    enc(grey, f"{prefix}_누끼_정면후면.mp4", 20)
    for v, name in (("front", "정면"), ("back", "후면")):
        d = f"{tmp}/{v}"
        os.makedirs(d)
        for j, k in enumerate(rng):
            cv2.imwrite(f"{d}/{j:04d}.png", lay[v][k])
        subprocess.run(["ffmpeg", "-v", "error", "-y", "-framerate", "70", "-i", f"{d}/%04d.png", "-c:v", "libvpx-vp9",
                        "-pix_fmt", "yuva420p", "-b:v", "1.2M", "-auto-alt-ref", "0", f"{prefix}_누끼_{name}.webm"],
                       check=True)
    shutil.rmtree(tmp)
    for p in sorted(glob.glob(f"{prefix}_누끼_*")):
        print(p, f"{os.path.getsize(p) / 1e6:.2f} MB")


if __name__ == "__main__":
    cmd = sys.argv[1]
    if cmd == "pack":
        pack(sys.argv[2], sys.argv[3])
    elif cmd == "unpack":
        unpack(sys.argv[2], sys.argv[3:])
    elif cmd == "preview":
        preview(sys.argv[2], sys.argv[3])
    else:
        sys.exit(__doc__)
