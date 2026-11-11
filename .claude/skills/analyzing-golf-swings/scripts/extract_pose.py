"""Run MediaPipe Pose Landmarker (heavy) over a video, save per-frame landmarks.

Usage: python extract_pose.py <video> <model.task> <out.npz>
Saves:
  img  : (N, 33, 4)  x_px, y_px, z(rel), visibility   (NaN when not detected)
  world: (N, 33, 3)  metric world coords (hip-centred)
  fps, w, h
"""
import sys
import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python as mpt
from mediapipe.tasks.python import vision

video, model, out = sys.argv[1:4]
cap = cv2.VideoCapture(video)
fps = cap.get(cv2.CAP_PROP_FPS)
w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

opts = vision.PoseLandmarkerOptions(
    base_options=mpt.BaseOptions(model_asset_path=model),
    running_mode=vision.RunningMode.VIDEO,
    num_poses=1,
    min_pose_detection_confidence=0.3,
    min_pose_presence_confidence=0.3,
    min_tracking_confidence=0.3,
)
lm = vision.PoseLandmarker.create_from_options(opts)

imgs, worlds = [], []
i = 0
while True:
    ok, frame = cap.read()
    if not ok:
        break
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    res = lm.detect_for_video(mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb),
                              int(round(i * 1000.0 / fps)))
    if res.pose_landmarks:
        p = res.pose_landmarks[0]
        imgs.append([[q.x * w, q.y * h, q.z * w, q.visibility] for q in p])
        pw = res.pose_world_landmarks[0]
        worlds.append([[q.x, q.y, q.z] for q in pw])
    else:
        imgs.append(np.full((33, 4), np.nan))
        worlds.append(np.full((33, 3), np.nan))
    i += 1

np.savez(out, img=np.array(imgs, float), world=np.array(worlds, float), fps=fps, w=w, h=h)
print(out, "frames", i, "missing", int(np.isnan(np.array(imgs)[:, 0, 0]).sum()))
