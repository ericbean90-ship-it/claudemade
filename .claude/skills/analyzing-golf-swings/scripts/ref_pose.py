"""MediaPipe pose on each separated mannequin layer (upscaled crop). Saves ref/pose_{view}.json"""
import sys, json, glob, numpy as np, cv2, mediapipe as mp
from mediapipe.tasks import python as mpt
from mediapipe.tasks.python import vision
S=sys.argv[1]; R=f"{S}/ref"
lm=vision.PoseLandmarker.create_from_options(vision.PoseLandmarkerOptions(
    base_options=mpt.BaseOptions(model_asset_path=f"{S}/models/pose_heavy.task"),
    running_mode=vision.RunningMode.IMAGE, num_poses=1,
    min_pose_detection_confidence=0.2, min_pose_presence_confidence=0.2))
uniq=json.load(open(f"{R}/uniq.json"))
for view in ('dtl','front'):
    out={}
    for k in uniq:
        im=cv2.imread(f"{R}/{view}/{k:04d}.png",cv2.IMREAD_UNCHANGED)
        a=im[...,3:4]/255.0; rgb=(im[...,:3]*a).astype(np.uint8)
        ys,xs=np.where(im[...,3]>40); x0,x1=max(xs.min()-40,0),min(xs.max()+40,im.shape[1]); y0,y1=max(ys.min()-40,0),min(ys.max()+40,im.shape[0])
        crop=cv2.resize(rgb[y0:y1,x0:x1],None,fx=4,fy=4,interpolation=cv2.INTER_CUBIC)
        # put on mid-grey background so the white mannequin reads like a person
        bg=np.full_like(crop,70); aa=cv2.resize(im[y0:y1,x0:x1,3],None,fx=4,fy=4)[...,None]/255.0
        crop=(crop*aa+bg*(1-aa)).astype(np.uint8)
        r=lm.detect(mp.Image(image_format=mp.ImageFormat.SRGB,data=cv2.cvtColor(crop,cv2.COLOR_BGR2RGB)))
        if r.pose_landmarks:
            out[k]=[[q.x*crop.shape[1]/4+x0,q.y*crop.shape[0]/4+y0,q.visibility] for q in r.pose_landmarks[0]]
    json.dump(out,open(f"{R}/pose_{view}.json","w"))
    print(view,'detected',len(out),'of',len(uniq))
