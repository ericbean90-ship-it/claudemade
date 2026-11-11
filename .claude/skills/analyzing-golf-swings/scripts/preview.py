"""Draw pro skeleton (stick) over the user's keyframes in both views for checking."""
import os
import sys, json, cv2, numpy as np
S=sys.argv[1]; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from geom import load_cams
from pro_model import ProGolfer
cf,cb,XG=load_cams(S)
ball=np.load(f'{S}/ball.npy')
kf=json.load(open(f'{S}/keyframes.json'))
pro=ProGolfer(XG,ball,kf)
def frames(v):
    c=cv2.VideoCapture(f'{S}/vid/{v}.mp4'); out=[]
    while True:
        ok,f=c.read()
        if not ok: break
        out.append(f)
    return out
FR={'front':frames('front'),'back':frames('back')}
E=[(11,12),(11,13),(13,15),(12,14),(14,16),(11,23),(12,24),(23,24),(23,25),(25,27),(24,26),(26,28),(27,29),(29,31),(27,31),(28,30),(30,32),(28,32),('neck','head')]
tiles=[]
for name,i in kf.items():
    J=pro.pose(i)
    row=[]
    for v,cam in (('front',cf),('back',cb)):
        im=FR[v][i].copy()
        P={k:cam.project(np.asarray(p))[0][0] for k,p in J.items()}
        for a,b in E:
            cv2.line(im,tuple(int(x) for x in P[a]),tuple(int(x) for x in P[b]),(255,255,0),2,cv2.LINE_AA)
        cv2.line(im,tuple(int(x) for x in P['grip']),tuple(int(x) for x in P['clubhead']),(0,255,0),2,cv2.LINE_AA)
        cv2.circle(im,tuple(int(x) for x in P['head']),int(abs(cam.f*0.10/cam.to_cam(J['head'][None])[0,2])),(255,255,0),2)
        cv2.putText(im,f'{name} f{i}',(8,24),0,0.7,(0,255,255),2)
        row.append(im)
    tiles.append(np.hstack(row))
grid=[np.hstack(tiles[k:k+2]) for k in range(0,len(tiles),2)]
cv2.imwrite(f'{S}/frames/preview.jpg',cv2.resize(np.vstack(grid),None,fx=0.5,fy=0.5),[cv2.IMWRITE_JPEG_QUALITY,85])
for k in range(0,len(tiles),2):
    cv2.imwrite(f'{S}/frames/preview_{k//2}.jpg',np.hstack(tiles[k:k+2]),[cv2.IMWRITE_JPEG_QUALITY,85])
print('ok')
