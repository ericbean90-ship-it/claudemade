# 레슨 프로 옆 타석 영상 (기본 비교 모델, 사용자 결정 2026-11-27)

## 무엇인가
- 사용자의 레슨 프로가 **사용자가 연습하는 바로 그 타석**에서 찍은 정면·후면 영상입니다.
- 프로를 영상에서 누끼 따서(배경 제거) 사용자 영상의 옆 타석에 세웁니다. 실제 사람이므로 체형을 바꾸지 않습니다.
- **완전히 불투명하게, 나란히 (사용자 결정 2026-11-27):** 반투명 겹치기는 만들지 않습니다. 어두운 바지나 흔들린 팔이 반쯤 비치는 곳은 `solid_alpha`가 메웁니다(몸 안쪽 100%, 가장자리 1~2px만 부드럽게).
- 드라이브 폴더의 `pro-asset-front.zip`, `pro-asset-back.zip`(각 3MB 미만)에 들어 있습니다.
  - 내용: 색 영상, 알파(누끼) 영상, 관절, 클럽 궤적, 배경판, 키프레임, 촬영 시각, 프로 자신의 지표(`pro_metrics.json`)
- **실제 인물입니다.** 묶음, 누끼, 원본 영상은 공개 저장소에 올리지 않습니다. 사용자 비공개 드라이브에만 둡니다.

## 매 회차 실행
```bash
python3 $SK/scripts/pro_asset.py unpack $W/pro $W/pro-asset-front.zip $W/pro-asset-back.zip
python3 $SK/scripts/pro_overlay.py $W $W/pro $W/out check   # check_pro_<P1|P4|P7|P10>.jpg
python3 $SK/scripts/pro_overlay.py $W $W/pro $W/out         # pro_front_nextbay.mp4, pro_dtl_nextbay.mp4, pro_keyframes.jpg
```
- **위치·크기:**
  - 두 영상의 빈 배경(사람, 로고, 시각 표시 제외)을 SIFT로 맞춰 닮음변환을 구합니다.
  - 같은 카메라면 배율 1.00, 이동 몇 px입니다.
  - 그다음 프로를 빈 바닥이 넓은 쪽 옆 타석으로 옮깁니다.
- **시간:** 프로의 P1~P10을 사용자 P1~P10에 맞추고, 가장 가까운 실제 프레임을 씁니다.
- **확인 이미지:** 각 화면 오른쪽 칸 "정합 확인: 같은 자리 50%"에서 프로가 사용자 자리에 겹쳐 서야 합니다. 발 위치가 크게 어긋나면 정합이 잘못된 것입니다.
- **다른 타석이면:** `background match failed`가 나옵니다. 다리 길이로 크기를 맞추므로 프로와 사용자의 상대 크기는 근사치입니다. 사용자에게 알립니다.

## 프로 영상을 새로 받았을 때 (묶음 다시 만들기)
```bash
P=<scratchpad>/pro-<파일명시각>
python3 $SK/scripts/stamp.py $P <front> <back>              # out/stamp.png 확인, 다르면 stamp.json 수정
bash $SK/scripts/run_pipeline.sh <front> <back> $P analyze  # 관절, 카메라 보정, 키프레임, 지표 (공 경고 무시 가능)
python3 $SK/scripts/ball_grid.py $P                          # 공 픽셀 읽기 → ball.py, ... metrics 다시 실행
python3 $SK/scripts/pro_cutout.py $P <fu> <fv> <bu> <bv>     # 누끼 + 클럽 (약 6분)
python3 $SK/scripts/pro_asset.py pack $P <out>               # pro-asset-front.zip, pro-asset-back.zip
python3 $SK/scripts/pro_asset.py preview $P <out>/<prefix>_레슨프로   # 누끼 미리보기 mp4 + 투명 webm
```
- 키프레임 시트(`out/keyframes_sheet.jpg`)를 사용자 영상과 똑같이 눈으로 확인합니다.
- **이 영상은 사용자 기록이 아닙니다.** `history/`나 `swing-*.json`으로 드라이브에 올리지 않습니다.
- 새 묶음을 올리면 예전 `pro-asset-*.zip`은 사용자에게 묻고 정리합니다.

## 누끼 방법 (`pro_cutout.py`)
- **사람:** Robust Video Matting(`rvm_mobilenetv3_fp32.onnx`, GitHub 공식 배포, 15MB)
  - 프레임 사이 기억이 있어서 깜빡임이 없고, 프레임당 0.05초입니다.
  - BiRefNet은 품질은 좋지만 프레임당 10~20초라 영상에는 쓰지 않습니다.
- **클럽:** 사람 누끼 모델은 가늘고 흐린 샤프트를 놓칩니다. 그래서 따로 찾습니다.
  1. 카메라가 고정이므로 배경판을 만듭니다. 프로가 가리지 않은 프레임만, 클럽이 움직이기 시작한 뒤만 씁니다. 후면 스크린 영상이 바뀌므로 앞뒤 ±0.7초 구간 배경판을 씁니다.
  2. 배경과 달라진 픽셀에서, 손에서 시작해 끊기지 않고 이어지는 직선을 찾습니다. 몸에 가려진 구간은 끊김으로 치지 않습니다.
  3. 정면·후면의 2D 각도로 **3D 샤프트 방향**을 구합니다. 길이는 어드레스 때 손-공 거리로 고정합니다. 어드레스와 임팩트에서는 클럽 헤드를 공에 둡니다.
  4. 각 화면으로 다시 투영해 은색 샤프트와 헤드를 그립니다. 샤프트 가운데가 머리나 몸통보다 카메라에서 멀면 몸 뒤로 그립니다.
  - 첫 영상 결과: P1~P10 전 구간에서 그려짐, 프레임 사이 회전 중앙값 약 2°
- **한계:** 클럽은 실제 픽셀이 아니라 찾은 각도로 다시 그린 선입니다. 임팩트 직전처럼 아주 빠른 구간은 앞뒤 프레임에서 보간합니다.

## 프로 실측 지표
- 묶음 안 `pro_metrics.json`에 사용자 지표와 같은 계산으로 낸 프로의 값이 있습니다(같은 카메라).
- 투어 평균 대신 비교 기준으로 쓸 수 있습니다. 리포트에 넣을지는 사용자에게 묻습니다.
- 실제 인물의 데이터이므로 수치는 저장소 문서에 적지 않습니다.
