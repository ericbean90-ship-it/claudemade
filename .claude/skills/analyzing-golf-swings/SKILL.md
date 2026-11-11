---
name: analyzing-golf-swings
description: Use when the user uploads golf swing videos (a face-on "front" clip and a down-the-line "back" clip of the same shot, e.g. rangex *_frontVideo.MP4 / *_backVideo.MP4), asks for swing coaching or problems, a pro overlay/ghost comparison, an overlay of the Modelgolf model-pro render, or asks whether their swing changed since last time (스윙 분석, 변화가 있나, 프로랑 겹쳐줘).
---

# 골프 스윙 영상 분석·코칭

## 개요
같은 샷을 찍은 정면 영상과 후면(DTL) 영상 두 개로 3D 지표를 측정합니다. 그 지표로 지난 회차와 비교하고, 코칭 리포트와 비교 영상을 만듭니다.

**두 가지 원칙**
- 모든 수치는 `metrics.json` 한 곳에서 나옵니다.
- 회차 기록은 구글 드라이브에만 보관합니다.

모든 답은 한국어로 씁니다. 먼저 큰 흐름을 한 문장으로 요약하고, 중요한 순서대로 쓰고, 어려운 개념은 비유로 설명합니다.

## 준비물
- **필수:** 같은 샷의 정면(face-on)과 후면(DTL) 영상. 파일명 타임스탬프가 같아야 합니다.
- **선택:** Modelgolf 모델프로 영상(4:09). 사용자가 올려야 합니다.
  - YouTube 다운로드는 서버가 403으로 막습니다. **우회 시도 금지.**

## 작업 순서
`SK=<이 스킬 폴더>`, `W=<scratchpad>/swing-<촬영시각>`

1. **환경 준비:** `bash $SK/scripts/setup.sh`
2. **분석:** `bash $SK/scripts/run_pipeline.sh <front> <back> $W analyze`
   - 포즈 추출, 카메라 보정, P1~P10 자동 검출, 지표 계산까지 합니다.
3. **키프레임 확인 (필수):** `$W/out/keyframes_sheet.jpg`를 **반드시 열어 봅니다.**
   - P4(탑)과 P7(임팩트)이 틀렸으면 `$W/keyframes.json`을 고칩니다.
   - 고친 뒤 `... $W metrics`를 다시 실행합니다.
   - 공 위치 기본값은 rangex 카메라 기준입니다. 카메라가 바뀌었으면 `BALL_*` 환경변수로 지정합니다.
4. **회차 비교 (구글 드라이브):**
   1. `search_files`로 `title contains '골프 스윙 기록'` 폴더를 찾습니다.
   2. 그 폴더(`parentId`)의 `swing-*.json` 파일을 모두 찾습니다.
   3. 파일마다 `download_file_content`를 호출합니다. `read_file_content`는 빈 값을 돌려주므로 쓰지 않습니다.
   4. 결과에 표시된 저장 경로의 파일을 `$W/history/`에 복사합니다.
   5. `python3 $SK/scripts/compare.py $W`를 실행합니다.
   - **exit 3 = 이전 회차와 같은 영상**입니다. 사용자에게 알리고 새로 분석하지 않습니다.
5. **리포트:** `reference/coaching.md`의 판정 기준과 템플릿을 따릅니다.
   - 숫자는 `$W/out/metrics.json`과 `compare.md`에서만 가져옵니다.
   - 키프레임 시트로 테이크어웨이 경로와 샤프트 기울기를 눈으로 확인하고, 그 결과를 보탭니다.
6. **영상 만들기:**
   - 프로 고스트: `... $W render` 실행 → `ghost_*.mp4`
   - 모델프로 영상을 받았다면 `reference/model-pro.md`를 따릅니다.
7. **기록 저장:**
   - 위 폴더에 `create_file`로 `swing-<YYYY-MM-DD_HHMMSS>.json`을 새로 만듭니다.
   - 옵션: `textContent` = `metrics.json` 내용, `contentMimeType: text/plain`, `disableConversionToGoogleType: true`
   - `notes`에 이번 근본 과제와 드릴을 적습니다.
   - 같은 영상이면 저장하지 않습니다.
8. **전달:** `SendUserFile`로 영상, 시트, `compare.md`를 보냅니다.

## 지키지 않으면 사고 나는 것
| 실수 | 대신 |
|---|---|
| 영상, 이미지, `metrics.json`을 저장소에 커밋 | 이 저장소는 **공개**입니다. 코드만 커밋합니다. |
| 키프레임 시트를 보지 않고 진행 | 자동 검출은 포즈 잡음에 흔들립니다. 반드시 눈으로 확인합니다. |
| 같은 파일을 새 회차로 기록 | `compare.py`의 MD5 검사(exit 3)를 따릅니다. |
| ±2cm, ±2° 안의 차이를 "좋아졌다"고 말함 | 측정 오차 범위입니다. "변화 없음"이라고 씁니다. |
| 모델프로 파생 프레임을 저장하거나 공유 | 타인 저작물입니다. 개인 분석용 결과 영상만 사용자에게 줍니다. |
