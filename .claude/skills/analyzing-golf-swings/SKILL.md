---
name: analyzing-golf-swings
description: Use when the user uploads golf swing videos (a face-on "front" clip and a down-the-line "back" clip of the same shot, e.g. rangex *_frontVideo.MP4 / *_backVideo.MP4), asks for swing coaching or problems, a pro overlay/ghost comparison, an overlay of the Modelgolf model-pro render, or asks whether their swing changed since last time (스윙 분석, 변화가 있나, 프로랑 겹쳐줘).
---

# 골프 스윙 영상 분석·코칭

## 개요
같은 샷을 찍은 정면 영상과 후면(DTL) 영상 두 개로 3D 지표를 측정합니다. 그 지표로 지난 회차와 비교하고, 코칭 리포트와 비교 영상을 만듭니다.

**원칙**
- 모든 수치는 `out/metrics.json` 한 곳에서 나옵니다.
- 회차 기록은 구글 드라이브의 `골프 스윙 기록` 폴더에만 둡니다. 회차마다 파일 하나씩 추가합니다.
- 답은 한국어로 씁니다. 큰 흐름을 한 문장으로 요약하며 시작하고, 중요한 순서대로 쓰고, 어려운 개념은 비유로 설명합니다.

## 준비
- **영상:** 같은 샷의 정면 영상과 후면 영상. 파일명 시각(`YYYYMMDD_HHMMSS`)이 같아야 합니다.
  - 파일명 시각은 UTC입니다. 영상 화면에 찍힌 시각은 +9시간(KST)입니다.
- **모델프로 영상 (선택):** Modelgolf 영상은 사용자가 올려야 합니다. YouTube 다운로드는 서버가 403으로 막으니 **우회하지 않습니다.**
- **변수:**
  - `SK` = 이 스킬 폴더, `T=$SK/scripts`
  - `W` = `<scratchpad>/swing-YYYY-MM-DD_HHMMSS` (파일명 시각)
- **환경:** 클라우드 환경 `claudemade`를 쓰면 의존성이 미리 설치돼 있습니다. 그래도 아래 1단계 `setup.sh`는 실행합니다.

## 작업 순서
1. **환경 확인:** `bash $T/setup.sh`
2. **지난 기록 가져오기 (구글 드라이브, 분석보다 먼저):**
   1. 폴더를 찾습니다: `search_files`, `title contains '골프 스윙 기록' and mimeType = 'application/vnd.google-apps.folder'`
      - 폴더가 없으면 첫 회차입니다. `create_file`로 폴더를 만듭니다 (`mimeType: application/vnd.google-apps.folder`).
   2. 기록 파일을 찾습니다: `search_files`, `parentId = '<폴더id>' and title contains 'swing-'`
      - `nextPageToken`이 오면 빈 결과가 나올 때까지 다음 페이지를 봅니다.
   3. 파일마다 `download_file_content`를 호출합니다. `read_file_content`는 빈 값을 돌려주니 쓰지 않습니다.
   4. 결과에 나온 저장 경로의 파일을 `$W/history/<드라이브 제목>`으로 복사합니다. 확장자는 상관없습니다.
3. **중복 검사:** `python3 $T/precheck.py <front> <back> $W`
   - **exit 3** = 이미 분석한 영상입니다. "같은 영상"이라고 알리고, 그때의 `notes`(과제)를 다시 안내합니다. 분석, 렌더, 저장은 하지 않습니다. **여기서 끝.**
   - **exit 2** = 정면과 후면의 촬영 시각이 다릅니다. 같은 샷인지 사용자에게 묻습니다.
4. **촬영 시각 확인:** `python3 $T/stamp.py $W <front> <back>`
   - `out/stamp.png`(영상 왼쪽 아래에 찍힌 날짜·시각을 확대한 것)를 열어 `stamp.json`의 `screen_time`과 같은지 확인합니다.
   - 다르면 `stamp.json`을 직접 고칩니다.
   - **모든 제작물(md, 사진, 영상) 파일명은 `stamp.json`의 `prefix`(예: `2026-11-11_1916`)로 시작합니다.** 파일명의 UTC 시각은 쓰지 않습니다.
   - 분석: `bash $T/run_pipeline.sh <front> <back> $W analyze` (약 3분)
5. **눈으로 확인 (필수, 둘 다):**
   - **공 위치:** `ball.py`가 경고를 냈다면(두 카메라의 공 위치가 3cm 넘게 어긋남), 카메라나 타석이 바뀐 것입니다.
     1. `python3 $T/ball_grid.py $W`를 실행합니다.
     2. `out/ball_front.jpg`와 `out/ball_dtl.jpg`에서 공 중심 픽셀을 읽습니다.
     3. `python3 $T/ball.py $W <fu> <fv> <bu> <bv>`를 실행합니다.
     4. `bash $T/run_pipeline.sh <front> <back> $W metrics`를 다시 실행합니다.
   - **키프레임:** `out/keyframes_sheet.jpg`를 엽니다. P4는 탑(클럽이 가장 뒤로 간 순간), P7은 임팩트(클럽이 공에 닿는 순간)여야 합니다.
     - 틀렸으면 `python3 $T/frames_strip.py $W <시작> <끝> 2`로 후보 프레임을 보고 `$W/keyframes.json`을 고칩니다.
     - 고친 뒤 `... $W metrics`를 다시 실행합니다.
6. **비교:** `python3 $T/compare.py $W` → `out/compare.md`
   - "카메라가 달라졌다"는 경고가 나오면 척추 앞숙임 같은 절대 각도는 비교에서 뺍니다.
7. **리포트 md 작성 (한국어, 매 회차 필수):**
   1. `python3 $T/report_md.py $W`를 실행합니다. 제목과 비교표가 들어간 `out/<prefix>_리포트.md` 뼈대가 생깁니다.
   2. 뼈대의 `<!-- 작성: ... -->` 자리를 모두 한국어로 채웁니다. 섹션은 다음 4개이고, 이 순서를 바꾸지 않습니다.
      - **스윙 분석 요약:** 큰 흐름 한 문장, 이번 근본 과제 하나, 비유
      - **지난 회차 대비 변화 (MM/DD → MM/DD):** 표 + 해석
      - **핵심 느낌:** 3가지
      - **드릴:** 3~4개 + 다음 회차 확인 기준
   3. 판정 기준, 근본 원인 고르는 법, 드릴 목록은 `reference/coaching.md`를 따릅니다. 키프레임 시트에서 눈으로 본 결과(테이크어웨이 경로, 샤프트 기울기)도 보탭니다.
   4. `grep -c '<!-- 작성' <md>`가 **0**이어야 다음 단계로 갑니다.
8. **영상 만들기:**
   - 프로 고스트: `... $W render` → `out/ghost_*.mp4`
   - 모델프로: 영상을 받았다면 `reference/model-pro.md`를 따릅니다.
   - 전후 비교 이미지: 지난 회차 `ghost_P4.jpg`, `ghost_P7.jpg`가 이 세션에 있을 때만 만듭니다.
9. **기록 저장:**
   - 위치: 2단계에서 찾은 폴더
   - 파일: `create_file`로 `swing-YYYY-MM-DD_HHMMSS.json`을 새로 만듭니다.
   - 옵션: `textContent` = `out/metrics.json`, `contentMimeType: text/plain`, `disableConversionToGoogleType: true`
   - 저장하기 전에 `notes`(이번 근본 과제와 드릴), `club`, 카메라 변경 메모를 채웁니다.
   - **제작물도 같은 폴더에 올립니다 (날짜 파일명):**
     1. `python3 $T/package.py $W`를 실행합니다. `upload/`에 `<prefix>_리포트.md`, `<prefix>_전후비교.jpg`, `<prefix>_프로고스트_합본.mp4`, `<prefix>_모델프로_*.mp4` 사본을 만들고, 파일마다 `file_ref`를 출력합니다.
     2. 파일마다 `create_file`을 호출합니다.
        - 옵션: `title` = 그 파일명, `parentId` = 폴더 id, `base64Content` = `file_ref`, `contentMimeType` = 출력된 mime, `disableConversionToGoogleType: true`
     3. 결과의 `fileSize`가 출력된 바이트 수와 같은지 확인합니다.
   - **크기 제한:** 커넥터 호출 한 번에 약 3MB까지만 보낼 수 있습니다. 3MB가 넘는 파일(영상 대부분)은 업로드가 거절됩니다.
     - 거절되면 파일 이름을 사용자에게 알리고 어떻게 할지 묻습니다. 임의로 줄이거나 나누어 보내지 않습니다.
     - 사용자가 처리 방식을 정하면 이 줄을 그 방식으로 바꿉니다.
10. **전달:**
    - `SendUserFile`로 리포트 md, 영상, 이미지를 보냅니다.
    - 채팅 답변도 리포트 md의 4개 섹션 그대로 **한국어로** 씁니다. 거기에 파일 목록과 한계를 덧붙입니다. 영어로 쓰지 않습니다.

## 지키지 않으면 사고 나는 것
| 실수 | 대신 |
|---|---|
| 영상, 이미지, `metrics.json`을 저장소에 커밋 | 이 저장소는 **공개**입니다. 코드만 커밋합니다. |
| 시트나 공 경고를 보지 않고 진행 | 카메라가 바뀌면 숫자 전체가 틀어집니다. 5단계는 생략하지 않습니다. |
| ±2cm, ±2° 안의 차이를 "좋아졌다"고 말함 | 측정 오차 범위입니다. "변화 없음"이라고 씁니다. |
| 카메라가 바뀐 회차의 절대 각도를 비교 | 어드레스 대비 변화량 지표만 비교합니다. |
| 모델프로 파생 프레임을 저장하거나 공유 | 타인 저작물입니다. 결과 영상은 사용자에게만 보냅니다. |
