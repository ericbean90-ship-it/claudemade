# -*- coding: utf-8 -*-
"""1단계: PDF 폴더 → 텍스트 파일(work/f01.txt …)

주의: Google Drive의 PDF 읽기 기능은 파일당 약 20%를 잘라먹는다.
반드시 원본 PDF를 PyMuPDF로 직접 파싱할 것. (이 스크립트가 그렇게 한다)

사용법:  python3 extract_pdfs.py <PDF폴더> [작업폴더]
PDF 파일명은 "…-1.pdf" … "…-25.pdf" 처럼 끝에 분할번호가 있어야 한다.
"""
import sys, os, re, glob
import pymupdf

def main(pdf_dir, work="work"):
    os.makedirs(work, exist_ok=True)
    files = glob.glob(os.path.join(pdf_dir, "*.pdf"))
    if not files:
        sys.exit(f"PDF를 찾지 못했습니다: {pdf_dir}")
    got = {}
    for p in sorted(files):
        m = re.search(r'-(\d+)\.pdf$', os.path.basename(p))
        if not m:
            print(f"  [건너뜀] 분할번호 없음: {os.path.basename(p)}")
            continue
        idx = int(m.group(1))
        doc = pymupdf.open(p)
        txt = "\n".join(pg.get_text() for pg in doc)
        n = doc.page_count
        doc.close()
        out = os.path.join(work, f"f{idx:02d}.txt")
        open(out, "w", encoding="utf-8").write(txt)
        got[idx] = (n, len(txt))
        print(f"  {os.path.basename(p):<40} → f{idx:02d}.txt  {n}쪽  {len(txt):,}자")
    print(f"\n총 {len(got)}개 파일, {sum(v[0] for v in got.values())}쪽 추출")
    missing = [i for i in range(1, max(got)+1) if i not in got]
    if missing:
        print(f"⚠ 분할번호 누락: {missing}")
    return got

if __name__ == "__main__":
    if len(sys.argv) < 2:
        sys.exit(__doc__)
    main(sys.argv[1], sys.argv[2] if len(sys.argv) > 2 else "work")
