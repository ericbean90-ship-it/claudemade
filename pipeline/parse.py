# -*- coding: utf-8 -*-
"""2단계: 텍스트 → 구조화 JSON

work/f01.txt(주계약+목차) 에서 특약 목차를 읽고,
work/f02.txt~f25.txt(본문)에서 특약별 구간과 조항을 잘라낸다.

산출물(work/):
  toc.txt        목차 특약 목록 (쪽\t특약명)
  body.txt       f02~f25 원문 연결본 (공백 보존)
  segs.json      특약별 본문 구간 [{page,name,opos,oend}]
  clauses.json   {특약명: {조항키: [제목, 본문]}}
  summary.json   특약별 메타 [{page,name,renew,grade3}]
  families.json  {계열: [특약명…]}
  terms.json     {용어: [{text,users}…]}  정의 조항 중복 제거본

사용법:  python3 parse.py [작업폴더]
"""
import sys, os, re, json
from collections import defaultdict

STRIP = "ㆍ・∙·"

def norm(s):
    """비교용 정규화: 공백·가운뎃점 제거, 이스케이프 해제"""
    s = s.replace("\\[", "[").replace("\\]", "]")
    for c in STRIP:
        s = s.replace(c, "")
    return re.sub(r'\s+', '', s)

# ---------------------------------------------------------------- 목차
def read_toc(work):
    """f01.txt 의 '상품 목차'에서 (쪽, 특약명) 목록을 읽는다.

    PDF 직접 추출본은 특약명과 쪽번호가 서로 다른 줄에 온다:
        (체증형)입원간병인사용(...)특약AC(무배당,무해약환급금형)
        63
    같은 줄에 오는 형태도 함께 허용한다.
    """
    t = open(os.path.join(work, "f01.txt"), encoding="utf-8").read()
    i = t.find("특약일반사항")
    if i < 0:
        raise SystemExit("f01.txt 에서 상품 목차를 찾지 못했습니다.")
    j = t.find("상황별 목차", i + 50)
    seg = t[i:j] if j > 0 else t[i:i+80000]

    lines = [l.strip() for l in seg.split("\n") if l.strip()]
    out, k = [], 0
    while k < len(lines):
        line = lines[k]
        m = re.match(r'^(.+특약.*?)\s+(\d{1,4})$', line)      # 같은 줄
        if m:
            name, page = m.group(1), int(m.group(2)); k += 1
        elif "특약" in line and k + 1 < len(lines) and re.fullmatch(r'\d{1,4}', lines[k+1]):
            name, page = line, int(lines[k+1]); k += 2          # 다음 줄
        else:
            k += 1; continue
        if "제1편일반사항" in norm(name):
            continue
        if norm(name) in ("특약", "제도성특약"):                 # 목차 소제목
            continue
        out.append((page, name))
    return out

# ---------------------------------------------------------------- 구간 분할
VARIANTS = {
 "입원간병인사용(181일이상,요양,정신,한방병원제외)특약A(무배당)":
   ["입원간병인사용(181일이상,요양/정신/한방병원제외)특약A(무배당)"],
 "입원간호간병통합서비스보장(181일이상,요양,정신,한방병원제외)특약A(무배당)":
   ["입원간호간병통합서비스보장(181일이상,요양/정신/한방병원제외)특약A(무배당)"],
}

def segment(work, toc):
    parts, i = [], 2
    while os.path.exists(os.path.join(work, f"f{i:02d}.txt")):
        parts.append(open(os.path.join(work, f"f{i:02d}.txt"), encoding="utf-8").read())
        i += 1
    body = "\n".join(parts)
    buf, idxmap = [], []
    for k, ch in enumerate(body):
        if ch.isspace() or ch in STRIP:
            continue
        buf.append(ch); idxmap.append(k)
    nbody = "".join(buf)

    segs, cur = [], 0
    for pg, name in toc:
        cands = [norm(name)] + [norm(v) for v in VARIANTS.get(name, [])]
        pos = None
        for key in cands:
            for m in re.finditer(re.escape(key), nbody):
                if m.start() < cur:
                    continue
                after = nbody[m.end():m.end()+400]
                # 목차/각주가 아닌 '본문 시작'인지 확인
                if ("제1편일반사항" in after or "약관제1편" in after
                        or "특약약관" in after or "제1조" in after[:250]):
                    pos = m.start(); break
            if pos is not None:
                break
        if pos is None:   # 완화된 재탐색
            for key in cands:
                m = re.search(re.escape(key), nbody[cur:])
                if m:
                    pos = cur + m.start(); break
        if pos is None:
            raise SystemExit(f"본문에서 특약을 찾지 못했습니다: {name}")
        segs.append({"page": pg, "name": name, "npos": pos, "opos": idxmap[pos]})
        cur = pos + 10
    for k, s in enumerate(segs):
        s["oend"] = segs[k+1]["opos"] if k+1 < len(segs) else len(body)
    return body, segs

# ---------------------------------------------------------------- 조항 분해
# 반드시 '줄머리'여야 한다. 본문 속 상호참조(제2-1조의8 [특약의 보장개시일])를
# 새 조항 제목으로 오인하면 조항 본문이 중간에서 잘린다.
HDR = re.compile(r'(?m)^[ \t]*제\s*2\s*-\s*(\d+)\s*조(?:\s*의\s*(\d+))?\s*\[([^\]]{2,40})\]')

def split_clauses(body, segs):
    data = {}
    for s in segs:
        t = body[s["opos"]:s["oend"]]
        ms = list(HDR.finditer(t))
        d = {}
        for i, m in enumerate(ms):
            no, sub, ti = m.groups()
            key = f"제2-{no}조" + (f"의{sub}" if sub else "")
            end = ms[i+1].start() if i+1 < len(ms) else len(t)
            if key not in d:
                d[key] = [re.sub(r'\s+', ' ', ti).strip(), t[m.end():end]]
        data[s["name"]] = d
    return data

# ---------------------------------------------------------------- 메타
def summarize(body, segs):
    out = []
    for s in segs:
        t = body[s["opos"]:s["oend"]]
        out.append({"page": s["page"], "name": s["name"],
                    "renew": "갱신형" in s["name"],
                    "grade3": "간편고지형" in t})
    return out

def family(n):
    if any(k in n for k in ["사후정리", "선지급서비스", "이륜자동차", "장애인전용",
                            "지정대리청구", "치료비선지급", "치료비 선지급",
                            "특정신체부위", "무사고"]):
        return "0.제도성특약"
    if n.startswith("70+"): return "1.70+ 전용"
    if n.startswith("80+"): return "2.80+ 전용"
    if any(k in n for k in ["간병인", "간호간병"]): return "3.입원간병"
    if "입원" in n and "암" not in n and "질환" not in n: return "4.입원"
    if "수술" in n: return "5.수술"
    if any(k in n for k in ["항암", "표적", "카티", "면역항암", "방사선"]): return "6.항암치료"
    if "암" in n: return "7.암(진단·치료)"
    if any(k in n for k in ["뇌", "심장", "심근", "순환계", "혈전", "부정맥",
                            "허혈", "2대질병", "3대질병"]): return "8.뇌·심혈관"
    return "9.기타 질병·재해"

# ---------------------------------------------------------------- 용어사전
def canon(t):
    """정의 비교용: 별표번호·조항번호 차이를 무시"""
    t = re.sub(r'별표\s*\d+', '별표N', t)
    t = re.sub(r'제2-\d+조(?:의\d+)?', '제2-N조', t)
    t = re.sub(r'[“”"\'’‘]', '"', t)
    t = re.sub(r'[ㆍ·・∙]', '·', t)
    return re.sub(r'\s+', '', t)

def build_terms(segs, clauses_mod):
    terms = defaultdict(dict)
    for s in segs:
        for _k, ti, tx in clauses_mod.defs(s["name"]):
            tx2 = re.sub(r'\s+', ' ', tx).strip()
            if len(tx2) < 40:
                continue
            q = re.findall(r'[“"]([^”"]{1,25})[”"]', ti)
            key = " / ".join(q) if q else ti
            c = canon(tx2); d = terms[key]
            if c not in d:
                d[c] = {"text": tx2, "users": []}
            d[c]["users"].append(s["name"])
            if len(tx2) > len(d[c]["text"]):
                d[c]["text"] = tx2
    return {k: [{"text": x["text"], "users": x["users"]} for x in d.values()]
            for k, d in terms.items()}

# ---------------------------------------------------------------- main
def main(work="work"):
    toc = read_toc(work)
    print(f"목차 특약: {len(toc)}개")
    body, segs = segment(work, toc)
    print(f"본문 분할: {len(segs)}개 (원문 {len(body):,}자)")
    clauses = split_clauses(body, segs)
    ncl = sum(len(v) for v in clauses.values())
    print(f"조항 분해: {ncl}개 (특약당 평균 {ncl//len(segs)}개)")

    J = lambda n, o: json.dump(o, open(os.path.join(work, n), "w", encoding="utf-8"),
                               ensure_ascii=False)
    open(os.path.join(work, "toc.txt"), "w", encoding="utf-8").write(
        "\n".join(f"{p}\t{n}" for p, n in toc))
    open(os.path.join(work, "body.txt"), "w", encoding="utf-8").write(body)
    J("segs.json", segs)
    J("clauses.json", clauses)
    summ = summarize(body, segs); J("summary.json", summ)
    fam = defaultdict(list)
    for s in segs:
        fam[family(s["name"])].append(s["name"])
    J("families.json", dict(fam))
    print("계열 분류:", ", ".join(f"{k.split('.')[1]} {len(v)}" for k, v in sorted(fam.items())))

    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    import clause_text
    clause_text.load(work)
    terms = build_terms(segs, clause_text)
    J("terms.json", terms)
    print(f"용어사전: 용어 {len(terms)}개 / 변형 {sum(len(v) for v in terms.values())}개")
    print(f"\n완료 → {work}/")

if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "work")
