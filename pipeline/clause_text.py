# -*- coding: utf-8 -*-
"""조항 본문 정제 유틸 (parse.py / generate.py 공용)

PDF에서 뽑은 조항 본문에는 아래 잡음이 섞여 있어 정제가 필요하다.
  - 쪽번호가 문장 사이에 단독 줄로 끼어듦
  - 단어가 줄바꿈으로 쪼개짐 (한글은 하이픈이 없어 그냥 붙음)
  - 도해(면책기간 그림)의 텍스트 파편이 문장 중간에 섞임
  - '설명 <…예시>' 박스가 문장 중간에 통째로 끼어듦
"""
import re, json, os

C = {}    # {특약명: {조항키: [제목, 본문]}}
SEG = []  # [{page,name,opos,oend}]
BODY = "" # 원문 연결본

def load(work="work"):
    global C, SEG, BODY
    C = json.load(open(os.path.join(work, "clauses.json"), encoding="utf-8"))
    SEG = json.load(open(os.path.join(work, "segs.json"), encoding="utf-8"))
    BODY = open(os.path.join(work, "body.txt"), encoding="utf-8").read()

# ---------------------------------------------------------------- 정제
PAGENUM = re.compile(r'(?m)^\s*\d{2,4}\s*$')
DIAG = [
    re.compile(r'(?:암|독감\(인플루엔자\)|전이암|유사암|치매)?보장개시일\s*\d+일\s*\([^)]*(?:제외기간|보장)[^)]*\)[^\n]{0,60}'),
    re.compile(r'\d+월\s*\d+일\s*\d+월\s*\d+일[^\n]{0,60}'),
    re.compile(r'계약일\[부활\(효력회복\)일\]'),
    re.compile(r'(?:○|●)\s*[^\n○●]{2,30}\(\d{4}\.\s*\d+\.\s*\d+\)'),
]
EXPL = re.compile(r'설명\s*<[^>]{0,80}>.*?(?=(?:[①-⑮])|(?:\s[가-하]\.)|$)', re.S)

def strip_diagram(t):
    for r in DIAG:
        t = r.sub(" ", t)
    t = EXPL.sub(" ", t)
    return re.sub(r'[ \t]{2,}', ' ', t)

def clean(t):
    if not t:
        return ""
    t = t.replace("­", "")
    t = PAGENUM.sub("", t)
    t = re.sub(r'LI\d{6,}\(\d+\)', '', t)
    # '지급사유'/'지급금액' 표 머리는 줄로 보존
    t = re.sub(r'(?m)^\s*(지급사유|지급금액)\s*$', r'\n@@\1@@\n', t)
    # 줄바꿈으로 쪼개진 단어 붙이기 — 단, 다음 줄이 항목 머리면 붙이지 않는다
    t = re.sub(r'([가-힣A-Za-z0-9%\)\]”"])[ \t]*\n[ \t]*'
               r'(?![\d]+\.|[①-⑮]|[가-하]\.|[-•◇◆※*]|<)([가-힣A-Za-z0-9(\[“"])', r'\1\2', t)
    t = t.replace("@@지급사유@@", "\n**지급사유** ").replace("@@지급금액@@", "\n**지급금액**\n")
    t = strip_diagram(t)
    t = re.sub(r'[ \t]{2,}', ' ', t)
    t = re.sub(r'\n{2,}', '\n', t)
    return t.strip()

def get(name, key):
    return C.get(name, {}).get(key, ["", ""])[1]

def title(name, key):
    return C.get(name, {}).get(key, ["", ""])[0]

# ---------------------------------------------------------------- 제2-2조
def benefit_block(name):
    """지급사유·지급금액 (약관 원문, 요약 아님)"""
    t = clean(get(name, "제2-2조"))
    t = re.sub(r'②\s*제1항의\s*[“"]보험연도[”"].*?$', '', t, flags=re.S)
    t = re.sub(r'\n?\s*설명\s*\n.*?(?=(?:\n[①-⑮])|$)', '\n', t, flags=re.S)
    t = re.sub(r'<보험연도 계산 예시>.*?$', '', t, flags=re.S)
    return re.sub(r'\n{2,}', '\n', t).strip()

# ---------------------------------------------------------------- 제2-3조
KEEP = re.compile(r'(한도|1회 입원당|1회당|1회에 한하여|보장개시|면책|제외기간|90일|중복|'
                  r'추가로 지급하지|지급하지 않|보장하지 않|지급되지 않|해당하지 않|만 지급|'
                  r'감액|지정|취소|연간|보험연도|간주|합산|소멸|체증|초과하여|최고|한합니다|'
                  r'봅니다|제외합니다|보지 않)')
DROP = re.compile(r'(적립 이율|유첨|이자는|지급기일의 다음|청구하여야 합니다|서류를 제출|알려 드립니다)')

def cut_explain(t):
    t = re.split(r'\n?\s*설명\s*\n', t)[0]
    t = re.split(r'<\s*제2-\d', t)[0]
    return re.split(r'<보험연도 계산 예시>', t)[0].strip()

def caveats(name, limit=8):
    """지급을 제한·조건짓는 조항만 (절차 조항 제외)"""
    t = clean(get(name, "제2-3조"))
    out = []
    for p in re.split(r'(?=[①②③④⑤⑥⑦⑧⑨⑩⑪⑫⑬⑭⑮])', t):
        p = cut_explain(re.sub(r'\s+', ' ', p).strip())
        if len(p) < 25 or DROP.search(p) or not KEEP.search(p):
            continue
        out.append(p if len(p) <= 420 else p[:420] + " …")
    return out[:limit]

# ---------------------------------------------------------------- 정의
DEFKEYS = ["제2-1조"] + [f"제2-1조의{i}" for i in range(2, 9)]

def defs(name):
    out = []
    for k in DEFKEYS:
        if k not in C.get(name, {}):
            continue
        ti, tx = title(name, k), clean(get(name, k))
        if not tx or tx.startswith("이 특약은 해당사항"):
            continue
        out.append((k, ti, strip_diagram(tx).strip()))
    return out

# ---------------------------------------------------------------- 별표 / 소멸
def annexes(name):
    from collections import Counter
    s = next(x for x in SEG if x["name"] == name)
    seg = BODY[s["opos"]:s["oend"]]
    items = ["별표" + (n or "") + " [" + re.sub(r'\s+', ' ', t).strip() + "]"
             for n, t in re.findall(r'별표\s*(\d+)?\s*\[([^\]]{2,45})\]', seg)]
    return [k for k, _ in Counter(items).most_common(12)]

UNIVERSAL = re.compile(r'(주계약이 해지|피보험자가.{0,12}사망한 경우|보험기간이 끝난|보험기간 만료)')

def extinction(name):
    """공통 3가지를 뺀, 이 특약 고유의 소멸사유"""
    t = clean(get(name, "제2-12조"))
    m = re.search(r'①(.*?)(?=②|$)', t, re.S)
    if not m:
        return []
    seg = re.sub(r'\s+', ' ', m.group(1))
    out = []
    for it in re.split(r'(?<![\d.])(?=\d\.\s)', seg)[1:]:
        it = it.strip()
        if len(it) < 12 or UNIVERSAL.search(it[:40]):
            continue
        it = re.sub(r'^\d\.\s*', '', it)
        out.append(it if len(it) <= 260 else it[:260] + " …")
    return out[:3]
