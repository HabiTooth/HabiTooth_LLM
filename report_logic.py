# report_logic.py
"""
HabiTooth LLM 리포트 로직.

배포 모델: Qwen/Qwen3.5-9B  (Apache 2.0)
  - 초기에는 Qwen2.5-3B(로컬 개발) → EEVE-Korean-Instruct-10.8B(배포 후보) 순으로 검토했고,
    동일 프롬프트·동일 검증 기준으로 비교한 결과 Qwen3.5-9B 를 채택했다.
    응답 8~9초(EEVE 75~155초), 지시 준수율·문장 다양성 모두 우위.
  - 이 파일은 torch / fastapi / pydantic 의존성이 전혀 없다.
    → test.py 가 importlib.reload() 로 다시 읽을 수 있고,
      프롬프트·카탈로그·집계 규칙만 고칠 때는 모델을 VRAM 에서 내렸다 올릴 필요가 없다.
    → 모델을 갈아타도 이 파일은 그대로 쓴다. 모델 비교 실험의 통제 변수이기도 하다.

[역할 분담 — 환각이 사용자에게 도달하는 경로를 차단하는 것이 설계 목표]
- 사실 (부위·병변·위험도·재측정 주기·이전 측정 대비 변화) : 100% 코드가 결정. LLM 관여 없음.
- 의학적 권고 (스케일링, 치실 사용법 등)                  : MANAGEMENT_CATALOG 의 고정 텍스트.
- 표현 (그 부위가 왜 어렵고 어떻게 닦을지, 그룹당 2문장)   : LLM 생성.
  → polish() 후처리 + assemble_report() 의 검증 15종 + 최대 4회 재생성으로 감싼다.
- notice / disclaimer                                     : 완전 고정.

[출력 구조]
- riskDetail  : 관찰 문장(코드) + 변화 문장(코드) + 관리 문장 2개(LLM) + 재측정/호전(코드)
- management  : 전부 코드. 어떤 항목을 고를지는 측정 결과에 따라 달라지고,
                문장 자체는 검증된 고정 텍스트를 쓴다. title 은 앱·Spring 과 계약된 값이므로
                절대 변경하지 않는다.
- notice      : 촬영이 유효하지 않은 구역이 있을 때만.
- disclaimer  : 고정.

[LOGIC_VERSION]
프롬프트나 규칙을 고칠 때마다 올린다. POST /dev/reload 응답에 그대로 나오므로
콜랩·서버에 올린 파일이 실제로 반영됐는지 즉시 확인할 수 있다.
"""

import itertools
import re
from collections import Counter

# ══════════════════════════════════════════════════════════════
# 상수
# ══════════════════════════════════════════════════════════════
# 이 값을 고칠 때마다 올린다. /dev/reload 와 /health 응답에 그대로 나와서
# 콜랩에 올린 파일이 실제로 반영됐는지 바로 확인할 수 있다.
LOGIC_VERSION = "2026-09-17-phrase-stats"

DISCLAIMER = ("HabiTooth의 분석 결과는 구강 상태 파악을 돕는 보조 진단 및 참고 목적으로만 제공되며, "
              "전문적인 의료 자문, 확진 및 치료를 대신하지 않습니다. "
              "염려되는 증상이 있으면 반드시 치과 전문의와 상담하세요.")

LESION_KR = {"CALCULUS": "치석", "PLAQUE": "치태"}
CHANGE_KR = {"NEW": " (새로 발견)", "PERSISTED": " (계속 관찰됨)", None: ""}
RISK_ORDER = {"CRITICAL": 0, "HIGH": 1, "MEDIUM": 2, "LOW": 3, "VERY_LOW": 4}

VIEW_KR = {
    "UPPER_RIGHT": "위턱 오른쪽", "UPPER_CENTER": "위턱 앞쪽", "UPPER_LEFT": "위턱 왼쪽",
    "LOWER_RIGHT": "아래턱 오른쪽", "LOWER_CENTER": "아래턱 앞쪽", "LOWER_LEFT": "아래턱 왼쪽",
    "OUTER_RIGHT": "바깥쪽 오른쪽", "OUTER_CENTER": "바깥쪽 앞니", "OUTER_LEFT": "바깥쪽 왼쪽",
}

# zoneScore 는 "높을수록 깨끗함" 이라고 가정했다. 반대라면 부등호를 뒤집을 것.
WEAK_ZONE_THRESHOLD = 70.0

# areaRatio → 정성 표현. 실제 분포 보고 조정할 것.
COVERAGE_BANDS = [(0.30, "넓게 퍼져 있어요"), (0.10, "여러 곳에서 관찰돼요"), (0.0, "일부에서 관찰돼요")]

MIN_DETAIL_LEN = 30
MAX_DETAIL_LEN = 220          # 문장이 늘어지는 것을 막는 상한
SENTENCES_PER_DETAIL = 2

# 부위 이름을 이 개수보다 많이 나열하게 되면 묶어서 요약한다.
MAX_REGIONS_LISTED = 3

# 로컬 개발 시 병변 수를 잘라 생성 시간을 줄이던 제한. 배포에서는 반드시 None.
DEV_LIMIT = None

# ══════════════════════════════════════════════════════════════
# 문장 후처리
# 모델 출력에서 반복적으로 관찰된 문제를 강제 교정한다.
# 규칙은 전부 실측 기반이다. 추측으로 넣지 말고, 실제로 나온 출력을 보고 추가할 것.
# detail 문장에만 적용된다. management 는 고정 텍스트라 교정이 필요 없다.
# 순서 주의: 더 좁은 패턴이 위, 넓은 패턴이 아래.
# ══════════════════════════════════════════════════════════════
def _has_final(syl: str) -> bool:
    """마지막 글자에 받침이 있는지."""
    code = ord(syl) - 0xAC00
    return 0 <= code <= 11171 and code % 28 != 0


def _is_yang(syl: str) -> bool:
    """중성이 양성모음(ㅏ, ㅗ)인지 → '아요' 계열."""
    code = ord(syl) - 0xAC00
    if not 0 <= code <= 11171:
        return False
    return (code % 588) // 28 in (0, 8)


def _hapsyo(m):
    """'~습니다' → '~아요/~어요' (모음조화)"""
    stem = m.group(1)
    return stem + ("아요" if _is_yang(stem) else "어요")


def _copula(m):
    """'~입니다' → '~이에요/~예요' (받침 유무)"""
    noun = m.group(1)
    return noun + ("이에요" if _has_final(noun) else "예요")


PHRASE_FIXES = [
    # ── 오타 ────────────────────────────────────────────────
    (r"닻아", "닦아"),
    (r"닥아", "닦아"),
    (r"되요", "돼요"),
    (r"돕세요", "도와주세요"),
    (r"쓸어올리", "쓸어 올리"),
    (r"닦아올리", "닦아 올리"),

    # ── 어휘 통일 ───────────────────────────────────────────
    (r"플라[크그]", "치태"),
    (r"치태석", "치석"),
    (r"치석막", "치태"),
    (r"바이오필름", "치태"),
    (r"씹는면", "씹는 면"),
    (r"치아면", "치아 면"),
    (r"치아와치아", "치아와 치아"),

    # ── 어색한 관용 표현 ───────────────────────────────────
    (r"치석(이|가)\s*잘\s*생겼(어요|습니다)", "치석이 쌓이기 쉬운 곳이에요"),
    (r"치태(이|가)\s*잘\s*생겼(어요|습니다)", "치태가 쌓이기 쉬운 곳이에요"),
    (r"잘\s*생겼(어요|습니다)", "잘 생기는 곳이에요"),
    (r"잘\s*(쌓이|생기)기\s*쉬운", r"\1기 쉬운"),
    (r"쉽게\s*(쌓이|생기)기\s*쉬운", r"\1기 쉬운"),
    (r"보임\s*(이\s*)?(돼요|됩니다|이에요)", "관찰돼요"),

    # ── 칫솔 동작의 목적어를 부위로 잘못 쓰는 패턴 ────────────
    (r"칫솔\s*모양을\s*잡고", "칫솔모 끝을 기울여 대고"),
    (r"칫솔\s*모양을", "칫솔모를"),
    (r"칫솔\s*모양(이|은)?", "칫솔모"),
    (r"칫솔\s+모(를|가|는|에|로)", r"칫솔모\1"),
    (r"칫솔\s+모\b", "칫솔모"),
    (r"(이\s*)?부위를(\s*세로로)", r"칫솔을\2"),
    (r"(이\s*)?부위를\s*(기울여|비스듬히)", r"칫솔을 \2"),
    (r"([가-힣]{2,4}\s?(?:뒷면|안쪽|바깥쪽|부위|면))을?\s*세로로\s*세(워|우)(서|고|며|세요)?",
     r"칫솔을 세로로 세워 \1까지"),

    # ── 명사 뒤에 "때"를 바로 붙이는 오류 ────────────────────
    (r"(양치|칫솔질|식사|사용|취침)\s*때(는|에|엔)?", r"\1할 때\2"),

    # ── 실측으로 잡힌 비문 (모델이 서두를 때 '못하'가 빠진다) ──
    # 뒤에 오는 말에 의존하지 않는다. 이 어형 자체가 비문이다.
    # 순서 주의: '~는 데' 형태를 먼저 처리해야 아래 일반 규칙에 먹히지 않는다.
    #
    # "들어가지 쉬운" 은 특히 위험하다. '못하'가 빠지면서 의미가 정반대가 된다.
    # (칫솔모가 안 들어간다 → 잘 들어간다)
    (r"들어가지\s*쉬운", "들어가지 못해 관리가 어려운"),
    (r"들어가지\s*어려운", "들어가지 못하는"),
    (r"닿지\s*쉬운", "닿지 않아 관리가 어려운"),
    (r"들어가지\s*는\s*데", "들어가지 못해"),
    (r"닿지\s*는\s*데", "닿지 않아"),
    (r"닦이지\s*는\s*데", "닦이지 않아"),
    (r"들어가지는(?=\s|$|[,.])", "들어가지 못하는"),
    (r"닿지는(?=\s|$|[,.])", "닿지 않는"),
    (r"닦이지는(?=\s|$|[,.])", "닦이지 않는"),

    # ── '~네요' / '~죠' 체 (감탄·확인조라 리포트 문체에 맞지 않는다) ──
    # 일반 규칙(r"([가-힣])죠" → r"\1예요")은 활용을 망가뜨린다.
    # ("힘들죠" → "힘들예요") 그래서 어형별로 적는다.
    # 여기 없는 '죠' 는 _FORMAL_LEAK 가 잡아 재시도시킨다.
    (r"곳이죠", "곳이에요"),
    (r"자리죠", "자리예요"),
    (r"부위죠", "부위예요"),
    (r"틈이죠", "틈이에요"),
    (r"힘들죠", "힘들어요"),
    (r"어렵죠", "어려워요"),
    (r"쉽죠", "쉬워요"),
    (r"있죠", "있어요"),
    (r"없죠", "없어요"),
    (r"되죠", "돼요"),
    (r"하죠", "해요"),
    (r"이네요", "이에요"),
    (r"하네요", "해요"),
    (r"있네요", "있어요"),
    (r"없네요", "없어요"),
    (r"좋네요", "좋아요"),
    (r"쉽네요", "쉬워요"),
    (r"어렵네요", "어려워요"),
    (r"힘드네요", "힘들어요"),

    # ── 합쇼체 → 해요체 통일 ───────────────────────────────
    # (management 는 고정 텍스트라 영향 없음. detail 에만 적용)
    # 순서 주의: 불규칙 활용을 먼저 처리하고, 규칙 활용은 아래 함수 치환에 맡긴다.
    (r"하십시오", "하세요"),
    (r"주십시오", "주세요"),
    (r"십시오", "세요"),
    # ㅂ불규칙 및 ㄹ탈락
    (r"쉽니다", "쉬워요"),
    (r"쉽습니다", "쉬워요"),
    (r"어렵습니다", "어려워요"),
    (r"가깝습니다", "가까워요"),
    (r"무겁습니다", "무거워요"),
    (r"가볍습니다", "가벼워요"),
    (r"힘듭니다", "힘들어요"),
    (r"다릅니다", "달라요"),
    (r"빠릅니다", "빨라요"),
    (r"느립니다", "느려요"),
    (r"깁니다", "길어요"),
    (r"아픕니다", "아파요"),
    (r"나쁩니다", "나빠요"),
    (r"큽니다", "커요"),
    # '입니다'가 서술격 조사가 아닌 동사 어미인 경우 (일괄 규칙보다 먼저)
    (r"쌓입니다", "쌓여요"),
    (r"보입니다", "보여요"),
    (r"쓰입니다", "쓰여요"),
    (r"모입니다", "모여요"),
    (r"놓입니다", "놓여요"),
    # 'ㅂ니다' 어미 동사
    (r"됩니다", "돼요"),
    (r"안 됩니다", "안 돼요"),
    (r"납니다", "나요"),
    (r"갑니다", "가요"),
    (r"옵니다", "와요"),
    (r"줍니다", "줘요"),
    (r"봅니다", "봐요"),
    (r"만듭니다", "만들어요"),
    (r"생깁니다", "생겨요"),
    (r"집니다", "져요"),
    (r"못합니다", "못해요"),
    (r"합니다", "해요"),
    (r"있읍니다", "있어요"),
    # 나머지 '~습니다' / '~입니다' 는 모음조화·받침을 보고 치환
    (r"([가-힣])습니다", _hapsyo),
    (r"([가-힣])입니다", _copula),

    # ── 영문/기호 혼입 ─────────────────────────────────────
    (r"\bplaque\b", "치태"),
    (r"\bcalculus\b", "치석"),
    (r"\btartar\b", "치석"),
    (r"[ \t]*[-–—]{2,}[ \t]*", " "),
    (r"^\s*json\s+", ""),
]


_MD_CHARS = re.compile(r"[*_`#>]+")
_LEAD_BULLET = re.compile(r"^\s*(?:[-•*·]|\d+\s*[.)])\s*")
_ROLE_LEAK = re.compile(r"(Human|Assistant|사용자|어시스턴트)\s*:")


# ══════════════════════════════════════════════════════════════
# PHRASE_FIXES 발동 통계
#
# 규칙 상당수가 EEVE 시절에 만들어졌고, 모델을 바꾼 뒤로는 한 번도 발동하지
# 않는 것이 섞여 있을 수 있다. 안 쓰이는 규칙은 무해하지 않다 — 규칙이 많을수록
# 서로 충돌해 멀쩡한 문장을 망가뜨릴 확률이 올라간다.
# (실제로 r"([가-힣])죠" → r"\1예요" 가 "힘들죠"를 "힘들예요"로 만든 적이 있다.)
#
# 추측으로 지우면 필요한 규칙을 지운다. 그래서 먼저 센다.
# GET /dev/phrase-stats 로 조회하고, 충분히 돌린 뒤 0회인 규칙만 정리한다.
#
# 주의: POST /dev/reload 는 모듈을 다시 읽으므로 카운터가 초기화된다.
# ══════════════════════════════════════════════════════════════
PHRASE_FIX_HITS = Counter()
POLISH_CALLS = 0


def phrase_fix_stats() -> dict:
    """발동 횟수 / 미발동 규칙 목록."""
    used = {pat: c for pat, c in PHRASE_FIX_HITS.items() if c}
    unused = [pat for pat, _ in PHRASE_FIXES if pat not in used]
    return {
        "logic_version": LOGIC_VERSION,
        "polish_calls": POLISH_CALLS,
        "rule_count": len(PHRASE_FIXES),
        "used_count": len(used),
        "unused_count": len(unused),
        "hits": dict(sorted(used.items(), key=lambda kv: -kv[1])),
        "unused": unused,
    }


def reset_phrase_stats():
    global POLISH_CALLS
    PHRASE_FIX_HITS.clear()
    POLISH_CALLS = 0


def polish(text) -> str:
    """LLM이 만든 문장의 오타·마크다운·문체를 교정한다."""
    if not isinstance(text, str):
        return ""
    # 역할 토큰 이후는 통째로 버린다 (모델이 대화를 이어 쓰는 경우)
    text = _ROLE_LEAK.split(text)[0]
    text = _MD_CHARS.sub("", text)
    text = _LEAD_BULLET.sub("", text)
    text = text.strip().strip('"').strip("'").strip("「」“”‘’").strip()
    global POLISH_CALLS
    POLISH_CALLS += 1
    for pat, rep in PHRASE_FIXES:
        text, n = re.subn(pat, rep, text)
        if n:
            PHRASE_FIX_HITS[pat] += n
    return re.sub(r"\s+", " ", text).strip()


_SENT_SPLIT = re.compile(r"(?<=[.!?])\s+")


def split_sentences(text: str):
    return [s.strip() for s in _SENT_SPLIT.split(text) if s.strip()]


def trim_sentences(text: str, n: int = SENTENCES_PER_DETAIL) -> str:
    """3문장 이상 나오면 앞의 n문장만 남긴다. 재시도보다 이쪽이 싸다."""
    sents = split_sentences(text)
    if len(sents) <= n:
        return text
    return " ".join(sents[:n])


# ══════════════════════════════════════════════════════════════
# 관리법 카탈로그 — 항목 선택도 문장도 코드가 담당한다. LLM 관여 없음.
# title 은 앱과 Spring 이 참조하는 계약된 값이다. 절대 변경 금지.
#
#   title  : 앱에 표시되는 제목
#   region : 문장에 끼울 부위를 어디서 고를지
#            "calculus" 치석 부위 / "molar" 어금니 부위 /
#            "anterior" 앞니 부위 / "top" 최우선 부위 / None 부위 없음
#   text   : {region} 자리표시자가 있는 본문
#   plain  : 해당 부위가 없을 때 쓰는 대체 본문 (region 이 None 이면 불필요)
# ══════════════════════════════════════════════════════════════
MANAGEMENT_CATALOG = {
    "SCALING": {
        "title": "치과 스케일링",
        "region": "calculus",
        "text": "{region}에서 발견된 치석은 칫솔질로는 떨어지지 않아요. 치석은 이미 단단하게 굳어 치아에 붙어 있어서, 치과에서 초음파 스케일러로 긁어내야 제거됩니다. 6개월에 한 번 정기적으로 받으면 잇몸 염증까지 함께 예방할 수 있어요.",
        "plain": "치석은 칫솔질로는 떨어지지 않아요. 이미 단단하게 굳어 치아에 붙어 있어서, 치과에서 초음파 스케일러로 긁어내야 제거됩니다. 6개월에 한 번 정기적으로 받으면 잇몸 염증까지 함께 예방할 수 있어요.",
    },
    "FOCUSED_BRUSHING": {
        "title": "집중 부위 칫솔질",
        "region": "top",
        "text": "양치를 시작할 때 {region}부터 닦아보세요. 보통 앞니부터 시작해 안쪽으로 갈수록 대충 닦고 끝내기 쉬운데, 순서를 바꾸면 가장 신경 써야 할 곳을 가장 꼼꼼하게 닦게 돼요. 거울을 보면서 그 부위에만 10초를 더 쓰는 것으로도 차이가 납니다.",
        "plain": "양치할 때 늘 마지막에 닦는 자리를 먼저 닦아보세요. 순서를 바꾸면 평소 대충 지나가던 곳을 가장 꼼꼼하게 닦게 돼요. 거울을 보면서 신경 쓰이는 부위에 10초를 더 쓰는 것으로도 차이가 납니다.",
    },
    "BRUSHING_METHOD": {
        "title": "올바른 칫솔질 방법",
        "region": None,
        "text": "칫솔모를 치아와 잇몸이 만나는 경계에 45도로 비스듬히 대고, 그 자리에서 짧고 가볍게 진동시키듯 움직여 주세요. 옆으로 세게 문지르면 치아가 패이고 잇몸이 내려앉을 수 있어요. 입안 전체를 도는 데 2분은 쓰는 것이 좋습니다.",
    },
    "FLOSS": {
        "title": "치실 사용",
        "region": "anterior",
        "text": "{region}처럼 치아 사이가 좁은 곳은 칫솔모가 들어가지 못해 치실이 필요해요. 치실을 톱질하듯 좌우로 움직여 천천히 내린 뒤, 한쪽 치아 옆면에 C자로 감아 위아래로 훑어주세요. 반대쪽 면도 같은 방법으로 한 번 더 닦아야 합니다.",
        "plain": "치아와 치아가 맞닿는 면은 칫솔모가 들어가지 못해 치실이 필요해요. 치실을 톱질하듯 좌우로 움직여 천천히 내린 뒤, 한쪽 치아 옆면에 C자로 감아 위아래로 훑어주세요. 반대쪽 면도 같은 방법으로 한 번 더 닦아야 합니다.",
    },
    "INTERDENTAL_BRUSH": {
        "title": "치간칫솔 사용",
        "region": "molar",
        "text": "치간칫솔은 {region}처럼 치아와 치아 사이에 공간이 있는 곳에 넣어 앞뒤로 움직이는 도구예요. 치아 바깥면이나 안쪽 면을 문지르는 용도가 아닙니다. 억지로 밀어 넣으면 잇몸이 상하니, 저항 없이 들어가는 가장 가는 굵기부터 시작하세요.",
        "plain": "치간칫솔은 치아와 치아 사이에 공간이 있는 곳에 넣어 앞뒤로 움직이는 도구예요. 치아 바깥면이나 안쪽 면을 문지르는 용도가 아닙니다. 억지로 밀어 넣으면 잇몸이 상하니, 저항 없이 들어가는 가장 가는 굵기부터 시작하세요.",
    },
    "MOUTHWASH": {
        "title": "구강세정제 사용",
        "region": None,
        "text": "구강세정제는 칫솔질을 대신하지 못하고 어디까지나 보조 수단이에요. 치태는 물리적으로 문질러야 떨어지기 때문입니다. 양치 직후보다는 외출 중이나 간식 뒤처럼 칫솔을 쓰기 어려울 때 사용하는 것이 잘 맞아요.",
    },
    "TONGUE_CLEANING": {
        "title": "혀 세정",
        "region": None,
        "text": "혀 표면의 설태에는 세균이 많이 모여 있어서, 그대로 두면 치아에 치태가 다시 쌓이는 속도가 빨라져요. 하루 한 번 양치 마지막에 혀 안쪽에서 바깥쪽으로 부드럽게 쓸어내 주세요. 세게 긁으면 혀 점막이 헐 수 있으니 힘은 빼야 합니다.",
    },
    "NIGHT_BRUSHING": {
        "title": "취침 전 칫솔질",
        "region": None,
        "text": "잠든 동안에는 침 분비가 크게 줄어 입안이 세균에 가장 취약해져요. 하루 중 자기 전 양치가 제일 중요한 이유입니다. 양치를 마친 뒤에는 물 외에 아무것도 먹지 않고 바로 자는 습관을 들여보세요.",
    },
    "DIET": {
        "title": "균형 잡힌 식단과 간식 조절",
        "region": None,
        "text": "당분이 많은 가공식품이나 캐러멜처럼 치아에 들러붙는 간식은 치태가 자라는 먹이가 돼요. 채소와 과일처럼 섬유질이 있는 음식은 씹는 동안 치아 표면을 자연스럽게 닦아줍니다. 간식 뒤 바로 양치가 어렵다면 물로 입안을 헹궈만 줘도 도움이 돼요.",
    },
    "IMMUNITY": {
        "title": "구강 면역력 관리",
        "region": None,
        "text": "잇몸 질환은 세균 자체보다 그에 맞서는 몸의 염증 반응으로 조직이 상하는 병이라, 같은 상태여도 면역력에 따라 정도가 달라져요. 충분한 수면과 균형 잡힌 식사, 스트레스 관리가 그대로 잇몸 상태로 이어집니다. 흡연은 잇몸 혈류를 줄여 회복을 가장 크게 방해해요.",
    },
    "HYDRATION": {
        "title": "수분 섭취와 구강 건조 관리",
        "region": None,
        "text": "침은 음식 찌꺼기를 씻어내고 산성으로 기운 입안을 되돌리는 자연 세정액이에요. 물을 자주 조금씩 마시면 이 기능이 유지됩니다. 코가 막혀 입으로 숨을 쉬거나 카페인을 많이 마시면 입안이 마르니 함께 살펴보세요.",
    },
    "TOOTHBRUSH_CARE": {
        "title": "칫솔 관리와 교체",
        "region": None,
        "text": "칫솔모가 옆으로 벌어지기 시작하면 치아 면에 제대로 닿지 않아 닦이는 양이 크게 줄어요. 보통 3개월이 기준이지만, 그 전에 벌어졌다면 바로 바꾸는 것이 좋습니다. 사용한 뒤에는 물기를 털고 통풍이 되는 곳에 세워 말려주세요.",
    },
    "REGULAR_CHECKUP": {
        "title": "정기 검진",
        "region": None,
        "text": "치석이나 잇몸 염증은 초기에 아프지 않아서 스스로 알아채기 어려워요. 6개월에 한 번 검진을 받으면 눈에 보이기 전 단계에서 잡을 수 있습니다. 지금처럼 상태가 좋을 때 기준을 만들어 두면 나중에 변화를 비교하기도 쉬워요.",
    },
    "WATER_FLOSSER": {
        "title": "구강세정기 사용",
        "region": None,
        "text": "구강세정기는 물줄기로 음식 찌꺼기를 밀어내 주지만, 치아 면에 붙은 치태까지 떼어내지는 못해요. 치실이나 치간칫솔을 대신하기보다 함께 쓰는 도구로 생각하세요. 교정 장치를 하고 있거나 손을 쓰기 불편할 때 특히 도움이 됩니다.",
    },
}

EXCLUSIVE_GROUPS = [
    {"FOCUSED_BRUSHING", "BRUSHING_METHOD"},            # 칫솔질 계열
    {"FLOSS", "INTERDENTAL_BRUSH", "WATER_FLOSSER"},    # 치아 사이 청결 계열
]

# ══════════════════════════════════════════════════════════════
# 부위 특성 — 개조식 메모.
#
# 완결된 한국어 문장으로 두면 LLM 이 그대로 베껴 쓴다. 실제로 그랬다.
# (힌트 "치아 사이 틈이 좁아 칫솔모가 들어가지 못하고, 잇몸 경계선을 지나치기 쉽다"
#  → 출력 "치아 사이 틈이 좁아 칫솔모가 들어가지 못하고 잇몸 경계선을 지나치기 쉬운 자리예요"
#  거의 동일. 케이스가 달라도 같은 문장이 나와 LLM 을 쓰는 의미가 사라졌다.)
#
# 그래서 문장이 아니라 사실 조각만 넘긴다. 베낄 문장이 없으면 모델이 조립해야 한다.
# assemble_report 의 _copies_hint 가 그래도 베낀 경우를 잡아 재시도시킨다.
#
# 조각을 쓸 때 규칙:
# - 완결된 문장으로 쓰지 말 것. 어미("~쉽다", "~어렵다")를 붙이면 베끼기 쉬워진다.
# - 명사구나 화살표 표기로 끊어 쓸 것.
# - 병변 이름("치석", "치태")을 넣지 말 것. 그룹이 섞이면 누수가 된다.
# ══════════════════════════════════════════════════════════════
REGION_HINT = {
    "오른쪽 위 어금니": ["볼에 가려 칫솔 진입 각도 확보 곤란",
                    "귀밑샘 침 출구 인접 → 침 속 미네랄 침착 빠름"],
    "왼쪽 위 어금니":   ["볼에 가려 칫솔 진입 각도 확보 곤란",
                    "귀밑샘 침 출구 인접 → 침 속 미네랄 침착 빠름"],
    "오른쪽 아래 어금니": ["혀에 가려 안쪽 면 접근 곤란",
                     "씹는 면 홈 깊음 → 음식물 잔류"],
    "왼쪽 아래 어금니":   ["혀에 가려 안쪽 면 접근 곤란",
                     "씹는 면 홈 깊음 → 음식물 잔류"],
    "오른쪽 위 앞니": ["치아 간 간격 좁음 → 칫솔모 진입 불가",
                  "잇몸 경계선 지나치기 쉬움"],
    "왼쪽 위 앞니":   ["치아 간 간격 좁음 → 칫솔모 진입 불가",
                  "잇몸 경계선 지나치기 쉬움"],
    "오른쪽 아래 앞니": ["혀밑샘·턱밑샘 침 출구 바로 옆",
                    "침 속 미네랄 침착 속도 최상위 부위"],
    "왼쪽 아래 앞니":   ["혀밑샘·턱밑샘 침 출구 바로 옆",
                    "침 속 미네랄 침착 속도 최상위 부위"],
}

# 위 힌트는 치석 기준이다. 치태 그룹에는 아래를 우선 사용한다.
REGION_HINT_PLAQUE = {
    "오른쪽 아래 앞니": ["혀가 늘 닿아 이미 닦였다고 착각하기 쉬움",
                    "안쪽 면 칫솔 접촉 부족"],
    "왼쪽 아래 앞니":   ["혀가 늘 닿아 이미 닦였다고 착각하기 쉬움",
                    "안쪽 면 칫솔 접촉 부족"],
    "오른쪽 아래 어금니": ["혀에 가려 안쪽 면 접근 곤란",
                     "씹는 면 홈 깊음 → 음식물 잔류"],
    "왼쪽 아래 어금니":   ["혀에 가려 안쪽 면 접근 곤란",
                     "씹는 면 홈 깊음 → 음식물 잔류"],
    "오른쪽 위 어금니": ["볼에 가려 칫솔 진입 각도 확보 곤란",
                    "양치 후반부 → 마무리 소홀해지는 자리"],
    "왼쪽 위 어금니":   ["볼에 가려 칫솔 진입 각도 확보 곤란",
                    "양치 후반부 → 마무리 소홀해지는 자리"],
}

MAX_HINTS_PER_GROUP = 3

# 힌트를 그대로 베꼈는지 판정하는 기준. 공백·기호를 뗀 뒤 이만큼 연속 일치하면 베낌으로 본다.
HINT_COPY_NGRAM = 8


def select_management(s: dict, k: int = 3):
    """측정 결과를 보고 관리법 k개를 규칙으로 확정한다."""
    sc = {}

    def add(key, w):
        sc[key] = sc.get(key, 0) + w

    if s["has_calculus"]:
        add("SCALING", 100)                                   # 치석 있으면 반드시 1위
    if s["has_plaque"]:
        add("BRUSHING_METHOD", 40)
    # 32개 전치아 케이스에서 치아 수 가산점이 SCALING(100)을 넘지 않도록 상한을 둔다.
    if s["molar_cnt"]:
        add("INTERDENTAL_BRUSH", 30 + 5 * min(s["molar_cnt"], 8))
    if s["anterior_cnt"]:
        add("FLOSS", 30 + 5 * min(s["anterior_cnt"], 8))
    if s["lower_anterior_calculus"]:                          # 31~43 = 악하선 침샘 근처
        add("NIGHT_BRUSHING", 35)
        add("HYDRATION", 18)
    if s["issue_cnt"] >= 3:
        add("DIET", 28)
    if s["top_risk"] in ("CRITICAL", "HIGH"):
        add("IMMUNITY", 24)
        add("FOCUSED_BRUSHING", 26)
    if s["weak_zones"]:
        add("FOCUSED_BRUSHING", 20)
    if not s["has_issue"]:
        add("REGULAR_CHECKUP", 50)
        add("TOOTHBRUSH_CARE", 40)
        add("FLOSS", 30)

    # 아무것도 안 걸렸을 때를 위한 기본 풀
    for key, w in (("MOUTHWASH", 10), ("TONGUE_CLEANING", 8),
                   ("TOOTHBRUSH_CARE", 7), ("REGULAR_CHECKUP", 6), ("WATER_FLOSSER", 3)):
        add(key, w)

    ranked = sorted(sc, key=lambda x: (-sc[x], x))
    picked, used = [], set()
    for key in ranked:
        gi = next((i for i, g in enumerate(EXCLUSIVE_GROUPS) if key in g), None)
        if gi is not None and gi in used:
            continue
        picked.append(key)
        if gi is not None:
            used.add(gi)
        if len(picked) == k:
            return picked
    for key in ranked:                      # 그룹 제약으로 모자라면 채움
        if key not in picked:
            picked.append(key)
            if len(picked) == k:
                break
    return picked


def pick_region(s: dict, kind, used=None):
    """
    관리법 문장에 끼울 부위 이름을 고른다.
    used 에 이미 쓴 부위를 넘기면 되도록 겹치지 않는 부위를 먼저 고른다.
    해당하는 부위가 아예 없으면 None.
    """
    if not kind or not s["has_issue"]:
        return None
    used = used or set()

    if kind == "top":
        cands = [i["region"] for i in s["issues"]]
    elif kind == "calculus":
        cands = [i["region"] for i in s["issues"] if i["lesion"] == "CALCULUS"]
    elif kind == "molar":
        cands = [i["region"] for i in s["issues"] if is_molar(i["tooth"])]
    elif kind == "anterior":
        cands = [i["region"] for i in s["issues"] if not is_molar(i["tooth"])]
    else:
        return None

    if not cands:
        return None
    return next((c for c in cands if c not in used), cands[0])


def build_management(s: dict, picked: list):
    """선택된 항목의 고정 문장을 부위 이름과 함께 조립한다."""
    out, used = [], set()
    for key in picked:
        entry = MANAGEMENT_CATALOG[key]
        region = pick_region(s, entry.get("region"), used)
        if region:
            used.add(region)
            detail = entry["text"].format(region=region)
        else:
            detail = entry.get("plain", entry["text"])
        out.append({"title": entry["title"], "detail": detail})
    return out


# ══════════════════════════════════════════════════════════════
# 집계
# ══════════════════════════════════════════════════════════════
def fdi_to_region(tooth: int) -> str:
    quadrant, index = tooth // 10, tooth % 10
    vertical = {1: "오른쪽 위", 2: "왼쪽 위", 3: "왼쪽 아래", 4: "오른쪽 아래"}.get(quadrant, "기타")
    kind = "어금니" if index >= 4 else "앞니"
    return f"{vertical} {kind}"


def is_molar(tooth: int) -> bool:
    return tooth % 10 >= 4


def coverage_kr(ratio: float) -> str:
    for threshold, label in COVERAGE_BANDS:
        if ratio >= threshold:
            return label
    return COVERAGE_BANDS[-1][1]


def summarize_regions(regions, max_items: int = MAX_REGIONS_LISTED) -> str:
    """
    부위가 많아지면 이름을 그대로 나열하지 않고 묶는다.
    전치아 32개 케이스에서 8개 부위가 다 나오면
      "위아래 양쪽 어금니, 위아래 양쪽 앞니" 처럼 줄어든다.
    """
    regions = list(dict.fromkeys(regions))          # 순서 유지 중복 제거
    if len(regions) <= max_items:
        return ", ".join(regions)

    parsed = []
    for r in regions:
        tokens = r.split()
        if len(tokens) < 2:
            parsed.append((r, ""))
        else:
            parsed.append((" ".join(tokens[:-1]), tokens[-1]))

    out = []
    for kind in ("어금니", "앞니"):
        verts = [v for v, k in parsed if k == kind]
        if not verts:
            continue
        up = [v for v in ("오른쪽 위", "왼쪽 위") if v in verts]
        down = [v for v in ("오른쪽 아래", "왼쪽 아래") if v in verts]
        etc = [v for v in verts if v not in up + down]

        if len(up) == 2 and len(down) == 2:
            out.append(f"위아래 양쪽 {kind}")
        else:
            if len(up) == 2:
                out.append(f"위쪽 양쪽 {kind}")
            else:
                out.extend(f"{v} {kind}" for v in up)
            if len(down) == 2:
                out.append(f"아래쪽 양쪽 {kind}")
            else:
                out.extend(f"{v} {kind}" for v in down)
        out.extend(f"{v} {kind}" for v in dict.fromkeys(etc))

    # 분류에 안 잡힌 부위(기타 등)는 뒤에 붙인다
    out.extend(r for r, (v, k) in zip(regions, parsed) if k not in ("어금니", "앞니"))
    return ", ".join(out) if out else ", ".join(regions[:max_items])


def aggregate_session(teeth, zones=None, prev_issue_set=None) -> dict:
    """
    teeth : analysis_tooth_area 행들 (dict)
            {toothNumber, lesionType, areaRatio, toothScore, riskLevel}
    zones : analysis_result 행들 (dict)
            {viewType, zoneScore, zoneValid, detectedToothCount, totalCalculusRatio, totalPlaqueRatio}
    """
    zones = zones or []
    teeth = [t for t in teeth if t.get("lesionType")]

    # 같은 (치아, 병변)은 areaRatio 가 가장 큰 행만 남긴다 (confidence 대신)
    issue_map = {}
    for t in teeth:
        key = (t["toothNumber"], t["lesionType"])
        cur = issue_map.get(key)
        if cur is None or t.get("areaRatio", 0.0) > cur.get("areaRatio", 0.0):
            issue_map[key] = t

    issues = []
    for (tooth, lesion), t in issue_map.items():
        if prev_issue_set is None:
            change = None
        elif (tooth, lesion) in prev_issue_set:
            change = "PERSISTED"
        else:
            change = "NEW"
        issues.append({
            "tooth": tooth,
            "region": fdi_to_region(tooth),
            "lesion": lesion,
            "lesion_kr": LESION_KR.get(lesion, lesion),
            "riskLevel": t.get("riskLevel", "MEDIUM"),
            "coverage": coverage_kr(t.get("areaRatio", 0.0)),
            "change": change,
        })
    issues.sort(key=lambda x: (RISK_ORDER.get(x["riskLevel"], 5), x["tooth"]))

    if DEV_LIMIT:      # 배포에서는 None 이라 통과
        kept, cnt = [], {}
        for it in issues:
            c = cnt.get(it["lesion"], 0)
            if c < DEV_LIMIT.get(it["lesion"], 999):
                kept.append(it)
                cnt[it["lesion"]] = c + 1
        issues = kept

    resolved = []
    if prev_issue_set is not None:
        gone = prev_issue_set - set(issue_map.keys())
        resolved = [f"{fdi_to_region(t)} {LESION_KR.get(l, l)}" for t, l in sorted(gone)]

    weak_zones, invalid_zones = [], []
    for z in zones:
        name = VIEW_KR.get(z.get("viewType"), z.get("viewType", "알 수 없는 구역"))
        if not z.get("zoneValid", True):
            invalid_zones.append(name)
        elif z.get("zoneScore", 100.0) < WEAK_ZONE_THRESHOLD:
            weak_zones.append(name)

    return {
        "issues": issues,
        "issue_cnt": len(issues),
        "has_issue": len(issues) > 0,
        "has_calculus": any(i["lesion"] == "CALCULUS" for i in issues),
        "has_plaque": any(i["lesion"] == "PLAQUE" for i in issues),
        "molar_cnt": sum(1 for i in issues if is_molar(i["tooth"])),
        "anterior_cnt": sum(1 for i in issues if not is_molar(i["tooth"])),
        "lower_anterior_calculus": any(
            i["lesion"] == "CALCULUS" and i["tooth"] // 10 in (3, 4) and i["tooth"] % 10 <= 3
            for i in issues),
        "top_risk": issues[0]["riskLevel"] if issues else None,
        "resolved": resolved,
        "weak_zones": weak_zones,
        "invalid_zones": invalid_zones,
    }


def _particle(word: str) -> str:
    """받침 유무로 '이/가'를 고른다. 치석이 / 치태가"""
    return "이" if (ord(word[-1]) - 0xAC00) % 28 else "가"


def group_by_lesion(s: dict):
    """치석/치태로 묶는다. issues 가 이미 위험도순이라 그 순서가 유지된다."""
    groups = []
    for lesion in ("CALCULUS", "PLAQUE"):
        members = [i for i in s["issues"] if i["lesion"] == lesion]
        if not members:
            continue
        seen, regions = set(), []
        for m in members:
            if m["region"] not in seen:
                seen.add(m["region"])
                regions.append(m["region"])
        other_kr = LESION_KR["PLAQUE" if lesion == "CALCULUS" else "CALCULUS"]
        hints = []
        for r in regions:
            frags = (REGION_HINT_PLAQUE.get(r) if lesion == "PLAQUE" else None) or REGION_HINT.get(r) or []
            for f in frags:
                if f in hints:
                    continue
                if other_kr in f:      # 다른 병변 이름이 든 조각은 넣지 않는다
                    continue
                hints.append(f)
        groups.append({
            "lesion": lesion,
            "lesion_kr": LESION_KR[lesion],
            "regions": regions,
            "regions_text": summarize_regions(regions),
            "tooth_cnt": len(members),
            "riskLevel": members[0]["riskLevel"],   # 그룹 내 최고 위험도
            "hints": hints[:MAX_HINTS_PER_GROUP],
            # 이전 측정과 비교한 변화. prevTeeth 가 없으면 change 가 전부 None 이라 0.
            "persisted_cnt": sum(1 for m in members if m["change"] == "PERSISTED"),
            "new_cnt": sum(1 for m in members if m["change"] == "NEW"),
            "has_prev": any(m["change"] is not None for m in members),
        })
    groups.sort(key=lambda g: RISK_ORDER.get(g["riskLevel"], 5))
    return groups


def build_observation(g: dict) -> str:
    """'A, B, C에서 치석이 관찰돼요.' — 코드가 만든다."""
    kr = g["lesion_kr"]
    return f'{g["regions_text"]}에서 {kr}{_particle(kr)} 관찰돼요.'


def _eun_neun(word: str) -> str:
    """받침 유무로 '은/는'을 고른다. 치석은 / 치태는"""
    return "은" if (ord(word[-1]) - 0xAC00) % 28 else "는"


def build_persisted_note(g: dict) -> str:
    """
    그룹 안에서 지난번에도 있던 치아가 몇 개인지. 코드가 만든다.
    prevTeeth 가 없으면(has_prev False) 빈 문자열 — 첫 측정에는 비교할 게 없다.
    """
    if not g.get("has_prev"):
        return ""
    p, n, total = g["persisted_cnt"], g["new_cnt"], g["tooth_cnt"]
    if p == 0:
        return "모두 지난번 측정에는 없던 자리예요."
    if n == 0:
        return "지난번 측정에도 있던 자리가 그대로 남아 있어요."
    return f"이 중 치아 {p}개는 지난번 측정에도 있던 자리예요."


def build_resolved_note(s: dict) -> str:
    """지난번에 있었지만 이번에 사라진 부위. 코드가 만든다."""
    if not s["resolved"]:
        return ""
    joined = ", ".join(s["resolved"])
    return f'지난번에 보였던 {joined}{_eun_neun(joined)} 이번 측정에서 보이지 않아요.'


def recheck_sentence(s: dict) -> str:
    if s["top_risk"] in ("CRITICAL", "HIGH"):
        return "2주 후 재측정을 권장해요."
    if s["has_issue"]:
        return "1개월 후 재측정을 권장해요."
    return "3개월 후 재측정을 권장해요."


def make_title(it) -> str:
    return f'{it["region"]} {it["lesion_kr"]}'


def build_notice(s: dict):
    """촬영이 유효하지 않은 구역은 '문제 없음'이 아니라는 걸 반드시 알린다."""
    if not s["invalid_zones"]:
        return None
    return (f'{", ".join(s["invalid_zones"])} 구역은 사진 상태가 좋지 않아 이번 분석에서 제외됐어요. '
            f'해당 구역은 결과에 반영되지 않았으니 다음 측정 때 다시 촬영해 주세요.')


# ══════════════════════════════════════════════════════════════
# 프롬프트 — LLM은 detail 문장만 쓴다
# ══════════════════════════════════════════════════════════════
SYSTEM_PROMPT = """당신은 구강 관리 앱의 리포트를 쓰는 치과위생사입니다.
사용자가 자기 입속 상태를 이해하도록 부위별 설명 문장을 작성합니다.

[출력 형식 — 가장 중요]
- 오직 JSON 객체 하나만 출력합니다. 앞뒤에 어떤 설명도 붙이지 않습니다.
- 코드블록 표시(```), 마크다운, 번호 매기기, 인사말, 요약을 쓰지 않습니다.
- JSON 을 출력한 뒤에는 즉시 멈춥니다. 대화를 이어 쓰지 않습니다.

[말투 규칙]
- 다정하고 부드러운 해요체("~있어요", "~닦아주세요"). 합쇼체("~합니다")를 쓰지 않습니다.
- 2인칭 호칭, 인사말, 서명을 쓰지 않습니다.
- "플라크" 금지, "치태"로 통일합니다. 굳은 것은 "치석"입니다.
- 관찰된 사실은 "~이 관찰돼요 / ~이 발견됐어요"로 씁니다.
  부위 특성은 "~이 쌓이기 쉬운 곳이에요"로 씁니다. 두 표현을 섞은 "잘 생겼어요"는 쓰지 않습니다.
- 명사 뒤에 "때"를 바로 붙이지 않습니다. "양치 때"(X) → "양치할 때"(O).
- 칫솔 동작의 목적어는 반드시 '칫솔' 또는 '칫솔모'입니다. 부위를 세우거나 기울인다고 쓰지 않습니다.
- 위험도 등급(CRITICAL, HIGH 등), 점수, 퍼센트, 재측정 주기를 문장에 쓰지 않습니다. 앱이 따로 표시합니다.
- 영어 단어를 쓰지 않습니다.

[문장 구조 — 아래 골격만 따른다. 내용은 '부위 특성 메모'에서 직접 만든다]
첫 문장: 그 부위에서 칫솔이 닿기 어려운 이유, 또는 무심코 놓치기 쉬운 이유. 해요체로 끝낸다.
둘째 문장: 허용된 칫솔 동작 표현 중 하나 + "움직여 주세요" 또는 "닦아주세요".

[문장 밀도 참고 — 치아와 무관한 주제의 예시입니다]
아래는 내용을 참고할 예시가 아닙니다. 문장의 길이와 문체만 보세요.
"손잡이가 짧아 손목을 크게 돌려야 하는 도구라 끝까지 힘이 잘 전달되지 않아요."
"서랍 안쪽은 손이 닿지 않아 한번 넣어두면 다시 꺼내기가 번거로워요."

첫 문장은 위 예시 정도의 길이와 밀도로 씁니다.
"~어려운 곳이에요"처럼 이유를 한마디로 끊지 마세요.
왜 그런지와 그래서 어떻게 되는지를 한 문장 안에 함께 담습니다.
메모가 2개 주어졌다면 둘을 이어서 한 문장으로 엮는 것이 가장 좋습니다.

치아 부위에 대한 예시 문장을 주지 않는 이유가 있습니다. 예시를 주면 그것을 그대로
베껴 모든 사용자의 리포트가 같은 문장이 되기 때문입니다. 부위 특성 메모는 측정
결과마다 다르므로, 매번 그 메모를 근거로 새 문장을 만들어야 합니다.
"""


# LLM 이 골라 쓸 수 있는 칫솔 동작 표현 (검증에도 쓴다)
BRUSH_MOTIONS = [
    "칫솔모 끝을 잇몸 경계에 45도로 대고 짧게 진동시키듯",
    "칫솔을 세로로 세워 안쪽 끝까지 넣고 작은 원을 그리듯",
    "칫솔모 끝을 잇몸 쪽으로 기울여 아래에서 위로 쓸어 올리듯",
    "칫솔 머리를 안쪽 깊이 넣고 앞뒤로 짧게 왕복하듯",
]


def build_report_prompt(s: dict, picked=None) -> str:
    if s["has_issue"]:
        groups = group_by_lesion(s)
        group_lines = "\n".join(
            f'  {i+1}. {g["lesion_kr"]} — {g["regions_text"]} (치아 {g["tooth_cnt"]}개)\n'
            + (("\n     [부위 특성 메모]\n"
                + "\n".join(f"       - {h}" for h in g["hints"]))
               if g["hints"] else "\n     [부위 특성 메모] 없음")
            for i, g in enumerate(groups))
        # 호전/변화 문구는 LLM 에 맡기지 않는다.
        # "정확히 2문장" 규칙과 trim_sentences 와 충돌해 3번째 문장이 잘려 나가기 때문이다.
        # build_persisted_note / build_resolved_note 가 코드로 만들어 붙인다.
        motions = "\n".join(f"  · {m}" for m in BRUSH_MOTIONS)
        body = f"""아래 {len(groups)}개 그룹 각각에 대해 관리 방법 문장을 순서대로 만들어 "details" 배열에 넣는다.
"details" 배열의 길이는 정확히 {len(groups)} 이어야 하고, 순서도 그룹 목록과 같아야 한다.

[그룹 목록]
{group_lines}

[작성 규칙]
- 각 항목은 정확히 2문장. 한 문장은 60자를 넘기지 않는다.
- 어디에 무엇이 관찰되는지는 앱이 이미 표시하므로 절대 다시 쓰지 않는다. 부위 이름을 나열하지 않는다.
- "관찰돼요", "발견됐어요" 같은 관찰 서술을 쓰지 않는다. 앱이 이미 쓴 문장이다.
- 첫 문장은 그 부위들이 왜 관리가 어려운지, 둘째 문장은 그 부위들을 어떻게 관리하면 좋은지.
- '부위 특성 메모'는 완결된 문장이 아니라 사실 조각이다. 이 조각을 근거로 삼되, 표현은 직접 만든다.\n  메모의 단어 배열을 그대로 이어 붙이지 않는다. 메모에 없는 해부학 설명을 지어내지도 않는다.\n  메모가 2개 이상이면 억지로 다 넣지 말고, 그 부위에 가장 중요한 것을 골라 한 문장으로 쓴다.
- 병변을 제거하는 방법이나 치과 치료를 언급하지 않는다. 앱이 '관리법' 항목에서 따로 안내한다.
  둘째 문장은 그 부위를 칫솔로 닦는 방법만 쓴다. "제거하기 위해", "없애려면" 같은 말을 쓰지 않는다.
- 두 그룹에 같은 동작 표현을 쓰지 않는다.\n- 지난 측정과의 비교나 호전 여부는 쓰지 않는다. 앱이 따로 표시한다.
- 칫솔 동작은 다음 표현 중에서만 골라 쓴다. 비유나 새로운 표현을 지어내지 않는다.
{motions}
- 입안 밖의 신체 부위(복부, 가슴, 목 등)를 언급하지 않는다.\n- 힘을 주어 닦으라고 쓰지 않는다. 힘주어 문지르면 치아가 패이고 잇몸이 내려앉는다.\n  "힘차게", "세게", "강하게" 같은 말을 쓰지 않는다.
- 각 그룹은 자기 그룹의 병변 이름만 쓴다. 치태 그룹에 "치석"을, 치석 그룹에 "치태"를 쓰지 않는다.
- "~된 부위들은"으로 문장을 시작하지 않는다. 부위를 가리키는 말 없이 바로 이유부터 쓴다."""
        example = ('{"details": [' +
                   ", ".join(['"첫 문장. 둘째 문장."'] * len(groups)) + ']}')
    else:
        body = ('문제가 발견되지 않았다. "details" 배열에 문장 하나만 넣는다: '
                '양호한 상태를 축하하고 지금 습관을 유지하도록 격려하는 2문장.\n'
                '- 재측정 주기는 쓰지 않는다. 앱이 따로 붙인다.')
        example = '{"details": ["첫 문장. 둘째 문장."]}'

    return f"""아래 스캔 결과를 보고 부위별 설명 문장을 작성하세요.

{body}

[출력 형식 — 이 형태 그대로, 다른 텍스트 없이]
{example}"""


# ══════════════════════════════════════════════════════════════
# 조립 + 검증
# ══════════════════════════════════════════════════════════════
def _overlaps(a: str, b: str, n: int = 5) -> bool:
    aw, bw = a.split(), b.split()
    if len(aw) < n or len(bw) < n:
        return False
    grams = {tuple(aw[i:i + n]) for i in range(len(aw) - n + 1)}
    return any(tuple(bw[i:i + n]) in grams for i in range(len(bw) - n + 1))


# 후처리(PHRASE_FIXES)가 놓친 문체 이탈. 해요체 문장에 남아 있으면 안 되는 어미들.
#   "니다" — 교정 목록에 없는 'ㅂ니다' 어형이 남은 경우 (예: 힘듭니다)
#   "네요"/"군요" — 감탄조. 리포트 문체가 아니다.
_FORMAL_LEAK = re.compile(r"니다|네요|군요|는군|랍니다|죠(?=[\s.,!?]|$)")

# 첫 문장이 너무 짧으면 힌트를 제대로 풀어 쓰지 않은 것이다.
#   예: "볼에 가려 칫솔이 닿기 어려운 곳이 많아요." (정보량 거의 없음)
MIN_FIRST_SENTENCE_LEN = 28


def _first_sentence_too_short(detail: str) -> bool:
    sents = split_sentences(detail)
    if not sents:
        return True
    return len(sents[0]) < MIN_FIRST_SENTENCE_LEN


def _normalize_for_copy(text: str) -> str:
    """베낌 판정용 정규화. 공백·기호·조사성 문자를 떼어 표면적 변형을 무력화한다."""
    return re.sub(r"[\s,.·→~\-()\[\]의를을이가은는와과도로에서]", "", text)


def _copies_hint(detail: str, hints, n: int = HINT_COPY_NGRAM) -> bool:
    """
    LLM 문장이 부위 특성 메모를 그대로 옮겨 썼는지 본다.
    정규화한 뒤 n자 연속 일치가 있으면 베낌으로 판정한다.
    """
    d = _normalize_for_copy(detail)
    for h in hints or []:
        hh = _normalize_for_copy(h)
        if len(hh) < n:
            continue
        for i in range(len(hh) - n + 1):
            if hh[i:i + n] in d:
                return True
    return False


BANNED_WORDS = ["이어폰", "빗자루", "붓처럼", "지우개", "칼처럼",
                "복부", "배 근처", "가슴", "목구멍", "식도", "위장", "코 근처",
                # 칫솔질에 힘을 주라는 조언은 의학적으로 틀렸다.
                # MANAGEMENT_CATALOG 의 BRUSHING_METHOD 와 정면으로 충돌한다.
                # 긍정 명령형만 잡는다. "세게 문지르지 않아도" 같은 부정형은
                # 올바른 조언이므로 차단하지 않는다.
                "힘차게", "힘껏", "빡빡",
                "세게 문질러", "세게 닦아", "강하게 문질러", "강하게 닦아",
                "힘주어 문질러", "힘을 주어 닦", "힘을 주어 문질",
                # 좁은 틈에 칫솔을 밀어 넣으라는 조언도 잇몸을 상하게 한다.
                # MANAGEMENT_CATALOG 의 INTERDENTAL_BRUSH 와 충돌한다.
                "밀어 넣어야", "밀어넣어야", "깊숙이 밀어", "억지로 넣"]

# 앱이 따로 표시하는 정보가 문장에 새어 나오는지
_RISK_LEAK = re.compile(r"(CRITICAL|HIGH|MEDIUM|LOW|위험도|점수|\d+\s*점|\d+\s*%|재측정)")
# 코드가 이미 쓴 관찰 서술이 중복되는지
_OBSERVE_LEAK = re.compile(r"(관찰(돼요|됩니다|됐어요|되었어요)|발견(됐어요|되었어요|됩니다))")
# 영문 누수
_LATIN_LEAK = re.compile(r"[A-Za-z]{3,}")
# 치료·제거 언급. detail 은 '왜 어려운지'와 '어떻게 닦는지'만 쓴다.
_TREATMENT_LEAK = re.compile(r"스케일|제거|떼어내|떼어 내|없애|없앨|긁어내|치과|시술|치료")


def assemble_report(s: dict, llm_out: dict, picked: list):
    raw = [d for d in llm_out.get("details", []) if isinstance(d, str)]
    details = [trim_sentences(polish(d)) for d in raw]

    groups = group_by_lesion(s) if s["has_issue"] else []
    expected = len(groups) if s["has_issue"] else 1
    length_ok = (len(details) == expected)
    details = (details + [""] * expected)[:expected]

    risk_detail = []
    if s["has_issue"]:
        for g, sent in zip(groups, details):
            # 관찰 → 이전 측정 대비 변화 → LLM 관리 문장
            parts = [build_observation(g), build_persisted_note(g), sent]
            risk_detail.append({
                "title": f'{g["lesion_kr"]} · 치아 {g["tooth_cnt"]}개',
                "detail": " ".join(p for p in parts if p).strip(),
                "riskLevel": g["riskLevel"],
            })
        # 재측정 주기는 첫 항목 끝, 사라진 부위는 마지막 항목 끝
        risk_detail[0]["detail"] = (risk_detail[0]["detail"] + " " + recheck_sentence(s)).strip()
        rn = build_resolved_note(s)
        if rn:
            risk_detail[-1]["detail"] = (risk_detail[-1]["detail"] + " " + rn).strip()
    else:
        risk_detail.append({
            "title": "전반적으로 양호한 상태입니다!",
            "detail": (details[0] + " " + recheck_sentence(s)).strip(),
            "riskLevel": "VERY_LOW",
        })

    report = {"riskDetail": risk_detail, "management": build_management(s, picked)}
    notice = build_notice(s)
    if notice:
        report["notice"] = notice

    # ── 검증은 LLM 이 쓴 문장에만 한다.
    #    관찰 문장은 코드가 만들어 검증할 필요가 없고,
    #    두 그룹의 관찰 문장이 서로 비슷해 dup 오탐을 낸다.
    too_short = any(len(d) < MIN_DETAIL_LEN for d in details)
    too_long = any(len(d) > MAX_DETAIL_LEN for d in details)
    bad_sentence_cnt = any(len(split_sentences(d)) != SENTENCES_PER_DETAIL for d in details)
    # 치료·제거 언급 누수.
    #   치석은 칫솔질로 제거되지 않는다. "치석을 제거하기 위해 칫솔로 닦으세요"는
    #   의학적으로 틀린 정보이고, MANAGEMENT_CATALOG 의 '치과 스케일링' 과 충돌한다.
    #   스케일링 안내는 management 항목이 이미 정확하게 다루므로 detail 에서는 금지한다.
    treatment_leak = any(_TREATMENT_LEAK.search(d) for d in details)
    dup_action = any(_overlaps(a, b) for a, b in itertools.combinations(details, 2))
    has_banned_word = any(b in d for d in details for b in BANNED_WORDS)
    risk_leak = any(_RISK_LEAK.search(d) for d in details)
    observe_leak = any(_OBSERVE_LEAK.search(d) for d in details)
    latin_leak = any(_LATIN_LEAK.search(d) for d in details)
    hint_copy = any(_copies_hint(d, g["hints"]) for g, d in zip(groups, details)) if groups else False
    formal_leak = any(_FORMAL_LEAK.search(d) for d in details)
    thin_first = any(_first_sentence_too_short(d) for d in details)

    lesion_mixup = False
    if s["has_issue"]:
        for g, sent in zip(groups, details):
            other = "치태" if g["lesion"] == "CALCULUS" else "치석"
            if other in sent:
                lesion_mixup = True
                break

    # 실패 항목만 모아 찍는다.
    # (이전에는 length_ok 처럼 정상일 때 True 인 변수와 섞여 나와 반대로 읽혔다.
    #  여기 나열되는 이름은 모두 "이게 문제다" 라는 뜻이다.)
    failures = {
        "detail_개수_불일치": not length_ok,
        "문장_너무_짧음": too_short,
        "문장_너무_김": too_long,
        "문장수_2개_아님": bad_sentence_cnt,
        "치료_제거_언급": treatment_leak,
        "두_그룹_표현_중복": dup_action,
        "금칙어": has_banned_word,
        "등급_수치_누수": risk_leak,
        "관찰서술_중복": observe_leak,
        "영문_누수": latin_leak,
        "병변_혼동": lesion_mixup,
        "힌트_베낌": hint_copy,
        "문체_이탈": formal_leak,
        "첫문장_밀도_부족": thin_first,
    }
    needs_retry = any(failures.values())

    if needs_retry:
        print("[검증] 실패: " + ", ".join(k for k, v in failures.items() if v))
    return report, needs_retry


HEALTHY_DETAIL = ("이번 측정에서는 치석이나 치태가 발견되지 않았어요. "
                  "지금의 관리 습관을 그대로 이어가시면 좋겠습니다.")

HEALTHY_RESOLVED = ("이번 측정에서는 치석이나 치태가 발견되지 않았어요. "
                    "지난번에 관찰됐던 {resolved}도 이번에는 보이지 않아 잘 관리되고 있어요.")


def build_healthy_report(s: dict, picked: list):
    if s["resolved"]:
        detail = HEALTHY_RESOLVED.format(resolved=", ".join(s["resolved"]))
    else:
        detail = HEALTHY_DETAIL
    report = {
        "riskDetail": [{
            "title": "전반적으로 양호한 상태입니다!",
            "detail": (detail + " " + recheck_sentence(s)).strip(),
            "riskLevel": "VERY_LOW",
        }],
        "management": build_management(s, picked),
        "disclaimer": DISCLAIMER,
    }
    notice = build_notice(s)
    if notice:
        report["notice"] = notice
    return report
