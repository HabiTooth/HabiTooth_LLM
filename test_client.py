# test_client.py — 새 스키마(zones + teeth)로 요청.
#                  report_logic.py / test.py 와 함께 쓰는 테스트 클라이언트.
#                  모델을 갈아타도 이 파일은 그대로 쓴다.
#
# 사용법
#   python test_client.py                    # mid(15건) 실제 생성
#   python test_client.py --case empty       # 병변 0건 (LLM 안 탐, 고정 문장 경로)
#   python test_client.py --case full        # 전치아 32개
#   python test_client.py --case stress      # 32개 치아 × 2병변 = 64행
#   python test_client.py --case full --dry  # 모델 안 돌리고 프롬프트만
#   python test_client.py --case all         # 전 케이스 순차 실행
#   python test_client.py --case v2-new-calculus --dry   # v2: 점수 분해·할 일·프롬프트·폴백만
#   python test_client.py --case v2 --v2     # v2 케이스 6개 실제 생성, score/interpretation/actions 만 출력
#   python test_client.py --base https://xxxx.trycloudflare.com   # Colab 터널
import argparse
import json
import sys
import time
from datetime import datetime

import requests

# ══════════════════════════════════════════════════════════════
# zones — analysis_result 행에 해당
# ══════════════════════════════════════════════════════════════
ZONES_DIRTY = [
    {"viewType": "UPPER_RIGHT",  "zoneScore": 52.0, "zoneValid": True,  "detectedToothCount": 5,
     "totalCalculusRatio": 0.31, "totalPlaqueRatio": 0.04},
    {"viewType": "OUTER_RIGHT",  "zoneScore": 68.0, "zoneValid": True,  "detectedToothCount": 4,
     "totalCalculusRatio": 0.18, "totalPlaqueRatio": 0.02},
    {"viewType": "UPPER_CENTER", "zoneScore": 81.0, "zoneValid": True,  "detectedToothCount": 6,
     "totalCalculusRatio": 0.0,  "totalPlaqueRatio": 0.12},
    {"viewType": "LOWER_LEFT",   "zoneScore": 74.0, "zoneValid": True,  "detectedToothCount": 5,
     "totalCalculusRatio": 0.09, "totalPlaqueRatio": 0.01},
    {"viewType": "OUTER_LEFT",   "zoneScore": 0.0,  "zoneValid": False, "detectedToothCount": 0,
     "totalCalculusRatio": 0.0,  "totalPlaqueRatio": 0.0},
]

ZONES_CLEAN = [
    {"viewType": v, "zoneScore": 94.0, "zoneValid": True, "detectedToothCount": 5,
     "totalCalculusRatio": 0.0, "totalPlaqueRatio": 0.0}
    for v in ("UPPER_RIGHT", "UPPER_CENTER", "UPPER_LEFT",
              "LOWER_RIGHT", "LOWER_CENTER", "LOWER_LEFT",
              "OUTER_RIGHT", "OUTER_CENTER", "OUTER_LEFT")
]

ZONES_ALL_DIRTY = [
    {"viewType": v, "zoneScore": 38.0, "zoneValid": True, "detectedToothCount": 6,
     "totalCalculusRatio": 0.42, "totalPlaqueRatio": 0.21}
    for v in ("UPPER_RIGHT", "UPPER_CENTER", "UPPER_LEFT",
              "LOWER_RIGHT", "LOWER_CENTER", "LOWER_LEFT",
              "OUTER_RIGHT", "OUTER_CENTER", "OUTER_LEFT")
]

# ══════════════════════════════════════════════════════════════
# teeth — analysis_tooth_area 행에 해당
# ══════════════════════════════════════════════════════════════
TEETH_MID = [
    # 치석 10건 (16 은 중복 행 → areaRatio 큰 쪽만 남는지 확인용)
    {"toothNumber": 16, "lesionType": "CALCULUS", "areaRatio": 0.34, "toothScore": 41.0, "riskLevel": "CRITICAL"},
    {"toothNumber": 16, "lesionType": "CALCULUS", "areaRatio": 0.22, "toothScore": 48.0, "riskLevel": "HIGH"},
    {"toothNumber": 17, "lesionType": "CALCULUS", "areaRatio": 0.28, "toothScore": 45.0, "riskLevel": "CRITICAL"},
    {"toothNumber": 26, "lesionType": "CALCULUS", "areaRatio": 0.19, "toothScore": 55.0, "riskLevel": "HIGH"},
    {"toothNumber": 27, "lesionType": "CALCULUS", "areaRatio": 0.16, "toothScore": 58.0, "riskLevel": "HIGH"},
    {"toothNumber": 31, "lesionType": "CALCULUS", "areaRatio": 0.24, "toothScore": 47.0, "riskLevel": "CRITICAL"},
    {"toothNumber": 32, "lesionType": "CALCULUS", "areaRatio": 0.18, "toothScore": 56.0, "riskLevel": "HIGH"},
    {"toothNumber": 41, "lesionType": "CALCULUS", "areaRatio": 0.21, "toothScore": 51.0, "riskLevel": "HIGH"},
    {"toothNumber": 36, "lesionType": "CALCULUS", "areaRatio": 0.07, "toothScore": 79.0, "riskLevel": "LOW"},
    {"toothNumber": 46, "lesionType": "CALCULUS", "areaRatio": 0.09, "toothScore": 75.0, "riskLevel": "LOW"},
    # 치태 6건
    {"toothNumber": 11, "lesionType": "PLAQUE", "areaRatio": 0.13, "toothScore": 66.0, "riskLevel": "MEDIUM"},
    {"toothNumber": 21, "lesionType": "PLAQUE", "areaRatio": 0.15, "toothScore": 63.0, "riskLevel": "MEDIUM"},
    {"toothNumber": 24, "lesionType": "PLAQUE", "areaRatio": 0.12, "toothScore": 68.0, "riskLevel": "MEDIUM"},
    {"toothNumber": 37, "lesionType": "PLAQUE", "areaRatio": 0.11, "toothScore": 70.0, "riskLevel": "MEDIUM"},
    {"toothNumber": 46, "lesionType": "PLAQUE", "areaRatio": 0.10, "toothScore": 72.0, "riskLevel": "MEDIUM"},
    {"toothNumber": 47, "lesionType": "PLAQUE", "areaRatio": 0.06, "toothScore": 81.0, "riskLevel": "LOW"},
]

ALL_TEETH = [q * 10 + i for q in (1, 2, 3, 4) for i in range(1, 9)]   # FDI 32개


def _risk(ratio: float) -> str:
    if ratio >= 0.30:
        return "CRITICAL"
    if ratio >= 0.18:
        return "HIGH"
    if ratio >= 0.10:
        return "MEDIUM"
    return "LOW"


def _row(tooth: int, lesion: str, ratio: float):
    return {
        "toothNumber": tooth,
        "lesionType": lesion,
        "areaRatio": round(ratio, 3),
        "toothScore": round(100 - ratio * 180, 1),
        "riskLevel": _risk(ratio),
    }


def build_full():
    """전치아 32개 — 치아 하나당 병변 1개. 어금니·하악전치는 치석, 상악전치는 치태."""
    rows = []
    for idx, t in enumerate(ALL_TEETH):
        pos = t % 10
        quad = t // 10
        if pos >= 4:                              # 소구치·대구치
            lesion = "CALCULUS"
        elif quad in (3, 4) and pos <= 2:         # 하악 전치 (악하선 근처)
            lesion = "CALCULUS"
        else:
            lesion = "PLAQUE"
        ratio = 0.06 + (idx % 9) * 0.035          # 0.06 ~ 0.34
        rows.append(_row(t, lesion, ratio))
    return rows


def build_stress():
    """32개 치아에 치석·치태 모두 — 64행. 집계/요약 상한 확인용."""
    rows = []
    for idx, t in enumerate(ALL_TEETH):
        rows.append(_row(t, "CALCULUS", 0.05 + (idx % 9) * 0.035))
        rows.append(_row(t, "PLAQUE", 0.04 + (idx % 7) * 0.030))
    return rows


# ══════════════════════════════════════════════════════════════
# v2 케이스 — 13구역 + 광원별 2행 + prevTeeth 객체형 + totalScore
# Spring 이 실제로 보내는 형태를 흉내낸다. zoneScore 는 병변별 치아 점수 평균(= 100 − 3×평균면적비).
# ══════════════════════════════════════════════════════════════
def _z(view, light, score, detected=4, calc=None, plq=None, valid=True):
    return {"viewType": view, "lightType": light, "zoneScore": score, "zoneValid": valid,
            "detectedToothCount": detected, "totalCalculusRatio": calc, "totalPlaqueRatio": plq}


def _t(tooth, lesion, ratio):
    return {"toothNumber": tooth, "lesionType": lesion, "areaRatio": ratio,
            "toothScore": round(100 - 3 * ratio, 1), "riskLevel": _risk(ratio / 100)}


# A. 신규 치석 지배 (목업 사용자): 3구역, 치태 거의 없음, 치석 4곳 전부 새로 생김, 점수 급락
V2_A_ZONES = [
    _z("LOWER_FRONT", "WHITE", 80.0, 4, calc=6.7),   _z("LOWER_FRONT", "UV", 99.4, 4, plq=0.2),
    _z("UPPER_FRONT", "WHITE", 88.0, 4, calc=4.0),   _z("UPPER_FRONT", "UV", 100.0, 4, plq=0.0),
    _z("OUTER_CENTER", "WHITE", 95.0, 6, calc=1.7),  _z("OUTER_CENTER", "UV", 99.0, 6, plq=0.3),
]
V2_A_TEETH = [_t(41, "CALCULUS", 18.0), _t(42, "CALCULUS", 8.7), _t(11, "CALCULUS", 12.0),
              _t(21, "CALCULUS", 4.0), _t(41, "PLAQUE", 0.8), _t(31, "PLAQUE", 0.6)]
V2_A_PREV = [{"toothNumber": 41, "lesionType": "PLAQUE", "areaRatio": 6.2}]

# B. 치태 지배 (첫 촬영, 비교 없음): 어금니 설측 치태 많음, 치석 없음
V2_B_ZONES = [
    _z("LOWER_LEFT_MOLAR", "WHITE", 100.0, 3, calc=0.0), _z("LOWER_LEFT_MOLAR", "UV", 70.0, 3, plq=10.0),
    _z("LOWER_RIGHT_MOLAR", "WHITE", 100.0, 3, calc=0.0), _z("LOWER_RIGHT_MOLAR", "UV", 76.0, 3, plq=8.0),
    _z("LOWER_FRONT", "WHITE", 100.0, 4, calc=0.0),       _z("LOWER_FRONT", "UV", 88.0, 4, plq=4.0),
]
V2_B_TEETH = [_t(36, "PLAQUE", 14.2), _t(37, "PLAQUE", 9.0), _t(46, "PLAQUE", 11.0), _t(47, "PLAQUE", 5.0),
              _t(41, "PLAQUE", 6.0), _t(31, "PLAQUE", 4.0)]

# C. 치석 유지 + 치태 감소 (개선 중): 지난번 대비 치태 줄고 치석 그대로
V2_C_ZONES = [
    _z("LOWER_FRONT", "WHITE", 90.8, 5, calc=3.1),  _z("LOWER_FRONT", "UV", 75.0, 5, plq=8.3),
    _z("UPPER_FRONT", "WHITE", 98.0, 4, calc=0.7),  _z("UPPER_FRONT", "UV", 92.0, 4, plq=2.7),
]
V2_C_PREV_ZONES = [
    _z("LOWER_FRONT", "WHITE", 90.8, 5, calc=3.1),  _z("LOWER_FRONT", "UV", 68.0, 5, plq=10.7),
    _z("UPPER_FRONT", "WHITE", 98.0, 4, calc=0.7),  _z("UPPER_FRONT", "UV", 86.0, 4, plq=4.7),
]
V2_C_TEETH = [_t(41, "CALCULUS", 12.1), _t(31, "CALCULUS", 3.2),
              _t(41, "PLAQUE", 23.0), _t(31, "PLAQUE", 14.2), _t(42, "PLAQUE", 4.5), _t(11, "PLAQUE", 5.0)]
V2_C_PREV = [{"toothNumber": 41, "lesionType": "CALCULUS", "areaRatio": 12.1},
             {"toothNumber": 31, "lesionType": "CALCULUS", "areaRatio": 3.0},
             {"toothNumber": 41, "lesionType": "PLAQUE", "areaRatio": 31.0},
             {"toothNumber": 31, "lesionType": "PLAQUE", "areaRatio": 18.0},
             {"toothNumber": 42, "lesionType": "PLAQUE", "areaRatio": 9.0},
             {"toothNumber": 11, "lesionType": "PLAQUE", "areaRatio": 7.0},
             {"toothNumber": 21, "lesionType": "PLAQUE", "areaRatio": 6.0}]

# D. 깨끗 (지난번 치태 사라짐)
V2_D_ZONES = [_z(v, l, 100.0, 4, calc=0.0 if l == "WHITE" else None, plq=0.0 if l == "UV" else None)
              for v in ("UPPER_FRONT", "LOWER_FRONT", "OUTER_CENTER", "LOWER_LEFT_MOLAR",
                        "LOWER_RIGHT_MOLAR", "UPPER_LEFT_MOLAR", "UPPER_RIGHT_MOLAR")
              for l in ("WHITE", "UV")]
V2_D_PREV = [{"toothNumber": 36, "lesionType": "PLAQUE", "areaRatio": 5.0},
             {"toothNumber": 41, "lesionType": "PLAQUE", "areaRatio": 3.0}]

# E. 저커버리지 (2구역만) + 치석 1곳 + 구형 prevTeeth 배열 + 무효 구역 1개
V2_E_ZONES = [
    _z("LOWER_FRONT", "WHITE", 94.0, 4, calc=2.0),  _z("LOWER_FRONT", "UV", 97.0, 4, plq=1.0),
    _z("OUTER_CENTER", "WHITE", None, 1, calc=None, valid=False),
]
V2_E_TEETH = [_t(41, "CALCULUS", 8.0), _t(31, "PLAQUE", 2.0)]
V2_E_PREV = [[41, "CALCULUS"], [36, "PLAQUE"]]

# F. 악화: 치석 유지 + 치태 증가, Spring totalScore 와 재계산이 어긋나는 경우(경고 로그 확인용)
V2_F_ZONES = [
    _z("LOWER_FRONT", "WHITE", 85.0, 5, calc=5.0),       _z("LOWER_FRONT", "UV", 70.0, 5, plq=10.0),
    _z("LOWER_LEFT_MOLAR", "WHITE", 80.0, 3, calc=6.7),  _z("LOWER_LEFT_MOLAR", "UV", 64.0, 3, plq=12.0),
    _z("UPPER_FRONT", "WHITE", 97.0, 4, calc=1.0),       _z("UPPER_FRONT", "UV", 82.0, 4, plq=6.0),
]
V2_F_TEETH = [_t(41, "CALCULUS", 15.0), _t(31, "CALCULUS", 10.0), _t(36, "CALCULUS", 20.0),
              _t(41, "PLAQUE", 18.0), _t(31, "PLAQUE", 12.0), _t(36, "PLAQUE", 22.0), _t(37, "PLAQUE", 14.0),
              _t(11, "PLAQUE", 9.0), _t(21, "PLAQUE", 3.0)]
V2_F_PREV = [{"toothNumber": 41, "lesionType": "CALCULUS", "areaRatio": 15.0},
             {"toothNumber": 31, "lesionType": "CALCULUS", "areaRatio": 9.0},
             {"toothNumber": 36, "lesionType": "CALCULUS", "areaRatio": 19.0},
             {"toothNumber": 41, "lesionType": "PLAQUE", "areaRatio": 10.0},
             {"toothNumber": 36, "lesionType": "PLAQUE", "areaRatio": 12.0}]

V2_CASES = {
    "v2-new-calculus": {"zones": V2_A_ZONES, "teeth": V2_A_TEETH, "prevTeeth": V2_A_PREV,
                        "totalScore": 87.3, "prevTotalScore": 96.0, "capturedZoneCount": 3},
    "v2-plaque":       {"zones": V2_B_ZONES, "teeth": V2_B_TEETH, "prevTeeth": None,
                        "capturedZoneCount": 3},
    "v2-improving":    {"zones": V2_C_ZONES, "teeth": V2_C_TEETH, "prevTeeth": V2_C_PREV,
                        "prevZones": V2_C_PREV_ZONES, "capturedZoneCount": 2},
    "v2-clean":        {"zones": V2_D_ZONES, "teeth": [], "prevTeeth": V2_D_PREV,
                        "totalScore": 100.0, "prevTotalScore": 96.0, "capturedZoneCount": 7},
    "v2-low-coverage": {"zones": V2_E_ZONES, "teeth": V2_E_TEETH, "prevTeeth": V2_E_PREV,
                        "capturedZoneCount": 2},
    "v2-worsening":    {"zones": V2_F_ZONES, "teeth": V2_F_TEETH, "prevTeeth": V2_F_PREV,
                        "totalScore": 70.0, "prevTotalScore": 78.0, "capturedZoneCount": 3},
}

CASES = {
    "empty":  {"zones": ZONES_CLEAN,     "teeth": [],             "prevTeeth": None},
    "mid":    {"zones": ZONES_DIRTY,     "teeth": TEETH_MID,      "prevTeeth": None},
    "full":   {"zones": ZONES_ALL_DIRTY, "teeth": build_full(),   "prevTeeth": None},
    "stress": {"zones": ZONES_ALL_DIRTY, "teeth": build_stress(), "prevTeeth": None},
    # 호전 문구 확인용 — 지난번엔 있었는데 이번엔 전부 사라진 케이스
    "resolved": {"zones": ZONES_CLEAN, "teeth": [],
                 "prevTeeth": [[16, "CALCULUS"], [31, "CALCULUS"], [11, "PLAQUE"]]},
    # 이전 측정이 있고 병변도 남아 있는 케이스 — 프롬프트에 resolved_note 가 붙는다.
    #   16,17,31,11 → PERSISTED / 13,23,35 → 이번에 사라짐(resolved) / 나머지 → NEW
    "mid-prev": {"zones": ZONES_DIRTY, "teeth": TEETH_MID,
                 "prevTeeth": [[16, "CALCULUS"], [17, "CALCULUS"], [31, "CALCULUS"],
                               [11, "PLAQUE"],
                               [13, "CALCULUS"], [23, "PLAQUE"], [35, "CALCULUS"]]},
    # 이전 측정이 있지만 사라진 게 하나도 없는 케이스 — resolved_note 가 안 붙는다.
    "mid-prev-nogain": {"zones": ZONES_DIRTY, "teeth": TEETH_MID,
                        "prevTeeth": [[16, "CALCULUS"], [11, "PLAQUE"]]},
    **V2_CASES,
}


def _print_v2(data: dict):
    r = data.get("result", data)
    sc = r.get("score") or {}
    print(f"\n[점수] {sc.get('total')} (지난 {sc.get('prevTotal')}, Δ{sc.get('delta')})  "
          f"감점 치석 {sc.get('calculusDeduction')} / 치태 {sc.get('plaqueDeduction')}  "
          f"상한 칫솔질 {sc.get('ceilingByBrushing')} / 스케일링 {sc.get('ceilingAfterScaling')}  "
          f"촬영 {sc.get('capturedZoneCount')}/{sc.get('totalZoneCount')}  reason={sc.get('primaryReason')} {sc.get('flags')}")
    if sc.get("scoreMismatch"):
        print(f"  ⚠ 점수 불일치: {sc['scoreMismatch']}")
    print(f"[해석 · {r.get('interpretationSource')}]")
    for x in r.get("interpretation", []):
        print(f"  - {x}")
    print("[할 일]")
    for a in r.get("actions", []):
        print(f"  {a['title']}  [{a['tag']}]  — {a['detail']}")


def run_case(base: str, case: str, dry: bool, timeout: int, v2_only: bool = False):
    payload = CASES[case]
    endpoint = "/dev/dry-run" if dry else "/api/llm/report/generate"
    n = len(payload["teeth"])
    print("=" * 70)
    print(f"[case={case}] teeth={n}행  endpoint={endpoint}  [{datetime.now():%H:%M:%S}]")
    start = time.time()
    try:
        r = requests.post(base + endpoint, json=payload, timeout=timeout)
        print(f"  응답 {r.status_code} [{datetime.now():%H:%M:%S}]")
        data = r.json()
        if dry:
            print("\n[집계]", json.dumps(data["session"], ensure_ascii=False))
            print("\n[점수]", json.dumps(data["score"], ensure_ascii=False))
            print("\n[할 일]")
            for a in data["actions"]:
                print(f"  {a['title']}  [{a['tag']}]  — {a['detail']}")
            print("\n[폴백 해석]")
            for x in data["fallback_interpretation"]:
                print(f"  - {x}")
            print("\n[LLM 입력]", json.dumps(data["llm_input"], ensure_ascii=False))
            if not v2_only:
                print("\n[구형 관리법]", data["legacy"]["picked_titles"])
                print("\n[해석 프롬프트]\n" + data["interp_prompt"])
        elif v2_only:
            _print_v2(data)
        else:
            print(json.dumps(data, ensure_ascii=False, indent=2))
    except requests.exceptions.Timeout:
        print(f"  X 타임아웃 [{datetime.now():%H:%M:%S}]")
    except Exception as e:
        print(f"  X 에러 [{datetime.now():%H:%M:%S}]: {e}")
    finally:
        print(f"\n  소요 시간: {time.time() - start:.1f}초\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="http://localhost:8001")
    ap.add_argument("--case", default="mid",
                    choices=list(CASES) + ["all", "v2"])
    ap.add_argument("--dry", action="store_true")
    ap.add_argument("--v2", action="store_true", help="score / interpretation / actions 만 출력")
    ap.add_argument("--timeout", type=int, default=1200)
    # 예전 호출 방식(python test_client.py dry) 호환
    if len(sys.argv) > 1 and sys.argv[1] == "dry":
        sys.argv[1] = "--dry"
    args = ap.parse_args()

    if args.case == "all":
        cases = list(CASES)
    elif args.case == "v2":
        cases = list(V2_CASES)
    else:
        cases = [args.case]
    for c in cases:
        run_case(args.base.rstrip("/"), c, args.dry, args.timeout, v2_only=args.v2 or c.startswith("v2-"))


if __name__ == "__main__":
    main()
