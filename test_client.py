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
}


def run_case(base: str, case: str, dry: bool, timeout: int):
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
            print("\n[집계]", json.dumps(
                {k: v for k, v in data["session"].items() if k != "issues"},
                ensure_ascii=False))
            print("\n[그룹]")
            for g in data["groups"]:
                print(f"  - {g['lesion_kr']} / {g['regions_text']} / 치아 {g['tooth_cnt']}개 / {g['riskLevel']}")
            print("\n[선택된 관리법]", data["picked_titles"])
            print("\n[프롬프트]\n" + data["prompt"])
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
                    choices=list(CASES) + ["all"])
    ap.add_argument("--dry", action="store_true")
    ap.add_argument("--timeout", type=int, default=1200)
    # 예전 호출 방식(python test_client.py dry) 호환
    if len(sys.argv) > 1 and sys.argv[1] == "dry":
        sys.argv[1] = "--dry"
    args = ap.parse_args()

    cases = list(CASES) if args.case == "all" else [args.case]
    for c in cases:
        run_case(args.base.rstrip("/"), c, args.dry, args.timeout)


if __name__ == "__main__":
    main()
