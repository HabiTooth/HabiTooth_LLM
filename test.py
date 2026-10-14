# test.py — HabiTooth LLM 리포트 서버 (포트 8001, 이미지 분석 서버와 별도 프로세스)
#           Bedrock 호출과 엔드포인트만 맡는다. 로컬 모델 로딩 없음.
#
# 배포 모델: qwen.qwen3-32b-v1:0 (Amazon Bedrock, us-west-2)
#   - boto3 bedrock-runtime 의 Converse API 로 호출한다.
#   - Qwen3 는 thinking 모드가 기본 ON 이다. Bedrock 은 chat_template 을 서버에서 적용하므로
#     enable_thinking=False 를 직접 줄 수 없다. 대신 Qwen3 공식 소프트 스위치인
#     "/no_think" 를 user 프롬프트 끝에 붙이고, 그래도 새는 <think> 블록은 extract_json 이 떼어낸다.
#     Converse 응답에 reasoningContent 블록이 섞여 오면 text 블록만 모은다.
#
# 실행:  uvicorn test:app --host 0.0.0.0 --port 8001
#
# 환경변수
#   AWS_ACCESS_KEY_ID / AWS_SECRET_ACCESS_KEY (또는 IAM 역할·프로파일) — 표준 boto3 자격증명
#   HABITOOTH_MODEL        기본 qwen.qwen3-32b-v1:0
#   HABITOOTH_REGION       기본 us-west-2
#   HABITOOTH_MAXNEW       기본 512 (maxTokens)
#   HABITOOTH_RETRY        기본 4 (마지막 시도는 temperature 0)
#   HABITOOTH_CONCURRENCY  기본 4 (동시에 Bedrock 에 보낼 리포트 수)
#   HABITOOTH_NO_THINK     기본 1 (프롬프트 끝에 /no_think 추가)
#   HABITOOTH_TIMEOUT      기본 120 (초, Bedrock read timeout)
#
# 프롬프트·카탈로그·집계 규칙은 report_logic.py 에 있고,
# 고친 뒤 POST /dev/reload 만 치면 서버 재기동 없이 반영된다.
import importlib
import json
import os
import re
import threading
import time
from contextlib import asynccontextmanager
from typing import List, Optional

import boto3
from botocore.config import Config
from botocore.exceptions import BotoCoreError, ClientError
from fastapi import FastAPI, HTTPException
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, field_validator
from starlette.requests import Request

import report_logic  # 반드시 모듈째로 import (reload 대상이라 from-import 하면 안 됨)

MODEL_ID = os.getenv("HABITOOTH_MODEL", "qwen.qwen3-32b-v1:0")
REGION = os.getenv("HABITOOTH_REGION", "us-west-2")
MAX_NEW_TOKENS = int(os.getenv("HABITOOTH_MAXNEW", "512"))
MAX_RETRY = int(os.getenv("HABITOOTH_RETRY", "4"))
CONCURRENCY = int(os.getenv("HABITOOTH_CONCURRENCY", "4"))
NO_THINK = os.getenv("HABITOOTH_NO_THINK", "1") == "1"
READ_TIMEOUT = int(os.getenv("HABITOOTH_TIMEOUT", "120"))

GEN_SEMAPHORE = threading.Semaphore(CONCURRENCY)
resources = {}


# ══════════════════════════════════════════════════════════════
# Bedrock 클라이언트
# ══════════════════════════════════════════════════════════════
def make_client():
    cfg = Config(
        region_name=REGION,
        read_timeout=READ_TIMEOUT,
        connect_timeout=10,
        retries={"max_attempts": 3, "mode": "adaptive"},   # Throttling 시 백오프
    )
    return boto3.client("bedrock-runtime", config=cfg)


def build_messages(system: str, user: str):
    if NO_THINK:
        user = user.rstrip() + "\n/no_think"
    return (
        [{"text": system}],
        [{"role": "user", "content": [{"text": user}]}],
    )


def converse(system: str, user: str, temperature: float, top_p: float):
    """Converse 한 번 호출. 반환: (text, usage, stop_reason)"""
    client = resources["client"]
    sys_blocks, messages = build_messages(system, user)
    resp = client.converse(
        modelId=MODEL_ID,
        system=sys_blocks,
        messages=messages,
        inferenceConfig={
            "maxTokens": MAX_NEW_TOKENS,
            "temperature": temperature,
            "topP": top_p,
        },
    )
    blocks = resp.get("output", {}).get("message", {}).get("content", [])
    # reasoningContent(thinking) 블록은 버리고 text 블록만 모은다
    text = "".join(b.get("text", "") for b in blocks if "text" in b).strip()
    return text, resp.get("usage", {}), resp.get("stopReason")


def ping():
    """기동 직후 자격증명·모델 접근 권한을 한 번 확인한다. 실패해도 기동은 막지 않는다."""
    try:
        t0 = time.time()
        text, usage, _ = converse("짧게 답하세요.", "OK 라고만 답하세요.", 0.0, 1.0)
        print(f"[LLM SERVER] Bedrock 연결 확인 / {time.time()-t0:.1f}초 / {usage} / {text[:40]!r}")
        resources["ping_ok"] = True
    except Exception as e:
        print(f"[LLM SERVER] Bedrock 연결 확인 실패(무시하고 계속): {e}")
        resources["ping_ok"] = False


@asynccontextmanager
async def lifespan(app: FastAPI):
    print(f"[LLM SERVER] Bedrock {MODEL_ID} @ {REGION} 클라이언트 생성")
    resources["client"] = make_client()
    ping()
    yield
    resources.clear()
    print("[LLM SERVER] 종료")


app = FastAPI(title="HabiTooth LLM Report Server (Bedrock)", lifespan=lifespan)
app.add_middleware(
    CORSMiddleware, allow_origins=["*"], allow_credentials=True,
    allow_methods=["*"], allow_headers=["*"],
)


@app.exception_handler(RequestValidationError)
async def validation_handler(request: Request, exc: RequestValidationError):
    print(f"[422] {exc.errors()}")
    print(f"[422] body={exc.body}")
    return JSONResponse(status_code=422, content={"detail": exc.errors()})


# ══════════════════════════════════════════════════════════════
# 요청/응답 스키마 — Spring 이 보내는 형식 (변경 없음)
# ══════════════════════════════════════════════════════════════
class ZoneIn(BaseModel):
    viewType: str
    zoneScore: Optional[float] = 100.0
    zoneValid: bool = True
    detectedToothCount: int = 0
    totalCalculusRatio: Optional[float] = 0.0
    totalPlaqueRatio: Optional[float] = 0.0

    @field_validator("zoneScore", "totalCalculusRatio", "totalPlaqueRatio", mode="before")
    @classmethod
    def null_to_default(cls, v, info):
        if v is None:
            return 100.0 if info.field_name == "zoneScore" else 0.0
        return v


class ToothAreaIn(BaseModel):
    toothNumber: int
    lesionType: str
    areaRatio: Optional[float] = 0.0
    toothScore: Optional[float] = 100.0
    riskLevel: str = "MEDIUM"

    @field_validator("areaRatio", "toothScore", mode="before")
    @classmethod
    def null_to_default(cls, v, info):
        if v is None:
            return 0.0 if info.field_name == "areaRatio" else 100.0
        return v


class ReportRequest(BaseModel):
    zones: Optional[List[ZoneIn]] = None
    teeth: List[ToothAreaIn] = []
    prevTeeth: Optional[List[List]] = None

    @field_validator("teeth", mode="before")
    @classmethod
    def none_to_empty(cls, v):
        return v or []


# ══════════════════════════════════════════════════════════════
# 생성
# ══════════════════════════════════════════════════════════════
_FENCE = re.compile(r"```(?:json)?", re.IGNORECASE)
_THINK = re.compile(r"<think>.*?</think>", re.DOTALL | re.IGNORECASE)


def extract_json(text: str):
    """모델이 코드블록으로 감싸거나 뒤에 설명을 붙이는 경우를 흡수한다.
    /no_think 를 줘도 빈 <think></think> 가 앞에 붙는 경우가 있어 떼어낸다."""
    text = _THINK.sub("", text)
    text = text.split("</think>")[-1]      # 닫히지 않은 think 블록
    text = _FENCE.sub("", text).strip()
    start = text.find("{")
    if start < 0:
        return None
    depth = 0
    for i in range(start, len(text)):
        if text[i] == "{":
            depth += 1
        elif text[i] == "}":
            depth -= 1
            if depth == 0:
                chunk = text[start:i + 1]
                try:
                    return json.loads(chunk)
                except json.JSONDecodeError:
                    fixed = re.sub(r",\s*([\]}])", r"\1", chunk)
                    try:
                        return json.loads(fixed)
                    except json.JSONDecodeError:
                        return None
    return None


def run_generation(s: dict, picked: list, max_retry: int = MAX_RETRY):
    user_prompt = report_logic.build_report_prompt(s, picked)
    last = None
    for i in range(max_retry):
        greedy = (i == max_retry - 1)      # 마지막 시도는 결정적으로
        temperature, top_p = (0.0, 1.0) if greedy else (0.45, 0.85)
        t0 = time.time()
        print(f"[LLM] 시도 {i+1}/{max_retry} 생성 시작... (temp={temperature})")
        try:
            text, usage, stop = converse(report_logic.SYSTEM_PROMPT, user_prompt, temperature, top_p)
        except ClientError as e:
            code = e.response.get("Error", {}).get("Code", "")
            print(f"[LLM] Bedrock 오류 {code}: {e}")
            if code in ("ThrottlingException", "ServiceUnavailableException", "ModelNotReadyException"):
                raise HTTPException(status_code=503, detail=f"Bedrock 일시 오류: {code}")
            if code in ("AccessDeniedException", "ValidationException", "ResourceNotFoundException"):
                raise HTTPException(status_code=502, detail=f"Bedrock 설정 오류: {code} — {e}")
            continue
        except BotoCoreError as e:
            print(f"[LLM] Bedrock 연결 오류: {e}")
            continue

        dt = time.time() - t0
        out_tok = usage.get("outputTokens", 0)
        print(f"[LLM] in={usage.get('inputTokens')} out={out_tok} / {dt:.1f}초 "
              f"({out_tok/max(dt, 1e-6):.1f} tok/s) stop={stop}")

        llm_out = extract_json(text)
        if llm_out is None:
            print(f"[LLM] JSON 파싱 실패, 재시도 | 원문 앞부분: {text[:120]!r}")
            continue

        report, needs_retry = report_logic.assemble_report(s, llm_out, picked)
        last = report
        if not needs_retry:
            break
        print("[LLM] 검증 실패, 재시도")

    if last is not None:
        last["disclaimer"] = report_logic.DISCLAIMER
    return last


# ══════════════════════════════════════════════════════════════
# 엔드포인트 — Spring 이 호출
# ══════════════════════════════════════════════════════════════
@app.post("/api/llm/report/generate")
def generate_llm_report(req: ReportRequest):
    if not GEN_SEMAPHORE.acquire(blocking=False):
        raise HTTPException(
            status_code=503,
            detail="리포트 생성 요청이 많습니다. 잠시 후 다시 시도해 주세요.",
        )
    try:
        zones = [z.model_dump() for z in (req.zones or [])]
        teeth = [t.model_dump() for t in (req.teeth or [])]
        prev_set = {(p[0], p[1]) for p in req.prevTeeth} if req.prevTeeth else None

        s = report_logic.aggregate_session(teeth, zones, prev_set)
        picked = report_logic.select_management(s, k=3)
        print(f"[LLM] 병변 {s['issue_cnt']}건 / 관리법 {picked}")

        if not s["has_issue"]:
            report = report_logic.build_healthy_report(s, picked)
            return {"isSuccess": True, "result": report}

        report = run_generation(s, picked)
        if report is None:
            raise HTTPException(
                status_code=502,
                detail=f"LLM 리포트 생성 실패 (재시도 {MAX_RETRY}회 초과)",
            )
        return {"isSuccess": True, "result": report}
    finally:
        GEN_SEMAPHORE.release()


@app.post("/dev/reload")
def dev_reload():
    """report_logic.py 만 다시 읽는다."""
    importlib.reload(report_logic)
    return {"reloaded": "report_logic",
            "logic_version": getattr(report_logic, "LOGIC_VERSION", "(없음 — 옛 파일)"),
            "catalog_size": len(report_logic.MANAGEMENT_CATALOG)}


@app.post("/dev/dry-run")
def dev_dry_run(req: ReportRequest):
    """모델을 호출하지 않고 집계·관리법 선택·프롬프트만 확인한다."""
    zones = [z.model_dump() for z in (req.zones or [])]
    teeth = [t.model_dump() for t in req.teeth]
    prev_set = {(p[0], p[1]) for p in req.prevTeeth} if req.prevTeeth else None
    s = report_logic.aggregate_session(teeth, zones, prev_set)
    picked = report_logic.select_management(s, k=3)
    prompt = report_logic.build_report_prompt(s, picked)
    sys_blocks, messages = build_messages(report_logic.SYSTEM_PROMPT, prompt)
    return {
        "session": s,
        "groups": report_logic.group_by_lesion(s),
        "picked": picked,
        "picked_titles": [report_logic.MANAGEMENT_CATALOG[k]["title"] for k in picked],
        "prompt": prompt,
        "chat_text": {"system": sys_blocks, "messages": messages},   # Bedrock 에 보내는 실제 페이로드
        "management": report_logic.build_management(s, picked),
    }


@app.get("/dev/phrase-stats")
def dev_phrase_stats():
    return report_logic.phrase_fix_stats()


@app.post("/dev/phrase-stats/reset")
def dev_phrase_stats_reset():
    report_logic.reset_phrase_stats()
    return {"reset": True}


@app.get("/health")
def health():
    return {"status": "ok", "backend": "bedrock", "model_id": MODEL_ID, "region": REGION,
            "ping_ok": resources.get("ping_ok"),
            "logic_version": getattr(report_logic, "LOGIC_VERSION", "(없음 — 옛 파일)")}


@app.get("/debug")
def debug():
    return {
        "model_id": MODEL_ID,
        "region": REGION,
        "max_tokens": MAX_NEW_TOKENS,
        "max_retry": MAX_RETRY,
        "concurrency": CONCURRENCY,
        "no_think": NO_THINK,
        "read_timeout": READ_TIMEOUT,
        "boto3": boto3.__version__,
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001)
