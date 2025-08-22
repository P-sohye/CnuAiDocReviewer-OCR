from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
import tempfile, os, time
import uvicorn

from ocr_pipeline import review_document  # OCR 핵심 로직

app = FastAPI(title="OCR Review Service")

# ───────── 상태/워밍업 메모리 상태 ─────────
LAST_WARMUP = {
    "status": "cold",          # cold | warming | warm | error | skipped
    "ts": None,                # epoch seconds
    "msg": None,               # optional detail
}

def _set_warmup(status: str, msg: str | None = None):
    LAST_WARMUP["status"] = status
    LAST_WARMUP["ts"] = int(time.time())
    LAST_WARMUP["msg"] = msg

# ───────── OCR 리뷰 엔드포인트 ─────────
@app.post("/ocr/review")
async def ocr_review(file: UploadFile = File(...)):
    """
    PDF/HWP 등 문서를 업로드 받아 OCR 검사 후 PASS / FAIL 결과 반환
    """
    tmp_path = None
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=f"_{file.filename}") as tmp:
            tmp.write(await file.read())
            tmp_path = tmp.name

        # OCR 분석 실행
        return review_document(tmp_path)
    except Exception as e:
        import traceback; traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"OCR 실패: {e}")
    finally:
        if tmp_path and os.path.exists(tmp_path):
            try:
                os.remove(tmp_path)
            except Exception:
                pass

# ───────── 헬스/레디니스 ─────────
@app.get("/healthz")
def healthz():
    # 단순 liveness
    return {"status": "ok"}

@app.get("/readyz")
def readyz():
    # 최근 워밍업 상태 반환(레디니스 용도)
    return {
        "status": LAST_WARMUP["status"],
        "last_warm_ts": LAST_WARMUP["ts"],
        "msg": LAST_WARMUP["msg"],
    }

# ───────── 워밍업 ─────────
@app.post("/warmup")
def warmup():
    """
    모델/엔진 초기 로딩을 미리 수행.
    - 환경변수 OCR_WARMUP_SAMPLE 로 샘플 파일 경로 지정 권장.
    - 샘플이 없으면 스킵(상태는 skipped).
    """
    sample = os.getenv("OCR_WARMUP_SAMPLE", "").strip()
    if not sample:
        _set_warmup("skipped", "no sample provided (set OCR_WARMUP_SAMPLE)")
        return JSONResponse({"status": "skipped", "msg": LAST_WARMUP["msg"]})

    if not os.path.exists(sample):
        _set_warmup("error", f"sample not found: {sample}")
        return JSONResponse(status_code=500, content={"status": "error", "msg": LAST_WARMUP["msg"]})

    try:
        _set_warmup("warming", f"warming with sample={os.path.basename(sample)}")
        # 실제 한 번 실행 (결과는 버리되, 로딩이 목적)
        _ = review_document(sample)
        _set_warmup("warm", "ok")
        return {"status": "warm"}
    except Exception as e:
        _set_warmup("error", f"{type(e).__name__}: {e}")
        return JSONResponse(status_code=500, content={"status": "error", "msg": LAST_WARMUP["msg"]})

# ───────── 애플리케이션 스타트업 훅 ─────────
@app.on_event("startup")
def auto_warmup_on_start():
    if os.getenv("OCR_AUTOWARMUP", "false").lower() == "true":
        try:
            warmup()
        except Exception as e:
            # warmup 내부에서 상태 셋팅함. 여기서는 삼켜서 부팅은 계속.
            pass

if __name__ == "__main__":
    # uvicorn 서버 실행 (localhost:8000 기본 포트)
    # 운영에서는 reload=False 권장
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)
