# app.py
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
import tempfile, os, time
import uvicorn

from ocr_pipeline import review_document

app = FastAPI(title="OCR Review Service")

LAST_WARMUP = {"status": "cold", "ts": None, "msg": None}

def _set_warmup(status: str, msg: str | None = None):
    LAST_WARMUP.update({"status": status, "ts": int(time.time()), "msg": msg})

@app.post("/ocr/review")
async def ocr_review(file: UploadFile = File(...)):
    tmp_path = None
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=f"_{file.filename}") as tmp:
            tmp.write(await file.read())
            tmp_path = tmp.name
        return review_document(tmp_path)
    except Exception as e:
        import traceback; traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"OCR 실패: {e}")
    finally:
        if tmp_path and os.path.exists(tmp_path):
            try: os.remove(tmp_path)
            except Exception: pass

@app.get("/healthz")
def healthz():
    return {"status": "ok"}

@app.get("/readyz")
def readyz():
    return {"status": LAST_WARMUP["status"], "last_warm_ts": LAST_WARMUP["ts"], "msg": LAST_WARMUP["msg"]}

@app.post("/warmup")
def warmup():
    sample = os.getenv("OCR_WARMUP_SAMPLE", "").strip()
    if not sample:
        _set_warmup("skipped", "no sample provided (set OCR_WARMUP_SAMPLE)")
        return JSONResponse({"status": "skipped", "msg": LAST_WARMUP["msg"]})
    if not os.path.exists(sample):
        _set_warmup("error", f"sample not found: {sample}")
        return JSONResponse(status_code=500, content={"status": "error", "msg": LAST_WARMUP["msg"]})
    try:
        _set_warmup("warming", f"warming with sample={os.path.basename(sample)}")
        _ = review_document(sample)
        _set_warmup("warm", "ok")
        return {"status": "warm"}
    except Exception as e:
        _set_warmup("error", f"{type(e).__name__}: {e}")
        return JSONResponse(status_code=500, content={"status": "error", "msg": LAST_WARMUP["msg"]})

@app.on_event("startup")
def auto_warmup_on_start():
    if os.getenv("OCR_AUTOWARMUP", "false").lower() == "true":
        try:
            warmup()
        except Exception:
            pass

if __name__ == "__main__":
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)
