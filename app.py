# app.py
from fastapi import FastAPI, File, UploadFile, HTTPException
import tempfile, os
import uvicorn

from ocr_pipeline import review_document  # OCR 핵심 로직

app = FastAPI(title="OCR Review Service")

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

if __name__ == "__main__":
    # uvicorn 서버 실행 (localhost:8000 기본 포트)
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)
