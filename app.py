# app.py
from fastapi import FastAPI, File, UploadFile
import tempfile
import uvicorn

from ocr_pipeline import review_document  # OCR 핵심 로직

app = FastAPI(title="OCR Review Service")

@app.post("/ocr/review")
async def ocr_review(file: UploadFile = File(...)):
    """
    PDF/HWP 등 문서를 업로드 받아 OCR 검사 후 PASS / FAIL 결과 반환
    """
    # 업로드된 파일을 임시 파일로 저장
    with tempfile.NamedTemporaryFile(delete=False, suffix=f"_{file.filename}") as tmp:
        tmp.write(await file.read())
        tmp_path = tmp.name

    # OCR 분석 실행
    result = review_document(tmp_path)

    return {"status": result}


if __name__ == "__main__":
    # uvicorn 서버 실행 (localhost:8000 기본 포트)
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)
