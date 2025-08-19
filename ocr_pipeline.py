# ocr_pipeline.py
# -*- coding: utf-8 -*-

import os
import re
import time
import tempfile
import numpy as np
import cv2
import pdfplumber
from pdf2image import convert_from_path
from paddleocr import PaddleOCR
from PyPDF2 import PdfReader

POPPLER_BIN = r"C:\poppler-25.07.0\Library\bin"

os.environ["PATH"] = POPPLER_BIN + os.pathsep + os.environ.get("PATH", "")

# ===== 전역 상수 =====
DPI = 200
MIN_COUNT = 300  # 자소서 섹션 최소 길이 기준(예시)

# ===== OCR 엔진 (서버 기동 시 1회 초기화) =====
# Colab과 달리 운영 서버에서는 서버 시작 시 한 번만 올려두고 재사용하는 것이 중요합니다.
ocr = PaddleOCR(lang='korean', use_textline_orientation=True, device='cpu')


# ===== 헬퍼 함수들 =====
def extract_ocr_items(img):
    """
    PaddleOCR.predict() 결과를 (text, confidence, bbox)의 리스트로 통일
    bbox는 [ [x1,y1], [x2,y2], [x3,y3], [x4,y4] ] 꼴
    """
    raw = ocr.predict(img)
    page = raw[0] if raw and raw[0] is not None else []
    results = []
    for line in page:
        # line = [bbox, [text, conf]]
        if isinstance(line, list) and len(line) == 2 and isinstance(line[1], (list, tuple)) and len(line[1]) == 2:
            text, conf = line[1][0], line[1][1]
            bbox = line[0]
            results.append((text, conf, bbox))
    return results


def count_sections(text):
    """자소서 섹션 길이(문자 수) 리스트 반환. 예: '1. ... 2. ...' 기반 분할."""
    return [len(p) for p in re.split(r"\d+\.\s*", text)[1:]]


def analyze_photo_characteristics(crop):
    """
    내부 픽셀 복잡도와 엣지 비율을 사용해 '사진처럼 보이는지' 스코어링 (0~1).
    표 등 단색/선분 위주 객체는 낮은 점수로 걸러짐.
    """
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    if gray.size < 500:
        return 0.0

    h, w = gray.shape
    inner_crop = gray[int(h * 0.1):int(h * 0.9), int(w * 0.1):int(w * 0.9)]
    inner_std = np.std(inner_crop) if inner_crop.size > 0 else 0

    if inner_std < 30:
        return 0.0

    edges = cv2.Canny(gray, 50, 150)
    edge_ratio = (cv2.countNonZero(edges) / edges.size) if edges.size > 0 else 0

    score = 0.0
    score += min(edge_ratio * 10, 1.0) * 0.5
    score += min(inner_std / 35.0, 1.0) * 0.5
    return score


def check_photo_at_fixed_location(img, debug=False):
    """
    고정 좌표 영역을 분석하여 사진 유무 판별 (템플릿 문서에 맞춘 좌표; mm 단위 예시)
    """
    PHOTO_AREA_MM = {'x': 15, 'y': 20, 'w': 30, 'h': 40}
    x1 = int((PHOTO_AREA_MM['x'] / 25.4) * DPI)
    y1 = int((PHOTO_AREA_MM['y'] / 25.4) * DPI)
    x2 = x1 + int((PHOTO_AREA_MM['w'] / 25.4) * DPI)
    y2 = y1 + int((PHOTO_AREA_MM['h'] / 25.4) * DPI)

    if debug:
        print(f"[PHOTO] ROI px=({x1},{y1})~({x2},{y2})")

    h, w, _ = img.shape
    crop = img[min(y1, h):min(y2, h), min(x1, w):min(x2, w)]
    if crop.size == 0:
        if debug:
            print("[PHOTO] ROI out of image bounds")
        return False, 0.0

    score = analyze_photo_characteristics(crop)
    has_photo = score > 0.5
    if debug:
        print(f"[PHOTO] has={has_photo}, score={score:.3f}")
    return has_photo, score


def detect_actual_photo(img, debug=False):
    """현재는 고정 좌표 기준으로만 검사 (필요 시 확장)"""
    return check_photo_at_fixed_location(img, debug)


def get_applicant_name_hybrid(pdf_path, ocr_page1_items, debug=False):
    """
    1페이지에서 성명 추출: pdfplumber 텍스트 → 미검출 시 OCR 텍스트 사용
    """
    try:
        with pdfplumber.open(pdf_path) as pdf:
            p1_text = pdf.pages[0].extract_text() or ''
            match = re.search(r'성\s*명\s*(?:\(한글\))?\s*([가-힣]{2,4})', p1_text)
            if match:
                name = match.group(1).strip()
                if debug:
                    print(f"[NAME] pdfplumber: {name}")
                return name
    except Exception:
        pass

    full_ocr_text = " ".join([item[0] for item in ocr_page1_items])
    match = re.search(r'성\s*명\s*(?:\(한글\))?\s*([가-힣]{2,4})', full_ocr_text)
    if match:
        name = match.group(1).strip()
        if debug:
            print(f"[NAME] OCR: {name}")
        return name

    if debug:
        print("[NAME] not found")
    return None


def check_consent_hybrid(pdf_path, ocr_page3_items, debug=False):
    """
    3페이지에서 '동의하십니까?(예)' 패턴 확인: 먼저 pdfplumber → 미검출 시 OCR 텍스트
    """
    consent_pattern = r'동의하십니까\?\s*\(\s*예\s*\)'
    try:
        with pdfplumber.open(pdf_path) as pdf:
            if len(pdf.pages) >= 3:
                p3_text = pdf.pages[2].extract_text() or ''
                if re.search(consent_pattern, re.sub(r'\s+', '', p3_text)):
                    if debug:
                        print("[CONSENT] Digital Text Match")
                    return True, "Digital Text Match"
    except Exception:
        pass

    full_ocr_text = "".join([item[0] for item in ocr_page3_items])
    if re.sub(r'\s+', '', full_ocr_text) and re.search(consent_pattern, re.sub(r'\s+', '', full_ocr_text)):
        if debug:
            print("[CONSENT] OCR Text Match")
        return True, "OCR Text Match"

    if debug:
        print("[CONSENT] Not Found")
    return False, None


# 추가: 서명 이미지 스코어러(간단판)
def analyze_signature_image(crop):
    """서명(펜 스트로크)처럼 보이는지 간단히 점수화 (0~1)."""
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (3, 3), 0)
    edges = cv2.Canny(gray, 50, 150)
    strokes = cv2.countNonZero(edges)
    density = strokes / (edges.size or 1)
    return float(min(density * 8, 1.0))


# 추가: OCR 기반 서명 근접 탐색(간단판)
def detect_signature_enhanced_ocr(page_items, applicant_name, img_shape):
    """
    OCR 텍스트에서 '성명/서명/신청인' 주변에 이름/획이 존재하면 서명이 있다고 간주(간단판).
    실제로는 좌표 기반 더 정교한 근접 탐색 필요.
    """
    if not page_items:
        return False, None

    anchor_words = ('성명', '서명', '신청인')
    has_anchor = any(any(a in t for a in anchor_words) for t, _, _ in page_items)
    name_seen = any((applicant_name and applicant_name in t) for t, _, _ in page_items)
    # 아주 러프하게: 앵커+이름이 같이 보이면 서명 있다고 가정
    return (has_anchor and name_seen), "OCR proximity"


def detect_signature_ultimate(pdf_path, page_num, img, page_items, applicant_name, debug=False):
    """
    pdfplumber(디지털 텍스트/임베디드 이미지) → 실패 시 OCR 근접 탐색 → ROI 이미지 스코어 순
    """
    if not applicant_name:
        if debug:
            print(f"[SIGN{page_num}] skip: no applicant name")
        return False, None

    try:
        with pdfplumber.open(pdf_path) as pdf:
            if len(pdf.pages) >= page_num:
                page = pdf.pages[page_num - 1]
                page_bottom = page.crop((0, page.height * 0.6, page.width, page.height))
                words = page_bottom.extract_words(x_tolerance=2, y_tolerance=3)

                anchor_box, anchor_index = None, -1
                for i, w in enumerate(words):
                    if applicant_name in w.get("text", ""):
                        ctx = "".join(x.get("text", "") for x in words[max(0, i - 3):i + 1])
                        if "성명" in ctx or "서명" in ctx:
                            anchor_box, anchor_index = w, i
                            break

                if anchor_box:
                    # 옆 단어가 (인)/(서명) 함정인지 검사
                    next_word_text = ""
                    if len(words) > anchor_index + 1:
                        nextw = words[anchor_index + 1]
                        if abs(nextw['top'] - anchor_box['top']) < 5:
                            next_word_text = nextw.get("text", "")

                    if next_word_text and next_word_text not in ["(인)", "(서명)"]:
                        if debug:
                            print(f"[SIGN{page_num}] typed signature: {next_word_text}")
                        return True, f"Typed Signature: {next_word_text}"

                    # 임베디드 이미지로 서명했는지 검사
                    for p_img in page_bottom.images:
                        if (p_img["x0"] < anchor_box["x1"] + 150 and
                            p_img["x1"] > anchor_box["x1"] and
                            p_img["top"] < anchor_box["bottom"] and
                            p_img["bottom"] > anchor_box["top"]):
                            if debug:
                                print(f"[SIGN{page_num}] embedded image signature")
                            return True, "PDF Embedded Image"

                    if next_word_text in ["(인)", "(서명)"]:
                        if debug:
                            print(f"[SIGN{page_num}] trap text found")
                        return False, f"Trap Text: {next_word_text}"

                    # 이미지 ROI 스코어
                    scale = DPI / 72.0
                    rx1 = int(anchor_box["x1"] * scale)
                    ry1 = int(anchor_box["top"] * scale) - 20
                    rx2 = rx1 + 250
                    ry2 = int(anchor_box["bottom"] * scale) + 20
                    roi = img[max(0, ry1):min(img.shape[0], ry2), max(0, rx1):min(img.shape[1], rx2)]
                    if roi.size > 0:
                        sig_score = analyze_signature_image(roi)
                        if debug:
                            print(f"[SIGN{page_num}] ROI score={sig_score:.3f}")
                        if sig_score > 0.6:
                            return True, f"Image Signature (Score: {sig_score:.2f})"

    except Exception as e:
        if debug:
            print(f"[SIGN{page_num}] pdfplumber error: {e}")

    # pdfplumber 실패 시 OCR 근접 탐색
    flag, text = detect_signature_enhanced_ocr(page_items, applicant_name, img.shape)
    if debug:
        print(f"[SIGN{page_num}] OCR proximity flag={flag}, text={text}")
    return flag, text


# ===== 메인 함수: review_document =====
def review_document(pdf_path: str, debug: bool = False) -> dict:
    """
    Colab 스크립트를 운영형으로 감싼 함수.
    입력: 저장된 PDF 파일 경로
    반환: { status: PASS|FAIL|ERROR, processing_time, details:{...}, section_counts:[...], reason? }
    """
    start = time.perf_counter()
    
    try:
        # 1) PDF → 이미지 (우선 Poppler 경로 명시)
        try:
            pil_pages = convert_from_path(pdf_path, dpi=DPI, poppler_path=POPPLER_BIN)
            imgs = [cv2.cvtColor(np.array(p), cv2.COLOR_RGB2BGR) for p in pil_pages]
        except Exception as e:
            # 🔁 폴백: Poppler 없이 PyMuPDF로 렌더링
            import fitz  # PyMuPDF
            zoom = DPI / 72.0
            mat = fitz.Matrix(zoom, zoom)
            imgs = []
            with fitz.open(pdf_path) as doc:
                for page in doc:
                    pix = page.get_pixmap(matrix=mat, alpha=False)
                    arr = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.h, pix.w, 3)
                    imgs.append(cv2.cvtColor(arr, cv2.COLOR_RGB2BGR))
            if not imgs:
                return {"status": "FAIL", "processing_time": "0.00s", "reason": "PDF 렌더링 실패(PyMuPDF)"}

        # 2) 전체 페이지 OCR 아이템
        all_page_items = [extract_ocr_items(img) for img in imgs]

        # 3) 1페이지: 이름, 사진
        applicant_name = get_applicant_name_hybrid(pdf_path, all_page_items[0], debug=debug)
        photo_ok, photo_score = detect_actual_photo(imgs[0], debug=debug)

        # 4) 3페이지: 동의 체크
        consent_ok, consent_method = (False, None)
        if len(imgs) >= 3:
            consent_ok, consent_method = check_consent_hybrid(pdf_path, all_page_items[2], debug=debug)

        # 5) 1/3페이지: 서명 확인
        sig1_ok, sig1_method = detect_signature_ultimate(
            pdf_path, 1, imgs[0], all_page_items[0], applicant_name, debug=debug
        )
        if len(imgs) >= 3:
            sig3_ok, sig3_method = detect_signature_ultimate(
                pdf_path, 3, imgs[2], all_page_items[2], applicant_name, debug=debug
            )
        else:
            sig3_ok, sig3_method = (False, None)

        # 6) 2페이지 자소서 섹션 길이
        section_lengths, section_flags = [], []
        try:
            if len(imgs) >= 2:
                reader = PdfReader(pdf_path)
                raw2 = reader.pages[1].extract_text() or ''
                section_lengths = count_sections(raw2)
                section_flags = [c >= MIN_COUNT for c in section_lengths] if section_lengths else []
        except Exception:
            # PyPDF2가 실패할 수 있으므로 실패해도 전체 실패로 두지 않음
            section_lengths, section_flags = [], []

        # 7) 최종 판정
        requirements = [
            applicant_name is not None,
            photo_ok,
            all(section_flags) if section_flags else False,
            consent_ok,
            sig1_ok,
            sig3_ok
        ]
        final_pass = all(requirements)
        status = "PASS" if final_pass else "FAIL"

        # 8) 결과 JSON
        elapsed = f"{time.perf_counter() - start:.2f}s"
        return {
            "status": status,
            "processing_time": elapsed,
            "details": {
                "지원자명": applicant_name if applicant_name else "❌ 미확인",
                "사진": "✅ 확인됨" if photo_ok else "❌ 누락",
                "사진(점수)": round(float(photo_score), 4),
                "동의서": "✅ 확인됨" if consent_ok else "❌ 누락",
                "동의서(근거)": consent_method,
                "서명1": "✅ 확인됨" if sig1_ok else "❌ 누락",
                "서명1(근거)": sig1_method,
                "서명3": "✅ 확인됨" if sig3_ok else "❌ 누락",
                "서명3(근거)": sig3_method,
                "자소서분량": "✅ 충족" if (section_flags and all(section_flags)) else "❌ 부족",
            },
            "section_counts": [
                {"label": f"섹션{i+1}", "count": cnt, "length": cnt}
                for i, cnt in enumerate(section_lengths)
            ],
        }

    except Exception as e:
        return {
            "status": "ERROR",
            "processing_time": f"{time.perf_counter() - start:.2f}s",
            "reason": str(e),
        }
