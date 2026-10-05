import io
import re
import os
from typing import Tuple, List

def extract_text_from_file(file_bytes: bytes, filename: str) -> Tuple[str, str]:
    """
    Extracts plain text from various file formats:
    - PDF (.pdf)
    - Images (.png, .jpg, .jpeg, .webp, .bmp, .tiff)
    - Word (.docx)
    - Text (.txt, .md, .rtf)
    
    Returns:
        (extracted_text, extraction_method)
    """
    ext = os.path.splitext(filename.lower())[1]
    
    # 1. Plain Text
    if ext in [".txt", ".md", ".rtf", ".csv"]:
        try:
            return file_bytes.decode("utf-8"), "Plain Text Decoder"
        except UnicodeDecodeError:
            return file_bytes.decode("latin-1", errors="ignore"), "Latin-1 Text Decoder"
            
    # 2. PDF Documents
    elif ext == ".pdf":
        text = ""
        # Try pdfplumber first
        try:
            import pdfplumber
            with pdfplumber.open(io.BytesIO(file_bytes)) as pdf:
                for page in pdf.pages:
                    page_text = page.extract_text()
                    if page_text:
                        text += page_text + "\n\n"
            if len(text.strip()) > 50:
                return text.strip(), "PDF Direct Parser (pdfplumber)"
        except Exception:
            pass
            
        # Fallback to pypdf
        try:
            import pypdf
            reader = pypdf.PdfReader(io.BytesIO(file_bytes))
            for page in reader.pages:
                page_text = page.extract_text()
                if page_text:
                    text += page_text + "\n\n"
            if len(text.strip()) > 50:
                return text.strip(), "PDF Direct Parser (pypdf)"
        except Exception:
            pass
            
        # OCR Fallback for scanned PDFs
        try:
            import pdf2image
            import pytesseract
            images = pdf2image.convert_from_bytes(file_bytes)
            ocr_text = ""
            for img in images:
                ocr_text += pytesseract.image_to_string(img) + "\n\n"
            if len(ocr_text.strip()) > 20:
                return ocr_text.strip(), "PDF OCR Engine (Tesseract)"
        except Exception:
            pass
            
        return text.strip() if text.strip() else "Could not extract text from this PDF. It may be password-protected or empty.", "PDF Fallback"

    # 3. Word Documents (.docx)
    elif ext in [".docx", ".doc"]:
        try:
            import docx
            doc = docx.Document(io.BytesIO(file_bytes))
            full_text = []
            for para in doc.paragraphs:
                if para.text.strip():
                    full_text.append(para.text)
            for table in doc.tables:
                for row in table.rows:
                    row_text = [cell.text.strip() for cell in row.cells if cell.text.strip()]
                    if row_text:
                        full_text.append(" | ".join(row_text))
            return "\n\n".join(full_text), "Word Document Parser (python-docx)"
        except Exception as e:
            return f"Error reading Word document: {str(e)}", "DOCX Parser Error"

    # 4. Images (OCR)
    elif ext in [".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tiff"]:
        try:
            from PIL import Image
            import pytesseract
            img = Image.open(io.BytesIO(file_bytes)).convert("RGB")
            text = pytesseract.image_to_string(img)
            if len(text.strip()) > 10:
                return text.strip(), "Image OCR Engine (pytesseract)"
        except Exception:
            pass
            
        # Fallback to easyocr if available
        try:
            import easyocr
            reader = easyocr.Reader(['en'])
            results = reader.readtext(file_bytes, detail=0)
            text = "\n".join(results)
            if len(text.strip()) > 10:
                return text.strip(), "Image OCR Engine (EasyOCR)"
        except Exception:
            pass
            
        return "Image uploaded. Please ensure 'pytesseract' is configured with Tesseract OCR binary to extract text from images.", "OCR Module Notice"

    else:
        # Generic attempt
        try:
            return file_bytes.decode("utf-8", errors="ignore"), "Generic Text Decoder"
        except Exception as e:
            return f"Unsupported file extension: {ext}", "Unknown File Format"

