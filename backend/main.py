import os
import io
import csv
from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, StreamingResponse, JSONResponse, FileResponse
from pydantic import BaseModel
from typing import Optional, List

from extractor import extract_text_from_file
from segmenter import segment_contract_into_clauses
from classifier import ContractRiskAnalyzer

app = FastAPI(
    title="LegalBERT Contract Risk Intelligence API",
    description="Real-world multi-format contract risk assessment powered by sayan-7/legalbert-contract-risk-classifier",
    version="1.0.0"
)

# Enable CORS for React Frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize Analyzer on Startup
analyzer = ContractRiskAnalyzer()

class RawTextRequest(BaseModel):
    text: str
    document_name: Optional[str] = "Pasted Contract Text"

@app.get("/api/health")
def health_check():
    return {
        "status": "healthy",
        "model": "sayan-7/legalbert-contract-risk-classifier",
        "device": str(analyzer.device)
    }

@app.post("/api/analyze-file")
async def analyze_file(file: UploadFile = File(...)):
    """Accepts PDF, PNG, JPG, DOCX, TXT, extracts text and performs complete risk audit."""
    try:
        file_bytes = await file.read()
        if len(file_bytes) == 0:
            raise HTTPException(status_code=400, detail="Uploaded file is empty.")

        # 1. Multi-format Text Extraction
        extracted_text, method = extract_text_from_file(file_bytes, file.filename)
        if not extracted_text or len(extracted_text.strip()) < 10:
            raise HTTPException(
                status_code=400, 
                detail=f"Could not extract sufficient text from {file.filename} using {method}. Please ensure the file contains readable text."
            )

        # 2. Intelligent Clause Segmentation
        clauses = segment_contract_into_clauses(extracted_text)
        if not clauses:
            raise HTTPException(status_code=400, detail="No valid contract clauses could be segmented from the text.")

        # 3. Model Inference & Risk Scoring
        analysis_result = analyzer.analyze_clauses(clauses)

        return {
            "success": True,
            "filename": file.filename,
            "extraction_method": method,
            "raw_text_length": len(extracted_text),
            "raw_text_preview": extracted_text[:500] + ("..." if len(extracted_text) > 500 else ""),
            "analysis": analysis_result
        }
    except Exception as e:
        return JSONResponse(status_code=500, content={"success": False, "error": str(e)})

@app.post("/api/analyze-text")
def analyze_text(request: RawTextRequest):
    """Analyzes raw pasted contract text."""
    try:
        if not request.text or len(request.text.strip()) < 10:
            raise HTTPException(status_code=400, detail="Text is too short for analysis.")

        clauses = segment_contract_into_clauses(request.text)
        analysis_result = analyzer.analyze_clauses(clauses)

        return {
            "success": True,
            "filename": request.document_name,
            "extraction_method": "Direct Text Input",
            "raw_text_length": len(request.text),
            "analysis": analysis_result
        }
    except Exception as e:
        return JSONResponse(status_code=500, content={"success": False, "error": str(e)})

@app.post("/api/export-csv")
def export_csv(clauses: List[dict]):
    """Generates and downloads a CSV report of the analyzed clauses."""
    output = io.StringIO()
    writer = csv.writer(output)
    writer.writerow(["ID", "Heading", "Risk Level", "Confidence (%)", "Provision Text", "Actionable Redline Fix"])

    for c in clauses:
        writer.writerow([
            c.get("id", ""),
            c.get("heading", ""),
            c.get("risk_label", ""),
            c.get("confidence", ""),
            c.get("text", ""),
            c.get("recommendation", "")
        ])

    output.seek(0)
    return StreamingResponse(
        io.BytesIO(output.getvalue().encode("utf-8")),
        media_type="text/csv",
        headers={"Content-Disposition": "attachment; filename=contract_risk_audit.csv"}
    )

# Serve Frontend HTML directly at root
frontend_html = os.path.join(os.path.dirname(__file__), "..", "frontend", "index.html")
@app.get("/", response_class=HTMLResponse)
def serve_root():
    if os.path.exists(frontend_html):
        with open(frontend_html, "r", encoding="utf-8") as f:
            return f.read()
    return "<h1>LegalBERT API is running. Frontend not found.</h1>"

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)

