# Place spaces import at the very top of app.py
try:
    import spaces
except ImportError:
    spaces = None

import os
import io
import re
import csv
import pandas as pd
import gradio as gr
from typing import List, Dict, Any, Tuple

import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# 1. Load Model from Hugging Face
MODEL_NAME = "sayan-7/legalbert-contract-risk-classifier"
ID2LABEL = {0: 'Standard Clause', 1: 'Review Recommended', 2: 'High Risk Clause'}
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

print(f"Loading model {MODEL_NAME} on {device}...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME).to(device)
model.eval()

# 2. Text Extraction Functions
def extract_text_from_file(file_obj) -> Tuple[str, str]:
    if file_obj is None:
        return "", "No file"
    
    file_path = file_obj.name if hasattr(file_obj, 'name') else str(file_obj)
    ext = os.path.splitext(file_path.lower())[1]
    
    # Text / Markdown
    if ext in [".txt", ".md", ".rtf", ".csv"]:
        with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
            return f.read(), "Plain Text"
            
    # PDF Documents
    elif ext == ".pdf":
        text = ""
        try:
            import pdfplumber
            with pdfplumber.open(file_path) as pdf:
                for page in pdf.pages:
                    t = page.extract_text()
                    if t:
                        text += t + "\n\n"
            if len(text.strip()) > 30:
                return text.strip(), "PDF Direct Parser"
        except Exception:
            pass
            
        try:
            import pypdf
            reader = pypdf.PdfReader(file_path)
            for page in reader.pages:
                t = page.extract_text()
                if t:
                    text += t + "\n\n"
            if len(text.strip()) > 30:
                return text.strip(), "PyPDF Parser"
        except Exception:
            pass
            
        return text.strip() if text.strip() else "Could not extract text from PDF.", "PDF Fallback"

    # Word Documents (.docx)
    elif ext in [".docx", ".doc"]:
        try:
            import docx
            doc = docx.Document(file_path)
            full_text = [p.text for p in doc.paragraphs if p.text.strip()]
            return "\n\n".join(full_text), "Word Document (.docx)"
        except Exception as e:
            return f"Error reading Word document: {e}", "Word Parser Error"

    # Images (OCR)
    elif ext in [".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tiff"]:
        try:
            from PIL import Image
            import pytesseract
            img = Image.open(file_path).convert("RGB")
            text = pytesseract.image_to_string(img)
            if len(text.strip()) > 10:
                return text.strip(), "Tesseract OCR"
        except Exception:
            pass
        return "Please ensure Tesseract is available for image OCR.", "Image OCR"

    return "Unsupported file format.", "Unknown"

# 3. Clause Segmentation
def segment_clauses(text: str) -> List[Dict[str, str]]:
    text = re.sub(r'(?i)\bpage\s+\d+(\s+of\s+\d+)?\b', '', text)
    text = re.sub(r'\n{3,}', '\n\n', text).strip()
    
    chunks = re.split(r'(?m)(?=^(?:SECTION|ARTICLE|CLAUSE|\d+\.|\([a-z0-9]+\))\s+[A-Z0-9])', text)
    if len(chunks) <= 2:
        chunks = text.split('\n\n')
        
    clauses = []
    counter = 1
    for c in chunks:
        clean_c = c.strip()
        words = clean_c.split()
        if len(words) < 6:
            continue
        first_line = clean_c.split('\n')[0].strip()
        heading = f"Clause #{counter}: " + " ".join(first_line.split()[:4])
        clauses.append({
            "id": counter,
            "heading": heading,
            "text": re.sub(r'\s+', ' ', clean_c).strip()
        })
        counter += 1
    return clauses

# 4. Redline Advisor
def get_redline_fix(text: str, label: str) -> str:
    tl = text.lower()
    if label == "High Risk Clause":
        if any(k in tl for k in ["indemn", "defend", "hold harmless"]):
            return "🚨 Cap indemnification liability to fees paid in preceding 12 months, and carve out gross negligence."
        elif any(k in tl for k in ["terminat", "close user", "suspend"]):
            return "🚨 Require mutual termination rights, 30 days' written notice, cure period, and pro-rata refund."
        elif any(k in tl for k in ["jury", "class action", "arbitrat"]):
            return "🚨 Preserve statutory claims and emergency injunctive relief; ensure neutral arbitration rules."
        elif any(k in tl for k in ["lien", "security interest", "collateral"]):
            return "🚨 Narrow lien strictly to financed equipment/assets rather than a blanket lien on all assets."
        elif any(k in tl for k in ["liquidated damages", "penalty"]):
            return "🚨 Remove pre-set punitive formulas or tie liquidated damages strictly to actual direct costs."
        else:
            return "🚨 Severe one-sided exposure. Seek bilateral reciprocity and liability caps."
    elif label == "Review Recommended":
        if any(k in tl for k in ["intellectual property", "work for hire", "invention"]):
            return "⚠️ Confirm pre-existing IP remains yours; grant customer license solely for contract deliverables."
        elif any(k in tl for k in ["confidential", "secret", "disclosure"]):
            return "⚠️ Ensure mutual 3-5 year confidentiality term with standard exceptions (public domain, independent development)."
        elif any(k in tl for k in ["audit", "inspection"]):
            return "⚠️ Limit audits to once per year, during normal business hours, upon 30 days' notice, under NDA."
        elif any(k in tl for k in ["assign", "transfer"]):
            return "⚠️ Add standard carveout allowing assignment without consent in connection with M&A / sale of assets."
        else:
            return "⚠️ Review commercial and operational feasibility with business stakeholders."
    return "✅ Standard commercial boilerplate. No changes required."

# 5. Full Audit Pipeline (decorated with @spaces.GPU if on Hugging Face Spaces)
def _predict(file_obj, raw_text_input):
    if file_obj is not None:
        extracted, method = extract_text_from_file(file_obj)
    elif raw_text_input and len(raw_text_input.strip()) > 10:
        extracted = raw_text_input.strip()
        method = "Direct Text Input"
    else:
        return "⚠️ Please upload a document or paste contract text.", "", None, None

    clauses = segment_clauses(extracted)
    if not clauses:
        return "⚠️ No readable clauses found in document.", "", None, None

    # Run inference
    batch_texts = [c["text"] for c in clauses]
    inputs = tokenizer(batch_texts, padding=True, truncation=True, max_length=256, return_tensors="pt").to(device)
    
    with torch.no_grad():
        logits = model(input_ids=inputs["input_ids"], attention_mask=inputs["attention_mask"]).logits
        probs = F.softmax(logits, dim=-1).cpu().numpy()

    table_data = []
    hr_count, rev_count, std_count = 0, 0, 0

    for i, c in enumerate(clauses):
        p = probs[i]
        pred_idx = int(p.argmax())
        label = ID2LABEL[pred_idx]
        conf = f"{p[pred_idx] * 100:.1f}%"

        if label == "High Risk Clause":
            hr_count += 1
            badge = "🔴 HIGH RISK"
        elif label == "Review Recommended":
            rev_count += 1
            badge = "🟡 REVIEW"
        else:
            std_count += 1
            badge = "🟢 STANDARD"

        fix = get_redline_fix(c["text"], label)
        table_data.append([c["id"], badge, conf, c["text"], fix])

    total = len(clauses)
    score = min(100, max(0, round((hr_count * 100 + rev_count * 40 + std_count * 5) / total)))
    
    if score >= 70 or hr_count >= 3:
        status_category = "CRITICAL RISK"
    elif score >= 40 or hr_count >= 1:
        status_category = "HIGH RISK"
    elif score >= 20:
        status_category = "MODERATE RISK"
    else:
        status_category = "LOW RISK / STANDARD"

    summary_md = f"""
### 📊 Executive Contract Audit Summary
* **Overall Risk Score**: **`{score} / 100`** ({status_category})
* **Extraction Method**: {method}
* **Total Clauses Analyzed**: {total}
* 🔴 **High Risk Clauses**: `{hr_count}` | 🟡 **Review Recommended**: `{rev_count}` | 🟢 **Standard Clauses**: `{std_count}`

> **Legal Diagnosis**: {f"⚠️ Found {hr_count} critical liability terms requiring immediate legal renegotiation." if hr_count > 0 else "✅ No severe high-exposure legal liabilities detected."}
"""

    # Create CSV export file
    csv_path = "contract_risk_audit.csv"
    df = pd.DataFrame(table_data, columns=["ID", "Risk Level", "Confidence", "Clause Text", "Redline Recommendation"])
    df.to_csv(csv_path, index=False)

    return summary_md, extracted[:1000] + ("..." if len(extracted) > 1000 else ""), df, csv_path

analyze_contract = spaces.GPU(_predict) if (spaces is not None and hasattr(spaces, 'GPU')) else _predict

# 6. Gradio User Interface
theme = gr.themes.Soft(
    primary_hue="indigo",
    secondary_hue="slate",
    neutral_hue="slate"
)

with gr.Blocks(title="LegalBERT Contract Risk Intelligence") as demo:
    gr.Markdown("# ⚖️ LegalBERT Contract Risk Intelligence Platform")
    gr.Markdown("Upload any **PDF (digital/scanned), Image (PNG/JPG), Word (.docx), or Text** contract for automatic clause segmentation, risk scoring, and redline advice.")
    
    with gr.Row():
        with gr.Column(scale=1):
            file_input = gr.File(label="📄 Drag & Drop Contract File (PDF, PNG, JPG, DOCX, TXT)")
            text_input = gr.Textbox(label="✍️ Or Paste Contract Text", lines=6, placeholder="Paste clauses or full contract text here...")
            analyze_btn = gr.Button("🔍 Audit Contract Risk", variant="primary", size="lg")
            
        with gr.Column(scale=1):
            summary_output = gr.Markdown("### 📊 Summary will appear here after analysis.")
            csv_output = gr.File(label="📥 Download Structured CSV Report")

    gr.Markdown("---")
    gr.Markdown("### 📋 Clause-by-Clause Legal Risk Breakdown & Actionable Redlines")
    table_output = gr.Dataframe(
        headers=["ID", "Risk Level", "Confidence", "Clause Text", "Redline Recommendation"],
        datatype=["number", "str", "str", "str", "str"],
        wrap=True
    )

    with gr.Accordion("🔍 View Extracted Raw Text Preview", open=False):
        raw_preview = gr.Textbox(label="Raw Text", lines=8)

    analyze_btn.click(
        fn=analyze_contract,
        inputs=[file_input, text_input],
        outputs=[summary_output, raw_preview, table_output, csv_output]
    )

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860, theme=theme)
