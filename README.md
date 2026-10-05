# ⚖️ LegalBERT Contract Risk Intelligence Platform

A full-stack, real-world contract risk assessment web application powered by **LegalBERT** (`sayan-7/legalbert-contract-risk-classifier`) on Hugging Face.

---

## 🌟 Key Features

- 📄 **Multi-Format Document Ingestion**: Upload **PDF** (digital & scanned OCR), **Images** (PNG, JPG, JPEG, WEBP), **Word** (`.docx`), or **Plain Text**.
- 🔍 **Smart Clause Segmentation**: Automatically segments long legal agreements into individual provisions and detects clause headings.
- 🎯 **AI Risk Classification**: Predicts `High Risk Clause`, `Review Recommended`, or `Standard Clause` with per-class confidence percentages.
- 📊 **Executive Risk Gauge**: Calculates a 0–100 Weighted Contract Risk Score and generates an executive brief for non-lawyers.
- 🛠️ **Actionable Redline Advisor**: Provides domain-specific negotiation recommendations and redline suggestions for every high-risk clause.
- 📥 **Export Reports**: Instant one-click export to CSV and printable audit reports.

---

## 🚀 Running on Localhost (Single Command)

### 1. Install Dependencies
```bash
cd backend
pip install -r requirements.txt
```

### 2. Start the Application
```bash
python main.py
```
*Or with Uvicorn:*
```bash
uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

### 3. Open in Browser
Open **`http://localhost:8000`** in your browser to access the full interactive dashboard!

---

## 🌐 Hosting & Deployment Options

### Option A: Free 1-Click Hosting on Hugging Face Spaces (Docker)
1. Create a new Space on [Hugging Face Spaces](https://huggingface.co/new-space).
2. Select **Docker** as the Space SDK.
3. Push this repository (`backend/`, `frontend/`, `Dockerfile`, `requirements.txt`).
4. Hugging Face will automatically build and run your web app on a public URL!

### Option B: Free Hosting on Render / Railway
1. Push this repository to GitHub.
2. Link your GitHub repo to [Render](https://render.com) as a **Web Service**.
3. Set **Start Command**: `uvicorn backend.main:app --host 0.0.0.0 --port $PORT`
4. Deploy!

