import os
import sys
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from typing import List, Dict, Any

# Ensure stdout supports UTF-8 on Windows
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

MODEL_NAME = "sayan-7/legalbert-contract-risk-classifier"

# Standard label mapping
ID2LABEL = {0: 'Standard Clause', 1: 'Review Recommended', 2: 'High Risk Clause'}
LABEL2ID = {v: k for k, v in ID2LABEL.items()}

class ContractRiskAnalyzer:
    def __init__(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"[INFO] Loading LegalBERT Risk Classifier from Hugging Face ({MODEL_NAME}) on {self.device}...")
        
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
            self.model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME).to(self.device)
            self.model.eval()
            print("[SUCCESS] LegalBERT model initialized successfully!")
        except Exception as e:
            print(f"[WARNING] Error loading from Hugging Face ({e}). Initializing base architecture fallback.")
            self.tokenizer = AutoTokenizer.from_pretrained("nlpaueb/legal-bert-base-uncased")
            self.model = AutoModelForSequenceClassification.from_pretrained(
                "nlpaueb/legal-bert-base-uncased", 
                num_labels=3
            ).to(self.device)
            self.model.eval()

    def generate_actionable_fix(self, text: str, risk_label: str) -> str:
        """Generates domain-specific legal redlines and negotiation recommendations."""
        text_lower = text.lower()
        
        if risk_label == "High Risk Clause":
            if any(k in text_lower for k in ["indemn", "defend", "hold harmless"]):
                return "Redline Fix: Cap indemnification to aggregate fees paid in preceding 12 months, and carve out gross negligence, willful misconduct, and breach by indemnified party."
            elif any(k in text_lower for k in ["terminat", "close user", "suspend"]):
                return "Redline Fix: Require mutual termination rights, minimum 30 days' written notice, a 15-day cure period for material breach, and pro-rata refund of unearned prepaid fees."
            elif any(k in text_lower for k in ["jury", "class action", "arbitrat"]):
                return "Redline Fix: Ensure mandatory arbitration includes mutual fee-shifting for bad faith, allows statutory small-claims carveouts, and does not extinguish emergency injunctive relief."
            elif any(k in text_lower for k in ["lien", "security interest", "collateral"]):
                return "Redline Fix: Narrow lien strictly to financed equipment/assets rather than a blanket lien on 'all assets and accounts' of the enterprise."
            elif any(k in text_lower for k in ["non-compete", "compete", "restrict"]):
                return "Redline Fix: Limit post-termination restrictions to 12 months maximum, narrow geographic radius, and remove worldwide blanket restrictions."
            elif any(k in text_lower for k in ["liquidated damages", "penalty"]):
                return "Redline Fix: Strike pre-set punitive formulas or tie liquidated damages strictly to verifiable direct administrative costs."
            elif any(k in text_lower for k in ["guarantee", "guarantor"]):
                return "Redline Fix: Remove personal individual guarantee; substitute with standard corporate parent comfort letter or security deposit."
            else:
                return "Redline Fix: Highly one-sided provision. Seek bilateral reciprocity, liability caps, and standard commercial carveouts."

        elif risk_label == "Review Recommended":
            if any(k in text_lower for k in ["intellectual property", "work for hire", "invention", "patent"]):
                return "Review Advice: Confirm pre-existing IP & tools remain party's sole property; grant limited customer license only for the contract deliverables."
            elif any(k in text_lower for k in ["confidential", "secret", "disclosure"]):
                return "Review Advice: Ensure mutual 3-to-5 year confidentiality term with standard exclusions (public domain, independently developed, compelled disclosure)."
            elif any(k in text_lower for k in ["audit", "inspection", "books and records"]):
                return "Review Advice: Limit audits to once per year, during normal business hours, upon 30 days' written notice, and under strict NDA."
            elif any(k in text_lower for k in ["assign", "transfer", "successor"]):
                return "Review Advice: Include standard exception allowing assignment without consent to affiliates or in connection with M&A / sale of assets."
            elif any(k in text_lower for k in ["payment", "interest", "fee", "invoice"]):
                return "Review Advice: Verify payment schedule (Net 30/45 days) and ensure late interest is capped at standard statutory rate."
            elif any(k in text_lower for k in ["warranty", "represent"]):
                return "Review Advice: Qualify representations by 'to party's knowledge' and require commercially reasonable standard of care."
            else:
                return "Review Advice: Operational/commercial term. Review operational feasibility and alignment with standard business practices."

        else:
            return "Standard Boilerplate: Standard commercial legal wording. No modification required."

    def analyze_clauses(self, clauses: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Runs batch inference across all extracted clauses."""
        if not clauses:
            return {
                "overall_score": 0,
                "risk_category": "Low Risk",
                "counts": {"High Risk": 0, "Review Recommended": 0, "Standard": 0},
                "clauses": [],
                "executive_summary": "No contract clauses detected in document."
            }

        analyzed = []
        high_risk_count = 0
        review_count = 0
        standard_count = 0

        # Batch tokenization & inference
        batch_size = 16
        for i in range(0, len(clauses), batch_size):
            chunk = clauses[i:i+batch_size]
            texts = [c["text"] for c in chunk]
            
            inputs = self.tokenizer(
                texts,
                padding=True,
                truncation=True,
                max_length=256,
                return_tensors="pt"
            ).to(self.device)

            with torch.no_grad():
                logits = self.model(input_ids=inputs["input_ids"], attention_mask=inputs["attention_mask"]).logits
                probs = F.softmax(logits, dim=-1).cpu().numpy()

            for j, c in enumerate(chunk):
                p = probs[j]
                pred_idx = int(p.argmax())
                pred_label = ID2LABEL[pred_idx]
                conf = float(p[pred_idx])

                if pred_label == "High Risk Clause":
                    high_risk_count += 1
                elif pred_label == "Review Recommended":
                    review_count += 1
                else:
                    standard_count += 1

                fix = self.generate_actionable_fix(c["text"], pred_label)

                analyzed.append({
                    "id": c["id"],
                    "heading": c["heading"],
                    "text": c["text"],
                    "word_count": c["word_count"],
                    "risk_label": pred_label,
                    "confidence": round(conf * 100, 2),
                    "confidence_breakdown": {
                        "Standard": round(float(p[0]) * 100, 1),
                        "Review": round(float(p[1]) * 100, 1),
                        "High Risk": round(float(p[2]) * 100, 1)
                    },
                    "recommendation": fix
                })

        # Calculate Overall Weighted Risk Score (0 to 100)
        total = len(clauses)
        # Weights: High Risk = 100 pts, Review = 40 pts, Standard = 5 pts
        raw_score = (high_risk_count * 100 + review_count * 40 + standard_count * 5) / total
        overall_score = min(100, max(0, round(raw_score)))

        if overall_score >= 70 or high_risk_count >= 3:
            risk_category = "Critical Risk"
            badge_color = "red"
        elif overall_score >= 40 or high_risk_count >= 1:
            risk_category = "High Risk"
            badge_color = "orange"
        elif overall_score >= 20 or review_count >= 3:
            risk_category = "Moderate Risk"
            badge_color = "yellow"
        else:
            risk_category = "Low Risk / Standard"
            badge_color = "green"

        # Generate Executive Summary Brief
        executive_summary = self._generate_executive_summary(
            total, high_risk_count, review_count, standard_count, overall_score, risk_category, analyzed
        )

        return {
            "overall_score": overall_score,
            "risk_category": risk_category,
            "badge_color": badge_color,
            "total_clauses": total,
            "counts": {
                "high_risk": high_risk_count,
                "review_recommended": review_count,
                "standard": standard_count
            },
            "clauses": analyzed,
            "executive_summary": executive_summary
        }

    def _generate_executive_summary(self, total, hr, rev, std, score, category, clauses):
        critical_clauses = [c for c in clauses if c["risk_label"] == "High Risk Clause"]
        review_clauses = [c for c in clauses if c["risk_label"] == "Review Recommended"]
        
        summary = f"This document contains {total} extracted provisions with an aggregate Risk Score of {score}/100 ({category}). "
        
        if hr > 0:
            summary += f"Found {hr} High-Risk provision(s) that require immediate legal attention (e.g. {', '.join([c['heading'] for c in critical_clauses[:3]])}). "
        else:
            summary += "No severe high-exposure legal liabilities were detected. "
            
        if rev > 0:
            summary += f"Found {rev} commercial term(s) requiring business review (e.g. IP ownership, confidentiality, and payment terms). "
            
        summary += f"The remaining {std} clauses consist of standard commercial boilerplate (Counterparts, Severability, Governing Law)."
        return summary

