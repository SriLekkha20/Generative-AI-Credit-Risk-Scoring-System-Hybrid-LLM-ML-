"""
FastAPI app exposing /score endpoint for credit risk.
"""

from typing import Optional

from fastapi import FastAPI
from pydantic import BaseModel, Field
from transformers import AutoTokenizer, AutoModel
import torch
import numpy as np

from src.pipeline import CreditRiskPipeline
from monitoring.metrics import log_request


EMBED_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"


def mean_pool(last_hidden_state, attention_mask):
    mask = attention_mask.unsqueeze(-1).expand(last_hidden_state.size()).float()
    masked = last_hidden_state * mask
    summed = torch.sum(masked, dim=1)
    counts = torch.clamp(mask.sum(dim=1), min=1e-9)
    return summed / counts


class Applicant(BaseModel):
    age: int = Field(..., ge=18, le=90)
    income: float = Field(..., ge=0)
    loan_amount: float = Field(..., ge=0)
    credit_history_length: int = Field(..., ge=0)
    remarks: Optional[str] = ""


app = FastAPI(title="Hybrid LLM + ML Credit Risk API")

pipeline = CreditRiskPipeline()
tokenizer = AutoTokenizer.from_pretrained(EMBED_MODEL_NAME)
embed_model = AutoModel.from_pretrained(EMBED_MODEL_NAME)


@app.get("/")
async def root():
    return {"status": "ok", "service": "credit-risk-genai"}


@app.post("/score")
async def score(applicant: Applicant):
    log_request("/score")

    features = {
        "age": applicant.age,
        "income": applicant.income,
        "loan_amount": applicant.loan_amount,
        "credit_history_length": applicant.credit_history_length,
    }

    risk_result = pipeline.score(features)

    explanation = {}
    if applicant.remarks:
        encoded = tokenizer(
            applicant.remarks,
            return_tensors="pt",
            truncation=True,
            padding=True,
            max_length=64,
        )
        with torch.no_grad():
            out = embed_model(**encoded)
            emb = mean_pool(out.last_hidden_state, encoded["attention_mask"])
        emb_np = emb.cpu().numpy()[0]
        neighbors = pipeline.similar_remarks(emb_np, top_k=3)
        explanation["similar_remarks"] = neighbors

    return {
        "risk": risk_result,
        "explanation": explanation,
    }
