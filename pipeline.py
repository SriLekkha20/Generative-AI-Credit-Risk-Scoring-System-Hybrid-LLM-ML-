"""
Utility functions to load the risk model, scaler, and text index.
"""

from pathlib import Path
from typing import Dict, Any

import faiss
import joblib
import numpy as np
import xgboost as xgb

MODEL_DIR = Path("models")


class CreditRiskPipeline:
    def __init__(self):
        self.model = xgb.Booster()
        self.model.load_model(MODEL_DIR / "xgb_model.json")
        self.scaler = joblib.load(MODEL_DIR / "scaler.joblib")

        index_path = MODEL_DIR / "remark_index.faiss"
        text_path = MODEL_DIR / "remark_texts.joblib"
        if index_path.exists() and text_path.exists():
            self.index = faiss.read_index(str(index_path))
            self.remark_texts = joblib.load(text_path)
        else:
            self.index = None
            self.remark_texts = []

    def score(self, features: Dict[str, float]) -> Dict[str, Any]:
        x = np.array(
            [
                features["age"],
                features["income"],
                features["loan_amount"],
                features["credit_history_length"],
            ],
            dtype=float,
        ).reshape(1, -1)

        x_scaled = self.scaler.transform(x)
        dmatrix = xgb.DMatrix(x_scaled)
        prob_default = float(self.model.predict(dmatrix)[0])
        risk_score = 1.0 - prob_default
        decision = "APPROVE" if risk_score >= 0.6 else "REVIEW"

        return {
            "risk_score": risk_score,
            "default_probability": prob_default,
            "decision": decision,
        }

    def similar_remarks(self, query_embedding: np.ndarray, top_k: int = 3):
        if self.index is None:
            return []

        query_embedding = query_embedding.astype("float32").reshape(1, -1)
        distances, indices = self.index.search(query_embedding, top_k)
        results = []
        for dist, idx in zip(distances[0], indices[0]):
            if idx < 0 or idx >= len(self.remark_texts):
                continue
            results.append(
                {
                    "remark": self.remark_texts[idx],
                    "distance": float(dist),
                }
            )
        return results
