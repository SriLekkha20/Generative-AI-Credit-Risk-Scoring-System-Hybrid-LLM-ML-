# Generative AI – Credit Risk Scoring System (Hybrid LLM + ML) 💳

This project demonstrates a simplified **hybrid LLM + ML** workflow for credit risk:

- Numeric features → XGBoost model
- Text remarks → Transformer embeddings + FAISS index
- Combined for risk reasoning & explainable responses
- REST API using FastAPI

> Note: This is a **demo implementation** designed for learning and GitHub portfolio use.

---

## 🧱 Components

- `src/train_xgboost.py`  
  Trains an XGBoost classifier on synthetic credit data.

- `src/build_embeddings.py`  
  Creates sentence embeddings for the `remarks` field and builds a FAISS index.

- `src/pipeline.py`  
  Utility functions to load models, scalers, and perform scoring.

- `api/main.py`  
  FastAPI application exposing `/score` endpoint.

- `monitoring/metrics.py`  
  Simple stub for logging request stats.

---

## 📦 Tech Stack

- Python
- XGBoost
- scikit-learn
- Transformers (sentence embeddings)
- FAISS (CPU)
- FastAPI
- Uvicorn

---

## 📂 Dataset

Synthetic dataset in `data/credit_data.csv` with columns:

- `age`
- `income`
- `loan_amount`
- `credit_history_length`
- `remarks`
- `default` (Y/N)

---

## 🚀 Setup

```bash
pip install -r requirements.txt
