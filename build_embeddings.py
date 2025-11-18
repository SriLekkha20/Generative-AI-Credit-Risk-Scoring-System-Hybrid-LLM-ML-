"""
Build sentence embeddings for credit 'remarks' and create a FAISS index.
"""

from pathlib import Path

import faiss
import numpy as np
import pandas as pd
from transformers import AutoModel, AutoTokenizer
import torch
import joblib

DATA_PATH = Path("data/credit_data.csv")
MODEL_DIR = Path("models")
MODEL_DIR.mkdir(parents=True, exist_ok=True)

EMBED_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"


def mean_pool(last_hidden_state, attention_mask):
    mask = attention_mask.unsqueeze(-1).expand(last_hidden_state.size()).float()
    masked = last_hidden_state * mask
    summed = torch.sum(masked, dim=1)
    counts = torch.clamp(mask.sum(dim=1), min=1e-9)
    return summed / counts


def main():
    df = pd.read_csv(DATA_PATH)
    remarks = df["remarks"].fillna("").tolist()

    tokenizer = AutoTokenizer.from_pretrained(EMBED_MODEL_NAME)
    model = AutoModel.from_pretrained(EMBED_MODEL_NAME)

    all_embeddings = []

    for text in remarks:
        encoded = tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            padding=True,
            max_length=64,
        )
        with torch.no_grad():
            out = model(**encoded)
            emb = mean_pool(out.last_hidden_state, encoded["attention_mask"])
        emb_np = emb.cpu().numpy()[0]
        all_embeddings.append(emb_np)

    embed_matrix = np.vstack(all_embeddings).astype("float32")

    dim = embed_matrix.shape[1]
    index = faiss.IndexFlatL2(dim)
    index.add(embed_matrix)

    faiss.write_index(index, str(MODEL_DIR / "remark_index.faiss"))
    joblib.dump(df["remarks"].tolist(), MODEL_DIR / "remark_texts.joblib")

    print("✅ Built FAISS index and saved remark texts.")


if __name__ == "__main__":
    main()
