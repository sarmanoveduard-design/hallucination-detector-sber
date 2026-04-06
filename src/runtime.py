from __future__ import annotations

from pathlib import Path

import joblib
from sentence_transformers import SentenceTransformer


MODEL_NAME = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"


_classifier = None
_embedding_model = None


def get_project_root() -> Path:
    return Path(__file__).resolve().parent.parent


def get_classifier():
    global _classifier

    if _classifier is None:
        model_path = get_project_root() / "models" / "full_detector.joblib"
        _classifier = joblib.load(model_path)

    return _classifier


def get_embedding_model() -> SentenceTransformer:
    global _embedding_model

    if _embedding_model is None:
        _embedding_model = SentenceTransformer(MODEL_NAME)

    return _embedding_model
