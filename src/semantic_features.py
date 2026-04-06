from __future__ import annotations

import numpy as np
import pandas as pd

from runtime import get_embedding_model


def _normalize_text(value: object) -> str:
    return str(value).strip()


def _cosine_similarity_matrix(
    a: np.ndarray,
    b: np.ndarray,
) -> np.ndarray:
    a_norm = np.linalg.norm(a, axis=1, keepdims=True)
    b_norm = np.linalg.norm(b, axis=1, keepdims=True)

    a_norm = np.clip(a_norm, a_min=1e-12, a_max=None)
    b_norm = np.clip(b_norm, a_min=1e-12, a_max=None)

    return (a * b).sum(axis=1, keepdims=True) / (a_norm * b_norm)


def extract_semantic_features(df: pd.DataFrame) -> np.ndarray:
    model = get_embedding_model()

    prompts = df["prompt"].map(_normalize_text).tolist()
    answers = df["model_answer"].map(_normalize_text).tolist()
    correct_answers = df.get("correct_answer", pd.Series([""] * len(df)))
    correct_answers = correct_answers.map(_normalize_text).tolist()

    prompt_embeddings = model.encode(
        prompts,
        convert_to_numpy=True,
        normalize_embeddings=False,
        show_progress_bar=False,
    )
    answer_embeddings = model.encode(
        answers,
        convert_to_numpy=True,
        normalize_embeddings=False,
        show_progress_bar=False,
    )
    correct_embeddings = model.encode(
        correct_answers,
        convert_to_numpy=True,
        normalize_embeddings=False,
        show_progress_bar=False,
    )

    prompt_answer_cosine = _cosine_similarity_matrix(
        prompt_embeddings,
        answer_embeddings,
    )
    answer_correct_cosine = _cosine_similarity_matrix(
        answer_embeddings,
        correct_embeddings,
    )

    has_correct_answer = np.asarray(
        [[float(text != "")] for text in correct_answers],
        dtype=np.float32,
    )

    answer_correct_cosine = answer_correct_cosine * has_correct_answer

    return np.hstack(
        [
            prompt_answer_cosine.astype(np.float32),
            answer_correct_cosine.astype(np.float32),
            has_correct_answer,
        ]
    )
