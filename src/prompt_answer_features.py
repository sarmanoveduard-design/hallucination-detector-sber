from __future__ import annotations

import re

import numpy as np
import pandas as pd


def normalize(text: str) -> str:
    text = str(text).lower().strip()
    text = re.sub(r"\s+", " ", text)
    return text


def tokens(text: str) -> set[str]:
    return set(re.findall(r"\w+", normalize(text)))


def extract_prompt_answer_features(df: pd.DataFrame) -> np.ndarray:
    features: list[list[float]] = []

    for _, row in df.iterrows():

        prompt = normalize(row["prompt"])
        answer = normalize(row["model_answer"])

        prompt_tokens = tokens(prompt)
        answer_tokens = tokens(answer)

        intersection = len(prompt_tokens & answer_tokens)
        union = len(prompt_tokens | answer_tokens)

        token_overlap = intersection / union if union else 0.0

        prompt_len = float(len(prompt))
        answer_len = float(len(answer))

        length_ratio = answer_len / (prompt_len + 1)

        answer_token_count = float(len(answer_tokens))
        prompt_token_count = float(len(prompt_tokens))

        number_in_prompt = bool(re.search(r"\d", prompt))
        number_in_answer = bool(re.search(r"\d", answer))

        number_mismatch = float(
            number_in_answer and not number_in_prompt
        )

        question_type = float(prompt.endswith("?"))

        features.append(
            [
                prompt_len,
                answer_len,
                length_ratio,
                prompt_token_count,
                answer_token_count,
                token_overlap,
                float(number_in_prompt),
                float(number_in_answer),
                number_mismatch,
                question_type,
            ]
        )

    return np.asarray(features, dtype=np.float32)
