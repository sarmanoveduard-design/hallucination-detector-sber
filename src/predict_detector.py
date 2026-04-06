from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from prompt_answer_features import extract_prompt_answer_features
from semantic_features import extract_semantic_features


THRESHOLD = 0.45


def build_similarity_block(df: pd.DataFrame) -> np.ndarray:
    features: list[list[float]] = []

    for _, row in df.iterrows():
        model_answer = str(row.get("model_answer", "")).lower()
        correct_answer = str(row.get("correct_answer", "")).lower()

        exact = float(model_answer == correct_answer and correct_answer != "")
        substring = float(
            correct_answer != "" and correct_answer in model_answer
        )
        reverse = float(
            correct_answer != "" and model_answer in correct_answer
        )
        len_diff = float(abs(len(model_answer) - len(correct_answer)))

        features.append(
            [
                exact,
                substring,
                reverse,
                len_diff,
            ]
        )

    return np.asarray(features, dtype=np.float32)


def prepare_features(df: pd.DataFrame) -> np.ndarray:
    prompt_answer_features = extract_prompt_answer_features(df)
    similarity_features = build_similarity_block(df)
    semantic_features = extract_semantic_features(df)

    return np.hstack(
        [
            prompt_answer_features,
            similarity_features,
            semantic_features,
        ]
    )


def predict(df: pd.DataFrame) -> pd.DataFrame:
    root = Path(__file__).resolve().parent.parent
    model_path = root / "models" / "full_detector.joblib"

    classifier = joblib.load(model_path)

    features = prepare_features(df)
    probabilities = classifier.predict_proba(features)[:, 1]
    predictions = (probabilities >= THRESHOLD).astype(int)

    result = df.copy()
    result["hallucination_probability"] = probabilities
    result["prediction"] = predictions

    return result


def main() -> None:
    sample = pd.DataFrame(
        [
            {
                "prompt": "В каком году рэпер King Von был убит?",
                "model_answer": (
                    "Рэпер King Von был убит 6 ноября 2020 года."
                ),
                "correct_answer": "",
            },
            {
                "prompt": (
                    "Сколько миллионов рублей было выделено на реализацию "
                    "программы защиты свидетелей в России на 2014—2018 годы?"
                ),
                "model_answer": (
                    "На реализацию программы защиты свидетелей было "
                    "выделено 350 миллионов рублей."
                ),
                "correct_answer": "",
            },
        ]
    )

    result = predict(sample)

    print(f"Threshold: {THRESHOLD:.2f}")
    print(
        result[
            [
                "prompt",
                "model_answer",
                "hallucination_probability",
                "prediction",
            ]
        ].to_string(index=False)
    )


if __name__ == "__main__":
    main()
