from pathlib import Path

import joblib
import pandas as pd
from sklearn.model_selection import train_test_split

from predict_detector import THRESHOLD, prepare_features


def main() -> None:
    root = Path(__file__).resolve().parent.parent
    data_path = root / "data" / "knowledge_bench_public.csv"
    model_path = root / "models" / "full_detector.joblib"

    df = pd.read_csv(data_path).copy()
    df["target"] = df["is_hallucination"].astype(int)

    _, test_df = train_test_split(
        df,
        test_size=0.2,
        random_state=42,
        stratify=df["target"],
    )

    classifier = joblib.load(model_path)

    x_test = prepare_features(test_df)
    y_proba = classifier.predict_proba(x_test)[:, 1]

    result = test_df.copy()
    result["hallucination_probability"] = y_proba
    result["prediction"] = (y_proba >= THRESHOLD).astype(int)

    false_positive = result[
        (result["target"] == 0) & (result["prediction"] == 1)
    ].copy()

    false_negative = result[
        (result["target"] == 1) & (result["prediction"] == 0)
    ].copy()

    false_positive = false_positive.sort_values(
        by="hallucination_probability",
        ascending=False,
    )

    false_negative = false_negative.sort_values(
        by="hallucination_probability",
        ascending=True,
    )

    print(f"Threshold: {THRESHOLD:.2f}")
    print(f"False positives: {len(false_positive)}")
    print(f"False negatives: {len(false_negative)}")

    print("\nTop 5 false positives:")
    print(
        false_positive[
            [
                "prompt",
                "model_answer",
                "correct_answer",
                "hallucination_probability",
            ]
        ]
        .head(5)
        .to_string(index=False)
    )

    print("\nTop 5 false negatives:")
    print(
        false_negative[
            [
                "prompt",
                "model_answer",
                "correct_answer",
                "hallucination_probability",
            ]
        ]
        .head(5)
        .to_string(index=False)
    )


if __name__ == "__main__":
    main()
