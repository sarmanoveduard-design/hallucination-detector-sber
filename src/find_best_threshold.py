from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, classification_report
from sklearn.model_selection import train_test_split

from predict_detector import prepare_features


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
    y_test = test_df["target"].to_numpy()
    y_proba = classifier.predict_proba(x_test)[:, 1]

    best_threshold = 0.5
    best_f1 = -1.0

    print(f"PR-AUC: {average_precision_score(y_test, y_proba):.6f}")
    print("\nThreshold scan:")

    for threshold in np.arange(0.1, 0.91, 0.05):
        y_pred = (y_proba >= threshold).astype(int)

        tp = int(((y_pred == 1) & (y_test == 1)).sum())
        fp = int(((y_pred == 1) & (y_test == 0)).sum())
        fn = int(((y_pred == 0) & (y_test == 1)).sum())

        precision = tp / (tp + fp) if (tp + fp) else 0.0
        recall = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = (
            2 * precision * recall / (precision + recall)
            if (precision + recall)
            else 0.0
        )

        print(
            f"threshold={threshold:.2f} "
            f"precision={precision:.4f} "
            f"recall={recall:.4f} "
            f"f1={f1:.4f}"
        )

        if f1 > best_f1:
            best_f1 = f1
            best_threshold = float(threshold)

    print(f"\nBest threshold: {best_threshold:.2f}")
    print(f"Best F1: {best_f1:.4f}")

    final_pred = (y_proba >= best_threshold).astype(int)

    print("\nClassification report at best threshold:")
    print(classification_report(y_test, final_pred, digits=4))


if __name__ == "__main__":
    main()
