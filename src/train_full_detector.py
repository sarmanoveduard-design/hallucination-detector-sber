from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, classification_report
from sklearn.model_selection import train_test_split

from prompt_answer_features import extract_prompt_answer_features
from semantic_features import extract_semantic_features


def build_similarity_block(df: pd.DataFrame) -> np.ndarray:
    features: list[list[float]] = []

    for _, row in df.iterrows():
        model_answer = str(row["model_answer"]).lower()
        correct_answer = str(row["correct_answer"]).lower()

        exact = float(model_answer == correct_answer)
        substring = float(correct_answer in model_answer)
        reverse = float(model_answer in correct_answer)
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


def main() -> None:
    root = Path(__file__).resolve().parent.parent
    data_path = root / "data" / "knowledge_bench_public.csv"
    models_dir = root / "models"
    models_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(data_path).copy()
    df["target"] = df["is_hallucination"].astype(int)

    train_df, test_df = train_test_split(
        df,
        test_size=0.2,
        random_state=42,
        stratify=df["target"],
    )

    prompt_answer_train = extract_prompt_answer_features(train_df)
    prompt_answer_test = extract_prompt_answer_features(test_df)

    similarity_train = build_similarity_block(train_df)
    similarity_test = build_similarity_block(test_df)

    semantic_train = extract_semantic_features(train_df)
    semantic_test = extract_semantic_features(test_df)

    x_train = np.hstack(
        [
            prompt_answer_train,
            similarity_train,
            semantic_train,
        ]
    )
    x_test = np.hstack(
        [
            prompt_answer_test,
            similarity_test,
            semantic_test,
        ]
    )

    classifier = LogisticRegression(
        max_iter=3000,
        class_weight="balanced",
        solver="liblinear",
        random_state=42,
    )

    classifier.fit(x_train, train_df["target"])

    proba = classifier.predict_proba(x_test)[:, 1]
    pred = (proba > 0.5).astype(int)

    pr_auc = average_precision_score(
        test_df["target"],
        proba,
    )

    print(f"Feature count: {x_train.shape[1]}")
    print(f"PR-AUC: {pr_auc:.6f}")
    print(
        classification_report(
            test_df["target"],
            pred,
            digits=4,
        )
    )

    joblib.dump(
        classifier,
        models_dir / "full_detector.joblib",
    )

    print("\nSaved:")
    print(models_dir / "full_detector.joblib")


if __name__ == "__main__":
    main()
