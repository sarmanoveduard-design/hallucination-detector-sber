from difflib import SequenceMatcher
from pathlib import Path
import re

import joblib
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, classification_report
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.model_selection import train_test_split


def normalize_text(text: str) -> str:
    text = str(text).strip().lower()
    text = re.sub(r"\s+", " ", text)
    return text


def tokenize(text: str) -> list[str]:
    return re.findall(r"\w+", normalize_text(text), flags=re.UNICODE)


def build_numeric_features(df: pd.DataFrame) -> np.ndarray:
    features: list[list[float]] = []

    for _, row in df.iterrows():
        model_answer = normalize_text(row["model_answer"])
        correct_answer = normalize_text(row["correct_answer"])

        model_tokens = set(tokenize(model_answer))
        correct_tokens = set(tokenize(correct_answer))

        intersection = len(model_tokens & correct_tokens)
        union = len(model_tokens | correct_tokens)

        token_jaccard = intersection / union if union else 0.0
        token_recall = (
            intersection / len(correct_tokens) if correct_tokens else 0.0
        )
        token_precision = (
            intersection / len(model_tokens) if model_tokens else 0.0
        )

        exact_match = float(model_answer == correct_answer)
        substring_match = float(correct_answer in model_answer)
        reverse_substring_match = float(model_answer in correct_answer)

        answer_len = float(len(model_answer))
        correct_len = float(len(correct_answer))
        len_diff = abs(answer_len - correct_len)

        seq_ratio = SequenceMatcher(
            None,
            model_answer,
            correct_answer,
        ).ratio()

        features.append(
            [
                exact_match,
                substring_match,
                reverse_substring_match,
                answer_len,
                correct_len,
                len_diff,
                float(len(model_tokens)),
                float(len(correct_tokens)),
                token_jaccard,
                token_recall,
                token_precision,
                seq_ratio,
            ]
        )

    return np.asarray(features, dtype=np.float32)


def main() -> None:
    project_root = Path(__file__).resolve().parent.parent
    data_path = project_root / "data" / "knowledge_bench_public.csv"
    models_dir = project_root / "models"
    models_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(data_path).copy()
    df["target"] = df["is_hallucination"].astype(int)
    df["model_answer_norm"] = df["model_answer"].astype(str).map(
        normalize_text
    )
    df["correct_answer_norm"] = df["correct_answer"].astype(str).map(
        normalize_text
    )

    train_df, test_df = train_test_split(
        df,
        test_size=0.2,
        random_state=42,
        stratify=df["target"],
    )

    word_vectorizer = TfidfVectorizer(
        lowercase=True,
        analyzer="word",
        ngram_range=(1, 2),
        min_df=2,
        sublinear_tf=True,
    )

    char_vectorizer = TfidfVectorizer(
        lowercase=True,
        analyzer="char_wb",
        ngram_range=(3, 5),
        min_df=2,
        sublinear_tf=True,
    )

    train_word_answer = word_vectorizer.fit_transform(
        train_df["model_answer_norm"]
    )
    test_word_answer = word_vectorizer.transform(
        test_df["model_answer_norm"]
    )

    train_word_correct = word_vectorizer.transform(
        train_df["correct_answer_norm"]
    )
    test_word_correct = word_vectorizer.transform(
        test_df["correct_answer_norm"]
    )

    train_char_answer = char_vectorizer.fit_transform(
        train_df["model_answer_norm"]
    )
    test_char_answer = char_vectorizer.transform(
        test_df["model_answer_norm"]
    )

    train_char_correct = char_vectorizer.transform(
        train_df["correct_answer_norm"]
    )
    test_char_correct = char_vectorizer.transform(
        test_df["correct_answer_norm"]
    )

    train_word_cosine = cosine_similarity(
        train_word_answer,
        train_word_correct,
    ).diagonal().reshape(-1, 1)

    test_word_cosine = cosine_similarity(
        test_word_answer,
        test_word_correct,
    ).diagonal().reshape(-1, 1)

    train_char_cosine = cosine_similarity(
        train_char_answer,
        train_char_correct,
    ).diagonal().reshape(-1, 1)

    test_char_cosine = cosine_similarity(
        test_char_answer,
        test_char_correct,
    ).diagonal().reshape(-1, 1)

    train_numeric = build_numeric_features(train_df)
    test_numeric = build_numeric_features(test_df)

    x_train = np.hstack(
        [train_word_cosine, train_char_cosine, train_numeric]
    )
    x_test = np.hstack(
        [test_word_cosine, test_char_cosine, test_numeric]
    )

    classifier = LogisticRegression(
        max_iter=3000,
        class_weight="balanced",
        solver="liblinear",
        random_state=42,
    )

    classifier.fit(x_train, train_df["target"])

    y_proba = classifier.predict_proba(x_test)[:, 1]
    y_pred = (y_proba >= 0.5).astype(int)

    pr_auc = average_precision_score(test_df["target"], y_proba)

    print(f"Train size: {len(train_df)}")
    print(f"Test size: {len(test_df)}")
    print(f"Feature count: {x_train.shape[1]}")
    print(f"PR-AUC: {pr_auc:.6f}")
    print("\nClassification report:")
    print(classification_report(test_df["target"], y_pred, digits=4))

    joblib.dump(
        word_vectorizer,
        models_dir / "sim_word_vectorizer.joblib",
    )
    joblib.dump(
        char_vectorizer,
        models_dir / "sim_char_vectorizer.joblib",
    )
    joblib.dump(
        classifier,
        models_dir / "similarity_detector.joblib",
    )

    print("\nSaved:")
    print(models_dir / "sim_word_vectorizer.joblib")
    print(models_dir / "sim_char_vectorizer.joblib")
    print(models_dir / "similarity_detector.joblib")


if __name__ == "__main__":
    main()
