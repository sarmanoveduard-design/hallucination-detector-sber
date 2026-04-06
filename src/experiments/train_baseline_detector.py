from pathlib import Path

import joblib
import pandas as pd
from scipy.sparse import hstack
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, classification_report
from sklearn.model_selection import train_test_split


def build_text(row: pd.Series) -> str:
    prompt = str(row["prompt"]).strip()
    model_answer = str(row["model_answer"]).strip()
    correct_answer = str(row["correct_answer"]).strip()

    return (
        f"PROMPT: {prompt}\n"
        f"MODEL_ANSWER: {model_answer}\n"
        f"CORRECT_ANSWER: {correct_answer}"
    )


def main() -> None:
    project_root = Path(__file__).resolve().parent.parent
    data_path = project_root / "data" / "knowledge_bench_public.csv"
    models_dir = project_root / "models"
    models_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(data_path).copy()
    df["text"] = df.apply(build_text, axis=1)
    df["target"] = df["is_hallucination"].astype(int)

    x_train, x_test, y_train, y_test = train_test_split(
        df["text"],
        df["target"],
        test_size=0.2,
        random_state=42,
        stratify=df["target"],
    )

    word_vectorizer = TfidfVectorizer(
        lowercase=True,
        analyzer="word",
        ngram_range=(1, 2),
        min_df=2,
        max_df=0.95,
        sublinear_tf=True,
    )

    char_vectorizer = TfidfVectorizer(
        lowercase=True,
        analyzer="char_wb",
        ngram_range=(3, 5),
        min_df=2,
        sublinear_tf=True,
    )

    x_train_word = word_vectorizer.fit_transform(x_train)
    x_test_word = word_vectorizer.transform(x_test)

    x_train_char = char_vectorizer.fit_transform(x_train)
    x_test_char = char_vectorizer.transform(x_test)

    x_train_features = hstack([x_train_word, x_train_char]).tocsr()
    x_test_features = hstack([x_test_word, x_test_char]).tocsr()

    classifier = LogisticRegression(
        max_iter=3000,
        class_weight="balanced",
        solver="liblinear",
        random_state=42,
    )

    classifier.fit(x_train_features, y_train)

    y_proba = classifier.predict_proba(x_test_features)[:, 1]
    y_pred = (y_proba >= 0.5).astype(int)

    pr_auc = average_precision_score(y_test, y_proba)

    print(f"Train size: {len(x_train)}")
    print(f"Test size: {len(x_test)}")
    print(f"Word features: {x_train_word.shape[1]}")
    print(f"Char features: {x_train_char.shape[1]}")
    print(f"Total features: {x_train_features.shape[1]}")
    print(f"PR-AUC: {pr_auc:.6f}")
    print("\nClassification report:")
    print(classification_report(y_test, y_pred, digits=4))

    joblib.dump(word_vectorizer, models_dir / "word_vectorizer.joblib")
    joblib.dump(char_vectorizer, models_dir / "char_vectorizer.joblib")
    joblib.dump(classifier, models_dir / "baseline_detector.joblib")

    print("\nSaved:")
    print(models_dir / "word_vectorizer.joblib")
    print(models_dir / "char_vectorizer.joblib")
    print(models_dir / "baseline_detector.joblib")


if __name__ == "__main__":
    main()
