from time import perf_counter

import pandas as pd
from sklearn.model_selection import train_test_split

from predict_detector import prepare_features
from runtime import get_classifier, get_embedding_model
from pathlib import Path


def main() -> None:
    root = Path(__file__).resolve().parent.parent
    data_path = root / "data" / "knowledge_bench_public.csv"

    df = pd.read_csv(data_path).copy()

    _, test_df = train_test_split(
        df,
        test_size=0.2,
        random_state=42,
        stratify=df["is_hallucination"],
    )

    sample_1 = test_df.head(1).copy()
    sample_10 = test_df.head(10).copy()
    sample_50 = test_df.head(50).copy()

    cold_start_begin = perf_counter()
    _ = get_embedding_model()
    classifier = get_classifier()
    cold_start_elapsed = perf_counter() - cold_start_begin

    print("Cold start:")
    print(f"  total_time_sec = {cold_start_elapsed:.4f}")
    print()

    for name, sample in (
        ("1 sample", sample_1),
        ("10 samples", sample_10),
        ("50 samples", sample_50),
    ):
        start = perf_counter()
        features = prepare_features(sample)
        _ = classifier.predict_proba(features)[:, 1]
        elapsed = perf_counter() - start

        per_sample_ms = (elapsed / len(sample)) * 1000

        print(f"{name}:")
        print(f"  total_time_sec = {elapsed:.4f}")
        print(f"  per_sample_ms  = {per_sample_ms:.2f}")
        print()


if __name__ == "__main__":
    main()
