from pathlib import Path

import pandas as pd


def main() -> None:
    project_root = Path(__file__).resolve().parent.parent
    data_path = project_root / "data" / "knowledge_bench_public.csv"

    df = pd.read_csv(data_path)

    print("Shape:", df.shape)
    print("\nColumns:")
    for column in df.columns:
        print(f"- {column}")

    print("\nTarget distribution:")
    print(df["is_hallucination"].value_counts(dropna=False))

    print("\nMissing values:")
    print(df.isna().sum())

    print("\nSample rows:")
    print(
        df[["prompt", "model_answer", "is_hallucination", "correct_answer"]]
        .head(5)
        .to_string(index=False)
    )


if __name__ == "__main__":
    main()
