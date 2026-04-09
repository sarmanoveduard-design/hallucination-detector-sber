from pathlib import Path

import pandas as pd

from predict_detector import predict


def main() -> None:
    root = Path(__file__).resolve().parent.parent
    input_path = root / "data" / "knowledge_bench_public.csv"
    output_path = root / "data" / "knowledge_bench_public_scores.csv"

    df = pd.read_csv(input_path)
    result = predict(df)

    result.to_csv(output_path, index=False, encoding="utf-8-sig")

    print("Done.")
    print(f"Input rows: {len(df)}")
    print(f"Saved to: {output_path}")


if __name__ == "__main__":
    main()
