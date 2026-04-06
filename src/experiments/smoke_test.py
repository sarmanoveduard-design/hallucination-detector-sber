from pathlib import Path

import pandas as pd


def main() -> None:
    project_root = Path(__file__).resolve().parent.parent
    data_path = project_root / "data" / "knowledge_bench_public.csv"
    notebook_path = project_root / "notebooks" / "baseline.ipynb"
    task_path = project_root / "configs" / "sber.md"

    print(f"Project root: {project_root}")
    print(f"Dataset exists: {data_path.exists()}")
    print(f"Notebook exists: {notebook_path.exists()}")
    print(f"Task file exists: {task_path.exists()}")

    if not data_path.exists():
        raise FileNotFoundError(f"Dataset not found: {data_path}")

    df = pd.read_csv(data_path)

    print("\nDataset loaded successfully.")
    print(f"Rows: {len(df)}")
    print(f"Columns: {list(df.columns)}")
    print("\nFirst 3 rows:")
    print(df.head(3).to_string(index=False))


if __name__ == "__main__":
    main()
