import argparse
import os

import pandas as pd
from sklearn.model_selection import train_test_split


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--raw_data", required=True, help="duong dan file creditcard.csv")
    p.add_argument("--output_dir", required=True)
    p.add_argument("--test_size", type=float, default=0.2)
    p.add_argument("--seed", type=int, default=42)
    a = p.parse_args()

    df = pd.read_csv(a.raw_data)
    X, y = df.drop(columns="Class"), df["Class"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=a.test_size, stratify=y, random_state=a.seed
    )

    os.makedirs(a.output_dir, exist_ok=True)
    X_train.assign(Class=y_train.values).to_parquet(
        os.path.join(a.output_dir, "train.parquet"), index=False
    )
    X_test.assign(Class=y_test.values).to_parquet(
        os.path.join(a.output_dir, "test.parquet"), index=False
    )
    print(f"Train: {len(X_train)} dong, fraud={int(y_train.sum())}")
    print(f"Test : {len(X_test)} dong, fraud={int(y_test.sum())}")


if __name__ == "__main__":
    main()