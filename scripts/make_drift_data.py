import argparse
import numpy as np
import pandas as pd

SHIFT_STD = {"V4": 1.5, "V11": 1.5, "V14": -1.5, "V17": 1.5}
AMOUNT_FACTOR = 2.5

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default="data/creditcard.csv")
    ap.add_argument("--n", type=int, default=20000)
    ap.add_argument("--seed", type=int, default=7)
    a = ap.parse_args()

    df = pd.read_csv(a.csv)
    std = df[list(SHIFT_STD)].std()

    normal = df.sample(a.n, random_state=a.seed).reset_index(drop=True)
    drifted = df.sample(a.n, random_state=a.seed + 1).reset_index(drop=True)
    for col, k in SHIFT_STD.items():
        drifted[col] = drifted[col] + k * std[col]
    drifted["Amount"] = drifted["Amount"] * AMOUNT_FACTOR

    normal.to_csv("data/prod_normal.csv", index=False)
    drifted.to_csv("data/prod_drifted.csv", index=False)
    pd.concat([df, drifted], ignore_index=True).to_csv("data/retrain_data.csv", index=False)

    print(f"prod_normal.csv  : {len(normal)} dong")
    print(f"prod_drifted.csv : {len(drifted)} dong (drift: {SHIFT_STD}, Amount x{AMOUNT_FACTOR})")
    print(f"retrain_data.csv : {len(df) + len(drifted)} dong (goc + drifted)")


if __name__ == "__main__":
    main()