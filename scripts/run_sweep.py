import os
import subprocess
import sys
import time

PY = sys.executable
RAW = "data/creditcard.csv"
PROCESSED = "outputs/processed"
ENV = {**os.environ, "MLFLOW_TRACKING_URI": "sqlite:///mlflow.db"}

# (model_type, imbalance, n_estimators, max_depth, learning_rate)
CONFIGS = [
    ("rf", "smote", 100, 12, 0.1),
    ("rf", "smote", 200, 12, 0.1),
    ("rf", "smote", 100, 8, 0.1),
    ("rf", "smote", 100, 16, 0.1),
    ("rf", "class_weight", 100, 12, 0.1),
    ("rf", "none", 100, 12, 0.1),
    ("lgbm", "smote", 200, 12, 0.1),
    ("lgbm", "smote", 400, 12, 0.05),
    ("lgbm", "class_weight", 200, 12, 0.1),
    ("lgbm", "none", 200, 12, 0.1),
]


def run(cmd):
    print(">", " ".join(cmd))
    code = subprocess.call(cmd, env=ENV)
    if code != 0:
        sys.exit(f"Lenh that bai (ma {code}). Dung lai.")


def main():
    if not os.path.exists(f"{PROCESSED}/train.parquet"):
        os.makedirs(PROCESSED, exist_ok=True)
        run([PY, "src/preprocess.py", "--raw_data", RAW, "--output_dir", PROCESSED])

    only = sys.argv[1] if len(sys.argv) > 1 else None
    t0 = time.time()
    for i, (mt, imb, n, d, lr) in enumerate(CONFIGS, 1):
        if only and mt != only:
            continue
        out = f"outputs/sweep/run{i:02d}"
        os.makedirs(f"{out}/metrics", exist_ok=True)
        print(f"\n=== [{i}/{len(CONFIGS)}] {mt} | {imb} | n_estimators={n} | max_depth={d} | lr={lr} ===")
        run([PY, "src/train.py",
             "--train_data", f"{PROCESSED}/train.parquet",
             "--test_data", f"{PROCESSED}/test.parquet",
             "--model_type", mt, "--imbalance", imb,
             "--n_estimators", str(n), "--max_depth", str(d), "--learning_rate", str(lr),
             "--model_output", f"{out}/model", "--metrics_output", f"{out}/metrics"])
    print(f"\nXong {len(CONFIGS)} cau hinh trong {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()