import sys

sys.path.insert(0, "src")

import numpy as np
import pandas as pd
from imblearn.pipeline import Pipeline as ImbPipeline
from lightgbm import LGBMClassifier
from sklearn.metrics import average_precision_score
from sklearn.model_selection import train_test_split
import train as T

train = pd.read_parquet("outputs/processed/train.parquet")
test = pd.read_parquet("outputs/processed/test.parquet")
X_train, y_train = train.drop(columns="Class"), train["Class"]
X_test, y_test = test.drop(columns="Class"), test["Class"]
X_fit, X_val, y_fit, y_val = train_test_split(
    X_train, y_train, test_size=0.2, stratify=y_train, random_state=42)

full = (y_fit == 0).sum() / (y_fit == 1).sum()
base = dict(n_estimators=200, learning_rate=0.1, random_state=42, n_jobs=-1, verbose=-1)

CONFIGS = {
    "A none, reg_lambda=1": dict(scale_pos_weight=1.0, reg_lambda=1.0),
    "B none, reg_lambda=10": dict(scale_pos_weight=1.0, reg_lambda=10.0),
    "C none, reg_lambda=10, min_child_samples=50, lr=0.05":
        dict(scale_pos_weight=1.0, reg_lambda=10.0, min_child_samples=50, learning_rate=0.05),
    "D spw=sqrt, reg_lambda=10": dict(scale_pos_weight=float(np.sqrt(full)), reg_lambda=10.0),
    "E spw=full, reg_lambda=10": dict(scale_pos_weight=float(full), reg_lambda=10.0),
}

rows = []
for name, extra in CONFIGS.items():
    clf = LGBMClassifier(**{**base, **extra})
    model = ImbPipeline([("prep", T.make_prep()), ("clf", clf)])
    model.fit(X_fit, y_fit)
    p_val = model.predict_proba(X_val)[:, 1]
    p_test = model.predict_proba(X_test)[:, 1]
    thr = T.pick_threshold(y_val, p_val, 0.85)
    pred = p_test >= thr
    tp = int((pred & (y_test.to_numpy() == 1)).sum())
    rows.append({
        "cau hinh": name,
        "AP val": round(average_precision_score(y_val, p_val), 4),
        "AP test": round(average_precision_score(y_test, p_test), 4),
        "xac suat khac nhau": len(np.unique(p_test)),
        "nguong": float(f"{thr:.3g}"),
        "bi gan gian lan": int(pred.sum()),
        "bat dung (tp/98)": tp,
        "recall": round(tp / int(y_test.sum()), 3),
        "precision": round(tp / max(int(pred.sum()), 1), 4),
    })
    print(rows[-1])

print()
print(pd.DataFrame(rows).to_string(index=False))