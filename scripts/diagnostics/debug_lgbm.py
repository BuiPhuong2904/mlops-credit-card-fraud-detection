import sys
import types

sys.path.insert(0, "src")

import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
from sklearn.metrics import average_precision_score
from sklearn.model_selection import train_test_split
import train as T

train = pd.read_parquet("outputs/processed/train.parquet")
test = pd.read_parquet("outputs/processed/test.parquet")
X_train, y_train = train.drop(columns="Class"), train["Class"]
X_test, y_test = test.drop(columns="Class"), test["Class"]
X_fit, X_val, y_fit, y_val = train_test_split(
    X_train, y_train, test_size=0.2, stratify=y_train, random_state=42)

print(f"fit: {len(y_fit)} dong, {int(y_fit.sum())} gian lan | val: {len(y_val)} dong, {int(y_val.sum())} gian lan"
      f" | test: {len(y_test)} dong, {int(y_test.sum())} gian lan")

for imb in ["none", "class_weight", "smote"]:
    a = types.SimpleNamespace(model_type="lgbm", imbalance=imb, n_estimators=None,
                              max_depth=12, learning_rate=0.1, seed=42)
    steps = [("prep", T.make_prep())]
    if imb == "smote":
        steps.append(("smote", SMOTE(random_state=42)))
    steps.append(("clf", T.make_classifier(a, y_fit)))
    model = ImbPipeline(steps)
    model.fit(X_fit, y_fit)

    p_val = model.predict_proba(X_val)[:, 1]
    p_test = model.predict_proba(X_test)[:, 1]
    thr = T.pick_threshold(y_val, p_val, 0.85)
    clf = model.named_steps["clf"]
    fraud, normal = p_test[y_test.to_numpy() == 1], p_test[y_test.to_numpy() == 0]

    print(f"\n===== lgbm | {imb} =====")
    print(f"so cay thuc te: {clf.booster_.num_trees()}")
    print(f"AP val = {average_precision_score(y_val, p_val):.4f} | AP test = {average_precision_score(y_test, p_test):.4f}")
    print(f"so gia tri xac suat khac nhau tren test: {len(np.unique(p_test))}")
    print(f"xac suat GIAN LAN  (min/25%/50%/75%/max): {np.round(np.quantile(fraud, [0, .25, .5, .75, 1]), 6)}")
    print(f"xac suat BINH THUONG (50%/99%/99.9%/max): {np.round(np.quantile(normal, [.5, .99, .999, 1]), 6)}")
    print(f"nguong chon = {thr:.3e} | so giao dich bi gan 'gian lan' tren test = {int((p_test >= thr).sum())}/{len(p_test)}")