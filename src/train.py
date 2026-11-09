import argparse
import json
import os
import shutil
import time

import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
from lightgbm import LGBMClassifier
from mlflow.models import infer_signature
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (average_precision_score, confusion_matrix, f1_score,
                             precision_recall_curve, precision_score, recall_score)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import RobustScaler


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--train_data", required=True)
    p.add_argument("--test_data", required=True)
    p.add_argument("--model_type", choices=["rf", "lgbm"], required=True)
    p.add_argument("--imbalance", choices=["smote", "class_weight", "none"], default="smote")
    p.add_argument("--n_estimators", type=int, default=None)
    p.add_argument("--max_depth", type=int, default=12)
    p.add_argument("--learning_rate", type=float, default=0.1)
    p.add_argument("--target_recall", type=float, default=0.85)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--model_output", required=True)
    p.add_argument("--metrics_output", required=True)
    return p.parse_args()


def make_prep():
    return ColumnTransformer(
        [("scale", RobustScaler(), ["Amount", "Time"])],
        remainder="passthrough",
        verbose_feature_names_out=False,
    ).set_output(transform="pandas")


def make_classifier(a, y_fit):
    if a.model_type == "rf":
        return RandomForestClassifier(
            n_estimators=a.n_estimators or 100,
            max_depth=a.max_depth,
            class_weight="balanced_subsample" if a.imbalance == "class_weight" else None,
            n_jobs=-1,
            random_state=a.seed,
        )
    if a.imbalance == "class_weight":
        spw = float(np.sqrt((y_fit == 0).sum() / (y_fit == 1).sum()))
    else:
        spw = 1.0
    reg_lambda = 0.0 if a.imbalance == "smote" else 10.0
    return LGBMClassifier(
        n_estimators=a.n_estimators or 200,
        learning_rate=a.learning_rate,
        scale_pos_weight=spw,
        reg_lambda=reg_lambda,
        random_state=a.seed,
        n_jobs=-1,
        verbose=-1,
    )


def pick_threshold(y_val, proba, target_recall):
    """Nguong co Precision cao nhat ma Recall van >= target_recall."""
    prec, rec, thr = precision_recall_curve(y_val, proba)
    prec, rec = prec[:-1], rec[:-1]
    ok = np.where(rec >= target_recall)[0]
    if len(ok) == 0:
        return float(thr[np.argmax(rec)])
    return float(thr[ok[np.argmax(prec[ok])]])


def main():
    a = parse_args()
    if "AZUREML_RUN_ID" not in os.environ:
        mlflow.set_experiment("fraud-detection")

    train = pd.read_parquet(a.train_data)
    test = pd.read_parquet(a.test_data)
    X_train, y_train = train.drop(columns="Class"), train["Class"]
    X_test, y_test = test.drop(columns="Class"), test["Class"]

    X_fit, X_val, y_fit, y_val = train_test_split(
        X_train, y_train, test_size=0.2, stratify=y_train, random_state=a.seed
    )

    steps = [("prep", make_prep())]
    if a.imbalance == "smote":
        steps.append(("smote", SMOTE(random_state=a.seed)))
    steps.append(("clf", make_classifier(a, y_fit)))
    model = ImbPipeline(steps)

    with mlflow.start_run():
        mlflow.set_tags({"model_type": a.model_type, "imbalance": a.imbalance})
        mlflow.log_params({
            "model_type": a.model_type,
            "imbalance": a.imbalance,
            "n_estimators": a.n_estimators or (100 if a.model_type == "rf" else 200),
            "max_depth": a.max_depth if a.model_type == "rf" else "n/a",
            "learning_rate": a.learning_rate if a.model_type == "lgbm" else "n/a",
            "target_recall": a.target_recall,
            "seed": a.seed,
        })

        t0 = time.time()
        model.fit(X_fit, y_fit)
        train_seconds = round(time.time() - t0, 1)

        thr = pick_threshold(y_val, model.predict_proba(X_val)[:, 1], a.target_recall)

        proba = model.predict_proba(X_test)[:, 1]
        pred = (proba >= thr).astype(int)
        tn, fp, fn, tp = confusion_matrix(y_test, pred).ravel()
        metrics = {
            "threshold": round(thr, 4),
            "recall": round(recall_score(y_test, pred), 4),
            "precision": round(precision_score(y_test, pred, zero_division=0), 4),
            "f1": round(f1_score(y_test, pred), 4),
            "pr_auc": round(average_precision_score(y_test, proba), 4),
            "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp),
            "train_seconds": train_seconds,
        }
        mlflow.log_metrics(metrics)

        final = Pipeline([("prep", model.named_steps["prep"]), ("clf", model.named_steps["clf"])])
        if os.path.isdir(a.model_output) and os.listdir(a.model_output):
            shutil.rmtree(a.model_output)
        mlflow.sklearn.save_model(
            final, a.model_output,
            signature=infer_signature(X_test.head(100), final.predict(X_test.head(100))),
            serialization_format=mlflow.sklearn.SERIALIZATION_FORMAT_CLOUDPICKLE,
        )
        with open(os.path.join(a.model_output, "threshold.json"), "w") as f:
            json.dump({"threshold": thr}, f)
        mlflow.log_artifacts(a.model_output, artifact_path="model")

    os.makedirs(a.metrics_output, exist_ok=True)
    with open(os.path.join(a.metrics_output, "metrics.json"), "w") as f:
        json.dump({"model_type": a.model_type, "imbalance": a.imbalance, **metrics}, f, indent=2)
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()