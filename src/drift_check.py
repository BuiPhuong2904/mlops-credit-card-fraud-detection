import argparse
import json
import os
import subprocess
import sys
import time
import numpy as np
import pandas as pd
from scipy.stats import ks_2samp

FEATURES = [f"V{i}" for i in range(1, 29)] + ["Amount"]

def psi(reference, current, bins=10):
    """PSI voi cac khoang chia theo phan vi cua du lieu tham chieu."""
    edges = np.unique(np.quantile(reference, np.linspace(0, 1, bins + 1)))
    edges[0], edges[-1] = -np.inf, np.inf
    ref_pct = np.histogram(reference, edges)[0] / len(reference)
    cur_pct = np.histogram(current, edges)[0] / len(current)
    ref_pct = np.clip(ref_pct, 1e-6, None)
    cur_pct = np.clip(cur_pct, 1e-6, None)
    return float(np.sum((cur_pct - ref_pct) * np.log(cur_pct / ref_pct)))


def run_az(*args):
    cmd = ["az", *args]
    print(">", " ".join(cmd))
    return subprocess.call(cmd, shell=(os.name == "nt"))


def trigger_retrain(retrain_data, pipeline, data_name, rg, ws):
    version = time.strftime("drift-%Y%m%d%H%M")
    code = run_az("ml", "data", "create", "--name", data_name, "--version", version,
                  "--type", "uri_file", "--path", retrain_data, "-g", rg, "-w", ws)
    if code != 0:
        print("Khong dang ky duoc data asset moi.")
        return code
    return run_az("ml", "job", "create", "-f", pipeline, "-g", rg, "-w", ws,
                  "--set", f"inputs.raw_data.path=azureml:{data_name}:{version}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--reference", default="data/creditcard.csv")
    ap.add_argument("--current", required=True)
    ap.add_argument("--psi-threshold", type=float, default=0.2)
    ap.add_argument("--ks-alpha", type=float, default=0.01)
    ap.add_argument("--min-drifted-features", type=int, default=3)
    ap.add_argument("--report", default="reports/drift_report.json")
    ap.add_argument("--trigger-retrain", action="store_true")
    ap.add_argument("--retrain-data", default="data/retrain_data.csv")
    ap.add_argument("--pipeline", default="pipelines/training_pipeline.yml")
    ap.add_argument("--data-name", default="creditcard-raw")
    ap.add_argument("--resource-group", default="rg-fraud-mlops")
    ap.add_argument("--workspace", default="mlw-fraud-detection")
    a = ap.parse_args()

    ref = pd.read_csv(a.reference, usecols=FEATURES)
    cur = pd.read_csv(a.current, usecols=FEATURES)

    rows = []
    for f in FEATURES:
        p = psi(ref[f].to_numpy(), cur[f].to_numpy())
        ks = ks_2samp(ref[f], cur[f])
        rows.append({"feature": f, "psi": round(p, 4), "ks_stat": round(float(ks.statistic), 4),
                     "ks_pvalue": float(ks.pvalue),
                     "drift": bool(p > a.psi_threshold and ks.pvalue < a.ks_alpha)})
    res = pd.DataFrame(rows).sort_values("psi", ascending=False)
    n_drift = int(res["drift"].sum())
    dataset_drift = n_drift >= a.min_drifted_features

    print(res.head(10).to_string(index=False, formatters={"ks_pvalue": "{:.2e}".format}))
    print(f"\nSo dac trung drift: {n_drift}/{len(FEATURES)} (nguong: >= {a.min_drifted_features})")
    print("KET LUAN:", "CO DRIFT" if dataset_drift else "KHONG DRIFT")

    os.makedirs(os.path.dirname(a.report), exist_ok=True)
    with open(a.report, "w") as f:
        json.dump({"current": a.current, "n_reference": len(ref), "n_current": len(cur),
                   "psi_threshold": a.psi_threshold, "ks_alpha": a.ks_alpha,
                   "n_drifted_features": n_drift, "dataset_drift": dataset_drift,
                   "features": res.to_dict(orient="records")}, f, indent=2)
    print("Da ghi", a.report)

    if dataset_drift:
        if a.trigger_retrain:
            sys.exit(trigger_retrain(a.retrain_data, a.pipeline, a.data_name,
                                     a.resource_group, a.workspace))
        sys.exit(2)
    sys.exit(0)


if __name__ == "__main__":
    main()