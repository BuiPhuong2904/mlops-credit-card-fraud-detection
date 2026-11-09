import argparse
import json
import os
import shutil
import sys


def load_metrics(metrics_dir):
    with open(os.path.join(metrics_dir, "metrics.json")) as f:
        return json.load(f)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--rf_metrics", required=True)
    p.add_argument("--rf_model", required=True)
    p.add_argument("--lgbm_metrics", required=True)
    p.add_argument("--lgbm_model", required=True)
    p.add_argument("--min_recall", type=float, default=0.80)
    p.add_argument("--select_by", choices=["pr_auc", "f1"], default="pr_auc")
    p.add_argument("--selected_model", required=True)
    p.add_argument("--decision", required=True)
    a = p.parse_args()

    candidates = [
        {"name": "rf", "metrics": load_metrics(a.rf_metrics), "model_dir": a.rf_model},
        {"name": "lgbm", "metrics": load_metrics(a.lgbm_metrics), "model_dir": a.lgbm_model},
    ]
    print(f"{'model':<6} {'recall':>7} {'precision':>10} {'f1':>7} {'pr_auc':>7}")
    for c in candidates:
        m = c["metrics"]
        print(f"{c['name']:<6} {m['recall']:>7} {m['precision']:>10} {m['f1']:>7} {m['pr_auc']:>7}")

    passed = [c for c in candidates if c["metrics"]["recall"] >= a.min_recall]
    os.makedirs(a.decision, exist_ok=True)
    summary = {c["name"]: c["metrics"] for c in candidates}

    if not passed:
        with open(os.path.join(a.decision, "decision.json"), "w") as f:
            json.dump({"passed": False, "min_recall": a.min_recall, "all": summary}, f, indent=2)
        print(f"KHONG model nao dat Recall >= {a.min_recall}. Dung pipeline.")
        sys.exit(1)

    best = max(passed, key=lambda c: c["metrics"][a.select_by])
    shutil.copytree(best["model_dir"], a.selected_model, dirs_exist_ok=True)
    decision = {
        "passed": True,
        "selected": best["name"],
        "select_by": a.select_by,
        "min_recall": a.min_recall,
        "metrics": best["metrics"],
        "all": summary,
    }
    with open(os.path.join(a.decision, "decision.json"), "w") as f:
        json.dump(decision, f, indent=2)
    print(f"-> Chon: {best['name']} (theo {a.select_by}, Recall >= {a.min_recall})")


if __name__ == "__main__":
    main()