import argparse
import json
import os
import mlflow

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model_dir", required=True)
    p.add_argument("--decision", required=True)
    p.add_argument("--model_name", default="fraud-detector")
    a = p.parse_args()

    with open(os.path.join(a.decision, "decision.json")) as f:
        decision = json.load(f)
    if not decision.get("passed"):
        raise SystemExit("Khong co model dat dieu kien, khong dang ky.")

    if "AZUREML_RUN_ID" not in os.environ:
        mlflow.set_experiment("fraud-detection")

    with mlflow.start_run() as run:
        mlflow.set_tags({"selected_model": decision["selected"], "select_by": decision["select_by"]})
        mlflow.log_metrics({k: v for k, v in decision["metrics"].items()
                            if k in ("recall", "precision", "f1", "pr_auc", "threshold")})
        mlflow.log_artifacts(a.model_dir, artifact_path="model")
        mv = mlflow.register_model(f"runs:/{run.info.run_id}/model", a.model_name)
        print(f"Da dang ky model '{mv.name}' version {mv.version} ({decision['selected']})")


if __name__ == "__main__":
    main()