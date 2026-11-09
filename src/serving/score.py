import json
import os
from scoring_core import parse_payload, predict

model = None
threshold = 0.5

def _find_model_dir(root):
    for dirpath, _, files in os.walk(root):
        if "MLmodel" in files:
            return dirpath
    raise FileNotFoundError(f"Khong tim thay MLmodel trong {root}")


def init():
    global model, threshold
    import mlflow.sklearn

    model_dir = _find_model_dir(os.environ["AZUREML_MODEL_DIR"])
    model = mlflow.sklearn.load_model(model_dir)
    threshold_file = os.path.join(model_dir, "threshold.json")
    if os.path.exists(threshold_file):
        with open(threshold_file) as f:
            threshold = float(json.load(f)["threshold"])
    print(f"Model loaded from {model_dir}, threshold={threshold}")


def run(raw_data):
    df = parse_payload(raw_data)
    labels, proba = predict(model, df, threshold)
    return {
        "predictions": labels,
        "probabilities": [round(p, 6) for p in proba],
        "threshold": threshold,
    }