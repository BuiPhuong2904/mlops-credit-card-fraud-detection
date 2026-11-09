import json
import numpy as np
import pandas as pd

FEATURES = ["Time"] + [f"V{i}" for i in range(1, 29)] + ["Amount"]
MAX_ROWS = 1000


class InputError(ValueError):
    """Du lieu dau vao khong dung dinh dang."""


def parse_payload(payload):
    """Nhan dict hoac chuoi JSON {"data": [ {...}, ... ]}.
    Tra ve DataFrame float64 co dung 30 cot theo thu tu FEATURES."""
    if isinstance(payload, (str, bytes, bytearray)):
        try:
            payload = json.loads(payload)
        except json.JSONDecodeError as e:
            raise InputError(f"JSON khong hop le: {e}")

    if not isinstance(payload, dict) or "data" not in payload:
        raise InputError('Body phai co dang {"data": [ {...}, ... ]}')

    rows = payload["data"]
    if isinstance(rows, dict):
        rows = [rows]
    if not isinstance(rows, list) or len(rows) == 0:
        raise InputError("'data' phai la danh sach khong rong")
    if len(rows) > MAX_ROWS:
        raise InputError(f"Toi da {MAX_ROWS} giao dich moi lan goi")
    if not all(isinstance(r, dict) for r in rows):
        raise InputError("Moi phan tu cua 'data' phai la mot object")

    df = pd.DataFrame(rows)
    missing = [c for c in FEATURES if c not in df.columns]
    if missing:
        raise InputError(f"Thieu cot: {missing}")

    df = df[FEATURES]  # bo cot thua, sap xep dung thu tu
    try:
        df = df.astype("float64")
    except (ValueError, TypeError):
        raise InputError("Tat ca cac cot phai la so")
    if not np.isfinite(df.to_numpy()).all():
        raise InputError("Co gia tri thieu hoac khong huu han (NaN/inf)")
    return df


def predict(model, df, threshold):
    """Tra ve (nhan 0/1, xac suat gian lan) theo nguong da chon luc huan luyen."""
    proba = model.predict_proba(df)[:, 1]
    labels = (proba >= threshold).astype(int)
    return labels.tolist(), proba.tolist()