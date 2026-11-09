import copy
import json
import os
import sys
import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src", "serving"))
import score  # noqa: E402
from scoring_core import FEATURES, MAX_ROWS, InputError, parse_payload, predict  # noqa: E402

SAMPLE = {c: 0.0 for c in FEATURES}
SAMPLE["Amount"] = 100.0


class FakeModel:
    """Gia lap model: xac suat tang theo cot Amount (de test nguong)"""

    def predict_proba(self, df):
        p = np.clip(df["Amount"].to_numpy() / 1000.0, 0, 1)
        return np.column_stack([1 - p, p])


def payload(*rows):
    return {"data": [copy.deepcopy(r) for r in rows]}


# ---------- dau vao hop le ----------
def test_valid_record_gives_30_ordered_float_columns():
    df = parse_payload(payload(SAMPLE))
    assert list(df.columns) == FEATURES
    assert df.shape == (1, 30)
    assert str(df.dtypes.unique()[0]) == "float64"


def test_accepts_json_string():
    df = parse_payload(json.dumps(payload(SAMPLE, SAMPLE)))
    assert df.shape == (2, 30)


def test_accepts_single_object_instead_of_list():
    df = parse_payload({"data": SAMPLE})
    assert df.shape == (1, 30)


def test_column_order_is_normalized():
    shuffled = dict(reversed(list(SAMPLE.items())))
    df = parse_payload(payload(shuffled))
    assert list(df.columns) == FEATURES


def test_extra_columns_are_ignored():
    row = {**SAMPLE, "Class": 1, "note": "abc"}
    df = parse_payload(payload(row))
    assert list(df.columns) == FEATURES


# ---------- dau vao sai ----------
def test_invalid_json_string_raises():
    with pytest.raises(InputError):
        parse_payload("{not json")


def test_missing_data_key_raises():
    with pytest.raises(InputError):
        parse_payload({"rows": [SAMPLE]})


def test_empty_data_raises():
    with pytest.raises(InputError):
        parse_payload({"data": []})


def test_missing_column_raises_and_names_it():
    row = copy.deepcopy(SAMPLE)
    del row["V14"]
    with pytest.raises(InputError, match="V14"):
        parse_payload(payload(row))


def test_non_numeric_value_raises():
    row = {**SAMPLE, "Amount": "abc"}
    with pytest.raises(InputError):
        parse_payload(payload(row))


def test_null_value_raises():
    row = {**SAMPLE, "V1": None}
    with pytest.raises(InputError):
        parse_payload(payload(row))


def test_infinite_value_raises():
    row = {**SAMPLE, "V2": float("inf")}
    with pytest.raises(InputError):
        parse_payload(payload(row))


def test_too_many_rows_raises():
    with pytest.raises(InputError):
        parse_payload(payload(*([SAMPLE] * (MAX_ROWS + 1))))


def test_row_that_is_not_an_object_raises():
    with pytest.raises(InputError):
        parse_payload({"data": [[1, 2, 3]]})


# ---------- du doan va nguong ----------
def test_threshold_decides_label():
    df = parse_payload(payload({**SAMPLE, "Amount": 100.0},
                               {**SAMPLE, "Amount": 500.0},
                               {**SAMPLE, "Amount": 900.0}))
    labels, _ = predict(FakeModel(), df, threshold=0.5)
    assert labels == [0, 1, 1]
    labels, _ = predict(FakeModel(), df, threshold=0.95)
    assert labels == [0, 0, 0]


def test_run_returns_only_0_or_1(monkeypatch):
    monkeypatch.setattr(score, "model", FakeModel())
    monkeypatch.setattr(score, "threshold", 0.5)
    out = score.run(json.dumps(payload({**SAMPLE, "Amount": 900.0}, SAMPLE)))
    assert set(out["predictions"]) <= {0, 1}
    assert out["predictions"] == [1, 0]
    assert out["threshold"] == 0.5


def test_run_rejects_bad_input(monkeypatch):
    monkeypatch.setattr(score, "model", FakeModel())
    with pytest.raises(InputError):
        score.run({"data": [{"Amount": 1}]})


# ---------- endpoint that ----------
LIVE = os.getenv("ENDPOINT_URL") and os.getenv("ENDPOINT_KEY")


@pytest.mark.skipif(not LIVE, reason="Chua dat ENDPOINT_URL / ENDPOINT_KEY")
def test_live_endpoint_valid_and_invalid_input():
    import requests

    headers = {"Authorization": f"Bearer {os.environ['ENDPOINT_KEY']}",
               "Content-Type": "application/json"}
    ok = requests.post(os.environ["ENDPOINT_URL"], json=payload(SAMPLE), headers=headers, timeout=60)
    assert ok.status_code == 200
    body = ok.json()
    assert set(body["predictions"]) <= {0, 1}

    bad = requests.post(os.environ["ENDPOINT_URL"], json={"data": [{"Amount": 1}]}, headers=headers, timeout=60)
    assert bad.status_code != 200