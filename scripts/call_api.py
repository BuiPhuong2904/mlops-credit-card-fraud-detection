import argparse
import json
import os
import subprocess
import sys
import time
import pandas as pd
import requests

ENDPOINT = "fraud-ep-b01"
RG = "rg-fraud-mlops"
WS = "mlw-fraud-detection"
FEATURES = ["Time"] + [f"V{i}" for i in range(1, 29)] + ["Amount"]


def az(*args):
    cmd = ["az", *args, "-g", RG, "-w", WS, "-o", "tsv"]
    return subprocess.check_output(cmd, shell=(os.name == "nt"), text=True).strip()


def get_endpoint():
    url = az("ml", "online-endpoint", "show", "-n", ENDPOINT, "--query", "scoring_uri")
    key = az("ml", "online-endpoint", "get-credentials", "-n", ENDPOINT, "--query", "primaryKey")
    return url, key


def call(url, key, body):
    headers = {"Authorization": f"Bearer {key}", "Content-Type": "application/json"}
    t0 = time.time()
    r = requests.post(url, json=body, headers=headers, timeout=60)
    return r, (time.time() - t0) * 1000


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default="data/creditcard.csv")
    ap.add_argument("--run-tests", action="store_true")
    a = ap.parse_args()

    url, key = get_endpoint()
    print("Scoring URI:", url)

    if a.run_tests:
        env = {**os.environ, "ENDPOINT_URL": url, "ENDPOINT_KEY": key}
        sys.exit(subprocess.call([sys.executable, "-m", "pytest", "tests", "-v"], env=env))

    df = pd.read_csv(a.csv)
    sample = pd.concat([df[df.Class == 0].sample(3, random_state=1),
                        df[df.Class == 1].sample(3, random_state=1)])
    body = {"data": sample[FEATURES].to_dict(orient="records")}

    r, ms = call(url, key, body)
    print(f"\n[Hop le] HTTP {r.status_code} - {ms:.0f} ms")
    out = r.json()
    if isinstance(out, str):
        out = json.loads(out)
    print("Thuc te   :", sample.Class.astype(int).tolist())
    print("Du doan   :", out["predictions"])
    print("Xac suat  :", out["probabilities"])
    print("Nguong    :", out["threshold"])

    r, ms = call(url, key, {"data": [{"Amount": 1}]})
    print(f"\n[Sai dinh dang] HTTP {r.status_code} - {ms:.0f} ms (mong doi khac 200)")
    print(r.text[:300])


if __name__ == "__main__":
    main()