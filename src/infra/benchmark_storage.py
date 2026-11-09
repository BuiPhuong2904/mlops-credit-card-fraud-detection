import argparse, os, time
import numpy as np
import pandas as pd
from azure.storage.blob import BlobServiceClient

parser = argparse.ArgumentParser()
parser.add_argument("--label", required=True, help="ten region, vd eastasia")
parser.add_argument("--skip-local", action="store_true")
args = parser.parse_args()

CSV, PQ = "data/creditcard.csv", "data/creditcard.parquet"
if not os.path.exists(PQ):
    pd.read_csv(CSV).to_parquet(PQ, index=False)


def timeit(fn, n):
    ts = []
    for _ in range(n):
        t0 = time.perf_counter()
        fn()
        ts.append(time.perf_counter() - t0)
    return round(float(np.mean(ts)), 2)


rows = []
if not args.skip_local:
    df = pd.read_csv(CSV)
    rows.append(("Local", "CSV doc (s)", timeit(lambda: pd.read_csv(CSV), 3)))
    rows.append(("Local", "CSV ghi (s)", timeit(lambda: df.to_csv("data/tmp.csv", index=False), 3)))
    rows.append(("Local", "Parquet ghi (s)", timeit(lambda: df.to_parquet(PQ, index=False), 3)))
    rows.append(("Local", "Parquet doc (s)", timeit(lambda: pd.read_parquet(PQ), 3)))
    print("Dung luong CSV (MB):", round(os.path.getsize(CSV) / 1e6, 1))
    print("Dung luong Parquet (MB):", round(os.path.getsize(PQ) / 1e6, 1))

svc = BlobServiceClient.from_connection_string(os.environ["AZURE_STORAGE_CONNECTION_STRING"])
cont = svc.get_container_client("benchmark")
if not cont.exists():
    cont.create_container()

for name, path in [("CSV", CSV), ("Parquet", PQ)]:
    blob = cont.get_blob_client(os.path.basename(path))

    def up(path=path, blob=blob):
        with open(path, "rb") as f:
            blob.upload_blob(f, overwrite=True)

    def down(blob=blob):
        blob.download_blob().readall()

    rows.append((args.label, f"{name} upload (s)", timeit(up, 2)))
    rows.append((args.label, f"{name} download (s)", timeit(down, 2)))

tiny = cont.get_blob_client("ping.txt")
tiny.upload_blob(b"x", overwrite=True)
tiny.get_blob_properties()  # khoi dong ket noi
lat = []
for _ in range(20):
    t0 = time.perf_counter()
    tiny.get_blob_properties()
    lat.append((time.perf_counter() - t0) * 1000)
rows.append((args.label, "Do tre TB 20 lan (ms)", round(float(np.mean(lat)), 1)))

out = pd.DataFrame(rows, columns=["Noi do", "Phep do", "Gia tri"])
print(out.to_string(index=False))
out.to_csv(f"data/benchmark_{args.label}.csv", index=False)