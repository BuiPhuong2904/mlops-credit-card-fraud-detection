import argparse
import random
import statistics
import time
import pandas as pd

from call_api import FEATURES, call, get_endpoint


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default="data/creditcard.csv")
    ap.add_argument("--n", type=int, default=300, help="so lan goi")
    ap.add_argument("--pause", type=float, default=0.5, help="nghi giua cac lan goi (giay)")
    ap.add_argument("--max-batch", type=int, default=5, help="so giao dich toi da moi lan goi")
    a = ap.parse_args()

    url, key = get_endpoint()
    df = pd.read_csv(a.csv)
    pool = df[FEATURES]
    random.seed(0)

    ok, fail, lat = 0, 0, []
    t_start = time.time()
    for i in range(a.n):
        k = random.randint(1, a.max_batch)
        rows = pool.sample(k).to_dict(orient="records")
        try:
            r, ms = call(url, key, {"data": rows})
            lat.append(ms)
            if r.status_code == 200:
                ok += 1
            else:
                fail += 1
        except Exception as e:
            fail += 1
            print("Loi ket noi:", type(e).__name__)
        if (i + 1) % 25 == 0:
            print(f"{i + 1}/{a.n} - ok={ok} fail={fail}")
        time.sleep(a.pause)

    dur = time.time() - t_start
    lat.sort()
    print(f"\nXong {a.n} lan goi trong {dur:.0f}s: thanh cong={ok}, loi={fail}")
    if lat:
        print(f"Do tre phia client (ms): trung binh={statistics.mean(lat):.0f}, "
              f"p50={lat[len(lat)//2]:.0f}, p95={lat[int(len(lat)*0.95)-1]:.0f}, max={lat[-1]:.0f}")


if __name__ == "__main__":
    main()