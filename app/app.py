import json
import os
import sys
import time
import pandas as pd
import requests
import streamlit as st

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts"))
from call_api import FEATURES, get_endpoint

CSV_PATH = "data/creditcard.csv"
REPORT_DIR = "reports"

st.set_page_config(page_title="Phát hiện gian lận thẻ tín dụng", layout="wide")
st.title("Hệ thống MLOps phát hiện gian lận thẻ tín dụng")

# ---------- Thanh ben: ket noi endpoint ----------
with st.sidebar:
    st.header("Kết nối endpoint")
    if st.button("Lấy URL và khóa từ Azure CLI"):
        try:
            url, key = get_endpoint()
            st.session_state["url"], st.session_state["key"] = url, key
            st.success("Đã lấy thông tin endpoint.")
        except Exception as e:
            st.error(f"Không lấy được: {type(e).__name__}. Kiểm tra az login và endpoint đang chạy.")
    url = st.text_input("Scoring URI", st.session_state.get("url", os.getenv("ENDPOINT_URL", "")))
    key = st.text_input("Khóa (không hiển thị)", st.session_state.get("key", os.getenv("ENDPOINT_KEY", "")),
                        type="password")


def call_api(rows):
    headers = {"Authorization": f"Bearer {key}", "Content-Type": "application/json"}
    t0 = time.time()
    r = requests.post(url, json={"data": rows}, headers=headers, timeout=60)
    return r, (time.time() - t0) * 1000


tab1, tab2 = st.tabs(["Dự đoán giao dịch", "Kết quả mô hình"])

# ---------- Tab 1: du doan ----------
with tab1:
    mode = st.radio("Nguồn dữ liệu", ["Mẫu từ dataset", "Dán JSON", "Tải file CSV"], horizontal=True)
    rows, truth = None, None

    if mode == "Mẫu từ dataset":
        if not os.path.exists(CSV_PATH):
            st.warning(f"Không thấy {CSV_PATH}.")
        else:
            c1, c2, c3 = st.columns(3)
            n_norm = c1.number_input("Số giao dịch bình thường", 0, 50, 3)
            n_fraud = c2.number_input("Số giao dịch gian lận", 0, 50, 3)
            seed = c3.number_input("Seed", 0, 9999, 1)
            if st.button("Lấy mẫu"):
                df = pd.read_csv(CSV_PATH)
                s = pd.concat([df[df.Class == 0].sample(int(n_norm), random_state=int(seed)),
                               df[df.Class == 1].sample(int(n_fraud), random_state=int(seed))])
                st.session_state["sample"] = s
            if "sample" in st.session_state:
                s = st.session_state["sample"]
                rows = s[FEATURES].to_dict(orient="records")
                truth = s["Class"].astype(int).tolist()
                st.dataframe(s[["Time", "V1", "V2", "V14", "Amount", "Class"]], use_container_width=True)

    elif mode == "Dán JSON":
        example = json.dumps({"data": [{**{c: 0.0 for c in FEATURES}, "Amount": 100.0}]}, indent=2)
        text = st.text_area('Body JSON dạng {"data": [ {...30 cột...} ]}', example, height=260)
        try:
            rows = json.loads(text)["data"]
        except Exception:
            st.error("JSON không hợp lệ.")
            rows = None

    else:
        up = st.file_uploader("CSV có các cột Time, V1..V28, Amount", type="csv")
        if up is not None:
            d = pd.read_csv(up)
            missing = [c for c in FEATURES if c not in d.columns]
            if missing:
                st.error(f"Thiếu cột: {missing}")
            else:
                rows = d[FEATURES].head(200).to_dict(orient="records")
                truth = d["Class"].astype(int).head(200).tolist() if "Class" in d.columns else None
                st.write(f"Gửi {len(rows)} giao dịch đầu tiên.")

    if st.button("Gọi API dự đoán", type="primary", disabled=not (rows and url and key)):
        try:
            r, ms = call_api(rows)
            if r.status_code != 200:
                st.error(f"HTTP {r.status_code}: {r.text[:300]}")
            else:
                out = r.json()
                if isinstance(out, str):
                    out = json.loads(out)
                pred_col = "Dự đoán (0=thường, 1=gian lận)"
                res = pd.DataFrame({pred_col: out["predictions"],
                                    "Xác suất gian lận": out["probabilities"]})
                res["Kết luận"] = res[pred_col].map({0: "Bình thường", 1: "GIAN LẬN"})
                if truth is not None and len(truth) == len(res):
                    res.insert(0, "Thực tế", truth)
                st.success(f"HTTP 200 - {ms:.0f} ms - ngưỡng {out['threshold']:.4f}")
                st.dataframe(res, use_container_width=True)
        except Exception as e:
            st.error(f"Lỗi gọi API: {type(e).__name__}")

# ---------- Tab 2: ket qua mo hinh ----------
with tab2:
    csv = os.path.join(REPORT_DIR, "experiments.csv")
    if not os.path.exists(csv):
        st.info("Chưa có reports/experiments.csv. Chạy scripts/run_sweep.py rồi scripts/plot_experiments.py.")
    else:
        st.subheader("Bảng kết quả các cấu hình")
        st.dataframe(pd.read_csv(csv).drop(columns=["label"], errors="ignore"), use_container_width=True)
        for fn in ["fig1_pr_auc_theo_cau_hinh.png", "fig2_recall_precision.png", "fig3_xu_ly_mat_can_bang.png"]:
            p = os.path.join(REPORT_DIR, fn)
            if os.path.exists(p):
                st.image(p)