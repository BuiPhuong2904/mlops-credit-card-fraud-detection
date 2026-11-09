import argparse
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

PARAMS = ["model_type", "imbalance", "n_estimators", "max_depth", "learning_rate"]
METRICS = ["recall", "precision", "f1", "pr_auc", "train_seconds"]
COLORS = {"rf": "#2a6fdb", "lgbm": "#e07b00"}
MARKERS = {"smote": "o", "class_weight": "s", "none": "^"}


def load_runs(tracking_uri, experiment):
    import mlflow

    mlflow.set_tracking_uri(tracking_uri)
    raw = mlflow.search_runs(experiment_names=[experiment],
                             filter_string="attributes.status = 'FINISHED'")
    if raw.empty:
        raise SystemExit(f"Khong co run nao trong experiment '{experiment}'.")
    df = pd.DataFrame()
    for c in PARAMS:
        df[c] = raw.get(f"params.{c}")
    for c in METRICS:
        df[c] = raw.get(f"metrics.{c}")
    df["start_time"] = raw["start_time"]
    return df


def clean(df):
    df = df.copy()
    for c in ["n_estimators", "max_depth", "learning_rate"] + METRICS:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.dropna(subset=["model_type", "pr_auc", "recall"])
    if "start_time" in df.columns:
        df = df.sort_values("start_time", ascending=False)
    df = df.drop_duplicates(subset=PARAMS, keep="first")
    depth = df["max_depth"].map(lambda v: "-" if pd.isna(v) else str(int(v)))
    df["label"] = (df["model_type"] + " | " + df["imbalance"].fillna("?") +
                   " | n=" + df["n_estimators"].fillna(0).astype(int).astype(str) +
                   " d=" + depth)
    df["label"] = df["label"] + df.groupby("label").cumcount().map(lambda k: "" if k == 0 else f" (#{k + 1})")
    return df.sort_values("pr_auc", ascending=False).reset_index(drop=True)


def fig_pr_auc(df, path):
    d = df.iloc[::-1].reset_index(drop=True)
    pos = list(range(len(d)))
    fig, ax = plt.subplots(figsize=(9, max(3.5, 0.45 * len(d) + 1)))
    ax.barh(pos, d["pr_auc"], color=[COLORS.get(m, "gray") for m in d["model_type"]])
    ax.set_yticks(pos)
    ax.set_yticklabels(d["label"])
    for y, v, r in zip(pos, d["pr_auc"], d["recall"]):
        ax.text(v + 0.005, y, f"{v:.3f}  (recall {r:.2f})", va="center", fontsize=8)
    ax.set_xlim(0, min(1.25, d["pr_auc"].max() + 0.3))
    ax.set_xlabel("PR-AUC (tập kiểm tra)")
    ax.set_title("So sánh PR-AUC theo cấu hình huấn luyện")
    handles = [plt.Rectangle((0, 0), 1, 1, color=c) for c in COLORS.values()]
    ax.legend(handles, COLORS.keys(), title="Loại mô hình", loc="lower right")
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def fig_recall_precision(df, path):
    fig, ax = plt.subplots(figsize=(7.5, 5.5))
    for (m, imb), g in df.groupby(["model_type", df["imbalance"].fillna("?")]):
        ax.scatter(g["recall"], g["precision"], s=70, color=COLORS.get(m, "gray"),
                   marker=MARKERS.get(imb, "D"), label=f"{m} | {imb}", alpha=0.85)
    ax.axvline(0.80, color="red", linestyle="--", linewidth=1)
    ax.text(0.801, ax.get_ylim()[1] * 0.97, "yêu cầu Recall > 0.80", color="red", fontsize=8, va="top")
    ax.set_xlabel("Recall (lớp gian lận)")
    ax.set_ylabel("Precision (lớp gian lận)")
    ax.set_title("Đánh đổi Recall - Precision theo loại mô hình và cách xử lý mất cân bằng")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def fig_imbalance(df, path):
    g = df.assign(imbalance=df["imbalance"].fillna("?")).groupby(["model_type", "imbalance"])[
        ["pr_auc", "recall"]].mean().reset_index()
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.2), sharey=False)
    for ax, metric, title in zip(axes, ["pr_auc", "recall"], ["PR-AUC trung bình", "Recall trung bình"]):
        piv = g.pivot(index="imbalance", columns="model_type", values=metric)
        piv.plot.bar(ax=ax, color=[COLORS.get(c, "gray") for c in piv.columns], rot=0, width=0.7)
        for cont in ax.containers:
            ax.bar_label(cont, fmt="%.3f", fontsize=8, padding=2)
        ax.set_ylim(0, 1.2)
        ax.set_title(title)
        ax.set_xlabel("Cách xử lý mất cân bằng")
        ax.grid(axis="y", alpha=0.3)
        ax.legend(title="Loại mô hình", loc="upper center", ncol=2, fontsize=8, title_fontsize=8)
    fig.suptitle("Ảnh hưởng của cách xử lý mất cân bằng lớp (trung bình các cấu hình cùng nhóm)")
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tracking-uri", default="sqlite:///mlflow.db")
    ap.add_argument("--experiment", default="fraud-detection")
    ap.add_argument("--from-csv", help="doc tu CSV thay vi MLflow (de thu nghiem)")
    ap.add_argument("--out", default="reports")
    a = ap.parse_args()

    df = pd.read_csv(a.from_csv) if a.from_csv else load_runs(a.tracking_uri, a.experiment)
    df = clean(df)
    os.makedirs(a.out, exist_ok=True)
    df.drop(columns=["start_time"], errors="ignore").to_csv(f"{a.out}/experiments.csv", index=False)
    fig_pr_auc(df, f"{a.out}/fig1_pr_auc_theo_cau_hinh.png")
    fig_recall_precision(df, f"{a.out}/fig2_recall_precision.png")
    fig_imbalance(df, f"{a.out}/fig3_xu_ly_mat_can_bang.png")
    print(df[["label", "recall", "precision", "f1", "pr_auc", "train_seconds"]].round(4).to_string(index=False))
    print(f"\nDa ghi {len(df)} run vao {a.out}/ (experiments.csv + 3 hinh PNG)")


if __name__ == "__main__":
    main()