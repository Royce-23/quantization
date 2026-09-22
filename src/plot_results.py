"""TV3 — vẽ biểu đồ trực tiếp từ kết quả thực nghiệm đã lưu."""

from __future__ import annotations

import csv
import json
from pathlib import Path

from .common import project_path


def _read_summary(path):
    with Path(path).open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    parsed = []
    for row in rows:
        parsed.append(
            {
                **row,
                "calibration_size": None if row["calibration_size"] == "" else int(row["calibration_size"]),
                "n": int(row["n"]),
                "accuracy_mean_pct": float(row["accuracy_mean_pct"]),
                "accuracy_std_pct": None if row["accuracy_std_pct"] == "" else float(row["accuracy_std_pct"]),
                "macro_f1_mean": float(row["macro_f1_mean"]),
                "macro_f1_std": None if row["macro_f1_std"] == "" else float(row["macro_f1_std"]),
                "model_size_mean_bytes": float(row["model_size_mean_bytes"]),
            }
        )
    return parsed


def _annotate_bars(axis, bars, formatter):
    for bar in bars:
        axis.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height(),
            formatter(bar.get_height()),
            ha="center",
            va="bottom",
            fontsize=9,
        )


def run(config):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    results_dir = project_path(config["paths"]["results"])
    summary_path = results_dir / "summary.csv"
    selection_path = results_dir / "selection.json"
    if not summary_path.is_file():
        raise FileNotFoundError("Thiếu results/summary.csv; chạy 'python main.py summarize'")
    if not selection_path.is_file():
        raise FileNotFoundError("Thiếu results/selection.json; chạy 'python main.py evaluate'")
    rows = _read_summary(summary_path)
    selection = json.loads(selection_path.read_text(encoding="utf-8"))
    output_dir = results_dir / "figures" / "quantization"
    output_dir.mkdir(parents=True, exist_ok=True)

    validation = [row for row in rows if row["split"] == "validation"]
    baseline = next(row for row in validation if row["model_kind"] == "fp32")
    int8 = sorted(
        (row for row in validation if row["model_kind"] == "int8"),
        key=lambda row: row["calibration_size"],
    )
    sizes = [row["calibration_size"] for row in int8]

    fig, axis = plt.subplots(figsize=(8, 5))
    axis.errorbar(
        sizes,
        [row["accuracy_mean_pct"] for row in int8],
        yerr=[row["accuracy_std_pct"] or 0.0 for row in int8],
        marker="o",
        capsize=5,
        linewidth=2,
        label="INT8: mean ± SD (3 seeds)",
    )
    axis.axhline(baseline["accuracy_mean_pct"], color="tab:orange", linestyle="--", label="FP32")
    axis.set_xticks(sizes)
    axis.set_xlabel("Số cửa sổ calibration (train)")
    axis.set_ylabel("Validation accuracy (%)")
    axis.set_title("Ảnh hưởng của calibration size đến accuracy INT8")
    axis.grid(alpha=0.25)
    axis.legend()
    fig.tight_layout()
    fig.savefig(output_dir / "accuracy_vs_calibration.png", dpi=180)
    plt.close(fig)

    fig, axis = plt.subplots(figsize=(8, 5))
    axis.errorbar(
        sizes,
        [row["macro_f1_mean"] for row in int8],
        yerr=[row["macro_f1_std"] or 0.0 for row in int8],
        marker="o",
        capsize=5,
        linewidth=2,
        color="tab:green",
        label="INT8: mean ± SD (3 seeds)",
    )
    axis.axhline(baseline["macro_f1_mean"], color="tab:orange", linestyle="--", label="FP32")
    axis.set_xticks(sizes)
    axis.set_xlabel("Số cửa sổ calibration (train)")
    axis.set_ylabel("Validation macro-F1")
    axis.set_title("Ảnh hưởng của calibration size đến macro-F1 INT8")
    axis.grid(alpha=0.25)
    axis.legend()
    fig.tight_layout()
    fig.savefig(output_dir / "macro_f1_vs_calibration.png", dpi=180)
    plt.close(fig)

    size_rows = [baseline, *int8]
    labels = ["FP32", *[f"INT8\nn={row['calibration_size']}" for row in int8]]
    fig, axis = plt.subplots(figsize=(8, 5))
    bars = axis.bar(
        labels,
        [row["model_size_mean_bytes"] / 1024 for row in size_rows],
        color=["tab:orange", "tab:blue", "tab:blue", "tab:blue"],
    )
    _annotate_bars(axis, bars, lambda value: f"{value:.1f} KB")
    axis.set_ylabel("Dung lượng file TFLite (KiB)")
    axis.set_title("Dung lượng FP32 và full-integer INT8")
    axis.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_dir / "model_size_comparison.png", dpi=180)
    plt.close(fig)

    labels_text = config["dataset"]["labels"]
    representative_stem = Path(selection["representative_model_file"]).stem
    matrix_files = [
        ("FP32", results_dir / "confusion_matrices" / "test_baseline_fp32.csv"),
        (
            f"INT8 n={selection['selected_calibration_size']}, seed={selection['representative_calibration_seed']}",
            results_dir / "confusion_matrices" / f"test_{representative_stem}.csv",
        ),
    ]
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))
    image = None
    for axis, (title, matrix_path) in zip(axes, matrix_files):
        if not matrix_path.is_file():
            raise FileNotFoundError(f"Thiếu confusion matrix: {matrix_path}")
        matrix = np.loadtxt(matrix_path, delimiter=",", dtype=np.int64)
        image = axis.imshow(matrix, cmap="Blues")
        axis.set_title(f"{title} — test")
        axis.set_xlabel("Nhãn dự đoán")
        axis.set_ylabel("Nhãn thật")
        axis.set_xticks(range(len(labels_text)), labels_text, rotation=45, ha="right", fontsize=8)
        axis.set_yticks(range(len(labels_text)), labels_text, fontsize=8)
        threshold = matrix.max() / 2
        for row in range(matrix.shape[0]):
            for column in range(matrix.shape[1]):
                axis.text(
                    column,
                    row,
                    str(matrix[row, column]),
                    ha="center",
                    va="center",
                    color="white" if matrix[row, column] > threshold else "black",
                    fontsize=8,
                )
    # Mỗi ô đã in số lượng nên không cần colorbar; bỏ colorbar giúp hai ma trận
    # không bị che khi tên lớp dài.
    fig.subplots_adjust(left=0.08, right=0.98, bottom=0.24, top=0.9, wspace=0.36)
    fig.savefig(output_dir / "confusion_matrix_test.png", dpi=180)
    plt.close(fig)

    print(f"Đã tạo 4 biểu đồ tại {output_dir}")
    return output_dir
