"""TV3 — tổng hợp metric theo split và calibration size."""

from __future__ import annotations

import csv
import json
import statistics
from collections import defaultdict

from .common import project_path


SUMMARY_FIELDS = [
    "checkpoint_id",
    "split",
    "model_kind",
    "calibration_size",
    "selected",
    "n",
    "accuracy_mean_pct",
    "accuracy_std_pct",
    "delta_accuracy_pp_vs_fp32",
    "macro_f1_mean",
    "macro_f1_std",
    "model_size_mean_bytes",
    "model_size_std_bytes",
    "size_reduction_pct",
    "compression_ratio",
]


def _mean(values):
    return sum(values) / len(values)


def _std(values):
    return None if len(values) < 2 else statistics.stdev(values)


def _fmt(value, digits=6):
    return "" if value is None else f"{value:.{digits}f}"


def run(config):
    results_dir = project_path(config["paths"]["results"])
    metrics_path = results_dir / "metrics.csv"
    selection_path = results_dir / "selection.json"
    if not metrics_path.is_file() or metrics_path.stat().st_size == 0:
        raise FileNotFoundError("Thiếu results/metrics.csv; chạy 'python main.py evaluate'")
    if not selection_path.is_file():
        raise FileNotFoundError("Thiếu results/selection.json; chạy 'python main.py evaluate'")

    with metrics_path.open(newline="", encoding="utf-8") as stream:
        raw_rows = list(csv.DictReader(stream))
    if not raw_rows:
        raise ValueError("results/metrics.csv không có dữ liệu thực nghiệm")
    checkpoint_ids = {row["checkpoint_id"] for row in raw_rows}
    if len(checkpoint_ids) != 1:
        raise ValueError("metrics.csv trộn nhiều checkpoint; không thể tổng hợp an toàn")
    selection = json.loads(selection_path.read_text(encoding="utf-8"))
    selected_size = int(selection["selected_calibration_size"])

    groups = defaultdict(list)
    for raw in raw_rows:
        kind = "fp32" if raw["calibration_size"] == "" else "int8"
        size = None if kind == "fp32" else int(raw["calibration_size"])
        row = {
            **raw,
            "calibration_size": size,
            "accuracy_pct": float(raw["accuracy_pct"]),
            "macro_f1": float(raw["macro_f1"]),
            "model_size_bytes": int(raw["model_size_bytes"]),
        }
        groups[(row["split"], kind, size)].append(row)

    baseline_by_split = {}
    for (split, kind, _), rows in groups.items():
        if kind == "fp32":
            if len(rows) != 1:
                raise ValueError(f"Split {split} phải có đúng 1 baseline FP32")
            baseline_by_split[split] = rows[0]

    summary = []
    for (split, kind, size), rows in sorted(
        groups.items(), key=lambda item: (item[0][0], item[0][1], item[0][2] or -1)
    ):
        baseline = baseline_by_split.get(split)
        if baseline is None:
            raise ValueError(f"Split {split} thiếu baseline FP32")
        accuracies = [row["accuracy_pct"] for row in rows]
        f1_scores = [row["macro_f1"] for row in rows]
        sizes = [row["model_size_bytes"] for row in rows]
        accuracy_mean = _mean(accuracies)
        size_mean = _mean(sizes)
        baseline_size = baseline["model_size_bytes"]
        summary.append(
            {
                "checkpoint_id": rows[0]["checkpoint_id"],
                "split": split,
                "model_kind": kind,
                "calibration_size": size,
                "selected": kind == "int8" and size == selected_size,
                "n": len(rows),
                "accuracy_mean_pct": accuracy_mean,
                "accuracy_std_pct": _std(accuracies),
                "delta_accuracy_pp_vs_fp32": accuracy_mean - baseline["accuracy_pct"],
                "macro_f1_mean": _mean(f1_scores),
                "macro_f1_std": _std(f1_scores),
                "model_size_mean_bytes": size_mean,
                "model_size_std_bytes": _std(sizes),
                "size_reduction_pct": 100.0 * (1.0 - size_mean / baseline_size),
                "compression_ratio": baseline_size / size_mean,
            }
        )

    summary_path = results_dir / "summary.csv"
    with summary_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=SUMMARY_FIELDS, lineterminator="\n")
        writer.writeheader()
        for row in summary:
            writer.writerow(
                {
                    **row,
                    "calibration_size": "" if row["calibration_size"] is None else row["calibration_size"],
                    "selected": "yes" if row["selected"] else "no",
                    "accuracy_mean_pct": _fmt(row["accuracy_mean_pct"]),
                    "accuracy_std_pct": _fmt(row["accuracy_std_pct"]),
                    "delta_accuracy_pp_vs_fp32": _fmt(row["delta_accuracy_pp_vs_fp32"]),
                    "macro_f1_mean": _fmt(row["macro_f1_mean"], 8),
                    "macro_f1_std": _fmt(row["macro_f1_std"], 8),
                    "model_size_mean_bytes": _fmt(row["model_size_mean_bytes"], 1),
                    "model_size_std_bytes": _fmt(row["model_size_std_bytes"], 1),
                    "size_reduction_pct": _fmt(row["size_reduction_pct"]),
                    "compression_ratio": _fmt(row["compression_ratio"]),
                }
            )

    validation_rows = [row for row in summary if row["split"] == "validation"]
    test_rows = [row for row in summary if row["split"] == "test"]
    markdown = [
        "# Tổng hợp kết quả thực nghiệm",
        "",
        f"Checkpoint: `{next(iter(checkpoint_ids))}`.",
        f"Calibration size được chọn bằng validation: **{selected_size}**.",
        "Độ lệch chuẩn là sample standard deviation qua các seed; n=1 ghi N/A.",
        "",
        "## Validation — dùng để chọn cấu hình",
        "",
        "| Model | n | Accuracy mean ± std (%) | Macro-F1 mean ± std | Bytes |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    for row in validation_rows:
        name = "FP32" if row["model_kind"] == "fp32" else f"INT8 n={row['calibration_size']}"
        acc_std = "N/A" if row["accuracy_std_pct"] is None else f"{row['accuracy_std_pct']:.4f}"
        f1_std = "N/A" if row["macro_f1_std"] is None else f"{row['macro_f1_std']:.6f}"
        markdown.append(
            f"| {name} | {row['n']} | {row['accuracy_mean_pct']:.4f} ± {acc_std} | "
            f"{row['macro_f1_mean']:.6f} ± {f1_std} | {row['model_size_mean_bytes']:.0f} |"
        )
    markdown.extend(
        [
            "",
            "## Test — chỉ sau khi chốt calibration size",
            "",
            "| Model | n | Accuracy mean ± std (%) | Δ so với FP32 (điểm %) | Macro-F1 | Giảm size |",
            "| --- | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for row in test_rows:
        name = "FP32" if row["model_kind"] == "fp32" else f"INT8 n={row['calibration_size']}"
        acc_std = "N/A" if row["accuracy_std_pct"] is None else f"{row['accuracy_std_pct']:.4f}"
        markdown.append(
            f"| {name} | {row['n']} | {row['accuracy_mean_pct']:.4f} ± {acc_std} | "
            f"{row['delta_accuracy_pp_vs_fp32']:+.4f} | {row['macro_f1_mean']:.6f} | "
            f"{row['size_reduction_pct']:.2f}% |"
        )
    markdown.extend(
        [
            "",
            "Không diễn giải kết quả validation như kết quả test cuối cùng. Các số trên là số đo từ artifact thật, không phải số minh họa.",
            "",
        ]
    )
    (results_dir / "summary.md").write_text("\n".join(markdown), encoding="utf-8")
    print(f"Đã ghi {summary_path} và results/summary.md")
    return summary_path
