"""TV3 — đánh giá FP32/INT8 trên cùng tensor và chọn cấu hình bằng validation."""

from __future__ import annotations

import csv
import json
from collections import defaultdict
from pathlib import Path

from .common import project_path


METRIC_FIELDS = [
    "experiment_id",
    "checkpoint_id",
    "model_id",
    "model_file",
    "split",
    "calibration_size",
    "calibration_seed",
    "accuracy_pct",
    "macro_f1",
    "model_size_bytes",
]


def _quantize_input(np, values, detail):
    dtype = np.dtype(detail["dtype"])
    values = np.asarray(values, dtype=np.float32)
    if dtype == np.dtype(np.float32):
        return values
    if not np.issubdtype(dtype, np.integer):
        raise ValueError(f"Input dtype TFLite không hỗ trợ: {dtype}")
    scale, zero_point = detail["quantization"]
    if scale <= 0:
        raise ValueError("Input integer thiếu quantization scale hợp lệ")
    limits = np.iinfo(dtype)
    quantized = np.rint(values / np.float32(scale) + np.float32(zero_point))
    return np.clip(quantized, limits.min, limits.max).astype(dtype)


def _dequantize_output(np, values, detail):
    dtype = np.dtype(detail["dtype"])
    if dtype == np.dtype(np.float32):
        return np.asarray(values, dtype=np.float32)
    scale, zero_point = detail["quantization"]
    if scale <= 0:
        raise ValueError("Output integer thiếu quantization scale hợp lệ")
    return (np.asarray(values, dtype=np.float32) - zero_point) * np.float32(scale)


def infer_tflite(tf, np, model_path, x, batch_size=128):
    """Chạy TFLite theo batch và trả logits đã giải lượng tử về float32."""
    x = np.asarray(x, dtype=np.float32)
    interpreter = tf.lite.Interpreter(model_path=str(model_path), num_threads=1)
    interpreter.allocate_tensors()
    first_input = interpreter.get_input_details()[0]
    signature = [int(value) for value in first_input.get("shape_signature", [])]
    dynamic_batch = bool(signature and signature[0] == -1)
    invoke_batch = min(int(batch_size), len(x)) if dynamic_batch else 1
    if dynamic_batch:
        interpreter.resize_tensor_input(
            first_input["index"], [invoke_batch, *x.shape[1:]], strict=False
        )
        interpreter.allocate_tensors()

    input_detail = interpreter.get_input_details()[0]
    output_detail = interpreter.get_output_details()[0]
    outputs = []
    for start in range(0, len(x), invoke_batch):
        current = x[start : start + invoke_batch]
        valid_count = len(current)
        if dynamic_batch and valid_count < invoke_batch:
            padded = np.zeros((invoke_batch, *x.shape[1:]), dtype=np.float32)
            padded[:valid_count] = current
            current = padded
        quantized = _quantize_input(np, current, input_detail)
        interpreter.set_tensor(input_detail["index"], quantized)
        interpreter.invoke()
        raw_output = interpreter.get_tensor(output_detail["index"])
        outputs.append(_dequantize_output(np, raw_output, output_detail)[:valid_count])
    return np.concatenate(outputs, axis=0).astype(np.float32, copy=False)


def confusion_matrix(np, y_true, y_pred, class_count):
    matrix = np.zeros((class_count, class_count), dtype=np.int64)
    np.add.at(matrix, (y_true, y_pred), 1)
    return matrix


def macro_f1(np, matrix):
    scores = []
    for class_id in range(len(matrix)):
        true_positive = matrix[class_id, class_id]
        false_positive = matrix[:, class_id].sum() - true_positive
        false_negative = matrix[class_id, :].sum() - true_positive
        denominator = 2 * true_positive + false_positive + false_negative
        scores.append(0.0 if denominator == 0 else 2 * true_positive / denominator)
    return float(np.mean(scores))


def _evaluate_one(tf, np, artifact, split, x, y, class_count, confusion_dir):
    model_path = project_path(artifact["model_file"])
    if not model_path.is_file():
        raise FileNotFoundError(f"Thiếu model: {artifact['model_file']}")
    logits = infer_tflite(tf, np, model_path, x)
    if logits.shape != (len(x), class_count):
        raise ValueError(f"Output {artifact['model_file']} sai shape: {logits.shape}")
    predictions = logits.argmax(axis=1).astype(np.int64)
    matrix = confusion_matrix(np, y, predictions, class_count)
    matrix_path = confusion_dir / f"{split}_{model_path.stem}.csv"
    np.savetxt(matrix_path, matrix, delimiter=",", fmt="%d")
    return {
        "experiment_id": f"{split}-{model_path.stem}",
        "checkpoint_id": artifact["checkpoint_id"],
        "model_id": artifact["model_id"],
        "model_file": artifact["model_file"],
        "split": split,
        "calibration_size": artifact["calibration_size"],
        "calibration_seed": artifact["calibration_seed"],
        "accuracy_pct": float(np.mean(predictions == y) * 100.0),
        "macro_f1": macro_f1(np, matrix),
        "model_size_bytes": model_path.stat().st_size,
    }


def _select_calibration_size(rows):
    grouped = defaultdict(list)
    for row in rows:
        if row["split"] == "validation" and row["calibration_size"] is not None:
            grouped[int(row["calibration_size"])].append(row)
    if not grouped:
        raise ValueError("Không có kết quả INT8 validation để chọn calibration size")

    aggregates = []
    for size, group in grouped.items():
        mean_accuracy = sum(item["accuracy_pct"] for item in group) / len(group)
        mean_f1 = sum(item["macro_f1"] for item in group) / len(group)
        aggregates.append((mean_accuracy, mean_f1, -size, size, group))
    _, _, _, selected_size, selected_group = max(aggregates)
    representative = max(
        selected_group,
        key=lambda item: (item["accuracy_pct"], item["macro_f1"], -int(item["calibration_seed"])),
    )
    return selected_size, representative


def _write_metrics(rows, destination):
    with Path(destination).open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=METRIC_FIELDS, lineterminator="\n")
        writer.writeheader()
        for row in rows:
            encoded = dict(row)
            for key in ("calibration_size", "calibration_seed"):
                if encoded[key] is None:
                    encoded[key] = ""
            encoded["accuracy_pct"] = f"{row['accuracy_pct']:.6f}"
            encoded["macro_f1"] = f"{row['macro_f1']:.8f}"
            writer.writerow(encoded)


def run(config):
    import numpy as np
    import tensorflow as tf

    model_metadata_path = project_path("models/model_metadata.json")
    dataset_path = project_path("data/processed/har.npz")
    if not model_metadata_path.is_file():
        raise FileNotFoundError("Thiếu models/model_metadata.json; chạy 'python main.py quantize'")
    if not dataset_path.is_file():
        raise FileNotFoundError("Thiếu data/processed/har.npz; chạy 'python main.py prepare'")

    metadata = json.loads(model_metadata_path.read_text(encoding="utf-8"))
    artifacts = metadata.get("models", [])
    fp32_models = [item for item in artifacts if item["model_kind"] == "fp32"]
    int8_models = [item for item in artifacts if item["model_kind"] == "int8"]
    if len(fp32_models) != 1 or not int8_models:
        raise ValueError("model_metadata.json phải có 1 FP32 và ít nhất 1 INT8")
    if metadata.get("labels") != config["dataset"]["labels"]:
        raise ValueError("Thứ tự labels trong metadata không khớp cấu hình")
    if any(not item.get("full_integer") for item in int8_models):
        raise ValueError("Có model INT8 chưa đạt full-integer")

    with np.load(dataset_path) as dataset:
        splits = {
            "validation": (dataset["x_val"], dataset["y_val"]),
            "test": (dataset["x_test"], dataset["y_test"]),
        }
    class_count = len(config["dataset"]["labels"])
    results_dir = project_path(config["paths"]["results"])
    confusion_dir = results_dir / "confusion_matrices"
    confusion_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    x_val, y_val = splits["validation"]
    for artifact in [fp32_models[0], *int8_models]:
        row = _evaluate_one(
            tf, np, artifact, "validation", x_val, y_val, class_count, confusion_dir
        )
        rows.append(row)
        print(
            f"validation {Path(artifact['model_file']).name}: "
            f"accuracy={row['accuracy_pct']:.4f}%, macro-F1={row['macro_f1']:.6f}"
        )

    selected_size, representative_row = _select_calibration_size(rows)
    selected_artifacts = [
        item for item in int8_models if int(item["calibration_size"]) == selected_size
    ]
    x_test, y_test = splits["test"]
    for artifact in [fp32_models[0], *selected_artifacts]:
        row = _evaluate_one(
            tf, np, artifact, "test", x_test, y_test, class_count, confusion_dir
        )
        rows.append(row)
        print(
            f"test {Path(artifact['model_file']).name}: "
            f"accuracy={row['accuracy_pct']:.4f}%, macro-F1={row['macro_f1']:.6f}"
        )

    selection = {
        "checkpoint_id": metadata["checkpoint_id"],
        "policy": metadata["selection_policy"],
        "selected_calibration_size": selected_size,
        "representative_model_file": representative_row["model_file"],
        "representative_calibration_seed": representative_row["calibration_seed"],
        "note": "Test was evaluated only after selecting size from validation results.",
    }
    metrics_path = results_dir / "metrics.csv"
    _write_metrics(rows, metrics_path)
    (results_dir / "selection.json").write_text(
        json.dumps(selection, indent=2), encoding="utf-8"
    )
    print(f"Đã chọn calibration size={selected_size} bằng validation")
    print(f"Đã ghi {len(rows)} dòng vào {metrics_path}")
    return metrics_path
