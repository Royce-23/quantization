import csv
import json
from pathlib import Path

import numpy as np
import pytest

from src.calibration import select_calibration
from src.evaluate import (
    _dequantize_output,
    _quantize_input,
    confusion_matrix,
    infer_tflite,
    macro_f1,
)


ROOT = Path(__file__).resolve().parents[1]


def test_calibration_is_reproducible_unique_and_train_bounded():
    x = np.arange(30 * 4 * 2, dtype=np.float32).reshape(30, 4, 2)
    selected_a, indices_a = select_calibration(x, 12, 123)
    selected_b, indices_b = select_calibration(x, 12, 123)
    np.testing.assert_array_equal(indices_a, indices_b)
    np.testing.assert_array_equal(selected_a, selected_b)
    assert len(np.unique(indices_a)) == 12
    np.testing.assert_array_equal(selected_a, x[indices_a])
    with pytest.raises(ValueError):
        select_calibration(x, 31, 123)


def test_affine_input_quantization_clips_and_dequantizes():
    detail = {"dtype": np.int8, "quantization": (0.5, -3)}
    values = np.array([-1000.0, -0.5, 0.0, 1.0, 1000.0], dtype=np.float32)
    quantized = _quantize_input(np, values, detail)
    np.testing.assert_array_equal(
        quantized, np.array([-128, -4, -3, -1, 127], dtype=np.int8)
    )
    recovered = _dequantize_output(np, quantized, detail)
    np.testing.assert_allclose(recovered[1:4], values[1:4], atol=0.25)


def test_confusion_matrix_and_macro_f1_contract():
    labels = np.arange(6, dtype=np.int64)
    matrix = confusion_matrix(np, labels, labels, 6)
    np.testing.assert_array_equal(matrix, np.eye(6, dtype=np.int64))
    assert macro_f1(np, matrix) == 1.0


def test_quantization_artifacts_are_full_integer_and_traceable():
    metadata_path = ROOT / "models" / "model_metadata.json"
    if not metadata_path.is_file():
        pytest.skip("Chạy 'python main.py quantize' để sinh artifact")
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    models = metadata["models"]
    assert len([item for item in models if item["model_kind"] == "fp32"]) == 1
    int8_models = [item for item in models if item["model_kind"] == "int8"]
    assert len(int8_models) == 9
    assert {(item["calibration_size"], item["calibration_seed"]) for item in int8_models} == {
        (size, seed) for size in (50, 200, 500) for seed in (11, 22, 33)
    }
    for item in int8_models:
        assert item["full_integer"] is True
        assert item["float_tensor_count"] == 0
        assert item["input"]["dtype"] == "int8"
        assert item["output"]["dtype"] == "int8"
        assert len(item["calibration_indices"]) == item["calibration_size"]
        assert len(set(item["calibration_indices"])) == item["calibration_size"]
        assert (ROOT / item["model_file"]).stat().st_size == item["size_bytes"]


def test_evaluation_keeps_test_out_of_calibration_selection():
    selection_path = ROOT / "results" / "selection.json"
    metrics_path = ROOT / "results" / "metrics.csv"
    if not selection_path.is_file() or not metrics_path.is_file():
        pytest.skip("Chạy evaluate để sinh kết quả")
    selection = json.loads(selection_path.read_text(encoding="utf-8"))
    with metrics_path.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    val_int8 = [row for row in rows if row["split"] == "validation" and row["calibration_size"]]
    test_int8 = [row for row in rows if row["split"] == "test" and row["calibration_size"]]
    assert len(val_int8) == 9
    assert len(test_int8) == 3
    assert {int(row["calibration_size"]) for row in test_int8} == {
        int(selection["selected_calibration_size"])
    }
    assert "validation" in selection["policy"]


def test_fp32_and_int8_tflite_run_on_same_normalized_samples():
    tf = pytest.importorskip("tensorflow")
    sanity_path = ROOT / "data" / "processed" / "sanity_samples.npz"
    fp32_path = ROOT / "models" / "baseline_fp32.tflite"
    int8_path = ROOT / "models" / "int8_n500_s22.tflite"
    if not all(path.is_file() for path in (sanity_path, fp32_path, int8_path)):
        pytest.skip("Chưa có đủ artifact để chạy TFLite")
    with np.load(sanity_path) as sanity:
        x = sanity["x"][:2].astype(np.float32)
    fp32_logits = infer_tflite(tf, np, fp32_path, x)
    int8_logits = infer_tflite(tf, np, int8_path, x)
    assert fp32_logits.shape == int8_logits.shape == (2, 6)
    assert np.isfinite(fp32_logits).all()
    assert np.isfinite(int8_logits).all()
