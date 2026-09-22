"""Test cho tính năng trích xuất tham số C và suy luận C (TASK-10).

Xem .bangiao/ke-hoach.md mục 8 cho tiêu chí nghiệm thu đầy đủ.
"""
import ast
import importlib
import json
import re
import shutil
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]

HAS_COMPILER = (shutil.which("cc") or shutil.which("gcc")) and shutil.which("make")


# ---------------------------------------------------------------------------
# 8.1 — Không cần TensorFlow, không cần compiler
# ---------------------------------------------------------------------------


def test_export_c_module_imports_without_heavy_deps():
    module = importlib.import_module("src.export_c")
    assert callable(module.run)

    source = (ROOT / "src" / "export_c.py").read_text(encoding="utf-8")
    tree = ast.parse(source)
    top_level_modules = set()
    for node in tree.body:
        if isinstance(node, ast.Import):
            for alias in node.names:
                top_level_modules.add(alias.name.split(".")[0])
        elif isinstance(node, ast.ImportFrom):
            if node.module:
                top_level_modules.add(node.module.split(".")[0])
    assert "numpy" not in top_level_modules
    assert "tensorflow" not in top_level_modules


def test_main_registers_export_c_stage():
    import main

    assert "export-c" in main.STAGES
    assert main.STAGES["export-c"] == "export_c"


def test_config_has_export_c_section():
    from src.common import load_config

    config = load_config()
    assert "export_c" in config
    export_cfg = config["export_c"]
    required_fields = {
        "checkpoint", "metadata", "preprocess", "sanity_samples",
        "params_npz", "manifest", "firmware_dir", "values_per_line",
        "float_format",
    }
    assert required_fields <= export_cfg.keys()
    # Bẫy CI quan trọng: không được thêm key mới vào config["paths"].
    assert set(config["paths"].keys()) == {
        "raw_data", "processed_data", "models", "results",
    }


PARAMS_HEADER = ROOT / "firmware" / "include" / "har_resnet_params.h"
PARAMS_SOURCE = ROOT / "firmware" / "src" / "har_resnet_params.c"
GOLDEN_HEADER = ROOT / "firmware" / "include" / "har_golden.h"
GOLDEN_SOURCE = ROOT / "firmware" / "src" / "har_golden.c"
MANIFEST_PATH = ROOT / "models" / "params_manifest.json"
METADATA_PATH = ROOT / "models" / "baseline_metadata.json"


def _require_generated(path):
    if not path.is_file():
        pytest.skip(f"Chưa sinh {path}; chạy 'python main.py export-c' trước")


def test_generated_params_header_contract():
    _require_generated(PARAMS_HEADER)
    text = PARAMS_HEADER.read_text(encoding="utf-8")
    metadata = json.loads(METADATA_PATH.read_text(encoding="utf-8"))

    assert "#define HAR_PARAM_COUNT   10726" in text
    assert "#define HAR_NUM_CLASSES     6" in text
    assert "#define HAR_INPUT_TIMESTEPS 128" in text
    assert "#define HAR_INPUT_CHANNELS  9" in text
    assert "#define HAR_FILTERS         32" in text
    assert f'#define HAR_CHECKPOINT_ID "{metadata["checkpoint_id"]}"' in text

    weight_declarations = re.findall(r"extern const float har_\w+\[", text)
    # 5 lớp Conv x (w,b) + dense x (w,b) = 12, cộng har_input_mean/std.
    assert len(weight_declarations) >= 12
    assert "extern const float har_input_mean[HAR_INPUT_CHANNELS];" in text
    assert "extern const float har_input_std[HAR_INPUT_CHANNELS];" in text


def test_generated_params_source_has_no_nan_inf():
    _require_generated(PARAMS_SOURCE)
    text = PARAMS_SOURCE.read_text(encoding="utf-8")
    lowered = text.lower()
    assert "nan" not in lowered
    assert "inf" not in lowered

    literals = re.findall(r"-?\d[\d.eE+-]*f(?=[,\s])", text)
    assert len(literals) == 10726 + 9 + 9


def test_manifest_matches_metadata():
    _require_generated(MANIFEST_PATH)
    manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    metadata = json.loads(METADATA_PATH.read_text(encoding="utf-8"))

    assert manifest["checkpoint_id"] == metadata["checkpoint_id"]
    assert manifest["parameter_count"] == 10726
    assert len(manifest["layers"]) == 6
    layer_names = {layer["name"] for layer in manifest["layers"]}
    assert layer_names == {
        "block1_conv1", "block1_conv2", "block1_projection",
        "block2_conv1", "block2_conv2", "activity_logits",
    }


# ---------------------------------------------------------------------------
# 8.2 — Cần NumPy, không cần TensorFlow
# ---------------------------------------------------------------------------

PARAMS_NPZ = ROOT / "models" / "params_fp32.npz"
SANITY_NPZ = ROOT / "data" / "processed" / "sanity_samples.npz"


def test_reference_forward_matches_saved_logits():
    if not PARAMS_NPZ.is_file():
        pytest.skip("Chưa sinh models/params_fp32.npz; chạy 'python main.py export-c' trước")
    import numpy as np

    from src.reference_forward import forward_batch, load_params

    params = load_params()
    with np.load(SANITY_NPZ) as sanity:
        x = sanity["x"]
        expected_logits = sanity["baseline_logits"]
        expected_pred = sanity["baseline_predictions"]

    logits = forward_batch(params, x)
    assert np.abs(logits - expected_logits).max() < 1e-4
    assert (logits.argmax(axis=1) == expected_pred).all()


def test_reference_ops_shapes():
    import numpy as np

    from src.reference_forward import conv1d_same, dense, global_max_pool, maxpool1d_2

    rng = np.random.default_rng(0)
    x = rng.normal(size=(128, 9)).astype(np.float32)
    w = rng.normal(size=(3, 9, 32)).astype(np.float32)
    b = rng.normal(size=(32,)).astype(np.float32)
    conv_out = conv1d_same(x, w, b)
    assert conv_out.shape == (128, 32)

    pooled = maxpool1d_2(conv_out)
    assert pooled.shape == (64, 32)

    gmp = global_max_pool(pooled)
    assert gmp.shape == (32,)

    dense_w = rng.normal(size=(32, 6)).astype(np.float32)
    dense_b = rng.normal(size=(6,)).astype(np.float32)
    logits = dense(gmp, dense_w, dense_b)
    assert logits.shape == (6,)


def test_conv1d_same_padding_behaviour():
    import numpy as np

    from src.reference_forward import conv1d_same

    x = np.arange(1, 6, dtype=np.float32).reshape(5, 1)  # [1,2,3,4,5]
    b = np.zeros((1,), dtype=np.float32)

    identity_kernel = np.array([[[0.0]], [[1.0]], [[0.0]]], dtype=np.float32)
    out_identity = conv1d_same(x, identity_kernel, b)
    np.testing.assert_allclose(out_identity.reshape(-1), x.reshape(-1))

    shift_kernel = np.array([[[1.0]], [[0.0]], [[0.0]]], dtype=np.float32)
    out_shift = conv1d_same(x, shift_kernel, b)
    expected = np.array([0.0, 1.0, 2.0, 3.0, 4.0], dtype=np.float32)
    np.testing.assert_allclose(out_shift.reshape(-1), expected)


def test_normalize_roundtrip():
    import numpy as np

    from src.reference_forward import normalize

    mean = np.arange(9, dtype=np.float32)
    std = np.full((9,), 2.0, dtype=np.float32)
    params = {"input_mean": mean, "input_std": std}

    raw_mean = np.tile(mean, (128, 1))
    np.testing.assert_allclose(normalize(params, raw_mean), 0.0, atol=1e-6)

    raw_mean_plus_std = np.tile(mean + std, (128, 1))
    np.testing.assert_allclose(normalize(params, raw_mean_plus_std), 1.0, atol=1e-6)


# ---------------------------------------------------------------------------
# 8.3 — Cần TensorFlow
# ---------------------------------------------------------------------------


def test_reference_matches_keras_on_sanity():
    tf = pytest.importorskip("tensorflow")
    import numpy as np

    from src.reference_forward import forward_batch, load_params

    if not PARAMS_NPZ.is_file():
        pytest.skip("Chưa sinh models/params_fp32.npz; chạy 'python main.py export-c' trước")

    model = tf.keras.models.load_model(ROOT / "models" / "baseline.keras")
    with np.load(SANITY_NPZ) as sanity:
        x = sanity["x"]
    keras_logits = model.predict(x, verbose=0)

    params = load_params()
    ref_logits = forward_batch(params, x)
    assert np.abs(keras_logits - ref_logits).max() < 1e-4


def test_exported_params_match_checkpoint():
    tf = pytest.importorskip("tensorflow")
    import numpy as np

    if not PARAMS_NPZ.is_file():
        pytest.skip("Chưa sinh models/params_fp32.npz; chạy 'python main.py export-c' trước")

    model = tf.keras.models.load_model(ROOT / "models" / "baseline.keras")
    with np.load(PARAMS_NPZ) as params_npz:
        for name in (
            "block1_conv1", "block1_conv2", "block1_projection",
            "block2_conv1", "block2_conv2", "activity_logits",
        ):
            kernel, bias = model.get_layer(name).get_weights()
            assert np.array_equal(kernel, params_npz[f"{name}_w"])
            assert np.array_equal(bias, params_npz[f"{name}_b"])


# ---------------------------------------------------------------------------
# 8.4 — Cần trình biên dịch C
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not HAS_COMPILER, reason="Cần cc/gcc và make để build firmware/")
def test_c_build_is_warning_free():
    firmware_dir = ROOT / "firmware"
    subprocess.run(["make", "clean"], cwd=firmware_dir, check=True,
                    capture_output=True, text=True)
    result = subprocess.run(["make"], cwd=firmware_dir, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    assert "warning:" not in result.stderr


@pytest.mark.skipif(not HAS_COMPILER, reason="Cần cc/gcc và make để build firmware/")
def test_c_inference_matches_python():
    firmware_dir = ROOT / "firmware"
    subprocess.run(["make"], cwd=firmware_dir, check=True, capture_output=True, text=True)
    metadata = json.loads(METADATA_PATH.read_text(encoding="utf-8"))
    result = subprocess.run(
        [str(firmware_dir / "build" / "test_inference")],
        cwd=firmware_dir, capture_output=True, text=True,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert "PASS" in result.stdout
    assert metadata["checkpoint_id"] in result.stdout


# ---------------------------------------------------------------------------
# 8.5 — Hồi quy
# ---------------------------------------------------------------------------


def test_main_check_still_passes():
    result = subprocess.run(
        ["python3", "main.py", "--check"], cwd=ROOT, capture_output=True, text=True,
    )
    assert result.returncode == 0
    assert "OK:" in result.stdout
