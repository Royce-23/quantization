"""Test cho tính năng trích xuất tham số C và suy luận C (TASK-10).

Xem docs/model_architecture_and_c_inference.md mục 5 cho tiêu chí đối chiếu.
"""
import ast
import hashlib
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


# ---------------------------------------------------------------------------
# 9 — Test mới cho TASK tách khối C / sơ đồ nhiều nhánh / describe_params.py
# ---------------------------------------------------------------------------


def test_format_value_emits_valid_c_float_literals():
    """9.1 — không cần numpy/TF, không cần compiler."""
    from src.export_c import _format_value

    fmt = "{:.9g}f"
    # Dùng list các tuple (không dùng dict!): 0.0 == -0.0 nên nếu dùng dict,
    # hai khoá này sẽ va chạm (collapse thành 1 key) và làm mất ca kiểm tra
    # dấu âm của 0 — đây là bẫy dễ mắc khi viết test cho 2 giá trị "bằng nhau
    # nhưng khác bit dấu".
    cases = [
        (0.0, "0.0f"),
        (-0.0, "-0.0f"),
        (1.0, "1.0f"),
        (3.0, "3.0f"),
        (1e-9, "1e-09f"),
        (-0.0123456791, "-0.0123456791f"),
    ]
    for value, expected in cases:
        text = _format_value(value, fmt)
        assert text == expected
        assert text.endswith("f")
        mantissa = text[:-1]
        assert any(ch in mantissa for ch in ".eE"), (
            f"literal C không hợp lệ (thiếu '.'/'e'): {text!r}"
        )

    with pytest.raises(ValueError):
        _format_value(float("nan"), fmt)
    with pytest.raises(ValueError):
        _format_value(float("inf"), fmt)


def test_reference_block_decomposition_matches_forward():
    """9.2 — cần numpy + models/params_fp32.npz (skip nếu thiếu)."""
    if not PARAMS_NPZ.is_file():
        pytest.skip("Chưa sinh models/params_fp32.npz; chạy 'python main.py export-c' trước")
    import numpy as np

    from src.reference_forward import (
        block1_forward, block2_forward, forward, head_forward, load_params,
    )

    params = load_params()
    with np.load(SANITY_NPZ) as sanity:
        x = sanity["x"]

    for i in range(x.shape[0]):
        x1 = block1_forward(params, x[i])
        assert x1.shape == (64, 32)
        x2 = block2_forward(params, x1)
        assert x2.shape == (64, 32)
        logits_blocks = head_forward(params, x2)
        assert logits_blocks.shape == (6,)

        logits_forward = forward(params, x[i])
        assert np.array_equal(logits_blocks, logits_forward)


def test_c_header_declares_block_level_api():
    """9.3."""
    text = (ROOT / "firmware" / "include" / "har_resnet.h").read_text(encoding="utf-8")
    for symbol in ("har_block1_forward", "har_block2_forward", "har_head_forward"):
        assert symbol in text
    assert "THREAD-SAFE" in text.upper()


def test_c_source_has_per_block_functions():
    """9.4."""
    text = (ROOT / "firmware" / "src" / "har_resnet.c").read_text(encoding="utf-8")
    for symbol in (
        "har_block1_forward", "har_block2_forward", "har_head_forward",
        "har_block1_branch_main", "har_block1_branch_shortcut",
        "har_block2_branch_main", "har_residual_merge_relu",
    ):
        assert f"{symbol}(" in text

    predict_start = text.index("void har_resnet_predict(")
    predict_body_start = text.index("{", predict_start)
    predict_body_end = text.index("\n}", predict_body_start)
    predict_body = text[predict_body_start:predict_body_end]

    assert "har_block1_forward(" in predict_body
    assert "har_block2_forward(" in predict_body
    assert "har_head_forward(" in predict_body
    assert "har_conv1d_same(" not in predict_body


def test_describe_params_module_imports_without_heavy_deps():
    """9.5."""
    module = importlib.import_module("src.describe_params")
    assert callable(module.run)

    source = (ROOT / "src" / "describe_params.py").read_text(encoding="utf-8")
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


def test_main_registers_describe_params_stage():
    """9.6."""
    import main

    assert "describe-params" in main.STAGES
    assert main.STAGES["describe-params"] == "describe_params"


def test_config_has_describe_params_section():
    """9.7."""
    from src.common import load_config

    config = load_config()
    assert "describe_params" in config
    assert set(config["describe_params"].keys()) >= {"params_npz", "manifest", "output"}
    # Bẫy CI quan trọng: config["paths"] vẫn đúng 4 key cũ, key mới KHÔNG được vào đây.
    assert set(config["paths"].keys()) == {
        "raw_data", "processed_data", "models", "results",
    }


PARAMETERS_REFERENCE = ROOT / "results" / "parameters_reference.md"


def test_parameters_reference_is_generated_and_consistent():
    """9.8."""
    if not PARAMETERS_REFERENCE.is_file():
        pytest.skip(f"Chưa sinh {PARAMETERS_REFERENCE}; chạy 'python main.py describe-params' trước")
    text = PARAMETERS_REFERENCE.read_text(encoding="utf-8")

    assert "TỰ ĐỘNG SINH" in text
    assert ("10726" in text) or ("10 726" in text)
    for layer_name in (
        "block1_conv1", "block1_conv2", "block1_projection",
        "block2_conv1", "block2_conv2", "activity_logits",
    ):
        assert layer_name in text
    assert "Block 1" in text
    assert "Block 2" in text
    assert ("Nhánh" in text) or ("branch" in text.lower())
    assert "Công thức" in text
    assert not re.search(r"\d{4}-\d{2}-\d{2}", text)


def test_describe_params_is_deterministic():
    """9.9 — cần numpy + params_fp32.npz (skip nếu thiếu)."""
    if not PARAMS_NPZ.is_file():
        pytest.skip("Chưa sinh models/params_fp32.npz; chạy 'python main.py export-c' trước")
    from src.common import load_config
    from src.describe_params import run

    config = load_config()
    path1 = run(config)
    digest1 = hashlib.sha256(path1.read_bytes()).hexdigest()
    path2 = run(config)
    digest2 = hashlib.sha256(path2.read_bytes()).hexdigest()
    assert digest1 == digest2


def test_pipeline_figure_shows_branches():
    """9.10."""
    pytest.importorskip("matplotlib")
    from src.common import load_config
    from src.visualize_pipeline import run

    config = load_config()
    png_path = run(config)
    svg_path = png_path.with_suffix(".svg")
    assert png_path.is_file() and png_path.stat().st_size > 0
    assert svg_path.is_file() and svg_path.stat().st_size > 0

    svg_text = svg_path.read_text(encoding="utf-8")
    labels = ("Nhánh A", "Nhánh B", "identity")
    missing = [label for label in labels if label not in svg_text]
    if missing:
        # matplotlib có thể nhúng text dạng path (không grep được) tuỳ backend/font;
        # trong trường hợp đó, đối chiếu trực tiếp trên mã nguồn vẽ hình.
        source = (ROOT / "src" / "visualize_pipeline.py").read_text(encoding="utf-8")
        for label in missing:
            assert label in source, f"Thiếu nhãn {label!r} cả trong SVG lẫn source code"


def test_pipeline_arrows_point_forward_and_boxes_dont_overlap():
    """9.12 — kiểm tra ĐỊNH LƯỢNG lane C, không chỉ nhìn ảnh bằng mắt.

    Bối cảnh: Reviewer phát hiện mũi tên nối ReLU -> Conv1D (thứ 2) trong cả
    Block 1 và Block 2 của `_lane_c_branches()` từng bị vẽ NGƯỢC CHIỀU (2 hộp
    liền kề chồng lấn, khe hở -0.10) khiến sơ đồ mô tả sai thứ tự thuật toán
    thật (trông như Conv1D -> ReLU). Bài test trước đó
    (`test_pipeline_figure_shows_branches`) chỉ grep nhãn chữ trong SVG nên
    không phát hiện được lỗi này. Bài test này đọc lại đúng toạ độ hộp/mũi
    tên từ `_lane_c_geometry()` (nguồn sự thật dùng để vẽ) và khẳng định:

    (a) không có cặp hộp liền kề nào (cùng hàng) chồng lấn — khoảng cách
        giữa mép phải hộp trái và mép trái hộp phải phải DƯƠNG;
    (b) mọi mũi tên nối 2 khối xử lý liên tiếp đều có toạ độ x đích LỚN HƠN
        x nguồn (luôn tiến sang phải, không lùi).
    """
    from src.visualize_pipeline import _lane_c_geometry

    geo = _lane_c_geometry()

    # (a) Không hộp liền kề nào (cùng hàng) chồng lấn.
    rows = {}
    for x_start, width, row_id in geo["boxes"]:
        rows.setdefault(row_id, []).append((x_start, width))
    assert rows, "geo['boxes'] rỗng — không có gì để kiểm tra"
    for row_id, items in rows.items():
        items.sort(key=lambda item: item[0])
        for (x_left, w_left), (x_right, _w_right) in zip(items[:-1], items[1:]):
            gap = x_right - (x_left + w_left)
            assert gap > 0, (
                f"Hộp chồng lấn trong hàng {row_id!r}: hộp trái kết thúc tại "
                f"{x_left + w_left}, hộp phải bắt đầu tại {x_right} (gap={gap})"
            )

    # (b) Mọi mũi tên đều tiến sang phải (x đích > x nguồn).
    assert geo["arrows"], "geo['arrows'] rỗng — không có gì để kiểm tra"
    for x1, x2, label in geo["arrows"]:
        assert x2 > x1, (
            f"Mũi tên {label!r} đi NGƯỢC CHIỀU: x1={x1} -> x2={x2} "
            "(x đích phải lớn hơn x nguồn)"
        )


BANGIAO_SCAN_GLOBS = (
    "src/*.py", "tests/*.py", "firmware/src/*.c", "firmware/include/*.h",
    "firmware/test/*.c", "docs/*.md", "README.md",
)


def test_no_dangling_bangiao_references():
    """9.11.

    LƯU Ý: KHÔNG được viết chuỗi thư mục kế hoạch (dấu chấm + "bangiao" + dấu
    gạch chéo) liền một khối trong CHÍNH file test này — glob "tests/*.py"
    bên dưới quét luôn tests/test_c_export.py, nên nếu viết chuỗi đó liền
    mạch (kể cả trong docstring/comment), test này sẽ tự phát hiện chính nó
    và luôn FAIL. Vì vậy token cần ghép từ 2 phần ở dòng dưới, và mọi mô tả
    bằng lời trong file này cũng phải tránh viết liền mạch chuỗi đó.
    """
    dead_ref_token = ".bangiao" + "/"
    offenders = []
    for pattern in BANGIAO_SCAN_GLOBS:
        for path in ROOT.glob(pattern):
            if dead_ref_token in path.read_text(encoding="utf-8"):
                offenders.append(str(path.relative_to(ROOT)))
    assert not offenders, f"Tham chiếu chết tới {dead_ref_token} trong: {offenders}"
