"""TV — Trích xuất tham số ResNetv1-6 (models/baseline.keras) sang mã C tĩnh
cho firmware/, kèm bản sao NumPy (.npz) và golden vector đối chiếu.

QUAN TRỌNG: `python main.py --check` chạy trên CI (Python 3.11) không cài
numpy/tensorflow và import mọi module trong STAGES ở top-level. Vì vậy toàn bộ
import numpy/tensorflow trong file này BẮT BUỘC nằm bên trong hàm `run`,
giống style của src/train.py và src/prepare_data.py — không được đưa lên đầu
file.
"""
from __future__ import annotations

import hashlib
import json
import platform
from pathlib import Path

from .common import ROOT, project_path

# Bảng đặc tả 5 lớp Conv1D theo đúng thứ tự forward trong src/model.py.
# kernel Keras có shape (k, cin, cout); flatten C-order giữ nguyên thứ tự này
# -> layout C là [k][cin][cout], đúng theo docs/model_architecture_and_c_inference.md mục 3.1.
CONV_LAYERS = [
    {"name": "block1_conv1", "k": 3, "cin": 9, "cout": 32},
    {"name": "block1_conv2", "k": 3, "cin": 32, "cout": 32},
    {"name": "block1_projection", "k": 1, "cin": 9, "cout": 32},
    {"name": "block2_conv1", "k": 3, "cin": 32, "cout": 32},
    {"name": "block2_conv2", "k": 3, "cin": 32, "cout": 32},
]
DENSE_LAYER = {"name": "activity_logits", "din": 32, "dout": 6}

INPUT_TIMESTEPS = 128
INPUT_CHANNELS = 9
FILTERS = 32
NUM_CLASSES = 6
TOTAL_PARAM_COUNT = 10726
GOLDEN_COUNT = 12

_AUTOGEN_BANNER = "/* TỰ ĐỘNG SINH bởi src/export_c.py — KHÔNG SỬA TAY. */"


def _sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _macro_prefix(layer_name):
    return f"HAR_{layer_name.upper()}"


def _format_value(value, float_format):
    text = float_format.format(float(value))
    lowered = text.lower()
    if "nan" in lowered or "inf" in lowered:
        raise ValueError(f"Giá trị không hợp lệ khi xuất sang C: {value!r} -> {text}")
    # Tách hậu tố 'f'/'F' của literal float C (nếu float_format có sẵn hậu tố này).
    if text and text[-1] in "fF":
        mantissa, suffix = text[:-1], text[-1]
    else:
        mantissa, suffix = text, ""
    # "0" -> "0.0", "-3" -> "-3.0": literal C phải có dấu chấm hoặc số mũ thì hậu tố
    # 'f' mới hợp lệ ("0f" là lỗi cú pháp, "0.0f" mới đúng). "1e-09" đã có 'e' -> giữ nguyên.
    if not any(ch in mantissa for ch in ".eE"):
        mantissa += ".0"
    return mantissa + suffix


def _format_c_array_body(flat_values, float_format, values_per_line):
    """Sinh phần thân mảng C, 1 dòng values_per_line phần tử, thụt lề 4 space."""
    lines = []
    total = len(flat_values)
    for start in range(0, total, values_per_line):
        chunk = flat_values[start:start + values_per_line]
        formatted = [_format_value(v, float_format) for v in chunk]
        row = ", ".join(formatted)
        is_last_chunk = start + values_per_line >= total
        lines.append(f"    {row}" + ("" if is_last_chunk else ","))
    return "\n".join(lines)


def _format_c_array_definition(c_type, name, size_expr, flat_values, float_format,
                                values_per_line, shape_comment, qualifier="const"):
    body = _format_c_array_body(flat_values, float_format, values_per_line)
    return (
        f"/* shape {shape_comment} */\n"
        f"{qualifier} {c_type} {name}[{size_expr}] = {{\n{body}\n}};\n"
    )


def _render_params_header(checkpoint_id):
    lines = [
        _AUTOGEN_BANNER,
        "#ifndef HAR_RESNET_PARAMS_H",
        "#define HAR_RESNET_PARAMS_H",
        "",
        f'#define HAR_CHECKPOINT_ID "{checkpoint_id}"',
        f"#define HAR_PARAM_COUNT   {TOTAL_PARAM_COUNT}",
        f"#define HAR_INPUT_TIMESTEPS {INPUT_TIMESTEPS}",
        f"#define HAR_INPUT_CHANNELS  {INPUT_CHANNELS}",
        f"#define HAR_FILTERS         {FILTERS}",
        f"#define HAR_NUM_CLASSES     {NUM_CLASSES}",
        "",
    ]
    for spec in CONV_LAYERS:
        prefix = _macro_prefix(spec["name"])
        lines.append(f"#define {prefix}_K {spec['k']}")
        lines.append(f"#define {prefix}_CIN {spec['cin']}")
        lines.append(f"#define {prefix}_COUT {spec['cout']}")
        lines.append(
            f"extern const float har_{spec['name']}_w[{prefix}_K * {prefix}_CIN * {prefix}_COUT];"
        )
        lines.append(f"extern const float har_{spec['name']}_b[{prefix}_COUT];")
        lines.append("")
    lines.append(f"#define HAR_DENSE_IN  {DENSE_LAYER['din']}")
    lines.append(f"#define HAR_DENSE_OUT {DENSE_LAYER['dout']}")
    lines.append(f"extern const float har_{DENSE_LAYER['name']}_w[HAR_DENSE_IN * HAR_DENSE_OUT];")
    lines.append(f"extern const float har_{DENSE_LAYER['name']}_b[HAR_DENSE_OUT];")
    lines.append("")
    lines.append("extern const float har_input_mean[HAR_INPUT_CHANNELS];")
    lines.append("extern const float har_input_std[HAR_INPUT_CHANNELS];")
    lines.append("extern const char *const har_label_names[HAR_NUM_CLASSES];")
    lines.append("")
    lines.append("#endif /* HAR_RESNET_PARAMS_H */")
    lines.append("")
    return "\n".join(lines)


def _render_params_source(params, labels, float_format, values_per_line):
    parts = [_AUTOGEN_BANNER, '#include "har_resnet_params.h"', ""]
    for spec in CONV_LAYERS:
        prefix = _macro_prefix(spec["name"])
        w = params[f"{spec['name']}_w"]
        b = params[f"{spec['name']}_b"]
        parts.append(_format_c_array_definition(
            "float", f"har_{spec['name']}_w",
            f"{prefix}_K * {prefix}_CIN * {prefix}_COUT",
            w.reshape(-1).tolist(), float_format, values_per_line,
            f"{tuple(int(d) for d in w.shape)} -> flatten [k][cin][cout]",
        ))
        parts.append(_format_c_array_definition(
            "float", f"har_{spec['name']}_b", f"{prefix}_COUT",
            b.reshape(-1).tolist(), float_format, values_per_line,
            f"{tuple(int(d) for d in b.shape)}",
        ))
    dense_w = params[f"{DENSE_LAYER['name']}_w"]
    dense_b = params[f"{DENSE_LAYER['name']}_b"]
    parts.append(_format_c_array_definition(
        "float", f"har_{DENSE_LAYER['name']}_w", "HAR_DENSE_IN * HAR_DENSE_OUT",
        dense_w.reshape(-1).tolist(), float_format, values_per_line,
        f"{tuple(int(d) for d in dense_w.shape)} -> flatten [din][dout]",
    ))
    parts.append(_format_c_array_definition(
        "float", f"har_{DENSE_LAYER['name']}_b", "HAR_DENSE_OUT",
        dense_b.reshape(-1).tolist(), float_format, values_per_line,
        f"{tuple(int(d) for d in dense_b.shape)}",
    ))
    parts.append(_format_c_array_definition(
        "float", "har_input_mean", "HAR_INPUT_CHANNELS",
        params["input_mean"].reshape(-1).tolist(), float_format, values_per_line,
        "(9,) trung bình z-score theo kênh",
    ))
    parts.append(_format_c_array_definition(
        "float", "har_input_std", "HAR_INPUT_CHANNELS",
        params["input_std"].reshape(-1).tolist(), float_format, values_per_line,
        "(9,) độ lệch chuẩn z-score theo kênh",
    ))
    label_lines = ",\n".join(f'    "{label}"' for label in labels)
    parts.append(
        "/* Tên nhãn theo đúng thứ tự lớp 0..5, dùng cho har_label_name(). */\n"
        f"const char *const har_label_names[HAR_NUM_CLASSES] = {{\n{label_lines}\n}};\n"
    )
    return "\n".join(parts)


def _render_golden_header():
    return (
        f"{_AUTOGEN_BANNER}\n"
        "#ifndef HAR_GOLDEN_H\n"
        "#define HAR_GOLDEN_H\n\n"
        '#include "har_resnet_params.h"\n\n'
        f"#define HAR_GOLDEN_COUNT {GOLDEN_COUNT}\n\n"
        "extern const float har_golden_input[HAR_GOLDEN_COUNT][HAR_INPUT_TIMESTEPS * HAR_INPUT_CHANNELS];\n"
        "extern const float har_golden_logits[HAR_GOLDEN_COUNT][HAR_NUM_CLASSES];\n"
        "extern const int   har_golden_label[HAR_GOLDEN_COUNT];       /* nhãn thật y */\n"
        "extern const int   har_golden_prediction[HAR_GOLDEN_COUNT];  /* argmax của Keras */\n\n"
        "#endif /* HAR_GOLDEN_H */\n"
    )


def _render_golden_source(golden_x, golden_logits, golden_y, golden_pred,
                           float_format, values_per_line):
    parts = [_AUTOGEN_BANNER, '#include "har_golden.h"', ""]
    parts.append(
        "/* 12 cửa sổ validation (2 mẫu/lớp) đã chuẩn hoá z-score sẵn, "
        "flatten [time][channel] — KHÔNG chuẩn hoá lại trong C. */"
    )
    rows = []
    for i in range(golden_x.shape[0]):
        body = _format_c_array_body(golden_x[i].reshape(-1).tolist(), float_format, values_per_line)
        rows.append(f"  {{\n{body}\n  }}")
    parts.append(
        "const float har_golden_input[HAR_GOLDEN_COUNT][HAR_INPUT_TIMESTEPS * HAR_INPUT_CHANNELS] = {\n"
        + ",\n".join(rows) + "\n};\n"
    )
    parts.append("/* Logits tham chiếu do models/baseline.keras (TensorFlow) sinh ra. */")
    rows = []
    for i in range(golden_logits.shape[0]):
        body = _format_c_array_body(golden_logits[i].reshape(-1).tolist(), float_format, values_per_line)
        rows.append(f"  {{\n{body}\n  }}")
    parts.append(
        "const float har_golden_logits[HAR_GOLDEN_COUNT][HAR_NUM_CLASSES] = {\n"
        + ",\n".join(rows) + "\n};\n"
    )
    label_values = ", ".join(str(int(v)) for v in golden_y.tolist())
    pred_values = ", ".join(str(int(v)) for v in golden_pred.tolist())
    parts.append(f"const int har_golden_label[HAR_GOLDEN_COUNT] = {{ {label_values} }};")
    parts.append(f"const int har_golden_prediction[HAR_GOLDEN_COUNT] = {{ {pred_values} }};")
    parts.append("")
    return "\n".join(parts)


def run(config):
    import numpy as np
    import tensorflow as tf

    export_cfg = config["export_c"]
    checkpoint_path = project_path(export_cfg["checkpoint"])
    metadata_path = project_path(export_cfg["metadata"])
    preprocess_path = project_path(export_cfg["preprocess"])
    sanity_path = project_path(export_cfg["sanity_samples"])
    params_npz_path = project_path(export_cfg["params_npz"])
    manifest_path = project_path(export_cfg["manifest"])
    firmware_dir = project_path(export_cfg["firmware_dir"])
    values_per_line = int(export_cfg["values_per_line"])
    float_format = export_cfg["float_format"]

    if not checkpoint_path.is_file():
        raise FileNotFoundError(
            f"Thiếu checkpoint {checkpoint_path}; chạy 'python main.py train' trước"
        )
    if not metadata_path.is_file():
        raise FileNotFoundError(f"Thiếu {metadata_path}; chạy 'python main.py train' trước")
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    expected_checkpoint_id = metadata["checkpoint_id"]
    actual_checkpoint_id = _sha256(checkpoint_path)
    if actual_checkpoint_id != expected_checkpoint_id:
        raise ValueError(
            "checkpoint_id của models/baseline.keras không khớp models/baseline_metadata.json "
            f"(thực tế {actual_checkpoint_id}, kỳ vọng {expected_checkpoint_id}). "
            "Baseline FP32 đã bàn giao không được sửa; nếu train lại, cập nhật cả metadata "
            "và sanity_samples.npz trước khi export-c."
        )

    model = tf.keras.models.load_model(checkpoint_path)

    params = {}
    layer_manifest = []
    total_params = 0
    for spec in CONV_LAYERS:
        layer = model.get_layer(spec["name"])
        kernel, bias = layer.get_weights()
        expected_kernel_shape = (spec["k"], spec["cin"], spec["cout"])
        expected_bias_shape = (spec["cout"],)
        if tuple(kernel.shape) != expected_kernel_shape:
            raise ValueError(
                f"Layer {spec['name']}: kernel shape {kernel.shape} != kỳ vọng {expected_kernel_shape}"
            )
        if tuple(bias.shape) != expected_bias_shape:
            raise ValueError(
                f"Layer {spec['name']}: bias shape {bias.shape} != kỳ vọng {expected_bias_shape}"
            )
        kernel = kernel.astype(np.float32, copy=False)
        bias = bias.astype(np.float32, copy=False)
        params[f"{spec['name']}_w"] = kernel
        params[f"{spec['name']}_b"] = bias
        total_params += int(kernel.size + bias.size)
        layer_manifest.append({
            "name": spec["name"],
            "kernel_shape": list(expected_kernel_shape),
            "bias_shape": list(expected_bias_shape),
            "params": int(kernel.size + bias.size),
        })

    dense_layer = model.get_layer(DENSE_LAYER["name"])
    dense_kernel, dense_bias = dense_layer.get_weights()
    expected_dense_kernel_shape = (DENSE_LAYER["din"], DENSE_LAYER["dout"])
    expected_dense_bias_shape = (DENSE_LAYER["dout"],)
    if tuple(dense_kernel.shape) != expected_dense_kernel_shape:
        raise ValueError(
            f"Layer {DENSE_LAYER['name']}: kernel shape {dense_kernel.shape} != "
            f"kỳ vọng {expected_dense_kernel_shape}"
        )
    if tuple(dense_bias.shape) != expected_dense_bias_shape:
        raise ValueError(
            f"Layer {DENSE_LAYER['name']}: bias shape {dense_bias.shape} != "
            f"kỳ vọng {expected_dense_bias_shape}"
        )
    dense_kernel = dense_kernel.astype(np.float32, copy=False)
    dense_bias = dense_bias.astype(np.float32, copy=False)
    params[f"{DENSE_LAYER['name']}_w"] = dense_kernel
    params[f"{DENSE_LAYER['name']}_b"] = dense_bias
    total_params += int(dense_kernel.size + dense_bias.size)
    layer_manifest.append({
        "name": DENSE_LAYER["name"],
        "kernel_shape": list(expected_dense_kernel_shape),
        "bias_shape": list(expected_dense_bias_shape),
        "params": int(dense_kernel.size + dense_bias.size),
    })

    if total_params != TOTAL_PARAM_COUNT:
        raise ValueError(f"Tổng tham số trích được {total_params} != kỳ vọng {TOTAL_PARAM_COUNT}")
    if int(model.count_params()) != TOTAL_PARAM_COUNT:
        raise ValueError(
            f"model.count_params()={model.count_params()} != kỳ vọng {TOTAL_PARAM_COUNT}"
        )

    if not preprocess_path.is_file():
        raise FileNotFoundError(f"Thiếu {preprocess_path}; chạy 'python main.py prepare' trước")
    preprocess = json.loads(preprocess_path.read_text(encoding="utf-8"))
    mean = np.asarray(preprocess["mean"], dtype=np.float32)
    std = np.asarray(preprocess["std"], dtype=np.float32)
    if mean.shape != (INPUT_CHANNELS,) or std.shape != (INPUT_CHANNELS,):
        raise ValueError("mean/std trong preprocess.json phải có shape (9,)")
    params["input_mean"] = mean
    params["input_std"] = std

    params_npz_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(params_npz_path, **params)

    if not sanity_path.is_file():
        raise FileNotFoundError(f"Thiếu {sanity_path}; chạy 'python main.py prepare' trước")
    with np.load(sanity_path) as sanity:
        if "baseline_logits" not in sanity.files or "baseline_predictions" not in sanity.files:
            raise ValueError(
                "data/processed/sanity_samples.npz thiếu 'baseline_logits'/'baseline_predictions'; "
                "chạy lại 'python main.py train' để sanity samples được ghi kèm logits tham chiếu "
                "trước khi export-c."
            )
        golden_x = sanity["x"].astype(np.float32, copy=False)
        golden_y = sanity["y"].astype(np.int64, copy=False)
        golden_logits = sanity["baseline_logits"].astype(np.float32, copy=False)
        golden_pred = sanity["baseline_predictions"].astype(np.int64, copy=False)

    if golden_x.shape != (GOLDEN_COUNT, INPUT_TIMESTEPS, INPUT_CHANNELS):
        raise ValueError(
            f"sanity_samples.npz['x'] shape {golden_x.shape} != "
            f"kỳ vọng ({GOLDEN_COUNT}, {INPUT_TIMESTEPS}, {INPUT_CHANNELS})"
        )
    if golden_logits.shape != (GOLDEN_COUNT, NUM_CLASSES):
        raise ValueError(
            f"sanity_samples.npz['baseline_logits'] shape {golden_logits.shape} != "
            f"kỳ vọng ({GOLDEN_COUNT}, {NUM_CLASSES})"
        )

    include_dir = firmware_dir / "include"
    src_dir = firmware_dir / "src"
    include_dir.mkdir(parents=True, exist_ok=True)
    src_dir.mkdir(parents=True, exist_ok=True)

    labels = config["dataset"]["labels"]

    generated_files = {
        include_dir / "har_resnet_params.h": _render_params_header(actual_checkpoint_id),
        src_dir / "har_resnet_params.c": _render_params_source(
            params, labels, float_format, values_per_line
        ),
        include_dir / "har_golden.h": _render_golden_header(),
        src_dir / "har_golden.c": _render_golden_source(
            golden_x, golden_logits, golden_y, golden_pred, float_format, values_per_line
        ),
    }
    for path, text in generated_files.items():
        path.write_text(text, encoding="utf-8")

    generated_manifest = []
    total_bytes = 0
    for path, text in generated_files.items():
        data = text.encode("utf-8")
        total_bytes += len(data)
        generated_manifest.append({
            "path": str(path.relative_to(ROOT)),
            "sha256": hashlib.sha256(data).hexdigest(),
            "bytes": len(data),
        })

    manifest = {
        "checkpoint_id": actual_checkpoint_id,
        "checkpoint_file": export_cfg["checkpoint"],
        "architecture": config["model"]["architecture"],
        "parameter_count": TOTAL_PARAM_COUNT,
        "layers": layer_manifest,
        "input": {
            "shape": [INPUT_TIMESTEPS, INPUT_CHANNELS],
            "dtype": "float32",
            "normalization": "z_score_per_channel",
            "channel_order": config["dataset"]["channel_order"],
        },
        "labels": labels,
        "generated_files": generated_manifest,
        "environment": {
            "python": platform.python_version(),
            "tensorflow": tf.__version__,
            "numpy": np.__version__,
        },
    }
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")

    summary_lines = [
        "Đã trích xuất tham số ResNetv1-6 sang C:",
        f"  checkpoint_id = {actual_checkpoint_id}",
        f"  tổng tham số  = {TOTAL_PARAM_COUNT}",
        f"  {params_npz_path.relative_to(ROOT)}",
        f"  {manifest_path.relative_to(ROOT)}",
    ]
    summary_lines += [f"  {item['path']} ({item['bytes']} byte)" for item in generated_manifest]
    summary_lines.append(f"  tổng dung lượng file C sinh ra = {total_bytes} byte")
    print("\n".join(summary_lines))
    return manifest_path
