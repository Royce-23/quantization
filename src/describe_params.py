"""TV — Tài liệu hoá CHI TIẾT toàn bộ tham số của ResNetv1-6 1D
(models/params_fp32.npz), sinh báo cáo `results/parameters_reference.md`.

Mục đích: mỗi tham số (layer, khối, nhánh, shape, ý nghĩa từng chiều, công
thức toán dùng tham số đó, tên biến C tương ứng, layout flatten) được mô tả
tường minh — bổ sung cho `models/params_manifest.json` (vốn chỉ có
kernel_shape/bias_shape/params, không có ngữ nghĩa).

QUAN TRỌNG: `python main.py --check` chạy trên CI (Python 3.11) không cài
numpy/tensorflow và import mọi module trong STAGES ở top-level. Vì vậy toàn
bộ `import numpy` trong file này BẮT BUỘC nằm bên trong hàm `run`, giống
style của `src/export_c.py` — KHÔNG được đưa lên đầu file. Module này KHÔNG
cần TensorFlow (chỉ đọc `.npz`).
"""
from __future__ import annotations

import json
from pathlib import Path

from .common import ROOT, project_path

# ---------------------------------------------------------------------------
# Đặc tả chi tiết từng tham số — đây là nội dung "python chi tiết rõ ràng"
# mà báo cáo results/parameters_reference.md sẽ dựa vào.
# ---------------------------------------------------------------------------

PARAM_SPECS = [
    {
        "layer": "block1_conv1",
        "block": "Block 1",
        "branch": "A (main path)",
        "op": "Conv1D K=3, padding=same, stride=1, use_bias=True",
        "source": "sensor_window (128, 9) đã z-score",
        "kernel_key": "block1_conv1_w",
        "kernel_shape": (3, 9, 32),
        "kernel_dims": (
            "K: vị trí trong kernel thời gian (3 bước, chỉ số 0..2)",
            "Cin: kênh vào (9 kênh cảm biến: total_acc x/y/z, body_gyro x/y/z, "
            "body_acc x/y/z)",
            "Cout: kênh ra (32 filter học được)",
        ),
        "bias_key": "block1_conv1_b",
        "bias_shape": (32,),
        "bias_dims": ("Cout: một bias cho mỗi filter",),
        "c_symbol": "har_block1_conv1_w / har_block1_conv1_b",
        "c_layout": "w[(k*Cin + ci)*Cout + co]  (flatten C-order, KHÔNG transpose)",
        "formula": "y[t][co] = b[co] + SUM_k SUM_ci W[k][ci][co] * xtilde[t+k-1][ci]",
        "out_shape": (128, 32),
        "role": "Trích đặc trưng thời gian đầu tiên, nâng 9 kênh cảm biến lên 32 filter.",
    },
    {
        "layer": "block1_conv2",
        "block": "Block 1",
        "branch": "A (main path)",
        "op": "Conv1D K=3, padding=same, stride=1, use_bias=True",
        "source": "output của block1_relu1 (128, 32) — ReLU(block1_conv1(x_norm))",
        "kernel_key": "block1_conv2_w",
        "kernel_shape": (3, 32, 32),
        "kernel_dims": (
            "K: vị trí trong kernel thời gian (3 bước, chỉ số 0..2)",
            "Cin: kênh vào (32 filter của block1_conv1 sau ReLU)",
            "Cout: kênh ra (32 filter)",
        ),
        "bias_key": "block1_conv2_b",
        "bias_shape": (32,),
        "bias_dims": ("Cout: một bias cho mỗi filter",),
        "c_symbol": "har_block1_conv2_w / har_block1_conv2_b",
        "c_layout": "w[(k*Cin + ci)*Cout + co]  (flatten C-order, KHÔNG transpose)",
        "formula": "y[t][co] = b[co] + SUM_k SUM_ci W[k][ci][co] * xtilde[t+k-1][ci]",
        "out_shape": (128, 32),
        "role": "Lớp Conv1D thứ hai của nhánh main Block 1, trước khi MaxPool/2.",
    },
    {
        "layer": "block1_projection",
        "block": "Block 1",
        "branch": "B (projection shortcut)",
        "op": "Conv1D K=1, padding=valid, stride=1, use_bias=True",
        "source": "sensor_window (128, 9) đã z-score — CÙNG nguồn với nhánh A",
        "kernel_key": "block1_projection_w",
        "kernel_shape": (1, 9, 32),
        "kernel_dims": (
            "K: vị trí trong kernel thời gian (K=1, chỉ 1 bước — tương đương "
            "Conv1x1, không trộn thông tin giữa các bước thời gian)",
            "Cin: kênh vào (9 kênh cảm biến)",
            "Cout: kênh ra (32 filter, khớp số kênh của nhánh main để cộng được)",
        ),
        "bias_key": "block1_projection_b",
        "bias_shape": (32,),
        "bias_dims": ("Cout: một bias cho mỗi filter",),
        "c_symbol": "har_block1_projection_w / har_block1_projection_b",
        "c_layout": "w[(k*Cin + ci)*Cout + co]  (flatten C-order, K=1 nên chỉ số k luôn 0)",
        "formula": "p[t][co] = b[co] + SUM_ci W[0][ci][co] * x_norm[t][ci]",
        "out_shape": (128, 32),
        "role": (
            "Đổi số kênh 9 -> 32 cho nhánh shortcut (projection shortcut kiểu "
            "ResNet) để cộng được với nhánh main sau khi cả hai cùng MaxPool/2."
        ),
    },
    {
        "layer": "block2_conv1",
        "block": "Block 2",
        "branch": "A (main path)",
        "op": "Conv1D K=3, padding=same, stride=1, use_bias=True",
        "source": "output của Block 1 — x1 (64, 32)",
        "kernel_key": "block2_conv1_w",
        "kernel_shape": (3, 32, 32),
        "kernel_dims": (
            "K: vị trí trong kernel thời gian (3 bước, chỉ số 0..2)",
            "Cin: kênh vào (32, output Block 1)",
            "Cout: kênh ra (32 filter)",
        ),
        "bias_key": "block2_conv1_b",
        "bias_shape": (32,),
        "bias_dims": ("Cout: một bias cho mỗi filter",),
        "c_symbol": "har_block2_conv1_w / har_block2_conv1_b",
        "c_layout": "w[(k*Cin + ci)*Cout + co]  (flatten C-order, KHÔNG transpose)",
        "formula": "y[t][co] = b[co] + SUM_k SUM_ci W[k][ci][co] * xtilde[t+k-1][ci]",
        "out_shape": (64, 32),
        "role": "Lớp Conv1D đầu tiên của nhánh main Block 2 (không hạ mẫu thời gian).",
    },
    {
        "layer": "block2_conv2",
        "block": "Block 2",
        "branch": "A (main path)",
        "op": "Conv1D K=3, padding=same, stride=1, use_bias=True",
        "source": "output của block2_relu1 (64, 32) — ReLU(block2_conv1(x1))",
        "kernel_key": "block2_conv2_w",
        "kernel_shape": (3, 32, 32),
        "kernel_dims": (
            "K: vị trí trong kernel thời gian (3 bước, chỉ số 0..2)",
            "Cin: kênh vào (32 filter của block2_conv1 sau ReLU)",
            "Cout: kênh ra (32 filter)",
        ),
        "bias_key": "block2_conv2_b",
        "bias_shape": (32,),
        "bias_dims": ("Cout: một bias cho mỗi filter",),
        "c_symbol": "har_block2_conv2_w / har_block2_conv2_b",
        "c_layout": "w[(k*Cin + ci)*Cout + co]  (flatten C-order, KHÔNG transpose)",
        "formula": "y[t][co] = b[co] + SUM_k SUM_ci W[k][ci][co] * xtilde[t+k-1][ci]",
        "out_shape": (64, 32),
        "role": (
            "Lớp Conv1D thứ hai của nhánh main Block 2, kết quả được cộng "
            "trực tiếp với nhánh identity (không có lớp weight nào ở nhánh B)."
        ),
    },
    {
        "layer": "activity_logits",
        "block": "Head (bộ phân loại)",
        "branch": "— (không có nhánh, sau điểm hợp nhánh cuối cùng)",
        "op": "Dense (fully connected), use_bias=True, KHÔNG activation",
        "source": "global_max_pool(output Block 2) — vector (32,)",
        "kernel_key": "activity_logits_w",
        "kernel_shape": (32, 6),
        "kernel_dims": (
            "Din: 32 đặc trưng sau GlobalMaxPooling1D",
            "Dout: 6 lớp hoạt động (WALKING..LAYING)",
        ),
        "bias_key": "activity_logits_b",
        "bias_shape": (6,),
        "bias_dims": ("Dout: một bias cho mỗi lớp",),
        "c_symbol": "har_activity_logits_w / har_activity_logits_b",
        "c_layout": "w[i*Dout + o]  (flatten C-order theo shape Keras (Din, Dout))",
        "formula": "z[o] = b[o] + SUM_i g[i] * W[i][o]",
        "out_shape": (6,),
        "role": "Lớp phân loại cuối, xuất logits thô (softmax không nằm trong model).",
    },
]

NORM_SPECS = [
    {
        "key": "input_mean",
        "shape": (9,),
        "role": "Trung bình (mean) từng kênh cảm biến, fit CHỈ trên tập train.",
        "formula": "xn[t][c] = (xraw[t][c] - mu[c]) / sigma[c]",
        "c_symbol": "har_input_mean",
    },
    {
        "key": "input_std",
        "shape": (9,),
        "role": "Độ lệch chuẩn (std) từng kênh cảm biến, fit CHỈ trên tập train.",
        "formula": "xn[t][c] = (xraw[t][c] - mu[c]) / sigma[c]",
        "c_symbol": "har_input_std",
    },
]

TOTAL_PARAM_COUNT = 10726

_AUTOGEN_BANNER = (
    "<!-- TỰ ĐỘNG SINH bởi src/describe_params.py — KHÔNG SỬA TAY. -->"
)


def _expected_npz_keys():
    keys = set()
    for spec in PARAM_SPECS:
        keys.add(spec["kernel_key"])
        keys.add(spec["bias_key"])
    for spec in NORM_SPECS:
        keys.add(spec["key"])
    return keys


def _percent_zero(flat):
    if flat.size == 0:
        return 0.0
    return float((flat == 0).sum()) / float(flat.size) * 100.0


def _format_num(value):
    return f"{value:.6f}"


def _render_overview_table(layer_param_counts, total):
    lines = [
        "## 2. Bảng tổng quan",
        "",
        "| # | layer | khối | Nhánh | op | kernel shape | bias shape | params | % tổng |",
        "| ---: | --- | --- | --- | --- | --- | --- | ---: | ---: |",
    ]
    for idx, spec in enumerate(PARAM_SPECS, start=1):
        count = layer_param_counts[spec["layer"]]
        pct = count / total * 100.0
        lines.append(
            f"| {idx} | `{spec['layer']}` | {spec['block']} | {spec['branch']} | "
            f"{spec['op']} | {spec['kernel_shape']} | {spec['bias_shape']} | "
            f"{count} | {pct:.2f}% |"
        )
    lines.append(f"| | | | | | | **Tổng** | **{total}** | **100.00%** |")
    lines.append("")
    return lines


def _render_layer_stats(flat):
    stats = {
        "min": float(flat.min()),
        "max": float(flat.max()),
        "mean": float(flat.mean()),
        "std": float(flat.std()),
        "max_abs": float(abs(flat).max()),
        "pct_zero": _percent_zero(flat),
    }
    return stats


def _render_layer_section(spec, params, layer_param_counts):
    kernel = params[spec["kernel_key"]]
    bias = params[spec["bias_key"]]
    kernel_stats = _render_layer_stats(kernel.reshape(-1))
    bias_stats = _render_layer_stats(bias.reshape(-1))
    total_layer = layer_param_counts[spec["layer"]]

    lines = [f"### {spec['layer']}", ""]
    lines.append(f"- **Vai trò**: {spec['role']}")
    lines.append(f"- **Nguồn input**: {spec['source']}")
    lines.append(f"- **Khối / Nhánh**: {spec['block']} / {spec['branch']}")
    lines.append(f"- **Phép toán**: {spec['op']}")
    lines.append(f"- **Shape kernel**: `{spec['kernel_shape']}` — ý nghĩa từng chiều:")
    for dim in spec["kernel_dims"]:
        lines.append(f"  - {dim}")
    lines.append(f"- **Shape bias**: `{spec['bias_shape']}` — ý nghĩa từng chiều:")
    for dim in spec["bias_dims"]:
        lines.append(f"  - {dim}")
    lines.append(
        f"- **Số phần tử**: kernel = {int(kernel.size)}, bias = {int(bias.size)}, "
        f"tổng = {total_layer}"
    )
    lines.append(f"- **Công thức toán**: `{spec['formula']}`")
    lines.append(f"- **Tên biến C**: `{spec['c_symbol']}`")
    lines.append(f"- **Layout C**: `{spec['c_layout']}`")
    lines.append(f"- **Shape output**: `{spec['out_shape']}`")
    lines.append("- **Thống kê kernel**: "
                 f"min={_format_num(kernel_stats['min'])}, "
                 f"max={_format_num(kernel_stats['max'])}, "
                 f"mean={_format_num(kernel_stats['mean'])}, "
                 f"std={_format_num(kernel_stats['std'])}, "
                 f"max|w|={_format_num(kernel_stats['max_abs'])}, "
                 f"%0={_format_num(kernel_stats['pct_zero'])}")
    lines.append("- **Thống kê bias**: "
                 f"min={_format_num(bias_stats['min'])}, "
                 f"max={_format_num(bias_stats['max'])}, "
                 f"mean={_format_num(bias_stats['mean'])}, "
                 f"std={_format_num(bias_stats['std'])}, "
                 f"max|w|={_format_num(bias_stats['max_abs'])}, "
                 f"%0={_format_num(bias_stats['pct_zero'])}")
    lines.append("")
    return lines


def _render_normalization_section(params, channel_order):
    mean = params["input_mean"].reshape(-1)
    std = params["input_std"].reshape(-1)
    lines = ["## 5. Chuẩn hoá đầu vào (z-score theo kênh)", ""]
    lines.append(f"- **Công thức**: `{NORM_SPECS[0]['formula']}`")
    lines.append(
        f"- **Tên biến C**: `{NORM_SPECS[0]['c_symbol']}` (mean), "
        f"`{NORM_SPECS[1]['c_symbol']}` (std)"
    )
    lines.append("")
    lines.append("| # | kênh | mu[c] | sigma[c] |")
    lines.append("| ---: | --- | ---: | ---: |")
    for idx, name in enumerate(channel_order):
        lines.append(f"| {idx} | `{name}` | {_format_num(float(mean[idx]))} | "
                      f"{_format_num(float(std[idx]))} |")
    lines.append("")
    return lines


def _render_memory_section(total):
    fp32_bytes = total * 4
    int8_bytes = total * 1
    lines = ["## 6. Ước lượng bộ nhớ tham số", ""]
    lines.append(
        f"- FP32 = {total} × 4 B = {fp32_bytes} B ({fp32_bytes / 1024:.2f} KiB)"
    )
    lines.append(
        f"- INT8 giả định = {total} × 1 B = {int8_bytes} B ({int8_bytes / 1024:.2f} KiB)"
    )
    lines.append(
        "- Đây là phép tính lý thuyết trên số tham số, KHÁC với dung lượng file "
        "`.tflite` đo thực tế ở `docs/quantization.md` (51 728 -> 25 104 byte) "
        "vì file TFLite còn chứa metadata/flatbuffer."
    )
    lines.append("")
    return lines


def _render_mapping_section():
    lines = ["## 7. Đối chiếu Keras <-> NumPy <-> C", ""]
    lines.append("| Khối | Layer Keras | Hàm NumPy (`src/reference_forward.py`) | Hàm C (`firmware/src/har_resnet.c`) |")
    lines.append("| --- | --- | --- | --- |")
    lines.append("| Block 1 | `block1_*` (5 layer) | `block1_forward()` | `har_block1_forward()` |")
    lines.append("| Block 2 | `block2_*` (4 layer) | `block2_forward()` | `har_block2_forward()` |")
    lines.append("| Head | `global_max_pool`, `activity_logits` | `head_forward()` | `har_head_forward()` |")
    lines.append("")
    return lines


def run(config):
    import numpy as np

    describe_cfg = config["describe_params"]
    params_npz_path = project_path(describe_cfg["params_npz"])
    manifest_path = project_path(describe_cfg["manifest"])
    output_path = project_path(describe_cfg["output"])

    if not params_npz_path.is_file():
        raise FileNotFoundError(
            f"Thiếu {params_npz_path}; chạy 'python main.py export-c' trước"
        )

    with np.load(params_npz_path) as npz:
        actual_keys = set(npz.files)
        expected_keys = _expected_npz_keys()
        if actual_keys != expected_keys:
            missing = expected_keys - actual_keys
            extra = actual_keys - expected_keys
            raise ValueError(
                "models/params_fp32.npz không khớp PARAM_SPECS/NORM_SPECS "
                f"(thiếu {sorted(missing)}, thừa {sorted(extra)})"
            )
        params = {key: npz[key].astype(np.float32, copy=True) for key in npz.files}

    layer_param_counts = {}
    total = 0
    for spec in PARAM_SPECS:
        kernel = params[spec["kernel_key"]]
        bias = params[spec["bias_key"]]
        if tuple(kernel.shape) != spec["kernel_shape"]:
            raise ValueError(
                f"{spec['layer']}: kernel shape thực tế {kernel.shape} != "
                f"khai báo {spec['kernel_shape']}"
            )
        if tuple(bias.shape) != spec["bias_shape"]:
            raise ValueError(
                f"{spec['layer']}: bias shape thực tế {bias.shape} != "
                f"khai báo {spec['bias_shape']}"
            )
        count = int(kernel.size + bias.size)
        layer_param_counts[spec["layer"]] = count
        total += count

    for spec in NORM_SPECS:
        value = params[spec["key"]]
        if tuple(value.shape) != spec["shape"]:
            raise ValueError(
                f"{spec['key']}: shape thực tế {value.shape} != khai báo {spec['shape']}"
            )

    if total != TOTAL_PARAM_COUNT:
        raise ValueError(f"Tổng tham số tính được {total} != kỳ vọng {TOTAL_PARAM_COUNT}")

    checkpoint_id = None
    if manifest_path.is_file():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        checkpoint_id = manifest.get("checkpoint_id")
        if manifest.get("parameter_count") != TOTAL_PARAM_COUNT:
            raise ValueError(
                f"models/params_manifest.json parameter_count "
                f"{manifest.get('parameter_count')} != {TOTAL_PARAM_COUNT}"
            )
        manifest_layer_names = {layer["name"] for layer in manifest.get("layers", [])}
        spec_layer_names = {spec["layer"] for spec in PARAM_SPECS}
        if manifest_layer_names != spec_layer_names:
            raise ValueError(
                "Danh sách layer trong models/params_manifest.json không khớp "
                f"PARAM_SPECS (manifest={sorted(manifest_layer_names)}, "
                f"specs={sorted(spec_layer_names)})"
            )

    channel_order = config["dataset"]["channel_order"]

    lines = [
        _AUTOGEN_BANNER,
        "",
        "# Tham số ResNetv1-6 1D — tài liệu chi tiết",
        "",
        "## 1. Tiêu đề và nguồn",
        "",
        f"- Nguồn tham số: `{params_npz_path.relative_to(ROOT)}`",
    ]
    if checkpoint_id is not None:
        lines.append(f"- checkpoint_id: `{checkpoint_id}`")
    lines.append(f"- Tổng tham số: **{total}**")
    lines.append("")

    lines += _render_overview_table(layer_param_counts, total)

    lines.append("## 3. Chi tiết từng layer")
    lines.append("")
    for spec in PARAM_SPECS:
        lines += _render_layer_section(spec, params, layer_param_counts)

    lines.append(
        "## 4. Thống kê số học (đã trình bày trong mục 3, kèm lại ở đây để "
        "dễ tra cứu và nối mạch sang phần INT8)"
    )
    lines.append("")
    lines.append("| layer | tensor | min | max | mean | std | max\\|w\\| | %0 |")
    lines.append("| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |")
    for spec in PARAM_SPECS:
        kernel_stats = _render_layer_stats(params[spec["kernel_key"]].reshape(-1))
        bias_stats = _render_layer_stats(params[spec["bias_key"]].reshape(-1))
        lines.append(
            f"| `{spec['layer']}` | kernel | {_format_num(kernel_stats['min'])} | "
            f"{_format_num(kernel_stats['max'])} | {_format_num(kernel_stats['mean'])} | "
            f"{_format_num(kernel_stats['std'])} | {_format_num(kernel_stats['max_abs'])} | "
            f"{_format_num(kernel_stats['pct_zero'])} |"
        )
        lines.append(
            f"| `{spec['layer']}` | bias | {_format_num(bias_stats['min'])} | "
            f"{_format_num(bias_stats['max'])} | {_format_num(bias_stats['mean'])} | "
            f"{_format_num(bias_stats['std'])} | {_format_num(bias_stats['max_abs'])} | "
            f"{_format_num(bias_stats['pct_zero'])} |"
        )
    lines.append("")

    lines += _render_normalization_section(params, channel_order)
    lines += _render_memory_section(total)
    lines += _render_mapping_section()

    text = "\n".join(lines) + "\n"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(text, encoding="utf-8")

    print(
        "Đã sinh báo cáo tham số chi tiết:\n"
        f"  {output_path.relative_to(ROOT)}\n"
        f"  tổng tham số = {total}\n"
        f"  số layer mô tả = {len(PARAM_SPECS)}"
    )
    return output_path
