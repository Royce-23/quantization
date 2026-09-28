"""Cài đặt forward pass ResNetv1-6 thuần NumPy — đúng thứ tự phép toán mà
`firmware/src/har_resnet.c` sẽ dùng (docs/model_architecture_and_c_inference.md mục 2).

Module này KHÔNG được đăng ký vào `STAGES` của main.py nên `python main.py
--check` (chạy trên CI không cài numpy) không import nó; vì vậy import numpy
ở top-level là chấp nhận được ở đây (khác với src/export_c.py, nơi bắt buộc
import numpy/tensorflow bên trong hàm).

Đây vừa là "tài liệu sống" của thuật toán, vừa là cầu nối để kiểm chứng thuật
toán trước khi viết code C: nếu forward_batch() không khớp logits của Keras
(baseline_logits trong data/processed/sanity_samples.npz) thì thuật toán mô
tả sai và KHÔNG được viết C dựa trên nó.
"""
from __future__ import annotations

import numpy as np

from .common import project_path

LAYER_ORDER = (
    "block1_conv1",
    "block1_conv2",
    "block1_projection",
    "block2_conv1",
    "block2_conv2",
    "activity_logits",
)


def load_params(npz_path="models/params_fp32.npz"):
    """Nạp models/params_fp32.npz thành dict tên_layer -> mảng float32."""
    path = project_path(npz_path)
    with np.load(path) as data:
        return {key: data[key].astype(np.float32, copy=True) for key in data.files}


def normalize(params, x_raw):
    """Z-score theo kênh: xn[t, c] = (x_raw[t, c] - mean[c]) / std[c]."""
    mean = params["input_mean"].astype(np.float32)
    std = params["input_std"].astype(np.float32)
    x_raw = np.asarray(x_raw, dtype=np.float32)
    return ((x_raw - mean) / std).astype(np.float32)


def conv1d_same(x, w, b):
    """Conv1D "same", stride 1, kernel lẻ K (K=1 -> pad=0, dùng chung cho
    block1_projection kiểu "valid").

    x: (L, Cin) float32
    w: (K, Cin, Cout) float32 — đúng layout Keras, KHÔNG transpose.
    b: (Cout,) float32
    Trả về: (L, Cout) float32

    Thứ tự tích luỹ giống hệt pseudocode mục 5.2: với mỗi vị trí thời gian t
    và kênh ra co, cộng dồn qua ki rồi qua ci (zero padding ngoài biên).
    """
    x = np.asarray(x, dtype=np.float32)
    w = np.asarray(w, dtype=np.float32)
    b = np.asarray(b, dtype=np.float32)
    length, cin = x.shape
    ksize, w_cin, cout = w.shape
    if w_cin != cin:
        raise ValueError(f"conv1d_same: Cin của input ({cin}) != Cin của kernel ({w_cin})")
    pad = ksize // 2
    out = np.zeros((length, cout), dtype=np.float32)
    for t in range(length):
        acc = b.copy().astype(np.float32)
        for ki in range(ksize):
            ts = t + ki - pad
            if ts < 0 or ts >= length:
                continue
            acc = acc + x[ts, :] @ w[ki, :, :]
            acc = acc.astype(np.float32)
        out[t, :] = acc
    return out


def relu(x):
    x = np.asarray(x, dtype=np.float32)
    return np.maximum(x, np.float32(0.0)).astype(np.float32)


def maxpool1d_2(x):
    """MaxPool1D pool=2, stride=2, padding valid. x: (L, C) với L chẵn."""
    x = np.asarray(x, dtype=np.float32)
    length, channels = x.shape
    if length % 2 != 0:
        raise ValueError("maxpool1d_2 yêu cầu độ dài thời gian chẵn")
    x = x.reshape(length // 2, 2, channels)
    return np.max(x, axis=1).astype(np.float32)


def add(a, b):
    return (np.asarray(a, dtype=np.float32) + np.asarray(b, dtype=np.float32)).astype(np.float32)


def global_max_pool(x):
    """GlobalMaxPooling1D: max theo trục thời gian. x: (L, C) -> (C,)."""
    x = np.asarray(x, dtype=np.float32)
    return np.max(x, axis=0).astype(np.float32)


def dense(x, w, b):
    """out[o] = bias[o] + sum_i in[i] * w[i, o]. x: (Din,), w: (Din, Dout)."""
    x = np.asarray(x, dtype=np.float32)
    w = np.asarray(w, dtype=np.float32)
    b = np.asarray(b, dtype=np.float32)
    return (x @ w + b).astype(np.float32)


def block1_forward(params, x_norm):
    """Block 1 — residual block projection shortcut + hạ mẫu trục thời gian.

    Nguồn input: sensor_window (128, 9) đã z-score (KHÔNG có stem conv trước).
    Nhánh A (main)     : Conv1D k=3 same -> ReLU -> Conv1D k=3 same -> MaxPool/2.
    Nhánh B (shortcut) : Conv1D k=1 valid (projection) -> MaxPool/2, CÙNG nguồn
                          input x_norm với nhánh A.
    Hợp nhánh          : y = ReLU(A + B)  (công thức 2.6 của tài liệu kiến trúc).
    Công thức đầy đủ: xem docs/model_architecture_and_c_inference.md mục 2.3, 2.5, 2.6.
    Shape: (128, 9) -> (64, 32).
    Hàm C tương ứng: firmware/src/har_resnet.c::har_block1_forward().
    """
    x_norm = np.asarray(x_norm, dtype=np.float32)

    # Nhánh chính (main): conv -> relu -> conv -> maxpool.
    main = conv1d_same(x_norm, params["block1_conv1_w"], params["block1_conv1_b"])
    main = relu(main)
    main = conv1d_same(main, params["block1_conv2_w"], params["block1_conv2_b"])
    main = maxpool1d_2(main)

    # Nhánh tắt (shortcut): projection 1x1 trên input gốc -> maxpool.
    shortcut = conv1d_same(x_norm, params["block1_projection_w"], params["block1_projection_b"])
    shortcut = maxpool1d_2(shortcut)

    x = relu(add(main, shortcut))  # x sau block1, shape (64, 32)
    return x


def block2_forward(params, x1):
    """Block 2 — residual block identity shortcut, giữ nguyên (64, 32).

    Nguồn input: x1 = output của KHỐI 1 (64, 32).
    Nhánh A (main)     : Conv1D k=3 same -> ReLU -> Conv1D k=3 same (không pool).
    Nhánh B (shortcut) : identity — KHÔNG có phép toán, dùng thẳng x1.
    Hợp nhánh          : y = ReLU(A + B).
    Công thức đầy đủ: xem docs/model_architecture_and_c_inference.md mục 2.3, 2.6.
    Shape: (64, 32) -> (64, 32).
    Hàm C tương ứng: firmware/src/har_resnet.c::har_block2_forward().
    """
    x1 = np.asarray(x1, dtype=np.float32)

    identity = x1
    main = conv1d_same(x1, params["block2_conv1_w"], params["block2_conv1_b"])
    main = relu(main)
    main = conv1d_same(main, params["block2_conv2_w"], params["block2_conv2_b"])
    x = relu(add(main, identity))  # (64, 32)
    return x


def head_forward(params, x2):
    """Head (bộ phân loại) — GlobalMaxPooling1D + Dense(32 -> 6), KHÔNG softmax.

    Nguồn input: x2 = output của KHỐI 2 (64, 32).
    Công thức: xem docs/model_architecture_and_c_inference.md mục 2.7, 2.8.
    Shape: (64, 32) -> (32,) -> (6,) — logits thô.
    Hàm C tương ứng: firmware/src/har_resnet.c::har_head_forward().
    """
    x2 = np.asarray(x2, dtype=np.float32)

    pooled = global_max_pool(x2)  # (32,)
    logits = dense(pooled, params["activity_logits_w"], params["activity_logits_b"])  # (6,)
    return logits


def forward(params, x_norm):
    """x_norm (128, 9) đã chuẩn hoá -> logits (6,). Đúng 3 lời gọi khối
    (block1_forward -> block2_forward -> head_forward), giữ nguyên xi thứ tự
    phép toán bên trong so với bản gốc — xem
    firmware/src/har_resnet.c::har_resnet_predict() (3 lời gọi khối tương ứng)."""
    x_norm = np.asarray(x_norm, dtype=np.float32)
    x1 = block1_forward(params, x_norm)
    x2 = block2_forward(params, x1)
    logits = head_forward(params, x2)
    return logits


def forward_batch(params, x_norm_batch):
    """x_norm_batch (N, 128, 9) -> logits (N, 6)."""
    x_norm_batch = np.asarray(x_norm_batch, dtype=np.float32)
    outputs = [forward(params, sample) for sample in x_norm_batch]
    return np.stack(outputs, axis=0).astype(np.float32)
