"""Cài đặt forward pass ResNetv1-6 thuần NumPy — đúng thứ tự phép toán mà
`firmware/src/har_resnet.c` sẽ dùng (mục 5.2/5.7 kế hoạch triển khai).

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


def forward(params, x_norm):
    """x_norm (128, 9) đã chuẩn hoá -> logits (6,). Đúng trình tự 13 bước
    dùng trong firmware/src/har_resnet.c (mục 5.4 kế hoạch)."""
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

    # Block 2: shortcut identity.
    identity = x
    main = conv1d_same(x, params["block2_conv1_w"], params["block2_conv1_b"])
    main = relu(main)
    main = conv1d_same(main, params["block2_conv2_w"], params["block2_conv2_b"])
    x = relu(add(main, identity))  # (64, 32)

    pooled = global_max_pool(x)  # (32,)
    logits = dense(pooled, params["activity_logits_w"], params["activity_logits_b"])  # (6,)
    return logits


def forward_batch(params, x_norm_batch):
    """x_norm_batch (N, 128, 9) -> logits (N, 6)."""
    x_norm_batch = np.asarray(x_norm_batch, dtype=np.float32)
    outputs = [forward(params, sample) for sample in x_norm_batch]
    return np.stack(outputs, axis=0).astype(np.float32)
