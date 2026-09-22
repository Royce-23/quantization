"""TV2 — chọn tập đại diện calibration từ model-train một cách tái lập."""

from __future__ import annotations


def select_calibration(x_train, size, seed):
    """Lấy ``size`` cửa sổ không lặp bằng RNG cục bộ.

    Hàm chỉ nhận tensor train do caller truyền vào. Việc không dùng RNG toàn
    cục giúp cùng ``seed`` luôn sinh đúng cùng danh sách chỉ số và không làm
    thay đổi trạng thái ngẫu nhiên của bước khác.
    """
    import numpy as np

    x_train = np.asarray(x_train)
    if x_train.ndim != 3:
        raise ValueError(f"x_train phải có dạng [N, time, channel], nhận {x_train.shape}")
    try:
        size = int(size)
        seed = int(seed)
    except (TypeError, ValueError) as exc:
        raise ValueError("size và seed phải là số nguyên") from exc
    if size <= 0:
        raise ValueError("calibration size phải lớn hơn 0")
    if size > len(x_train):
        raise ValueError(
            f"calibration size {size} vượt quá số mẫu train {len(x_train)}"
        )

    rng = np.random.default_rng(seed)
    indices = rng.choice(len(x_train), size=size, replace=False).astype(np.int64)
    return x_train[indices].astype(np.float32, copy=False), indices
