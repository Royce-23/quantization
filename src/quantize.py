"""TV2 — Xuất mô hình và chạy khảo sát calibration.
Bộ khung TODO, chưa triển khai thuật toán.
"""
from .common import pending


def run(config):
    # TODO: tải cùng baseline.keras cho tất cả cấu hình.
    # TODO: xuất baseline_fp32.tflite và INT8 cho từng size/seed.
    # TODO: representative data từ train đã chuẩn hóa, float32.
    # TODO: kiểm tra input/output dtype, scales, zero-points, float fallback.
    # TODO: lưu model_metadata.json theo CONTRACT.
    pending("TV2", "quantize.run")
