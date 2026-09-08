"""TV3 — Đánh giá mô hình bằng cùng quy trình.
Bộ khung TODO, chưa triển khai thuật toán.
"""
from .common import pending


def run(config):
    # TODO: tải artifact và metadata; dùng cùng mẫu/nhãn cho FP32 và INT8.
    # TODO: quantize input đúng scale/zero-point; clip trước khi cast integer.
    # TODO: không chuẩn hóa lần hai; kiểm tra outputs và label mapping.
    # TODO: accuracy %, macro-F1 trên tất cả lớp, bytes file, confusion matrix.
    # TODO: mặc định validation; test chỉ sau khi chốt cấu hình.
    # TODO: mỗi cấu hình/seed/split lưu một dòng metrics.csv.
    pending("TV3", "evaluate.run")
