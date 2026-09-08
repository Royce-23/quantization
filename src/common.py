"""Giao diện chung. Chỉ dùng thư viện chuẩn để bộ khung chạy ngay."""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def load_config(path="configs/project.json"):
    target = Path(path)
    if not target.is_absolute():
        target = ROOT / target
    with target.open(encoding="utf-8") as stream:
        config = json.load(stream)
    required = {"project", "paths", "dataset", "model", "training", "quantization", "evaluation"}
    missing = required - config.keys()
    if missing:
        raise ValueError(f"Thiếu mục cấu hình: {sorted(missing)}")
    return config


def project_path(relative):
    """Đổi đường dẫn tương đối thành đường dẫn tính từ gốc repository."""
    return ROOT / relative


def pending(owner, task):
    raise NotImplementedError(f"CHƯA TRIỂN KHAI — {owner}: {task}. Xem docs/CONTRACT.md.")
