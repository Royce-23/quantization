"""TV1 — tải, kiểm tra, chia và chuẩn hóa tín hiệu UCI-HAR."""

from __future__ import annotations

import json
import shutil
import tempfile
import urllib.request
import zipfile
from pathlib import Path

from .common import project_path


DATASET_URL = (
    "https://archive.ics.uci.edu/ml/machine-learning-databases/00240/"
    "UCI%20HAR%20Dataset.zip"
)
DATASET_DIRNAME = "UCI HAR Dataset"


def _safe_extract(archive, destination):
    """Giải nén sau khi loại đường dẫn có thể thoát khỏi thư mục đích."""
    destination = Path(destination).resolve()
    for member in archive.infolist():
        member_path = (destination / member.filename).resolve()
        if destination not in member_path.parents and member_path != destination:
            raise ValueError(f"Đường dẫn không an toàn trong ZIP: {member.filename}")
    archive.extractall(destination)


def ensure_dataset(raw_root):
    """Trả về thư mục UCI-HAR, tự tải atomically nếu chưa có."""
    raw_root = Path(raw_root)
    dataset_dir = raw_root / DATASET_DIRNAME
    required = dataset_dir / "train" / "Inertial Signals" / "total_acc_x_train.txt"
    if required.is_file():
        return dataset_dir

    raw_root.mkdir(parents=True, exist_ok=True)
    print(f"Đang tải UCI-HAR từ {DATASET_URL}")
    with tempfile.TemporaryDirectory(dir=raw_root) as temporary:
        temporary = Path(temporary)
        archive_path = temporary / "uci_har.zip"
        urllib.request.urlretrieve(DATASET_URL, archive_path)
        with zipfile.ZipFile(archive_path) as archive:
            _safe_extract(archive, temporary)
        extracted = temporary / DATASET_DIRNAME
        if not extracted.is_dir():
            raise RuntimeError("ZIP không chứa thư mục UCI HAR Dataset như mong đợi")
        if dataset_dir.exists():
            shutil.rmtree(dataset_dir)
        shutil.move(str(extracted), str(dataset_dir))
    return dataset_dir


def load_official_split(dataset_dir, split_name, channel_order):
    """Đọc split chính thức thành tensor [samples, 128, 9]."""
    import numpy as np

    if split_name not in {"train", "test"}:
        raise ValueError("split_name chỉ nhận 'train' hoặc 'test'")
    split_dir = Path(dataset_dir) / split_name
    signal_dir = split_dir / "Inertial Signals"
    channels = [
        np.atleast_2d(
            np.loadtxt(
                signal_dir / f"{signal}_{split_name}.txt", dtype=np.float32
            )
        )
        for signal in channel_order
    ]
    x = np.stack(channels, axis=-1).astype(np.float32, copy=False)
    y = np.atleast_1d(
        np.loadtxt(split_dir / f"y_{split_name}.txt", dtype=np.int64)
    ) - 1
    subjects = np.atleast_1d(
        np.loadtxt(split_dir / f"subject_{split_name}.txt", dtype=np.int64)
    )
    if x.shape[1:] != (128, 9):
        raise ValueError(f"Shape {split_name} không hợp lệ: {x.shape}")
    if not (len(x) == len(y) == len(subjects)):
        raise ValueError(f"Số mẫu x/y/subject của {split_name} không khớp")
    if not np.isfinite(x).all():
        raise ValueError(f"{split_name} chứa NaN hoặc infinity")
    if set(np.unique(y).tolist()) != set(range(6)):
        raise ValueError(f"Nhãn {split_name} không đủ 0..5")
    return x, y, subjects


def split_and_normalize(official_train, official_test, validation_subjects):
    """Tách validation theo subject và fit z-score chỉ trên model train."""
    import numpy as np

    x_official, y_official, subject_official = official_train
    x_test_raw, y_test, subject_test = official_test
    validation_subjects = sorted(set(int(item) for item in validation_subjects))
    unknown = set(validation_subjects) - set(subject_official.tolist())
    if unknown:
        raise ValueError(f"Subject validation không thuộc official train: {sorted(unknown)}")
    val_mask = np.isin(subject_official, validation_subjects)
    train_mask = ~val_mask
    if not train_mask.any() or not val_mask.any():
        raise ValueError("Train và validation đều phải có ít nhất một mẫu")

    x_train_raw = x_official[train_mask]
    mean = x_train_raw.mean(axis=(0, 1), keepdims=True).astype(np.float32)
    std = x_train_raw.std(axis=(0, 1), keepdims=True).astype(np.float32)
    std = np.maximum(std, np.float32(1e-6))

    def normalize(values):
        return ((values - mean) / std).astype(np.float32)

    arrays = {
        "x_train": normalize(x_train_raw),
        "y_train": y_official[train_mask],
        "subject_train": subject_official[train_mask],
        "x_val": normalize(x_official[val_mask]),
        "y_val": y_official[val_mask],
        "subject_val": subject_official[val_mask],
        "x_test": normalize(x_test_raw),
        "y_test": y_test,
        "subject_test": subject_test,
    }
    return arrays, mean.reshape(-1), std.reshape(-1)


def _sanity_indices(labels, per_class=2):
    import numpy as np

    return np.concatenate(
        [np.flatnonzero(labels == class_id)[:per_class] for class_id in range(6)]
    )


def prepare_dataset(config):
    """Tạo toàn bộ artifact dữ liệu theo docs/CONTRACT.md."""
    import numpy as np

    raw_root = project_path(config["paths"]["raw_data"])
    output_dir = project_path(config["paths"]["processed_data"])
    output_dir.mkdir(parents=True, exist_ok=True)
    dataset_dir = ensure_dataset(raw_root)
    channel_order = config["dataset"]["channel_order"]
    official_train = load_official_split(dataset_dir, "train", channel_order)
    official_test = load_official_split(dataset_dir, "test", channel_order)
    if len(official_train[0]) != 7352 or len(official_test[0]) != 2947:
        raise ValueError("Số mẫu không khớp phiên bản UCI-HAR dùng trong paper")

    arrays, mean, std = split_and_normalize(
        official_train,
        official_test,
        config["dataset"]["validation_subjects"],
    )
    np.savez_compressed(output_dir / "har.npz", **arrays)

    preprocessing = {
        "dtype": "float32",
        "input_shape": config["dataset"]["input_shape"],
        "channel_order": channel_order,
        "labels": config["dataset"]["labels"],
        "source_label_base": 1,
        "model_label_base": 0,
        "normalization": "z_score_per_channel_fit_on_model_train_only",
        "normalization_applied": True,
        "mean": mean.tolist(),
        "std": std.tolist(),
        "std_floor": 1e-6,
    }
    (output_dir / "preprocess.json").write_text(
        json.dumps(preprocessing, indent=2), encoding="utf-8"
    )
    split_metadata = {
        "policy": config["dataset"]["split_policy"],
        "train_subjects": sorted(set(arrays["subject_train"].tolist())),
        "validation_subjects": sorted(set(arrays["subject_val"].tolist())),
        "test_subjects": sorted(set(arrays["subject_test"].tolist())),
        "sample_counts": {
            "train": len(arrays["x_train"]),
            "validation": len(arrays["x_val"]),
            "test": len(arrays["x_test"]),
        },
        "official_counts": {"train": 7352, "test": 2947},
    }
    (output_dir / "split.json").write_text(
        json.dumps(split_metadata, indent=2), encoding="utf-8"
    )

    sanity_indices = _sanity_indices(arrays["y_val"])
    np.savez_compressed(
        output_dir / "sanity_samples.npz",
        x=arrays["x_val"][sanity_indices],
        y=arrays["y_val"][sanity_indices],
        validation_indices=sanity_indices,
    )
    print(
        "Đã chuẩn bị UCI-HAR: "
        f"train={len(arrays['x_train'])}, val={len(arrays['x_val'])}, "
        f"test={len(arrays['x_test'])}, shape={arrays['x_train'].shape[1:]}"
    )
    return output_dir / "har.npz"


def run(config):
    return prepare_dataset(config)
