"""Tạo các biểu đồ kiểm chứng riêng cho dữ liệu và baseline FP32 của TV1."""

from __future__ import annotations

import csv
import json
from pathlib import Path

from .common import load_config, project_path


def _require(path, instruction):
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Thiếu {path}. {instruction}")
    return path


def _read_history(path):
    with Path(path).open(encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    if not rows:
        raise ValueError("Lịch sử huấn luyện đang trống")
    required = {"epoch", "accuracy", "loss", "val_accuracy", "val_loss"}
    if not required.issubset(rows[0]):
        raise ValueError(f"Lịch sử huấn luyện thiếu cột: {sorted(required - rows[0].keys())}")
    return {key: [float(row[key]) for row in rows] for key in required}


def _plot_training_curves(plt, history, best_epoch, destination):
    figure, axes = plt.subplots(1, 2, figsize=(12, 4.6))
    epochs = history["epoch"]
    axes[0].plot(epochs, history["accuracy"], label="Train", linewidth=1.4)
    axes[0].plot(epochs, history["val_accuracy"], label="Validation", linewidth=1.4)
    axes[0].axvline(best_epoch, color="black", linestyle="--", alpha=0.6,
                    label=f"Checkpoint: epoch {best_epoch}")
    axes[0].set(title="Độ chính xác theo epoch", xlabel="Epoch", ylabel="Accuracy")

    axes[1].plot(epochs, history["loss"], label="Train", linewidth=1.4)
    axes[1].plot(epochs, history["val_loss"], label="Validation", linewidth=1.4)
    axes[1].axvline(best_epoch, color="black", linestyle="--", alpha=0.6)
    axes[1].set(title="Hàm mất mát theo epoch", xlabel="Epoch", ylabel="Cross-entropy")
    for axis in axes:
        axis.grid(alpha=0.25)
        axis.legend()
    figure.suptitle("Baseline FP32 — ResNetv1-6 trên UCI-HAR", fontweight="bold")
    figure.tight_layout()
    figure.savefig(destination, dpi=180, bbox_inches="tight")
    plt.close(figure)


def _plot_class_distribution(plt, np, dataset, labels, destination):
    class_ids = np.arange(len(labels))
    width = 0.25
    figure, axis = plt.subplots(figsize=(12, 5.5))
    for offset, (name, key) in zip(
        (-width, 0, width),
        (("Train", "y_train"), ("Validation", "y_val"), ("Test", "y_test")),
    ):
        counts = np.bincount(dataset[key], minlength=len(labels))
        bars = axis.bar(class_ids + offset, counts, width, label=name)
        axis.bar_label(bars, fontsize=8, padding=2)
    axis.set(
        title="Phân bố lớp trong các tập dữ liệu",
        xlabel="Hoạt động",
        ylabel="Số cửa sổ",
        xticks=class_ids,
        xticklabels=labels,
    )
    plt.setp(axis.get_xticklabels(), rotation=25, ha="right")
    axis.grid(axis="y", alpha=0.25)
    axis.legend()
    figure.tight_layout()
    figure.savefig(destination, dpi=180, bbox_inches="tight")
    plt.close(figure)


def _confusion_matrix(np, y_true, y_pred, class_count):
    matrix = np.zeros((class_count, class_count), dtype=np.int64)
    np.add.at(matrix, (y_true, y_pred), 1)
    return matrix


def _plot_confusion_matrix(plt, np, matrix, labels, accuracy, destination):
    figure, axis = plt.subplots(figsize=(8.5, 7.5))
    image = axis.imshow(matrix, cmap="Blues")
    axis.set(
        title=f"Ma trận nhầm lẫn FP32 trên test — accuracy {accuracy:.2%}",
        xlabel="Nhãn dự đoán",
        ylabel="Nhãn thật",
        xticks=np.arange(len(labels)),
        yticks=np.arange(len(labels)),
        xticklabels=labels,
        yticklabels=labels,
    )
    plt.setp(axis.get_xticklabels(), rotation=35, ha="right")
    threshold = matrix.max() / 2
    for row in range(matrix.shape[0]):
        for column in range(matrix.shape[1]):
            axis.text(
                column,
                row,
                str(matrix[row, column]),
                ha="center",
                va="center",
                color="white" if matrix[row, column] > threshold else "black",
                fontsize=9,
            )
    figure.colorbar(image, ax=axis, fraction=0.046, pad=0.04)
    figure.tight_layout()
    figure.savefig(destination, dpi=180, bbox_inches="tight")
    plt.close(figure)


def _plot_architecture(plt, patches, destination):
    figure, axis = plt.subplots(figsize=(18, 6.5))
    axis.set_xlim(0, 21)
    axis.set_ylim(0, 7)
    axis.axis("off")

    def box(x, y, width, height, text, color):
        patch = patches.FancyBboxPatch(
            (x, y), width, height,
            boxstyle="round,pad=0.04,rounding_size=0.08",
            facecolor=color, edgecolor="#263238", linewidth=1.2,
        )
        axis.add_patch(patch)
        axis.text(x + width / 2, y + height / 2, text,
                  ha="center", va="center", fontsize=9)

    def arrow(x1, y1, x2, y2, dashed=False):
        axis.annotate(
            "", xy=(x2, y2), xytext=(x1, y1),
            arrowprops={"arrowstyle": "->", "linewidth": 1.3,
                        "linestyle": "--" if dashed else "-", "color": "#37474f"},
        )

    box(0.3, 3.2, 1.4, 1.0, "Input\n128 × 9", "#e3f2fd")
    box(2.4, 4.4, 2.0, 1.0, "Conv1D 32, k=3\nReLU", "#fff3e0")
    box(4.9, 4.4, 2.0, 1.0, "Conv1D 32, k=3\nMaxPool 2", "#fff3e0")
    box(3.6, 1.4, 2.2, 1.0, "Projection shortcut\nConv1D k=1 + Pool", "#f3e5f5")
    box(7.5, 3.2, 1.3, 1.0, "Add\nReLU", "#e8f5e9")
    box(9.5, 4.4, 2.0, 1.0, "Conv1D 32, k=3\nReLU", "#fff3e0")
    box(12.0, 4.4, 2.0, 1.0, "Conv1D 32, k=3", "#fff3e0")
    box(14.8, 3.2, 1.3, 1.0, "Add\nReLU", "#e8f5e9")
    box(16.8, 3.2, 2.0, 1.0, "GlobalMaxPool1D", "#e0f7fa")
    box(19.4, 3.2, 1.2, 1.0, "Dense\n6 logits", "#ffebee")

    arrow(1.7, 3.7, 2.4, 4.9)
    arrow(4.4, 4.9, 4.9, 4.9)
    arrow(1.7, 3.7, 3.6, 1.9, dashed=True)
    arrow(6.9, 4.9, 7.5, 3.8)
    arrow(5.8, 1.9, 7.5, 3.6, dashed=True)
    arrow(8.8, 3.7, 9.5, 4.9)
    arrow(11.5, 4.9, 12.0, 4.9)
    arrow(14.0, 4.9, 14.8, 3.8)
    arrow(8.8, 3.5, 14.8, 3.5, dashed=True)
    arrow(16.1, 3.7, 16.8, 3.7)
    arrow(18.8, 3.7, 19.4, 3.7)
    axis.text(10.5, 6.35, "ResNetv1-6 1D — điều chỉnh từ Hình 4 của Novac et al. (2021)",
              ha="center", fontsize=15, fontweight="bold")
    axis.text(10.5, 5.85, "32 filters • 10.726 tham số • đầu vào cửa sổ cảm biến UCI-HAR",
              ha="center", fontsize=10, color="#455a64")
    figure.tight_layout()
    figure.savefig(destination, dpi=180, bbox_inches="tight")
    plt.close(figure)


def run(config):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.patches as patches
    import matplotlib.pyplot as plt
    import numpy as np
    import tensorflow as tf

    processed_dir = project_path(config["paths"]["processed_data"])
    model_dir = project_path(config["paths"]["models"])
    results_dir = project_path(config["paths"]["results"])
    history_path = _require(
        results_dir / "tv1_training_history.csv", "Chạy 'python main.py train' trước."
    )
    dataset_path = _require(
        processed_dir / "har.npz", "Chạy 'python main.py prepare' trước."
    )
    model_path = _require(
        model_dir / "baseline.keras", "Chạy 'python main.py train' trước."
    )
    metadata_path = _require(model_dir / "baseline_metadata.json", "Thiếu metadata TV1.")
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    labels = config["dataset"]["labels"]
    output_dir = results_dir / "figures" / "tv1"
    output_dir.mkdir(parents=True, exist_ok=True)

    history = _read_history(history_path)
    training_path = output_dir / "training_curves.png"
    _plot_training_curves(
        plt, history, metadata["metrics"]["best_epoch"], training_path
    )

    with np.load(dataset_path) as loaded:
        dataset = {key: loaded[key] for key in loaded.files}
    distribution_path = output_dir / "class_distribution.png"
    _plot_class_distribution(plt, np, dataset, labels, distribution_path)

    model = tf.keras.models.load_model(model_path, compile=False)
    predictions = model.predict(
        dataset["x_test"], batch_size=config["training"]["batch_size"], verbose=0
    ).argmax(axis=1)
    accuracy = float(np.mean(predictions == dataset["y_test"]))
    matrix = _confusion_matrix(np, dataset["y_test"], predictions, len(labels))
    confusion_path = output_dir / "confusion_matrix_fp32.png"
    _plot_confusion_matrix(plt, np, matrix, labels, accuracy, confusion_path)
    np.savetxt(output_dir / "confusion_matrix_fp32.csv", matrix, fmt="%d", delimiter=",")

    architecture_path = output_dir / "model_architecture.png"
    _plot_architecture(plt, patches, architecture_path)
    generated = [training_path, distribution_path, confusion_path, architecture_path]
    print("Đã tạo biểu đồ TV1:")
    for path in generated:
        print(f"- {path.relative_to(project_path('.'))}")
    print(f"Accuracy kiểm tra lại từ checkpoint: {accuracy:.4%}")
    return output_dir


if __name__ == "__main__":
    run(load_config())
