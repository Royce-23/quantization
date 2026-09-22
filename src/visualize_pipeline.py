"""Tạo sơ đồ trình bày luồng xử lý và thuật toán của đề tài."""

from __future__ import annotations

from .common import project_path


def _box(axis, x, y, width, height, text, color, fontsize=11, weight="normal"):
    from matplotlib.patches import FancyBboxPatch

    patch = FancyBboxPatch(
        (x, y), width, height,
        boxstyle="round,pad=0.012,rounding_size=0.018",
        linewidth=1.5, edgecolor="#334155", facecolor=color,
    )
    axis.add_patch(patch)
    axis.text(x + width / 2, y + height / 2, text, ha="center", va="center",
              fontsize=fontsize, weight=weight, color="#0f172a", wrap=True)


def _arrow(axis, x1, y1, x2, y2, color="#475569"):
    axis.annotate("", xy=(x2, y2), xytext=(x1, y1), arrowprops={
        "arrowstyle": "-|>", "color": color, "linewidth": 1.7,
        "shrinkA": 4, "shrinkB": 4,
    })


def _lane(axis, y, title, color, boxes):
    from matplotlib.patches import Rectangle

    axis.add_patch(Rectangle((0.25, y), 17.5, 2.55, facecolor=color[0],
                             edgecolor=color[1], linewidth=1.2))
    axis.text(0.55, y + 2.27, title, fontsize=13, weight="bold", color=color[2])
    for x, text, fill in boxes:
        _box(axis, x, y + 0.53, 1.85, 1.15, text, fill, fontsize=9)
    for left, right in zip(boxes[:-1], boxes[1:]):
        _arrow(axis, left[0] + 1.86, y + 1.10, right[0] - 0.04, y + 1.10)


def run(config):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle

    output_dir = project_path(config["paths"]["results"]) / "figures" / "pipeline"
    output_dir.mkdir(parents=True, exist_ok=True)
    figure, axis = plt.subplots(figsize=(18, 11))
    axis.set_xlim(0, 18)
    axis.set_ylim(0, 11)
    axis.axis("off")
    figure.patch.set_facecolor("white")
    axis.text(9, 10.65, "QUANTIZATION HAR — LUỒNG XỬ LÝ VÀ THUẬT TOÁN",
              ha="center", va="center", fontsize=22, weight="bold", color="#0f172a")
    axis.text(9, 10.27,
              "Nhận diện hoạt động từ cảm biến UCI-HAR và so sánh FP32 với PTQ full-integer INT8",
              ha="center", va="center", fontsize=11, color="#475569")

    _lane(axis, 7.05, "A. DỮ LIỆU VÀ BASELINE FP32",
          ("#eff6ff", "#bfdbfe", "#1d4ed8"), [
              (0.6, "UCI-HAR\n7352 train + 2947 test\n128 × 9 cảm biến", "#dbeafe"),
              (3.05, "Chia dữ liệu\ntrain / validation / test\ntách theo subject", "#dbeafe"),
              (5.5, "Z-score\nmean, std fit trên train\nkhông rò rỉ test", "#dbeafe"),
              (7.95, "ResNetv1-6 1D\nConv/Pool + residual\n10.726 tham số", "#bfdbfe"),
              (10.4, "Huấn luyện FP32\nSGD, 300 epoch\nchọn checkpoint tốt nhất", "#bfdbfe"),
              (12.85, "baseline.keras\nFP32 checkpoint\n92,60% test", "#93c5fd"),
          ])
    _lane(axis, 4.0, "B. PTQ INT8, ĐÁNH GIÁ VÀ BÁO CÁO",
          ("#f0fdf4", "#bbf7d0", "#15803d"), [
              (0.6, "baseline.keras\nđầu vào FP32 đã\nchuẩn hóa", "#dcfce7"),
              (3.05, "Calibration từ train\nn = 50 / 200 / 500\nseed = 11 / 22 / 33", "#dcfce7"),
              (5.5, "Tìm scale và\nzero-point\ncho activation", "#dcfce7"),
              (7.95, "TFLite PTQ\nfull-integer INT8\ninput/output = int8", "#bbf7d0"),
              (10.4, "Validation\n9 cấu hình INT8\nchọn n = 500", "#bbf7d0"),
              (12.85, "Test cuối\naccuracy, macro-F1\ndung lượng, confusion matrix", "#86efac"),
          ])
    _arrow(axis, 13.77, 7.56, 13.77, 5.70, color="#15803d")
    axis.text(14.25, 6.65, "cùng checkpoint", ha="center", va="center",
              fontsize=8, color="#15803d")

    axis.add_patch(Rectangle((0.25, 0.35), 17.5, 3.05, facecolor="#fff7ed",
                             edgecolor="#fed7aa", linewidth=1.2))
    axis.text(0.55, 3.08, "C. THUẬT TOÁN SUY LUẬN RESNET 1D", fontsize=13,
              weight="bold", color="#c2410c")
    steps = [
        (0.6, "Input\n128 × 9", "#ffedd5", 1.48),
        (2.25, "Conv1D k=3\n32 filters", "#ffedd5", 1.48),
        (3.9, "ReLU → Conv1D\nk=3", "#ffedd5", 1.48),
        (5.55, "MaxPool /2\n128 → 64", "#ffedd5", 1.48),
        (7.2, "Cộng shortcut\nConv1×1 + Pool\n→ ReLU", "#fdba74", 1.72),
        (9.15, "Block 2\nConv → ReLU → Conv\n+ identity → ReLU", "#fdba74", 1.72),
        (11.1, "Global Max Pool\n64 × 32 → 32", "#fed7aa", 1.48),
        (13.05, "Dense 6 logits\nargmax → nhãn", "#fdba74", 1.48),
    ]
    for x, text, color, width in steps:
        _box(axis, x, 1.48, width, 1.05, text, color, fontsize=8.5)
    for left, right in zip(steps[:-1], steps[1:]):
        _arrow(axis, left[0] + left[3], 2.0, right[0] - 0.04, 2.0, color="#c2410c")
    axis.text(9, 0.72,
              "INT8: q = clip(round(x / scale + zero_point), -128, 127)   |   x ≈ scale × (q − zero_point)",
              ha="center", va="center", fontsize=10, color="#7c2d12",
              bbox={"boxstyle": "round,pad=0.35", "facecolor": "#ffedd5", "edgecolor": "#fdba74"})
    figure.tight_layout(pad=0.5)
    figure.savefig(output_dir / "algorithm_flow.png", dpi=180, bbox_inches="tight")
    figure.savefig(output_dir / "algorithm_flow.svg", bbox_inches="tight")
    plt.close(figure)
    print(f"Đã tạo sơ đồ tại {output_dir}")
    return output_dir / "algorithm_flow.png"
