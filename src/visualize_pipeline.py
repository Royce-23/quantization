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
    return x + width / 2, y + height / 2


def _arrow(axis, x1, y1, x2, y2, color="#475569", style="-"):
    axis.annotate("", xy=(x2, y2), xytext=(x1, y1), arrowprops={
        "arrowstyle": "-|>", "color": color, "linewidth": 1.7,
        "shrinkA": 4, "shrinkB": 4, "linestyle": style,
    })


def _merge_node(axis, x, y, radius=0.26, color="#f97316"):
    """Vẽ node hợp nhánh (Add) bằng ký hiệu vòng tròn nhỏ ⊕."""
    from matplotlib.patches import Circle

    axis.add_patch(Circle((x, y), radius, facecolor="#fed7aa",
                          edgecolor=color, linewidth=1.8, zorder=5))
    axis.text(x, y, "⊕", ha="center", va="center", fontsize=14,
              weight="bold", color="#9a3412", zorder=6)
    axis.text(x, y - radius - 0.16, "Add", ha="center", va="top",
              fontsize=7.5, color="#9a3412")
    return x, y


def _lane(axis, y, title, color, boxes):
    from matplotlib.patches import Rectangle

    axis.add_patch(Rectangle((0.25, y), 19.3, 2.55, facecolor=color[0],
                             edgecolor=color[1], linewidth=1.2))
    axis.text(0.55, y + 2.27, title, fontsize=13, weight="bold", color=color[2])
    for x, text, fill in boxes:
        _box(axis, x, y + 0.53, 1.9, 1.15, text, fill, fontsize=9)
    for left, right in zip(boxes[:-1], boxes[1:]):
        _arrow(axis, left[0] + 1.91, y + 1.10, right[0] - 0.04, y + 1.10)


def _lane_c_geometry():
    """Hình học (toạ độ) của lane C, tách khỏi việc vẽ.

    Lý do tách riêng: sau khi Reviewer phát hiện mũi tên `ReLU -> Conv1D
    (thứ 2)` bị vẽ NGƯỢC CHIỀU trong cả Block 1 lẫn Block 2 (2 hộp liền kề
    có khe hở ÂM, tức là chồng lấn), hàm này trở thành NGUỒN SỰ THẬT DUY
    NHẤT cho toạ độ hộp/mũi tên, để:

    1. `_lane_c_branches()` dùng đúng các toạ độ này khi vẽ (không còn số
       toạ độ hard-code rải rác, dễ lệch tay như trước).
    2. `tests/test_c_export.py::test_pipeline_arrows_point_forward_and_boxes_dont_overlap`
       đọc lại được đúng các toạ độ này để kiểm tra ĐỊNH LƯỢNG (không chỉ
       nhìn ảnh bằng mắt): không hộp liền kề nào chồng lấn (khoảng cách
       dương) và mọi mũi tên đều có x đích > x nguồn (luôn tiến, không lùi).

    Quy ước đặt tên trả về:
      - "boxes": list các (x_start, width, row_id) — mỗi `row_id` là 1 hàng
        hộp nối tiếp nhau bằng mũi tên (a1 = Block1 nhánh main, b1 = Block1
        nhánh shortcut, a2 = Block2 nhánh main, head = khối phân loại cuối).
      - "arrows": list các (x1, x2, label) — x1 là điểm xuất phát (mép phải
        hộp nguồn + lề nhỏ), x2 là điểm đến (mép trái hộp đích - lề nhỏ),
        đúng công thức đã dùng khi vẽ mũi tên bằng `_arrow()`.
    """
    ROW_H = 0.95   # chiều cao mỗi hàng hộp (A/B) trong lane C
    BOX_GAP_MIN = 0.15  # khe hở tối thiểu giữa 2 hộp liền kề (theo yêu cầu Reviewer)

    # ------------------------------------------------------------------
    # Input + Block 1 — zone y trong [4.3, 6.3], dưới tiêu đề lane (6.62)
    # ------------------------------------------------------------------
    top1_y, bot1_y = 5.35, 4.30

    a1_w = 1.5
    # Bước = bề rộng hộp + khe hở (>= BOX_GAP_MIN) => khoảng cách LUÔN dương,
    # khác với bản cũ (bước cố định 1.6/1.4 tuỳ hộp) từng khiến hộp thứ 2 và
    # thứ 3 chồng lên nhau (khe hở -0.10) và mũi tên đi NGƯỢC.
    a1_step = a1_w + 0.20
    a1_boxes = [
        (2.15 + i * a1_step, text) for i, text in enumerate([
            "Conv1D k=3\nsame\n(128,9)->(128,32)",
            "ReLU\n(128,32)",
            "Conv1D k=3\nsame\n(128,32)->(128,32)",
            "MaxPool /2\n128->64\n(64,32)",
        ])
    ]

    b1_w = 2.15
    b1_boxes = [
        (2.15, "Conv1D k=1\n(projection)\n(128,9)->(128,32)"),
        (4.55, "MaxPool /2\n128->64\n(64,32)"),
    ]

    # merge1_x phải đủ xa hộp cuối a1_boxes để mũi tên "hộp cuối -> merge"
    # vẫn tiến (x2 > x1) với khe hở >= BOX_GAP_MIN.
    merge1_x, merge1_y = 9.20, 5.30
    x1_box_x = merge1_x + 0.50  # giữ đúng khoảng cách tương đối bản gốc

    # ------------------------------------------------------------------
    # Block 2 — zone y trong [2.1, 4.1], identity shortcut
    # ------------------------------------------------------------------
    ZONE_OFFSET = 2.2
    top2_y, bot2_y = top1_y - ZONE_OFFSET, bot1_y - ZONE_OFFSET

    a2_w = 1.5
    a2_step = a2_w + 0.20
    a2_boxes = [
        (11.55 + i * a2_step, text) for i, text in enumerate([
            "Conv1D k=3\nsame\n(64,32)->(64,32)",
            "ReLU\n(64,32)",
            "Conv1D k=3\nsame\n(64,32)->(64,32)",
        ])
    ]

    merge2_x, merge2_y = 16.90, merge1_y - ZONE_OFFSET
    x2_box_x = merge2_x + 0.50

    head_y = 0.45
    head_w = 1.9
    head_boxes = [
        (0.5, "Global Max Pool\n(64,32)->(32,)"),
        (2.6, "Dense 32->6\n(logits thô)"),
        (4.7, "argmax\n→ nhãn 0..5"),
    ]

    boxes = (
        [(x, a1_w, "block1_main") for x, _ in a1_boxes]
        + [(x, b1_w, "block1_shortcut") for x, _ in b1_boxes]
        + [(x, a2_w, "block2_main") for x, _ in a2_boxes]
        + [(x, head_w, "head") for x, _ in head_boxes]
    )

    arrows = []
    for left, right in zip(a1_boxes[:-1], a1_boxes[1:]):
        arrows.append((left[0] + a1_w + 0.01, right[0] - 0.04, "block1_main"))
    arrows.append((b1_boxes[0][0] + b1_w + 0.01, b1_boxes[1][0] - 0.04, "block1_shortcut"))
    arrows.append((a1_boxes[-1][0] + a1_w + 0.01, merge1_x - 0.22, "block1_main->merge1"))
    arrows.append((b1_boxes[-1][0] + b1_w + 0.01, merge1_x - 0.22, "block1_shortcut->merge1"))
    arrows.append((merge1_x + 0.26, x1_box_x - 0.04, "merge1->x1"))
    for left, right in zip(a2_boxes[:-1], a2_boxes[1:]):
        arrows.append((left[0] + a2_w + 0.01, right[0] - 0.04, "block2_main"))
    arrows.append((a2_boxes[-1][0] + a2_w + 0.01, merge2_x - 0.22, "block2_main->merge2"))
    arrows.append((11.55, merge2_x - 0.22, "block2_identity->merge2"))
    arrows.append((merge2_x + 0.26, x2_box_x - 0.04, "merge2->x2"))
    for left, right in zip(head_boxes[:-1], head_boxes[1:]):
        arrows.append((left[0] + head_w + 0.01, right[0] - 0.04, "head"))

    return {
        "ROW_H": ROW_H,
        "BOX_GAP_MIN": BOX_GAP_MIN,
        "top1_y": top1_y, "bot1_y": bot1_y,
        "merge1_x": merge1_x, "merge1_y": merge1_y,
        "a1_boxes": a1_boxes, "a1_w": a1_w,
        "b1_boxes": b1_boxes, "b1_w": b1_w,
        "x1_box_x": x1_box_x,
        "top2_y": top2_y, "bot2_y": bot2_y,
        "merge2_x": merge2_x, "merge2_y": merge2_y,
        "a2_boxes": a2_boxes, "a2_w": a2_w,
        "x2_box_x": x2_box_x,
        "head_y": head_y, "head_boxes": head_boxes, "head_w": head_w,
        "boxes": boxes,
        "arrows": arrows,
    }


def _lane_c_branches(axis):
    """Lane C: sơ đồ luồng THUẬT TOÁN SUY LUẬN RESNET v1-6 1D CÓ NHÁNH.

    Cấu trúc (đúng đặc tả mục 4.1/4.4 tài liệu kiến trúc): 2 điểm rẽ nhánh
    (Block 1: A1/B1, Block 2: A2/B2) + 2 điểm hợp nhánh (⊕) + head phân loại.
    Toạ độ lấy từ `_lane_c_geometry()` (xem docstring hàm đó để biết vì sao
    tách riêng).
    """
    from matplotlib.patches import Rectangle

    geo = _lane_c_geometry()

    lane_y, lane_h = 0.35, 6.9
    axis.add_patch(Rectangle((0.25, lane_y), 19.3, lane_h, facecolor="#fff7ed",
                             edgecolor="#fed7aa", linewidth=1.2))
    axis.text(0.55, lane_y + lane_h - 0.28,
              "C. THUẬT TOÁN SUY LUẬN RESNET v1-6 1D — LUỒNG CÓ NHÁNH",
              fontsize=13, weight="bold", color="#c2410c")

    box_color_main = "#ffedd5"
    box_color_shortcut = "#fdba74"
    box_color_out = "#fed7aa"

    ROW_H = geo["ROW_H"]

    # ------------------------------------------------------------------
    # Input + Block 1 — zone y trong [4.3, 6.3], dưới tiêu đề lane (6.62)
    # ------------------------------------------------------------------
    top1_y, bot1_y = geo["top1_y"], geo["bot1_y"]
    merge1_x, merge1_y = geo["merge1_x"], geo["merge1_y"]
    a1_boxes, a1_w = geo["a1_boxes"], geo["a1_w"]
    b1_boxes, b1_w = geo["b1_boxes"], geo["b1_w"]

    in_cx, in_cy = _box(axis, 0.5, bot1_y, 1.35, (top1_y + ROW_H) - bot1_y,
                        "Input\nxn (128×9)", "#fdba74", fontsize=8.5, weight="bold")

    axis.text(2.15, top1_y + ROW_H + 0.15, "Nhánh A (main)", fontsize=8,
              weight="bold", color="#9a3412")
    axis.text(2.15, bot1_y - 0.15, "Nhánh B (shortcut)", fontsize=8,
              weight="bold", color="#9a3412", va="top")

    _arrow(axis, 1.88, in_cy + 0.18, 2.15, top1_y + ROW_H / 2, color="#c2410c")
    _arrow(axis, 1.88, in_cy - 0.18, 2.15, bot1_y + ROW_H / 2, color="#c2410c")

    for x, text in a1_boxes:
        _box(axis, x, top1_y, a1_w, ROW_H, text, box_color_main, fontsize=7.3)
    for left, right in zip(a1_boxes[:-1], a1_boxes[1:]):
        _arrow(axis, left[0] + a1_w + 0.01, top1_y + ROW_H / 2, right[0] - 0.04, top1_y + ROW_H / 2)

    for x, text in b1_boxes:
        _box(axis, x, bot1_y, b1_w, ROW_H, text, box_color_shortcut, fontsize=7.3)
    _arrow(axis, b1_boxes[0][0] + b1_w + 0.01, bot1_y + ROW_H / 2,
           b1_boxes[1][0] - 0.04, bot1_y + ROW_H / 2)

    _arrow(axis, a1_boxes[-1][0] + a1_w + 0.01, top1_y + ROW_H / 2, merge1_x - 0.22, merge1_y + 0.16)
    _arrow(axis, b1_boxes[-1][0] + b1_w + 0.01, bot1_y + ROW_H / 2, merge1_x - 0.22, merge1_y - 0.16)
    _merge_node(axis, merge1_x, merge1_y)

    x1_box_x = geo["x1_box_x"]
    x1_cx, x1_cy = _box(axis, x1_box_x, merge1_y - 0.575, 1.75, 1.15,
                        "ReLU\n→ x1 (64×32)", box_color_out, fontsize=8, weight="bold")
    _arrow(axis, merge1_x + 0.26, merge1_y, x1_box_x - 0.04, x1_cy, color="#c2410c")

    # ------------------------------------------------------------------
    # Block 2 — zone y trong [2.1, 4.1], identity shortcut
    # ------------------------------------------------------------------
    top2_y, bot2_y = geo["top2_y"], geo["bot2_y"]
    merge2_x, merge2_y = geo["merge2_x"], geo["merge2_y"]
    a2_boxes, a2_w = geo["a2_boxes"], geo["a2_w"]

    axis.text(11.55, top2_y + ROW_H + 0.15, "Nhánh A (main)", fontsize=8,
              weight="bold", color="#9a3412")
    axis.text(11.55, bot2_y + ROW_H / 2 - 0.25, "Nhánh B (identity)", fontsize=8,
              weight="bold", color="#9a3412", va="top")

    _arrow(axis, x1_cx, x1_cy - 0.58, 11.55, top2_y + ROW_H / 2, color="#c2410c")
    _arrow(axis, x1_cx, x1_cy - 0.58, 11.55, bot2_y + ROW_H / 2, color="#c2410c")

    for x, text in a2_boxes:
        _box(axis, x, top2_y, a2_w, ROW_H, text, box_color_main, fontsize=7.3)
    for left, right in zip(a2_boxes[:-1], a2_boxes[1:]):
        _arrow(axis, left[0] + a2_w + 0.01, top2_y + ROW_H / 2, right[0] - 0.04, top2_y + ROW_H / 2)

    _arrow(axis, a2_boxes[-1][0] + a2_w + 0.01, top2_y + ROW_H / 2, merge2_x - 0.22, merge2_y + 0.16)
    _arrow(axis, 11.55, bot2_y + ROW_H / 2, merge2_x - 0.22, merge2_y - 0.16,
           color="#c2410c", style="--")
    axis.text(14.6, bot2_y + ROW_H / 2 + 0.10, "identity (không phép toán)", fontsize=7,
              color="#9a3412", style="italic", va="bottom")
    _merge_node(axis, merge2_x, merge2_y)

    x2_box_x = geo["x2_box_x"]
    x2_cx, x2_cy = _box(axis, x2_box_x, merge2_y - 0.575, 1.9, 1.15,
                        "ReLU\n→ x2 (64×32)", box_color_out, fontsize=8, weight="bold")
    _arrow(axis, merge2_x + 0.26, merge2_y, x2_box_x - 0.04, x2_cy, color="#c2410c")

    # ------------------------------------------------------------------
    # Head — zone y trong [0.45, 1.6], nối từ x2 bằng đường gấp khúc (elbow)
    # để tránh cắt chéo qua toàn bộ sơ đồ.
    # ------------------------------------------------------------------
    head_y, head_boxes, head_w = geo["head_y"], geo["head_boxes"], geo["head_w"]
    for x, text in head_boxes:
        _box(axis, x, head_y, head_w, 1.15, text, "#fdba74", fontsize=8.2)
    for left, right in zip(head_boxes[:-1], head_boxes[1:]):
        _arrow(axis, left[0] + head_w + 0.01, head_y + 0.575, right[0] - 0.04, head_y + 0.575,
               color="#c2410c")

    elbow_y = 1.85  # nằm dưới đáy khối 2 (bot2_y ~2.1) và trên đỉnh Head (head_y+1.15=1.6)
    first_head_cx = head_boxes[0][0] + 0.95
    axis.plot([x2_cx, x2_cx], [x2_cy - 0.58, elbow_y], color="#c2410c", linewidth=1.7, zorder=3)
    axis.plot([x2_cx, first_head_cx], [elbow_y, elbow_y], color="#c2410c", linewidth=1.7, zorder=3)
    _arrow(axis, first_head_cx, elbow_y, first_head_cx, head_y + 1.19, color="#c2410c")

    # ------------------------------------------------------------------
    # Ghi chú công thức
    # ------------------------------------------------------------------
    axis.text(11.9, 1.15,
              "FP32:  y[t][co] = b[co] + Σ_k Σ_ci W[k][ci][co]·x̃[t+k−p][ci]",
              ha="center", va="center", fontsize=9, color="#7c2d12",
              bbox={"boxstyle": "round,pad=0.3", "facecolor": "#ffedd5", "edgecolor": "#fdba74"})
    axis.text(11.9, 0.65,
              "INT8:  q = clip(round(x / scale + zero_point), -128, 127)   |   "
              "x ≈ scale × (q − zero_point)",
              ha="center", va="center", fontsize=8.5, color="#7c2d12",
              bbox={"boxstyle": "round,pad=0.3", "facecolor": "#ffedd5", "edgecolor": "#fdba74"})


def run(config):
    import matplotlib

    matplotlib.use("Agg")
    # matplotlib.rcParams["svg.hashsalt"]: mặc định là None, khiến mỗi lần
    # chạy matplotlib bốc 1 salt ngẫu nhiên (uuid4) để sinh id cho các thẻ
    # <clipPath>/<pattern> trong SVG (ví dụ clip-path="url(#pXXXXXXXXXX)").
    # Kết quả: dù tắt <dc:date>, nội dung .svg vẫn đổi ở MỌI lần chạy dù hình
    # vẽ giống hệt nhau, làm git báo file "bị sửa" (diff rác). Đặt salt cố
    # định để id luôn giống nhau khi input/hình vẽ không đổi -> .svg
    # deterministic thật sự.
    matplotlib.rcParams["svg.hashsalt"] = "quantization-har-pipeline"
    import matplotlib.pyplot as plt

    output_dir = project_path(config["paths"]["results"]) / "figures" / "pipeline"
    output_dir.mkdir(parents=True, exist_ok=True)
    figure, axis = plt.subplots(figsize=(20, 15))
    axis.set_xlim(0, 20)
    axis.set_ylim(0, 15)
    axis.axis("off")
    figure.patch.set_facecolor("white")
    axis.text(10, 14.55, "QUANTIZATION HAR — LUỒNG XỬ LÝ VÀ THUẬT TOÁN",
              ha="center", va="center", fontsize=22, weight="bold", color="#0f172a")
    axis.text(10, 14.17,
              "Nhận diện hoạt động từ cảm biến UCI-HAR và so sánh FP32 với PTQ full-integer INT8",
              ha="center", va="center", fontsize=11, color="#475569")

    _lane(axis, 10.6, "A. DỮ LIỆU VÀ BASELINE FP32",
          ("#eff6ff", "#bfdbfe", "#1d4ed8"), [
              (0.6, "UCI-HAR\n7352 train + 2947 test\n128 × 9 cảm biến", "#dbeafe"),
              (3.15, "Chia dữ liệu\ntrain / validation / test\ntách theo subject", "#dbeafe"),
              (5.7, "Z-score\nmean, std fit trên train\nkhông rò rỉ test", "#dbeafe"),
              (8.25, "ResNetv1-6 1D\nConv/Pool + residual\n10.726 tham số", "#bfdbfe"),
              (10.8, "Huấn luyện FP32\nSGD, 300 epoch\nchọn checkpoint tốt nhất", "#bfdbfe"),
              (13.35, "baseline.keras\nFP32 checkpoint\n92,60% test", "#93c5fd"),
          ])
    _lane(axis, 7.6, "B. PTQ INT8, ĐÁNH GIÁ VÀ BÁO CÁO",
          ("#f0fdf4", "#bbf7d0", "#15803d"), [
              (0.6, "baseline.keras\nđầu vào FP32 đã\nchuẩn hóa", "#dcfce7"),
              (3.15, "Calibration từ train\nn = 50 / 200 / 500\nseed = 11 / 22 / 33", "#dcfce7"),
              (5.7, "Tìm scale và\nzero-point\ncho activation", "#dcfce7"),
              (8.25, "TFLite PTQ\nfull-integer INT8\ninput/output = int8", "#bbf7d0"),
              (10.8, "Validation\n9 cấu hình INT8\nchọn n = 500", "#bbf7d0"),
              (13.35, "Test cuối\naccuracy, macro-F1\ndung lượng, confusion matrix", "#86efac"),
          ])
    _arrow(axis, 14.3, 11.15, 14.3, 10.20, color="#15803d")
    axis.text(14.8, 10.68, "cùng checkpoint", ha="center", va="center",
              fontsize=8, color="#15803d")

    _lane_c_branches(axis)

    figure.tight_layout(pad=0.5)
    figure.savefig(output_dir / "algorithm_flow.png", dpi=180, bbox_inches="tight")
    # metadata={"Date": None} tắt hẳn thẻ <dc:date> mà matplotlib mặc định
    # nhúng vào SVG (ngày giờ chạy) — nếu không tắt, mỗi lần chạy lại (kể cả
    # khi nội dung hình không đổi) sha256 của .svg vẫn khác nhau, khiến
    # `git status` báo "bị sửa" dù không có thay đổi thật (diff rác).
    figure.savefig(output_dir / "algorithm_flow.svg", bbox_inches="tight",
                    metadata={"Date": None})
    plt.close(figure)
    print(f"Đã tạo sơ đồ tại {output_dir}")
    return output_dir / "algorithm_flow.png"
