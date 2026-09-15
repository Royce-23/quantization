# Quantization HAR — Nhóm 3 sinh viên

Khảo sát ảnh hưởng của PTQ INT8 đến độ chính xác và dung lượng mô hình nhận diện hoạt động từ cảm biến UCI-HAR.

Phần TV1 (dữ liệu và baseline FP32) đã được triển khai, chạy đủ 300 epoch và bàn giao checkpoint. Phần TV2–TV3 vẫn giữ trạng thái `CHƯA TRIỂN KHAI` cho đến khi thành viên phụ trách hoàn thiện.

## Bắt đầu ngay trong VS Code

1. Giải nén ZIP; mở thư mục `quantization-har-starter` bằng File > Open Folder.
2. Mở Terminal tại thư mục có README này.
3. Chạy `python main.py --check`. Windows có thể dùng `py main.py --check` nếu lệnh python không có.
4. Đọc [phân công](PHAN_CONG_CONG_VIEC.md), điền tên thành viên và deadline.
5. Đọc [giao diện bàn giao](docs/CONTRACT.md), sau đó mở file được giao và tìm `TODO`.
6. Xem [hướng dẫn GitHub](docs/GITHUB_GUIDE.md) để đẩy toàn bộ mã nguồn đã giải nén lên repository.

## Ai viết phần nào?

| Thành viên | File chính | Kết quả bàn giao |
| --- | --- | --- |
| TV1 | `src/prepare_data.py`, `src/model.py`, `src/train.py` | Dữ liệu, cấu hình, checkpoint FP32 |
| TV2 | `src/calibration.py`, `src/quantize.py` | FP32 xuất, INT8 và cấu hình calibration |
| TV3 | `src/evaluate.py`, `src/summarize_results.py`, `src/plot_results.py` | Metric, bảng, biểu đồ và phân tích |

`src/common.py`, `configs/`, `main.py` là giao diện chung: báo cả nhóm khi thay đổi.

## Lệnh điều phối đã có sẵn

```bash
python main.py --check
python main.py --help
python main.py prepare
python main.py train
python main.py visualize-tv1
python main.py quantize
python main.py evaluate
python main.py summarize
python main.py plot
```

`prepare` và `train` đã chạy được. Các lệnh `quantize`, `evaluate`, `summarize`, `plot` hiện vẫn dẫn tới module TODO và thoát mã 2. Mỗi thành viên giữ hàm `run(config)` để main gọi được sau khi hoàn thiện.

## Môi trường nghiên cứu

Tạo môi trường riêng bằng `python -m venv .venv`. Trên Windows PowerShell, có thể gọi trực tiếp `.venv\Scripts\python.exe` mà không cần kích hoạt môi trường.

Phần TV1 đã được kiểm chứng với Python 3.10.12, TensorFlow CPU 2.16.1 và NumPy 1.26.4. Cài thư viện rồi chạy:

```bash
python -m pip install -r requirements.txt
python main.py prepare
python main.py train
```

Lệnh `prepare` tự tải UCI-HAR nếu `data/raw/UCI HAR Dataset` chưa tồn tại. Lệnh `train` dùng cấu hình trong `configs/project.json` và lưu checkpoint tốt nhất theo validation accuracy.
Lệnh `visualize-tv1` tạo bốn hình PNG trong `results/figures/tv1/` để kiểm tra dữ liệu, lịch sử huấn luyện, kiến trúc và ma trận nhầm lẫn.

## Phạm vi nghiên cứu

- Dataset: [UCI-HAR](https://doi.org/10.24432/C54S4K), tự tải riêng; không kèm dữ liệu trong ZIP.
- Paper nền: [Quantization and Deployment of Deep Neural Networks on Microcontrollers](https://doi.org/10.3390/s21092984).
- Một baseline FP32; INT8 PTQ; calibration 50/200/500 cửa sổ là cấu hình khảo sát đề xuất.
- Giữ nguyên checkpoint giữa các cấu hình calibration. Calibration lấy từ train; chọn cấu hình bằng validation; test dùng cuối cùng.
- So sánh accuracy, macro-F1, dung lượng file. Không suy ra RAM/Flash, năng lượng hoặc tốc độ MCU từ laptop.
- Baseline dùng ResNetv1-6 ở Hình 4 của paper, điều chỉnh Conv/Pool thành 1D cho tensor UCI-HAR `(128, 9)`, chọn 32 filters.
- Huấn luyện bám mục 6.1.1: SGD, batch 64, 300 epoch, learning rate 0,05, momentum 0,9, weight decay `5e-4`, giảm learning rate ×0,13 tại epoch 100/200/250 và Mixup.
- Khác paper: nhóm tách validation theo subject, dùng Mixup alpha 0,2 do paper không công bố alpha và khảo sát TFLite PTQ INT8 thay vì gọi đây là tái hiện nguyên bản MicroAI QAT.

## Kết quả TV1 đã đo

| Chỉ số | Kết quả |
| --- | ---: |
| Validation accuracy | 95,82% |
| Validation macro-F1 | 0,9621 |
| Test accuracy | 92,60% |
| Test macro-F1 | 0,9271 |
| Số tham số | 10.726 |
| Checkpoint `.keras` | 148.495 byte |
| Epoch được chọn | 84/300 |

Xem mô tả và giới hạn tại [docs/data_and_baseline.md](docs/data_and_baseline.md).

## Thư mục

| Thư mục | Mục đích |
| --- | --- |
| `src/` | Mã nguồn theo từng người |
| `configs/` | Cấu hình dùng chung |
| `docs/` | Giao diện bàn giao, ghi chép paper, báo cáo |
| `data/raw/`, `data/processed/` | Dữ liệu cục bộ, không commit dữ liệu lớn |
| `models/` | Checkpoint và model xuất, không commit model lớn |
| `results/` | CSV, biểu đồ đã kiểm tra |
| `.github/` | Mẫu Issue, PR và CI kiểm tra bộ khung |

Xem [mốc tiến độ](docs/MILESTONES.md). CI chỉ kiểm tra cú pháp và cấu hình bộ khung, không xác nhận chất lượng nghiên cứu.
