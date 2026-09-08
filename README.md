# Quantization HAR — Nhóm 3 sinh viên

Khảo sát ảnh hưởng của PTQ INT8 đến độ chính xác và dung lượng mô hình nhận diện hoạt động từ cảm biến UCI-HAR.

**Đây là bộ khung để nhóm viết tiếp, chưa có mô hình huấn luyện hay kết quả nghiên cứu.** Các bước nghiên cứu báo rõ `CHUA TRIỂN KHAI` cho đến khi thành viên hoàn thiện. Chương trình kiểm tra bộ khung chạy được bằng thư viện chuẩn Python, không cần GPU, dataset hoặc TensorFlow.

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
python main.py quantize
python main.py evaluate
python main.py summarize
python main.py plot
```

Các lệnh từ `prepare` đến `plot` hiện chỉ dẫn tới module TODO và thoát mã 2. Đây là trạng thái có chủ ý, không phải đã chạy thí nghiệm thành công. Mỗi thành viên giữ hàm `run(config)` để main gọi được sau khi hoàn thiện.

## Môi trường nghiên cứu

Tạo môi trường riêng bằng `python -m venv .venv`. Trên Windows PowerShell, có thể gọi trực tiếp `.venv\Scripts\python.exe` mà không cần kích hoạt môi trường.

`requirements.txt` hiện không cài thêm thư viện. Khi nhóm bắt đầu code AI, thống nhất phiên bản Python/TensorFlow và điền các thư viện cần thiết. Sau đó cài bằng `.venv\Scripts\python.exe -m pip install -r requirements.txt`. Chỉ ghi phiên bản đã chạy được; gói ZIP chưa kiểm chứng môi trường huấn luyện AI.

## Phạm vi nghiên cứu

- Dataset: [UCI-HAR](https://doi.org/10.24432/C54S4K), tự tải riêng; không kèm dữ liệu trong ZIP.
- Paper nền: [Quantization and Deployment of Deep Neural Networks on Microcontrollers](https://doi.org/10.3390/s21092984).
- Một baseline FP32; INT8 PTQ; calibration 50/200/500 cửa sổ là cấu hình khảo sát đề xuất.
- Giữ nguyên checkpoint giữa các cấu hình calibration. Calibration lấy từ train; chọn cấu hình bằng validation; test dùng cuối cùng.
- So sánh accuracy, macro-F1, dung lượng file. Không suy ra RAM/Flash, năng lượng hoặc tốc độ MCU từ laptop.
- Kiến trúc và đầu vào còn chờ đối chiếu paper. Nếu thay kiến trúc, ghi rõ là áp dụng phương pháp trên mô hình điều chỉnh.

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
