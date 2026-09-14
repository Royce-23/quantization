# Dữ liệu và baseline FP32 — TV1

## 1. Phần bài báo được sử dụng

Nguồn chính là Novac và cộng sự, *Quantization and Deployment of Deep Neural Networks on Microcontrollers*, Sensors 2021, DOI [10.3390/s21092984](https://doi.org/10.3390/s21092984).

Các quyết định TV1 bám theo bài báo:

- Mục 6 và Hình 4: ResNetv1-6, hai khối residual, convolution kernel 3, projection shortcut kernel 1, max pooling và lớp phân loại cuối. Đối với UCI-HAR, paper quy định convolution và pooling là một chiều.
- Mục 6.1.1: dùng tín hiệu cảm biến theo cửa sổ thay vì 561 đặc trưng tính sẵn. Mỗi mẫu dài 2,56 giây, gồm 128 thời điểm ở 50 Hz và 9 kênh; có 6 lớp hoạt động.
- Paper dùng 7.352 mẫu train, 2.947 mẫu test chính thức và z-score theo train.
- Baseline FP32 được huấn luyện bằng SGD, batch 64, 300 epoch, learning rate ban đầu 0,05, momentum 0,9 và weight decay `5e-4`. Learning rate nhân 0,13 tại epoch 100, 200 và 250. Paper cũng dùng Mixup.

Paper khảo sát nhiều độ rộng 16, 24, 32, 40, 48, 64 và 80 filters và báo trung bình 15 lần chạy. Đề tài này chọn cố định 32 filters vì mục tiêu chính là so sánh FP32 với PTQ INT8, không khảo sát độ rộng mạng.

## 2. Điểm điều chỉnh so với paper

Đây là triển khai dựa trên phương pháp của paper, không phải tái hiện nguyên bản:

1. Paper huấn luyện bằng toàn bộ official train; dự án tách subject 27–30 làm validation để chọn checkpoint mà không trộn cùng người giữa train và validation.
2. Paper không công bố hệ số Mixup, nên dự án ghi rõ lựa chọn `alpha=0.2`.
3. Paper dùng MicroAI và QAT cho nhánh INT8 chính; đề tài nhóm khảo sát TFLite PTQ INT8 với các kích thước calibration 50/200/500.
4. Paper báo trung bình 15 lần chạy. Kết quả hiện tại là một lần chạy có seed 42; chưa được hiểu là tái hiện giá trị trung bình của paper.
5. Không suy ra RAM, Flash, latency hay năng lượng MCU từ dung lượng file và phép đo trên laptop.

## 3. Dữ liệu và tiền xử lý

Nguồn dữ liệu: [UCI Human Activity Recognition Using Smartphones](https://doi.org/10.24432/C54S4K).

Thứ tự tensor đầu vào `[time, channel]` được cố định như sau:

1. `total_acc_x`, `total_acc_y`, `total_acc_z`;
2. `body_gyro_x`, `body_gyro_y`, `body_gyro_z`;
3. `body_acc_x`, `body_acc_y`, `body_acc_z`.

Thứ tự này theo ba nhóm tín hiệu được paper mô tả; paper không công bố thứ tự tensor chi tiết. Mọi module phía sau phải đọc `data/processed/preprocess.json`, không tự sắp xếp lại kênh.

| Split | Subject | Số mẫu |
| --- | --- | ---: |
| Train | 1, 3, 5, 6, 7, 8, 11, 14, 15, 16, 17, 19, 21, 22, 23, 25, 26 | 5.867 |
| Validation | 27, 28, 29, 30 | 1.485 |
| Test chính thức | 2, 4, 9, 10, 12, 13, 18, 20, 24 | 2.947 |

Mỗi kênh được chuẩn hóa theo `(x - mean_train) / std_train`. Mean/std chỉ tính từ 5.867 mẫu model-train. Code kiểm tra shape `(128, 9)`, nhãn đủ 0–5, số mẫu official, độ dài x/y/subject và NaN/infinity trước khi ghi artifact.

## 4. Kiến trúc FP32

Luồng chính của ResNetv1-6 1D:

```text
Input(128×9)
  ├─ Conv1D(32, k=3) → ReLU → Conv1D(32, k=3) → MaxPool(2) ─┐
  └─ Conv1D(32, k=1) → MaxPool(2) ---------------------------┤ Add → ReLU
       ├─ Conv1D(32, k=3) → ReLU → Conv1D(32, k=3) ----------┤ Add → ReLU
       └─ identity -------------------------------------------┘
  → GlobalMaxPool1D → Dense(6 logits)
```

Mạng có 10.726 tham số. Tên layer và shape có thể kiểm tra trong `src/model.py`.

## 5. Lệnh chạy và môi trường

Đã kiểm chứng trên Python 3.10.12, TensorFlow CPU 2.16.1 và NumPy 1.26.4:

```bash
python -m pip install -r requirements.txt
python main.py --check
python main.py prepare
python main.py train
```

`prepare` tự tải dữ liệu nếu cần. `train` luôn đọc `configs/project.json`, dùng seed 42, chạy đủ 300 epoch và lưu model có validation accuracy cao nhất.

## 6. Kết quả thực nghiệm

| Chỉ số | Kết quả |
| --- | ---: |
| Validation accuracy | 95,8249% |
| Validation macro-F1 | 0,9621 |
| Test accuracy | 92,6026% |
| Test macro-F1 | 0,9271 |
| Epoch checkpoint | 84/300 |
| Số tham số | 10.726 |
| Dung lượng `baseline.keras` | 148.495 byte (145,0 KiB) |

Checkpoint ID (SHA-256):

```text
aa323b3e55d43881f13c381dabfbf24c9eadea0972c0be2b75791a53b3bc08
```

Giá trị test khoảng 92,6% nằm trong vùng kết quả được thể hiện trên Hình 5 của paper, nhưng không được coi là trùng khớp trực tiếp vì dự án chỉ chạy một seed, dành bốn subject cho validation và dùng TensorFlow/Keras thay cho pipeline MicroAI của tác giả.

## 7. Artifact bàn giao TV2/TV3

- `data/processed/har.npz`: tensor float32 đã chuẩn hóa; file lớn không commit, tạo lại bằng `python main.py prepare`.
- `data/processed/preprocess.json`: mean/std, channel order và label order.
- `data/processed/split.json`: subject và số mẫu từng split.
- `data/processed/sanity_samples.npz`: 12 mẫu validation, logits, prediction và checkpoint ID để kiểm tra tích hợp.
- `models/baseline.keras`: checkpoint FP32 chính thức.
- `models/baseline_metadata.json`: hash, cấu hình, phiên bản và metric.
- `results/tv1_training_history.csv`: log 300 epoch.

TV2 phải dùng duy nhất checkpoint trên cho tất cả cấu hình calibration. Calibration chỉ lấy từ `x_train`; TV3 so sánh TFLite FP32 và INT8 trong cùng định dạng để tránh sai lệch do container file.
