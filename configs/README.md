# Cấu hình chung

`project.json` đã chốt phần dữ liệu, ResNetv1-6 và huấn luyện FP32 của TV1 sau khi đối chiếu Hình 4, mục 6 và 6.1.1 của paper. `seed=42`, các mức calibration và seed calibration là đề xuất nhóm, không phải cấu hình của paper.

`labels.json` cố định ánh xạ nhãn gốc 1–6 của UCI-HAR sang output 0–5 của mô hình.

TV1 xác nhận đầu vào, nhãn, mạng và huấn luyện. TV2 xác nhận quantization. TV3 xác nhận cách đánh giá. Cả nhóm review thay đổi file này.
