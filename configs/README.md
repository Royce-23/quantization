# Cấu hình chung

`project.json` đã chốt phần dữ liệu, ResNetv1-6 và huấn luyện FP32 của TV1 sau khi đối chiếu Hình 4, mục 6 và 6.1.1 của paper. `seed=42`, các mức calibration và seed calibration là đề xuất nhóm, không phải cấu hình của paper.

`labels.json` cố định ánh xạ nhãn gốc 1–6 của UCI-HAR sang output 0–5 của mô hình.

TV1 đã xác nhận đầu vào, nhãn, mạng và huấn luyện. TV2 đã chạy full-integer
PTQ cho đủ 3 size × 3 seed. TV3 đã xác nhận quy trình chọn bằng validation,
đánh giá test và cách tính metric. Cả nhóm cần review nếu thay đổi file này.

Section `export_c` (thêm cho `python main.py export-c`, xem
[`docs/model_architecture_and_c_inference.md`](../docs/model_architecture_and_c_inference.md))
khai báo đường dẫn checkpoint/metadata/preprocess/sanity dùng để trích tham
số, nơi ghi `params_fp32.npz`/`params_manifest.json` và thư mục
`firmware/`, cùng quy tắc định dạng số khi sinh mã C (`values_per_line`,
`float_format`). Cố ý **không** thêm key mới vào `paths` để không phá
`python main.py --check` (kiểm tra mọi thư mục trong `paths` tồn tại).
