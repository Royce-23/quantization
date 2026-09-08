# Cấu hình chung

`project.json` có `null` ở các thông số cần đọc paper và quyết định. Không tự xem null là 0. `seed=42`, các mức calibration và seed calibration là đề xuất nhóm, không phải cấu hình đã xác minh của paper.

TV1 xác nhận đầu vào, nhãn, mạng và huấn luyện. TV2 xác nhận quantization. TV3 xác nhận cách đánh giá. Cả nhóm review thay đổi file này.
