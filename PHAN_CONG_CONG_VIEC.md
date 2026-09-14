# Phân công nghiên cứu Quantization — Nhóm 3 sinh viên

> Đề tài: Khảo sát ảnh hưởng của lượng tử hóa sau huấn luyện INT8 đến độ chính xác và dung lượng mô hình nhận diện hoạt động từ dữ liệu cảm biến.
>
> Phạm vi: nghiên cứu thực nghiệm trên laptop, dùng UCI-HAR; đọc paper, triển khai một phần, đánh giá và giải thích kết quả. Chưa triển khai phần cứng.

## 1. Thành viên và phạm vi

Điền tên, tài khoản GitHub và hạn hoàn thành trước khi bắt đầu. Các ô chưa đánh dấu là việc chưa hoàn thành.

| Vai trò | Họ tên | GitHub | Phụ trách | Nhánh đề xuất |
| --- | --- | --- | --- | --- |
| TV1 | Điền tên | Điền username | Dữ liệu và baseline FP32 | `feature/baseline` |
| TV2 | Điền tên | Điền username | PTQ INT8 và calibration | `feature/quantization` |
| TV3 | Điền tên | Điền username | Đánh giá và phân tích | `feature/evaluation` |

- Một dataset, một cấu hình mạng baseline, một kỹ thuật PTQ INT8.
- Thí nghiệm mở rộng: thay đổi số cửa sổ calibration (dữ liệu đại diện phục vụ lượng tử hóa).
- Chỉ số chính: accuracy, macro-F1, dung lượng file mô hình; ma trận nhầm lẫn hỗ trợ phân tích.
- Không yêu cầu firmware, cảm biến thật, UART, đo RAM/Flash MCU hoặc năng lượng.
- Không yêu cầu đề xuất thuật toán mới hoặc công bố tạp chí.
- Cả ba cùng đọc paper, viết code và viết phần báo cáo của mình.

## 2. Tài liệu nền và quyết định chung

- Paper chính đã chọn: [Quantization and Deployment of Deep Neural Networks on Microcontrollers](https://doi.org/10.3390/s21092984).
- Dataset: [UCI Human Activity Recognition Using Smartphones](https://doi.org/10.24432/C54S4K).
- Phạm vi dự kiến: phần UCI-HAR và nhánh INT8 PTQ đã thảo luận; phải đối chiếu cấu hình với paper trước khi triển khai.
- Nếu thay kiến trúc hoặc quy trình: ghi rõ là áp dụng phương pháp trên cấu hình điều chỉnh, không gọi là tái hiện nguyên cấu hình của paper.

| Quyết định | Giá trị nhóm chốt |
| --- | --- |
| Cấu hình mạng / vị trí trong paper | ResNetv1-6, 32 filters; Hình 4 và mục 6.1.1 |
| Kích thước đầu vào và thứ tự kênh | `(128, 9)`; xem `data/processed/preprocess.json` |
| Bộ chia train / validation / test | Giữ test chính thức; tách validation theo người trong train |
| Công thức chuẩn hóa | Chỉ tính tham số từ train; lưu để dùng chung |
| Framework và phiên bản | Python 3.10.12, TensorFlow CPU 2.16.1, NumPy 1.26.4 |
| Checkpoint FP32 chính thức | `models/baseline.keras`; SHA-256 bắt đầu `aa323b3e55d4` |
| Ngày bắt đầu / hạn nộp | Điền ngày |

## 3. TV1 — Dữ liệu và mô hình FP32

### Checklist

- [x] Đọc phần dữ liệu, kiến trúc, huấn luyện trong paper; ghi thông số chưa được công bố.
- [x] Nạp các cửa sổ tín hiệu UCI-HAR; xác nhận không nhầm với bảng đặc trưng tính sẵn.
- [x] Kiểm tra số mẫu, nhãn, chiều tensor, thứ tự kênh và giá trị bất thường.
- [x] Chuẩn bị bộ chia train/validation/test, tránh trộn dữ liệu cùng người qua train và validation.
- [x] Tính chuẩn hóa từ train; lưu tham số và ánh xạ nhãn.
- [x] Dựng mạng và bàn giao sớm một mô hình huấn luyện thử cho TV2/TV3.
- [x] Huấn luyện baseline; chọn checkpoint bằng validation.
- [x] Chốt checkpoint, cấu hình, seed và hướng dẫn chạy lại.
- [x] Viết phần báo cáo: bài toán, dữ liệu, kiến trúc và huấn luyện.

### File phụ trách dự kiến

| File | Công dụng |
| --- | --- |
| `src/prepare_data.py` | Đọc, chia và chuẩn hóa dữ liệu |
| `src/model.py` | Định nghĩa mạng |
| `src/train.py` | Huấn luyện và lưu checkpoint |
| `configs/preprocess.json` | Tham số chuẩn hóa và thứ tự kênh |
| `configs/labels.json` | Ánh xạ nhãn |
| `docs/data_and_baseline.md` | Ghi chép và nội dung báo cáo |

**Bàn giao:** dữ liệu đầu vào được mô tả rõ, checkpoint FP32, cấu hình, mẫu kiểm chứng và lệnh chạy thực tế. Dữ liệu/checkpoint lớn có thể bàn giao bằng vị trí tải thống nhất, không cần commit toàn bộ vào Git.

**Hoàn thành khi:** TV2 và TV3 tải mô hình và dự đoán được trên cùng mẫu mà không sửa thủ công tiền xử lý.

## 4. TV2 — Lượng tử hóa INT8

### Checklist

- [ ] Đọc và giải thích FP32, INT8, scale, zero-point, PTQ và calibration.
- [ ] Nhận mô hình thử và quy ước đầu vào từ TV1.
- [ ] Viết hàm lấy calibration từ train, không dùng validation/test.
- [ ] Xuất mô hình FP32 và INT8 để đánh giá bằng cùng họ runtime.
- [ ] Kiểm tra kiểu dữ liệu đầu vào/đầu ra, scale, zero-point và toán tử còn float nếu có.
- [ ] Thử dự đoán; kiểm tra chuyển đổi số nguyên và tránh chuẩn hóa hai lần.
- [ ] Chạy các mức calibration 50, 200, 500 cửa sổ trên cùng checkpoint.
- [ ] Nếu nguồn lực cho phép, lặp ba seed lấy mẫu cho mỗi mức.
- [ ] Lưu cấu hình, seed, tên mô hình và log từng lần chạy.
- [ ] Viết phần báo cáo: phương pháp và thực nghiệm lượng tử hóa.

### File phụ trách dự kiến

| File | Công dụng |
| --- | --- |
| `src/calibration.py` | Chọn dữ liệu đại diện |
| `src/quantize.py` | Xuất FP32 và INT8 |
| `configs/experiments.json` | Cấu hình và seed thí nghiệm |
| `docs/quantization.md` | Giải thích và ghi chép phương pháp |

**Bàn giao:** mô hình FP32 xuất, các mô hình INT8, thông tin đầu vào/đầu ra và bảng ánh xạ mô hình–cấu hình.

**Hoàn thành khi:** chương trình đánh giá của TV3 chạy được mọi cấu hình và truy được nguồn checkpoint/calibration.

## 5. TV3 — Đánh giá và phân tích

### Checklist

- [ ] Thống nhất cách tính accuracy, macro-F1 và dung lượng file.
- [ ] Viết bộ đánh giá FP32/INT8; kiểm tra đúng nhãn và xử lý đầu vào INT8.
- [ ] Tính ma trận nhầm lẫn và kiểm tra nhãn trục biểu đồ.
- [ ] Đo dung lượng file thực tế, dùng KiB = byte / 1024.
- [ ] Đánh giá trên validation để hỗ trợ chọn cấu hình.
- [ ] Sau khi chốt, đánh giá cuối trên test bằng cùng danh sách mẫu.
- [ ] Tổng hợp trung bình và độ lệch chuẩn nếu có nhiều seed.
- [ ] Vẽ biểu đồ accuracy theo số mẫu calibration và dung lượng FP32–INT8.
- [ ] Đối chiếu phần kết quả tương ứng trong paper; ghi rõ khác biệt cấu hình.
- [ ] Viết kết quả, thảo luận và giới hạn; ghép báo cáo với hai thành viên còn lại.

### File phụ trách dự kiến

| File | Công dụng |
| --- | --- |
| `src/evaluate.py` | Đánh giá chung |
| `src/summarize_results.py` | Tổng hợp các lần chạy |
| `src/plot_results.py` | Vẽ biểu đồ |
| `results/metrics.csv` | Kết quả thực nghiệm |
| `docs/results_and_discussion.md` | Phân tích và đối chiếu |

**Bàn giao:** chương trình đánh giá, bảng số liệu có cấu hình nguồn, đồ thị và nhận xét được dữ liệu hỗ trợ.

**Hoàn thành khi:** nhóm chạy lại được bảng/biểu đồ và không có số liệu minh họa bị trình bày như kết quả thật.

## 6. Theo dõi 5 phase

| Phase | TV1 | TV2 | TV3 | Hạn hoàn thành |
| --- | --- | --- | --- | --- |
| 1. Exploration | Dữ liệu và kiến trúc | Phương pháp PTQ | Cách đánh giá | Điền ngày |
| 2. Reproduction | Baseline FP32 | Xuất INT8 | Bộ đánh giá | Điền ngày |
| 3. Benchmark | Kiểm tra baseline | Kiểm tra chuyển đổi | So sánh FP32–INT8 | Điền ngày |
| 4. Exploration | Kiểm tra dữ liệu | Thí nghiệm calibration | Tổng hợp và biểu đồ | Điền ngày |
| 5. Research | Phân tích dữ liệu | Phân tích lượng tử | Thảo luận và giới hạn | Điền ngày |

- [ ] Phase 1: đọc lướt các paper đã tìm, đọc kỹ một bài, chốt phần thực nghiệm và đầu vào chung.
- [ ] Phase 2: FP32 và INT8 chạy được qua cùng quy trình đánh giá.
- [ ] Phase 3: có bảng và biểu đồ baseline đã được nhóm kiểm tra.
- [ ] Phase 4: hoàn thành khảo sát kích thước calibration.
- [ ] Phase 5: hoàn thiện báo cáo, slide và buổi chạy demo trên laptop.

Phase 5 tập trung giải thích kết quả và đề xuất hướng tiếp tục. Chưa khẳng định đóng góp mới nếu chưa kiểm tra nghiên cứu liên quan.

## 7. Quy tắc thí nghiệm chung

1. Dùng chung bộ chia dữ liệu, thứ tự kênh, nhãn và tham số chuẩn hóa.
2. Mọi cấu hình calibration dùng cùng checkpoint FP32; không huấn luyện lại cho từng mức.
3. Giữ cùng cách lấy mẫu khi thay số lượng mẫu; lưu seed. Calibration chỉ lấy từ train.
4. Chọn cấu hình bằng validation; test dùng cho đánh giá cuối, không dùng để điều chỉnh tiếp.
5. Nếu lặp seed, báo cáo đủ các lần chạy thay vì chỉ lấy kết quả đẹp nhất.
6. Dung lượng file khác RAM chạy mô hình. Không suy ra RAM/Flash MCU hay tốc độ MCU từ thí nghiệm laptop.
7. Các mức 50/200/500 là cấu hình khảo sát đề xuất, không phải kết quả hoặc cấu hình đã xác nhận của paper.
8. Code có AI hỗ trợ phải được người phụ trách đọc hiểu, kiểm tra và giải thích được.

### Bảng kết quả cuối dự kiến

| Cấu hình | Calibration | Accuracy (%) | Macro-F1 (0–1) | Dung lượng (KiB) |
| --- | ---: | ---: | ---: | ---: |
| FP32 | Không áp dụng | Chưa đo | Chưa đo | Chưa đo |
| INT8-A | 50 | Chưa đo | Chưa đo | Chưa đo |
| INT8-B | 200 | Chưa đo | Chưa đo | Chưa đo |
| INT8-C | 500 | Chưa đo | Chưa đo | Chưa đo |

Trong `metrics.csv`, nên lưu: `experiment_id`, `checkpoint_id`, `model_file`, `split`, `calibration_size`, `calibration_seed`, `accuracy_pct`, `macro_f1`, `model_size_bytes`. Mỗi cấu hình/seed là một dòng; bảng tổng hợp được tạo từ dữ liệu này.

## 8. Phối hợp trên GitHub

### Quy trình nhóm

1. Mỗi công việc nhỏ tạo một Issue, gán đúng người phụ trách và phase. Thành viên cần được thêm vào repository để gán việc phù hợp.
2. Mỗi người làm trên nhánh riêng như bảng thành viên; cập nhật từ nhánh chung trước khi làm tiếp.
3. Commit theo từng việc rõ ràng; không sửa file của người khác mà chưa trao đổi.
4. Mở Pull Request (PR) về `main`; mô tả đã đổi gì, cách chạy và kết quả kiểm tra.
5. Một thành viên khác đọc PR trước khi nhóm ghép vào `main`.
6. Ghi số Issue/PR vào công việc liên quan. Đánh dấu checklist bằng cách đổi `[ ]` thành `[x]` rồi lưu file.

Đây là quy ước làm việc đề xuất; file Markdown không tự tạo Issue, gán người hoặc bật bảo vệ nhánh.

### Phân công đọc chéo

| Tác giả PR | Người đọc chính |
| --- | --- |
| TV1 | TV2 |
| TV2 | TV3 |
| TV3 | TV1 |

### Danh sách Issue đề xuất

| Mã công việc (không phải số Issue GitHub) | Tiêu đề | Người phụ trách | Phụ thuộc |
| --- | --- | --- | --- |
| TASK-01 | Chốt cấu hình, đầu vào và phần paper tái hiện | Cả nhóm; TV1 tổng hợp | Không |
| TASK-02 | Chuẩn bị UCI-HAR và chuẩn hóa | TV1 | TASK-01 |
| TASK-03 | Huấn luyện baseline FP32 | TV1 | TASK-02 |
| TASK-04 | Xây quy trình xuất INT8 | TV2 | Mô hình thử từ TV1 |
| TASK-05 | Viết chương trình đánh giá chung | TV3 | Đầu vào thống nhất và mô hình thử |
| TASK-06 | So sánh baseline FP32–INT8 | TV3 | TASK-03, TASK-04, TASK-05 |
| TASK-07 | Chạy khảo sát calibration | TV2 | TASK-06 |
| TASK-08 | Tổng hợp kết quả và biểu đồ | TV3 | TASK-07 |
| TASK-09 | Hoàn thiện báo cáo và demo | Cả nhóm | TASK-08 |

### Mẫu nội dung Issue

```markdown
## Mục tiêu
Mô tả một kết quả cụ thể cần đạt.

## Phụ trách / hạn hoàn thành
- Thành viên:
- Phase:
- Hạn:

## Đầu vào và phụ thuộc
- File / cấu hình / Issue liên quan:

## Việc cần làm
- [ ] Công việc 1
- [ ] Công việc 2

## Điều kiện hoàn thành
- [ ] Có sản phẩm bàn giao và cách chạy
- [ ] Đã kiểm tra kết quả
- [ ] Có PR và người đọc chéo
```

## 9. Quy ước file trong repository

Các đường dẫn dưới đây là cấu trúc dự kiến, chưa có code triển khai đi kèm tài liệu này.

| Đường dẫn | Nội dung | Người quản lý |
| --- | --- | --- |
| `README.md` | Giới thiệu và lệnh chạy đã kiểm chứng | Cả nhóm |
| `PHAN_CONG_CONG_VIEC.md` | Phân công và tiến độ | Cả nhóm |
| `requirements.txt` | Phiên bản thư viện đã chạy được | TV1 khởi tạo, cả nhóm thống nhất |
| `src/` | Các script đã phân công | Theo từng file |
| `configs/` | Tham số và cấu hình dùng chung | TV1 + TV2 |
| `docs/` | Ghi chú paper và nội dung báo cáo | Cả nhóm |
| `results/` | CSV và biểu đồ cuối | TV3 |
| `data/`, `models/` | Dữ liệu và mô hình cục bộ | TV1 + TV2 |

Không commit dữ liệu lớn, môi trường Python, cache hoặc checkpoint tạm. Ghi cách tải dữ liệu và vị trí bàn giao mô hình trong README; commit code, cấu hình, bảng số liệu nhỏ và tài liệu cần thiết. Các tên file dùng chung chỉ đổi sau khi báo hai thành viên còn lại.

## 10. Checklist nộp bài

- [ ] Một baseline FP32 với cấu hình rõ ràng.
- [ ] Quy trình PTQ INT8 chạy lại được.
- [ ] Một khảo sát calibration có kiểm soát.
- [ ] Bảng accuracy, macro-F1 và dung lượng từ kết quả thật.
- [ ] Biểu đồ có tên trục, đơn vị và chú thích.
- [ ] Báo cáo nêu phần tái hiện, phần thay đổi và giới hạn.
- [ ] README ghi môi trường, dữ liệu và thứ tự chạy.
- [ ] Cả ba hiểu luồng dữ liệu → FP32 → INT8 → đánh giá.
- [ ] Mỗi người trình bày phần mình phụ trách; demo hoàn toàn trên laptop.
