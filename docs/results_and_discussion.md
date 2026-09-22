# Kết quả và thảo luận

Các số dưới đây được đọc từ model TFLite thật qua `src/evaluate.py`, không
chép từ paper và không phải số minh họa. FP32 và INT8 dùng đúng cùng tensor
đã chuẩn hóa, thứ tự nhãn và checkpoint.

## Quy tắc đánh giá

- Chạy FP32 và cả 9 model INT8 trên validation.
- Gom ba seed theo từng calibration size; chọn size có mean accuracy cao nhất.
- Nếu bằng nhau, ưu tiên mean macro-F1 rồi calibration size nhỏ hơn.
- Sau khi chốt size mới mở test; đánh giá FP32 và cả ba seed của size đã chọn.
- Macro-F1 tính đều trên 6 lớp. Độ lệch chuẩn là sample SD qua ba seed.

Quy trình này tránh dùng test để lựa chọn cấu hình. Kết quả validation dùng
cho lựa chọn, không được trình bày như kết quả tổng quát hóa cuối cùng.

## Kết quả validation

| Model | Accuracy mean ± SD (%) | Macro-F1 mean ± SD |
| --- | ---: | ---: |
| FP32 | 95,8249 | 0,962063 |
| INT8, n=50 | 95,6678 ± 0,1695 | 0,960656 ± 0,001528 |
| INT8, n=200 | 95,7127 ± 0,1029 | 0,961052 ± 0,000940 |
| INT8, n=500 | **95,7576 ± 0,1782** | **0,961448 ± 0,001619** |

`n=500` có mean validation accuracy cao nhất trong ba cấu hình INT8 nên được
chọn trước khi chạy test. Chênh lệch giữa ba size nhỏ; kết luận phù hợp là
500 tốt nhất trong phạm vi ba seed đã chạy, không khẳng định luôn tốt nhất.

## Kết quả test cuối cùng

| Model | Accuracy mean ± SD (%) | Macro-F1 mean ± SD | Dung lượng |
| --- | ---: | ---: | ---: |
| FP32 TFLite | 92,6026 | 0,927098 | 51.728 byte |
| INT8 n=500 (3 seed) | 92,1841 ± 0,2722 | 0,923170 ± 0,002580 | 25.104 byte |

So với FP32, INT8 giảm trung bình **0,4185 điểm phần trăm accuracy** và giảm
**51,47% dung lượng file**, tương đương file nhỏ hơn **2,06 lần**. Tỉ lệ file
không đạt đúng 4 lần dù trọng số từ 32 bit xuống 8 bit vì model nhỏ và file
TFLite còn chứa graph, shape, scale/zero-point và metadata cố định.

Ma trận nhầm lẫn cho thấy lỗi lớn nhất của cả hai model vẫn là `SITTING` và
`STANDING`. Đây là hai hoạt động tĩnh có tín hiệu quán tính giống nhau;
lượng tử hóa không làm thay đổi bản chất điểm khó chính này.

## Lệnh tái lập

```bash
python main.py quantize
python main.py evaluate
python main.py summarize
python main.py plot
python -m pytest -q
```

Số liệu chi tiết nằm trong `results/metrics.csv`, thống kê trong
`results/summary.csv` và hình tại `results/figures/quantization/`.

## Giới hạn kết luận

- Chỉ khảo sát một kiến trúc, một dataset và ba calibration size.
- Chỉ có ba seed calibration; khoảng biến thiên chưa đại diện cho mọi mẫu.
- Không đo latency, RAM, Flash hay năng lượng trên phần cứng.
- Kết quả là TFLite PTQ INT8, không so sánh trực tiếp như thể cùng phương pháp
  với MicroAI QAT trong paper.
