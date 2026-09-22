# Tổng hợp kết quả thực nghiệm

Checkpoint: `aa323b3e55d1f904edbb05fb6d2a817b748b6e0ad26b7eb0166777c94535c74e`.
Calibration size được chọn bằng validation: **500**.
Độ lệch chuẩn là sample standard deviation qua các seed; n=1 ghi N/A.

## Validation — dùng để chọn cấu hình

| Model | n | Accuracy mean ± std (%) | Macro-F1 mean ± std | Bytes |
| --- | ---: | ---: | ---: | ---: |
| FP32 | 1 | 95.8249 ± N/A | 0.962063 ± N/A | 51728 |
| INT8 n=50 | 3 | 95.6678 ± 0.1695 | 0.960656 ± 0.001528 | 25104 |
| INT8 n=200 | 3 | 95.7127 ± 0.1029 | 0.961052 ± 0.000940 | 25104 |
| INT8 n=500 | 3 | 95.7576 ± 0.1782 | 0.961448 ± 0.001619 | 25104 |

## Test — chỉ sau khi chốt calibration size

| Model | n | Accuracy mean ± std (%) | Δ so với FP32 (điểm %) | Macro-F1 | Giảm size |
| --- | ---: | ---: | ---: | ---: | ---: |
| FP32 | 1 | 92.6026 ± N/A | +0.0000 | 0.927098 | 0.00% |
| INT8 n=500 | 3 | 92.1841 ± 0.2722 | -0.4185 | 0.923170 | 51.47% |

Không diễn giải kết quả validation như kết quả test cuối cùng. Các số trên là số đo từ artifact thật, không phải số minh họa.
