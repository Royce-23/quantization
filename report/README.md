# Báo cáo LaTeX

`main.tex` là báo cáo (tiếng Việt, XeLaTeX); `main.pdf` là bản đã biên dịch.

## Biên dịch

```bash
cd report
tectonic -X compile main.tex        # hoặc: xelatex main.tex  (chạy 2 lần)
```

Cần các font TeX Gyre (Termes, Heros, Cursor) — có sẵn trong TeX Live và Overleaf (chọn XeLaTeX).
Hình nằm trong `figures/` nên thư mục `report/` tự chứa, có thể tải nguyên thư mục lên Overleaf.

## Số liệu trong báo cáo

Mọi bảng và con số trong `generated/` được sinh từ kết quả thật của repository, không gõ tay:

```bash
python main.py export-c            # nếu chưa có models/params_fp32.npz
python report/scripts/gen_tables.py
```

Script đọc `results/*.csv`, `models/*.json`, `models/params_fp32.npz`, `data/processed/split.json`,
và tự kiểm tra chéo với `results/parameters_reference.md` (thống kê tham số) cùng ma trận nhầm lẫn
(tổng đường chéo phải bằng số mẫu đúng ở `results/metrics.csv`).

Tên thành viên, MSSV, giảng viên và trường trên trang bìa để dấu chấm; điền trực tiếp trong `main.tex`.
