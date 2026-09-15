# data/processed

`python main.py prepare` sinh:

- `har.npz`: train/validation/test đã chuẩn hóa (không commit vì dung lượng lớn);
- `preprocess.json`: mean/std, thứ tự kênh, nhãn;
- `split.json`: subject và số mẫu từng split;
- `sanity_samples.npz`: 12 mẫu validation kèm output checkpoint sau khi train.

Không chuẩn hóa lần hai khi TV2/TV3 đọc `har.npz`.
