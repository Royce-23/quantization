# KNOWN ISSUES / BẪY ĐÃ BIẾT

File này ghi lại các lỗi/bẫy dạng "tưởng là nguyên nhân A nhưng thật ra là B"
(root cause không hiển nhiên, dễ tái phát nếu người khác không biết) đã gặp
trong quá trình phát triển repo này. Mỗi mục theo format:

```
## <tiêu đề ngắn>
- Triệu chứng:
- Tưởng là do:
- Nguyên nhân thật:
- Cách tránh / sửa:
- Liên quan tới:
```

---

## `test_generated_params_source_has_no_nan_inf` cấm chuỗi con `"inf"`/`"nan"`

- Triệu chứng: thêm một comment tiếng Việt bình thường (ví dụ có chữ
  "inference", "info", "infinity", "nanh", "nano"...) vào phần văn bản do
  `_render_params_*` trong `src/export_c.py` sinh ra khiến
  `tests/test_c_export.py::test_generated_params_source_has_no_nan_inf` FAIL.
- Tưởng là do: giá trị NaN/Inf thật sự lọt vào trọng số đã export.
- Nguyên nhân thật: test kiểm tra bằng **substring** trên toàn bộ nội dung
  file (`"inf" not in text.lower()`), không phải kiểm tra riêng các literal
  số — nên bất kỳ từ tiếng Anh nào chứa "inf"/"nan" làm chuỗi con cũng bị
  bắt nhầm, kể cả khi không liên quan gì tới giá trị số.
- Cách tránh / sửa: không thêm comment mới vào các hàm `_render_params_*`
  của `src/export_c.py`; nếu cần thêm comment, tránh các từ chứa "inf"/"nan"
  làm substring.
- Liên quan tới: `src/export_c.py`, `tests/test_c_export.py:~100-108`.

## `main.py --check` chạy trên CI không có numpy/tensorflow/matplotlib

- Triệu chứng: `python main.py --check` chạy OK trên máy dev (có đủ thư
  viện) nhưng FAIL trên CI (`.github/workflows/scaffold-check.yml`).
- Tưởng là do: lỗi cấu hình CI hoặc thiếu bước cài đặt.
- Nguyên nhân thật: `main.py --check` import **mọi** module trong `STAGES`
  ở top-level (`importlib.import_module`); CI Python 3.11 không cài
  numpy/tensorflow/matplotlib. Nếu một module mới (ví dụ
  `src/describe_params.py`) import numpy/tensorflow ở top-level (ngoài hàm
  `run()`), `--check` sẽ ImportError ngay trên CI dù chạy được trên máy dev.
- Cách tránh / sửa: mọi module được đăng ký vào `STAGES` chỉ được import
  numpy/tensorflow/matplotlib **bên trong hàm `run()`** (hoặc helper được
  gọi từ `run()`), không bao giờ ở top-level. Có test AST kiểm tra việc này
  (xem `test_describe_params_module_imports_without_heavy_deps`,
  `test_export_c_module_imports_without_heavy_deps`).
- Liên quan tới: `main.py`, `src/export_c.py`, `src/describe_params.py`,
  `src/visualize_pipeline.py`.

## `test_config_has_export_c_section` khoá cứng `config["paths"]`

- Triệu chứng: thêm một đường dẫn cấu hình mới (ví dụ đường dẫn output cho
  1 stage mới) vào `config["paths"]` trong `configs/project.json` làm
  `test_config_has_export_c_section` FAIL dù giá trị hoàn toàn hợp lý.
- Tưởng là do: test lỗi thời, cần cập nhật lại tập key kỳ vọng.
- Nguyên nhân thật: test cố tình `assert set(config["paths"].keys()) == {...
  4 key cố định ...}` để khoá cứng giao diện `paths` dùng chung toàn dự án
  (tránh các thành viên khác nhau vô tình thêm key trùng ý nghĩa khác tên).
  Đường dẫn cấu hình riêng cho 1 tính năng mới phải nằm ở mục **top-level
  riêng** (ví dụ `config["describe_params"]`), không phải trong `paths`.
- Cách tránh / sửa: không thêm key mới vào `config["paths"]`; tạo mục cấu
  hình top-level riêng (giống `config["export_c"]`).
- Liên quan tới: `configs/project.json`, `tests/test_c_export.py`.

## Dict Python coi `0.0` và `-0.0` là CÙNG một key (viết test cho dấu 0 dễ sai)

- Triệu chứng: viết test kiểm tra `_format_value` cho cả `0.0` lẫn `-0.0`
  bằng cách gom vào một `dict` literal (`{0.0: "0.0f", -0.0: "-0.0f", ...}`)
  rồi lặp `for value, expected in cases.items()` — test FAIL với thông báo
  `assert '0.0f' == '-0.0f'` dù hàm `_format_value` hoạt động đúng (đã tự
  kiểm chứng độc lập bằng `python -c` cho ra đúng `-0.0f`).
- Tưởng là do: `_format_value` xử lý sai dấu của số 0 âm.
- Nguyên nhân thật: trong Python, `0.0 == -0.0` là `True` và `hash(0.0) ==
  hash(-0.0)`, nên khi dùng làm **key của dict**, hai giá trị này va chạm
  (collapse) thành 1 key duy nhất — dict giữ lại **key object đầu tiên**
  (`0.0`, dấu dương) nhưng **value bị ghi đè bởi lần insert sau** (`"-0.0f"`
  của `-0.0`). Kết quả: khi lặp, nhận được `value = 0.0` (dương) nhưng
  `expected = "-0.0f"` (từ lần insert `-0.0` bị collapse) — lỗi nằm ở TEST,
  không nằm ở hàm được kiểm thử.
- Cách tránh / sửa: không dùng `dict` khi cần phân biệt `0.0` và `-0.0` (hay
  bất kỳ cặp giá trị nào `==` nhau nhưng cần giữ riêng biệt, kể cả `True`/`1`
  và `1.0`); dùng `list[tuple]` thay thế. Xem
  `tests/test_c_export.py::test_format_value_emits_valid_c_float_literals`
  để có ví dụ đã sửa kèm comment giải thích.
- Liên quan tới: `src/export_c.py::_format_value`, `tests/test_c_export.py`.

## Test quét "tham chiếu chết tới thư mục ghi chú nội bộ" có thể tự phát hiện CHÍNH nó (false positive)

Bối cảnh: repo dùng quy trình bàn giao 4-agent (Planner → Coder → Tester →
Reviewer) ghi chú nội bộ vào một thư mục con nằm trong `.gitignore` (không
commit, không xuất hiện trên repository công khai). Có một test quét toàn
bộ mã nguồn để đảm bảo không còn chỗ nào trong code/docs sản phẩm trỏ (chết)
tới thư mục đó — vì người đọc repo từ bên ngoài (kể cả bản thân CI) không
bao giờ có thư mục này nên mọi tham chiếu tới nó đều là tham chiếu chết.

- Triệu chứng: viết `tests/test_c_export.py::test_no_dangling_bangiao_references`
  để quét toàn bộ `tests/*.py` tìm literal đường dẫn thư mục ghi chú nội bộ
  nói trên — nhưng chính bản thân file test này (glob `tests/*.py` bao gồm
  cả chính nó) chứa nguyên văn literal đó ngay trong logic kiểm tra, nên
  test luôn tự FAIL với chính nó là "offender" dù không có tham chiếu chết
  thật nào còn sót lại trong repo.
- Tưởng là do: còn sót tham chiếu tới thư mục ghi chú nội bộ ở đâu đó trong
  code/docs sản phẩm mà chưa dọn hết.
- Nguyên nhân thật: bất kỳ file nào định nghĩa logic "tìm chuỗi X" bằng cách
  viết literal chuỗi X liền mạch trong mã nguồn của chính nó, rồi bị chính
  glob pattern của nó quét trúng, sẽ luôn tự khớp — đây là kiểu lỗi "kiểm tra
  tự tham chiếu" (self-referential check), không phải lỗi ở nội dung repo.
- Cách tránh / sửa: ghép chuỗi cần tìm từ 2+ phần tại runtime (nối tên thư
  mục với dấu gạch chéo bằng phép cộng chuỗi) thay vì viết liền mạch, để bản
  thân file chứa logic kiểm tra không chứa nguyên văn chuỗi bị cấm; đồng
  thời tránh nhắc lại nguyên văn chuỗi đó trong docstring/comment giải thích
  (comment cũng nằm trong phạm vi bị quét).
- Liên quan tới: `tests/test_c_export.py::test_no_dangling_bangiao_references`.

## Tắt `<dc:date>` trong SVG (matplotlib) chưa đủ để file `.svg` deterministic

- Triệu chứng: đã truyền `metadata={"Date": None}` cho `figure.savefig(...svg...)`
  để loại timestamp (xác nhận đúng: `grep "dc:date"` trên file sinh ra không
  còn kết quả nào) — nhưng chạy `python main.py visualize-pipeline` 2 lần
  liên tiếp, `sha256sum` của `.svg` vẫn khác nhau ở MỌI lần chạy, dù nội dung
  hình vẽ (toạ độ, màu, chữ) giống hệt nhau. `diff` 2 file `.svg` cho thấy
  chênh lệch nằm ở các thuộc tính `clip-path="url(#pXXXXXXXXXX)"` — chuỗi
  hex 10 ký tự đổi mỗi lần chạy.
- Tưởng là do: vẫn còn sót thẻ metadata khác chứa timestamp (ví dụ
  `Creator`/`Format`) mà `metadata={"Date": None}` chưa che hết.
- Nguyên nhân thật: matplotlib sinh id cho các thẻ `<clipPath>`/`<pattern>`
  trong SVG bằng cách băm (`sha256`) một "salt" + nội dung; salt này lấy từ
  `matplotlib.rcParams["svg.hashsalt"]`, mặc định là `None` — khi `None`,
  matplotlib tự bốc một `uuid.uuid4()` NGẪU NHIÊN làm salt cho MỖI LẦN GỌI
  `savefig`, hoàn toàn độc lập với việc có tắt `Date` metadata hay không.
  Đây là 2 cơ chế không dùng chung của matplotlib (`_write_metadata` cho
  ngày giờ, `_make_id`/`svg.hashsalt` cho id phần tử), nên sửa cái này không
  tự động sửa cái kia.
- Cách tránh / sửa: đặt `matplotlib.rcParams["svg.hashsalt"] = "<chuỗi cố
  định bất kỳ>"` (một lần, trước khi gọi `savefig`) SONG SONG với
  `metadata={"Date": None}` — thiếu 1 trong 2 thì `.svg` vẫn không
  deterministic. Đã kiểm chứng: chạy `visualize-pipeline` 3 lần liên tiếp,
  sha256 của cả `.png` lẫn `.svg` giống hệt nhau sau khi đặt cả 2.
- Liên quan tới: `src/visualize_pipeline.py::run()`.
