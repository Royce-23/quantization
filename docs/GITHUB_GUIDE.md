# Đưa bộ khung lên GitHub và bắt đầu làm nhóm

## Trưởng nhóm — repository mới

1. Giải nén ZIP. Mở thư mục có README, main.py và src trong VS Code.
2. Tạo repository trống trên tài khoản GitHub của nhóm/trưởng nhóm. Khi dùng lệnh bên dưới, không khởi tạo thêm README trên web để tránh hai lịch sử ban đầu.
3. Mở Terminal trong thư mục dự án và chạy từng dòng:

```bash
git init
git add .
git commit -m "chore: initialize quantization research starter"
git branch -M main
```

4. Thêm địa chỉ repository thật, thay OWNER và REPO trước khi chạy:

```bash
git remote add origin https://github.com/OWNER/REPO.git
git push -u origin main
```

Nếu Git yêu cầu danh tính, cấu hình tên/email của chính bạn. Thêm hai bạn còn lại làm collaborator và để họ chấp nhận lời mời.

**Đẩy các file đã giải nén, không chỉ upload file ZIP.** Dùng Git giúp giữ cả .github và .gitignore. Starter không tự tạo repo hoặc mời người.

## Nếu đã có repository

Clone repo hiện có vào một thư mục riêng, tạo nhánh cho việc thêm bộ khung, rồi sao chép file cần thiết vào đó. Kiểm tra trước khi ghi đè README, requirements hoặc .gitignore. Không chạy `git init` hay thay remote của dự án cũ một cách tùy tiện. Commit và mở PR để nhóm xem thay đổi.

## Thành viên — bắt đầu công việc

Thay URL và nhánh bằng thông tin thật. Lệnh dưới dùng nhánh mẫu cho TV1:

```bash
git clone https://github.com/OWNER/REPO.git
cd REPO
git switch -c feature/baseline
python main.py --check
```

TV2 dùng `feature/quantization`; TV3 dùng `feature/evaluation`. Viết phần được giao; xem README và CONTRACT. Sau khi sửa:

```bash
git status
git add src/prepare_data.py
git commit -m "feat: prepare HAR data"
git push -u origin feature/baseline
```

Đường dẫn ở `git add` là ví dụ của TV1; thay bằng file thực sự sửa. Mở Pull Request từ nhánh của mình vào main; một bạn khác đọc trước khi merge.

## Lấy cập nhật chung

Trước khi lấy cập nhật, commit công việc đang làm hoặc xử lý thay đổi chưa lưu. Trên nhánh công việc:

```bash
git fetch origin
git merge origin/main
```

Nếu có conflict, cùng người sở hữu file giải quyết; không force-push để bỏ qua. Tiếp tục làm trên nhánh riêng. Không đưa dữ liệu lớn, .venv hoặc checkpoint tạm vào Git.

## Checklist và Issue

Điền tên trong PHAN_CONG_CONG_VIEC.md. Tạo Issue bằng mẫu có sẵn, gán người và milestone. Nhóm tự tạo 5 milestone tương ứng 5 phase; file này không tự tạo chúng.

Trong Markdown, đổi `[ ]` thành `[x]` để cập nhật việc hoàn thành. CI xanh chỉ chứng minh bộ khung/cú pháp hợp lệ, chưa chứng minh thuật toán đúng.
