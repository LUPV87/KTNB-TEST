# 🛡️ Ứng dụng Phát Hiện Giao Dịch Gian Lận & Rủi Ro

Ứng dụng web được xây dựng dựa trên Streamlit và thư viện Machine Learning Scikit-Learn nhằm mục đích tự động hóa quá trình nhận diện, phân tích rủi ro và phát hiện các hành vi giao dịch gian lận tài chính (`default = 1`) từ các thuộc tính đầu vào được ẩn hóa từ `X_1` đến `X_14`.

## ✨ Các tính năng chính

- **Cấu hình động đa mô hình:** Lựa chọn linh hoạt giữa 3 thuật toán phổ biến: **Random Forest**, **Decision Tree**, và **Logistic Regression** kèm tùy chỉnh sâu các tham số như `n_estimators`, `max_iter`, `random_state` ngay trên giao diện Sidebar.
- **Trực quan hóa đồ thị tương tác:** Thống kê mô tả và biểu diễn đồ thị tần suất, phân phối lớp nhị phân đa chiều bằng thư viện Plotly.
- **Hai chế độ dự đoán thông minh:**
  - *Nhập thủ công*: Kiểm tra nhanh hồ sơ một khách hàng đơn lẻ thông qua Form điền số liệu (hỗ trợ tự điền trung vị thông minh).
  - *Xử lý tệp hàng loạt*: Tải lên file chứa danh sách nhiều giao dịch cùng lúc, hệ thống tự động chấm điểm rủi ro, tính toán xác suất phần trăm và xuất file kết quả dạng `.csv` chuẩn hóa.

## 🛠️ Hướng dẫn cài đặt và khởi chạy

**Bước 1:** Đảm bảo máy tính của bạn đã cài đặt Python (Khuyến nghị phiên bản `>= 3.9`).

**Bước 2:** Di chuyển vào thư mục chứa 3 tệp tin ứng dụng (`app.py`, `requirements.txt`, `README.md`) và thực hiện cài đặt các gói thư viện phụ thuộc bằng dòng lệnh:
```bash
pip install -r requirements.txt
