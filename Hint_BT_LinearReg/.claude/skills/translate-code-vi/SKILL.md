---
name: translate-code-vi
description: Dịch nội dung file .ipynb, .py, .md sang tiếng Việt, giữ nguyên 100% code, tên biến, tên hàm, cú pháp lệnh. Dùng khi người dùng yêu cầu dịch file .py/.ipynb/.md sang tiếng Việt hoặc việt hóa notebook/script.
---

# Dịch file .ipynb / .py / .md sang tiếng Việt

## Nguyên tắc chung

LUÔN dịch: markdown cells, docstring, comment, chuỗi text hiển thị cho người đọc (print, thông báo, label biểu đồ).

KHÔNG BAO GIỜ dịch: tên biến/hàm/class/module/file/cột dataframe, từ khóa Python, tên thư viện, chuỗi tham số kỹ thuật, đường dẫn, URL, code logic, số liệu, output đã lưu sẵn.

Nếu không chắc, ưu tiên giữ nguyên.

## File .py
Dịch docstring, comment, chuỗi hiển thị. Giữ nguyên biến/hàm/class, f-string trong {}, tham số kỹ thuật. Kiểm tra số dòng code không đổi sau khi dịch.

## File .ipynb
Đọc bằng nbformat/json, không sửa tay qua text editor thô. Dịch toàn bộ markdown cell. Với code cell chỉ dịch comment/docstring. Giữ nguyên outputs, execution_count, metadata. Ghi lại với ensure_ascii=False để tiếng Việt hiển thị đúng.

## File .md
Dịch heading, đoạn văn, bullet, nội dung bảng. Giữ nguyên code block, inline code, URL.

## Quy trình
1. Xác định file cần dịch.
2. Đọc đúng cách theo loại file.
3. Dịch theo nguyên tắc trên.
4. Lưu thành file mới hậu tố _vi, giữ file gốc.
5. Báo lại tên file kết quả và các chỗ không chắc chắn.
