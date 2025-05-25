# Hướng dẫn cài đặt và chạy dự án LLM với Mistral

## 1. Cài đặt các phần mềm phụ thuộc

- **Tesseract OCR:**  
  Tải và cài đặt tại:  
  https://github.com/tesseract-ocr/tesseract?tab=readme-ov-file#installing-tesseract

- **Poppler:**  
  Tải và giải nén tại:  
  https://github.com/oschwartz10612/poppler-windows/releases/

- **Thêm đường dẫn vào biến môi trường PATH:**  
  - Thêm thư mục cài đặt `Tesseract-OCR` và thư mục `bin` của Poppler vào biến môi trường `PATH` trên Windows.

## 1.1. Cài đặt NVIDIA CUDA (nếu dùng GPU)

- **Driver NVIDIA:**  
  Tải và cài đặt driver mới nhất cho card đồ họa tại:  
  https://www.nvidia.com/Download/index.aspx

- **CUDA Toolkit:**  
  Tải và cài đặt CUDA Toolkit (nên dùng bản >= 11.8):  
  https://developer.nvidia.com/cuda-downloads

- **Kiểm tra CUDA:**  
  Sau khi cài đặt, kiểm tra bằng lệnh Python:
  ```python
  import torch
  print(torch.cuda.is_available())
  ```
  Nếu trả về `True` là đã cài thành công.

---

**Lưu ý:**  
- Nếu chỉ chạy trên CPU, bạn có thể bỏ qua bước này.
- Nếu dùng Google Colab hoặc server cloud, các bước này thường đã được cài sẵn.
## 2. Cài đặt thư viện Python

- Cài đặt Python >= 3.10 (khuyến nghị dùng Anaconda hoặc Miniconda).
- Cài đặt các thư viện cần thiết:
  ```sh
  pip install -r requirement.txt
  ```

## 3. Chuẩn bị dữ liệu và mô hình

- Đặt file PDF vào thư mục [`docs/`](docs/).
- Tải model Mistral `.gguf` (ví dụ: [mistral-7b-instruct-v0.2.Q4_K_M.gguf](https://huggingface.co/TheBloke/Mistral-7B-Instruct-v0.2-GGUF)) và đặt vào thư mục dự án.

## 4. Thiết lập biến môi trường

- Tạo file `.env` với nội dung:
  ```
  MISTRAL_MODEL_PATH=mistral-7b-instruct-v0.2.Q4_K_M.gguf
  ```

## 5. Chạy dự án

- **Chạy giao diện dòng lệnh:**
  ```sh
  python main.py
  ```

- **Chạy giao diện đồ họa:**
  ```sh
  python ui.py
  ```

## 6. Ghi chú

- Nếu muốn fine-tune model, xem file [`finetune_mistral_qlora.py`](finetune_mistral_qlora.py).
- Đảm bảo đã cài đặt đầy đủ driver GPU (nếu sử dụng GPU để train/fine-tune).
- Thư mục checkpoint và file model lớn đã được thêm vào `.gitignore`, không cần push lên GitHub.

---