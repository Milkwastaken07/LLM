# Hướng dẫn cài đặt, train và sử dụng dự án LLM với Mistral

## 1. Cài đặt các phần mềm phụ thuộc

- **Tesseract OCR:**  
  Tải và cài đặt tại:  
  https://github.com/tesseract-ocr/tesseract?tab=readme-ov-file#installing-tesseract

- **Poppler:**  
  Tải và giải nén tại:  
  https://github.com/oschwartz10612/poppler-windows/releases/

- **Thêm đường dẫn vào biến môi trường PATH:**  
  - Thêm thư mục cài đặt `Tesseract-OCR` và thư mục `bin` của Poppler vào biến môi trường `PATH` trên Windows.

## 2. Cài đặt NVIDIA CUDA (nếu dùng GPU)

- **Driver NVIDIA:**  
  https://www.nvidia.com/Download/index.aspx

- **CUDA Toolkit (>= 11.8):**  
  https://developer.nvidia.com/cuda-downloads

- **Kiểm tra CUDA:**  
  ```python
  import torch
  print(torch.cuda.is_available())
  ```
  Nếu trả về `True` là đã cài thành công.

## 3. Cài đặt thư viện Python

- Cài đặt Python >= 3.10 (khuyến nghị dùng Anaconda hoặc Miniconda).
- Cài đặt các thư viện cần thiết:
  ```sh
  pip install -r requirement.txt
  ```

## 4. Cấu trúc thư mục dự án

```
LLM/
├── .env
├── .gitignore
├── chat.py
├── loader.py
├── main.py
├── mistral-7b-instruct-v0.2.Q4_K_M.gguf
├── README.md
├── requirement.txt
├── retriever.py
├── test.py
├── ui.py
├── __pycache__/
├── chroma_store/
│   └── chroma.sqlite3
├── docs/
│   └── my_docs.pdf
├── mistral-7b-viquad-merged/
│   ├── config.json
│   ├── model.safetensors
│   └── ...
├── mistral-viquad-qlora/
│   └── ...
├── stanford_alpaca-main/
│   └── checkpoints/
├── training/
│   ├── finetune_mistral_qlora.py
│   ├── load_viquad.py
│   └── mergeModel.py
```

## 5. Chuẩn bị dữ liệu và mô hình

- Đặt file PDF vào thư mục [`docs/`](docs/).
- Tải model Mistral `.gguf` (ví dụ: [mistral-7b-instruct-v0.2.Q4_K_M.gguf](https://huggingface.co/TheBloke/Mistral-7B-Instruct-v0.2-GGUF)) và đặt vào thư mục dự án.

## 6. Thiết lập biến môi trường

- Tạo file `.env` với nội dung:
  ```
  MISTRAL_MODEL_PATH=mistral-7b-instruct-v0.2.Q4_K_M.gguf
  ```
  Nếu đã merge model sau khi fine-tune, sửa lại thành:
  ```
  MISTRAL_MODEL_PATH=./mistral-7b-viquad-merged
  ```

## 7. Hướng dẫn train (fine-tune) model

1. **Chuẩn bị dữ liệu:**  
   - Chạy script chuyển dữ liệu ViQuAD sang định dạng Alpaca:
     ```sh
     python training/load_viquad.py
     ```
   - File `viquad_alpaca.json` sẽ được tạo.

2. **Fine-tune với QLoRA:**  
   - Thay đổi dữ liệu fine-tune bằng cách sửa dòng:
     ```python
     data_path = "viquad_alpaca.json"
     ```
   - Chạy script fine-tune:
     ```sh
     python training/finetune_mistral_qlora.py
     ```
   - Checkpoint sẽ lưu ở thư mục `mistral-viquad-qlora/`.

4. **Merge adapter vào model gốc:**  
   - Sau khi fine-tune xong, merge adapter LoRA vào model gốc:
     ```sh
     python training/mergeModel.py
     ```
   - Model đã merge sẽ nằm ở `mistral-7b-viquad-merged/`.

## 8. Chạy dự án

- **Chạy giao diện dòng lệnh:**
  ```sh
  python main.py
  ```

- **Chạy giao diện đồ họa:**
  ```sh
  python ui.py
  ```

## 9. Ghi chú

- Nếu chỉ muốn inference nhanh, chỉ cần file `.gguf` và cài `llama-cpp-python`.
- Nếu muốn sử dụng model đã fine-tune, đảm bảo biến môi trường trỏ đúng thư mục model đã merge.
- Không push file model lớn, checkpoint lên GitHub (đã có trong `.gitignore`).

---
