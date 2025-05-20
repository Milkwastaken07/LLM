import os
import pdfplumber
import logging
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.documents import Document

# Thêm import cho OCR
import pytesseract
from pdf2image import convert_from_path
from PIL import Image

logging.getLogger("pdfminer").setLevel(logging.ERROR)

def ocr_pdf_page(pdf_path, page_number):
    """Chuyển trang PDF thành ảnh và nhận diện chữ bằng OCR."""
    images = convert_from_path(pdf_path, first_page=page_number+1, last_page=page_number+1)
    if images:
        text = pytesseract.image_to_string(images[0], lang='eng+vie')
        return text
    return ""

def load_and_split_documents(folder_path):
    documents = []
    if not os.path.exists(folder_path):
        print(f"❌ Thư mục {folder_path} không tồn tại.")
        return []

    for filename in os.listdir(folder_path):
        if filename.endswith(".pdf"):
            file_path = os.path.join(folder_path, filename)
            try:
                with pdfplumber.open(file_path) as pdf:
                    full_text = ""
                    for i, page in enumerate(pdf.pages):
                        text_layer = page.extract_text() or ""
                        ocr_text = ocr_pdf_page(file_path, i)
                        # So sánh độ dài, lấy cái nhiều ký tự hơn
                        if len(text_layer.strip()) >= len(ocr_text.strip()):
                            chosen_text = text_layer
                        else:
                            chosen_text = ocr_text
                        if chosen_text and chosen_text.strip():
                            full_text += chosen_text + "\n"
                    if full_text.strip():
                        documents.append(full_text)
            except Exception as e:
                print(f"❌ Lỗi khi đọc {filename}: {e}")

    if documents:
        # Tách từng tài liệu thành các đoạn nhỏ
        splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        splits = []
        for doc in documents:
            splits.extend(splitter.split_text(doc))
        doc_objects = [Document(page_content=chunk) for chunk in splits]
        print(f"✅ Đã load {len(documents)} file PDF")
        print(f"✅ Tách thành {len(splits)} đoạn nhỏ")
        return doc_objects
    else:
        print("⚠️ Không tìm thấy tài liệu PDF nào trong thư mục.")
        return []