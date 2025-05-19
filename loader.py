import os
import pdfplumber
import logging
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.documents import Document

logging.getLogger("pdfminer").setLevel(logging.ERROR)

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
                    for page in pdf.pages:
                        text = page.extract_text()
                        if text:
                            full_text += text + "\n"
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