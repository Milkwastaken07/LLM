from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import os

def load_and_split_documents(folder_path):
    documents = []
    if not os.path.exists(folder_path):
        print(f"❌ Thư mục {folder_path} không tồn tại.")
        return []

    for filename in os.listdir(folder_path):
        if filename.endswith(".pdf"):
            loader = PyPDFLoader(os.path.join(folder_path, filename))
            documents.extend(loader.load())

    if documents:
        splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        splits = splitter.split_documents(documents)
        print(f"✅ Đã load {len(documents)} trang từ PDF")
        print(f"✅ Tách thành {len(splits)} đoạn nhỏ")
        return splits
    else:
        print("⚠️ Không tìm thấy tài liệu PDF nào trong thư mục.")
        return []

