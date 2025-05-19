from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma

def build_retriever(doc_objects):
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vectorstore = Chroma.from_documents(doc_objects, embedding=embeddings, persist_directory="chroma_store")
    retriever = vectorstore.as_retriever()
    return retriever
