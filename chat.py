import os
from dotenv import load_dotenv
from llama_cpp import Llama

load_dotenv()

# Đường dẫn tới file model Mistral (ví dụ: "mistral-7b-instruct-v0.2.Q4_K_M.gguf")
MISTRAL_MODEL_PATH = os.getenv("MISTRAL_MODEL_PATH", "mistral-7b-instruct-v0.2.Q4_K_M.gguf")

llm = Llama(
    model_path=MISTRAL_MODEL_PATH,
    n_ctx=4096,
    n_threads=8,
    n_gpu_layers=20,  # Tùy GPU, có thể giảm nếu lỗi
    verbose=False
)

def ask_question_with_retriever(query, retriever):
    context_docs = retriever.get_relevant_documents(query)
    context_text = "\n\n".join([doc.page_content for doc in context_docs])
    prompt = (
        "Dựa vào ngữ cảnh sau, hãy trả lời câu hỏi một cách chi tiết, đúng trọng tâm, "
        "giải thích rõ ràng, có thể liệt kê các ý chính nếu cần.\n\n"
        f"Ngữ cảnh:\n{context_text}\n\nCâu hỏi: {query}\nTrả lời:"
    )
    output = llm(prompt, max_tokens=1024, stop=["</s>"])
    return output["choices"][0]["text"].strip()

def ask_question_direct(query):
    prompt = f"Trả lời chi tiết, đúng trọng tâm: {query}\nTrả lời:"
    output = llm(prompt, max_tokens=1024, stop=["</s>"])
    return output["choices"][0]["text"].strip()