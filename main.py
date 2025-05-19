from loader import load_and_split_documents
from retriever import build_retriever
from chat import ask_question_with_retriever, ask_question_direct

splits = load_and_split_documents("docs")
def chat_with_gemini_and_rag():
    retriever = build_retriever(splits)
    print("🔍 Sử dụng RAG với tài liệu.")
    while True:
        query = input("💬 Câu hỏi: ")
        if query.lower() in ["exit", "quit"]:
            break
        if query.lower() in ["normal"]:
            chat_with_gemini()
            break
        print("🔍 Đang sử dụng RAG với tài liệu.")
        answer = ask_question_with_retriever(query, retriever)
        print("🤖 Gemini + RAG:", answer)

def chat_with_gemini():
    print("💡 Sử dụng Gemini trực tiếp.")
    while True:
        query = input("💬 Câu hỏi: ")
        if query.lower() in ["exit", "quit"]:
            break
        if query.lower() in ["rag"]:
            if splits:
                chat_with_gemini_and_rag()
                break
            else:
                print("⚠️ Không có tài liệu để sử dụng RAG.")
                continue
        answer = ask_question_direct(query)
        print("🤖 Gemini:", answer)

def main():
    if splits:
        chat_with_gemini_and_rag()
    else:
        chat_with_gemini()

if __name__ == "__main__":
    main()
