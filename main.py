from loader import load_and_split_documents
from retriever import build_retriever
from chat import ask_question_with_retriever, ask_question_direct

def main():
    # Load và tách tài liệu PDF
    splits = load_and_split_documents("docs")

    if splits:
        retriever = build_retriever(splits)
        print("🔍 Đang sử dụng RAG với tài liệu.")
        while True:
            query = input("💬 Câu hỏi: ")
            if query.lower() in ["exit", "quit"]:
                break
            answer = ask_question_with_retriever(query, retriever)
            print("🤖 Gemini + RAG:", answer)
    else:
        print("💡 Không có tài liệu. Sử dụng Gemini trực tiếp.")
        while True:
            query = input("💬 Câu hỏi: ")
            if query.lower() in ["exit", "quit"]:
                break
            answer = ask_question_direct(query)
            print("🤖 Gemini:", answer)

if __name__ == "__main__":
    main()
