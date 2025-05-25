from loader import load_and_split_documents
from retriever import build_retriever
from chat import ask_question_with_retriever, ask_question_direct

splits = load_and_split_documents("docs")

def chat_with_mistral_and_rag():
    retriever = build_retriever(splits)
    print("🔍 Sử dụng RAG với tài liệu.")
    while True:
        query = input("💬 Câu hỏi: ")
        if query.lower() in ["exit", "quit"]:
            break
        if query.lower() in ["normal"]:
            chat_with_mistral()
            break
        print("🔍 Đang sử dụng RAG với tài liệu.")
        answer = ask_question_with_retriever(query, retriever)
        print("🤖 Mistral + RAG:", answer)

def chat_with_mistral():
    print("💡 Sử dụng Mistral trực tiếp.")
    while True:
        query = input("💬 Câu hỏi: ")
        if query.lower() in ["exit", "quit"]:
            break
        if query.lower() in ["rag"]:
            if splits:
                chat_with_mistral_and_rag()
                break
            else:
                print("⚠️ Không có tài liệu để sử dụng RAG.")
                continue
        answer = ask_question_direct(query)
        print("🤖 Mistral:", answer)

def main():
    if splits:
        chat_with_mistral_and_rag()
    else:
        chat_with_mistral()

if __name__ == "__main__":
    main()