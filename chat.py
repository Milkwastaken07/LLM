import os
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.runnables import RunnableConfig

load_dotenv()

llm = ChatGoogleGenerativeAI(
    model="gemini-2.0-flash",
    temperature=0.2,
    google_api_key=os.getenv("GOOGLE_API_KEY")
)

def ask_question_with_retriever(query, retriever):
    context_docs = retriever.get_relevant_documents(query)
    context_text = "\n\n".join([doc.page_content for doc in context_docs])
    prompt = f"Dựa vào ngữ cảnh sau, hãy trả lời câu hỏi.\n\nNgữ cảnh:\n{context_text}\n\nCâu hỏi: {query}"
    response = llm.invoke(prompt)
    return response.content

def ask_question_direct(query):
    response = llm.invoke(query)
    return response.content
