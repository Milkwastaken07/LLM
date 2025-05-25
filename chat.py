import os
from dotenv import load_dotenv
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

load_dotenv()

MODEL_PATH = None
if os.path.isdir("./mistral-7b-viquad-merged"):
    MODEL_PATH = "./mistral-7b-viquad-merged"
elif os.path.isfile("mistral-7b-instruct-v0.2.Q4_K_M.gguf"):
    MODEL_PATH = "mistral-7b-instruct-v0.2.Q4_K_M.gguf"
else:
    raise FileNotFoundError("Không tìm thấy model đã fine-tune hoặc file .gguf!")

print("MODEL_PATH:", MODEL_PATH)
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(MODEL_PATH, device_map="auto", torch_dtype=torch.float16, trust_remote_code=True)
model.eval()

def ask_question_with_retriever(query, retriever):
    context_docs = retriever.get_relevant_documents(query)
    context_text = "\n\n".join([doc.page_content for doc in context_docs])
    prompt = (
        f"Trả lời câu hỏi dựa trên đoạn văn.input:\n{context_text}\n\nCâu hỏi: {query}\nTrả lời:"
    )
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        outputs = model.generate(**inputs, max_new_tokens=1024, do_sample=True)
    return tokenizer.decode(outputs[0], skip_special_tokens=True).split("Trả lời:")[-1].strip()

def ask_question_direct(query):
    prompt = f"Trả lời chi tiết, đúng trọng tâm: {query}\nTrả lời:"
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        outputs = model.generate(**inputs, max_new_tokens=1024, do_sample=True)
    return tokenizer.decode(outputs[0], skip_special_tokens=True).split("Trả lời:")[-1].strip()