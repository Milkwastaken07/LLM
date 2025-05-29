import os
from dotenv import load_dotenv

load_dotenv()

MODEL_PATH = None
USE_TRANSFORMERS = False

if os.path.isdir("./mistral-7b-viquad-merged"):
    MODEL_PATH = "./mistral-7b-viquad-merged"
    USE_TRANSFORMERS = True
elif os.path.isfile("mistral-7b-instruct-v0.2.Q4_K_M.gguf"):
    MODEL_PATH = "mistral-7b-instruct-v0.2.Q4_K_M.gguf"
    USE_TRANSFORMERS = False
else:
    raise FileNotFoundError("Không tìm thấy model đã fine-tune hoặc file .gguf!")

print("MODEL_PATH:", MODEL_PATH)

if USE_TRANSFORMERS:
    from transformers import AutoModelForCausalLM, AutoTokenizer
    import torch
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(MODEL_PATH, device_map="auto", torch_dtype=torch.float16, trust_remote_code=True)
    model.eval()

    def ask_question_with_retriever(query, retriever):
        context_docs = retriever.get_relevant_documents(query)
        context_text = context_docs[0].page_content if context_docs else ""
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
else:
    from llama_cpp import Llama

    llm = Llama(model_path=MODEL_PATH, n_ctx=2048, n_threads=8, n_gpu_layers=32)

    def ask_question_with_retriever(query, retriever):
        context_docs = retriever.get_relevant_documents(query)
        context_text = context_docs[0].page_content if context_docs else ""
        prompt = (
            f"Trả lời câu hỏi dựa trên đoạn văn.input:\n{context_text}\n\nCâu hỏi: {query}\nTrả lời:"
        )
        output = llm(prompt, max_tokens=256, stop=["</s>", "Trả lời:"])
        return output["choices"][0]["text"].strip()

    def ask_question_direct(query):
        prompt = f"Trả lời chi tiết, đúng trọng tâm: {query}\nTrả lời:"
        output = llm(prompt, max_tokens=256, stop=["</s>", "Trả lời:"])
        return output["choices"][0]["text"].strip()