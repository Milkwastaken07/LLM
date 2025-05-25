import json
from datasets import load_dataset

ds = load_dataset("taidng/UIT-ViQuAD2.0")
train_data = ds['train']

alpaca_records = []
for item in train_data:
    context = item['context']
    question = item['question']
    answer = item['answers']['text'][0] if item['answers']['text'] else ""
    alpaca_records.append({
        "instruction": "Trả lời câu hỏi dựa trên đoạn văn.",
        "input": f"Đoạn văn: {context}\nCâu hỏi: {question}",
        "output": answer
    })

with open("viquad_alpaca.json", "w", encoding="utf-8") as f:
    for rec in alpaca_records:
        f.write(json.dumps(rec, ensure_ascii=False) + "\n")