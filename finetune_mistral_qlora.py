from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    BitsAndBytesConfig,
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
import torch

model_name = "mistralai/Mistral-7B-Instruct-v0.2"
data_path = "viquad_alpaca.json"

# Load dataset
dataset = load_dataset("json", data_files={"train": data_path})

# Load tokenizer & model in 4bit (dùng BitsAndBytesConfig mới)
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4"
)

tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    quantization_config=bnb_config,
    device_map="auto",
    trust_remote_code=True
)

# Prepare model for QLoRA
model = prepare_model_for_kbit_training(model)

# QLoRA config
lora_config = LoraConfig(
    r=8,
    lora_alpha=16,
    target_modules=["q_proj", "v_proj", "k_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM"
)
model = get_peft_model(model, lora_config)

def preprocess(example):
    prompt = f"{example['instruction']}\n{example['input']}\nĐáp án: {example['output']}"
    return tokenizer(prompt, truncation=True, max_length=256, padding="max_length")

tokenized = dataset["train"].map(preprocess, batched=False)

training_args = TrainingArguments(
    output_dir="./mistral-viquad-qlora",
    per_device_train_batch_size=4,  # tăng batch size lên 6
    num_train_epochs=1,
    save_steps=100,
    logging_steps=10,
    learning_rate=5e-4,  # tăng learning rate
    fp16=True,
    report_to="none"
)

def data_collator(features):
    return {
        "input_ids": torch.stack([torch.tensor(f["input_ids"]) for f in features]),
        "attention_mask": torch.stack([torch.tensor(f["attention_mask"]) for f in features]),
        "labels": torch.stack([torch.tensor(f["input_ids"]) for f in features]),
    }

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized,
    data_collator=data_collator,
)

trainer.train(resume_from_checkpoint="./mistral-viquad-qlora/checkpoint-200")