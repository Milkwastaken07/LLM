from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel, PeftConfig
import torch

# Đường dẫn model gốc và adapter
base_model_path = "mistralai/Mistral-7B-Instruct-v0.2"  # hoặc đường dẫn local .gguf nếu đã convert sang transformers
adapter_path = "./mistral-viquad-qlora/checkpoint-7114"  # thay CHECKPOINT_ID bằng checkpoint lớn nhất, ví dụ 7114
output_dir = "./mistral-7b-viquad-merged"

# Nếu máy yếu, có thể dùng quantization_config như khi train
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4"
)

# Load model gốc và adapter
model = AutoModelForCausalLM.from_pretrained(
    base_model_path,
    device_map="auto",
    trust_remote_code=True,
    quantization_config=bnb_config
)
model = PeftModel.from_pretrained(model, adapter_path)
model = model.merge_and_unload()  # merge LoRA vào model gốc

# Lưu model đã merge
model.save_pretrained(output_dir)
tokenizer = AutoTokenizer.from_pretrained(base_model_path, trust_remote_code=True)
tokenizer.save_pretrained(output_dir)

print(f"Đã merge và lưu model vào {output_dir}")