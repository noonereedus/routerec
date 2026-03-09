import os
import torch
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM

os.environ["HF_HOME"] = "/scratch/u5fi/byebyevlad.u5fi/hf_cache"
os.environ["HF_DATASETS_CACHE"] = "/scratch/u5fi/byebyevlad.u5fi/hf_cache"

print("Loading SynthTRIPs...")
ds = load_dataset("ashmib/SynthTRIPs")
example = ds["test"][0]
prompt = example["text"]
print("Prompt preview:\n")
print(prompt[:500])
print("\n---\n")

model_id = "openai/gpt-oss-20b"

tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    torch_dtype=torch.bfloat16,
    device_map="auto",
)

model.eval()

# ---- Tokenize ----
inputs = tokenizer(
    prompt,
    return_tensors="pt",
    truncation=True,
    max_length=4096
)

print("Token length:", inputs["input_ids"].shape[1])

inputs = {k: v.to(model.device) for k, v in inputs.items()}

# ---- Generate ----
with torch.no_grad():
    outputs = model.generate(
        **inputs,
        max_new_tokens=500,
        do_sample=False
    )

result = tokenizer.decode(outputs[0], skip_special_tokens=True)

print("\nMODEL OUTPUT:\n")
print(result)