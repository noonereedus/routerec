from transformers import pipeline
import torch

model_id = "openai/gpt-oss-20b"

pipe = pipeline(
    "text-generation",
    model=model_id,
    dtype="auto",
    device_map="auto",
)

messages = [
    {"role": "user", "content": "Explain University of Birmingham's strength in comparison to Columbia University."},
]

outputs = pipe(
    messages,
    max_new_tokens=256,
)
print(outputs[0]["generated_text"][-1])