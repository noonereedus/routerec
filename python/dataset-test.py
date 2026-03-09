import os
from datasets import load_dataset

os.environ["HF_HOME"] = "/scratch/u5fi/byebyevlad.u5fi/hf_cache"
os.environ["HF_DATASETS_CACHE"] = "/scratch/u5fi/byebyevlad.u5fi/hf_cache"

print("Loading SynthTRIPs...")

ds = load_dataset("ashmib/SynthTRIPs")

print(ds["test"].column_names)
print(ds["test"][0])
