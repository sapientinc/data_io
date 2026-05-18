from datasets import load_dataset
import pandas as pd

from utils import write_jsonl


dataset = load_dataset('TIGER-Lab/WebInstruct-verified', split='train')
result = []


for row in dataset:
    result.append({
        "condition": "direct",
        "instruction": row["question"],
        "response": row["answer"]
    })

write_jsonl(f"data/webinstruct_verified.jsonl", result)
