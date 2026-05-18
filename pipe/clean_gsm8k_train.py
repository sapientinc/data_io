from datasets import load_dataset
from utils import write_jsonl
import re

dataset_name = "openai/gsm8k"

dataset = load_dataset(dataset_name, "main", split="train")

result = []
for row in dataset:
    answer = row["answer"].split("#### ")
    assert len(answer) == 2

    result.append({
        "condition": "direct",
        "instruction": row["question"].strip(),
        "response": answer[-1].strip()
    })

print(f"Total records loaded: {len(result)}")
write_jsonl("data/gsm8k_train.jsonl", result)
