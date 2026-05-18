from datasets import load_dataset

from utils import write_jsonl


dataset = load_dataset("KbsdJames/Omni-MATH", split="test")
result = []


for row in dataset:
    result.append({
        "condition": "cot",
        "instruction": row["problem"],
        "response": row["solution"].strip()
    })
    result.append({
        "condition": "direct",
        "instruction": row["problem"],
        "response": row["answer"].strip()
    })

write_jsonl(f"data/omnimath.jsonl", result)
