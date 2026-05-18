from datasets import load_dataset
import pandas as pd

from utils import write_jsonl


dataset = load_dataset('facebook/natural_reasoning', split='train')
result = []


for row in dataset:
    reference_answer = row["reference_answer"].strip()
    if len(reference_answer):
        # Skip all "proofs". Rough rule-based filter.
        lower_question = row["question"].lower()
        if "prove" not in lower_question and "show that" not in lower_question:
            result.append({
                "condition": "noisy,direct",
                "instruction": row["question"],
                "response": reference_answer
            })

write_jsonl(f"data/natural_reasoning.jsonl", result)
