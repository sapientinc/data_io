from datasets import load_dataset
import pandas as pd

from utils import write_jsonl


dataset_dict = load_dataset('facebook/principia-collection')
result = []


for split_name, dataset in dataset_dict.items():
    for row in dataset:
        result.append({
            "condition": "synth,direct",
            "instruction": row["problem_statement"],
            "response": row["answer"]
        })

write_jsonl(f"data/principia_collection.jsonl", result)
