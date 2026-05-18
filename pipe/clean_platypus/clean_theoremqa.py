import pandas as pd
from datasets import load_dataset
from utils import write_jsonl


dataset = load_dataset("TIGER-Lab/TheoremQA")
new_data = []

# Open the csv file
df = pd.DataFrame(dataset['test'])

# Transform the data
for _, row in df.iterrows():
    if row['Picture'] is None:
        instruction = row['Question']
        transformed_data = {
            "condition": "direct",
            "instruction": instruction,
            "response": row['Answer']
        }
        new_data.append(transformed_data)


# Save the transformed data to a new json file
write_jsonl(f"data/Platypus/theoremqa.jsonl", new_data)
