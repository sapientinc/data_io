from datasets import load_dataset
import pandas as pd

from utils import write_jsonl


dataset = load_dataset('metaeval/ScienceQA_text_only') 
df = pd.concat([pd.DataFrame(dataset[split]) for split in ("train", "validation", "test")])

# Transform the data
new_data = []
for _, row in df.iterrows():
    question = row["question"].strip()
    choices = row["choices"]
    lecture = row["lecture"].strip()
    solution = row["solution"].strip()
    answer = row["answer"]

    formatted_choices = '\n'.join([f'{chr(65+i)}: {choice.strip()}' for i, choice in enumerate(choices)])

    if solution:
        # with rationale
        if lecture:
            input = f"Solve the following question using the information provided in the lecture.\n\n{question}\nOptions:\n{formatted_choices}\n\nLecture: {lecture}"
        else:
            input = f"{question}\nOptions:\n{formatted_choices}"

        new_data.append({
            "condition": "cot",
            "instruction": input,
            "response": f"{solution}\n\nAnswer: {chr(65+answer)}"
        })

    # without rationale
    if lecture:
        input = f"Choose the correct option letter for the following question based on the information from the lecture.\n\n{question}\nOptions:\n{formatted_choices}\n\nLecture: {lecture}"
    else:
        input = f"{question}\nOptions:\n{formatted_choices}"

    new_data.append({
        "condition": "direct",
        "instruction": input,
        "response": f"{chr(65+answer)}"
    })


# Save the transformed data to a new json file
write_jsonl(f"data/Platypus/scienceqa.jsonl", new_data)
