import os
import json
import os
from utils import write_jsonl


# The directory where the json files are stored
# dir_path = 'original'
target_path = "raw_data/Platypus/scibench/dataset/original"


json_files = [f for f in os.listdir(target_path) if f.endswith('.json')]


new_data = []

# Iterate over all the files
for json_file in json_files:
    file_path = os.path.join(target_path, json_file)

    # Open each json file
    with open(file_path, 'r') as f:
        # Load the data
        file_data = json.load(f)
        
        # Transform the data
        for d in file_data:
            problem = d.get('problem_text', '').strip()
            solution = d.get('solution', '').strip()
            answer_latex = d.get('answer_latex', '').strip()
            answer_number = d.get('answer_number', '').strip()
            if answer_latex == f"${answer_number}$":
                answer_latex = answer_number

            if solution:
                new_data.append({
                    "condition": "cot",
                    "instruction": problem,
                    "response": solution
                })

            new_data.append({
                "condition": "direct",
                "instruction": problem,
                "response": answer_latex
            })

# Save the transformed data to a new json file
write_jsonl(f"data/Platypus/scibench.jsonl", new_data)
