from datasets import load_dataset, get_dataset_config_names
from utils import write_jsonl

def _last_boxed_only_string(string):
    idx = string.rfind("\\boxed")
    if idx < 0:
        idx = string.rfind("\\fbox")
        if idx < 0:
            return None

    i = idx
    left_brace_idx = None
    right_brace_idx = None
    num_left_braces_open = 0
    while i < len(string):
        if string[i] == "{":
            num_left_braces_open += 1
            if left_brace_idx is None:
                left_brace_idx = i
        elif string[i] == "}":
            num_left_braces_open -= 1
            if num_left_braces_open == 0:
                right_brace_idx = i
                break

        i += 1
    
    if left_brace_idx is None or right_brace_idx is None:
        return None

    return string[left_brace_idx + 1: right_brace_idx].strip()

dataset_name = "EleutherAI/hendrycks_math"

# 1. Dynamically get all subset names (e.g., 'algebra', 'geometry', etc.)
subsets = get_dataset_config_names(dataset_name)
print(f"Found subsets: {subsets}")

result = []

# 2. Iterate through each subset and aggregate the data
for subset in subsets:
    dataset = load_dataset(dataset_name, subset, split="train")
    
    for row in dataset:
        result.append({
            "condition": "cot",
            "instruction": row["problem"],
            "response": row["solution"].strip()
        })

        # No CoT variant
        ground_truth_answer = _last_boxed_only_string(row["solution"])
        if ground_truth_answer:
            result.append({
                "condition": "direct",
                "instruction": row["problem"],
                "response": ground_truth_answer.strip()
            })

print(f"Total records loaded: {len(result)}")
write_jsonl("data/math_train.jsonl", result)