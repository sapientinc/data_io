from datasets import load_dataset
from utils import write_jsonl

dataset = load_dataset("HuggingFaceH4/no_robots")
result = []

for split in dataset.values():  # pyright: ignore[reportAttributeAccessIssue]
    for row in split:
        msgs = row["messages"]
        if len(msgs) < 2:
            continue

        # Check for system prompt at index 0
        system_content = ""
        start_idx = 0
        
        if msgs[0]["role"] == "system":
            system_content = msgs[0]["content"] + "\n\n"
            start_idx = 1

        # Ensure valid User -> Assistant structure for the first turn
        if (len(msgs) > start_idx + 1 and 
            msgs[start_idx]["role"] == "user" and 
            msgs[start_idx + 1]["role"] == "assistant"):

            result.append({
                "condition": "cot",
                "instruction": system_content + msgs[start_idx]["content"],
                "response": msgs[start_idx + 1]["content"]
            })

write_jsonl("data/no_robots.jsonl", result)
