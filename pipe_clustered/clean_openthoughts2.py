import os
import re
from datasets import load_dataset
import pyarrow as pa
import pyarrow.parquet as pq


OUTPUT_DIR = "data_clustered/openthoughts2"
REMOVE_SOURCES = {
    'dolphin', 'evolcodegolf', 'glaive', 'magicoder', 'sharegpt', 'codefeedback',  # Remove code
    'nvidia_math'  # Already included
}

dataset = load_dataset('open-thoughts/OpenThoughts2-1M', split='train')

result = {"instruction": [], "response": [], "condition": []}


def remove_think_tags(text):
    pattern = r'<think>.*?</think>'
    return re.sub(pattern, '', text, flags=re.DOTALL)


for row in dataset:
    if row["source"] not in REMOVE_SOURCES:
        # Check there are exactly two conversations: user and assistant
        assert len(row["conversations"]) == 2
        assert row["conversations"][0]["from"] == "user"
        assert row["conversations"][1]["from"] == "assistant"

        input = row["conversations"][0]["value"]
        output = row["conversations"][1]["value"]

        # Filter out code based on simple heuristics
        output_lower = output.lower()
        if "python" not in input.lower() and "python" not in output_lower and "```" not in output_lower:
            result["condition"].append("synth,cot")
            result["instruction"].append(input)
            result["response"].append(remove_think_tags(output))

# Write
os.makedirs(OUTPUT_DIR, exist_ok=True)
pq.write_table(pa.Table.from_pydict(result), os.path.join(OUTPUT_DIR, f"all.parquet"))
