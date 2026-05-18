import os
import re
from datasets import load_dataset
import pyarrow as pa
import pyarrow.parquet as pq


OUTPUT_DIR = "data_clustered/acereason"

dataset = load_dataset('nvidia/AceReason-1.1-SFT', split='train')

result = {"instruction": [], "response": [], "condition": []}


def remove_think_tags(text):
    pattern = r'<think>.*?</think>'
    return re.sub(pattern, '', text, flags=re.DOTALL)


for row in dataset:
    if row["category"] == "math":
        # Should be only one reasoning trace
        assert row["output"].count("<think>") == 1
        result["condition"].append("synth,cot")
        result["instruction"].append(row["input"])
        result["response"].append(remove_think_tags(row["output"]))


# Write
os.makedirs(OUTPUT_DIR, exist_ok=True)
pq.write_table(pa.Table.from_pydict(result), os.path.join(OUTPUT_DIR, f"all.parquet"))
