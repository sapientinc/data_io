import os
from datasets import load_dataset
import pyarrow as pa
import pyarrow.parquet as pq


OUTPUT_DIR = "data_clustered/openmathinstruct2"
ORIGINAL_SOURCES = {"math", "gsm8k"}

dataset = load_dataset('nvidia/OpenMathInstruct-2', split='train')

result_cot = {"instruction": [], "response": [], "condition": []}
result_direct = {"instruction": [], "response": [], "condition": []}


for row in dataset:
    result_cot["condition"].append("synth,cot")
    result_cot["instruction"].append(row["problem"])
    result_cot["response"].append(row["generated_solution"])

    if row["problem_source"] not in ORIGINAL_SOURCES:
        result_direct["condition"].append("synth,direct")
        result_direct["instruction"].append(row["problem"])
        result_direct["response"].append(row["expected_answer"])


# Write
os.makedirs(OUTPUT_DIR, exist_ok=True)
pq.write_table(pa.Table.from_pydict(result_cot), os.path.join(OUTPUT_DIR, f"cot.parquet"))
pq.write_table(pa.Table.from_pydict(result_direct), os.path.join(OUTPUT_DIR, f"direct.parquet"))
