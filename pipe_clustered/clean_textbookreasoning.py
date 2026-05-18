import os

from datasets import load_dataset
import pyarrow as pa
import pyarrow.parquet as pq


OUTPUT_DIR = "data_clustered/textbookreasoning"


dataset = load_dataset('MegaScience/TextbookReasoning', split='train')

result_cot = {"instruction": [], "response": [], "condition": []}
result_direct = {"instruction": [], "response": [], "condition": []}


for row in dataset:
    # Synthetic
    result_cot["condition"].append("synth,cot")
    result_cot["instruction"].append(row["question"])
    result_cot["response"].append(row["answer"])

    # Real.
    lower_question = row["question"].lower()
    # Skip all "proofs". Rough rule-based filter.
    if "prove" not in lower_question and "show that" not in lower_question:
        result_direct["condition"].append("noisy,direct")
        result_direct["instruction"].append(row["question"])
        result_direct["response"].append(row["reference_answer"])

# Write
os.makedirs(OUTPUT_DIR, exist_ok=True)
pq.write_table(pa.Table.from_pydict(result_cot), os.path.join(OUTPUT_DIR, f"cot.parquet"))
pq.write_table(pa.Table.from_pydict(result_direct), os.path.join(OUTPUT_DIR, f"direct.parquet"))
