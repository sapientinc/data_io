import os

import csv
import pyarrow as pa
import pyarrow.parquet as pq
from huggingface_hub import hf_hub_download


DATASET = "sapientinc/sudoku-extreme"
OUTPUT_DIR = "data_clustered/sudoku_extreme"


result = {"instruction": [], "response": [], "condition": []}

with open(hf_hub_download(DATASET, f"train.csv", repo_type="dataset"), newline="") as csvfile:
    reader = csv.reader(csvfile)
    next(reader)  # Skip header
    for source, q, a, rating in reader:
        assert len(q) == 81 and len(a) == 81

        result["instruction"].append(f"Solve the Sudoku\n\n{q.replace('.', '0')}")
        result["response"].append(f"{a}")
        result["condition"].append("direct")

# Write
os.makedirs(OUTPUT_DIR, exist_ok=True)
pq.write_table(
    pa.Table.from_pydict(result),
    os.path.join(OUTPUT_DIR, f"all.parquet")
)
