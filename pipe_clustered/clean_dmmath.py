import os
import string
from glob import glob
from tqdm import tqdm

import csv
import pyarrow as pa
import pyarrow.parquet as pq


DATASET_DIR = "raw_data/mathematics_dataset-v1.0"
OUTPUT_DIR = "data_clustered/dmmath"
SUBSETS = ["train-easy", "train-medium", "train-hard"]


def safe_filename(filename):
    # Remove or replace unsafe characters
    safe_chars = set(string.ascii_letters + string.digits + "_-. ")
    return "".join(c if c in safe_chars else "_" for c in filename)


def main():
    for set_name in SUBSETS:
        filenames = glob(os.path.join(DATASET_DIR, set_name, "*.txt"))

        # Train data
        for filename in tqdm(filenames):
            result = {"instruction": [], "response": [], "condition": []}
            task_name = safe_filename(f"{set_name}__{os.path.basename(filename).removesuffix('.txt')}")

            # Decode
            with open(filename, "r") as f:
                lines = [line.strip() for line in f.readlines()]
                assert len(lines) % 2 == 0

                x_list, y_list = lines[::2], lines[1::2]

            for x, y in zip(x_list, y_list):
                result["instruction"].append(x)
                result["response"].append(y)
                result["condition"].append("direct")

            # Write
            os.makedirs(OUTPUT_DIR, exist_ok=True)
            pq.write_table(
                pa.Table.from_pydict(result),
                os.path.join(OUTPUT_DIR, f"{task_name}.parquet")
            )


if __name__ == "__main__":
    main()
