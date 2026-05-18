from glob import glob
import os
import random
import string

from tqdm import tqdm
import pyarrow as pa
import pyarrow.parquet as pq


def safe_filename(filename):
    # Remove or replace unsafe characters
    safe_chars = set(string.ascii_letters + string.digits + "_-. ")
    return "".join(c if c in safe_chars else "_" for c in filename)


def clean_flan(input_dir: str, included_subsets: dict[str, list[str]], output_path: str):
    os.makedirs(output_path, exist_ok=True)

    for condition, subset_names in included_subsets.items():
        for subset_name in subset_names:
            # Load this subset
            subset_data = {}

            filenames = glob(os.path.join(input_dir, subset_name, "*.parquet"))
            for filename in tqdm(filenames):
                # Load table
                table = pq.read_table(filename)
                for task_name, inputs, targets in zip(table.column("_task_name"), table.column("inputs"), table.column("targets")):
                    task_name = str(task_name)

                    subset_data.setdefault(task_name, {"instruction": [], "response": [], "condition": []})
                    subset_task_data = subset_data[task_name]
                    subset_task_data["instruction"].append(str(inputs))
                    subset_task_data["response"].append(str(targets))
                    subset_task_data["condition"].append(condition)
            
            # Write
            for task_name, task_data in subset_data.items():
                tqdm.write(f"Task {task_name}: {len(task_data['instruction'])} records")
                pq.write_table(pa.Table.from_pydict(task_data), os.path.join(output_path, f"{subset_name}__{safe_filename(task_name)}.parquet"))
            del subset_data


def main():
    clean_flan(
        input_dir="raw_data/FLAN",
        included_subsets={
            "direct": [
                # Few-shot
                "dialog_fsopt_data",
                "flan_fsopt_data", "flan_fsnoopt_data",
                "niv2_fsopt_data",
                "t0_fsopt_data", "t0_fsnoopt_data",

                # Zero-shot
                "dialog_zsopt_data",
                "flan_zsopt_data", "flan_zsnoopt_data",
                "niv2_zsopt_data",
                "t0_zsopt_data", "t0_zsnoopt_data",
            ],
            "cot": [
                "cot_fsopt_data",
                "cot_zsopt_data",
            ]
        },
        output_path=f"data_clustered/flan"
    )


if __name__ == "__main__":
    main()
