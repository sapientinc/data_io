import tarfile
import os
import pyarrow as pa
import pyarrow.parquet as pq
from tqdm import tqdm
from collections import defaultdict

def process_amps_archive(input_path, output_dir):
    # Store data grouped by the combined "topic_task" key
    # Structure: { "topic_task": { "condition": [], "instruction": [], "response": [] } }
    tasks_data = defaultdict(lambda: {"condition": [], "instruction": [], "response": []})

    skipped_count = 0
    total_records = 0
    
    print(f"Processing {input_path}...")
    
    try:
        with tarfile.open(input_path, "r:gz") as tar:
            for member in tqdm(tar, desc="Reading archive"):
                if not member.isfile() or not member.name.endswith('.txt'):
                    continue
                
                # Path parsing: amps/mathematica/<topic>/<task>/filename.txt
                path_parts = member.name.strip("/").split("/")
                
                # Ensure we have enough depth for negative indexing
                if len(path_parts) < 3:
                    continue
                
                # Validation: Ensure it's part of the mathematica dataset
                if len(path_parts) >= 4 and path_parts[-4] != "mathematica":
                      continue

                topic = path_parts[-3]
                subtask = path_parts[-2]
                
                # Create the specific key for grouping and filename
                task_key = f"{topic}_{subtask}"
                
                # Determine condition based on the subtask folder name
                condition = "noisy,cot" if subtask.endswith("w_steps") else "noisy,direct"
                
                f = tar.extractfile(member)
                if f is None:
                    skipped_count += 1
                    continue
                
                try:
                    content = f.read().decode('utf-8')
                except UnicodeDecodeError:
                    skipped_count += 1
                    continue

                # --- Modified Parsing Logic ---
                # 1. Check if starts with "Problem:" and remove it
                content = content.removeprefix("Problem:")

                # 2. Split by the first "Answer:"
                # maxsplit=1 ensures we only split on the first occurrence
                parts = content.split("Answer:", 1)
                
                if len(parts) < 2:
                    # "Answer:" tag missing
                    print ("Can't extract ans: ", content) # Uncomment for debug
                    skipped_count += 1
                    continue
                
                instruction_text = parts[0].strip()
                response_text = parts[1].strip()
                
                if not instruction_text or not response_text:
                    skipped_count += 1
                    continue
                # ------------------------------

                # Append to the specific topic_task dictionary
                tasks_data[task_key]["condition"].append(condition)
                tasks_data[task_key]["instruction"].append(instruction_text)
                tasks_data[task_key]["response"].append(response_text)
                total_records += 1

    except FileNotFoundError:
        print(f"Error: The file {input_path} was not found.")
        return

    print(f"Parsing complete. Found {total_records} records.")
    print(f"Writing parquet files to '{output_dir}'...")

    os.makedirs(output_dir, exist_ok=True)

    # Write one parquet file per combined topic_task
    for task_name, columns in tqdm(tasks_data.items(), desc="Writing files"):
        table = pa.Table.from_pydict(columns)
        
        # Filename: topic_task.parquet
        file_path = os.path.join(output_dir, f"{task_name}.parquet")
        
        pq.write_table(table, file_path)

    print(f"Done. Skipped files: {skipped_count}")

if __name__ == "__main__":
    INPUT_FILE = "raw_data/amps.tar.gz"
    OUTPUT_DIR = "data_clustered/ampsmathematica" 
    
    process_amps_archive(INPUT_FILE, OUTPUT_DIR)
