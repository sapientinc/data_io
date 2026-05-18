from glob import glob
from utils import write_jsonl
import orjson
from tqdm import tqdm


DATASET_PATH = "raw_data/amps/khan"


result = []
for filename in tqdm(glob(f"{DATASET_PATH}/**/*.json", recursive=True)):
    with open(filename, "rb") as f:
        item = orjson.loads(f.read())
        result.append({
            "condition": "noisy,cot",
            "instruction": item["problem"],
            "response": "\n".join(item["hints"]).strip()
        })


write_jsonl(f"data/amps_khan.jsonl", result)
