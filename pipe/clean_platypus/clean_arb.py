import os
import orjson

from utils import write_jsonl


DATA_PATH = "raw_data/Platypus/ARB"
DESCRIPTION_MAP = {
    "math.json": ("Solve the math problem.", "cot"),
    "reading.json": ("Solve the reading comprehension problem.", "cot"),
    "law.json": ("Choose the correct option letter.", "direct"),
    "science.json": ("Solve the science problem.", "cot"),
    "physics.json": ("Solve the physics problem.", "cot")
}


for filename, (description, condition) in DESCRIPTION_MAP.items():
    with open(os.path.join(DATA_PATH, filename), "rb") as f:
        data = orjson.loads(f.read())

    data = [{"instruction": f"{description}\n\n{x['instruction']}", "response": x["response"], "condition": condition} for x in data]

    write_jsonl(f"data/Platypus/arb_{filename}l", data)
