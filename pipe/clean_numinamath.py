from datasets import load_dataset

from utils import write_jsonl


dataset = load_dataset("AI-MO/NuminaMath-1.5", split="train")
result = []


for row in dataset:
    if not row["synthetic"] and row["problem_is_valid"] == "Yes" and row["solution_is_valid"] == "Yes" and row["solution"] is not None and row["answer"] is not None:
        if "http" not in row["problem"] and "http" not in row["solution"] and "Translate the text above into English" not in row["solution"]:
            problem = row["problem"].strip()
            solution = row["solution"].strip()
            if problem and solution:
                result.append({
                    "condition": "noisy,cot",
                    "instruction": problem,
                    "response": solution
                })

                if row["question_type"] != "proof" and row["answer"] != "proof":
                    answer = row["answer"].strip()
                    if answer:
                        result.append({
                            "condition": "noisy,direct",
                            "instruction": problem,
                            "response": answer
                        })

write_jsonl(f"data/numinamath.jsonl", result)
