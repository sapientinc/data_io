from dataclasses import dataclass, fields
from typing import Literal, Optional, Sequence, Any
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
import yaml

from tqdm import tqdm
import numpy as np
import pydantic
from omegaconf import OmegaConf


LongContextMode = Literal["drop", "truncate"]


class PrefixConfig(pydantic.BaseModel):
    max_per_file: Optional[int] = None
    long_context: LongContextMode = "truncate"
    repeat: int = 1


class Config(pydantic.BaseModel):
    tokenized_path: str = "data_tokenized_bpe_65k"
    output_path: str = "/dev/shm/sampled"
    prefix_config_path: str = "prefix_config.yaml"

    seed: int = 0
    epochs: int = 10

    context_size: int = 4096 + 1  # +1: Account for AR shift
    min_resp_length: int = 2  # at least one content token + a EOS = 2 tokens


@dataclass
class TaskIndices:
    inst_start: np.ndarray
    inst_len: np.ndarray
    resp_start: np.ndarray
    resp_len: np.ndarray


@dataclass
class Task:
    name: str
    indices: TaskIndices

    # Prefix
    prefix_config: PrefixConfig

    # Mmap base addr
    mmap_base_offset: int = 0
    mmap_length: int = 0

    # Permutation
    coverage: Optional[np.ndarray] = None
    perm: Optional[np.ndarray] = None
    perm_cursor: int = 0


class V1DatasetMeta(pydantic.BaseModel):
    tokenizer_info: dict[str, Any] = {}
    vocab_size: Optional[int] = None
    max_seq_len: int
    total_length: int


def truncate_and_filter(task: Task, config: Config):
    # Filter too short / empty responses
    keep_mask = task.indices.resp_len >= config.min_resp_length
    # Handle long context
    allowed_resp = config.context_size - np.minimum(task.indices.inst_len, config.context_size)  # avoid unsigned overflow, min 0
    if task.prefix_config.long_context == "truncate":
        # Response must be non-empty
        keep_mask &= allowed_resp >= 1
        # Truncate: Cap response length to fit context
        task.indices.resp_len = np.minimum(task.indices.resp_len, allowed_resp)
    else:
        # Drop mode: Keep only if resp not too short, and total length fits
        keep_mask &= task.indices.resp_len <= allowed_resp

    for f in fields(TaskIndices):
        setattr(task.indices, f.name, getattr(task.indices, f.name)[keep_mask])


def concat_tokens(tasks: list[Task], config: Config, tokenizer_info: dict[str, Any], num_workers: int = 32) -> None:
    def _copy_thread(task: Task) -> None:
        mmap_src = np.load(Path(config.tokenized_path) / task.name / "tokens.npy", mmap_mode="r")
        mmap_array[task.mmap_base_offset: task.mmap_base_offset + task.mmap_length] = mmap_src

    # Precompute offset
    total_tokens = 0
    for task in tasks:
        task_len = int(np.sum(task.indices.inst_len) + np.sum(task.indices.resp_len))
        task.mmap_base_offset = total_tokens
        task.mmap_length = task_len
        total_tokens += task_len

    # Create a big mmap of concatenated tokens with dynamic dtype
    target_dtype = np.int32  # Defaults to int32

    vocab_size = tokenizer_info.get("vocab_size")
    if vocab_size is not None:
        if vocab_size <= np.iinfo(np.uint8).max:
            target_dtype = np.uint8
        elif vocab_size <= np.iinfo(np.uint16).max:
            target_dtype = np.uint16

    Path(config.output_path).mkdir(parents=True, exist_ok=True)
    mmap_array = np.lib.format.open_memmap(Path(config.output_path) / "tokens.npy", mode="w+", dtype=target_dtype, shape=(total_tokens, ))

    # Read and copy tokens (threaded)
    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        futures = [executor.submit(_copy_thread, task) for task in tasks]
        # Monitor progress
        for f in tqdm(as_completed(futures), total=len(futures), desc="Writing tokens"):
            f.result()

    # --- Step 4: Cleanup ---
    mmap_array.flush()


def gen_report(tasks: list[Task], config: Config, pct_levels: Sequence[int] = [0, 1, 5, 25, 50, 75, 95, 99, 100]):
    @dataclass
    class Stats:
        rows: int = 0
        toks: int = 0
        inst_toks: int = 0
        resp_toks: int = 0
        
        cov_rows: int = 0
        cov_toks: int = 0
        cov_inst: int = 0
        cov_resp: int = 0
        cov_rows_truncated: int = 0
        
        # New additions for unique sampling
        unique_cov_rows: int = 0
        unique_cov_toks: int = 0
        
        # # Lists for histograms (store only for covered items)
        # h_inst: list[np.ndarray] = field(default_factory=list)
        # h_resp: list[np.ndarray] = field(default_factory=list)
        # h_total: list[np.ndarray] = field(default_factory=list)

    # Aggregation
    cat_stats = {}
    task_stats = {}
    for task in tqdm(tasks, desc="Generating report"):
        s = Stats()

        # 1. Context length
        inst = task.indices.inst_len
        resp = task.indices.resp_len

        s.rows = len(inst)
        s.inst_toks = int(np.sum(inst))
        s.resp_toks = int(np.sum(resp))
        s.toks = s.inst_toks + s.resp_toks
        
        # 2. Coverage logic
        if task.coverage is not None:
            s.cov_rows = int(np.sum(task.coverage))
            s.cov_inst = int(np.sum(inst * task.coverage))
            s.cov_resp = int(np.sum(resp * task.coverage))
            s.cov_toks = s.cov_inst + s.cov_resp
            
            # Unique coverage logic
            covered_mask = task.coverage > 0
            s.unique_cov_rows = int(np.sum(covered_mask))
            s.unique_cov_toks = int(np.sum((inst + resp)[covered_mask]))

        task_stats[task.name] = s

        # Add to Category
        for cat_name in ("GLOBAL", task.name.split("__")[0]):
            cat_stats.setdefault(cat_name, Stats())
            cat_s = cat_stats[cat_name]
            for f in fields(Stats):
                val = getattr(s, f.name)
                if f.name.startswith("h_"):
                    getattr(cat_s, f.name).extend(val)
                else:
                    setattr(cat_s, f.name, getattr(cat_s, f.name) + val)

    # # Compute percentiles
    # for stats_dict in [cat_stats, task_stats]:
    #     for s in stats_dict.values():
    #         for f in fields(Stats):
    #             if f.name.startswith("h_"):
    #                 pct_values = np.percentile(np.concatenate(getattr(s, f.name)), pct_levels)
    #                 setattr(s, f.name, pct_values)
    
    # Output Generators
    def fmt_num(n): return f"{n:,}"
    def fmt_num_pct(n, total): return f"{fmt_num(n)} ({n/max(1, total):.1%})"

    def print_coverage_table(data: dict[str, Stats], ref: Stats, title: str):
        print(f"\n### {title} Coverage Stats")
        header = f"| Name | Rows (%) | Tokens (%) | Cov Rows (%) | Cov Toks (%) | Cov IToks (%) | Cov RToks (%) |"
        print(header)
        print("".join(char if char == "|" else "-" for char in header))
        
        for name, s in sorted(data.items()):
            print(f"| **{name}** | {fmt_num_pct(s.rows, ref.rows)} | {fmt_num_pct(s.toks, ref.toks)} | {fmt_num_pct(s.cov_rows, ref.cov_rows)} | {fmt_num_pct(s.cov_toks, ref.cov_toks)} | {fmt_num_pct(s.cov_inst, ref.cov_inst)} | {fmt_num_pct(s.cov_resp, ref.cov_resp)} |")

    # def print_dist_table(data: dict[str, Stats], title: str):
    #     print(f"\n### {title} Context Distribution (Covered Only)")
    #     # Columns: Name, Dropped, Truncated, TotalLen P50, Inst P50, Resp P50
    #     header = f"| Name | Trunc (%) |"
    #     for typ in ("T", "I", "R"):
    #         for pct in pct_levels:
    #             header += f" {typ}{pct} |"
    #     print(header)
    #     print("".join(char if char == "|" else "-" for char in header))

    #     for name, s in sorted(data.items()):
    #         row = f"| {name} | {fmt_num_pct(s.cov_rows_truncated, s.cov_rows)} | "
    #         for arr in [s.h_total, s.h_inst, s.h_resp]:
    #             for val in arr:
    #                 row += f" {fmt_num(int(val))} |"
    #         print(row)

    # Executing Reports
    print_coverage_table(cat_stats, cat_stats["GLOBAL"], "Category")
    print_coverage_table(task_stats, cat_stats["GLOBAL"], "Task")

    # Global Uniques Summary
    print("\n### Global Summary")
    global_s = cat_stats["GLOBAL"]
    print(f"**Total Unique Tokens Sampled:** {fmt_num(global_s.unique_cov_toks)} out of {fmt_num(global_s.toks)} ({global_s.unique_cov_toks / max(1, global_s.toks):.2%})")
    print(f"**Total Unique Rows Sampled:** {fmt_num(global_s.unique_cov_rows)} out of {fmt_num(global_s.rows)} ({global_s.unique_cov_rows / max(1, global_s.rows):.2%})\n")


def main():
    config = Config(**OmegaConf.to_container(OmegaConf.from_cli(), resolve=True))  # pyright: ignore[reportCallIssue]

    # Load tokenizer info
    with open(Path(config.tokenized_path) / "tokenizer_info.json", "r") as f:
        tokenizer_info = json.load(f)
    # Load prefix config
    with open(config.prefix_config_path, 'r') as f:
        # Load raw yaml data
        prefix_config_list = yaml.safe_load(f)

    # Scan tokenized_path/<DATASET_NAME>.
    # Read tokenized_path/<DATASET_NAME>/inst_start.npy, inst_len.npy, resp_start.npy, resp_len.npy
    # [start: start+len] are indexes of each document to tokens.npy (a large numpy file of concatenated token ids)
    tasks: list[Task] = []
    for dataset_dir in tqdm(sorted(Path(config.tokenized_path).iterdir()), desc="Reading indices"):
        task_name = dataset_dir.name
        if not dataset_dir.is_dir():
            continue

        # Match prefixes
        prefix_config = None
        for prefix_config_item in prefix_config_list:
            if task_name.startswith(prefix_config_item["prefix"]):
                if prefix_config is not None:
                    print(f"Warning: Multiple possible prefixes for task {task_name}, pick config {prefix_config.model_dump_json()}")
                    break
                prefix_config = PrefixConfig(**prefix_config_item)

        if prefix_config is None:
            prefix_config = PrefixConfig()

        # Add task
        tasks.append(Task(name=task_name,
                          indices=TaskIndices(**{f.name: np.load(dataset_dir / f"{f.name}.npy")
                                              for f in fields(TaskIndices)}),
                          prefix_config=prefix_config))

    # Concatenate all tokens into a single large file
    concat_tokens(tasks, config, tokenizer_info)

    # Prefilter
    for task in tasks:
        truncate_and_filter(task, config)

    # Generate epoch indices
    rng = np.random.Generator(np.random.Philox(seed=config.seed))

    total_rows = sum(len(task.indices.inst_start) for task in tasks)
    buffer = TaskIndices(**{f.name: np.empty((total_rows, ), dtype=np.int64)
                            for f in fields(TaskIndices)})

    total_tokens = 0
    for epoch in tqdm(range(config.epochs), desc="Generating epoch indices"):
        # Generate one epoch
        buffer_cursor = 0
        for task in tasks:
            assert task.prefix_config is not None
            rows_to_sample = min(task.prefix_config.max_per_file, len(task.indices.inst_start)) if task.prefix_config.max_per_file is not None else len(task.indices.inst_start)
            rows_to_sample *= task.prefix_config.repeat
            rows_fetched = 0

            # stats
            if task.coverage is None:
                task.coverage = np.zeros((len(task.indices.inst_start), ), dtype=np.int32)

            while rows_fetched < rows_to_sample:
                # Reset permutation if exhausted
                if task.perm is None or task.perm_cursor >= len(task.perm):
                    task.perm = rng.permutation(len(task.indices.inst_len))
                    task.perm_cursor = 0

                remaining = len(task.perm) - task.perm_cursor
                # If we need more than we have left in current perm, take all remaining
                take = min(remaining, rows_to_sample - rows_fetched)
                rows_fetched += take

                # Slice indices
                batch_indices = task.perm[task.perm_cursor: task.perm_cursor + take]
                task.perm_cursor += take

                # Fill buffer
                buffer.inst_start[buffer_cursor: buffer_cursor + take] = task.indices.inst_start[batch_indices] + task.mmap_base_offset
                buffer.inst_len[buffer_cursor: buffer_cursor + take] = task.indices.inst_len[batch_indices]
                buffer.resp_start[buffer_cursor: buffer_cursor + take] = task.indices.resp_start[batch_indices] + task.mmap_base_offset
                buffer.resp_len[buffer_cursor: buffer_cursor + take] = task.indices.resp_len[batch_indices]
                buffer_cursor += take

                # Stats
                task.coverage[batch_indices] += 1

        # Stats for metadata
        total_tokens += np.sum(buffer.inst_len[:buffer_cursor]) + np.sum(buffer.resp_len[:buffer_cursor])

        # Random permutation
        perm = rng.permutation(buffer_cursor)
        # Write epoch to disk
        epoch_dir = Path(config.output_path) / f"epoch_{epoch}"
        epoch_dir.mkdir(parents=True, exist_ok=True)
        for f in fields(TaskIndices):
            np.save(epoch_dir / f"{f.name}.npy", getattr(buffer, f.name)[perm])

    # Save metadata
    with open(Path(config.output_path) / "metadata.json", "w") as f:
        f.write(V1DatasetMeta(
            tokenizer_info=tokenizer_info,
            max_seq_len=config.context_size,
            total_length=int(round(total_tokens / config.epochs))
        ).model_dump_json())

    # Generate report
    gen_report(tasks, config)


if __name__ == "__main__":
    main()
