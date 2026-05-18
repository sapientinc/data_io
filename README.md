![](./assets/banner.png)

# Data IO

This is the data pipeline used in the pretraining process of HRM-Text. Unlike LLM pretraining pipelines that ingest web documents for language modeling, HRM-Text Data IO produces instruction-style question-answer pairs and builds sampled tokenized datasets for training.

## Overview

The pipeline consists of four main stages:

1. **Data Cleaning**: Convert raw datasets into standardized instruction/response format
2. **Tokenizer Training**: Train BPE tokenizer
3. **Tokenization**: Convert text to token IDs using a Rust-based high-performance tokenizer
4. **Stratified Sampling**: Create balanced training datasets with configurable sampling strategies

### Directory Structure

```
data_io/
├── pipe/                     # Data cleaning scripts (legacy, small datasets)
├── pipe_clustered/           # Data cleaning scripts (large clustered datasets)
├── raw_data/                 # Raw, source datasets
├── data/                     # Cleaned legacy, small datasets (JSONL format)
├── data_clustered/           # Cleaned large-scale datasets (Parquet format)
├── tokenizer/                # Rust tokenizer implementation
├── trained_tokenizers/       # Trained tokenizers
├── data_tokenized_*/         # Tokenized output (numpy arrays and metadata)
├── prefix_config.yaml        # Stratified sampling configuration
└── sample_tokenized.py       # Stratified sampling & epoch creation
```

## Guidelines

Before you start, please make sure that you are in the project directory and have installed pip requirements:

```bash
cd data_io
pip install -r requirements.txt
```

Install Rust/Cargo before tokenizer training or tokenization.

> 💡 The cleaning scripts requires ~512GiB of RAM. You can download [cleaned data](https://huggingface.co/datasets/sapientinc/HRM-Text-data-io-cleaned-20260515) to skip Raw Data Preparation and Data Cleaning and go directly to Tokenization.

### Raw Data Preparation

Most cleaning scripts read source datasets from Hugging Face Hub. Some scripts read local files from `raw_data/`.

The following local raw datasets are required before running the full cleaning pipeline.

```bash
# FLAN
hf download Open-Orca/FLAN --repo-type dataset --local-dir ./raw_data/FLAN
# SYNTH
hf download PleIAs/SYNTH --repo-type dataset --local-dir ./raw_data/SYNTH
# Platypus
mkdir -p ./raw_data/Platypus
hf download imone/ARB --repo-type dataset --local-dir ./raw_data/Platypus/ARB
git clone https://github.com/mandyyyyii/scibench.git ./raw_data/Platypus/scibench
```

Download the following and unzip to `raw_data/`.

- [amps.tar.gz](https://drive.google.com/file/d/1hQsua3TkpEmcJD_UWQx8dmNdEZPyxw23/view?usp=sharing)
- [mathematics_dataset-v1.0.tar.gz](https://console.cloud.google.com/storage/browser/mathematics-dataset)

### Data Cleaning

Transform raw datasets into standardized format using the cleaning scripts. Run the needed scripts in `pipe` and `pipe_clustered`, for example:

```bash
python -m pipe.clean_platypus.clean_arb
python -m pipe.clean_gsm8k_train
python -m pipe.clean_math_train
# ... other cleaners

python -m pipe_clustered.clean_acereason
# ... other clustered cleaners
```

Cleaned data is written to `data/` and `data_clustered/`.

**Output Format:**

JSON:
```jsonc
{
  "condition": "cot,noisy",  // tags attached to this item, separated by comma
  "instruction": "Question or prompt text",
  "response": "Answer or completion text"
}
```
Parquet: Same as above, in columnar format.

### (OPTIONAL) Tokenizer training

Trained tokenizers are already in `trained_tokenizers/`. **Optional:** If you want to train a new one, run the following:

```bash
(cd tokenizer && cargo run --release --bin train_tokenizer -- ../data ../data_clustered -o ../trained_tokenizers/bpe/tokenizer.json)
```

### Tokenization

Convert text to token IDs using the high-performance Rust tokenizer:

```bash
(cd tokenizer && cargo run --release --bin tokenizer -- ../data ../data_clustered --tokenizer-path ../trained_tokenizers/bpe/tokenizer.json -o ../data_tokenized_bpe_65k)
```

It supports incremental processing. When source data changes, it will remove orphans and re-tokenize newly updated files.

**Output:** For each source `.jsonl` or `.parquet` file, creates one output subdirectory containing:
- `tokens.npy`: Concatenated token IDs
- `inst_start.npy`, `inst_len.npy`: Instruction boundaries
- `resp_start.npy`, `resp_len.npy`: Response boundaries
- `metadata.json`: For caching only (source file modification time, size)

The output root also contains `tokenizer_info.json`.

### (ON TRAINING NODES ONLY) Stratified Sampling

**On each node that is about to launch training**, create balanced training datasets from tokenized dataset with stratified sampling in memory (`/dev/shm`):

```bash
python sample_tokenized.py epochs=10 > show_analytics.md
```

Override configuration values with `key=value` arguments ([OmegaConf CLI argument format](https://omegaconf.readthedocs.io/en/2.3_branch/usage.html#id15)).

**Configuration Options:**
```python
tokenized_path: str = "data_tokenized_bpe_65k" # Input directory
output_path: str = "/dev/shm/sampled"          # Output directory (RAM disk)
prefix_config_path: str = "prefix_config.yaml" # Stratified sampling configuration

seed: int = 0                                  # Random seed
epochs: int = 10                               # Number of training epochs

context_size: int = 4096 + 1                   # Max sequence length (including +1 AR shift)
min_resp_length: int = 2                       # Minimum response length. All responses shorter than this will be dropped. Default: at least one content token + an EOS = 2 tokens
```

**Stratified sampling configuration file (specified in prefix_config_path):**

The sampler matches file prefixes in order. Once a match is found, the following rules apply:

```python
max_per_file: Optional[int] = None  # Maximum rows to sample from this file per epoch
long_context: Literal["drop", "truncate"] = "truncate"  # What to do if the context exceeds maximum
repeat: int = 1  # Repeat the dataset for X times. Used for upsampling small datasets
```

**Output:**
- `tokens.npy`: Concatenated token array (memory-mapped)
- `epoch_N/`: Per-epoch index arrays (inst_start, inst_len, resp_start, resp_len)
- `metadata.json`: Dataset statistics (vocab size, max sequence length, total tokens)

**Analytics:** The script writes Markdown statistics to stdout, usually redirected to a file:

Reports include:
- Coverage statistics by category and task
- Total unique rows and tokens sampled

## Citation

Citation information can be found in the accompanying paper.

## Contributions

We welcome contributions to scale and improve this pretraining pipeline! Because pretraining data quality directly impacts model performance, we require validation for all changes. Please align your Pull Request with one of the following categories:

### 1. Optimizations (No Result Change)

*For code refactoring, speedups, or memory footprint reductions.*

**Rule:** The final output must remain identical to the main branch.
**Validation Required:**
* Provide before/after performance metrics (execution time, peak RAM usage).
* Prove output equivalence by verifying the checksums (e.g., SHA256) of the generated `.npy` arrays.

### 2. Major Changes (Behavior Modifying)

*For modifying sampling strategies, updating the tokenizer, or adding/altering datasets.*

**Rule:** Any change that alters the token distribution, vocabulary, or sequence boundaries must be treated as a breaking change. **Validation & Benchmarking:**
* **Analytics:** Attach the complete Markdown output generated by `sample_tokenized.py` to your PR to show dataset coverage and sampled-token counts.
* **Model Evaluation:** It is strongly recommended to conduct a pretraining run at any scale and provide downstream benchmark results comparing the baseline to your proposed changes.
* **Pareto Efficiency & Merging:** We evaluate data modifications based on their position on the Pareto frontier of compute cost (training tokens) versus performance.
  * **Main Branch:** We merge highly efficient changes directly into `main`. This includes strict improvements (fewer tokens yielding better or equal performance) and high-ROI additions (a slight increase in tokens yielding a large performance jump).
  * **Alternative Branches:** Changes that push the frontier inward at lower compute but reduced performance, or outward at a high compute cost for better performance are valuable but will be merged into separate, dedicated branches rather than `main`.

### Submitting Your PR

Title your PR with a clear prefix (e.g., `[Opt]` or `[Major]`) and include the required validation proofs in the description. For other types of changes, please open an issue to discuss.

## License

Apache 2.0
