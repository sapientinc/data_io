import polars as pl
from pathlib import Path
from tqdm import tqdm

# Setup paths
input_dir = Path("./raw_data/SYNTH")
output_dir = Path("./data_clustered/SYNTH")
output_dir.mkdir(parents=True, exist_ok=True)

def process_files():
    # Use glob to find all parquet files
    for file_path in tqdm(input_dir.glob("*.parquet")):
        # Lazy processing for memory efficiency and parallel execution
        df = pl.scan_parquet(file_path)

        # 1. Filter
        df = df.filter(
            (pl.col("language") == "en") &
            (~pl.col("query_seed_url").str.contains("Pleias self-knowledge")) &
            (pl.col("exercise") != "cooking")
        )

        # 2. Transform & Select
        df = df.select([
            # instruction logic
            pl.when(pl.col("exercise") == "rag")
            .then(pl.col("query") + pl.col("constraints"))
            .otherwise(pl.col("query"))
            .alias("instruction"),

            # condition logic
            pl.when(pl.col("exercise").is_in({'creative writing', 'rag', 'memorization', 'constrained writing', 'editing'})).then(pl.lit("synth,cot"))
            .when(pl.col("exercise").is_in({'math mcq', 'mcq'})).then(pl.lit("synth,direct"))
            .when(pl.col("exercise") == "math exercise").then(pl.lit("synth,noisy,cot"))
            .otherwise(pl.lit(""))
            .alias("condition"),

            # response
            pl.col("synthetic_answer").alias("response")
        ])

        # 3. Write (Collect triggers the execution)
        df.sink_parquet(output_dir / file_path.name)

if __name__ == "__main__":
    process_files()
