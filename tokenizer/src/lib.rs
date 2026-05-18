use anyhow::{anyhow, Result};
use std::fs::File;
use std::io::BufReader;
use std::path::{Path, PathBuf};
use walkdir::WalkDir;
use parquet::arrow::arrow_reader::ParquetRecordBatchReaderBuilder;
use arrow::array::{Array, GenericStringArray};
use arrow::datatypes::DataType;
use serde::Deserialize;

pub struct FoundFile {
    pub path: PathBuf,
    pub safe_name: String,
}

/// Scans directories for parquet and jsonl files and computes safe names
pub fn scan_inputs(dirs: &[PathBuf]) -> Result<Vec<FoundFile>> {
    let mut files = Vec::new();
    for dir in dirs {
        for entry in WalkDir::new(dir).into_iter().filter_map(|e| e.ok()) {
            if entry.file_type().is_file() {
                let path = entry.path();
                let ext = path.extension().and_then(|s| s.to_str());
                if matches!(ext, Some("parquet" | "jsonl")) {
                    let safe_name = path.strip_prefix(dir)?.to_string_lossy().replace(['/', '\\'], "__");
                    files.push(FoundFile {
                        path: path.to_path_buf(),
                        safe_name,
                    });
                }
            }
        }
    }
    Ok(files)
}

pub fn read_any_stream<F>(path: &Path, callback: F) -> Result<()>
where F: FnMut(&str, &str, &str) {
    let ext = path.extension().and_then(|s| s.to_str()).unwrap_or("");
    match ext {
        "parquet" => read_parquet_stream(path, callback),
        "jsonl" => read_jsonl_stream(path, callback),
        _ => Err(anyhow!("Unsupported extension: {}", ext)),
    }
}

// --- Zero-Copy Readers ---

fn read_jsonl_stream<F>(path: &Path, mut callback: F) -> anyhow::Result<()>
where F: FnMut(&str, &str, &str) {
    // Minimal row struct only for JSONL deserialization
    #[derive(Debug, Deserialize)]
    struct JsonRow { // No lifetimes needed
        condition: String,
        instruction: String,
        response: String,
    }

    let file = File::open(path)?;
    let reader = BufReader::new(file);
    // Deserialize directly into a struct with &str to avoid allocation
    // We strictly assume the JSON lines contain string fields.
    let iter = serde_json::Deserializer::from_reader(reader).into_iter::<JsonRow>();
    for item in iter {
        match item {
            Ok(row) => callback(&row.condition, &row.instruction, &row.response),
            Err(e) => return Err(anyhow!("JSON Error: {}", e)),
        }
    }
    Ok(())
}

fn read_parquet_stream<F>(path: &Path, mut callback: F) -> anyhow::Result<()>
where F: FnMut(&str, &str, &str) {
    let file = File::open(path)?;
    let reader = ParquetRecordBatchReaderBuilder::try_new(file)?.build()?;

    for batch in reader {
        let batch = batch?;
        // We try both Utf8 (i32 offsets) and LargeUtf8 (i64 offsets)
        let c_col = batch.column_by_name("condition").ok_or_else(|| anyhow!("Missing condition"))?;
        let i_col = batch.column_by_name("instruction").ok_or_else(|| anyhow!("Missing instruction"))?;
        let r_col = batch.column_by_name("response").ok_or_else(|| anyhow!("Missing response"))?;

        // Inner processing loop macro to deduplicate code for DataType types
        macro_rules! process_batch {
            ($c_arr:expr, $i_arr:expr, $r_arr:expr) => {
                for i in 0..batch.num_rows() {
                    let c = $c_arr.value(i);
                    let inst = $i_arr.value(i);
                    let resp = $r_arr.value(i);
                    callback(c, inst, resp);
                }
            }
        }

        match (c_col.data_type(), i_col.data_type(), r_col.data_type()) {
            (DataType::Utf8, DataType::Utf8, DataType::Utf8) => {
                process_batch!(
                c_col.as_any().downcast_ref::<GenericStringArray<i32>>().unwrap(),
                i_col.as_any().downcast_ref::<GenericStringArray<i32>>().unwrap(),
                r_col.as_any().downcast_ref::<GenericStringArray<i32>>().unwrap()
            );
            }
            (DataType::LargeUtf8, DataType::LargeUtf8, DataType::LargeUtf8) => {
                process_batch!(
                c_col.as_any().downcast_ref::<GenericStringArray<i64>>().unwrap(),
                i_col.as_any().downcast_ref::<GenericStringArray<i64>>().unwrap(),
                r_col.as_any().downcast_ref::<GenericStringArray<i64>>().unwrap()
            );
            }
            _ => {
                return Err(anyhow!("Skipping batch with mixed/unsupported string types in {:?}", path));
            }
        }
    }
    Ok(())
}
