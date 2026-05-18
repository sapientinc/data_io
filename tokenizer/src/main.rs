use anyhow::{anyhow, Context, Result};
use clap::Parser;
use core_affinity;
use crossbeam_queue::SegQueue;
use indicatif::{ProgressBar, ProgressStyle};
use serde::{Deserialize, Serialize};
use serde_json::json;
use std::cmp::Reverse;
use std::collections::{HashMap, HashSet};
use std::fs::{self, File};
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::SystemTime;
use std::thread;
use tokenizers::Tokenizer;
use ndarray::ArrayView1;
use ndarray_npy::write_npy;

use tokenizer::{read_any_stream, scan_inputs};

// Optimize memory allocation
#[global_allocator]
static GLOBAL: mimalloc::MiMalloc = mimalloc::MiMalloc;

// --- CLI Arguments ---
#[derive(Parser, Debug)]
#[command(author, version, about)]
struct Args {
    #[arg(required = true, num_args = 1..)]
    dirs: Vec<PathBuf>,
    #[arg(short, long, required = true)]
    output_dir: PathBuf,

    #[arg(short, long, default_value = "Qwen/Qwen3-Next-80B-A3B-Instruct")]
    tokenizer_path: String,

    #[arg(long, default_value = "<|im_start|>")]
    boq: String,
    #[arg(long, default_value = "<|im_end|>")]
    eoq: String,
    #[arg(long, default_value = "<|box_end|>")]
    eoa: String,

    #[arg(long, value_delimiter = ',', num_args = 1..,
    default_values = ["direct=<|object_ref_start|>", "cot=<|object_ref_end|>", "noisy=<|quad_start|>", "synth=<|quad_end|>"])]
    conditions: Vec<String>,
}

// --- Data Structures ---

#[derive(Debug, Serialize, Deserialize)]
struct FileMetadata {
    source_mtime: u64,
    source_size: u64,
}

#[derive(Debug)]
struct WorkItem {
    input_path: PathBuf,
    output_subdir: PathBuf,
}

struct TokenizerContext {
    boq_id: u32,
    eoq_id: u32,
    eoa_id: u32,
    condition_ids: HashMap<String, u32>,
}

fn main() -> Result<()> {
    // Critical: Prevent tokenizer threads from fighting our file threads
    unsafe {
        std::env::set_var("TOKENIZERS_PARALLELISM", "false");
    }

    let args = Args::parse();
    let conditions: HashMap<String, String> = args.conditions.iter()
        .map(|s| s.split_once('=').map(|(k, v)| (k.to_owned(), v.to_owned())).unwrap())
        .collect();

    // 1. Setup Tokenizer
    println!("Loading tokenizer...");
    let tokenizer_path = Path::new(&args.tokenizer_path);
    let mut tokenizer = if tokenizer_path.exists() {
        // Resolve directory to tokenizer.json if necessary
        let file_path = if tokenizer_path.is_dir() {
            tokenizer_path.join("tokenizer.json")
        } else {
            tokenizer_path.to_path_buf()
        };
        Tokenizer::from_file(&file_path)
    } else {
        Tokenizer::from_pretrained(&args.tokenizer_path, None)
    }.map_err(|e| anyhow!(e))?;

    tokenizer.set_encode_special_tokens(true);  // Treat all special tokens as text.
    let tokenizer = Arc::new(tokenizer);

    // Tokenizer info
    // Create JSON object dynamically from Args and calculated data
    let tokenizer_info_json = json!({
        "tokenizer_path": args.tokenizer_path,
        "boq": args.boq,
        "eoq": args.eoq,
        "eoa": args.eoa,
        "condition_mapping": conditions,
        "vocab_size": tokenizer.get_vocab_size(true)
    });
    fs::create_dir_all(&args.output_dir)?;
    serde_json::to_writer(File::create(args.output_dir.join("tokenizer_info.json"))?, &tokenizer_info_json)?;

    // Setup context
    let ctx = Arc::new(TokenizerContext {
        boq_id: tokenizer.token_to_id(&args.boq).context("BOQ missing")?,
        eoq_id: tokenizer.token_to_id(&args.eoq).context("EOQ missing")?,
        eoa_id: tokenizer.token_to_id(&args.eoa).context("EOA missing")?,
        condition_ids: conditions.into_iter()
            .map(|(k, v)| Ok((k, tokenizer.token_to_id(&v).context("Condition missing")?)))
            .collect::<Result<_>>()?
    });

    // 2. Scan Inputs
    println!("Scanning inputs...");
    let mut expected_outputs = HashSet::new();
    let mut pending_work = Vec::new();

    let files = scan_inputs(&args.dirs)?;
    for f in files {
        let out_subdir = args.output_dir.join(&f.safe_name);
        expected_outputs.insert(out_subdir.clone());

        if should_process(&f.path, &out_subdir) {
            pending_work.push(WorkItem {
                input_path: f.path,
                output_subdir: out_subdir,
            });
        }
    }

    // Longest Processing Time First (LPT): Sort by size (Large -> Small) using cached metadata
    pending_work.sort_by_cached_key(|item| {
        Reverse(fs::metadata(&item.input_path)
                .map(|m| m.len())
                .unwrap_or(0))
    });
    let work_queue = SegQueue::new();
    for item in pending_work {
        work_queue.push(item);
    }

    // 3. Prune Orphans (Output exists, but input gone)
    if args.output_dir.exists() {
        println!("Pruning orphans...");
        for entry in fs::read_dir(&args.output_dir)? {
            let entry = entry?;
            let path = entry.path();
            if path.is_dir() && !expected_outputs.contains(&path) {
                println!("Removing orphan: {:?}", path.file_name().unwrap());
                fs::remove_dir_all(path)?;
            }
        }
    }

    // 4. Processing
    let total_work = work_queue.len();
    let core_ids = core_affinity::get_core_ids().unwrap();
    let num_threads = core_ids.len().saturating_sub(1).max(1);
    println!("Processing {} files on {} threads...", total_work, num_threads);

    let pb = Arc::new(ProgressBar::new(total_work as u64));
    pb.set_style(ProgressStyle::default_bar().template("{spinner:.green} {msg} {bar:40} {pos}/{len} {elapsed} ETA {eta}")?);
    let queue = Arc::new(work_queue);

    let mut handles = Vec::new();
    for i in 0..num_threads {
        let ctx = ctx.clone();
        let q = queue.clone();
        let tok = tokenizer.clone();
        let pb = pb.clone();
        let core_id = core_ids[i % core_ids.len()];

        handles.push(thread::spawn(move || {
            core_affinity::set_for_current(core_id);

            while let Some(item) = q.pop() {
                if let Err(e) = process_file(&item, &ctx, &tok) {
                    eprintln!("Failed {:?}: {}", item.input_path, e);
                }
                pb.inc(1);
            }
        }));
    }
    for h in handles { h.join().unwrap(); }
    pb.finish_with_message("Done.");

    Ok(())
}

// --- Core Logic ---

fn should_process(input: &Path, output_dir: &Path) -> bool {
    let meta_path = output_dir.join("metadata.json");
    if !meta_path.exists() { return true; }

    let input_meta = match fs::metadata(input) {
        Ok(m) => m,
        Err(_) => return true,
    };

    let cached: FileMetadata = match File::open(&meta_path).and_then(|f| Ok(serde_json::from_reader(f)?)) {
        Ok(c) => c,
        Err(_) => return true,
    };

    let mtime = input_meta.modified().unwrap_or(SystemTime::UNIX_EPOCH)
        .duration_since(SystemTime::UNIX_EPOCH).unwrap().as_secs();

    cached.source_size != input_meta.len() || cached.source_mtime != mtime
}

fn process_file(item: &WorkItem, ctx: &TokenizerContext, tok: &Tokenizer) -> Result<()> {
    // Vectors for output (re-used across rows to minimize allocations)
    // Use u64. u32 indices may overflow for large corpora
    let mut all_tokens: Vec<u32> = Vec::with_capacity(1024 * 1024);
    let mut inst_start = Vec::with_capacity(1024);
    let mut inst_len = Vec::with_capacity(1024);
    let mut resp_start = Vec::with_capacity(1024);
    let mut resp_len = Vec::with_capacity(1024);

    // Optimized closure to handle tokenization logic once
    let mut process_row = |condition: &str, instruction: &str, response: &str| {
        if let Ok(inst_enc) = tok.encode_fast(instruction, false) {
            if let Ok(resp_enc) = tok.encode_fast(response, false) {
                // Instruction: BOQ + Condition tokens + Encoded + EOQ
                let i_start = all_tokens.len();
                all_tokens.push(ctx.boq_id);
                for c in condition.split(',') {
                    all_tokens.push(ctx.condition_ids[c]);
                }
                all_tokens.extend_from_slice(inst_enc.get_ids());
                all_tokens.push(ctx.eoq_id);
                
                inst_start.push(i_start as u64);
                inst_len.push((all_tokens.len() - i_start) as u64);

                // Response: Encoded + EOA
                let r_start = all_tokens.len();
                all_tokens.extend_from_slice(resp_enc.get_ids());
                all_tokens.push(ctx.eoa_id);

                resp_start.push(r_start as u64);
                resp_len.push((all_tokens.len() - r_start) as u64);
            }
        }
    };

    // Read
    read_any_stream(&item.input_path, &mut process_row)?;

    // Write Output
    fs::create_dir_all(&item.output_subdir)?;

    write_npy(&item.output_subdir.join("tokens.npy"), &ArrayView1::from(&all_tokens))?;
    write_npy(&item.output_subdir.join("inst_start.npy"), &ArrayView1::from(&inst_start))?;
    write_npy(&item.output_subdir.join("inst_len.npy"), &ArrayView1::from(&inst_len))?;
    write_npy(&item.output_subdir.join("resp_start.npy"), &ArrayView1::from(&resp_start))?;
    write_npy(&item.output_subdir.join("resp_len.npy"), &ArrayView1::from(&resp_len))?;

    // Write Metadata
    let meta = fs::metadata(&item.input_path)?;
    let mtime = meta.modified()?.duration_since(SystemTime::UNIX_EPOCH)?.as_secs();

    let info = FileMetadata { source_mtime: mtime, source_size: meta.len() };
    let f = File::create(item.output_subdir.join("metadata.json"))?;
    serde_json::to_writer(f, &info)?;

    Ok(())
}
