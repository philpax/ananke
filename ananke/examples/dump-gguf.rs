//! Dump a GGUF file's tensor table with per-layer expert/non-expert breakdown.
//!
//! Usage: cargo run --release --example dump-gguf -- /path/to/file.gguf

use std::{collections::BTreeMap, env};

use ananke::{gguf, system::LocalFs};

fn main() {
    let path = env::args()
        .nth(1)
        .expect("usage: dump-gguf <path-to-first-shard.gguf>");
    let summary =
        gguf::read(&LocalFs, std::path::Path::new(&path)).expect("gguf read failed");

    println!(
        "architecture: {}  block_count: {:?}  tensors: {}  shards: {}  total: {:.2} GiB",
        summary.architecture,
        summary.block_count,
        summary.tensors.len(),
        summary.shards.len(),
        summary.total_tensor_bytes as f64 / 1024.0_f64.powi(3)
    );
    println!();

    let n_layers = summary.block_count.unwrap_or(0);

    // Aggregate by tensor "category" across all layers.
    let mut by_category: BTreeMap<String, u64> = BTreeMap::new();
    for (name, tensor) in &summary.tensors {
        let category = categorise(name);
        *by_category.entry(category.into()).or_default() += tensor.byte_size;
    }

    println!(
        "{:<40} {:>12} {:>10}",
        "category", "total GiB", "% of model"
    );
    println!("{}", "-".repeat(65));
    let mut cats: Vec<_> = by_category.iter().collect();
    cats.sort_by(|a, b| b.1.cmp(a.1));
    for (cat, bytes) in cats {
        let gib = *bytes as f64 / 1024.0_f64.powi(3);
        let pct = *bytes as f64 / summary.total_tensor_bytes as f64 * 100.0;
        println!("{cat:<40} {gib:>12.3} {pct:>10.2}");
    }
    println!();

    // Per-layer breakdown for the first 3 layers (representative sample).
    println!("per-layer sample (blk.0 .. blk.2):");
    for layer in 0..n_layers.min(3) {
        println!("  blk.{layer}:");
        let prefix = format!("blk.{layer}.");
        let mut entries: Vec<(&str, u64)> = summary
            .tensors
            .iter()
            .filter(|(name, _)| name.starts_with(&prefix))
            .map(|(name, t)| (name.as_str(), t.byte_size))
            .collect();
        entries.sort_by_key(|e| std::cmp::Reverse(e.1));
        for (name, bytes) in entries {
            let mib = bytes as f64 / 1024.0_f64.powi(2);
            println!("    {name:<48} {mib:>10.1} MiB");
        }
    }
}

fn categorise(name: &str) -> &'static str {
    if !name.starts_with("blk.") {
        if name == "output.weight" {
            return "output.weight (head)";
        }
        if name == "token_embd.weight" {
            return "token_embd.weight";
        }
        return "non-layer (other)";
    }
    // blk.N.<kind>
    let Some((_, rest)) = name.split_once('.').and_then(|(_, r)| r.split_once('.')) else {
        return "unknown layer tensor";
    };
    if rest.starts_with("attn_") {
        "blk.*.attn_*"
    } else if rest.starts_with("ffn_gate_exps") {
        "blk.*.ffn_gate_exps (experts)"
    } else if rest.starts_with("ffn_up_exps") {
        "blk.*.ffn_up_exps (experts)"
    } else if rest.starts_with("ffn_down_exps") {
        "blk.*.ffn_down_exps (experts)"
    } else if rest.starts_with("ffn_gate_inp") {
        "blk.*.ffn_gate_inp (router)"
    } else if rest.starts_with("ffn_gate_shexp") {
        "blk.*.ffn_gate_shexp (shared expert)"
    } else if rest.starts_with("ffn_up_shexp") {
        "blk.*.ffn_up_shexp (shared expert)"
    } else if rest.starts_with("ffn_down_shexp") {
        "blk.*.ffn_down_shexp (shared expert)"
    } else if rest.starts_with("ffn_gate") {
        "blk.*.ffn_gate (dense)"
    } else if rest.starts_with("ffn_up") {
        "blk.*.ffn_up (dense)"
    } else if rest.starts_with("ffn_down") {
        "blk.*.ffn_down (dense)"
    } else if rest.contains("norm") {
        "blk.*.*_norm"
    } else {
        "blk.* other"
    }
}
