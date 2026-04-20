//! Dump a GGUF file's tensor table with per-layer expert/non-expert breakdown.
//!
//! Usage: cargo run --example dump-gguf -- /path/to/file.gguf

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

    // Surface the metadata keys the estimator cares about — if any of
    // these are missing under the arch's canonical prefix the KV math
    // silently collapses to zero (we hit this with nemotron's "deci"
    // arch). Loop over whatever prefixes show up in the metadata table
    // so the operator sees the actual keys rather than assumptions.
    println!();
    println!("attention + context metadata (both scalar and array shapes):");
    let prefixes_of_interest = [
        "attention.head_count_kv",
        "attention.head_count",
        "attention.key_length",
        "attention.value_length",
        "attention.key_length_swa",
        "attention.value_length_swa",
        "attention.sliding_window",
        "attention.sliding_window_pattern",
        "attention.shared_kv_layers",
        "attention.layer_types",
        "full_attention_interval",
        "context_length",
        "embedding_length",
        "block_count",
    ];
    let keys: Vec<_> = summary.metadata.keys().cloned().collect();
    for suffix in prefixes_of_interest {
        for k in &keys {
            if k.ends_with(suffix) {
                // Some keys are u32 scalars (head_count, context_length);
                // others are u32 arrays (sliding_window_pattern as a
                // per-layer mask, head_count_kv on variable-KV families).
                // Print both interpretations so the operator sees what's
                // actually in the GGUF.
                let value = summary.metadata.get(k);
                let scalar = value.and_then(|v| v.as_u32());
                let array = value.and_then(|v| v.as_u32_array());
                let bools = value.and_then(|v| v.as_bool_array());
                match (scalar, array.as_deref(), bools.as_deref()) {
                    (Some(s), Some(a), _) if a.len() == 1 && a[0] == s => {
                        println!("  {k} = {s}");
                    }
                    (Some(s), _, _) => println!("  {k} = {s}"),
                    (None, Some(a), _) => {
                        let preview: Vec<String> = a.iter().take(8).map(|v| v.to_string()).collect();
                        let tail = if a.len() > 8 { ", …" } else { "" };
                        println!(
                            "  {k} = [{}{}] (len={})",
                            preview.join(","),
                            tail,
                            a.len()
                        );
                    }
                    (None, None, Some(b)) => {
                        let preview: Vec<&str> = b.iter().take(8).map(|v| if *v { "T" } else { "F" }).collect();
                        let tail = if b.len() > 8 { ", …" } else { "" };
                        println!(
                            "  {k} = [{}{}] (bools, len={}, {}/{} true)",
                            preview.join(","),
                            tail,
                            b.len(),
                            b.iter().filter(|v| **v).count(),
                            b.len(),
                        );
                    }
                    (None, None, None) => println!("  {k} = <non-integer>"),
                }
            }
        }
    }
    // Dump every metadata key that mentions attention / sliding / window
    // verbatim. Useful for chasing architecture-specific metadata names
    // (gemma3's SWA pattern, nemotron's per-layer attention, etc.).
    println!();
    println!("all attention-related keys:");
    for k in &keys {
        let lk = k.to_lowercase();
        if lk.contains("attention") || lk.contains("sliding") || lk.contains("window") {
            println!("  {k}");
        }
    }
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

    // Non-layer tensor list: token_embd, output head, and anything else that
    // doesn't match `blk.N.*`. Useful when adding a new architecture so we
    // can see which tensors `collect_non_layer` will route to CPU (token
    // embedding, PLE) vs to GPU 0 (output head, everything else).
    println!("non-layer tensors (not under blk.N.*):");
    let mut non_layer: Vec<(&str, u64)> = summary
        .tensors
        .iter()
        .filter(|(name, _)| !name.starts_with("blk."))
        .map(|(name, t)| (name.as_str(), t.byte_size))
        .collect();
    non_layer.sort_by_key(|e| std::cmp::Reverse(e.1));
    for (name, bytes) in non_layer {
        let mib = bytes as f64 / 1024.0_f64.powi(2);
        println!("    {name:<48} {mib:>10.1} MiB");
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
