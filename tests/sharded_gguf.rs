//! Integration test: the sharded GGUF reader aggregates tensors from two shard
//! files when `split.count = 2` and `split.no` is set in each shard.

mod common;

use ananke::gguf;
use common::synth_gguf;
use tempfile::TempDir;

#[test]
fn sharded_gguf_aggregates_tensor_count() {
    let dir = TempDir::new().unwrap();

    // Shard filenames must follow the pattern `{base}-NNNNN-of-MMMMM.gguf`.
    let shard0 = dir.path().join("model-00001-of-00002.gguf");
    let shard1 = dir.path().join("model-00002-of-00002.gguf");

    // Shard 0: declares split.count = 2, split.no = 0.
    synth_gguf::Builder::new()
        .kv_string("general.architecture", "qwen3")
        .kv_u32("qwen3.block_count", 4)
        .kv_u32("split.count", 2)
        .kv_u32("split.no", 0)
        .tensor_f16("blk.0.attn_q.weight", 512 * 1024)
        .tensor_f16("blk.1.attn_q.weight", 512 * 1024)
        .write_to(&shard0);

    // Shard 1: declares split.count = 2, split.no = 1.
    synth_gguf::Builder::new()
        .kv_string("general.architecture", "qwen3")
        .kv_u32("qwen3.block_count", 4)
        .kv_u32("split.count", 2)
        .kv_u32("split.no", 1)
        .tensor_f16("blk.2.attn_q.weight", 512 * 1024)
        .tensor_f16("blk.3.attn_q.weight", 512 * 1024)
        .write_to(&shard1);

    let summary = gguf::read(&shard0).expect("sharded read must succeed");

    // The aggregated summary should contain tensors from both shards.
    assert_eq!(
        summary.tensors.len(),
        4,
        "aggregated tensor count must equal sum across shards"
    );
    assert_eq!(
        summary.shards.len(),
        2,
        "shard list must contain both shard paths"
    );
    assert!(
        summary.tensors.contains_key("blk.0.attn_q.weight"),
        "tensors from shard 0 must be present"
    );
    assert!(
        summary.tensors.contains_key("blk.3.attn_q.weight"),
        "tensors from shard 1 must be present"
    );
}
