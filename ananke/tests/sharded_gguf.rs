//! Integration test: the sharded GGUF reader aggregates tensors from two shard
//! files when `split.count = 2` and `split.no` is set in each shard.
#![cfg(feature = "test-fakes")]

mod common;

use std::path::Path;

use ananke::{gguf, system::InMemoryFs};
use common::synth_gguf;

#[test]
fn sharded_gguf_aggregates_tensor_count() {
    // Shard filenames must follow the pattern `{base}-NNNNN-of-MMMMM.gguf`.
    let shard0 = Path::new("/fake/model-00001-of-00002.gguf");
    let shard1 = Path::new("/fake/model-00002-of-00002.gguf");

    // Shard 0: declares split.count = 2, split.no = 0.
    let bytes0 = synth_gguf::Builder::new()
        .kv_string("general.architecture", "qwen3")
        .kv_u32("qwen3.block_count", 4)
        .kv_u32("split.count", 2)
        .kv_u32("split.no", 0)
        .tensor_f16("blk.0.attn_q.weight", 512 * 1024)
        .tensor_f16("blk.1.attn_q.weight", 512 * 1024)
        .build();

    // Shard 1: declares split.count = 2, split.no = 1.
    let bytes1 = synth_gguf::Builder::new()
        .kv_string("general.architecture", "qwen3")
        .kv_u32("qwen3.block_count", 4)
        .kv_u32("split.count", 2)
        .kv_u32("split.no", 1)
        .tensor_f16("blk.2.attn_q.weight", 512 * 1024)
        .tensor_f16("blk.3.attn_q.weight", 512 * 1024)
        .build();

    let fs = InMemoryFs::new();
    fs.insert(shard0, bytes0);
    fs.insert(shard1, bytes1);

    let summary = gguf::read(&fs, shard0).expect("sharded read must succeed");

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
