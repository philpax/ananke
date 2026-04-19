//! Multi-shard GGUF aggregation.

use std::path::{Path, PathBuf};

use super::{
    reader::{ReadError, read_single},
    types::GgufSummary,
};

/// Read a GGUF model. If the file is shard 0 of a multi-shard set
/// (metadata has `split.count > 1`), walk all shards and return an
/// aggregated summary. Otherwise return the single-file summary.
pub fn read(path: &Path) -> Result<GgufSummary, ReadError> {
    let first = read_single(path)?;
    let split_count = first
        .metadata
        .get("split.count")
        .and_then(|v| v.as_u32())
        .unwrap_or(1);
    if split_count <= 1 {
        return Ok(first);
    }

    // If the user pointed at a non-zero shard, normalise to shard 0 first.
    let split_no = first
        .metadata
        .get("split.no")
        .and_then(|v| v.as_u32())
        .unwrap_or(0);
    let base_path = if split_no != 0 {
        shard_path(path, 0, split_count).ok_or_else(|| {
            ReadError(format!(
                "could not derive shard 0 path from {}",
                path.display()
            ))
        })?
    } else {
        path.to_path_buf()
    };

    let zero = if split_no != 0 {
        read_single(&base_path)?
    } else {
        first
    };

    let mut agg_tensors = zero.tensors.clone();
    let mut total_bytes = zero.total_tensor_bytes;
    let mut shards = vec![zero.path.clone()];

    for idx in 1..split_count {
        let sp = shard_path(&base_path, idx, split_count).ok_or_else(|| {
            ReadError(format!(
                "shard {idx}/{split_count}: could not derive path from {}",
                base_path.display()
            ))
        })?;
        let part = read_single(&sp)
            .map_err(|e| ReadError(format!("shard {idx}/{split_count}: {}", e.0)))?;
        let part_count = part
            .metadata
            .get("split.count")
            .and_then(|v| v.as_u32())
            .unwrap_or(1);
        if part_count != split_count {
            return Err(ReadError(format!(
                "shard {idx}/{split_count}: split.count mismatch (expected {split_count}, got {part_count})"
            )));
        }
        total_bytes += part.total_tensor_bytes;
        for (name, mut tensor) in part.tensors {
            tensor.shard_idx = idx as u16;
            agg_tensors.insert(name, tensor);
        }
        shards.push(sp);
    }

    Ok(GgufSummary {
        path: base_path,
        total_tensor_bytes: total_bytes,
        tensors: agg_tensors,
        metadata: zero.metadata.clone(),
        block_count: zero.block_count,
        architecture: zero.architecture.clone(),
        shards,
    })
}

/// Compute the shard path for `{basename}-{NNNNN}-of-{MMMMM}.gguf`
/// given an input that is any shard (we derive basename by stripping
/// the `-NNNNN-of-MMMMM.gguf` suffix).
pub(crate) fn shard_path(any_shard: &Path, zero_based_idx: u32, count: u32) -> Option<PathBuf> {
    let stem = any_shard.file_name()?.to_str()?;
    // Expected pattern: <basename>-NNNNN-of-MMMMM.gguf
    let (base, _rest) = stem.rsplit_once(".gguf")?;
    let base_no_shard = base.rsplit_once("-of-")?.0.rsplit_once('-')?.0;
    let filename = format!(
        "{base_no_shard}-{:05}-of-{:05}.gguf",
        zero_based_idx + 1,
        count
    );
    let parent = any_shard.parent().unwrap_or_else(|| Path::new(""));
    Some(parent.join(filename))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn shard_path_derives() {
        let p = PathBuf::from("/m/Qwen3-235B-Instruct-UD-Q2_K_XL-00001-of-00002.gguf");
        let s0 = shard_path(&p, 0, 2).unwrap();
        let s1 = shard_path(&p, 1, 2).unwrap();
        assert_eq!(
            s0,
            PathBuf::from("/m/Qwen3-235B-Instruct-UD-Q2_K_XL-00001-of-00002.gguf")
        );
        assert_eq!(
            s1,
            PathBuf::from("/m/Qwen3-235B-Instruct-UD-Q2_K_XL-00002-of-00002.gguf")
        );
    }
}
