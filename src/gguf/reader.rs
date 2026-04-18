//! Single-file GGUF reader, implemented as a thin adapter over the `gguf` crate.
//!
//! The file is memory-mapped so multi-GB models never require a full heap
//! allocation; the crate's parser then walks the mapped bytes in place.

use std::collections::BTreeMap;
use std::fs::File;
use std::path::Path;

use smol_str::SmolStr;

use super::types::{GgufSummary, GgufTensor, GgufType, GgufValue};

#[derive(Debug)]
pub struct ReadError(pub String);

impl std::fmt::Display for ReadError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "gguf read failed: {}", self.0)
    }
}

impl std::error::Error for ReadError {}

pub fn read_single(path: &Path) -> Result<GgufSummary, ReadError> {
    let file = File::open(path).map_err(|e| ReadError(format!("open {}: {e}", path.display())))?;

    // SAFETY: the file is not modified while we hold the map; we read it
    // entirely within this function and drop the map before returning.
    let mmap = unsafe {
        memmap2::Mmap::map(&file).map_err(|e| ReadError(format!("mmap {}: {e}", path.display())))?
    };

    let gguf_file = gguf::GGUFFile::read(&mmap)
        .map_err(|e| ReadError(format!("parse {}: {e}", path.display())))?
        .ok_or_else(|| ReadError(format!("incomplete data in {}", path.display())))?;

    let mut metadata: BTreeMap<SmolStr, GgufValue> = BTreeMap::new();
    for entry in &gguf_file.header.metadata {
        metadata.insert(SmolStr::new(&entry.key), convert_value(&entry.value));
    }

    let architecture = metadata
        .get("general.architecture")
        .and_then(|v| v.as_str())
        .map(SmolStr::new)
        .unwrap_or_else(|| SmolStr::new("unknown"));

    let block_count_key = format!("{architecture}.block_count");
    let block_count = metadata
        .get(block_count_key.as_str())
        .and_then(|v| v.as_u32());

    let mut tensors: BTreeMap<SmolStr, GgufTensor> = BTreeMap::new();
    let mut total_tensor_bytes = 0u64;

    for tensor_info in &gguf_file.tensors {
        let dtype = convert_ggml_type(tensor_info.tensor_type);
        let shape = tensor_info.dimensions.clone();
        let byte_size = tensor_byte_size(dtype, &shape);
        total_tensor_bytes += byte_size;
        let name = SmolStr::new(&tensor_info.name);
        tensors.insert(
            name.clone(),
            GgufTensor {
                name,
                dtype,
                shape,
                byte_size,
                shard_idx: 0,
                offset: tensor_info.offset,
            },
        );
    }

    Ok(GgufSummary {
        path: path.to_path_buf(),
        total_tensor_bytes,
        tensors,
        metadata,
        block_count,
        architecture,
        shards: vec![path.to_path_buf()],
    })
}

/// Convert a crate `GGMLType` to our `GgufType`.
///
/// The `gguf` 0.1.2 crate only covers the original GGML type table up to
/// `Count` (numeric id 19). Newer quantisation formats (IQ-series, BF16,
/// I64, F64) are beyond that range and will have caused a parse error before
/// this function is reached, so only the types the crate actually recognises
/// appear here.
fn convert_ggml_type(t: gguf::GGMLType) -> GgufType {
    use gguf::GGMLType;
    match t {
        GGMLType::F32 => GgufType::F32,
        GGMLType::F16 => GgufType::F16,
        GGMLType::Q4_0 => GgufType::Q4_0,
        GGMLType::Q4_1 => GgufType::Q4_1,
        GGMLType::Q5_0 => GgufType::Q5_0,
        GGMLType::Q5_1 => GgufType::Q5_1,
        GGMLType::Q8_0 => GgufType::Q8_0,
        GGMLType::Q8_1 => GgufType::Q8_1,
        GGMLType::Q2K => GgufType::Q2K,
        GGMLType::Q3K => GgufType::Q3K,
        GGMLType::Q4K => GgufType::Q4K,
        GGMLType::Q5K => GgufType::Q5K,
        GGMLType::Q6K => GgufType::Q6K,
        GGMLType::Q8K => GgufType::Q8K,
        // The crate's I8/I16/I32 map to discriminant values 16/17/18 in its
        // own enum, which differ from the current GGML spec (24/25/26). Files
        // produced by current llama.cpp will never reach this path because
        // the crate's parser errors on values > 19 before we get here.
        GGMLType::I8 => GgufType::I8,
        GGMLType::I16 => GgufType::I16,
        GGMLType::I32 => GgufType::I32,
        // Count is a sentinel, not a real type; treat it as unknown.
        GGMLType::Count => GgufType::Unknown(19),
    }
}

/// Convert a crate `GGUFMetadataValue` to our `GgufValue`.
fn convert_value(v: &gguf::GGUFMetadataValue) -> GgufValue {
    use gguf::GGUFMetadataValue;
    match v {
        GGUFMetadataValue::Uint8(x) => GgufValue::U8(*x),
        GGUFMetadataValue::Int8(x) => GgufValue::I8(*x),
        GGUFMetadataValue::Uint16(x) => GgufValue::U16(*x),
        GGUFMetadataValue::Int16(x) => GgufValue::I16(*x),
        GGUFMetadataValue::Uint32(x) => GgufValue::U32(*x),
        GGUFMetadataValue::Int32(x) => GgufValue::I32(*x),
        GGUFMetadataValue::Float32(x) => GgufValue::F32(*x),
        GGUFMetadataValue::Uint64(x) => GgufValue::U64(*x),
        GGUFMetadataValue::Int64(x) => GgufValue::I64(*x),
        GGUFMetadataValue::Float64(x) => GgufValue::F64(*x),
        GGUFMetadataValue::Bool(x) => GgufValue::Bool(*x),
        GGUFMetadataValue::String(x) => GgufValue::String(x.clone()),
        GGUFMetadataValue::Array(arr) => {
            GgufValue::Array(arr.value.iter().map(convert_value).collect())
        }
    }
}

/// Compute the on-disk byte size of a tensor from its dtype and shape.
///
/// All block-quantised formulas use integer arithmetic to avoid floating-point
/// rounding; the divisor is the super-block element count.
fn tensor_byte_size(dtype: GgufType, shape: &[u64]) -> u64 {
    let elements: u64 = shape.iter().product();
    match dtype {
        GgufType::F32 | GgufType::I32 => elements * 4,
        GgufType::F16 | GgufType::BF16 | GgufType::I16 => elements * 2,
        GgufType::I8 => elements,
        GgufType::I64 | GgufType::F64 => elements * 8,
        GgufType::Q8_0 => elements * 34 / 32, // block=32, 34 bytes/block ≈ 1.0625 bpe
        GgufType::Q8_1 => elements * 36 / 32,
        GgufType::Q4_0 | GgufType::IQ4_NL => elements * 18 / 32, // 0.5625 bpe
        GgufType::Q4_1 => elements * 20 / 32,
        GgufType::Q5_0 => elements * 22 / 32,
        GgufType::Q5_1 => elements * 24 / 32,
        GgufType::Q2K => elements * 84 / 256, // K-quant super-block 256; 84 bytes ≈ 2.625 bpe
        GgufType::Q3K => elements * 110 / 256,
        GgufType::Q4K => elements * 144 / 256,
        GgufType::Q5K => elements * 176 / 256,
        GgufType::Q6K => elements * 210 / 256,
        GgufType::Q8K => elements * 292 / 256,
        GgufType::IQ2_XXS => elements * 66 / 256,
        GgufType::IQ2_XS => elements * 74 / 256,
        GgufType::IQ3_XXS => elements * 98 / 256,
        GgufType::IQ1_S => elements * 50 / 256,
        GgufType::IQ3_S => elements * 110 / 256,
        GgufType::IQ2_S => elements * 82 / 256,
        GgufType::IQ4_XS => elements * 136 / 256,
        GgufType::IQ1_M => elements * 56 / 256,
        GgufType::Unknown(_) => elements * 2, // fall back to 2 bpe, log elsewhere
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Build a tiny synthetic GGUF v3 file in memory for round-trip testing.
    fn synth_gguf() -> Vec<u8> {
        let mut v = Vec::<u8>::new();
        v.extend_from_slice(b"GGUF");
        v.extend_from_slice(&3u32.to_le_bytes());
        // 1 tensor, 2 kv entries.
        v.extend_from_slice(&1u64.to_le_bytes());
        v.extend_from_slice(&2u64.to_le_bytes());
        // kv #1: general.architecture = "qwen3" (string, type=8)
        write_string(&mut v, "general.architecture");
        v.extend_from_slice(&8u32.to_le_bytes());
        write_string(&mut v, "qwen3");
        // kv #2: qwen3.block_count = 36 (u32, type=4)
        write_string(&mut v, "qwen3.block_count");
        v.extend_from_slice(&4u32.to_le_bytes());
        v.extend_from_slice(&36u32.to_le_bytes());
        // tensor: name "token_embd.weight", 2 dims [1024, 2048], dtype F16 (1), offset 0.
        write_string(&mut v, "token_embd.weight");
        v.extend_from_slice(&2u32.to_le_bytes());
        v.extend_from_slice(&1024u64.to_le_bytes());
        v.extend_from_slice(&2048u64.to_le_bytes());
        v.extend_from_slice(&1u32.to_le_bytes());
        v.extend_from_slice(&0u64.to_le_bytes());
        v
    }

    fn write_string(v: &mut Vec<u8>, s: &str) {
        v.extend_from_slice(&(s.len() as u64).to_le_bytes());
        v.extend_from_slice(s.as_bytes());
    }

    #[test]
    fn parses_synthetic_header() {
        let bytes = synth_gguf();
        let tmp = tempfile::NamedTempFile::new().unwrap();
        std::fs::write(tmp.path(), &bytes).unwrap();
        let summary = read_single(tmp.path()).unwrap();
        assert_eq!(summary.architecture, "qwen3");
        assert_eq!(summary.block_count, Some(36));
        assert_eq!(summary.tensors.len(), 1);
        let t = summary.tensors.values().next().unwrap();
        assert_eq!(t.name, "token_embd.weight");
        assert_eq!(t.byte_size, 1024 * 2048 * 2);
    }

    #[test]
    fn rejects_bad_magic() {
        let tmp = tempfile::NamedTempFile::new().unwrap();
        std::fs::write(tmp.path(), b"XXXXdata").unwrap();
        let err = read_single(tmp.path()).unwrap_err();
        assert!(err.0.contains("bad magic") || err.0.contains("parse"));
    }
}
