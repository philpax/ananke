//! Single-file GGUF reader.
//!
//! GGUF v3 layout: magic "GGUF" (4 bytes), version u32, tensor_count u64,
//! kv_count u64, then kv_count metadata entries, then tensor_count
//! tensor-info entries, then alignment padding, then the tensor data.
//! This reader walks the header only; it never mmaps or loads tensor data.

use std::{
    collections::BTreeMap,
    io::{BufReader, Read},
    path::Path,
};

use smol_str::SmolStr;

use super::types::{GgufSummary, GgufTensor, GgufType, GgufValue};
use crate::system::Fs;

const MAGIC: &[u8; 4] = b"GGUF";

#[derive(Debug)]
pub struct ReadError(pub String);

impl std::fmt::Display for ReadError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "gguf read failed: {}", self.0)
    }
}

impl std::error::Error for ReadError {}

pub fn read_single(fs: &dyn Fs, path: &Path) -> Result<GgufSummary, ReadError> {
    let file = fs
        .open(path)
        .map_err(|e| ReadError(format!("open {}: {e}", path.display())))?;
    let mut r = BufReader::new(file);

    let mut magic = [0u8; 4];
    r.read_exact(&mut magic)
        .map_err(|e| ReadError(format!("read magic: {e}")))?;
    if &magic != MAGIC {
        return Err(ReadError(format!("bad magic: {magic:?}")));
    }

    let version = read_u32(&mut r)?;
    if version != 3 && version != 2 {
        return Err(ReadError(format!("unsupported GGUF version {version}")));
    }

    let tensor_count = read_u64(&mut r)?;
    let kv_count = read_u64(&mut r)?;

    let mut metadata = BTreeMap::new();
    for _ in 0..kv_count {
        let key = read_string(&mut r)?;
        let value_type = read_u32(&mut r)?;
        let value = read_value(&mut r, value_type)?;
        metadata.insert(SmolStr::new(&key), value);
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

    let mut tensors = BTreeMap::new();
    let mut total_tensor_bytes = 0u64;

    for _ in 0..tensor_count {
        let name = read_string(&mut r)?;
        let n_dims = read_u32(&mut r)?;
        let mut shape = Vec::with_capacity(n_dims as usize);
        for _ in 0..n_dims {
            shape.push(read_u64(&mut r)?);
        }
        let dtype = GgufType::from_u32(read_u32(&mut r)?);
        if let GgufType::Unknown(id) = dtype {
            // Fail loudly rather than fall back to an F16-size guess. An
            // unknown dtype means the estimator would either over- or
            // under-reserve by a large factor, and silent fallback hid the
            // gpt-oss MXFP4 case for weeks. Fix: add the id to
            // `GgufType::from_u32` + `tensor_byte_size`.
            return Err(ReadError(format!(
                "{}: tensor `{name}` uses unsupported GGUF dtype id {id}; \
                 extend ananke::gguf::types::GgufType to cover it",
                path.display()
            )));
        }
        let offset = read_u64(&mut r)?;
        let byte_size = tensor_byte_size(dtype, &shape);
        total_tensor_bytes += byte_size;
        let sname = SmolStr::new(&name);
        tensors.insert(
            sname.clone(),
            GgufTensor {
                name: sname,
                dtype,
                shape,
                byte_size,
                shard_idx: 0,
                offset,
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

fn read_u8<R: Read>(r: &mut R) -> Result<u8, ReadError> {
    let mut b = [0u8; 1];
    r.read_exact(&mut b)
        .map_err(|e| ReadError(format!("read u8: {e}")))?;
    Ok(b[0])
}

fn read_i8<R: Read>(r: &mut R) -> Result<i8, ReadError> {
    Ok(read_u8(r)? as i8)
}

fn read_u16<R: Read>(r: &mut R) -> Result<u16, ReadError> {
    let mut b = [0u8; 2];
    r.read_exact(&mut b)
        .map_err(|e| ReadError(format!("read u16: {e}")))?;
    Ok(u16::from_le_bytes(b))
}

fn read_i16<R: Read>(r: &mut R) -> Result<i16, ReadError> {
    Ok(read_u16(r)? as i16)
}

fn read_u32<R: Read>(r: &mut R) -> Result<u32, ReadError> {
    let mut b = [0u8; 4];
    r.read_exact(&mut b)
        .map_err(|e| ReadError(format!("read u32: {e}")))?;
    Ok(u32::from_le_bytes(b))
}

fn read_i32<R: Read>(r: &mut R) -> Result<i32, ReadError> {
    Ok(read_u32(r)? as i32)
}

fn read_u64<R: Read>(r: &mut R) -> Result<u64, ReadError> {
    let mut b = [0u8; 8];
    r.read_exact(&mut b)
        .map_err(|e| ReadError(format!("read u64: {e}")))?;
    Ok(u64::from_le_bytes(b))
}

fn read_i64<R: Read>(r: &mut R) -> Result<i64, ReadError> {
    Ok(read_u64(r)? as i64)
}

fn read_f32<R: Read>(r: &mut R) -> Result<f32, ReadError> {
    Ok(f32::from_bits(read_u32(r)?))
}

fn read_f64<R: Read>(r: &mut R) -> Result<f64, ReadError> {
    Ok(f64::from_bits(read_u64(r)?))
}

fn read_bool<R: Read>(r: &mut R) -> Result<bool, ReadError> {
    Ok(read_u8(r)? != 0)
}

fn read_string<R: Read>(r: &mut R) -> Result<String, ReadError> {
    let len = read_u64(r)? as usize;
    let mut buf = vec![0u8; len];
    r.read_exact(&mut buf)
        .map_err(|e| ReadError(format!("read string bytes: {e}")))?;
    String::from_utf8(buf).map_err(|e| ReadError(format!("utf8: {e}")))
}

fn read_value<R: Read>(r: &mut R, tag: u32) -> Result<GgufValue, ReadError> {
    Ok(match tag {
        0 => GgufValue::U8(read_u8(r)?),
        1 => GgufValue::I8(read_i8(r)?),
        2 => GgufValue::U16(read_u16(r)?),
        3 => GgufValue::I16(read_i16(r)?),
        4 => GgufValue::U32(read_u32(r)?),
        5 => GgufValue::I32(read_i32(r)?),
        6 => GgufValue::F32(read_f32(r)?),
        7 => GgufValue::Bool(read_bool(r)?),
        8 => GgufValue::String(read_string(r)?),
        9 => {
            let inner = read_u32(r)?;
            let n = read_u64(r)? as usize;
            let mut v = Vec::with_capacity(n);
            for _ in 0..n {
                v.push(read_value(r, inner)?);
            }
            GgufValue::Array(v)
        }
        10 => GgufValue::U64(read_u64(r)?),
        11 => GgufValue::I64(read_i64(r)?),
        12 => GgufValue::F64(read_f64(r)?),
        other => return Err(ReadError(format!("unknown metadata type {other}"))),
    })
}

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
        GgufType::TQ1_0 => elements * 54 / 256, // ternary 1.7 bpe
        GgufType::TQ2_0 => elements * 66 / 256, // ternary 2.0 bpe
        GgufType::MXFP4 => elements * 17 / 32,  // 1-byte e8m0 scale + 16 bytes 4-bit / 32 elems
        GgufType::Unknown(_) => {
            // The reader rejects unknown dtypes before this point, so
            // reaching here is a programming error. Render defensively
            // rather than panicking mid-estimate.
            elements * 2
        }
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
        let fs = crate::system::InMemoryFs::new().with("/fake.gguf", synth_gguf());
        let summary = read_single(&fs, Path::new("/fake.gguf")).unwrap();
        assert_eq!(summary.architecture, "qwen3");
        assert_eq!(summary.block_count, Some(36));
        assert_eq!(summary.tensors.len(), 1);
        let t = summary.tensors.values().next().unwrap();
        assert_eq!(t.name, "token_embd.weight");
        assert_eq!(t.byte_size, 1024 * 2048 * 2);
    }

    #[test]
    fn rejects_bad_magic() {
        let fs = crate::system::InMemoryFs::new().with("/bad.gguf", b"XXXXdata".to_vec());
        let err = read_single(&fs, Path::new("/bad.gguf")).unwrap_err();
        assert!(err.0.contains("bad magic"));
    }
}
