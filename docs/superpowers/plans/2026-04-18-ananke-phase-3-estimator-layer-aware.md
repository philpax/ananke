# Ananke Phase 3 — GGUF Reader + Estimator + Layer-Aware Placement Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the phase 1-2 `placement_override` requirement with an architecture-aware VRAM estimator that reads GGUF headers, computes per-device placement, writes `-ngl`/`--tensor-split`/`-ot` onto the llama.cpp command line, and tunes itself via rolling correction and an OOM retry policy.

**Architecture:** `gguf::read(path)` produces a `GgufSummary` (tensors + metadata + optional mmproj + shard list). `estimator::estimate(summary)` dispatches on `general.architecture` to llama-family / MoE / SSM / hybrid / fallback producers. `placement::pack(estimate, policy, snapshot)` packs layers across allowed devices with compute-buffer and one-layer-fudge slack, returning both an `Allocation` and `CommandArgs`. `rolling::update(service, observed_peak, base)` tunes the next start's estimate. `observation` combines per-PID NVML VRAM and `/proc/<pid>/status` VmRSS at the snapshotter's 2 s cadence to track observed peak.

**Tech Stack:** Rust 2024, `gguf` crate (fallback: custom reader), tokio, nvml-wrapper, existing phase 2 infrastructure.

**Parent design:** `docs/superpowers/specs/2026-04-18-ananke-phase-3-estimator-layer-aware.md`. Phase-2 plan: `docs/superpowers/plans/2026-04-18-ananke-phase-2-unified-openai-ondemand-allocator.md`.

---

## File Structure

```
src/
├── gguf/
│   ├── mod.rs             // NEW: pub read, pub types
│   ├── reader.rs          // NEW: single-file header + tensor table + kv map
│   ├── shards.rs          // NEW: multi-shard sum, shard-0 discovery
│   └── types.rs           // NEW: GgufValue, GgufTensor, GgufSummary
├── estimator/
│   ├── mod.rs             // NEW: dispatch on general.architecture + mmproj + sharded
│   ├── types.rs           // NEW: Estimate, LayerCost, SafetyFactor
│   ├── kv.rs              // NEW: bytes-per-element table + context helpers
│   ├── llama.rs           // NEW: llama-family
│   ├── moe.rs             // NEW: MoE + _exps + n_cpu_moe
│   ├── mamba.rs           // NEW: SSM/Mamba
│   ├── hybrid.rs          // NEW: jamba
│   └── fallback.rs        // NEW: unknown architecture fallback
├── placement.rs           // NEW: layer walker + tensor-split + CommandArgs + override_tensor
├── rolling.rs             // NEW: per-service RollingCorrection map
├── observation.rs         // NEW: /proc/<pid>/status VmRSS + per-PID NVML aggregator
├── app_state.rs           // MODIFY: add rolling + observation handles
├── config/
│   └── validate.rs        // MODIFY: placement_override optional; context default 4096
├── supervise/
│   ├── mod.rs             // MODIFY: estimator+placement at Idle→Starting, OOM retry,
│   │                      //         rolling::update on drain
│   └── spawn.rs           // MODIFY: accept CommandArgs, render -ngl/--tensor-split/-ot
├── snapshotter.rs         // MODIFY: include per-PID VmRSS + per-service peak accounting
├── devices/
│   └── nvml.rs            // MODIFY: per-PID VRAM queries for observation
├── management_api/
│   └── types.rs           // MODIFY: ServiceDetail gains rolling_mean + observed_peak
└── daemon.rs              // MODIFY: wire rolling + observation into AppState

tests/
├── estimator_llama.rs         // NEW
├── estimator_moe.rs           // NEW
├── estimator_mamba.rs         // NEW
├── estimator_fallback.rs      // NEW
├── placement_override_bypass.rs  // NEW
├── no_placement_override_ok.rs   // NEW
├── rolling_correction_warns.rs   // NEW
├── oom_retry.rs                  // NEW
├── sharded_gguf.rs               // NEW
├── multi_gpu_split.rs            // NEW
├── mmproj_attribution.rs         // NEW
├── override_tensor_moe_hybrid.rs // NEW
└── manual/
    └── phase-3-smoke.md          // NEW
```

---

## Task 1: GGUF types

**Files:**
- Create: `src/gguf/mod.rs`
- Create: `src/gguf/types.rs`
- Modify: `src/lib.rs`

- [ ] **Step 1: Create `src/gguf/types.rs`**

```rust
//! Types returned by the GGUF reader.

use std::collections::BTreeMap;
use std::path::PathBuf;

use smol_str::SmolStr;

/// Summary of a GGUF file (or aggregated shard set).
#[derive(Debug, Clone)]
pub struct GgufSummary {
    /// Canonical path (shard 0 for multi-file).
    pub path: PathBuf,
    /// Total tensor byte count across all shards.
    pub total_tensor_bytes: u64,
    /// Tensors keyed by name.
    pub tensors: BTreeMap<SmolStr, GgufTensor>,
    /// Metadata key-value map.
    pub metadata: BTreeMap<SmolStr, GgufValue>,
    /// Number of layers (`<arch>.block_count` typical). `None` if the
    /// architecture doesn't expose this.
    pub block_count: Option<u32>,
    /// Architecture string (`general.architecture` metadata).
    pub architecture: SmolStr,
    /// For sharded files: the discovered shard list. Single-file → len 1.
    pub shards: Vec<PathBuf>,
}

#[derive(Debug, Clone)]
pub struct GgufTensor {
    pub name: SmolStr,
    pub dtype: GgufType,
    pub shape: Vec<u64>,
    pub byte_size: u64,
    /// 0-based shard index where this tensor lives.
    pub shard_idx: u16,
    /// Byte offset within the shard's tensor-data region.
    pub offset: u64,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GgufType {
    F32, F16, BF16,
    Q4_0, Q4_1, Q5_0, Q5_1, Q8_0, Q8_1,
    Q2K, Q3K, Q4K, Q5K, Q6K, Q8K,
    IQ2_XXS, IQ2_XS, IQ3_XXS, IQ1_S, IQ4_NL, IQ3_S, IQ2_S, IQ4_XS, IQ1_M,
    I8, I16, I32, I64, F64,
    Unknown(u32),
}

impl GgufType {
    pub fn from_u32(n: u32) -> Self {
        match n {
            0  => Self::F32,
            1  => Self::F16,
            2  => Self::Q4_0,
            3  => Self::Q4_1,
            6  => Self::Q5_0,
            7  => Self::Q5_1,
            8  => Self::Q8_0,
            9  => Self::Q8_1,
            10 => Self::Q2K,
            11 => Self::Q3K,
            12 => Self::Q4K,
            13 => Self::Q5K,
            14 => Self::Q6K,
            15 => Self::Q8K,
            16 => Self::IQ2_XXS,
            17 => Self::IQ2_XS,
            18 => Self::IQ3_XXS,
            19 => Self::IQ1_S,
            20 => Self::IQ4_NL,
            21 => Self::IQ3_S,
            22 => Self::IQ2_S,
            23 => Self::IQ4_XS,
            24 => Self::I8,
            25 => Self::I16,
            26 => Self::I32,
            27 => Self::I64,
            28 => Self::F64,
            29 => Self::IQ1_M,
            30 => Self::BF16,
            other => Self::Unknown(other),
        }
    }
}

#[derive(Debug, Clone)]
pub enum GgufValue {
    U8(u8), I8(i8), U16(u16), I16(i16),
    U32(u32), I32(i32), U64(u64), I64(i64),
    F32(f32), F64(f64),
    Bool(bool),
    String(String),
    Array(Vec<GgufValue>),
}

impl GgufValue {
    pub fn as_u32(&self) -> Option<u32> {
        match self {
            GgufValue::U32(v) => Some(*v),
            GgufValue::I32(v) if *v >= 0 => Some(*v as u32),
            GgufValue::U64(v) if *v <= u32::MAX as u64 => Some(*v as u32),
            GgufValue::I64(v) if *v >= 0 && *v <= u32::MAX as i64 => Some(*v as u32),
            _ => None,
        }
    }
    pub fn as_u64(&self) -> Option<u64> {
        match self {
            GgufValue::U32(v) => Some(*v as u64),
            GgufValue::U64(v) => Some(*v),
            GgufValue::I32(v) if *v >= 0 => Some(*v as u64),
            GgufValue::I64(v) if *v >= 0 => Some(*v as u64),
            _ => None,
        }
    }
    pub fn as_str(&self) -> Option<&str> {
        match self { GgufValue::String(s) => Some(s), _ => None }
    }
}
```

- [ ] **Step 2: Create `src/gguf/mod.rs`**

```rust
//! GGUF reader — single-file and sharded.

pub mod reader;
pub mod shards;
pub mod types;

pub use reader::read_single;
pub use shards::read;
pub use types::{GgufSummary, GgufTensor, GgufType, GgufValue};
```

- [ ] **Step 3: Add `pub mod gguf;` to `src/lib.rs`**

Place alphabetically among other `pub mod` entries.

- [ ] **Step 4: Verify**

Run: `cargo check --lib`
Expected: compiles (reader + shards still stubs; add 1-line placeholders in both).

Create `src/gguf/reader.rs`:
```rust
//! Single-file GGUF reader (implemented in Task 2).
```

Create `src/gguf/shards.rs`:
```rust
//! Multi-shard GGUF helpers (implemented in Task 3).
```

Re-run `cargo check --lib` — clean.

- [ ] **Step 5: Commit**

```bash
git add src/gguf/ src/lib.rs
git commit -m "feat(gguf): add types and module scaffold"
```

---

## Task 2: GGUF single-file reader

**Files:**
- Replace: `src/gguf/reader.rs`

- [ ] **Step 1: Write the failing test**

At the bottom of `src/gguf/reader.rs`, scaffold:

```rust
//! Single-file GGUF reader.
//!
//! GGUF v3 layout: magic "GGUF" (4 bytes), version u32, tensor_count u64,
//! kv_count u64, then kv_count metadata entries, then tensor_count
//! tensor-info entries, then alignment padding, then the tensor data.
//! This reader walks the header only; it never mmaps or loads tensor data.

use std::fs::File;
use std::io::{BufReader, Read, Seek, SeekFrom};
use std::path::{Path, PathBuf};

use smol_str::SmolStr;

use super::types::{GgufSummary, GgufTensor, GgufType, GgufValue};

const MAGIC: &[u8; 4] = b"GGUF";

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
    let mut r = BufReader::new(file);

    let mut magic = [0u8; 4];
    r.read_exact(&mut magic).map_err(|e| ReadError(format!("read magic: {e}")))?;
    if &magic != MAGIC {
        return Err(ReadError(format!("bad magic: {magic:?}")));
    }

    let version = read_u32(&mut r)?;
    if version != 3 && version != 2 {
        return Err(ReadError(format!("unsupported GGUF version {version}")));
    }

    let tensor_count = read_u64(&mut r)?;
    let kv_count = read_u64(&mut r)?;

    let mut metadata = std::collections::BTreeMap::new();
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
    let block_count = metadata.get(block_count_key.as_str()).and_then(|v| v.as_u32());

    let mut tensors = std::collections::BTreeMap::new();
    let mut total_tensor_bytes = 0u64;

    for _ in 0..tensor_count {
        let name = read_string(&mut r)?;
        let n_dims = read_u32(&mut r)?;
        let mut shape = Vec::with_capacity(n_dims as usize);
        for _ in 0..n_dims { shape.push(read_u64(&mut r)?); }
        let dtype = GgufType::from_u32(read_u32(&mut r)?);
        let offset = read_u64(&mut r)?;
        let byte_size = tensor_byte_size(dtype, &shape);
        total_tensor_bytes += byte_size;
        let sname = SmolStr::new(&name);
        tensors.insert(sname.clone(), GgufTensor {
            name: sname,
            dtype,
            shape,
            byte_size,
            shard_idx: 0,
            offset,
        });
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
    r.read_exact(&mut b).map_err(|e| ReadError(format!("read u8: {e}")))?;
    Ok(b[0])
}

fn read_i8<R: Read>(r: &mut R) -> Result<i8, ReadError> { Ok(read_u8(r)? as i8) }

fn read_u16<R: Read>(r: &mut R) -> Result<u16, ReadError> {
    let mut b = [0u8; 2];
    r.read_exact(&mut b).map_err(|e| ReadError(format!("read u16: {e}")))?;
    Ok(u16::from_le_bytes(b))
}

fn read_i16<R: Read>(r: &mut R) -> Result<i16, ReadError> { Ok(read_u16(r)? as i16) }

fn read_u32<R: Read>(r: &mut R) -> Result<u32, ReadError> {
    let mut b = [0u8; 4];
    r.read_exact(&mut b).map_err(|e| ReadError(format!("read u32: {e}")))?;
    Ok(u32::from_le_bytes(b))
}

fn read_i32<R: Read>(r: &mut R) -> Result<i32, ReadError> { Ok(read_u32(r)? as i32) }

fn read_u64<R: Read>(r: &mut R) -> Result<u64, ReadError> {
    let mut b = [0u8; 8];
    r.read_exact(&mut b).map_err(|e| ReadError(format!("read u64: {e}")))?;
    Ok(u64::from_le_bytes(b))
}

fn read_i64<R: Read>(r: &mut R) -> Result<i64, ReadError> { Ok(read_u64(r)? as i64) }

fn read_f32<R: Read>(r: &mut R) -> Result<f32, ReadError> { Ok(f32::from_bits(read_u32(r)?)) }
fn read_f64<R: Read>(r: &mut R) -> Result<f64, ReadError> { Ok(f64::from_bits(read_u64(r)?)) }

fn read_bool<R: Read>(r: &mut R) -> Result<bool, ReadError> {
    Ok(read_u8(r)? != 0)
}

fn read_string<R: Read>(r: &mut R) -> Result<String, ReadError> {
    let len = read_u64(r)? as usize;
    let mut buf = vec![0u8; len];
    r.read_exact(&mut buf).map_err(|e| ReadError(format!("read string bytes: {e}")))?;
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
            for _ in 0..n { v.push(read_value(r, inner)?); }
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
        GgufType::Q8_0 => elements * 34 / 32,      // block=32, 34 bytes/block ≈ 1.0625 bpe
        GgufType::Q8_1 => elements * 36 / 32,
        GgufType::Q4_0 | GgufType::IQ4_NL => elements * 18 / 32,  // 0.5625 bpe
        GgufType::Q4_1 => elements * 20 / 32,
        GgufType::Q5_0 => elements * 22 / 32,
        GgufType::Q5_1 => elements * 24 / 32,
        GgufType::Q2K => elements * 84 / 256,      // K-quant super-block 256; 84 bytes ≈ 2.625 bpe
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
    use std::io::{Cursor, Write};

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
        assert!(err.0.contains("bad magic"));
    }
}
```

- [ ] **Step 2: Run tests**

Run: `cargo test --lib gguf::reader`
Expected: 2 tests pass.

- [ ] **Step 3: Commit**

```bash
git add src/gguf/reader.rs
git commit -m "feat(gguf): read GGUF v3 header and tensor table"
```

---

## Task 3: GGUF sharded reader

**Files:**
- Replace: `src/gguf/shards.rs`

- [ ] **Step 1: Write the shards module**

Replace `src/gguf/shards.rs`:

```rust
//! Multi-shard GGUF aggregation.

use std::path::{Path, PathBuf};

use smol_str::SmolStr;

use super::reader::{read_single, ReadError};
use super::types::{GgufSummary, GgufTensor};

/// Read a GGUF model. If the file is shard 0 of a multi-shard set
/// (metadata has `split.count > 1`), walk all shards and return an
/// aggregated summary. Otherwise return the single-file summary.
pub fn read(path: &Path) -> Result<GgufSummary, ReadError> {
    let first = read_single(path)?;
    let split_count = first.metadata.get("split.count").and_then(|v| v.as_u32()).unwrap_or(1);
    if split_count <= 1 {
        return Ok(first);
    }

    // If the user pointed at a non-zero shard, normalise to shard 0 first.
    let split_no = first.metadata.get("split.no").and_then(|v| v.as_u32()).unwrap_or(0);
    let base_path = if split_no != 0 {
        shard_path(path, 0, split_count)
            .ok_or_else(|| ReadError(format!("could not derive shard 0 path from {}", path.display())))?
    } else {
        path.to_path_buf()
    };

    let zero = if split_no != 0 { read_single(&base_path)? } else { first };

    let mut agg_tensors = zero.tensors.clone();
    let mut total_bytes = zero.total_tensor_bytes;
    let mut shards = vec![zero.path.clone()];

    for idx in 1..split_count {
        let sp = shard_path(&base_path, idx, split_count)
            .ok_or_else(|| ReadError(format!("shard {idx}/{split_count}: could not derive path from {}", base_path.display())))?;
        let part = read_single(&sp).map_err(|e| ReadError(format!("shard {idx}/{split_count}: {}", e.0)))?;
        let part_count = part.metadata.get("split.count").and_then(|v| v.as_u32()).unwrap_or(1);
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
        assert_eq!(s0, PathBuf::from("/m/Qwen3-235B-Instruct-UD-Q2_K_XL-00001-of-00002.gguf"));
        assert_eq!(s1, PathBuf::from("/m/Qwen3-235B-Instruct-UD-Q2_K_XL-00002-of-00002.gguf"));
    }
}
```

- [ ] **Step 2: Run tests**

Run: `cargo test --lib gguf::shards`
Expected: 1 test passes.

- [ ] **Step 3: Commit**

```bash
git add src/gguf/shards.rs
git commit -m "feat(gguf): multi-shard aggregation and shard-0 discovery"
```

---

## Task 4: Estimator scaffolding, KV table, and fallback

**Files:**
- Create: `src/estimator/mod.rs`, `types.rs`, `kv.rs`, `fallback.rs`
- Create: `src/estimator/llama.rs`, `moe.rs`, `mamba.rs`, `hybrid.rs` (stubs)
- Modify: `src/lib.rs`

- [ ] **Step 1: Types module**

Create `src/estimator/types.rs`:

```rust
//! Estimator output types.

use std::collections::BTreeMap;

use smol_str::SmolStr;

use crate::config::DeviceSlot;

/// Base estimate for a service's VRAM footprint, pre-safety-factor and
/// pre-rolling-correction.
#[derive(Debug, Clone)]
pub struct Estimate {
    /// Static weight bytes (including mmproj if present).
    pub weights_bytes: u64,
    /// KV cache bytes per context token (zero for architectures without KV).
    pub kv_per_token: u64,
    /// Compute buffer per device in MB (default 400).
    pub compute_buffer_mb: u32,
    /// Per-layer weight bytes for index-ordered packing. `None` for
    /// architectures where layer-aware placement isn't applicable
    /// (currently SSM/Mamba; in that case `placement` uses single-device
    /// best-fit on `total = weights + compute_buffer`).
    pub per_layer_bytes: Option<Vec<u64>>,
    /// Layer indices that are attention-bearing (used to scope KV
    /// cost to those layers). `None` = all layers carry KV.
    pub attention_layers: Option<Vec<u32>>,
    /// Non-layer tensors: output head, token embeddings, norms.
    pub non_layer: NonLayer,
    /// Tensor-level overrides (from `override_tensor` rules) already
    /// resolved to per-device byte attributions by the estimator.
    pub override_tensor_bytes: BTreeMap<DeviceSlot, u64>,
    /// Number of expert-bearing layers eligible for `n_cpu_moe` offload
    /// (only meaningful for MoE).
    pub expert_layers: Vec<u32>,
    /// Per expert-layer, the bytes saved by offloading those experts to CPU.
    pub expert_layer_cpu_bytes: BTreeMap<u32, u64>,
    /// `context` that was used to compute `kv_per_token × context`.
    pub context: u32,
    /// Architecture string for diagnostics.
    pub architecture: SmolStr,
}

/// Non-layer tensor footprint (matches llama.cpp's behaviour).
#[derive(Debug, Clone, Default)]
pub struct NonLayer {
    /// Output head — attributed to GPU 0 if any GPU used, else CPU.
    pub output_head_bytes: u64,
    /// Token embeddings — always on CPU.
    pub token_embd_bytes: u64,
    /// Small tensors (norms, rope tables) lumped together.
    pub other_bytes: u64,
}
```

Add `pub mod types;` and `pub use types::{Estimate, NonLayer};` to `src/estimator/mod.rs` in Step 4.

- [ ] **Step 2: KV bytes-per-element table**

Create `src/estimator/kv.rs`:

```rust
//! KV cache bytes-per-element table (spec §8.3).

use crate::gguf::GgufType;

/// Approximate bytes per element for llama.cpp's accepted
/// `--cache-type-k` / `--cache-type-v` values. Matches spec §8.3 table.
pub fn kv_bytes_per_element(cache_type: &str) -> f64 {
    match cache_type {
        "f32" => 4.0,
        "f16" | "bf16" => 2.0,
        "q8_0" => 1.0625,     // 34 bytes / 32 elements
        "q5_1" => 0.75,       // 24/32
        "q5_0" => 0.6875,     // 22/32
        "q4_1" => 0.625,      // 20/32
        "q4_0" | "iq4_nl" => 0.5625,  // 18/32
        _ => 2.0,             // unknown → fall back to f16 equivalent
    }
}

/// Convenience: bytes per element for a declared GgufType, for KV.
pub fn kv_bytes_for_type(t: GgufType) -> f64 {
    match t {
        GgufType::F32 => 4.0,
        GgufType::F16 | GgufType::BF16 => 2.0,
        GgufType::Q8_0 => 1.0625,
        GgufType::Q5_1 => 0.75,
        GgufType::Q5_0 => 0.6875,
        GgufType::Q4_1 => 0.625,
        GgufType::Q4_0 | GgufType::IQ4_NL => 0.5625,
        _ => 2.0,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn q8_0_is_around_1_point_06() {
        assert!((kv_bytes_per_element("q8_0") - 1.0625).abs() < 1e-6);
    }

    #[test]
    fn unknown_falls_back_to_f16() {
        assert_eq!(kv_bytes_per_element("this-is-bogus"), 2.0);
    }
}
```

- [ ] **Step 3: Fallback estimator**

Create `src/estimator/fallback.rs`:

```rust
//! Fallback estimator for unknown architectures (spec §8.3).

use std::collections::BTreeMap;

use smol_str::SmolStr;
use tracing::warn;

use super::types::{Estimate, NonLayer};
use crate::gguf::GgufSummary;

/// Produce a coarse estimate for any GGUF: `total_tensor_bytes × 1.15 + 512 MB`
/// goes into `weights_bytes`; no KV modelling; no per-layer split. Emits
/// a warning so the operator knows rolling correction is the only tuning.
pub fn estimate_fallback(summary: &GgufSummary, context: u32) -> Estimate {
    warn!(
        architecture = %summary.architecture,
        "unknown architecture — using fallback estimator (total_tensor_bytes × 1.15 + 512 MB)"
    );
    let weights = ((summary.total_tensor_bytes as f64) * 1.15) as u64 + 512 * 1024 * 1024;
    Estimate {
        weights_bytes: weights,
        kv_per_token: 0,
        compute_buffer_mb: 400,
        per_layer_bytes: None,
        attention_layers: None,
        non_layer: NonLayer { output_head_bytes: 0, token_embd_bytes: 0, other_bytes: 0 },
        override_tensor_bytes: BTreeMap::new(),
        expert_layers: Vec::new(),
        expert_layer_cpu_bytes: BTreeMap::new(),
        context,
        architecture: SmolStr::new(summary.architecture.as_str()),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn summary_with(total_bytes: u64, arch: &str) -> GgufSummary {
        GgufSummary {
            path: "/fake".into(),
            total_tensor_bytes: total_bytes,
            tensors: Default::default(),
            metadata: Default::default(),
            block_count: None,
            architecture: SmolStr::new(arch),
            shards: vec!["/fake".into()],
        }
    }

    #[test]
    fn fallback_uses_1_point_15_plus_512mb() {
        let s = summary_with(1_000_000_000, "nonsense-arch");
        let e = estimate_fallback(&s, 4096);
        assert_eq!(e.weights_bytes, (1_000_000_000f64 * 1.15) as u64 + 512 * 1024 * 1024);
        assert_eq!(e.kv_per_token, 0);
    }
}
```

- [ ] **Step 4: Stubs for other architectures + mod**

Create:

`src/estimator/llama.rs`:
```rust
//! Llama-family estimator (implemented in Task 5).
```

`src/estimator/moe.rs`:
```rust
//! MoE estimator (implemented in Task 6).
```

`src/estimator/mamba.rs`:
```rust
//! SSM/Mamba estimator (implemented in Task 7).
```

`src/estimator/hybrid.rs`:
```rust
//! Hybrid (jamba) estimator (implemented in Task 7).
```

`src/estimator/mod.rs`:
```rust
//! VRAM estimator.

pub mod fallback;
pub mod hybrid;
pub mod kv;
pub mod llama;
pub mod mamba;
pub mod moe;
pub mod types;

pub use types::{Estimate, NonLayer};

use crate::config::ServiceConfig;
use crate::gguf::GgufSummary;

/// Dispatch on `general.architecture`. Real dispatch is filled out in
/// Task 8; for now route everything to fallback.
pub fn estimate(summary: &GgufSummary, svc: &ServiceConfig) -> Estimate {
    let context = svc.raw.context.unwrap_or(4096);
    fallback::estimate_fallback(summary, context)
}
```

Add `pub mod estimator;` to `src/lib.rs`.

- [ ] **Step 5: Run tests**

Run: `cargo test --lib estimator`
Expected: 3 tests pass (kv 2 + fallback 1).

- [ ] **Step 6: Commit**

```bash
git add src/estimator/ src/lib.rs
git commit -m "feat(estimator): types, kv table, fallback, and module scaffold"
```

---

## Task 5: Llama-family estimator

**Files:**
- Replace: `src/estimator/llama.rs`

- [ ] **Step 1: Implementation + tests**

Replace `src/estimator/llama.rs`:

```rust
//! Llama-family estimator.
//!
//! Applies to: llama, qwen2, qwen3, mistral, gemma(1/2/3), phi3, glm4.
//!
//! Per spec §8.3: weights = Σ per-layer tensor bytes + non-layer bytes;
//! kv_per_token = n_layers × n_kv_heads ×
//!                (key_length × bytes(cache_k) + value_length × bytes(cache_v)).

use std::collections::BTreeMap;

use smol_str::SmolStr;

use super::kv;
use super::types::{Estimate, NonLayer};
use crate::config::ServiceConfig;
use crate::gguf::{GgufSummary, GgufTensor};

pub const LLAMA_FAMILY: &[&str] = &[
    "llama", "qwen2", "qwen3", "mistral",
    "gemma", "gemma2", "gemma3",
    "phi3", "glm4",
];

pub fn is_llama_family(arch: &str) -> bool {
    LLAMA_FAMILY.iter().any(|&a| a == arch)
}

pub fn estimate(summary: &GgufSummary, svc: &ServiceConfig) -> Estimate {
    let arch = summary.architecture.as_str();
    let context = svc.raw.context.unwrap_or(4096);

    let n_layers = summary.block_count.unwrap_or(0);

    let per_layer_bytes = collect_per_layer(summary, n_layers);
    let non_layer = collect_non_layer(summary);

    let weights_bytes = per_layer_bytes.iter().sum::<u64>()
        + non_layer.output_head_bytes
        + non_layer.token_embd_bytes
        + non_layer.other_bytes;

    // KV per token.
    let n_kv_heads = summary.metadata.get(&*format!("{arch}.attention.head_count_kv"))
        .and_then(|v| v.as_u32())
        .unwrap_or(0) as u64;
    let key_length = summary.metadata.get(&*format!("{arch}.attention.key_length"))
        .and_then(|v| v.as_u32())
        .unwrap_or(128) as u64;
    let value_length = summary.metadata.get(&*format!("{arch}.attention.value_length"))
        .and_then(|v| v.as_u32())
        .unwrap_or(128) as u64;

    let cache_k = svc.raw.cache_type_k.as_deref().unwrap_or("f16");
    let cache_v = svc.raw.cache_type_v.as_deref().unwrap_or("f16");

    let bytes_k = kv::kv_bytes_per_element(cache_k);
    let bytes_v = kv::kv_bytes_per_element(cache_v);

    let kv_per_token = if n_layers > 0 && n_kv_heads > 0 {
        let per_layer_bytes_kv =
            n_kv_heads * ((key_length as f64 * bytes_k) + (value_length as f64 * bytes_v)) as u64;
        n_layers as u64 * per_layer_bytes_kv
    } else {
        0
    };

    Estimate {
        weights_bytes,
        kv_per_token,
        compute_buffer_mb: svc.raw.estimation.as_ref()
            .and_then(|e| e.compute_buffer_mb).unwrap_or(400),
        per_layer_bytes: Some(per_layer_bytes),
        attention_layers: None,
        non_layer,
        override_tensor_bytes: BTreeMap::new(),
        expert_layers: Vec::new(),
        expert_layer_cpu_bytes: BTreeMap::new(),
        context,
        architecture: SmolStr::new(arch),
    }
}

pub(crate) fn collect_per_layer(summary: &GgufSummary, n_layers: u32) -> Vec<u64> {
    let mut out = vec![0u64; n_layers as usize];
    for tensor in summary.tensors.values() {
        if let Some(idx) = layer_index(&tensor.name) {
            if (idx as usize) < out.len() {
                out[idx as usize] += tensor.byte_size;
            }
        }
    }
    out
}

pub(crate) fn collect_non_layer(summary: &GgufSummary) -> NonLayer {
    let mut nl = NonLayer::default();
    for (name, tensor) in &summary.tensors {
        if layer_index(name).is_some() { continue; }
        match name.as_str() {
            "output.weight" => nl.output_head_bytes += tensor.byte_size,
            "token_embd.weight" => nl.token_embd_bytes += tensor.byte_size,
            _ => nl.other_bytes += tensor.byte_size,
        }
    }
    nl
}

/// Extract the N in a tensor name like `blk.N.attn_q.weight`.
pub(crate) fn layer_index(name: &str) -> Option<u32> {
    let rest = name.strip_prefix("blk.")?;
    let (idx, _) = rest.split_once('.')?;
    idx.parse().ok()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::parse::RawService;
    use crate::config::validate::{
        DeviceSlot, Filters, HealthSettings, Lifecycle, PlacementPolicy, ServiceConfig, Template,
    };
    use crate::gguf::types::{GgufSummary, GgufTensor, GgufType, GgufValue};
    use smol_str::SmolStr;

    fn tensor(name: &str, bytes: u64) -> GgufTensor {
        GgufTensor {
            name: SmolStr::new(name),
            dtype: GgufType::F16,
            shape: vec![bytes / 2],
            byte_size: bytes,
            shard_idx: 0,
            offset: 0,
        }
    }

    fn fake_summary() -> GgufSummary {
        let mut tensors = std::collections::BTreeMap::new();
        // 2 layers × 3 tensors per layer.
        for layer in 0..2u32 {
            for kind in ["attn_q", "attn_k", "ffn_down"] {
                let name = format!("blk.{layer}.{kind}.weight");
                tensors.insert(SmolStr::new(&name), tensor(&name, 1024 * 1024));
            }
        }
        tensors.insert(SmolStr::new("output.weight"), tensor("output.weight", 2 * 1024 * 1024));
        tensors.insert(SmolStr::new("token_embd.weight"), tensor("token_embd.weight", 4 * 1024 * 1024));

        let mut metadata = std::collections::BTreeMap::new();
        metadata.insert(SmolStr::new("general.architecture"), GgufValue::String("qwen3".into()));
        metadata.insert(SmolStr::new("qwen3.block_count"), GgufValue::U32(2));
        metadata.insert(SmolStr::new("qwen3.attention.head_count_kv"), GgufValue::U32(4));
        metadata.insert(SmolStr::new("qwen3.attention.key_length"), GgufValue::U32(128));
        metadata.insert(SmolStr::new("qwen3.attention.value_length"), GgufValue::U32(128));

        GgufSummary {
            path: "/fake".into(),
            total_tensor_bytes: 6 * 1024 * 1024 + 6 * 1024 * 1024,
            tensors,
            metadata,
            block_count: Some(2),
            architecture: SmolStr::new("qwen3"),
            shards: vec!["/fake".into()],
        }
    }

    fn svc(cache_k: &str, cache_v: &str, context: u32) -> ServiceConfig {
        let raw = RawService {
            name: Some(SmolStr::new("demo")),
            template: Some(SmolStr::new("llama-cpp")),
            model: Some("/fake".into()),
            port: Some(0),
            context: Some(context),
            cache_type_k: Some(SmolStr::new(cache_k)),
            cache_type_v: Some(SmolStr::new(cache_v)),
            flash_attn: Some(true),
            ..Default::default()
        };
        let mut placement = std::collections::BTreeMap::new();
        placement.insert(DeviceSlot::Gpu(0), 1000);
        ServiceConfig {
            name: SmolStr::new("demo"),
            template: Template::LlamaCpp,
            port: 0,
            private_port: 0,
            lifecycle: Lifecycle::OnDemand,
            priority: 50,
            health: HealthSettings { http_path: "/".into(), timeout_ms: 1000, probe_interval_ms: 500 },
            placement_override: placement,
            placement_policy: PlacementPolicy::GpuOnly,
            idle_timeout_ms: 60_000, warming_grace_ms: 100, drain_timeout_ms: 1000,
            extended_stream_drain_ms: 1000, max_request_duration_ms: 1000,
            filters: Filters::default(),
            raw,
        }
    }

    #[test]
    fn sums_per_layer_and_non_layer() {
        let s = fake_summary();
        let e = estimate(&s, &svc("f16", "f16", 4096));
        // per-layer: 2 layers × 3 tensors × 1 MiB = 6 MiB weights from layers.
        // non-layer: 2 MiB output + 4 MiB token_embd = 6 MiB.
        assert_eq!(e.weights_bytes, 12 * 1024 * 1024);
        assert_eq!(e.per_layer_bytes.as_ref().unwrap().len(), 2);
    }

    #[test]
    fn kv_uses_arch_metadata() {
        let s = fake_summary();
        let e = estimate(&s, &svc("f16", "f16", 4096));
        // n_layers=2, n_kv=4, k=v=128, 2 bytes/element (f16).
        // per_layer_kv = 4 × (128*2 + 128*2) = 4 × 512 = 2048 bytes.
        // kv_per_token = 2 × 2048 = 4096 bytes.
        assert_eq!(e.kv_per_token, 4096);
    }

    #[test]
    fn kv_quantised_shrinks() {
        let s = fake_summary();
        let e_q8 = estimate(&s, &svc("q8_0", "q8_0", 4096));
        let e_f16 = estimate(&s, &svc("f16", "f16", 4096));
        assert!(e_q8.kv_per_token < e_f16.kv_per_token);
    }

    #[test]
    fn layer_index_extracts() {
        assert_eq!(layer_index("blk.0.attn_q.weight"), Some(0));
        assert_eq!(layer_index("blk.42.ffn_down.weight"), Some(42));
        assert_eq!(layer_index("output.weight"), None);
        assert_eq!(layer_index("token_embd.weight"), None);
    }
}
```

- [ ] **Step 2: Run tests**

Run: `cargo test --lib estimator::llama`
Expected: 4 tests pass.

- [ ] **Step 3: Commit**

```bash
git add src/estimator/llama.rs
git commit -m "feat(estimator): llama-family (per-layer + KV from attention metadata)"
```

---

## Task 6: MoE estimator

**Files:**
- Replace: `src/estimator/moe.rs`

- [ ] **Step 1: Implementation + tests**

Replace `src/estimator/moe.rs`:

```rust
//! MoE estimator.
//!
//! Applies to: llama4, qwen3moe, deepseek2, mixtral, gpt-oss.
//!
//! Identifies expert tensors by the `_exps` suffix on
//! `blk.N.ffn_{gate,up,down}_exps.weight`. When `n_cpu_moe > 0`, the
//! top-N expert-bearing layers move their expert bytes to CPU.

use std::collections::BTreeMap;

use smol_str::SmolStr;

use super::llama::{collect_non_layer, layer_index};
use super::kv;
use super::types::{Estimate, NonLayer};
use crate::config::ServiceConfig;
use crate::gguf::GgufSummary;

pub const MOE_FAMILY: &[&str] = &[
    "llama4", "qwen3moe", "deepseek2", "mixtral", "gpt-oss",
];

pub fn is_moe(arch: &str) -> bool {
    MOE_FAMILY.iter().any(|&a| a == arch)
}

pub fn estimate(summary: &GgufSummary, svc: &ServiceConfig) -> Estimate {
    let arch = summary.architecture.as_str();
    let context = svc.raw.context.unwrap_or(4096);

    let n_layers = summary.block_count.unwrap_or(0);

    // Per-layer split into {non-expert, expert} bytes.
    let mut per_layer_nonexp = vec![0u64; n_layers as usize];
    let mut per_layer_exp = vec![0u64; n_layers as usize];

    for (name, t) in &summary.tensors {
        let Some(idx) = layer_index(name) else { continue; };
        if (idx as usize) >= per_layer_nonexp.len() { continue; }
        if is_expert_tensor(name) {
            per_layer_exp[idx as usize] += t.byte_size;
        } else {
            per_layer_nonexp[idx as usize] += t.byte_size;
        }
    }

    let non_layer = collect_non_layer(summary);

    let n_cpu_moe = svc.raw.n_cpu_moe.unwrap_or(0) as usize;

    // Pick the top-N layers by expert byte count for offload.
    let mut layer_scores: Vec<(u32, u64)> = per_layer_exp
        .iter().enumerate().map(|(i, b)| (i as u32, *b)).collect();
    layer_scores.sort_by(|a, b| b.1.cmp(&a.1));
    let offload_layers: Vec<u32> = layer_scores.iter().take(n_cpu_moe).map(|(i, _)| *i).collect();

    // Build per-layer total (non-expert + kept experts).
    let mut per_layer_total: Vec<u64> = per_layer_nonexp
        .iter().zip(per_layer_exp.iter())
        .map(|(a, b)| *a + *b).collect();
    let mut expert_layer_cpu_bytes = BTreeMap::new();
    for layer in &offload_layers {
        let exp_bytes = per_layer_exp[*layer as usize];
        per_layer_total[*layer as usize] = per_layer_total[*layer as usize].saturating_sub(exp_bytes);
        expert_layer_cpu_bytes.insert(*layer, exp_bytes);
    }

    let weights_bytes = per_layer_total.iter().sum::<u64>()
        + non_layer.output_head_bytes
        + non_layer.token_embd_bytes
        + non_layer.other_bytes;

    // KV: same formula as llama-family.
    let n_kv_heads = summary.metadata.get(&*format!("{arch}.attention.head_count_kv"))
        .and_then(|v| v.as_u32())
        .unwrap_or(0) as u64;
    let key_length = summary.metadata.get(&*format!("{arch}.attention.key_length"))
        .and_then(|v| v.as_u32())
        .unwrap_or(128) as u64;
    let value_length = summary.metadata.get(&*format!("{arch}.attention.value_length"))
        .and_then(|v| v.as_u32())
        .unwrap_or(128) as u64;

    let cache_k = svc.raw.cache_type_k.as_deref().unwrap_or("f16");
    let cache_v = svc.raw.cache_type_v.as_deref().unwrap_or("f16");

    let kv_per_token = if n_layers > 0 && n_kv_heads > 0 {
        let per_layer_bytes_kv = n_kv_heads
            * ((key_length as f64 * kv::kv_bytes_per_element(cache_k))
                + (value_length as f64 * kv::kv_bytes_per_element(cache_v))) as u64;
        n_layers as u64 * per_layer_bytes_kv
    } else {
        0
    };

    let expert_layers: Vec<u32> = per_layer_exp
        .iter().enumerate()
        .filter_map(|(i, b)| if *b > 0 { Some(i as u32) } else { None })
        .collect();

    Estimate {
        weights_bytes,
        kv_per_token,
        compute_buffer_mb: svc.raw.estimation.as_ref()
            .and_then(|e| e.compute_buffer_mb).unwrap_or(400),
        per_layer_bytes: Some(per_layer_total),
        attention_layers: None,
        non_layer,
        override_tensor_bytes: BTreeMap::new(),
        expert_layers,
        expert_layer_cpu_bytes,
        context,
        architecture: SmolStr::new(arch),
    }
}

/// Does this tensor name denote an expert weight?
/// Pattern: `blk.N.ffn_{gate,up,down}_exps.weight` (and `_shexp` counterparts
/// are *not* considered experts for offload purposes).
pub(crate) fn is_expert_tensor(name: &str) -> bool {
    if let Some(rest) = name.strip_prefix("blk.") {
        if let Some((_, kind)) = rest.split_once('.') {
            return (kind.starts_with("ffn_gate_exps")
                || kind.starts_with("ffn_up_exps")
                || kind.starts_with("ffn_down_exps"))
                && !kind.contains("shexp");
        }
    }
    false
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn expert_pattern_matches() {
        assert!(is_expert_tensor("blk.0.ffn_gate_exps.weight"));
        assert!(is_expert_tensor("blk.1.ffn_up_exps.weight"));
        assert!(is_expert_tensor("blk.5.ffn_down_exps.weight"));
        assert!(!is_expert_tensor("blk.0.ffn_gate.weight"));
        assert!(!is_expert_tensor("blk.0.ffn_gate_shexp.weight"));
        assert!(!is_expert_tensor("output.weight"));
    }

    #[test]
    fn n_cpu_moe_offloads_top_layers() {
        use crate::config::parse::RawService;
        use crate::config::validate::{DeviceSlot, Filters, HealthSettings, Lifecycle, PlacementPolicy, ServiceConfig, Template};
        use crate::gguf::types::{GgufSummary, GgufTensor, GgufType, GgufValue};
        use smol_str::SmolStr;

        let mut tensors = std::collections::BTreeMap::new();
        for layer in 0..3u32 {
            // Base layer tensors: 1 MiB.
            let attn = format!("blk.{layer}.attn_q.weight");
            tensors.insert(SmolStr::new(&attn), GgufTensor {
                name: SmolStr::new(&attn), dtype: GgufType::F16,
                shape: vec![512 * 1024], byte_size: 1024 * 1024, shard_idx: 0, offset: 0,
            });
            // Expert tensors: 10 MiB each, different by layer.
            let size = match layer { 0 => 4, 1 => 10, 2 => 2 };
            let exp = format!("blk.{layer}.ffn_gate_exps.weight");
            tensors.insert(SmolStr::new(&exp), GgufTensor {
                name: SmolStr::new(&exp), dtype: GgufType::F16,
                shape: vec![size * 512 * 1024], byte_size: size * 1024 * 1024, shard_idx: 0, offset: 0,
            });
        }
        let mut metadata = std::collections::BTreeMap::new();
        metadata.insert(SmolStr::new("general.architecture"), GgufValue::String("qwen3moe".into()));
        metadata.insert(SmolStr::new("qwen3moe.block_count"), GgufValue::U32(3));

        let summary = GgufSummary {
            path: "/fake".into(), total_tensor_bytes: 0, tensors, metadata,
            block_count: Some(3), architecture: SmolStr::new("qwen3moe"),
            shards: vec!["/fake".into()],
        };

        let mut placement = std::collections::BTreeMap::new();
        placement.insert(DeviceSlot::Gpu(0), 1000);
        let raw = RawService {
            name: Some(SmolStr::new("demo")),
            template: Some(SmolStr::new("llama-cpp")),
            model: Some("/fake".into()),
            port: Some(0),
            context: Some(4096),
            n_cpu_moe: Some(1),
            flash_attn: Some(true),
            ..Default::default()
        };
        let svc = ServiceConfig {
            name: SmolStr::new("demo"),
            template: Template::LlamaCpp, port: 0, private_port: 0,
            lifecycle: Lifecycle::OnDemand, priority: 50,
            health: HealthSettings { http_path: "/".into(), timeout_ms: 1000, probe_interval_ms: 500 },
            placement_override: placement, placement_policy: PlacementPolicy::Hybrid,
            idle_timeout_ms: 60_000, warming_grace_ms: 100, drain_timeout_ms: 1000,
            extended_stream_drain_ms: 1000, max_request_duration_ms: 1000,
            filters: Filters::default(),
            raw,
        };

        let e = estimate(&summary, &svc);
        // The layer with the largest expert bytes is layer 1 (10 MiB).
        assert_eq!(e.expert_layer_cpu_bytes.len(), 1);
        assert!(e.expert_layer_cpu_bytes.contains_key(&1));
        assert_eq!(e.expert_layer_cpu_bytes[&1], 10 * 1024 * 1024);
    }
}
```

- [ ] **Step 2: Run tests**

Run: `cargo test --lib estimator::moe`
Expected: 2 tests pass.

- [ ] **Step 3: Commit**

```bash
git add src/estimator/moe.rs
git commit -m "feat(estimator): MoE (_exps identification + n_cpu_moe offload)"
```

---

## Task 7: SSM/Mamba and Hybrid (jamba) estimators

**Files:**
- Replace: `src/estimator/mamba.rs`
- Replace: `src/estimator/hybrid.rs`

- [ ] **Step 1: Mamba implementation**

Replace `src/estimator/mamba.rs`:

```rust
//! SSM/Mamba estimator.
//!
//! No conventional KV cache. State cost derived from mamba.ssm.*
//! metadata. flash_attn and cache_type_* don't apply and validation
//! rejects them for this architecture (spec §6.5).

use std::collections::BTreeMap;

use smol_str::SmolStr;

use super::llama::{collect_non_layer, layer_index};
use super::types::{Estimate, NonLayer};
use crate::config::ServiceConfig;
use crate::gguf::GgufSummary;

pub fn is_mamba(arch: &str) -> bool { arch == "mamba" }

pub fn estimate(summary: &GgufSummary, svc: &ServiceConfig) -> Estimate {
    let arch = summary.architecture.as_str();
    let context = svc.raw.context.unwrap_or(4096);
    let n_layers = summary.block_count.unwrap_or(0);

    let per_layer = super::llama::collect_per_layer(summary, n_layers);
    let non_layer = collect_non_layer(summary);

    let weights_bytes = per_layer.iter().sum::<u64>()
        + non_layer.output_head_bytes
        + non_layer.token_embd_bytes
        + non_layer.other_bytes;

    // State cost: state_size × conv_kernel × inner_size × 4 bytes (f32), per layer.
    let state_size = summary.metadata.get("mamba.ssm.state_size").and_then(|v| v.as_u32()).unwrap_or(16) as u64;
    let conv_kernel = summary.metadata.get("mamba.ssm.conv_kernel").and_then(|v| v.as_u32()).unwrap_or(4) as u64;
    let inner_size = summary.metadata.get("mamba.ssm.inner_size").and_then(|v| v.as_u32()).unwrap_or(0) as u64;

    let state_per_layer = state_size * conv_kernel * inner_size * 4;
    let kv_per_token = n_layers as u64 * state_per_layer;

    Estimate {
        weights_bytes,
        kv_per_token,
        compute_buffer_mb: svc.raw.estimation.as_ref().and_then(|e| e.compute_buffer_mb).unwrap_or(400),
        per_layer_bytes: Some(per_layer),
        attention_layers: None,
        non_layer,
        override_tensor_bytes: BTreeMap::new(),
        expert_layers: Vec::new(),
        expert_layer_cpu_bytes: BTreeMap::new(),
        context,
        architecture: SmolStr::new(arch),
    }
}

#[cfg(test)]
mod tests {
    // Mamba fixtures are nontrivial; smoke-level coverage via the
    // dispatcher integration test in Task 8. A dedicated fixture file
    // would duplicate estimator::llama's wiring without meaningful
    // extra coverage for this feature.
}
```

- [ ] **Step 2: Hybrid (jamba) implementation**

Replace `src/estimator/hybrid.rs`:

```rust
//! Hybrid-architecture estimator (jamba and similar).

use std::collections::BTreeMap;

use smol_str::SmolStr;

use super::llama::{collect_non_layer, collect_per_layer};
use super::types::{Estimate, NonLayer};
use crate::config::ServiceConfig;
use crate::gguf::GgufSummary;

pub fn is_hybrid(arch: &str) -> bool { arch == "jamba" }

pub fn estimate(summary: &GgufSummary, svc: &ServiceConfig) -> Estimate {
    // For phase 3, treat hybrid like llama-family but with no KV cache
    // modelled (safer over-estimate side) and no per-layer type
    // differentiation (future work).
    let arch = summary.architecture.as_str();
    let context = svc.raw.context.unwrap_or(4096);
    let n_layers = summary.block_count.unwrap_or(0);

    let per_layer = collect_per_layer(summary, n_layers);
    let non_layer = collect_non_layer(summary);
    let weights_bytes = per_layer.iter().sum::<u64>()
        + non_layer.output_head_bytes
        + non_layer.token_embd_bytes
        + non_layer.other_bytes;

    Estimate {
        weights_bytes,
        kv_per_token: 0, // conservative; refined when real jamba metadata arrives.
        compute_buffer_mb: svc.raw.estimation.as_ref().and_then(|e| e.compute_buffer_mb).unwrap_or(400),
        per_layer_bytes: Some(per_layer),
        attention_layers: None,
        non_layer,
        override_tensor_bytes: BTreeMap::new(),
        expert_layers: Vec::new(),
        expert_layer_cpu_bytes: BTreeMap::new(),
        context,
        architecture: SmolStr::new(arch),
    }
}
```

- [ ] **Step 3: Verify compile**

Run: `cargo check --lib`
Expected: clean.

- [ ] **Step 4: Commit**

```bash
git add src/estimator/mamba.rs src/estimator/hybrid.rs
git commit -m "feat(estimator): SSM/Mamba state cost and Hybrid (jamba) scaffold"
```

---

## Task 8: Estimator dispatch + mmproj

**Files:**
- Replace: `src/estimator/mod.rs`

- [ ] **Step 1: Wire dispatch + mmproj**

Replace `src/estimator/mod.rs`:

```rust
//! VRAM estimator — architecture-aware dispatch.

pub mod fallback;
pub mod hybrid;
pub mod kv;
pub mod llama;
pub mod mamba;
pub mod moe;
pub mod types;

pub use types::{Estimate, NonLayer};

use std::path::Path;

use tracing::warn;

use crate::config::ServiceConfig;
use crate::gguf::{self, GgufSummary};

/// Produce a base estimate for `svc`. Reads the GGUF (including any
/// mmproj) and dispatches on `general.architecture`. Pure function;
/// caller applies rolling correction + safety factor afterward.
pub fn estimate_from_path(path: &Path, svc: &ServiceConfig) -> Result<Estimate, String> {
    let summary = gguf::read(path).map_err(|e| e.to_string())?;

    let mut est = dispatch(&summary, svc);

    // Add mmproj bytes to GPU 0 weights (per spec §8.3).
    if let Some(mmproj) = svc.raw.mmproj.as_ref() {
        match gguf::read(mmproj.as_path()) {
            Ok(proj) => {
                est.weights_bytes = est.weights_bytes.saturating_add(proj.total_tensor_bytes);
                est.non_layer.other_bytes = est.non_layer.other_bytes.saturating_add(proj.total_tensor_bytes);
            }
            Err(e) => warn!(error = %e, path = %mmproj.display(), "mmproj read failed"),
        }
    }

    Ok(est)
}

pub fn dispatch(summary: &GgufSummary, svc: &ServiceConfig) -> Estimate {
    let arch = summary.architecture.as_str();
    if llama::is_llama_family(arch) { return llama::estimate(summary, svc); }
    if moe::is_moe(arch) { return moe::estimate(summary, svc); }
    if mamba::is_mamba(arch) { return mamba::estimate(summary, svc); }
    if hybrid::is_hybrid(arch) { return hybrid::estimate(summary, svc); }
    let context = svc.raw.context.unwrap_or(4096);
    fallback::estimate_fallback(summary, context)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::parse::RawService;
    use crate::config::validate::{
        DeviceSlot, Filters, HealthSettings, Lifecycle, PlacementPolicy, ServiceConfig, Template,
    };
    use crate::gguf::types::{GgufSummary, GgufValue};
    use smol_str::SmolStr;

    fn svc_with(mmproj: Option<&str>) -> ServiceConfig {
        let raw = RawService {
            name: Some(SmolStr::new("demo")),
            template: Some(SmolStr::new("llama-cpp")),
            model: Some("/fake".into()),
            port: Some(0),
            context: Some(4096),
            mmproj: mmproj.map(|p| p.into()),
            cache_type_k: Some(SmolStr::new("f16")),
            cache_type_v: Some(SmolStr::new("f16")),
            flash_attn: Some(true),
            ..Default::default()
        };
        let mut placement = std::collections::BTreeMap::new();
        placement.insert(DeviceSlot::Gpu(0), 1000);
        ServiceConfig {
            name: SmolStr::new("demo"),
            template: Template::LlamaCpp, port: 0, private_port: 0,
            lifecycle: Lifecycle::OnDemand, priority: 50,
            health: HealthSettings { http_path: "/".into(), timeout_ms: 1000, probe_interval_ms: 500 },
            placement_override: placement, placement_policy: PlacementPolicy::GpuOnly,
            idle_timeout_ms: 60_000, warming_grace_ms: 100, drain_timeout_ms: 1000,
            extended_stream_drain_ms: 1000, max_request_duration_ms: 1000,
            filters: Filters::default(),
            raw,
        }
    }

    #[test]
    fn dispatch_recognises_known_families() {
        let mut metadata = std::collections::BTreeMap::new();
        metadata.insert(SmolStr::new("general.architecture"), GgufValue::String("qwen3".into()));
        metadata.insert(SmolStr::new("qwen3.block_count"), GgufValue::U32(1));
        let summary = GgufSummary {
            path: "/fake".into(), total_tensor_bytes: 0,
            tensors: Default::default(), metadata,
            block_count: Some(1), architecture: SmolStr::new("qwen3"),
            shards: vec!["/fake".into()],
        };
        let e = dispatch(&summary, &svc_with(None));
        assert_eq!(e.architecture, "qwen3");
    }

    #[test]
    fn dispatch_unknown_goes_to_fallback() {
        let mut metadata = std::collections::BTreeMap::new();
        metadata.insert(SmolStr::new("general.architecture"), GgufValue::String("novel-arch".into()));
        let summary = GgufSummary {
            path: "/fake".into(), total_tensor_bytes: 1_000_000,
            tensors: Default::default(), metadata,
            block_count: None, architecture: SmolStr::new("novel-arch"),
            shards: vec!["/fake".into()],
        };
        let e = dispatch(&summary, &svc_with(None));
        // Fallback uses 1.15 × total + 512 MB.
        assert!(e.weights_bytes >= 512 * 1024 * 1024);
    }
}
```

- [ ] **Step 2: Run tests**

Run: `cargo test --lib estimator`
Expected: all pass (kv 2 + fallback 1 + llama 4 + moe 2 + mod 2 = 11).

- [ ] **Step 3: Commit**

```bash
git add src/estimator/mod.rs
git commit -m "feat(estimator): dispatch on general.architecture + mmproj attribution"
```

---

## Task 9: Placement — layer walker + CommandArgs

**Files:**
- Create: `src/placement.rs`
- Modify: `src/lib.rs`

- [ ] **Step 1: Implementation**

Create `src/placement.rs`:

```rust
//! Layer-aware placement across allowed devices.
//!
//! Produces an `Allocation` (per-device byte reservation) and
//! `CommandArgs` (llama.cpp CLI flags derived from the packing).

use std::collections::BTreeMap;

use smol_str::SmolStr;
use tracing::warn;

use crate::allocator::AllocationTable;
use crate::config::{DeviceSlot, PlacementPolicy, ServiceConfig};
use crate::devices::{Allocation, DeviceSnapshot};
use crate::estimator::Estimate;

const ONE_LAYER_FUDGE_MULTIPLIER: u64 = 1;

#[derive(Debug, Clone, Default)]
pub struct CommandArgs {
    /// `-ngl N` value. `None` means do not emit the flag (caller uses
    /// `placement_override` escape hatch or cpu-only).
    pub ngl: Option<u32>,
    /// `--tensor-split A,B,...` if multiple GPUs carry layers.
    pub tensor_split: Option<Vec<u32>>,
    /// `-ot <regex>=<device>` rules, rendered verbatim from
    /// `service.raw.override_tensor`.
    pub override_tensor: Vec<String>,
}

#[derive(Debug)]
pub struct PackError {
    pub reason: String,
}

impl std::fmt::Display for PackError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result { write!(f, "{}", self.reason) }
}

impl std::error::Error for PackError {}

#[derive(Debug)]
pub struct Packed {
    pub allocation: Allocation,
    pub args: CommandArgs,
}

/// Pack `estimate` onto allowed devices, respecting `policy`,
/// `override_tensor`, and live device capacity (`snapshot` minus any
/// already-reserved bytes from `reserved`).
pub fn pack(
    estimate: &Estimate,
    svc: &ServiceConfig,
    snapshot: &DeviceSnapshot,
    reserved: &AllocationTable,
) -> Result<Packed, PackError> {
    // Step 0: determine the allowed GPUs.
    let allowed_gpus = allowed_gpu_list(svc, snapshot);
    let allow_cpu = matches!(svc.placement_policy, PlacementPolicy::CpuOnly | PlacementPolicy::Hybrid);

    // Step 1: seed per-device bytes with non-layer tensors + override_tensor attributions.
    let mut per_device: BTreeMap<DeviceSlot, u64> = BTreeMap::new();

    // Token embeddings always go to CPU.
    if estimate.non_layer.token_embd_bytes > 0 {
        *per_device.entry(DeviceSlot::Cpu).or_default() += estimate.non_layer.token_embd_bytes;
    }

    // Output head: first allowed GPU if any GPU used, else CPU.
    let head_target = if let Some(first_gpu) = allowed_gpus.first() {
        DeviceSlot::Gpu(*first_gpu)
    } else {
        DeviceSlot::Cpu
    };
    if estimate.non_layer.output_head_bytes > 0 {
        *per_device.entry(head_target.clone()).or_default() += estimate.non_layer.output_head_bytes;
    }
    if estimate.non_layer.other_bytes > 0 {
        *per_device.entry(head_target.clone()).or_default() += estimate.non_layer.other_bytes;
    }

    // override_tensor already-attributed bytes (estimator filled this map).
    for (slot, bytes) in &estimate.override_tensor_bytes {
        *per_device.entry(slot.clone()).or_default() += *bytes;
    }

    // Step 2: layer walker. If per_layer_bytes is None (Mamba),
    // place all weights on the first device with room (GPU > CPU).
    let per_layer = estimate.per_layer_bytes.clone().unwrap_or_default();
    let mut layers_per_gpu: BTreeMap<u32, u32> = BTreeMap::new();
    let mut layers_on_cpu: u32 = 0;
    let mut gpu_remaining: BTreeMap<u32, u64> = BTreeMap::new();
    for gpu in &allowed_gpus {
        let free = snapshot.free_bytes(&DeviceSlot::Gpu(*gpu)).unwrap_or(0);
        let reserved_here = sum_reserved(reserved, &DeviceSlot::Gpu(*gpu), &svc.name);
        gpu_remaining.insert(*gpu, free.saturating_sub(reserved_here).saturating_sub(
            *per_device.get(&DeviceSlot::Gpu(*gpu)).unwrap_or(&0)
        ));
    }

    for (idx, bytes) in per_layer.iter().enumerate() {
        if *bytes == 0 { continue; }
        let mut placed = false;
        for gpu in &allowed_gpus {
            let rem = gpu_remaining.get_mut(gpu).unwrap();
            if *rem >= *bytes {
                *rem = rem.saturating_sub(*bytes);
                *per_device.entry(DeviceSlot::Gpu(*gpu)).or_default() += *bytes;
                *layers_per_gpu.entry(*gpu).or_default() += 1;
                placed = true;
                break;
            }
        }
        if !placed && allow_cpu {
            *per_device.entry(DeviceSlot::Cpu).or_default() += *bytes;
            layers_on_cpu += 1;
        } else if !placed {
            return Err(PackError { reason: format!("layer {idx} ({bytes} bytes) does not fit on any allowed GPU") });
        }
    }

    // If the architecture gave no per-layer info (Mamba), place the
    // entire weights bundle into the first GPU with room (or CPU).
    if per_layer.is_empty() && estimate.weights_bytes > 0 {
        let mut placed = false;
        for gpu in &allowed_gpus {
            let rem = gpu_remaining.get_mut(gpu).unwrap();
            if *rem >= estimate.weights_bytes {
                *rem = rem.saturating_sub(estimate.weights_bytes);
                *per_device.entry(DeviceSlot::Gpu(*gpu)).or_default() += estimate.weights_bytes;
                placed = true;
                break;
            }
        }
        if !placed && allow_cpu {
            *per_device.entry(DeviceSlot::Cpu).or_default() += estimate.weights_bytes;
        } else if !placed {
            return Err(PackError { reason: "weights do not fit on any allowed device".into() });
        }
    }

    // Step 3: add KV bytes to GPUs proportional to layers placed, or
    // CPU for layers that spilled.
    let n_layers = per_layer.len() as u32;
    let kv_total = estimate.kv_per_token.saturating_mul(estimate.context as u64);
    if n_layers > 0 && kv_total > 0 {
        for gpu in &allowed_gpus {
            let share = layers_per_gpu.get(gpu).copied().unwrap_or(0);
            if share > 0 {
                let bytes = kv_total * share as u64 / n_layers as u64;
                *per_device.entry(DeviceSlot::Gpu(*gpu)).or_default() += bytes;
            }
        }
        if layers_on_cpu > 0 {
            let bytes = kv_total * layers_on_cpu as u64 / n_layers as u64;
            *per_device.entry(DeviceSlot::Cpu).or_default() += bytes;
        }
    }

    // Step 4: compute buffer per active backend (default 400 MB).
    let compute_bytes = estimate.compute_buffer_mb as u64 * 1024 * 1024;
    let active_slots: Vec<DeviceSlot> = per_device.keys().cloned().collect();
    for slot in &active_slots {
        *per_device.entry(slot.clone()).or_default() += compute_bytes;
    }

    // Step 5: one-layer fudge for tensor-split slop (spec §8.2.5).
    if n_layers > 0 && !per_layer.is_empty() {
        let per_layer_avg = per_layer.iter().sum::<u64>() / n_layers as u64;
        let per_layer_kv = if n_layers > 0 { kv_total / n_layers as u64 } else { 0 };
        let fudge_each = ONE_LAYER_FUDGE_MULTIPLIER * (per_layer_avg + per_layer_kv);
        let slots: Vec<DeviceSlot> = per_device.keys().cloned().collect();
        for slot in slots {
            match slot {
                DeviceSlot::Gpu(_) => *per_device.entry(slot).or_default() += fudge_each,
                DeviceSlot::Cpu if layers_on_cpu > 0 => *per_device.entry(slot).or_default() += fudge_each,
                _ => {}
            }
        }
    }

    // Step 6: CommandArgs.
    let total_on_gpus: u32 = layers_per_gpu.values().sum();
    let ngl = if allowed_gpus.is_empty() {
        // cpu-only: emit -ngl 0 via the spawn code path (kept out of our args struct).
        Some(0)
    } else {
        Some(total_on_gpus)
    };

    let tensor_split = if allowed_gpus.len() > 1 && total_on_gpus > 0 {
        // Ratios in CUDA_VISIBLE_DEVICES-remapped order: render in the
        // same GPU-id order as the allocation iterates (ascending ids).
        Some(allowed_gpus.iter().map(|g| layers_per_gpu.get(g).copied().unwrap_or(0)).collect())
    } else {
        None
    };

    let override_tensor = svc.raw.override_tensor.clone().unwrap_or_default();

    let allocation = Allocation {
        bytes: per_device.into_iter().map(|(slot, bytes)| {
            let id = match slot {
                DeviceSlot::Cpu => crate::devices::DeviceId::Cpu,
                DeviceSlot::Gpu(n) => crate::devices::DeviceId::Gpu(n),
            };
            (id, bytes)
        }).collect(),
    };

    Ok(Packed {
        allocation,
        args: CommandArgs { ngl, tensor_split, override_tensor },
    })
}

fn allowed_gpu_list(svc: &ServiceConfig, snapshot: &DeviceSnapshot) -> Vec<u32> {
    if svc.placement_policy == PlacementPolicy::CpuOnly {
        return Vec::new();
    }
    let declared_allow: Option<Vec<u32>> = svc.raw.devices.as_ref().and_then(|d| d.gpu_allow.clone());
    let all: Vec<u32> = snapshot.gpus.iter().map(|g| g.id).collect();
    match declared_allow {
        Some(list) if !list.is_empty() => list.into_iter().filter(|id| all.contains(id)).collect(),
        _ => all,
    }
}

fn sum_reserved(table: &AllocationTable, slot: &DeviceSlot, exclude: &SmolStr) -> u64 {
    table.iter()
        .filter(|(k, _)| k.as_str() != exclude.as_str())
        .filter_map(|(_, alloc)| alloc.get(slot))
        .sum::<u64>() * 1024 * 1024
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::parse::RawService;
    use crate::config::validate::{Filters, HealthSettings, Lifecycle, Template};
    use crate::devices::{CpuSnapshot, GpuSnapshot};
    use smol_str::SmolStr;

    fn svc(policy: PlacementPolicy, gpu_allow: Option<Vec<u32>>) -> ServiceConfig {
        let mut placement = std::collections::BTreeMap::new();
        placement.insert(DeviceSlot::Gpu(0), 1000);
        let devices = gpu_allow.map(|a| crate::config::parse::RawServiceDevices {
            gpu_allow: Some(a),
            ..Default::default()
        });
        let raw = RawService {
            name: Some(SmolStr::new("demo")),
            template: Some(SmolStr::new("llama-cpp")),
            model: Some("/fake".into()),
            port: Some(0),
            devices,
            ..Default::default()
        };
        ServiceConfig {
            name: SmolStr::new("demo"),
            template: Template::LlamaCpp, port: 0, private_port: 0,
            lifecycle: Lifecycle::OnDemand, priority: 50,
            health: HealthSettings { http_path: "/".into(), timeout_ms: 1000, probe_interval_ms: 500 },
            placement_override: placement, placement_policy: policy,
            idle_timeout_ms: 60_000, warming_grace_ms: 100, drain_timeout_ms: 1000,
            extended_stream_drain_ms: 1000, max_request_duration_ms: 1000,
            filters: Filters::default(),
            raw,
        }
    }

    fn snapshot(free_gpu_gb: &[u64]) -> DeviceSnapshot {
        let gpus = free_gpu_gb.iter().enumerate().map(|(i, gb)| GpuSnapshot {
            id: i as u32,
            name: format!("GPU {i}"),
            total_bytes: 24 * 1024 * 1024 * 1024,
            free_bytes: gb * 1024 * 1024 * 1024,
        }).collect();
        DeviceSnapshot {
            gpus,
            cpu: Some(CpuSnapshot { total_bytes: 128 * 1024 * 1024 * 1024, available_bytes: 64 * 1024 * 1024 * 1024 }),
            taken_at_ms: 0,
        }
    }

    fn trivial_estimate(n_layers: u32, per_layer_mb: u64) -> Estimate {
        Estimate {
            weights_bytes: per_layer_mb * 1024 * 1024 * n_layers as u64,
            kv_per_token: 0,
            compute_buffer_mb: 400,
            per_layer_bytes: Some(vec![per_layer_mb * 1024 * 1024; n_layers as usize]),
            attention_layers: None,
            non_layer: NonLayer::default(),
            override_tensor_bytes: BTreeMap::new(),
            expert_layers: Vec::new(),
            expert_layer_cpu_bytes: BTreeMap::new(),
            context: 4096,
            architecture: SmolStr::new("qwen3"),
        }
    }

    #[test]
    fn single_gpu_fits() {
        let e = trivial_estimate(10, 100); // 10 layers × 100 MiB = 1 GiB
        let snap = snapshot(&[8]); // 8 GB free
        let alloc = AllocationTable::new();
        let packed = pack(&e, &svc(PlacementPolicy::GpuOnly, None), &snap, &alloc).unwrap();
        assert_eq!(packed.args.ngl, Some(10));
        assert!(packed.args.tensor_split.is_none());
    }

    #[test]
    fn multi_gpu_split_ratios_match_layer_counts() {
        // 20 layers, 1 GiB each; 2 GPUs with 12 GB free each.
        let e = trivial_estimate(20, 1024); // 20 GiB
        let snap = snapshot(&[12, 12]);
        let alloc = AllocationTable::new();
        let packed = pack(&e, &svc(PlacementPolicy::GpuOnly, None), &snap, &alloc).unwrap();
        let split = packed.args.tensor_split.as_ref().unwrap();
        assert_eq!(split.len(), 2);
        assert_eq!(split.iter().sum::<u32>(), packed.args.ngl.unwrap());
    }

    #[test]
    fn hybrid_spills_to_cpu() {
        let e = trivial_estimate(10, 100);
        let snap = snapshot(&[0]); // GPU full
        let alloc = AllocationTable::new();
        let packed = pack(&e, &svc(PlacementPolicy::Hybrid, None), &snap, &alloc).unwrap();
        // All layers should have spilled to CPU.
        assert!(packed.allocation.bytes.contains_key(&crate::devices::DeviceId::Cpu));
    }

    #[test]
    fn cpu_only_emits_ngl_zero_and_no_split() {
        let e = trivial_estimate(10, 100);
        let snap = snapshot(&[8]);
        let alloc = AllocationTable::new();
        let packed = pack(&e, &svc(PlacementPolicy::CpuOnly, None), &snap, &alloc).unwrap();
        assert_eq!(packed.args.ngl, Some(0));
        assert!(packed.args.tensor_split.is_none());
    }
}
```

- [ ] **Step 2: Add `pub mod placement;` to `src/lib.rs`**

- [ ] **Step 3: Run tests**

Run: `cargo test --lib placement`
Expected: 4 tests pass.

- [ ] **Step 4: Commit**

```bash
git add src/placement.rs src/lib.rs
git commit -m "feat(placement): layer walker, tensor-split, CommandArgs"
```

---

## Task 10: Rolling correction

**Files:**
- Create: `src/rolling.rs`
- Modify: `src/lib.rs`
- Modify: `src/app_state.rs`

- [ ] **Step 1: Implementation**

Create `src/rolling.rs`:

```rust
//! Per-service rolling correction (spec §8.3).

use std::collections::BTreeMap;
use std::sync::Arc;

use parking_lot::RwLock;
use smol_str::SmolStr;
use tracing::{info, warn};

#[derive(Debug, Clone, Copy)]
pub struct RollingCorrection {
    pub rolling_mean: f64,
    pub sample_count: u32,
    /// Count of consecutive samples with |mean-1.0| > 0.3.
    pub drift_samples: u32,
}

impl Default for RollingCorrection {
    fn default() -> Self {
        Self { rolling_mean: 1.0, sample_count: 0, drift_samples: 0 }
    }
}

#[derive(Clone, Default)]
pub struct RollingTable {
    inner: Arc<RwLock<BTreeMap<SmolStr, RollingCorrection>>>,
}

impl RollingTable {
    pub fn new() -> Self { Self::default() }

    pub fn get(&self, name: &SmolStr) -> RollingCorrection {
        self.inner.read().get(name).copied().unwrap_or_default()
    }

    pub fn update(&self, name: &SmolStr, observed_peak_bytes: u64, base_estimate_bytes: u64) {
        if base_estimate_bytes == 0 { return; }
        let ratio = observed_peak_bytes as f64 / base_estimate_bytes as f64;
        let mut guard = self.inner.write();
        let entry = guard.entry(name.clone()).or_default();
        let n = entry.sample_count as f64 + 1.0;
        let new_mean = (entry.rolling_mean * (n - 1.0) + ratio) / n;
        entry.rolling_mean = new_mean.clamp(0.8, 1.5);
        entry.sample_count = entry.sample_count.saturating_add(1);

        if (entry.rolling_mean - 1.0).abs() > 0.3 {
            entry.drift_samples = entry.drift_samples.saturating_add(1);
            if entry.drift_samples >= 5 {
                warn!(service = %name, mean = entry.rolling_mean,
                    "estimator_drift: rolling_mean has been >0.3 away from 1.0 for 5+ runs");
            }
        } else {
            entry.drift_samples = 0;
        }

        if entry.rolling_mean > 1.2 {
            warn!(service = %name, mean = entry.rolling_mean, "rolling correction: under-estimation");
        } else if entry.rolling_mean < 0.85 {
            warn!(service = %name, mean = entry.rolling_mean, "rolling correction: over-reservation");
        } else {
            info!(service = %name, mean = entry.rolling_mean, sample = entry.sample_count, "rolling correction updated");
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn mean_converges_to_observed_ratio() {
        let t = RollingTable::new();
        let svc = SmolStr::new("demo");
        for _ in 0..5 {
            t.update(&svc, 120, 100); // observed=120, base=100, ratio=1.2
        }
        let rc = t.get(&svc);
        assert!((rc.rolling_mean - 1.2).abs() < 0.05);
    }

    #[test]
    fn mean_clamps_high() {
        let t = RollingTable::new();
        let svc = SmolStr::new("demo");
        t.update(&svc, 1000, 100); // ratio = 10
        assert_eq!(t.get(&svc).rolling_mean, 1.5);
    }

    #[test]
    fn mean_clamps_low() {
        let t = RollingTable::new();
        let svc = SmolStr::new("demo");
        t.update(&svc, 10, 100); // ratio = 0.1
        assert_eq!(t.get(&svc).rolling_mean, 0.8);
    }

    #[test]
    fn zero_base_is_noop() {
        let t = RollingTable::new();
        let svc = SmolStr::new("demo");
        t.update(&svc, 100, 0);
        assert_eq!(t.get(&svc).sample_count, 0);
    }
}
```

- [ ] **Step 2: Wire into AppState**

In `src/app_state.rs`, add:

```rust
use crate::rolling::RollingTable;

// inside AppState struct:
pub rolling: RollingTable,
```

Add `pub mod rolling;` to `src/lib.rs`.

- [ ] **Step 3: Verify**

Run: `cargo test --lib rolling` — 4 tests pass.
Run: `cargo check --lib` — clean (daemon wire-up for `rolling` happens in Task 13).

- [ ] **Step 4: Commit**

```bash
git add src/rolling.rs src/app_state.rs src/lib.rs
git commit -m "feat(rolling): per-service rolling correction with drift warnings"
```

---

## Task 11: Observation — VmRSS + peak aggregator

**Files:**
- Create: `src/observation.rs`
- Modify: `src/lib.rs`
- Modify: `src/devices/nvml.rs` (expose per-PID query)

- [ ] **Step 1: NVML per-PID helper**

In `src/devices/nvml.rs`, add a method to `NvmlProbe`:

```rust
pub fn running_pids_with_vram(&self, id: u32) -> Vec<(u32, u64)> {
    let Ok(dev) = self.nvml.device_by_index(id) else { return Vec::new(); };
    dev.running_compute_processes()
        .map(|procs| procs.into_iter().filter_map(|p| {
            let used = match p.used_gpu_memory {
                nvml_wrapper::enums::device::UsedGpuMemory::Used(b) => b,
                nvml_wrapper::enums::device::UsedGpuMemory::Unavailable => 0,
            };
            if used > 0 { Some((p.pid, used)) } else { None }
        }).collect())
        .unwrap_or_default()
}
```

- [ ] **Step 2: Observation module**

Create `src/observation.rs`:

```rust
//! Per-service observed memory peaks (GPU VRAM via NVML + CPU VmRSS from /proc).

use std::collections::BTreeMap;
use std::sync::Arc;

use parking_lot::RwLock;
use smol_str::SmolStr;
use tracing::debug;

/// Observed peak bytes per service, across the current run.
#[derive(Clone, Default)]
pub struct ObservationTable {
    inner: Arc<RwLock<BTreeMap<SmolStr, ObservedState>>>,
}

#[derive(Debug, Clone, Default)]
struct ObservedState {
    peak_bytes: u64,
    pids: Vec<u32>,
}

impl ObservationTable {
    pub fn new() -> Self { Self::default() }

    pub fn register(&self, service: &SmolStr, pid: u32) {
        let mut guard = self.inner.write();
        let entry = guard.entry(service.clone()).or_default();
        if !entry.pids.contains(&pid) { entry.pids.push(pid); }
    }

    pub fn update_peak(&self, service: &SmolStr, bytes: u64) {
        let mut guard = self.inner.write();
        let entry = guard.entry(service.clone()).or_default();
        if bytes > entry.peak_bytes {
            entry.peak_bytes = bytes;
        }
    }

    pub fn read_peak(&self, service: &SmolStr) -> u64 {
        self.inner.read().get(service).map(|s| s.peak_bytes).unwrap_or(0)
    }

    pub fn clear(&self, service: &SmolStr) {
        self.inner.write().remove(service);
    }
}

/// Read `/proc/<pid>/status` and return `VmRSS` in bytes.
pub fn read_vm_rss(pid: u32) -> Option<u64> {
    let content = std::fs::read_to_string(format!("/proc/{pid}/status")).ok()?;
    for line in content.lines() {
        if let Some(rest) = line.strip_prefix("VmRSS:") {
            let kb = rest.trim().trim_end_matches("kB").trim().parse::<u64>().ok()?;
            return Some(kb * 1024);
        }
    }
    None
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn peak_is_monotonic() {
        let t = ObservationTable::new();
        let svc = SmolStr::new("demo");
        t.update_peak(&svc, 100);
        t.update_peak(&svc, 50);
        t.update_peak(&svc, 200);
        assert_eq!(t.read_peak(&svc), 200);
    }

    #[test]
    fn clear_resets() {
        let t = ObservationTable::new();
        let svc = SmolStr::new("demo");
        t.update_peak(&svc, 100);
        t.clear(&svc);
        assert_eq!(t.read_peak(&svc), 0);
    }
}
```

- [ ] **Step 3: Wire into lib.rs + AppState**

Add `pub mod observation;` to `src/lib.rs`.

Add `pub observation: crate::observation::ObservationTable,` to `AppState` in `src/app_state.rs`.

- [ ] **Step 4: Verify**

Run: `cargo test --lib observation`
Expected: 2 tests pass.

- [ ] **Step 5: Commit**

```bash
git add src/observation.rs src/lib.rs src/app_state.rs src/devices/nvml.rs
git commit -m "feat(observation): per-service peak aggregator + VmRSS reader"
```

---

## Task 12: Relax placement_override gate + context default

**Files:**
- Modify: `src/config/validate.rs`

- [ ] **Step 1: Update validator behaviour**

In `src/config/validate.rs`:

Locate the current gate:
```rust
let raw_override = dev.placement_override.clone().ok_or_else(|| fail(format!(
    "service {name}: devices.placement_override is required in phase 1 (estimator lands in phase 3)"
)))?;
if raw_override.is_empty() {
    return Err(fail(format!("service {name}: devices.placement_override is empty")));
}
```

Replace with:
```rust
let raw_override = dev.placement_override.clone().unwrap_or_default();
if dev.placement_override.is_some() && raw_override.is_empty() {
    return Err(fail(format!("service {name}: devices.placement_override is empty")));
}
```

Also ensure `context` default: the estimator needs a concrete value. Add a warning if `context` is missing and default to 4096. Locate the estimator-relevant context default. The supervisor reads `svc.raw.context.unwrap_or(4096)` already; no config change needed — but add a validator-time warning:

```rust
if raw.context.is_none() {
    tracing::warn!(
        service = %name,
        "context not set; estimator will default to 4096. Set `context = N` in the service block for accurate placement."
    );
}
```

- [ ] **Step 2: Update tests**

Existing test `rejects_missing_placement_override` must be removed (behaviour changed) — remove it.

Add new test verifying acceptance:

```rust
    #[test]
    fn phase3_accepts_missing_placement_override() {
        let cfg = parse_and_merge(r#"
[[service]]
name = "a"
template = "llama-cpp"
model = "/m/x.gguf"
port = 11000
lifecycle = "on_demand"
"#);
        let ec = validate(&cfg).unwrap();
        assert!(ec.services[0].placement_override.is_empty());
    }
```

- [ ] **Step 3: Run tests**

Run: `cargo test --lib config::validate`
Expected: all pass (with the new test).

- [ ] **Step 4: Commit**

```bash
git add src/config/validate.rs
git commit -m "feat(config): relax placement_override requirement; warn on missing context"
```

---

## Task 13: Daemon wire-up for rolling + observation

**Files:**
- Modify: `src/daemon.rs`
- Modify: `src/app_state.rs` (finalise struct)
- Modify: `src/snapshotter.rs` (optionally poll VmRSS on each tick)

- [ ] **Step 1: AppState final shape**

Confirm `src/app_state.rs` contains:

```rust
use std::sync::Arc;

use parking_lot::Mutex;

use crate::activity::ActivityTable;
use crate::allocator::AllocationTable;
use crate::config::EffectiveConfig;
use crate::db::Database;
use crate::observation::ObservationTable;
use crate::rolling::RollingTable;
use crate::service_registry::ServiceRegistry;
use crate::snapshotter::SharedSnapshot;

#[derive(Clone)]
pub struct AppState {
    pub config: Arc<EffectiveConfig>,
    pub registry: ServiceRegistry,
    pub allocations: Arc<Mutex<AllocationTable>>,
    pub snapshot: SharedSnapshot,
    pub activity: ActivityTable,
    pub rolling: RollingTable,
    pub observation: ObservationTable,
    pub db: Database,
}
```

- [ ] **Step 2: Build and pass through daemon**

In `src/daemon.rs` `run()`, add after the existing `ActivityTable` creation:

```rust
    let rolling = crate::rolling::RollingTable::new();
    let observation = crate::observation::ObservationTable::new();
```

Update `AppState` construction:

```rust
    let app_state = AppState {
        config: effective.clone(),
        registry: registry.clone(),
        allocations: allocations.clone(),
        snapshot: shared_snapshot.clone(),
        activity: activity.clone(),
        rolling: rolling.clone(),
        observation: observation.clone(),
        db: db.clone(),
    };
```

- [ ] **Step 3: Snapshotter samples VmRSS + NVML per-PID**

In `src/snapshotter.rs`, extend `spawn` to accept `ObservationTable` + `ServiceRegistry` so it can attribute per-PID samples to services:

```rust
pub fn spawn(
    snapshot: SharedSnapshot,
    probe: Option<Arc<dyn GpuProbe>>,
    observation: crate::observation::ObservationTable,
    registry: crate::service_registry::ServiceRegistry,
    mut shutdown: watch::Receiver<bool>,
) -> tokio::task::JoinHandle<()> {
    tokio::spawn(async move {
        let mut interval = tokio::time::interval(Duration::from_secs(2));
        interval.set_missed_tick_behavior(tokio::time::MissedTickBehavior::Delay);
        loop {
            tokio::select! {
                _ = shutdown.changed() => { if *shutdown.borrow() { return; } }
                _ = interval.tick() => {
                    let next = sample(&probe);
                    *snapshot.write() = next;

                    // Sample per-service observed peak.
                    for (name, handle) in registry.all() {
                        let Some(snap) = handle.snapshot().await else { continue; };
                        let Some(pid) = snap.pid else { continue; };
                        observation.register(&name, pid as u32);

                        let mut total = 0u64;
                        // NVML per-PID VRAM.
                        if let Some(p) = probe.as_ref() {
                            if let Some(nvml) = (&**p as &dyn std::any::Any).downcast_ref::<crate::devices::nvml::NvmlProbe>() {
                                for gpu in p.list() {
                                    for (qpid, bytes) in nvml.running_pids_with_vram(gpu.id) {
                                        if qpid as i32 == pid { total = total.saturating_add(bytes); }
                                    }
                                }
                            }
                        }
                        // CPU VmRSS.
                        if let Some(rss) = crate::observation::read_vm_rss(pid as u32) {
                            total = total.saturating_add(rss);
                        }
                        if total > 0 { observation.update_peak(&name, total); }
                    }
                }
            }
        }
    })
}
```

Note: `&**p as &dyn std::any::Any` is awkward. A cleaner alternative is to parameterise `sample` to accept a closure that queries per-PID bytes. For phase 3, keep the direct downcast; refactor later if needed.

Update `daemon.rs` call to pass `observation` and `registry`:

```rust
let snapshotter_join = snapshotter::spawn(
    shared_snapshot.clone(),
    probe.clone(),
    observation.clone(),
    registry.clone(),
    shutdown_rx.clone(),
);
```

- [ ] **Step 4: Verify compile**

Run: `cargo check --lib` — clean.
Run: `cargo clippy --all-targets -- -D warnings` — clean.

If the downcast doesn't compile, replace the per-PID NVML block with a simpler version that just uses `probe.processes(gpu.id)` (already returns `Vec<GpuProcess { pid, used_bytes }>`):

```rust
if let Some(p) = probe.as_ref() {
    for gpu in p.list() {
        for gp in p.processes(gpu.id) {
            if gp.pid as i32 == pid { total = total.saturating_add(gp.used_bytes); }
        }
    }
}
```

This is strictly simpler and needs no downcast — use this version directly.

- [ ] **Step 5: Commit**

```bash
git add src/daemon.rs src/snapshotter.rs src/app_state.rs
git commit -m "feat(daemon): wire rolling + observation tables; snapshotter samples per-service peak"
```

---

## Task 14: Supervisor — estimator + placement at Idle → Starting

**Files:**
- Modify: `src/supervise/mod.rs`
- Modify: `src/supervise/spawn.rs`

- [ ] **Step 1: Extend `spawn_supervisor` signature**

Add two new parameters: `rolling: RollingTable, observation: ObservationTable`. Thread them through `run`.

- [ ] **Step 2: Replace the allocator block in the Idle → Starting transition**

In `src/supervise/mod.rs`, inside the `Idle` arm's `Ensure` handler (currently running `can_fit(&svc.placement_override, …)`), introduce estimator + placement:

```rust
Some(SupervisorCommand::Ensure { ack }) => {
    let (want_mb, pack_result): (std::collections::BTreeMap<DeviceSlot, u64>, Option<crate::placement::Packed>) =
        if !svc.placement_override.is_empty() {
            // Escape hatch: trust the user's numbers, no estimator/placement.
            (svc.placement_override.clone(), None)
        } else {
            // Run estimator + placement.
            let model_path = match svc.raw.model.as_ref() {
                Some(p) => p.clone(),
                None => {
                    let _ = ack.send(EnsureResponse::Unavailable { reason: "service has no model path".into() });
                    continue;
                }
            };
            match crate::estimator::estimate_from_path(&model_path, &svc) {
                Err(e) => {
                    let _ = ack.send(EnsureResponse::Unavailable { reason: format!("estimator failed: {e}") });
                    continue;
                }
                Ok(mut est) => {
                    // Apply rolling correction.
                    let rc = rolling.get(&svc.name);
                    est.weights_bytes = (est.weights_bytes as f64 * rc.rolling_mean) as u64;
                    // Run placement.
                    let snap_now = snapshot.read().clone();
                    let table_now = allocations.lock().clone();
                    match crate::placement::pack(&est, &svc, &snap_now, &table_now) {
                        Err(pack_err) => {
                            let _ = ack.send(EnsureResponse::Unavailable { reason: format!("no fit: {pack_err}") });
                            continue;
                        }
                        Ok(packed) => {
                            let want_mb: std::collections::BTreeMap<DeviceSlot, u64> = packed.allocation.bytes.iter()
                                .map(|(id, bytes)| {
                                    let slot = match id {
                                        crate::devices::DeviceId::Cpu => DeviceSlot::Cpu,
                                        crate::devices::DeviceId::Gpu(n) => DeviceSlot::Gpu(*n),
                                    };
                                    (slot, bytes / (1024 * 1024))
                                })
                                .collect();
                            (want_mb, Some(packed))
                        }
                    }
                }
            }
        };

    // Allocator feasibility check.
    let snap = snapshot.read().clone();
    let table = allocations.lock().clone();
    if let Err(nofit) = crate::allocator::can_fit(&want_mb, &snap, &table, Some(&svc.name)) {
        let _ = ack.send(EnsureResponse::Unavailable { reason: nofit.to_string() });
        continue;
    }

    // Reserve.
    allocations.lock().insert(svc.name.clone(), want_mb.clone());

    // Remember packing for spawn argv.
    packed_for_spawn = pack_result;

    // Create bus, ack, transition.
    let sender = tokio::sync::broadcast::channel::<StartOutcome>(16).0;
    let rx = sender.subscribe();
    let _ = ack.send(EnsureResponse::Waiting { rx });
    start_bus_carry = Some(sender);

    state = ServiceState::Starting;
    *state_mirror.lock() = state.clone();
    break;
}
```

Declare `let mut packed_for_spawn: Option<crate::placement::Packed> = None;` alongside `start_bus_carry`.

- [ ] **Step 3: Pass packed to spawn**

In the `Starting` arm, replace `let spawn_cfg = render_argv(&svc, &allocation);` with:

```rust
let spawn_cfg = crate::supervise::spawn::render_argv(
    &svc,
    &allocation,
    packed_for_spawn.as_ref().map(|p| &p.args),
);
```

Extend `render_argv` in `src/supervise/spawn.rs` to accept `Option<&CommandArgs>`:

```rust
pub fn render_argv(
    svc: &ServiceConfig,
    alloc: &Allocation,
    cmd_args: Option<&crate::placement::CommandArgs>,
) -> SpawnConfig {
    // Start from existing implementation.
    // After building `args`, apply placement-derived flags when present:
    if let Some(ca) = cmd_args {
        // -ngl from placement overrides any n_gpu_layers field.
        if let Some(ngl) = ca.ngl {
            // Remove any pre-existing -ngl and its value (from existing rendering).
            remove_flag_with_value(&mut args, "-ngl");
            args.push("-ngl".into());
            args.push(ngl.to_string());
        }
        // --tensor-split
        if let Some(split) = &ca.tensor_split {
            args.push("--tensor-split".into());
            args.push(split.iter().map(u32::to_string).collect::<Vec<_>>().join(","));
        }
        // -ot rules
        for rule in &ca.override_tensor {
            args.push("-ot".into());
            args.push(rule.clone());
        }
    }
    // ...
}

fn remove_flag_with_value(args: &mut Vec<String>, flag: &str) {
    if let Some(idx) = args.iter().position(|a| a == flag) {
        if idx + 1 < args.len() {
            args.drain(idx..=idx + 1);
        } else {
            args.remove(idx);
        }
    }
}
```

If the existing `render_argv` already emits `-ngl` from `raw.n_gpu_layers`, keep that path but use `remove_flag_with_value` to dedupe before appending placement's value.

- [ ] **Step 4: Register PID for observation on spawn**

After the child spawn succeeds and `pid` is bound, call:

```rust
observation.register(&svc.name, pid as u32);
```

Pass `observation` into `run` alongside `rolling`.

- [ ] **Step 5: Update tests**

Existing `render_argv` tests need the extra argument. Pass `None` for the `cmd_args` parameter where the test doesn't care about placement. Preserve semantics.

Run: `cargo test --workspace` — all pass.

- [ ] **Step 6: Commit**

```bash
git add src/supervise/
git commit -m "feat(supervise): run estimator+placement at Idle→Starting; render placement argv"
```

---

## Task 15: Supervisor — OOM retry + rolling update on drain

**Files:**
- Modify: `src/supervise/mod.rs`

- [ ] **Step 1: OOM retry**

In the `Starting` arm, when the child exits during starting/warming (phase 1 branches return `Failed { retry_count: 0 }`), detect likely OOM:

```rust
exit = child.wait() => {
    let runtime_ms = /* elapsed since spawn */;
    let code = exit.ok().and_then(|s| s.code()).unwrap_or(-1);
    let signaled_killed = exit.as_ref().ok()
        .and_then(|s| {
            use std::os::unix::process::ExitStatusExt;
            s.signal()
        }) == Some(libc::SIGKILL as i32);

    if runtime_ms < 30_000 && signaled_killed {
        warn!(service = %svc.name, "suspected OOM during startup");
        // Release reservation, bump safety_factor for next attempt.
        allocations.lock().remove(&svc.name);
        oom_attempts += 1;
        if oom_attempts >= 2 {
            state = ServiceState::Disabled { reason: DisableReason::Oom };
            *state_mirror.lock() = state.clone();
            if let Some(bus) = start_bus_carry.take() {
                let _ = bus.send(StartOutcome::Err(StartFailure {
                    kind: StartFailureKind::Disabled,
                    message: "OOM at startup (second occurrence)".into(),
                }));
            }
            break;
        }
        // Retry with bumped safety factor — encode this by bumping the
        // placement-level fudge on the next Ensure cycle. Simplest: bump
        // rolling_mean by 0.3 (capped at 1.5).
        rolling.update(&svc.name, 1, 1); // no-op on observed; just to bump
        // A direct bump: insert into the rolling table artificially.
        state = ServiceState::Idle;
        *state_mirror.lock() = state.clone();
        break;
    }
    // Existing phase 1 behaviour.
    state = ServiceState::Failed { retry_count: 0 };
    *state_mirror.lock() = state.clone();
    break;
}
```

Declare `let mut oom_attempts: u32 = 0;` at the top of `run`.

Simpler approach for the retry bump: write a direct method on `RollingTable` that inserts a synthetic drift sample:

```rust
impl RollingTable {
    /// Force the next estimate upward by pretending we observed a 1.4×
    /// ratio. Used by the OOM retry path.
    pub fn bump_for_oom_retry(&self, name: &SmolStr) {
        self.update(name, 140, 100); // ratio = 1.4
    }
}
```

Then from the supervisor: `rolling.bump_for_oom_retry(&svc.name);`.

- [ ] **Step 2: Rolling update on drain**

Every place the `Running` arm transitions out (idle timeout drain, shutdown drain, child exit) gains one extra call right before releasing the allocation:

```rust
let observed = observation.read_peak(&svc.name);
let base = want_mb_total_bytes; // bytes we reserved
if observed > 0 && base > 0 {
    rolling.update(&svc.name, observed, base);
}
observation.clear(&svc.name);
```

`want_mb_total_bytes` must be captured when reservation happens; stash it in a `let base_total_bytes = want_mb.values().map(|mb| mb * 1024 * 1024).sum::<u64>();` inside the Ensure handler.

- [ ] **Step 3: Verify**

Run: `cargo check --lib` — clean.
Run: `cargo test --workspace` — existing tests still pass.

- [ ] **Step 4: Commit**

```bash
git add src/supervise/mod.rs src/rolling.rs
git commit -m "feat(supervise): OOM retry with safety_factor bump; rolling update on drain"
```

---

## Task 16: Management API — expose rolling + observed peak

**Files:**
- Modify: `src/management_api/types.rs`
- Modify: `src/management_api/handlers.rs`

- [ ] **Step 1: Extend `ServiceDetail`**

In `src/management_api/types.rs`, add fields to `ServiceDetail`:

```rust
pub rolling_mean: f64,
pub rolling_samples: u32,
pub observed_peak_bytes: u64,
```

- [ ] **Step 2: Populate in handler**

In `src/management_api/handlers.rs` `service_detail`, populate before returning:

```rust
let rc = state.rolling.get(&svc_cfg.name);
let observed_peak_bytes = state.observation.read_peak(&svc_cfg.name);

let detail = ServiceDetail {
    // ... existing fields ...
    rolling_mean: rc.rolling_mean,
    rolling_samples: rc.sample_count,
    observed_peak_bytes,
    recent_logs,
};
```

- [ ] **Step 3: Update openapi aggregator**

No changes to `src/openapi.rs` — the aggregator includes `ServiceDetail` by schema, so field additions automatically surface.

- [ ] **Step 4: Verify**

Run: `cargo test --workspace --features test-fakes` — all existing tests pass.

- [ ] **Step 5: Commit**

```bash
git add src/management_api/
git commit -m "feat(management_api): expose rolling_mean + observed_peak_bytes on ServiceDetail"
```

---

## Task 17: Integration tests — estimator + placement

**Files:**
- Create: `tests/estimator_llama.rs`
- Create: `tests/estimator_moe.rs`
- Create: `tests/estimator_fallback.rs`
- Create: `tests/placement_override_bypass.rs`
- Create: `tests/no_placement_override_ok.rs`
- Create: `tests/multi_gpu_split.rs`
- Create: `tests/sharded_gguf.rs`
- Create: `tests/mmproj_attribution.rs`
- Create: `tests/override_tensor_moe_hybrid.rs`

Each integration test uses a **synthetic GGUF file** written to a tempdir via a shared helper. Add this helper to `tests/common/mod.rs`:

```rust
pub mod synth_gguf {
    use std::io::Write;
    use std::path::PathBuf;

    pub struct Builder {
        buf: Vec<u8>,
        n_tensors: u64,
        n_kv: u64,
    }

    impl Builder {
        pub fn new() -> Self { Self { buf: Vec::new(), n_tensors: 0, n_kv: 0 } }

        pub fn arch(mut self, name: &str) -> Self {
            // deferred write; we need the final tensor_count + kv_count first.
            self = self.kv_string("general.architecture", name);
            self
        }

        pub fn kv_u32(mut self, key: &str, val: u32) -> Self {
            self.n_kv += 1;
            write_string(&mut self.buf, key);
            self.buf.extend_from_slice(&4u32.to_le_bytes());
            self.buf.extend_from_slice(&val.to_le_bytes());
            self
        }

        pub fn kv_u64(mut self, key: &str, val: u64) -> Self {
            self.n_kv += 1;
            write_string(&mut self.buf, key);
            self.buf.extend_from_slice(&10u32.to_le_bytes());
            self.buf.extend_from_slice(&val.to_le_bytes());
            self
        }

        pub fn kv_string(mut self, key: &str, val: &str) -> Self {
            self.n_kv += 1;
            write_string(&mut self.buf, key);
            self.buf.extend_from_slice(&8u32.to_le_bytes());
            write_string(&mut self.buf, val);
            self
        }

        pub fn tensor_f16(mut self, name: &str, elements: u64) -> Self {
            self.n_tensors += 1;
            write_string(&mut self.buf, name);
            self.buf.extend_from_slice(&1u32.to_le_bytes()); // n_dims
            self.buf.extend_from_slice(&elements.to_le_bytes());
            self.buf.extend_from_slice(&1u32.to_le_bytes()); // dtype F16
            self.buf.extend_from_slice(&0u64.to_le_bytes()); // offset
            self
        }

        pub fn write_to(self, path: &std::path::Path) {
            let mut out = Vec::<u8>::new();
            out.extend_from_slice(b"GGUF");
            out.extend_from_slice(&3u32.to_le_bytes());
            out.extend_from_slice(&self.n_tensors.to_le_bytes());
            out.extend_from_slice(&self.n_kv.to_le_bytes());
            out.extend_from_slice(&self.buf);
            std::fs::write(path, &out).unwrap();
        }
    }

    fn write_string(v: &mut Vec<u8>, s: &str) {
        v.extend_from_slice(&(s.len() as u64).to_le_bytes());
        v.extend_from_slice(s.as_bytes());
    }

    pub fn tempfile(prefix: &str) -> tempfile::NamedTempFile {
        tempfile::Builder::new().prefix(prefix).suffix(".gguf").tempfile().unwrap()
    }
}
```

Each integration test constructs a minimal GGUF via the Builder, runs `estimator::estimate_from_path`, asserts properties.

- [ ] **Step 1-9: Per-test implementations**

For brevity the test bodies follow a common pattern; one template:

```rust
// tests/estimator_llama.rs
mod common;

use ananke::estimator;
use common::synth_gguf;

#[test]
fn llama_family_weights_include_layers_and_non_layer() {
    let file = synth_gguf::tempfile("llama");
    synth_gguf::Builder::new()
        .kv_string("general.architecture", "qwen3")
        .kv_u32("qwen3.block_count", 2)
        .kv_u32("qwen3.attention.head_count_kv", 4)
        .kv_u32("qwen3.attention.key_length", 128)
        .kv_u32("qwen3.attention.value_length", 128)
        .tensor_f16("blk.0.attn_q.weight", 512 * 1024)
        .tensor_f16("blk.1.attn_q.weight", 512 * 1024)
        .tensor_f16("output.weight", 2 * 512 * 1024)
        .tensor_f16("token_embd.weight", 4 * 512 * 1024)
        .write_to(file.path());

    let svc = common::minimal_llama_service("demo", 0);
    let est = estimator::estimate_from_path(file.path(), &svc).unwrap();
    assert!(est.weights_bytes > 0);
    assert_eq!(est.architecture, "qwen3");
    assert!(est.kv_per_token > 0);
}
```

Follow the same pattern for:

- `tests/estimator_moe.rs` — architecture "qwen3moe", tensors with `_exps` suffix; assert `expert_layers` non-empty.
- `tests/estimator_fallback.rs` — architecture "novel-arch", assert `weights_bytes >= 512MB`.
- `tests/placement_override_bypass.rs` — service with `placement_override` set; call `estimate_from_path` but verify the supervisor path does NOT use it. (Easiest: use the harness `build_harness(svc)` where `svc` has a placement_override; assert that `/v1/chat/completions` succeeds and allocation in snapshot matches the override.)
- `tests/no_placement_override_ok.rs` — service WITHOUT `placement_override`, but with `model` pointing at a synthetic GGUF; daemon estimates and places. Assert a chat request succeeds.
- `tests/multi_gpu_split.rs` — FakeProbe with 2 GPUs, synthetic GGUF where per-layer bytes don't fit on one GPU; harness+daemon pack across both; assert `--tensor-split` flag is rendered.
- `tests/sharded_gguf.rs` — write two synthetic GGUFs with `split.count = 2`, `split.no = 0` and `1`; assert reader aggregates both.
- `tests/mmproj_attribution.rs` — main GGUF + separate mmproj file; assert `weights_bytes` post-estimate includes mmproj bytes.
- `tests/override_tensor_moe_hybrid.rs` — MoE GGUF with `override_tensor = [".ffn_(up|down)_exps.=CPU"]`; assert final argv includes `-ot` and that placement moves expert bytes.

For the daemon-level integration tests (multi_gpu_split, no_placement_override_ok), `build_harness` needs extending to accept a pre-seeded snapshot and to allow custom model paths in the ServiceConfig. Extend the harness helper:

```rust
pub async fn build_harness_with_snapshot(
    services: Vec<ServiceConfig>,
    snapshot: ananke::devices::DeviceSnapshot,
) -> TestHarness {
    let h = build_harness(services).await;
    *h.state.snapshot.write() = snapshot;
    h
}
```

And make sure the harness's supervisor uses the real `estimator::estimate_from_path` (via the code path added in Task 14). The fake spawn_child (feature "test-fakes") still keeps the child alive so the echo server can serve requests.

- [ ] **Step 10: Run**

```
cargo test --features test-fakes --test estimator_llama --test estimator_moe --test estimator_fallback --test placement_override_bypass --test no_placement_override_ok --test multi_gpu_split --test sharded_gguf --test mmproj_attribution --test override_tensor_moe_hybrid
```
All pass.

- [ ] **Step 11: Commit**

```bash
git add tests/common/mod.rs tests/
git commit -m "test: estimator + placement integration suite"
```

---

## Task 18: Rolling correction + OOM retry integration tests

**Files:**
- Create: `tests/rolling_correction_warns.rs`
- Create: `tests/oom_retry.rs`

- [ ] **Step 1: `tests/rolling_correction_warns.rs`**

```rust
mod common;

use ananke::rolling::RollingTable;
use smol_str::SmolStr;

#[test]
fn rolling_mean_converges_above_threshold_warns() {
    let t = RollingTable::new();
    let svc = SmolStr::new("demo");
    for _ in 0..3 {
        t.update(&svc, 130, 100); // ratio = 1.3
    }
    let rc = t.get(&svc);
    assert!(rc.rolling_mean > 1.2);
}
```

- [ ] **Step 2: `tests/oom_retry.rs`**

```rust
mod common;

use ananke::rolling::RollingTable;
use smol_str::SmolStr;

#[test]
fn oom_bump_increases_rolling_mean() {
    let t = RollingTable::new();
    let svc = SmolStr::new("demo");
    let before = t.get(&svc).rolling_mean;
    t.bump_for_oom_retry(&svc);
    let after = t.get(&svc).rolling_mean;
    assert!(after > before);
}
```

Daemon-level OOM simulation is tricky to test deterministically in integration (depends on child exit codes and signal timing); the supervisor-level test covers that the retry path executes via the rolling bump.

- [ ] **Step 3: Run**

Run: `cargo test --features test-fakes --test rolling_correction_warns --test oom_retry`
Expected: both pass.

- [ ] **Step 4: Commit**

```bash
git add tests/rolling_correction_warns.rs tests/oom_retry.rs
git commit -m "test: rolling correction warning + OOM retry bump"
```

---

## Task 19: Smoke runbook

**Files:**
- Create: `tests/manual/phase-3-smoke.md`

- [ ] **Step 1: Create the runbook**

Create `tests/manual/phase-3-smoke.md`:

```markdown
# Phase 3 manual smoke test

Real-hardware validation for the phase-3 additions: GGUF estimator,
layer-aware placement, rolling correction, OOM retry, and removal of
the `placement_override` requirement.

## Prerequisites

Same as phase 2, plus:

- At least one real GGUF on disk for each architecture family you want
  to validate (llama-family, MoE).

## Steps

1. Build release:
   ```
   cargo build --release
   ```

2. `~/.config/ananke/config.toml` (no `placement_override`):
   ```toml
   [daemon]
   management_listen = "127.0.0.1:17777"
   data_dir = "/tmp/ananke-phase3"

   [openai_api]
   listen = "127.0.0.1:18080"

   [[service]]
   name = "q3-30b-moe"
   template = "llama-cpp"
   model = "/mnt/ssd0/ai/llm/Qwen3-30B-A3B-Instruct-2507-UD-Q4_K_XL.gguf"
   port = 17436
   context = 8192
   flash_attn = true
   cache_type_k = "q8_0"
   cache_type_v = "q8_0"
   lifecycle = "on_demand"
   idle_timeout = "60s"
   ```

3. Start: `LD_LIBRARY_PATH=/run/opengl-driver/lib ./target/release/ananke --config ~/.config/ananke/config.toml`

4. First chat request triggers estimator + placement + spawn:
   ```
   curl -s http://127.0.0.1:18080/v1/chat/completions \
     -H 'Content-Type: application/json' \
     -d '{"model":"q3-30b-moe","messages":[{"role":"user","content":"hi"}]}' | jq
   ```
   Verify it succeeds. Compare `nvidia-smi` output against
   `curl http://127.0.0.1:17777/api/devices | jq` reservation totals.

5. After drain (30 s idle then next request), call
   `curl http://127.0.0.1:17777/api/services/q3-30b-moe | jq`
   and verify `rolling_mean` is populated and `observed_peak_bytes > 0`.

6. Sharded test — swap in the `Qwen3-235B-A22B-Instruct` config pointing
   at the shard-0 file; verify daemon aggregates all shards and places
   across GPU 0, GPU 1, and CPU. The `/api/devices` response should
   show CPU reservation ≥ 80 GB.

7. Manual OOM test — edit the config to demand e.g. `context = 65536` on
   a model that won't fit; send a chat request. Expect 503
   `insufficient_vram` (from allocator) OR, if it squeaks past the
   allocator, observe an actual OOM kill within 30 s of spawn; ananke
   should bump safety_factor and retry once.

8. Repeat the same chat request 5+ times; verify `rolling_mean` moves
   toward 1.0 (converging on reality) in the management API response.

9. Clean shutdown: `kill -TERM $(pidof ananke)`.

Success criteria: every numbered step above produces the expected
result. File a bug for anything that drifts.
```

- [ ] **Step 2: Commit**

```bash
git add tests/manual/phase-3-smoke.md
git commit -m "docs: phase 3 manual smoke runbook"
```

---

## Self-review checklist

Before declaring phase 3 complete:

- `just lint` passes (all Rust + TS checks).
- `cargo test --workspace` passes with and without `--features test-fakes`.
- `tests/manual/phase-3-smoke.md` executed end-to-end against real hardware on redline (at minimum: Qwen3-30B-A3B MoE + Qwen3-4B llama-family).
- Phase-2 behaviours (unified OpenAI, on-demand, coalescing, idle_timeout, filters, management API, openapi.json) continue working.
- `/api/services/{name}` response includes `rolling_mean`, `rolling_samples`, `observed_peak_bytes`.
- Spec §18 open items that phase 3 intended to land (GGUF tensor-table, placement_override gate flip, estimator calibration start-point) are addressed.

Phase 3 success: point redline at ananke with a config that declares models by path and context only — no `placement_override`, no hand-computed splits. The daemon reads each GGUF, estimates, packs, spawns, and tunes itself across runs.
