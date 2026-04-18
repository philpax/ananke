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
#[allow(non_camel_case_types)]
pub enum GgufType {
    F32,
    F16,
    BF16,
    Q4_0,
    Q4_1,
    Q5_0,
    Q5_1,
    Q8_0,
    Q8_1,
    Q2K,
    Q3K,
    Q4K,
    Q5K,
    Q6K,
    Q8K,
    IQ2_XXS,
    IQ2_XS,
    IQ3_XXS,
    IQ1_S,
    IQ4_NL,
    IQ3_S,
    IQ2_S,
    IQ4_XS,
    IQ1_M,
    I8,
    I16,
    I32,
    I64,
    F64,
    Unknown(u32),
}

impl GgufType {
    pub fn from_u32(n: u32) -> Self {
        match n {
            0 => Self::F32,
            1 => Self::F16,
            2 => Self::Q4_0,
            3 => Self::Q4_1,
            6 => Self::Q5_0,
            7 => Self::Q5_1,
            8 => Self::Q8_0,
            9 => Self::Q8_1,
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
    U8(u8),
    I8(i8),
    U16(u16),
    I16(i16),
    U32(u32),
    I32(i32),
    U64(u64),
    I64(i64),
    F32(f32),
    F64(f64),
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
        match self {
            GgufValue::String(s) => Some(s),
            _ => None,
        }
    }
}
