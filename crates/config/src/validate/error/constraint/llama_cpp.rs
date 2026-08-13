//! Constraint violations specific to the llama-cpp template.

use std::fmt;

/// Structured reason for a llama-cpp template constraint violation.
#[derive(Debug, Clone, PartialEq, Eq)]
#[allow(missing_docs)]
pub enum LlamaCppReason {
    ModelMissing,
    SpecTypeWrongDialect {
        spec_type: String,
        expected: &'static str,
    },
    MlaOutOfRange {
        value: u32,
    },
    DsaRequiresF16Kv {
        key: &'static str,
        value: String,
    },
    AttnMaxBatchZero,
    QuantizedKvRequiresFlashAttn {
        key: &'static str,
        value: String,
    },
    DraftModelRequiresSpecType,
    LauncherEmpty,
    ExpertOffloadInvalid {
        value: String,
        expected: String,
    },
    NumaInvalid {
        value: String,
        expected: String,
    },
}

impl fmt::Display for LlamaCppReason {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::ModelMissing => write!(f, "template llama-cpp requires `model`"),
            Self::SpecTypeWrongDialect {
                spec_type,
                expected,
            } => write!(
                f,
                "spec_type `{spec_type}` uses the wrong dialect (expected {expected})"
            ),
            Self::MlaOutOfRange { value } => {
                write!(f, "runtime.mla={value} is invalid (ik_llama accepts 0-3)")
            }
            Self::DsaRequiresF16Kv { key, value } => {
                write!(f, "runtime.dsa=true requires f16 KV, but {key}={value}")
            }
            Self::AttnMaxBatchZero => {
                write!(f, "runtime.attn_max_batch must be > 0")
            }
            Self::QuantizedKvRequiresFlashAttn { key, value } => {
                write!(
                    f,
                    "{key}={value} requires flash_attn=true (llama.cpp requires FA for quantised KV)"
                )
            }
            Self::DraftModelRequiresSpecType => {
                write!(f, "draft_model requires spec_type to be set")
            }
            Self::LauncherEmpty => write!(f, "launcher is present but empty"),
            Self::ExpertOffloadInvalid { value, expected } => {
                write!(
                    f,
                    "expert_offload `{value}` is invalid (expected {expected}, or an integer layer count)"
                )
            }
            Self::NumaInvalid { value, expected } => {
                write!(f, "numa `{value}` is invalid (expected {expected})")
            }
        }
    }
}
