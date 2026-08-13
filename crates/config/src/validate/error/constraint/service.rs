//! Constraint violations in the placement, lifecycle, and split vocabulary
//! shared by every template.

use std::fmt;

/// Structured reason for a service-level constraint violation.
#[derive(Debug, Clone, PartialEq, Eq)]
#[allow(missing_docs)]
pub enum ServiceReason {
    LifecycleOneshotInvalid,
    LifecycleUnknown { value: String },
    ModalityUnknown { value: String },
    CpuOnlyWithGpuLayers { n_gpu_layers: i32 },
    PlacementUnknown { value: String },
    PlacementOverrideEmpty,
    PlacementOverrideKeyInvalid { key: String },
    PlacementOverrideZero { key: String },
    GpuOnlyWithCpuOverride,
    SplitUnknown { value: String, expected: String },
    ExpertOffloadConflictsShardedSplit { split: String },
    ExpertOffloadRequiresHybridPlacement,
    ShardedSplitRequiresGpuOnly { split: String },
    ShardedSplitLlamaCppOnly { split: String },
    ShardedSplitConflictsOverrideTensor { split: String },
    TensorSplitWeightsRequiresSharded,
}

impl fmt::Display for ServiceReason {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::LifecycleOneshotInvalid => {
                write!(
                    f,
                    "lifecycle `oneshot` is invalid in a [[service]] block (API-only)"
                )
            }
            Self::LifecycleUnknown { value } => write!(f, "unknown lifecycle `{value}`"),
            Self::ModalityUnknown { value } => {
                write!(f, "unknown modality `{value}` (valid: `chat`, `embedding`)")
            }
            Self::CpuOnlyWithGpuLayers { n_gpu_layers } => {
                write!(
                    f,
                    "devices.placement=cpu-only with n_gpu_layers={n_gpu_layers} is invalid"
                )
            }
            Self::PlacementUnknown { value } => write!(f, "unknown placement `{value}`"),
            Self::PlacementOverrideEmpty => {
                write!(f, "devices.placement_override is empty")
            }
            Self::PlacementOverrideKeyInvalid { key } => {
                write!(f, "invalid placement_override key `{key}`")
            }
            Self::PlacementOverrideZero { key } => {
                write!(f, "placement_override for {key} is zero")
            }
            Self::GpuOnlyWithCpuOverride => {
                write!(f, "placement=gpu-only but placement_override includes cpu")
            }
            Self::SplitUnknown { value, expected } => {
                write!(f, "unknown devices.split `{value}` (expected {expected})")
            }
            Self::ExpertOffloadConflictsShardedSplit { split } => {
                write!(
                    f,
                    "expert_offload cannot be combined with devices.split=`{split}` (sharded split is GPU-only; expert offload targets the CPU)"
                )
            }
            Self::ExpertOffloadRequiresHybridPlacement => {
                write!(
                    f,
                    "expert_offload requires placement=hybrid (expert tensors offload to CPU)"
                )
            }
            Self::ShardedSplitRequiresGpuOnly { split } => {
                write!(
                    f,
                    "devices.split=`{split}` requires placement=gpu-only (tensor/row split cannot spill to CPU)"
                )
            }
            Self::ShardedSplitLlamaCppOnly { split } => {
                write!(
                    f,
                    "devices.split=`{split}` is only valid for llama-cpp services"
                )
            }
            Self::ShardedSplitConflictsOverrideTensor { split } => {
                write!(
                    f,
                    "devices.split=`{split}` cannot be combined with override_tensor"
                )
            }
            Self::TensorSplitWeightsRequiresSharded => {
                write!(
                    f,
                    "devices.tensor_split_weights is only valid with a sharded split mode (`row` or `tensor`)"
                )
            }
        }
    }
}
