//! Canonical vocabularies for enum-valued config fields.
//!
//! Each accepted string literal for a validated enum field (`devices.split`,
//! `numa`, `expert_offload`, …) lives here exactly once. Both the schema
//! documentation (this crate's [`crate::docs`]) and the daemon's config
//! validation (the `ananke` crate's `NumaStrategy`, `SplitMode`,
//! `OffloadMode`) reference these constants, so adding or renaming an
//! accepted value is a one-line change that cannot leave the parser, the
//! error messages, and the docs disagreeing.

/// `devices.split` / `--split-mode` values.
pub mod split_mode {
    /// Whole-layer pipeline across GPUs (the default).
    pub const LAYER: &str = "layer";
    /// Tensor parallelism, llama.cpp's older implementation.
    pub const ROW: &str = "row";
    /// Tensor parallelism, llama.cpp's newer implementation.
    pub const TENSOR: &str = "tensor";
    /// Every accepted value, in declaration order.
    pub const ALL: &[&str] = &[LAYER, ROW, TENSOR];
}

/// `numa` / `--numa` values.
pub mod numa {
    /// Spread threads and interleave memory across all NUMA nodes.
    pub const DISTRIBUTE: &str = "distribute";
    /// Confine threads and allocation to a single node.
    pub const ISOLATE: &str = "isolate";
    /// Defer placement to an external `numactl` mask.
    pub const NUMACTL: &str = "numactl";
    /// Every accepted value, in declaration order.
    pub const ALL: &[&str] = &[DISTRIBUTE, ISOLATE, NUMACTL];
}

/// `expert_offload` string values. The field also accepts an integer layer
/// count, which has no fixed string form and so is not listed here.
pub mod expert_offload {
    /// No expert offload; whole-layer CPU spill only.
    pub const OFF: &str = "off";
    /// The packer offloads the minimum experts to fit live VRAM.
    pub const AUTO: &str = "auto";
    /// Every accepted string value, in declaration order.
    pub const ALL: &[&str] = &[OFF, AUTO];
}

/// Render a value vocabulary as a quoted, comma-separated list for
/// operator-facing docs and validation errors, e.g. `"layer", "row",
/// "tensor"`.
pub fn quoted_list(values: &[&str]) -> String {
    values
        .iter()
        .map(|v| format!("\"{v}\""))
        .collect::<Vec<_>>()
        .join(", ")
}
