//! VRAM estimator — architecture-aware dispatch.

pub mod fallback;
pub mod hybrid;
pub mod kv;
pub mod llama;
pub mod mamba;
pub mod moe;
pub mod override_tensor;
pub mod types;

use std::path::Path;

use tracing::{info, warn};
pub use types::{Estimate, NonLayer};

use crate::{
    config::ServiceConfig,
    gguf::{self, GgufSummary},
};

/// Produce a base estimate for `svc`. Reads the GGUF (including any
/// mmproj) and dispatches on `general.architecture`. Pure function;
/// caller applies rolling correction + safety factor afterward.
pub fn estimate_from_path(path: &Path, svc: &ServiceConfig) -> Result<Estimate, String> {
    let summary = gguf::read(path).map_err(|e| e.to_string())?;

    info!(
        service = %svc.name,
        architecture = %summary.architecture,
        block_count = ?summary.block_count,
        tensor_count = summary.tensors.len(),
        total_tensor_gb = summary.total_tensor_bytes / (1024 * 1024 * 1024),
        shard_count = summary.shards.len(),
        "gguf summary",
    );

    let mut est = dispatch(&summary, svc);

    info!(
        service = %svc.name,
        weights_gb = est.weights_bytes / (1024 * 1024 * 1024),
        per_layer_len = est.per_layer_bytes.as_ref().map(|v| v.len()).unwrap_or(0),
        kv_per_token = est.kv_per_token,
        "post-dispatch estimate",
    );

    // Apply user-declared override_tensor rules BEFORE mmproj so matched
    // tensors leave the layer/non-layer budget cleanly (spec §8.2.4).
    if let Some(rules) = svc.raw.override_tensor.as_ref() {
        override_tensor::parse_and_apply(&mut est, &summary, rules);
    }

    // Add mmproj bytes to GPU 0 weights (per spec §8.3).
    if let Some(mmproj) = svc.raw.mmproj.as_ref() {
        match gguf::read(mmproj.as_path()) {
            Ok(proj) => {
                est.weights_bytes = est.weights_bytes.saturating_add(proj.total_tensor_bytes);
                est.non_layer.other_bytes = est
                    .non_layer
                    .other_bytes
                    .saturating_add(proj.total_tensor_bytes);
            }
            Err(e) => warn!(error = %e, path = %mmproj.display(), "mmproj read failed"),
        }
    }

    Ok(est)
}

pub fn dispatch(summary: &GgufSummary, svc: &ServiceConfig) -> Estimate {
    let arch = summary.architecture.as_str();
    if llama::is_llama_family(arch) {
        return llama::estimate(summary, svc);
    }
    if moe::is_moe(arch) {
        return moe::estimate(summary, svc);
    }
    if mamba::is_mamba(arch) {
        return mamba::estimate(summary, svc);
    }
    if hybrid::is_hybrid(arch) {
        return hybrid::estimate(summary, svc);
    }
    let context = svc.raw.context.unwrap_or(4096);
    fallback::estimate_fallback(summary, context)
}

#[cfg(test)]
mod tests {
    use smol_str::SmolStr;

    use super::*;
    use crate::{
        config::{
            parse::RawService,
            validate::{
                AllocationMode, DeviceSlot, Filters, HealthSettings, Lifecycle, PlacementPolicy,
                ServiceConfig, Template,
            },
        },
        gguf::types::{GgufSummary, GgufValue},
    };

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
            template: Template::LlamaCpp,
            port: 0,
            private_port: 0,
            lifecycle: Lifecycle::OnDemand,
            priority: 50,
            health: HealthSettings {
                http_path: "/".into(),
                timeout_ms: 1000,
                probe_interval_ms: 500,
            },
            placement_override: placement,
            placement_policy: PlacementPolicy::GpuOnly,
            idle_timeout_ms: 60_000,
            warming_grace_ms: 100,
            drain_timeout_ms: 1000,
            extended_stream_drain_ms: 1000,
            max_request_duration_ms: 1000,
            filters: Filters::default(),
            allocation_mode: AllocationMode::None,
            command: None,
            workdir: None,
            openai_compat: true,
            raw,
        }
    }

    #[test]
    fn dispatch_recognises_known_families() {
        let mut metadata = std::collections::BTreeMap::new();
        metadata.insert(
            SmolStr::new("general.architecture"),
            GgufValue::String("qwen3".into()),
        );
        metadata.insert(SmolStr::new("qwen3.block_count"), GgufValue::U32(1));
        let summary = GgufSummary {
            path: "/fake".into(),
            total_tensor_bytes: 0,
            tensors: Default::default(),
            metadata,
            block_count: Some(1),
            architecture: SmolStr::new("qwen3"),
            shards: vec!["/fake".into()],
        };
        let e = dispatch(&summary, &svc_with(None));
        assert_eq!(e.architecture, "qwen3");
    }

    #[test]
    fn dispatch_unknown_goes_to_fallback() {
        let mut metadata = std::collections::BTreeMap::new();
        metadata.insert(
            SmolStr::new("general.architecture"),
            GgufValue::String("novel-arch".into()),
        );
        let summary = GgufSummary {
            path: "/fake".into(),
            total_tensor_bytes: 1_000_000,
            tensors: Default::default(),
            metadata,
            block_count: None,
            architecture: SmolStr::new("novel-arch"),
            shards: vec!["/fake".into()],
        };
        let e = dispatch(&summary, &svc_with(None));
        // Fallback uses 1.15 × total + 512 MB.
        assert!(e.weights_bytes >= 512 * 1024 * 1024);
    }
}
