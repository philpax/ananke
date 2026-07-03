//! Per-service cache of GGUF metadata + estimator output.
//!
//! The estimator is pure over `(GGUF bytes, service config)`. The
//! daemon already runs it on every spawn to size the placement; the
//! `ServiceDetail` handler wants the same data without paying a fresh
//! GGUF read per HTTP poll. This cache memoises the result keyed on
//! service name and invalidates on two signals:
//!
//!   1. The configured `model` / `mmproj` paths change — the operator
//!      pointed the service at a different file.
//!   2. The estimator-relevant config fingerprint changes (context,
//!      override_tensor, cache_type_*, compute_buffer_mb,
//!      allow_fallback). The model on disk is the same but the
//!      estimate's numbers aren't, so the cached entry is stale.
//!
//! Cache lifetime is per daemon run. Editing the GGUF in place
//! without changing its path or any config field won't invalidate;
//! restart the daemon to refresh.

use std::{collections::BTreeMap, path::PathBuf, sync::Arc};

use ananke_api::services::detail::{EstimateSummary, ModelInfo};
use parking_lot::RwLock;
use smol_str::SmolStr;

use crate::{
    estimator::Estimate,
    gguf::{GgufSummary, GgufValue},
};

/// Cloneable handle to the daemon-wide estimate cache. Backed by a
/// single `RwLock<BTreeMap>` — most accesses are reads (detail polls)
/// with the occasional write on cache miss or invalidation.
#[derive(Clone, Default)]
pub struct EstimateCache {
    inner: Arc<RwLock<BTreeMap<SmolStr, CacheEntry>>>,
}

#[derive(Clone)]
pub struct CacheEntry {
    /// Key for invalidation: re-measure when the configured model
    /// path changes.
    pub model_path: PathBuf,
    /// Same, for the vision projector.
    pub mmproj_path: Option<PathBuf>,
    /// `EstimatorInputs::config_fingerprint()` at the moment this
    /// entry was inserted. A mismatch on lookup means the operator
    /// changed a context / cache_type / override_tensor / … setting
    /// that affects the estimate numbers without changing the GGUF
    /// path.
    pub config_fingerprint: u64,
    pub model_info: ModelInfo,
    pub estimate: EstimateSummary,
    /// The full internal estimate the summary was projected from. Kept so the
    /// detail handler can re-run the packer (for the placement preview)
    /// without re-parsing the GGUF — the summary alone is lossy (no per-layer
    /// breakdown, MTP, or non-layer split).
    pub estimate_full: Estimate,
}

impl CacheEntry {
    /// Build a cache entry from a fresh estimator run. Centralises
    /// the GGUF → `ModelInfo` and `Estimate` → `EstimateSummary`
    /// projections so the management detail handler and the
    /// supervisor's spawn-time cache warming produce byte-identical
    /// entries from the same inputs.
    pub fn build(
        summary: &GgufSummary,
        estimate: &Estimate,
        model_path: PathBuf,
        mmproj_path: Option<PathBuf>,
        config_fingerprint: u64,
    ) -> Self {
        let file_name = model_path
            .file_name()
            .map(|os| os.to_string_lossy().to_string())
            .unwrap_or_else(|| model_path.to_string_lossy().to_string());
        let trained_context_key = format!("{}.context_length", summary.architecture);
        let trained_context_length = summary
            .metadata
            .get(trained_context_key.as_str())
            .and_then(|v| v.as_u32());
        let model_name = summary
            .metadata
            .get("general.name")
            .and_then(GgufValue::as_str)
            .map(str::to_string)
            .filter(|s| !s.is_empty());
        let license = summary
            .metadata
            .get("general.license")
            .and_then(GgufValue::as_str)
            .map(str::to_string)
            .filter(|s| !s.is_empty());
        let parameter_count = summary
            .metadata
            .get("general.parameter_count")
            .and_then(GgufValue::as_u64);

        let has_mmproj = mmproj_path.is_some();
        let model_info = ModelInfo {
            architecture: summary.architecture.to_string(),
            model_name,
            license,
            parameter_count,
            total_tensor_bytes: summary.total_tensor_bytes,
            block_count: summary.block_count,
            shard_count: summary.shards.len() as u32,
            trained_context_length,
            file_name,
            has_mmproj,
        };

        let kv_bytes_for_context = estimate
            .kv_per_token
            .saturating_mul(estimate.context as u64);
        let compute_buffer_bytes_per_device = (estimate.compute_buffer_mb as u64) * 1024 * 1024;
        let estimate_summary = EstimateSummary {
            weights_bytes: estimate.weights_bytes,
            kv_per_token: estimate.kv_per_token,
            configured_context: estimate.context,
            kv_bytes_for_context,
            compute_buffer_bytes_per_device,
        };

        Self {
            model_path,
            mmproj_path,
            config_fingerprint,
            model_info,
            estimate: estimate_summary,
            estimate_full: estimate.clone(),
        }
    }
}

impl EstimateCache {
    pub fn new() -> Self {
        Self::default()
    }

    /// Return the cached entry only if every invalidation key still
    /// matches: model + mmproj paths AND the
    /// [`EstimatorInputs::config_fingerprint`] the caller is asking
    /// against. Any mismatch is treated as a cache miss so the caller
    /// re-runs the estimator.
    pub fn get(
        &self,
        name: &SmolStr,
        model_path: &std::path::Path,
        mmproj_path: Option<&std::path::Path>,
        config_fingerprint: u64,
    ) -> Option<CacheEntry> {
        let inner = self.inner.read();
        let entry = inner.get(name)?;
        if entry.model_path != model_path {
            return None;
        }
        match (&entry.mmproj_path, mmproj_path) {
            (Some(a), Some(b)) if a == b => {}
            (None, None) => {}
            _ => return None,
        }
        if entry.config_fingerprint != config_fingerprint {
            return None;
        }
        Some(entry.clone())
    }

    pub fn insert(&self, name: SmolStr, entry: CacheEntry) {
        self.inner.write().insert(name, entry);
    }

    /// Drop the entry for `name`. Called by the reconciler when a
    /// service is removed or its config changes in a way that might
    /// affect the estimate.
    pub fn invalidate(&self, name: &SmolStr) {
        self.inner.write().remove(name);
    }

    /// Drop every entry. Used on full config reload as a coarse
    /// invalidation when per-service tracking would be more code than
    /// it saves.
    pub fn clear(&self) {
        self.inner.write().clear();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const FP: u64 = 0xC0FFEE;

    fn make_entry(model: &str, mmproj: Option<&str>, fp: u64) -> CacheEntry {
        CacheEntry {
            model_path: PathBuf::from(model),
            mmproj_path: mmproj.map(PathBuf::from),
            config_fingerprint: fp,
            model_info: ModelInfo {
                architecture: "llama".into(),
                model_name: None,
                license: None,
                parameter_count: None,
                total_tensor_bytes: 1,
                block_count: None,
                shard_count: 1,
                trained_context_length: None,
                file_name: "x.gguf".into(),
                has_mmproj: mmproj.is_some(),
            },
            estimate: EstimateSummary {
                weights_bytes: 1,
                kv_per_token: 0,
                configured_context: 4096,
                kv_bytes_for_context: 0,
                compute_buffer_bytes_per_device: 0,
            },
            estimate_full: Estimate {
                weights_bytes: 1,
                kv_per_token: 0,
                compute_buffer_mb: 0,
                mtp_bytes: 0,
                per_layer_bytes: None,
                attention_layers: None,
                non_layer: crate::estimator::NonLayer::default(),
                override_tensor_bytes: std::collections::BTreeMap::new(),
                expert_layers: Vec::new(),
                expert_tensors: None,
                context: 4096,
                architecture: SmolStr::new("llama"),
            },
        }
    }

    #[test]
    fn hit_when_paths_and_fingerprint_match() {
        let cache = EstimateCache::new();
        let name = SmolStr::new("svc");
        cache.insert(name.clone(), make_entry("/a/model.gguf", None, FP));
        assert!(
            cache
                .get(&name, std::path::Path::new("/a/model.gguf"), None, FP)
                .is_some()
        );
    }

    #[test]
    fn miss_when_model_path_differs() {
        let cache = EstimateCache::new();
        let name = SmolStr::new("svc");
        cache.insert(name.clone(), make_entry("/a/model.gguf", None, FP));
        assert!(
            cache
                .get(&name, std::path::Path::new("/a/other.gguf"), None, FP)
                .is_none(),
            "different model path must invalidate"
        );
    }

    #[test]
    fn miss_when_mmproj_added_or_removed() {
        let cache = EstimateCache::new();
        let name = SmolStr::new("svc");
        cache.insert(
            name.clone(),
            make_entry("/a/model.gguf", Some("/a/mmproj.gguf"), FP),
        );
        assert!(
            cache
                .get(&name, std::path::Path::new("/a/model.gguf"), None, FP)
                .is_none(),
            "mmproj removal must invalidate"
        );
        assert!(
            cache
                .get(
                    &name,
                    std::path::Path::new("/a/model.gguf"),
                    Some(std::path::Path::new("/b/mmproj.gguf")),
                    FP,
                )
                .is_none(),
            "mmproj path change must invalidate"
        );
    }

    /// Pin the new fingerprint-based invalidation: paths unchanged
    /// but a non-path estimator-relevant setting (context,
    /// override_tensor, etc.) changed since insertion → cache miss so
    /// the handler re-runs the estimator against the updated config.
    #[test]
    fn miss_when_config_fingerprint_differs() {
        let cache = EstimateCache::new();
        let name = SmolStr::new("svc");
        cache.insert(name.clone(), make_entry("/a/model.gguf", None, FP));
        assert!(
            cache
                .get(
                    &name,
                    std::path::Path::new("/a/model.gguf"),
                    None,
                    FP.wrapping_add(1)
                )
                .is_none(),
            "fingerprint mismatch must invalidate even when paths match"
        );
    }

    #[test]
    fn invalidate_removes_entry() {
        let cache = EstimateCache::new();
        let name = SmolStr::new("svc");
        cache.insert(name.clone(), make_entry("/a/model.gguf", None, FP));
        cache.invalidate(&name);
        assert!(
            cache
                .get(&name, std::path::Path::new("/a/model.gguf"), None, FP)
                .is_none()
        );
    }
}
