//! Per-service cache of GGUF metadata + estimator output.
//!
//! The estimator is pure over `(GGUF bytes, service config)`. The
//! daemon already runs it on every spawn to size the placement; the
//! `ServiceDetail` handler wants the same data without paying a fresh
//! GGUF read per HTTP poll. This cache memoises the result keyed on
//! service name and invalidates whenever the configured `model` /
//! `mmproj` paths change — the cheapest signal that the underlying
//! file might be different from what we measured.
//!
//! Cache lifetime is per daemon run. Editing the GGUF in place
//! without changing its path won't invalidate; restart the daemon (or
//! touch the service's config) to refresh.

use std::{collections::BTreeMap, path::PathBuf, sync::Arc};

use ananke_api::{EstimateSummary, ModelInfo};
use parking_lot::RwLock;
use smol_str::SmolStr;

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
    pub model_info: ModelInfo,
    pub estimate: EstimateSummary,
}

impl EstimateCache {
    pub fn new() -> Self {
        Self::default()
    }

    /// Return the cached entry only if the (model_path, mmproj_path)
    /// pair still matches what the caller is asking about. A mismatch
    /// means the operator pointed the service at a different file
    /// since the last measurement — we treat that as a cache miss so
    /// the caller re-reads.
    pub fn get(
        &self,
        name: &SmolStr,
        model_path: &std::path::Path,
        mmproj_path: Option<&std::path::Path>,
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

    fn make_entry(model: &str, mmproj: Option<&str>) -> CacheEntry {
        CacheEntry {
            model_path: PathBuf::from(model),
            mmproj_path: mmproj.map(PathBuf::from),
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
        }
    }

    #[test]
    fn hit_when_paths_match() {
        let cache = EstimateCache::new();
        let name = SmolStr::new("svc");
        cache.insert(name.clone(), make_entry("/a/model.gguf", None));
        assert!(
            cache
                .get(&name, std::path::Path::new("/a/model.gguf"), None)
                .is_some()
        );
    }

    #[test]
    fn miss_when_model_path_differs() {
        let cache = EstimateCache::new();
        let name = SmolStr::new("svc");
        cache.insert(name.clone(), make_entry("/a/model.gguf", None));
        assert!(
            cache
                .get(&name, std::path::Path::new("/a/other.gguf"), None)
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
            make_entry("/a/model.gguf", Some("/a/mmproj.gguf")),
        );
        assert!(
            cache
                .get(&name, std::path::Path::new("/a/model.gguf"), None)
                .is_none(),
            "mmproj removal must invalidate"
        );
        assert!(
            cache
                .get(
                    &name,
                    std::path::Path::new("/a/model.gguf"),
                    Some(std::path::Path::new("/b/mmproj.gguf"))
                )
                .is_none(),
            "mmproj path change must invalidate"
        );
    }

    #[test]
    fn invalidate_removes_entry() {
        let cache = EstimateCache::new();
        let name = SmolStr::new("svc");
        cache.insert(name.clone(), make_entry("/a/model.gguf", None));
        cache.invalidate(&name);
        assert!(
            cache
                .get(&name, std::path::Path::new("/a/model.gguf"), None)
                .is_none()
        );
    }
}
