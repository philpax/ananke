//! The production model configurations, read from `models.toml`.
//!
//! Each entry mirrors what a `ServiceConfig` would carry for that model, so an
//! estimate taken here is the one the daemon would produce. The file is
//! gitignored because its paths are system-specific; `models.toml.example`
//! documents the shape.

use std::path::{Path, PathBuf};

use ananke_config::{
    flags::expert_offload::AUTO,
    placement::{OffloadMode, PlacementInputs, PlacementPolicy, SplitMode},
};
use ananke_estimate::{EstimatorInputs, Fork, Speculation};
use ananke_gguf::GgufType;
use serde::Deserialize;

use crate::plan::library::Library;

/// Where the model files live, relative to which `ModelConfig::model` is
/// resolved.
///
/// Resolved through `$LLM_DIR` the same way a plan's paths are, so a scoreboard
/// and a plan name the same file on a machine that keeps its library elsewhere.
pub fn model_root() -> PathBuf {
    Library::from_env().root().to_path_buf()
}

/// How many cards a model that names neither `visible_devices` nor `gpus` spans:
/// the campaign machine's pair.
const DEFAULT_CARDS: u32 = 2;

#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ModelsFile {
    #[serde(default)]
    pub model: Vec<ModelConfig>,
}

/// One production model's configuration.
///
/// `deny_unknown_fields` deliberately: a mistyped key here would otherwise be
/// dropped in silence, and the estimate would come out confidently wrong. The
/// file says `mtp = true`: a struct that declares only `spec_type` drops it,
/// ignoring MTP for three models and reading the scoreboard 16% low on one.
#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ModelConfig {
    pub name: String,
    pub model: String,
    #[serde(default)]
    pub mmproj: Option<String>,
    #[serde(default)]
    pub draft_model: Option<String>,
    pub context: u32,
    #[serde(default)]
    pub ubatch: Option<u32>,
    #[serde(default)]
    pub cache_type_k: Option<String>,
    #[serde(default)]
    pub cache_type_v: Option<String>,
    #[serde(default)]
    pub parallel: Option<u32>,
    #[serde(default)]
    pub flash_attn: Option<bool>,
    #[serde(default)]
    pub kv_unified: Option<bool>,
    #[serde(default)]
    pub cache_ram_mb: Option<u32>,
    #[serde(default)]
    pub split_mode: Option<String>,
    #[serde(default)]
    pub visible_devices: Option<u32>,
    #[serde(default)]
    pub n_cpu_moe: Option<u32>,
    #[serde(default)]
    pub expert_offload: Option<String>,
    #[serde(default)]
    pub ik_llama: bool,
    #[serde(default)]
    pub ik_dsa: bool,
    /// Whether the service runs with speculative decoding at all.
    #[serde(default)]
    pub mtp: bool,
    /// The `--spec-type` value, when it is not `draft-mtp` — dflash and any
    /// future mechanism. Absent means `draft-mtp`, so every existing
    /// `mtp = true` entry keeps meaning what it always has.
    #[serde(default)]
    pub spec_type: Option<String>,
    #[serde(default)]
    pub gpus: Option<Vec<u32>>,
}

impl ModelConfig {
    /// The GGUF's absolute path.
    pub fn model_path(&self) -> PathBuf {
        model_root().join(&self.model)
    }

    /// The vision projector's absolute path, if this model has one.
    pub fn mmproj_path(&self) -> Option<PathBuf> {
        self.mmproj.as_ref().map(|p| model_root().join(p))
    }

    /// The separate draft GGUF's absolute path, if this model has one.
    pub fn draft_path(&self) -> Option<PathBuf> {
        self.draft_model.as_ref().map(|p| model_root().join(p))
    }

    /// How many cards this model spans.
    pub fn cards(&self) -> u32 {
        self.visible_devices
            .or_else(|| self.gpus.as_ref().map(|g| g.len() as u32))
            .unwrap_or(DEFAULT_CARDS)
            .max(1)
    }

    /// The split this model runs under.
    ///
    /// Parsed with `SplitMode::from_flag` so this file's spelling of the flag and
    /// the config validator's cannot drift.
    pub fn split(&self) -> SplitMode {
        self.split_mode
            .as_deref()
            .and_then(SplitMode::from_flag)
            .unwrap_or(SplitMode::Layer)
    }

    /// How much of a mixture of experts is host-resident.
    pub fn offload(&self) -> OffloadMode {
        match (self.n_cpu_moe, self.expert_offload.as_deref()) {
            (Some(n), _) => OffloadMode::Layers(n),
            (None, Some(AUTO)) => OffloadMode::Auto,
            _ => OffloadMode::Off,
        }
    }

    /// Which llama.cpp this cell was measured against.
    pub fn fork(&self) -> Fork {
        if self.ik_llama {
            Fork::Ik { dsa: self.ik_dsa }
        } else {
            Fork::Mainline
        }
    }

    /// The estimator inputs for this model.
    ///
    /// `model`, `mmproj`, and `draft` are passed in because `EstimatorInputs`
    /// borrows its paths and they must outlive the returned value.
    pub fn estimator_inputs<'a>(
        &'a self,
        model: &'a Path,
        mmproj: Option<&'a Path>,
        draft: Option<&'a Path>,
    ) -> EstimatorInputs<'a> {
        EstimatorInputs {
            name: &self.name,
            model,
            mmproj,
            context: self.context,
            ubatch: self.ubatch,
            worst_graph_tokens: None,
            ik_assignment_options_supported: true,
            visible_devices: self.cards(),
            host_resident_experts: self.offload().is_enabled(),
            split_mode: self.split(),
            cache_type_k: self.cache_type_k.as_deref().and_then(GgufType::from_name),
            cache_type_v: self.cache_type_v.as_deref().and_then(GgufType::from_name),
            override_tensor: &[],
            compute_buffer_mb: None,
            speculation: match (self.mtp, draft) {
                (true, Some(path)) => Speculation::SeparateDraft {
                    path,
                    spec_type: self.spec_type.as_deref().unwrap_or("draft-mtp"),
                },
                (true, None) => Speculation::EmbeddedMtp,
                (false, _) => Speculation::None,
            },
            fork: self.fork(),
            parallel: self.parallel,
            flash_attn: self.flash_attn,
            kv_unified: self.kv_unified,
            cache_ram_mb: self.cache_ram_mb,
        }
    }

    /// The placement for this model.
    pub fn placement_inputs(&self) -> PlacementInputs {
        PlacementInputs {
            policy: if self.offload().is_enabled() {
                PlacementPolicy::Hybrid
            } else {
                PlacementPolicy::GpuOnly
            },
            split_mode: self.split(),
            gpu_allow: self
                .gpus
                .clone()
                .unwrap_or_else(|| (0..self.cards()).collect()),
            expert_offload: self.offload(),
            ik_llama: self.ik_llama,
            ..PlacementInputs::named(&self.name)
        }
    }
}

/// Read `models.toml`.
pub fn load(path: &Path) -> Result<Vec<ModelConfig>, String> {
    let text = std::fs::read_to_string(path).map_err(|e| {
        format!(
            "{}: {e} — copy models.toml.example and edit the paths",
            path.display()
        )
    })?;
    let parsed: ModelsFile =
        toml::from_str(&text).map_err(|e| format!("{}: {e}", path.display()))?;
    Ok(parsed.model)
}
