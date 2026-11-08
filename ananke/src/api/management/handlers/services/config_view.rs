//! Pure mappings from a validated service config to the detail DTOs.

use ananke_api::services::detail::{
    AllocationConfig, CommandDetail, IkParams, PlacementConfig, PlacementPolicy as WirePolicy,
    RuntimeInfo, ServingConfig, SplitConfig, SplitMode as WireSplitMode,
};
use ananke_config::{placement::SplitMode, units::MIB};
use ananke_estimate::{host_buffer::prompt_cache_bytes, service_inputs::estimator_inputs};

use crate::config::{AllocationMode, OffloadMode, PlacementPolicy, ServiceConfig};

/// Which llama-server implementation serves a llama-cpp service, plus
/// the fork-specific parameters when it is not mainline. Computed for the
/// detail view so the frontend can render a runtime card without a spawn.
pub(crate) fn runtime_info(svc_cfg: &ServiceConfig) -> Option<RuntimeInfo> {
    let lc = svc_cfg.llama_cpp()?;
    let ik = match lc.runtime.ik() {
        None => {
            return Some(RuntimeInfo {
                kind: "llama-cpp".into(),
                ik: None,
            });
        }
        Some(ik) => ik,
    };
    Some(RuntimeInfo {
        kind: "ik-llama".into(),
        ik: Some(IkParams {
            mla: ik.mla,
            dsa: ik.dsa,
            attn_max_batch: ik.attn_max_batch,
            runtime_repack: ik.runtime_repack,
        }),
    })
}

/// Curated serving knobs for the detail view, including the derived
/// per-slot context (`context / parallel` under a statically-split KV
/// pool — the number that actually bounds a request, and one no single
/// config key states).
///
/// The knobs `extra_args` can override (parallelism, unified KV, flash
/// attention, ubatch, prompt cache) come from the estimator's inputs, which
/// resolve them the way the launch command does, so the card, the estimate,
/// and the running server cannot disagree.
pub(crate) fn serving_config(svc_cfg: &ServiceConfig) -> Option<ServingConfig> {
    let lc = svc_cfg.llama_cpp()?;
    let inputs = estimator_inputs(svc_cfg)?;
    let parallel = inputs.parallel.unwrap_or(1);
    let kv_unified = inputs.kv_unified.unwrap_or(false);
    let effective_context_per_slot = lc
        .context
        .and_then(|c| (parallel > 1 && !kv_unified).then_some(c / parallel.max(1)));
    Some(ServingConfig {
        binary: lc.binary.to_string_lossy().into_owned(),
        cache_type_k: lc.cache_type_k.as_deref().unwrap_or("f16").to_string(),
        cache_type_v: lc.cache_type_v.as_deref().unwrap_or("f16").to_string(),
        flash_attn: inputs.flash_attn.unwrap_or(false),
        parallel,
        n_gpu_layers: lc.n_gpu_layers,
        context: lc.context,
        cache_ram_bytes: prompt_cache_bytes(&inputs),
        kv_unified,
        effective_context_per_slot,
        spec_type: lc.spec_type.as_deref().map(str::to_string),
        draft_model: lc
            .draft_model
            .as_deref()
            .and_then(|p| p.file_name().map(|f| f.to_string_lossy().into_owned())),
        expert_offload: match lc.expert_offload {
            OffloadMode::Off => "off".to_string(),
            OffloadMode::Auto => "auto".to_string(),
            OffloadMode::Layers(n) => format!("{n} layers"),
        },
        batch_size: lc.batch_size,
        ubatch_size: inputs.ubatch,
        threads: lc.threads,
        threads_batch: lc.threads_batch,
        numa: lc.numa.map(|n| n.as_flag().to_string()),
        mmap: lc.mmap.unwrap_or(true),
        mlock: lc.mlock.unwrap_or(false),
    })
}

/// The placement constraints the service's config sets.
pub(crate) fn placement_config(svc_cfg: &ServiceConfig) -> PlacementConfig {
    PlacementConfig {
        policy: match svc_cfg.placement_policy {
            PlacementPolicy::GpuOnly => WirePolicy::GpuOnly,
            PlacementPolicy::CpuOnly => WirePolicy::CpuOnly,
            PlacementPolicy::Hybrid => WirePolicy::Hybrid,
        },
        gpu_allow: svc_cfg.gpu_allow.clone(),
        gpu_headroom_bytes: svc_cfg.gpu_headroom_mb * MIB,
        split: svc_cfg.llama_cpp().map(|_| SplitConfig {
            mode: match svc_cfg.split_mode {
                SplitMode::Layer => WireSplitMode::Layer,
                SplitMode::Row => WireSplitMode::Row,
                SplitMode::Tensor => WireSplitMode::Tensor,
            },
            tensor_split_weights: svc_cfg.tensor_split_weights.clone(),
        }),
        allocation: match svc_cfg.allocation_mode {
            AllocationMode::None => AllocationConfig::Estimated,
            AllocationMode::Static { reserve_mb } => AllocationConfig::Static {
                reserve_bytes: reserve_mb * MIB,
            },
            AllocationMode::Dynamic {
                min_mb,
                max_mb,
                min_borrower_runtime_ms,
            } => AllocationConfig::Dynamic {
                min_bytes: min_mb * MIB,
                max_bytes: max_mb * MIB,
                min_borrower_runtime_ms,
            },
        },
    }
}

/// A command service's settings that its launch command does not show.
pub(crate) fn command_detail(svc_cfg: &ServiceConfig) -> Option<CommandDetail> {
    let cmd = svc_cfg.command()?;
    Some(CommandDetail {
        upstream_model: cmd
            .openai_proxy
            .as_ref()
            .map(|p| p.upstream_model.to_string()),
        shutdown_command: cmd.shutdown_command.clone(),
        private_port_fixed: cmd.private_port_override.is_some(),
        workdir: cmd
            .workdir
            .as_ref()
            .map(|w| w.to_string_lossy().into_owned()),
    })
}

#[cfg(test)]
mod tests {
    use std::path::PathBuf;

    use ananke_config::validate::{
        OpenAiProxyConfig, TemplateConfig,
        test_fixtures::{minimal_command_service, minimal_llama_cpp_service},
    };

    use super::*;

    #[test]
    fn tensor_split_carries_its_weights() {
        let mut svc = minimal_llama_cpp_service("demo");
        svc.placement_policy = PlacementPolicy::GpuOnly;
        svc.gpu_allow = vec![0, 1];
        svc.split_mode = SplitMode::Tensor;
        svc.tensor_split_weights = Some(vec![1.0, 2.0]);
        svc.gpu_headroom_mb = 512;
        let placement = placement_config(&svc);
        assert_eq!(placement.policy, WirePolicy::GpuOnly);
        assert_eq!(placement.gpu_allow, vec![0, 1]);
        assert_eq!(placement.gpu_headroom_bytes, 512 * MIB);
        assert_eq!(
            placement.split,
            Some(SplitConfig {
                mode: WireSplitMode::Tensor,
                tensor_split_weights: Some(vec![1.0, 2.0]),
            })
        );
        assert_eq!(placement.allocation, AllocationConfig::Estimated);
    }

    #[test]
    fn layer_split_has_no_weights() {
        let svc = minimal_llama_cpp_service("demo");
        let split = placement_config(&svc).split.unwrap();
        assert_eq!(split.mode, WireSplitMode::Layer);
        assert_eq!(split.tensor_split_weights, None);
    }

    #[test]
    fn command_service_has_no_split_and_reports_its_settings() {
        let mut svc = minimal_command_service("comfy", vec!["run".into()]);
        svc.allocation_mode = AllocationMode::Dynamic {
            min_mb: 1024,
            max_mb: 4096,
            min_borrower_runtime_ms: 30_000,
        };
        let TemplateConfig::Command(cmd) = &mut svc.template_config else {
            unreachable!("minimal_command_service builds a command template");
        };
        cmd.shutdown_command = Some(vec!["stop".into(), "now".into()]);
        cmd.private_port_override = Some(9000);
        cmd.workdir = Some(PathBuf::from("/srv/comfy"));
        cmd.openai_proxy = Some(OpenAiProxyConfig {
            upstream_model: "upstream".into(),
        });

        let placement = placement_config(&svc);
        assert_eq!(placement.split, None);
        assert_eq!(
            placement.allocation,
            AllocationConfig::Dynamic {
                min_bytes: 1024 * MIB,
                max_bytes: 4096 * MIB,
                min_borrower_runtime_ms: 30_000,
            }
        );
        assert_eq!(
            command_detail(&svc),
            Some(CommandDetail {
                upstream_model: Some("upstream".into()),
                shutdown_command: Some(vec!["stop".into(), "now".into()]),
                private_port_fixed: true,
                workdir: Some("/srv/comfy".into()),
            })
        );
        assert_eq!(serving_config(&svc), None);
    }

    #[test]
    fn static_allocation_is_in_bytes() {
        let mut svc = minimal_command_service("comfy", vec!["run".into()]);
        svc.allocation_mode = AllocationMode::Static { reserve_mb: 2048 };
        assert_eq!(
            placement_config(&svc).allocation,
            AllocationConfig::Static {
                reserve_bytes: 2048 * MIB
            }
        );
        assert!(command_detail(&svc).is_some_and(|c| !c.private_port_fixed));
    }

    #[test]
    fn llama_cpp_service_has_no_command_detail() {
        assert_eq!(command_detail(&minimal_llama_cpp_service("demo")), None);
    }

    #[test]
    fn cache_ram_defaults_and_follows_extra_args() {
        let svc = minimal_llama_cpp_service("demo");
        let default_bytes = serving_config(&svc).unwrap().cache_ram_bytes;
        assert!(default_bytes > 0);

        let mut svc = minimal_llama_cpp_service("demo");
        svc.extra_args = vec!["-cram".into(), "0".into()];
        assert_eq!(serving_config(&svc).unwrap().cache_ram_bytes, 0);
    }

    #[test]
    fn kv_unified_follows_extra_args() {
        let mut svc = minimal_llama_cpp_service("demo");
        assert!(!serving_config(&svc).unwrap().kv_unified);
        svc.extra_args = vec!["-kvu".into()];
        assert!(serving_config(&svc).unwrap().kv_unified);
    }
}
