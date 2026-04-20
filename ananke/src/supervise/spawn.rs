//! Linux-only: render llama-server argv from an `EffectiveConfig` service
//! entry. Actual child spawning lives behind the
//! [`crate::system::ProcessSpawner`] trait, with the production
//! [`crate::system::LocalSpawner`] applying `prctl(PR_SET_PDEATHSIG, SIGTERM)`
//! so the child dies if the daemon exits unexpectedly.

use std::collections::BTreeMap;

use crate::{
    allocator::placement::CommandArgs,
    config::validate::{LlamaCppConfig, PlacementPolicy, ServiceConfig, TemplateConfig},
    devices::{Allocation, cuda_env},
};

pub struct SpawnConfig {
    pub binary: String,
    pub args: Vec<String>,
    pub env: BTreeMap<String, String>,
}

/// Render the child command line plus env from a validated `ServiceConfig`,
/// its `Allocation`, and optional placement `CommandArgs`.
///
/// When `cmd_args` is `Some`, the placement engine has already computed
/// `-ngl`/`--tensor-split`/`-ot` values. Any existing `-ngl` flags from the
/// static config path are replaced by the placement-derived value.
pub fn render_argv(
    svc: &ServiceConfig,
    alloc: &Allocation,
    cmd_args: Option<&CommandArgs>,
) -> Result<SpawnConfig, crate::templates::SubstituteError> {
    match &svc.template_config {
        TemplateConfig::LlamaCpp(lc) => Ok(render_llama_cpp_argv(svc, lc, alloc, cmd_args)),
        TemplateConfig::Command(_) => render_command_argv(svc, alloc),
    }
}

fn render_llama_cpp_argv(
    svc: &ServiceConfig,
    lc: &LlamaCppConfig,
    alloc: &Allocation,
    cmd_args: Option<&CommandArgs>,
) -> SpawnConfig {
    let mut args: Vec<String> = Vec::new();

    args.push("-m".into());
    args.push(lc.model.to_string_lossy().into_owned());
    if let Some(mmproj) = &lc.mmproj {
        args.push("--mmproj".into());
        args.push(mmproj.to_string_lossy().into_owned());
    }
    if let Some(ctx) = lc.context {
        args.push("-c".into());
        args.push(ctx.to_string());
    }

    if let Some(ca) = cmd_args {
        // Placement engine provided -ngl; ignore the static config path.
        if let Some(ngl) = ca.ngl {
            args.push("-ngl".into());
            args.push(ngl.to_string());
        }
    } else {
        match svc.placement_policy {
            PlacementPolicy::CpuOnly => {
                args.push("-ngl".into());
                args.push("0".into());
            }
            PlacementPolicy::GpuOnly | PlacementPolicy::Hybrid => {
                if let Some(ngl) = lc.n_gpu_layers {
                    args.push("-ngl".into());
                    args.push(ngl.to_string());
                } else {
                    args.push("-ngl".into());
                    args.push("999".into());
                }
            }
        }
    }

    if lc.flash_attn == Some(true) {
        args.push("-fa".into());
        args.push("on".into());
    }
    if let Some(k) = &lc.cache_type_k {
        args.push("--cache-type-k".into());
        args.push(k.to_string());
    }
    if let Some(v) = &lc.cache_type_v {
        args.push("--cache-type-v".into());
        args.push(v.to_string());
    }
    if lc.jinja.unwrap_or(false) {
        args.push("--jinja".into());
    }
    if let Some(p) = &lc.chat_template_file {
        args.push("--chat-template-file".into());
        args.push(p.to_string_lossy().into_owned());
    }
    if let Some(t) = lc.threads {
        args.push("--threads".into());
        args.push(t.to_string());
    }
    if let Some(t) = lc.threads_batch {
        args.push("--threads-batch".into());
        args.push(t.to_string());
    }
    if let Some(b) = lc.batch_size {
        args.push("-b".into());
        args.push(b.to_string());
    }
    if let Some(b) = lc.ubatch_size {
        args.push("-ub".into());
        args.push(b.to_string());
    }
    if lc.mmap == Some(false) {
        args.push("--no-mmap".into());
    }
    if lc.mlock == Some(true) {
        args.push("--mlock".into());
    }
    if let Some(p) = lc.parallel {
        args.push("-np".into());
        args.push(p.to_string());
    }

    if let Some(ca) = cmd_args {
        // Placement-derived tensor-split and override-tensor rules take
        // precedence; lc.override_tensor is subsumed into CommandArgs by
        // the placement engine already.
        if let Some(ref split) = ca.tensor_split {
            let split_str = split
                .iter()
                .map(|n| n.to_string())
                .collect::<Vec<_>>()
                .join(",");
            args.push("--tensor-split".into());
            args.push(split_str);
        }
        for rule in &ca.override_tensor {
            args.push("-ot".into());
            args.push(rule.clone());
        }
    } else {
        for rule in &lc.override_tensor {
            args.push("-ot".into());
            args.push(rule.clone());
        }
    }

    // Sampling params are passed as extra flags when set.
    let s = &lc.sampling;
    if let Some(t) = s.temperature {
        args.push("--temp".into());
        args.push(t.to_string());
    }
    if let Some(p) = s.top_p {
        args.push("--top-p".into());
        args.push(p.to_string());
    }
    if let Some(k) = s.top_k {
        args.push("--top-k".into());
        args.push(k.to_string());
    }
    if let Some(m) = s.min_p {
        args.push("--min-p".into());
        args.push(m.to_string());
    }
    if let Some(r) = s.repeat_penalty {
        args.push("--repeat-penalty".into());
        args.push(r.to_string());
    }
    args.extend(svc.extra_args.iter().cloned());
    args.push("--host".into());
    args.push("127.0.0.1".into());
    args.push("--port".into());
    args.push(svc.private_port.to_string());

    let mut env: BTreeMap<String, String> = svc.env.clone();
    env.insert("CUDA_VISIBLE_DEVICES".into(), cuda_env::render(alloc));

    SpawnConfig {
        binary: "llama-server".into(),
        args,
        env,
    }
}

/// Assemble the [`PlaceholderContext`] a command-template argv renders
/// against. Shared by spawn-time and shutdown-time so both paths resolve
/// `{port}` / `{gpu_ids}` / `{vram_mb}` / `{name}` identically.
fn placeholder_context<'a>(
    svc: &'a ServiceConfig,
    alloc: &'a Allocation,
) -> crate::templates::PlaceholderContext<'a> {
    use crate::config::AllocationMode;
    let static_vram_mb = match svc.allocation_mode {
        AllocationMode::Static { vram_mb } => Some(vram_mb),
        _ => None,
    };
    crate::templates::PlaceholderContext {
        name: &svc.name,
        port: svc.private_port,
        // Command template has no model path; {model} resolves to empty.
        model: None,
        allocation: alloc,
        static_vram_mb,
    }
}

/// Render any command-template argv (the main `command` or the sibling
/// `shutdown_command`) under the same substitution rules. Hard-fails on
/// substitution errors — callers surface them as `StartFailure` /
/// shutdown-run warnings rather than launching with literal
/// `{placeholder}` tokens in argv.
fn render_command_like(
    argv: &[String],
    svc: &ServiceConfig,
    alloc: &Allocation,
) -> Result<SpawnConfig, crate::templates::SubstituteError> {
    let binary = argv.first().cloned().unwrap_or_default();
    let tail: Vec<String> = argv.iter().skip(1).cloned().collect();

    let ctx = placeholder_context(svc, alloc);
    let user_env: BTreeMap<String, String> = svc.env.clone();
    let (args, env_substituted) = crate::templates::substitute_argv(&tail, &user_env, &ctx)?;

    let mut env = BTreeMap::new();
    for (k, v) in env_substituted {
        env.insert(k, v);
    }
    env.insert("CUDA_VISIBLE_DEVICES".into(), cuda_env::render(alloc));

    Ok(SpawnConfig { binary, args, env })
}

/// Render argv for the optional `shutdown_command` sibling of a
/// command-template service, if one is configured. Returns `None` when
/// the service has no shutdown command or isn't a command-template
/// service. Propagates substitution errors so the caller logs them
/// (instead of launching the shutdown with unresolved `{placeholder}`s).
pub fn render_shutdown_argv(
    svc: &ServiceConfig,
    alloc: &Allocation,
) -> Option<Result<SpawnConfig, crate::templates::SubstituteError>> {
    let TemplateConfig::Command(cmd_cfg) = &svc.template_config else {
        return None;
    };
    let argv = cmd_cfg.shutdown_command.as_ref()?;
    if argv.is_empty() {
        return None;
    }
    Some(render_command_like(argv, svc, alloc))
}

/// Render argv for a `Command`-template service. Substitutes `{port}`,
/// `{gpu_ids}`, `{vram_mb}`, `{model}`, `{name}`.
fn render_command_argv(
    svc: &ServiceConfig,
    alloc: &Allocation,
) -> Result<SpawnConfig, crate::templates::SubstituteError> {
    let TemplateConfig::Command(cmd_cfg) = &svc.template_config else {
        unreachable!("render_command_argv called on non-command service")
    };
    render_command_like(&cmd_cfg.command, svc, alloc)
}

#[cfg(test)]
mod tests {
    use std::{collections::BTreeMap, path::PathBuf};

    use smol_str::SmolStr;

    use super::*;
    use crate::config::validate::{
        AllocationMode, DeviceSlot, Lifecycle, PlacementPolicy, ServiceConfig,
        test_fixtures::{expect_llama_cpp, minimal_command_service, minimal_service},
    };

    fn base_service() -> ServiceConfig {
        let mut placement = BTreeMap::new();
        placement.insert(DeviceSlot::Gpu(0), 10240);
        let mut svc = minimal_service("demo");
        svc.port = 11435;
        svc.private_port = 41000;
        svc.lifecycle = Lifecycle::Persistent;
        svc.placement_override = placement;
        svc.placement_policy = PlacementPolicy::GpuOnly;
        let lc = expect_llama_cpp(&mut svc);
        lc.model = PathBuf::from("/m/x.gguf");
        lc.context = Some(8192);
        lc.flash_attn = Some(true);
        lc.cache_type_k = Some(SmolStr::new("q8_0"));
        lc.cache_type_v = Some(SmolStr::new("q8_0"));
        svc
    }

    #[test]
    fn renders_core_flags() {
        let svc = base_service();
        let alloc = Allocation::from_override(&svc.placement_override);
        let cmd = render_argv(&svc, &alloc, None).unwrap();
        assert_eq!(cmd.binary, "llama-server");
        assert!(cmd.args.contains(&"-m".to_string()));
        assert!(cmd.args.iter().any(|a| a == "/m/x.gguf"));
        assert!(cmd.args.iter().any(|a| a == "-c"));
        assert!(cmd.args.iter().any(|a| a == "8192"));
        assert!(cmd.args.iter().any(|a| a == "-fa"));
        assert!(cmd.args.iter().any(|a| a == "--port"));
        assert!(cmd.args.iter().any(|a| a == "41000"));
        assert_eq!(cmd.env.get("CUDA_VISIBLE_DEVICES").unwrap(), "0");
    }

    #[test]
    fn renders_mmproj_when_present() {
        let mut svc = base_service();
        expect_llama_cpp(&mut svc).mmproj = Some(PathBuf::from("/m/x-mmproj.gguf"));
        let alloc = Allocation::from_override(&svc.placement_override);
        let cmd = render_argv(&svc, &alloc, None).unwrap();
        let idx = cmd.args.iter().position(|a| a == "--mmproj").unwrap();
        assert_eq!(cmd.args[idx + 1], "/m/x-mmproj.gguf");
    }

    #[test]
    fn cpu_only_renders_ngl_zero_and_empty_cuda_env() {
        let mut svc = base_service();
        svc.placement_policy = PlacementPolicy::CpuOnly;
        svc.placement_override.clear();
        svc.placement_override.insert(DeviceSlot::Cpu, 10240);
        let alloc = Allocation::from_override(&svc.placement_override);
        let cmd = render_argv(&svc, &alloc, None).unwrap();
        let ngl_idx = cmd.args.iter().position(|a| a == "-ngl").unwrap();
        assert_eq!(cmd.args[ngl_idx + 1], "0");
        assert_eq!(cmd.env.get("CUDA_VISIBLE_DEVICES").unwrap(), "");
    }

    #[test]
    fn placement_cmd_args_override_ngl_and_add_tensor_split() {
        let svc = base_service();
        let alloc = Allocation::from_override(&svc.placement_override);
        let ca = CommandArgs {
            ngl: Some(24),
            tensor_split: Some(vec![12, 12]),
            override_tensor: vec![],
        };
        let cmd = render_argv(&svc, &alloc, Some(&ca)).unwrap();
        let ngl_idx = cmd.args.iter().position(|a| a == "-ngl").unwrap();
        assert_eq!(cmd.args[ngl_idx + 1], "24");
        let ts_idx = cmd.args.iter().position(|a| a == "--tensor-split").unwrap();
        assert_eq!(cmd.args[ts_idx + 1], "12,12");
    }

    /// Regression for the scenario-01 `CUDA_VISIBLE_DEVICES=` empty-env bug:
    /// `SupervisorInit::allocation` is built from `placement_override` when the
    /// registry is constructed. For any estimator-driven service (no override),
    /// that bundle is empty, so rendering `render_argv` against it would emit
    /// `CUDA_VISIBLE_DEVICES=` and the child would silently fall back to CPU.
    /// The supervisor must thread the *packed* allocation into `render_argv`.
    /// This test demonstrates the discriminator: the two allocations produce
    /// different env values, so a regression that swaps back to `init.allocation`
    /// would be caught by the supervisor-level smoke path.
    #[test]
    fn render_uses_supplied_allocation_for_cuda_env() {
        let svc = base_service();
        // Empty override (estimator-driven) → `init.allocation` is empty.
        let empty_alloc = Allocation::from_override(&BTreeMap::new());
        let empty_cmd = render_argv(&svc, &empty_alloc, None).unwrap();
        assert_eq!(
            empty_cmd.env.get("CUDA_VISIBLE_DEVICES").unwrap(),
            "",
            "init.allocation with no GPU entries must render as empty (CPU fallback)"
        );

        // A packed allocation that placed layers on GPU 1 → env should list it.
        let mut placed = BTreeMap::new();
        placed.insert(DeviceSlot::Gpu(1), 4096);
        let packed_alloc = Allocation::from_override(&placed);
        let packed_cmd = render_argv(&svc, &packed_alloc, None).unwrap();
        assert_eq!(packed_cmd.env.get("CUDA_VISIBLE_DEVICES").unwrap(), "1");
    }

    #[test]
    fn command_template_renders_placeholders() {
        let command_argv = vec![
            "python".into(),
            "main.py".into(),
            "--port".into(),
            "{port}".into(),
        ];
        let mut placement = BTreeMap::new();
        placement.insert(DeviceSlot::Gpu(0), 6144);
        let mut svc = minimal_command_service("comfy", command_argv);
        svc.port = 8188;
        svc.private_port = 48188;
        svc.placement_override = placement.clone();
        svc.placement_policy = PlacementPolicy::GpuOnly;
        svc.allocation_mode = AllocationMode::Static { vram_mb: 6144 };
        let alloc = Allocation::from_override(&placement);
        let cfg = render_argv(&svc, &alloc, None).unwrap();
        assert_eq!(cfg.binary, "python");
        assert!(
            cfg.args.iter().any(|a| a == "48188"),
            "expected port substituted; got {:?}",
            cfg.args
        );
        assert!(
            cfg.args.iter().all(|a| a != "{port}"),
            "raw placeholder leaked into args: {:?}",
            cfg.args
        );
    }
}
