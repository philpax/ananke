//! Render llama-server argv from an `EffectiveConfig` service entry, and spawn
//! the child with `prctl(PR_SET_PDEATHSIG, SIGTERM)`.

use std::collections::BTreeMap;
#[cfg(not(feature = "test-fakes"))]
use std::ffi::OsString;

#[cfg(not(feature = "test-fakes"))]
use nix::sys::prctl;
#[cfg(not(feature = "test-fakes"))]
use nix::sys::signal::Signal;
use tokio::process::{Child, Command};

use crate::config::validate::{PlacementPolicy, ServiceConfig, Template};
use crate::devices::{Allocation, cuda_env};
use crate::errors::ExpectedError;

pub struct SpawnConfig {
    pub binary: String,
    pub args: Vec<String>,
    pub env: BTreeMap<String, String>,
}

/// Render the child command line plus env from a validated `ServiceConfig`
/// and its `Allocation`.
pub fn render_argv(svc: &ServiceConfig, alloc: &Allocation) -> SpawnConfig {
    let mut args: Vec<String> = Vec::new();

    match svc.template {
        Template::LlamaCpp => {
            let raw = &svc.raw;
            args.push("-m".into());
            args.push(raw.model.as_ref().unwrap().to_string_lossy().into_owned());
            if let Some(mmproj) = &raw.mmproj {
                args.push("--mmproj".into());
                args.push(mmproj.to_string_lossy().into_owned());
            }
            if let Some(ctx) = raw.context {
                args.push("-c".into());
                args.push(ctx.to_string());
            }
            match svc.placement_policy {
                PlacementPolicy::CpuOnly => {
                    args.push("-ngl".into());
                    args.push("0".into());
                }
                PlacementPolicy::GpuOnly | PlacementPolicy::Hybrid => {
                    if let Some(ngl) = raw.n_gpu_layers {
                        args.push("-ngl".into());
                        args.push(ngl.to_string());
                    } else {
                        args.push("-ngl".into());
                        args.push("999".into());
                    }
                }
            }
            if raw.flash_attn == Some(true) {
                args.push("-fa".into());
                args.push("on".into());
            }
            if let Some(k) = &raw.cache_type_k {
                args.push("--cache-type-k".into());
                args.push(k.to_string());
            }
            if let Some(v) = &raw.cache_type_v {
                args.push("--cache-type-v".into());
                args.push(v.to_string());
            }
            if raw.jinja.unwrap_or(false) {
                args.push("--jinja".into());
            }
            if let Some(p) = &raw.chat_template_file {
                args.push("--chat-template-file".into());
                args.push(p.to_string_lossy().into_owned());
            }
            if let Some(t) = raw.threads {
                args.push("--threads".into());
                args.push(t.to_string());
            }
            if let Some(t) = raw.threads_batch {
                args.push("--threads-batch".into());
                args.push(t.to_string());
            }
            if let Some(b) = raw.batch_size {
                args.push("-b".into());
                args.push(b.to_string());
            }
            if let Some(b) = raw.ubatch_size {
                args.push("-ub".into());
                args.push(b.to_string());
            }
            if raw.mmap == Some(false) {
                args.push("--no-mmap".into());
            }
            if raw.mlock == Some(true) {
                args.push("--mlock".into());
            }
            if let Some(p) = raw.parallel {
                args.push("-np".into());
                args.push(p.to_string());
            }
            if let Some(rules) = &raw.override_tensor {
                for rule in rules {
                    args.push("-ot".into());
                    args.push(rule.clone());
                }
            }
            // Sampling params are passed as extra flags when set.
            if let Some(s) = &raw.sampling {
                if let Some(t) = s.get("temperature") {
                    args.push("--temp".into());
                    args.push(t.to_string());
                }
                if let Some(p) = s.get("top_p") {
                    args.push("--top-p".into());
                    args.push(p.to_string());
                }
                if let Some(k) = s.get("top_k") {
                    args.push("--top-k".into());
                    args.push(k.to_string());
                }
                if let Some(m) = s.get("min_p") {
                    args.push("--min-p".into());
                    args.push(m.to_string());
                }
                if let Some(r) = s.get("repeat_penalty") {
                    args.push("--repeat-penalty".into());
                    args.push(r.to_string());
                }
            }
            if let Some(extra) = &raw.extra_args {
                args.extend(extra.iter().cloned());
            }
            if let Some(extra) = &raw.extra_args_append {
                args.extend(extra.iter().cloned());
            }
            args.push("--host".into());
            args.push("127.0.0.1".into());
            args.push("--port".into());
            args.push(svc.private_port.to_string());
        }
    }

    let mut env = BTreeMap::new();
    if let Some(user_env) = &svc.raw.env {
        for (k, v) in user_env {
            env.insert(k.clone(), v.clone());
        }
    }
    env.insert("CUDA_VISIBLE_DEVICES".into(), cuda_env::render(alloc));

    SpawnConfig {
        binary: "llama-server".into(),
        args,
        env,
    }
}

/// Spawn the real llama-server child process.
///
/// Uses `prctl(PR_SET_PDEATHSIG, SIGTERM)` so the child is killed if the
/// daemon exits before explicitly terminating it.
#[cfg(not(feature = "test-fakes"))]
pub async fn spawn_child(cfg: &SpawnConfig) -> Result<Child, ExpectedError> {
    let mut cmd = Command::new(&cfg.binary);
    cmd.args(cfg.args.iter().map(OsString::from));
    cmd.env_clear();
    for (k, v) in &cfg.env {
        cmd.env(k, v);
    }
    cmd.stdin(std::process::Stdio::null());
    cmd.stdout(std::process::Stdio::piped());
    cmd.stderr(std::process::Stdio::piped());
    cmd.kill_on_drop(true);
    // SAFETY: pre_exec runs in the child after fork, before exec. Only
    // async-signal-safe operations are permitted here; prctl(2) is
    // async-signal-safe on Linux.
    unsafe {
        cmd.pre_exec(|| {
            prctl::set_pdeathsig(Signal::SIGTERM).map_err(std::io::Error::other)?;
            Ok(())
        });
    }
    cmd.spawn().map_err(|e| {
        ExpectedError::config_unparseable(
            std::path::PathBuf::from("<spawn>"),
            format!("spawn {}: {e}", cfg.binary),
        )
    })
}

/// Fake spawn for integration tests.
///
/// Spawns a long-running `/bin/sh -c 'sleep 300'` that keeps the process
/// alive while the echo server (co-located in the harness on the same
/// `private_port`) serves HTTP. `kill_on_drop(true)` ensures cleanup.
#[cfg(feature = "test-fakes")]
pub async fn spawn_child(_cfg: &SpawnConfig) -> Result<Child, ExpectedError> {
    let mut cmd = Command::new("/bin/sh");
    cmd.args(["-c", "sleep 300"]);
    cmd.stdin(std::process::Stdio::null());
    cmd.stdout(std::process::Stdio::piped());
    cmd.stderr(std::process::Stdio::piped());
    cmd.kill_on_drop(true);
    cmd.spawn().map_err(|e| {
        ExpectedError::config_unparseable(
            std::path::PathBuf::from("<fake-spawn>"),
            format!("fake spawn failed: {e}"),
        )
    })
}

#[cfg(test)]
mod tests {
    use std::collections::BTreeMap;
    use std::path::PathBuf;

    use smol_str::SmolStr;

    use super::*;
    use crate::config::parse::RawService;
    use crate::config::validate::{
        DeviceSlot, HealthSettings, Lifecycle, PlacementPolicy, ServiceConfig, Template,
    };

    fn base_service() -> ServiceConfig {
        let mut placement = BTreeMap::new();
        placement.insert(DeviceSlot::Gpu(0), 10240);
        let raw = RawService {
            name: Some(SmolStr::new("demo")),
            template: Some(SmolStr::new("llama-cpp")),
            model: Some(PathBuf::from("/m/x.gguf")),
            port: Some(11435),
            context: Some(8192),
            flash_attn: Some(true),
            cache_type_k: Some(SmolStr::new("q8_0")),
            cache_type_v: Some(SmolStr::new("q8_0")),
            ..Default::default()
        };
        ServiceConfig {
            name: SmolStr::new("demo"),
            template: Template::LlamaCpp,
            port: 11435,
            private_port: 41000,
            lifecycle: Lifecycle::Persistent,
            priority: 50,
            health: HealthSettings {
                http_path: "/v1/models".into(),
                timeout_ms: 180_000,
                probe_interval_ms: 5_000,
            },
            placement_override: placement,
            placement_policy: PlacementPolicy::GpuOnly,
            filters: Default::default(),
            idle_timeout_ms: 600_000,
            warming_grace_ms: 60_000,
            drain_timeout_ms: 30_000,
            extended_stream_drain_ms: 30_000,
            max_request_duration_ms: 600_000,
            raw,
        }
    }

    #[test]
    fn renders_core_flags() {
        let svc = base_service();
        let alloc = Allocation::from_override(&svc.placement_override);
        let cmd = render_argv(&svc, &alloc);
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
        svc.raw.mmproj = Some(PathBuf::from("/m/x-mmproj.gguf"));
        let alloc = Allocation::from_override(&svc.placement_override);
        let cmd = render_argv(&svc, &alloc);
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
        let cmd = render_argv(&svc, &alloc);
        let ngl_idx = cmd.args.iter().position(|a| a == "-ngl").unwrap();
        assert_eq!(cmd.args[ngl_idx + 1], "0");
        assert_eq!(cmd.env.get("CUDA_VISIBLE_DEVICES").unwrap(), "");
    }
}
