//! Substitute `{port}`, `{gpu_ids}`, `{vram_mb}`, `{model}`, `{name}`
//! in command-template argv and env values.

use std::collections::BTreeMap;

use crate::devices::{Allocation, DeviceId};

#[derive(Debug, Clone)]
pub struct PlaceholderContext<'a> {
    pub name: &'a str,
    pub port: u16,
    pub model: Option<&'a str>,
    pub allocation: &'a Allocation,
    /// Only populated for single-GPU static allocations; `None` on
    /// dynamic or multi-device, where `{vram_mb}` is a config error.
    pub static_vram_mb: Option<u64>,
}

#[derive(Debug)]
pub enum SubstituteError {
    VramMbOnDynamic,
    VramMbMultiDevice,
    UnknownPlaceholder(String),
    /// The launcher splat `{args}` must occupy a launcher entry on its
    /// own; it cannot be embedded inside a larger argv string because
    /// the expansion produces multiple arguments, not a single one.
    SplatInsideArg,
}

impl std::fmt::Display for SubstituteError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            SubstituteError::VramMbOnDynamic => {
                write!(f, "{{vram_mb}} is invalid with a dynamic allocation")
            }
            SubstituteError::VramMbMultiDevice => {
                write!(
                    f,
                    "{{vram_mb}} is valid only with a single-GPU static allocation"
                )
            }
            SubstituteError::UnknownPlaceholder(s) => {
                write!(f, "unknown placeholder {{{s}}}")
            }
            SubstituteError::SplatInsideArg => {
                write!(
                    f,
                    "splat placeholder {{args}} must be the entire launcher entry, \
                     not embedded inside a larger string"
                )
            }
        }
    }
}

impl std::error::Error for SubstituteError {}

/// Substitute every `{placeholder}` in `input` using `ctx`. Returns a
/// fresh owned String. Unknown placeholders produce a hard error so
/// typos surface rather than leaking literal `{oops}` into the argv.
///
/// `{{` and `}}` are escapes for a literal `{` / `}` — handy when the
/// embedded script itself uses braces (Python dicts, shell parameter
/// expansion, …) and shouldn't be mistaken for a placeholder.
pub fn substitute(input: &str, ctx: &PlaceholderContext<'_>) -> Result<String, SubstituteError> {
    let mut out = String::with_capacity(input.len());
    let mut rest = input;
    while !rest.is_empty() {
        // `{{` → literal `{`.
        if let Some(after) = rest.strip_prefix("{{") {
            out.push('{');
            rest = after;
            continue;
        }
        // `}}` → literal `}`.
        if let Some(after) = rest.strip_prefix("}}") {
            out.push('}');
            rest = after;
            continue;
        }
        if let Some(after_brace) = rest.strip_prefix('{') {
            // Placeholder: consume up to the matching `}`.
            if let Some(close) = after_brace.find('}') {
                let key = &after_brace[..close];
                let replacement = resolve(key, ctx)?;
                out.push_str(&replacement);
                rest = &after_brace[close + 1..];
                continue;
            }
            // Unmatched `{` at EOL — copy literal and stop.
            out.push('{');
            rest = after_brace;
            continue;
        }
        // Regular char run up to the next `{` or `}`.
        let next = rest.find(['{', '}']).unwrap_or(rest.len());
        out.push_str(&rest[..next]);
        rest = &rest[next..];
    }
    Ok(out)
}

pub fn resolve(key: &str, ctx: &PlaceholderContext<'_>) -> Result<String, SubstituteError> {
    match key {
        "port" => Ok(ctx.port.to_string()),
        "name" => Ok(ctx.name.to_string()),
        "model" => Ok(ctx.model.unwrap_or("").to_string()),
        "gpu_ids" => {
            let mut ids: Vec<u32> = ctx
                .allocation
                .bytes
                .keys()
                .filter_map(|id| match id {
                    DeviceId::Cpu => None,
                    DeviceId::Gpu(n) => Some(*n),
                })
                .collect();
            ids.sort_unstable();
            Ok(ids.iter().map(u32::to_string).collect::<Vec<_>>().join(","))
        }
        "vram_mb" => ctx
            .static_vram_mb
            .map(|mb| mb.to_string())
            .ok_or(SubstituteError::VramMbOnDynamic),
        other => Err(SubstituteError::UnknownPlaceholder(other.to_string())),
    }
}

/// Substitute a llama-cpp `launcher` argv template, expanding the splat
/// `{args}` placeholder to the full list of llama-server flags ananke
/// would otherwise have emitted. `{args}` must occupy a launcher entry
/// on its own — `["--foo={args}"]` is rejected because the expansion
/// produces multiple argv entries, not a single one.
///
/// Every other launcher entry passes through [`substitute`], so the
/// usual placeholders (`{model}`, `{port}`, `{name}`, `{gpu_ids}`) are
/// resolved in-place.
pub fn substitute_launcher_argv(
    launcher: &[String],
    llama_args: &[String],
    ctx: &PlaceholderContext<'_>,
) -> Result<Vec<String>, SubstituteError> {
    let mut out: Vec<String> = Vec::with_capacity(launcher.len() + llama_args.len());
    for entry in launcher {
        if entry == "{args}" {
            out.extend(llama_args.iter().cloned());
            continue;
        }
        if entry.contains("{args}") {
            return Err(SubstituteError::SplatInsideArg);
        }
        out.push(substitute(entry, ctx)?);
    }
    Ok(out)
}

/// Apply substitution across a whole argv vector and env map. Stops at
/// the first substitution error.
pub fn substitute_argv(
    argv: &[String],
    env: &BTreeMap<String, String>,
    ctx: &PlaceholderContext<'_>,
) -> Result<(Vec<String>, BTreeMap<String, String>), SubstituteError> {
    let argv_out: Vec<String> = argv
        .iter()
        .map(|a| substitute(a, ctx))
        .collect::<Result<Vec<_>, _>>()?;
    let mut env_out = BTreeMap::new();
    for (k, v) in env {
        env_out.insert(k.clone(), substitute(v, ctx)?);
    }
    Ok((argv_out, env_out))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::validate::DeviceSlot;

    fn alloc_gpu0_only() -> Allocation {
        let mut map = std::collections::BTreeMap::new();
        map.insert(DeviceSlot::Gpu(0), 6000);
        Allocation::from_override(&map)
    }

    fn alloc_cpu_only() -> Allocation {
        let mut map = std::collections::BTreeMap::new();
        map.insert(DeviceSlot::Cpu, 1000);
        Allocation::from_override(&map)
    }

    #[test]
    fn substitutes_common_placeholders() {
        let alloc = alloc_gpu0_only();
        let ctx = PlaceholderContext {
            name: "demo",
            port: 8188,
            model: Some("/m/x.gguf"),
            allocation: &alloc,
            static_vram_mb: Some(6000),
        };
        let out = substitute(
            "python main.py --port {port} --model {model} --gpu {gpu_ids} --vram {vram_mb}",
            &ctx,
        )
        .unwrap();
        assert_eq!(
            out,
            "python main.py --port 8188 --model /m/x.gguf --gpu 0 --vram 6000"
        );
    }

    #[test]
    fn vram_mb_on_dynamic_fails() {
        let alloc = alloc_gpu0_only();
        let ctx = PlaceholderContext {
            name: "demo",
            port: 8188,
            model: None,
            allocation: &alloc,
            static_vram_mb: None,
        };
        let err = substitute("--vram {vram_mb}", &ctx).unwrap_err();
        assert!(matches!(err, SubstituteError::VramMbOnDynamic));
    }

    #[test]
    fn gpu_ids_empty_for_cpu_only() {
        let alloc = alloc_cpu_only();
        let ctx = PlaceholderContext {
            name: "demo",
            port: 8188,
            model: None,
            allocation: &alloc,
            static_vram_mb: None,
        };
        let out = substitute("{gpu_ids}", &ctx).unwrap();
        assert_eq!(out, "");
    }

    #[test]
    fn unknown_placeholder_errors() {
        let alloc = alloc_gpu0_only();
        let ctx = PlaceholderContext {
            name: "demo",
            port: 8188,
            model: None,
            allocation: &alloc,
            static_vram_mb: None,
        };
        let err = substitute("{bogus}", &ctx).unwrap_err();
        assert!(matches!(err, SubstituteError::UnknownPlaceholder(_)));
    }

    #[test]
    fn literal_braces_pass_through() {
        let alloc = alloc_gpu0_only();
        let ctx = PlaceholderContext {
            name: "demo",
            port: 8188,
            model: None,
            allocation: &alloc,
            static_vram_mb: None,
        };
        // No close brace → literal.
        let out = substitute("prefix {not closed", &ctx).unwrap();
        assert_eq!(out, "prefix {not closed");
    }

    #[test]
    fn double_braces_escape_to_literals() {
        let alloc = alloc_gpu0_only();
        let ctx = PlaceholderContext {
            name: "demo",
            port: 8188,
            model: None,
            allocation: &alloc,
            static_vram_mb: None,
        };
        // `{{` / `}}` are escapes; the embedded script keeps its braces.
        let out = substitute("print(d[{{'k': 1}}]) on {port}", &ctx).unwrap();
        assert_eq!(out, "print(d[{'k': 1}]) on 8188");
    }
}
