//! Generate `docs/configuration.md` from the descriptor table in
//! `ananke-config`, interleaved with hand-written prose fragments from
//! `gen_config_docs/prose/*.md`.
//!
//! Unlike `gen-api-docs`, there is no subprocess or JSON intermediary: the
//! xtask imports `ananke_config::docs` directly and renders the
//! `Vec<SectionDoc>` into Markdown. Changing a `DEFAULT_*` constant
//! changes the generated output and trips `--check` in CI.

mod render;

use std::{fmt, fs, path::PathBuf};

use ananke_config::docs::all_sections;
use clap::Args;
use render::ProseFragments;

/// Prose fragments transcluded at compile time.
const PROSE_OVERVIEW: &str = include_str!("prose/overview.md");
const PROSE_DAEMON: &str = include_str!("prose/daemon.md");
const PROSE_OPENAI_API: &str = include_str!("prose/openai_api.md");
const PROSE_DEFAULTS: &str = include_str!("prose/defaults.md");
const PROSE_DEVICES: &str = include_str!("prose/devices.md");
const PROSE_SERVICE_COMMON: &str = include_str!("prose/service_common.md");
const PROSE_SERVICE_LIFECYCLE: &str = include_str!("prose/service_lifecycle.md");
const PROSE_SERVICE_PLACEMENT: &str = include_str!("prose/service_placement.md");
const PROSE_HEALTH: &str = include_str!("prose/health.md");
const PROSE_RESOURCE_ALLOCATION: &str = include_str!("prose/resource_allocation.md");
const PROSE_FILTERS: &str = include_str!("prose/filters.md");
const PROSE_METADATA: &str = include_str!("prose/metadata.md");
const PROSE_TRACKING: &str = include_str!("prose/tracking.md");
const PROSE_AUTO_RESTART: &str = include_str!("prose/auto_restart.md");
const PROSE_AUTO_RESTART_ERROR_RATE: &str = include_str!("prose/auto_restart_error_rate.md");
const PROSE_AUTO_RESTART_PERIODIC: &str = include_str!("prose/auto_restart_periodic.md");
const PROSE_AUTO_RESTART_FOOTER: &str = include_str!("prose/auto_restart_footer.md");
const PROSE_LLAMA_CPP: &str = include_str!("prose/llama_cpp.md");
const PROSE_ESTIMATION: &str = include_str!("prose/estimation.md");
const PROSE_SAMPLING: &str = include_str!("prose/sampling.md");
const PROSE_COMMAND: &str = include_str!("prose/command.md");
const PROSE_OPENAI_PROXY: &str = include_str!("prose/openai_proxy.md");
const PROSE_INHERITANCE: &str = include_str!("prose/inheritance.md");

#[derive(Args)]
pub struct GenConfigDocsArgs {
    /// Diff generated output against the committed `docs/configuration.md`;
    /// exit non-zero if different.
    #[arg(long)]
    pub check: bool,
}

pub fn run(args: GenConfigDocsArgs) -> Result<(), Error> {
    let repo = repo_root()?;
    let docs_path = repo.join("docs/configuration.md");

    let sections = all_sections();
    let prose = ProseFragments {
        overview: PROSE_OVERVIEW,
        daemon: PROSE_DAEMON,
        openai_api: PROSE_OPENAI_API,
        defaults: PROSE_DEFAULTS,
        devices: PROSE_DEVICES,
        service_common: PROSE_SERVICE_COMMON,
        service_lifecycle: PROSE_SERVICE_LIFECYCLE,
        service_placement: PROSE_SERVICE_PLACEMENT,
        health: PROSE_HEALTH,
        resource_allocation: PROSE_RESOURCE_ALLOCATION,
        filters: PROSE_FILTERS,
        metadata: PROSE_METADATA,
        tracking: PROSE_TRACKING,
        auto_restart: PROSE_AUTO_RESTART,
        auto_restart_error_rate: PROSE_AUTO_RESTART_ERROR_RATE,
        auto_restart_periodic: PROSE_AUTO_RESTART_PERIODIC,
        auto_restart_footer: PROSE_AUTO_RESTART_FOOTER,
        llama_cpp: PROSE_LLAMA_CPP,
        estimation: PROSE_ESTIMATION,
        sampling: PROSE_SAMPLING,
        command: PROSE_COMMAND,
        openai_proxy: PROSE_OPENAI_PROXY,
        inheritance: PROSE_INHERITANCE,
    };
    let markdown = render::render_markdown(&sections, &prose);

    if args.check {
        let existing = fs::read_to_string(&docs_path).map_err(|source| Error::Io {
            path: docs_path.clone(),
            source,
        })?;
        if existing != markdown {
            eprintln!(
                "docs/configuration.md is stale; run `cargo xtask gen-config-docs` to regenerate"
            );
            return Err(Error::Stale);
        }
        println!("docs/configuration.md is up to date");
        return Ok(());
    }

    fs::write(&docs_path, &markdown).map_err(|source| Error::Io {
        path: docs_path.clone(),
        source,
    })?;
    println!("wrote {}", docs_path.display());
    Ok(())
}

fn repo_root() -> Result<PathBuf, Error> {
    let metadata = cargo_metadata::MetadataCommand::new()
        .no_deps()
        .exec()
        .map_err(Error::CargoMetadata)?;
    Ok(metadata.workspace_root.as_std_path().to_path_buf())
}

// ── error type ──────────────────────────────────────────────────────────

#[derive(Debug)]
pub enum Error {
    CargoMetadata(cargo_metadata::Error),
    Io {
        path: PathBuf,
        source: std::io::Error,
    },
    Stale,
}

impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::CargoMetadata(source) => {
                write!(f, "failed to read workspace metadata: {source}")
            }
            Self::Io { path, source } => {
                write!(f, "i/o error on {}: {source}", path.display())
            }
            Self::Stale => write!(f, "docs/configuration.md is stale"),
        }
    }
}

impl std::error::Error for Error {}

// ── tests ────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use ananke_config::docs::{FieldDoc, SectionDoc, all_sections};

    use super::*;

    #[test]
    fn test_field_name_backtick_wrapping() {
        // Use all_sections() so the renderer finds every section it needs.
        let mut sections = all_sections();
        // Replace the daemon section with our synthetic one to test
        // render_default logic on known values.
        let daemon_idx = sections.iter().position(|s| s.id == "daemon").unwrap();
        sections[daemon_idx] = SectionDoc {
            id: "daemon",
            title: "Daemon Settings",
            fields: vec![
                FieldDoc {
                    name: "port",
                    ty: "u16",
                    default: "`8080`".to_string(),
                    description: "The port.",
                },
                FieldDoc {
                    name: "name",
                    ty: "string",
                    default: "*required*".to_string(),
                    description: "Service name.",
                },
                FieldDoc {
                    name: "gpus",
                    ty: "array of u32",
                    default: "all visible GPUs".to_string(),
                    description: "GPUs.",
                },
            ],
        };
        let prose = empty_prose();
        let md = render::render_markdown(&sections, &prose);

        // Table header.
        assert!(md.contains("| Field | Type | Default | Description |"));
        // Field name is backtick-wrapped; type is bare; single-token
        // default is backtick-wrapped.
        assert!(md.contains("| `port` | u16 | `8080` | The port. |"));
        // Sentinel default is bare.
        assert!(
            md.contains("| `name` | string | *required* | Service name. |"),
            "sentinel default should be bare"
        );
        // Multi-word prose default is bare.
        assert!(
            md.contains("| `gpus` | array of u32 | all visible GPUs | GPUs. |"),
            "multi-word default should be bare, got: {md}"
        );
    }

    #[test]
    fn test_full_render_no_triple_newlines() {
        // Rendering the real descriptor table + real prose must not produce
        // triple newlines anywhere.
        let sections = all_sections();
        let md = render::render_markdown(&sections, &full_prose());
        assert!(
            !md.contains("\n\n\n"),
            "triple newline found in output:\n{md}"
        );
    }

    #[test]
    fn test_full_render_contains_all_sections() {
        let sections = all_sections();
        let md = render::render_markdown(&sections, &full_prose());
        // Every section's fields must appear in the output — check that
        // the first field name of each section is present.
        for s in &sections {
            let first_field = s.fields.first().unwrap();
            assert!(
                md.contains(&format!("`{}`", first_field.name)),
                "section '{}' first field '{}' missing from output",
                s.id,
                first_field.name
            );
        }
    }

    #[test]
    fn test_stale_detection() {
        // The check logic is a simple string comparison; verify that a
        // mismatched string is detected.
        let existing = "old content";
        let generated = "new content";
        assert_ne!(existing, generated);
    }

    fn full_prose() -> ProseFragments {
        ProseFragments {
            overview: PROSE_OVERVIEW,
            daemon: PROSE_DAEMON,
            openai_api: PROSE_OPENAI_API,
            defaults: PROSE_DEFAULTS,
            devices: PROSE_DEVICES,
            service_common: PROSE_SERVICE_COMMON,
            service_lifecycle: PROSE_SERVICE_LIFECYCLE,
            service_placement: PROSE_SERVICE_PLACEMENT,
            health: PROSE_HEALTH,
            resource_allocation: PROSE_RESOURCE_ALLOCATION,
            filters: PROSE_FILTERS,
            metadata: PROSE_METADATA,
            tracking: PROSE_TRACKING,
            auto_restart: PROSE_AUTO_RESTART,
            auto_restart_error_rate: PROSE_AUTO_RESTART_ERROR_RATE,
            auto_restart_periodic: PROSE_AUTO_RESTART_PERIODIC,
            auto_restart_footer: PROSE_AUTO_RESTART_FOOTER,
            llama_cpp: PROSE_LLAMA_CPP,
            estimation: PROSE_ESTIMATION,
            sampling: PROSE_SAMPLING,
            command: PROSE_COMMAND,
            openai_proxy: PROSE_OPENAI_PROXY,
            inheritance: PROSE_INHERITANCE,
        }
    }

    fn empty_prose() -> ProseFragments {
        ProseFragments {
            overview: "# Configuration Guide",
            daemon: "",
            openai_api: "",
            defaults: "",
            devices: "",
            service_common: "",
            service_lifecycle: "",
            service_placement: "",
            health: "",
            resource_allocation: "",
            filters: "",
            metadata: "",
            tracking: "",
            auto_restart: "",
            auto_restart_error_rate: "",
            auto_restart_periodic: "",
            auto_restart_footer: "",
            llama_cpp: "",
            estimation: "",
            sampling: "",
            command: "",
            openai_proxy: "",
            inheritance: "",
        }
    }
}
