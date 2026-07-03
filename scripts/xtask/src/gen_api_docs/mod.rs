//! Generate `docs/api.md` from the dumped OpenAPI spec, interleaved with
//! hand-written prose from `gen_api_docs/prose/*.md`.
//!
//! The generator shells out to `cargo run --example dump-openapi`, parses
//! the JSON, renders Markdown for endpoints, and assembles the full
//! document by interleaving generated sections with the prose fragments.
//!
//! Each endpoint shows a **TypeScript type** for its request body and
//! success response. All schemas are expanded inline at their point of
//! use — there is no separate schema reference section.

mod render;
mod ts;

use std::{
    fmt, fs,
    path::{Path, PathBuf},
    process::{Command, Stdio},
};

use clap::Args;
use render::ProseFragments;
use serde_json::Value;

/// Prose fragments transcluded at compile time.
const PROSE_OVERVIEW: &str = include_str!("prose/overview.md");
const PROSE_OPENAI_API: &str = include_str!("prose/openai_api.md");
const PROSE_MANAGEMENT_API: &str = include_str!("prose/management_api.md");
const PROSE_PER_SERVICE_PROXY: &str = include_str!("prose/per_service_proxy.md");
const PROSE_SERVICE_STATES: &str = include_str!("prose/service_states.md");
const PROSE_EVENTS_WS: &str = include_str!("prose/events_ws.md");
const PROSE_LOGS_WS: &str = include_str!("prose/logs_ws.md");
const PROSE_ERRORS: &str = include_str!("prose/errors.md");
const PROSE_PROMETHEUS: &str = include_str!("prose/prometheus.md");
const PROSE_OPENAPI_SPEC: &str = include_str!("prose/openapi_spec.md");
const PROSE_FRONTEND: &str = include_str!("prose/frontend.md");

#[derive(Args)]
pub struct GenApiDocsArgs {
    /// Diff generated output against the committed `docs/api.md`; exit
    /// non-zero if different.
    #[arg(long)]
    pub check: bool,
}

pub fn run(args: GenApiDocsArgs) -> Result<(), Error> {
    let repo = repo_root()?;
    let docs_path = repo.join("docs/api.md");

    let spec = dump_openapi(&repo)?;
    let prose = ProseFragments {
        overview: PROSE_OVERVIEW,
        openai_api: PROSE_OPENAI_API,
        management_api: PROSE_MANAGEMENT_API,
        per_service_proxy: PROSE_PER_SERVICE_PROXY,
        service_states: PROSE_SERVICE_STATES,
        events_ws: PROSE_EVENTS_WS,
        logs_ws: PROSE_LOGS_WS,
        errors: PROSE_ERRORS,
        prometheus: PROSE_PROMETHEUS,
        openapi_spec: PROSE_OPENAPI_SPEC,
        frontend: PROSE_FRONTEND,
    };
    let markdown = render::render_markdown(&spec, &prose);

    if args.check {
        let existing = fs::read_to_string(&docs_path).map_err(|source| Error::Io {
            path: docs_path.clone(),
            source,
        })?;
        if existing != markdown {
            eprintln!("docs/api.md is stale; run `cargo xtask gen-api-docs` to regenerate");
            return Err(Error::Stale);
        }
        println!("docs/api.md is up to date");
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

fn dump_openapi(repo: &Path) -> Result<Value, Error> {
    let manifest = repo.join("Cargo.toml");
    let output = Command::new("cargo")
        .args([
            "run",
            "--quiet",
            "--manifest-path",
            manifest.to_str().unwrap(),
            "--package",
            "ananke",
            "--example",
            "dump-openapi",
        ])
        .env("ANANKE_SKIP_FRONTEND_BUILD", "1")
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .output()
        .map_err(Error::CargoSpawn)?;
    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        return Err(Error::DumpOpenapi {
            stderr: stderr.into_owned(),
        });
    }
    let stdout = String::from_utf8_lossy(&output.stdout);
    serde_json::from_str(&stdout).map_err(|source| Error::JsonParse { source })
}

// ── error type ──────────────────────────────────────────────────────────

#[derive(Debug)]
pub enum Error {
    CargoMetadata(cargo_metadata::Error),
    CargoSpawn(std::io::Error),
    DumpOpenapi {
        stderr: String,
    },
    JsonParse {
        source: serde_json::Error,
    },
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
            Self::CargoSpawn(source) => write!(f, "failed to spawn `cargo`: {source}"),
            Self::DumpOpenapi { stderr } => {
                let trimmed = stderr.trim();
                if trimmed.is_empty() {
                    write!(f, "`cargo run --example dump-openapi` failed")
                } else {
                    write!(f, "`cargo run --example dump-openapi` failed: {trimmed}")
                }
            }
            Self::JsonParse { source } => write!(f, "failed to parse OpenAPI JSON: {source}"),
            Self::Io { path, source } => write!(f, "i/o error on {}: {source}", path.display()),
            Self::Stale => write!(f, "docs/api.md is stale"),
        }
    }
}

impl std::error::Error for Error {}

// ── tests ────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use serde_json::json;

    use super::*;

    #[test]
    fn test_synthetic_openapi_to_markdown() {
        let spec = json!({
            "openapi": "3.0.0",
            "info": { "title": "Test", "version": "0.0.1" },
            "paths": {
                "/api/widgets": {
                    "get": {
                        "summary": "List widgets",
                        "description": "Returns all widgets.",
                        "parameters": [
                            {
                                "name": "limit",
                                "in": "query",
                                "required": false,
                                "description": "Max items"
                            }
                        ],
                        "responses": {
                            "200": {
                                "description": "OK",
                                "content": {
                                    "application/json": {
                                        "schema": {
                                            "$ref": "#/components/schemas/WidgetList"
                                        }
                                    }
                                }
                            },
                            "404": {
                                "description": "Not found"
                            }
                        }
                    },
                    "post": {
                        "summary": "Create widget",
                        "requestBody": {
                            "content": {
                                "application/json": {
                                    "schema": {
                                        "$ref": "#/components/schemas/WidgetRequest"
                                    }
                                }
                            }
                        },
                        "responses": {
                            "201": {
                                "description": "Created",
                                "content": {
                                    "application/json": {
                                        "schema": {
                                            "$ref": "#/components/schemas/Widget"
                                        }
                                    }
                                }
                            }
                        }
                    }
                },
                "/api/widgets/{id}": {
                    "get": {
                        "summary": "Get widget",
                        "parameters": [
                            {"name": "id", "in": "path", "required": true}
                        ],
                        "responses": {
                            "200": {
                                "description": "OK",
                                "content": {
                                    "application/json": {
                                        "schema": {
                                            "$ref": "#/components/schemas/Widget"
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            },
            "components": {
                "schemas": {
                    "WidgetList": {
                        "type": "object",
                        "description": "A list of widgets.",
                        "required": ["count"],
                        "properties": {
                            "count": {
                                "type": "integer",
                                "description": "Number of widgets."
                            }
                        }
                    },
                    "WidgetRequest": {
                        "type": "object",
                        "description": "Create widget request.",
                        "required": ["name"],
                        "properties": {
                            "name": {
                                "type": "string",
                                "description": "Widget name."
                            }
                        }
                    },
                    "Widget": {
                        "type": "object",
                        "description": "A widget.",
                        "required": ["id", "name"],
                        "properties": {
                            "id": {"type": "integer"},
                            "name": {"type": "string"}
                        }
                    },
                    "ApiErrorCodeSlug": {
                        "type": "string",
                        "enum": ["model_not_found", "insufficient_vram", "other"]
                    }
                }
            }
        });

        let prose = ProseFragments {
            overview: "Test overview.",
            openai_api: "",
            management_api: "",
            per_service_proxy: "",
            service_states: "",
            events_ws: "",
            logs_ws: "",
            errors: "",
            prometheus: "",
            openapi_spec: "",
            frontend: "",
        };

        let md = render::render_markdown(&spec, &prose);

        // Title present.
        assert!(md.contains("# Ananke API"));
        // Endpoint in summary table.
        assert!(md.contains("`GET`"));
        assert!(md.contains("`/api/widgets`"));
        // Endpoint detail section.
        assert!(md.contains("GET /api/widgets"));
        // Parameter table.
        assert!(md.contains("`limit`"));
        assert!(md.contains("query"));
        // Response table with type name.
        assert!(md.contains("WidgetList"));
        // TypeScript block present.
        assert!(md.contains("```typescript"));
        // TS type for request body: field name + type annotation.
        assert!(md.contains("name: string"));
        // TS type for response: count as number (TS convention).
        assert!(md.contains("count: number"));
        // No schema reference section — everything is inlined.
        assert!(!md.contains("## Schema reference"));
        // No type name labels on request/response headers.
        assert!(!md.contains("**Request body** (`"));
        assert!(!md.contains("**Response (200)** (`"));
        // Error code table.
        assert!(md.contains("`model_not_found`"));
        assert!(md.contains("`insufficient_vram`"));
    }
}
