//! Release flow: bump the workspace version, commit, and tag locally.
//!
//! The flow deliberately stops before any remote-affecting operation
//! (no `git push`, no `gh release`) so the operator can inspect the
//! commit and the tag before publishing them. Once `git push origin
//! v<version>` lands, the Release workflow (`.github/workflows/release.yml`)
//! takes over and produces the actual GitHub release.

use std::{
    fmt, fs,
    path::{Path, PathBuf},
    process::{Command, Stdio},
};

use clap::Parser;
use toml_edit::{DocumentMut, value};

#[derive(Parser)]
pub struct Args {
    /// New workspace version (without leading `v`), e.g. 0.2.0.
    pub version: String,

    /// Proceed even if the working tree has uncommitted changes.
    #[arg(long)]
    pub allow_dirty: bool,

    /// Branch on which a release is allowed. Defaults to `main`.
    #[arg(long, default_value = "main")]
    pub branch: String,

    /// Proceed even if the current branch does not match `--branch`.
    #[arg(long)]
    pub allow_branch_mismatch: bool,

    /// Print the actions that would be taken without performing them.
    #[arg(long)]
    pub dry_run: bool,
}

pub fn run(args: Args) -> Result<(), Error> {
    validate_version(&args.version)?;

    let repo = repo_root()?;
    let cargo_toml = repo.join("Cargo.toml");
    let cargo_lock = repo.join("Cargo.lock");
    let package_json = repo.join("frontend/package.json");
    let package_lock = repo.join("frontend/package-lock.json");

    let old = read_workspace_version(&cargo_toml)?;
    if old == args.version {
        return Err(Error::SameVersion { version: old });
    }

    if !args.allow_dirty {
        ensure_clean_tree(&repo)?;
    }
    ensure_on_branch(&repo, &args.branch, args.allow_branch_mismatch)?;
    let tag = format!("v{}", args.version);
    ensure_tag_absent(&repo, &tag)?;

    let plan = Plan::build(
        &cargo_toml,
        &cargo_lock,
        &package_json,
        &package_lock,
        &old,
        &args.version,
    )?;

    println!("Bumping workspace version: {old} -> {}", args.version);
    println!("Tag to create:             {tag}");
    println!("Files to update:");
    for file in &plan.files {
        println!("  {}", file.path.display());
    }

    if args.dry_run {
        println!();
        println!("Dry run; no files written.");
        return Ok(());
    }

    plan.write()?;

    git(
        &repo,
        &[
            "add",
            "Cargo.toml",
            "Cargo.lock",
            "frontend/package.json",
            "frontend/package-lock.json",
        ],
    )?;
    git(&repo, &["commit", "-m", &format!("chore(release): {tag}")])?;
    git(&repo, &["tag", "-a", &tag, "-m", &format!("Release {tag}")])?;

    println!();
    println!("Created commit and annotated tag {tag}.");
    println!("Push when ready:");
    println!("  git push --follow-tags");
    Ok(())
}

#[derive(Debug)]
pub enum Error {
    InvalidVersion {
        value: String,
        reason: String,
    },
    SameVersion {
        version: String,
    },
    DirtyTree {
        lines: usize,
    },
    WrongBranch {
        expected: String,
        actual: String,
    },
    TagExists(String),
    GitSpawn(std::io::Error),
    GitCommand {
        command: String,
        stderr: String,
    },
    Io {
        path: PathBuf,
        source: std::io::Error,
    },
    TomlParse {
        path: PathBuf,
        source: toml_edit::TomlError,
    },
    JsonParse {
        path: PathBuf,
        source: serde_json::Error,
    },
    MissingKey {
        path: PathBuf,
        key: String,
    },
    VersionMismatch {
        path: PathBuf,
        package: String,
        expected: String,
        found: String,
    },
    WorkspaceCratesIncomplete {
        found: usize,
        expected: usize,
    },
    UnexpectedMatchCount {
        path: PathBuf,
        needle: String,
        found: usize,
        expected: usize,
    },
    CargoMetadata(cargo_metadata::Error),
}

impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidVersion { value, reason } => {
                write!(f, "invalid version `{value}`: {reason}")
            }
            Self::SameVersion { version } => {
                write!(f, "workspace is already at version {version}")
            }
            Self::DirtyTree { lines } => write!(
                f,
                "working tree has {lines} uncommitted change(s); rerun with --allow-dirty to override"
            ),
            Self::WrongBranch { expected, actual } => write!(
                f,
                "on branch `{actual}`, expected `{expected}`; rerun with --allow-branch-mismatch to override"
            ),
            Self::TagExists(tag) => write!(f, "tag `{tag}` already exists"),
            Self::GitSpawn(source) => write!(f, "failed to spawn `git`: {source}"),
            Self::GitCommand { command, stderr } => {
                let trimmed = stderr.trim();
                if trimmed.is_empty() {
                    write!(f, "`{command}` failed")
                } else {
                    write!(f, "`{command}` failed: {trimmed}")
                }
            }
            Self::Io { path, source } => write!(f, "i/o error on {}: {source}", path.display()),
            Self::TomlParse { path, source } => {
                write!(f, "failed to parse {} as TOML: {source}", path.display())
            }
            Self::JsonParse { path, source } => {
                write!(f, "failed to parse {} as JSON: {source}", path.display())
            }
            Self::MissingKey { path, key } => {
                write!(f, "missing key `{key}` in {}", path.display())
            }
            Self::VersionMismatch {
                path,
                package,
                expected,
                found,
            } => write!(
                f,
                "{}: package `{package}` is at version {found}, expected {expected}",
                path.display()
            ),
            Self::WorkspaceCratesIncomplete { found, expected } => write!(
                f,
                "found {found} of {expected} expected workspace crate entries in Cargo.lock"
            ),
            Self::UnexpectedMatchCount {
                path,
                needle,
                found,
                expected,
            } => write!(
                f,
                "expected {expected} occurrence(s) of `{needle}` in {}, found {found}",
                path.display()
            ),
            Self::CargoMetadata(source) => write!(f, "failed to read workspace metadata: {source}"),
        }
    }
}

impl std::error::Error for Error {}

struct Plan {
    files: Vec<PlannedFile>,
}

struct PlannedFile {
    path: PathBuf,
    new_content: String,
}

impl Plan {
    fn build(
        cargo_toml: &Path,
        cargo_lock: &Path,
        package_json: &Path,
        package_lock: &Path,
        old: &str,
        new: &str,
    ) -> Result<Self, Error> {
        let members = workspace_member_names(cargo_toml)?;
        let files = vec![
            PlannedFile {
                path: cargo_toml.to_path_buf(),
                new_content: rewrite_cargo_toml(cargo_toml, new)?,
            },
            PlannedFile {
                path: cargo_lock.to_path_buf(),
                new_content: rewrite_cargo_lock(cargo_lock, &members, old, new)?,
            },
            PlannedFile {
                path: package_json.to_path_buf(),
                new_content: rewrite_package_json(package_json, old, new)?,
            },
            PlannedFile {
                path: package_lock.to_path_buf(),
                new_content: rewrite_package_lock(package_lock, old, new)?,
            },
        ];
        Ok(Self { files })
    }

    fn write(&self) -> Result<(), Error> {
        for file in &self.files {
            write(&file.path, &file.new_content)?;
        }
        Ok(())
    }
}

fn workspace_member_names(cargo_toml: &Path) -> Result<Vec<String>, Error> {
    let metadata = cargo_metadata::MetadataCommand::new()
        .manifest_path(cargo_toml)
        .no_deps()
        .exec()
        .map_err(Error::CargoMetadata)?;
    Ok(metadata
        .workspace_packages()
        .into_iter()
        .map(|p| p.name.clone())
        .collect())
}

fn validate_version(v: &str) -> Result<(), Error> {
    if let Some(stripped) = v.strip_prefix('v') {
        return Err(Error::InvalidVersion {
            value: v.to_string(),
            reason: format!("drop the leading `v` (use `{stripped}` instead)"),
        });
    }
    let core = v.split(['-', '+']).next().unwrap_or(v);
    let parts: Vec<&str> = core.split('.').collect();
    let core_ok = parts.len() == 3
        && parts
            .iter()
            .all(|p| !p.is_empty() && p.chars().all(|c| c.is_ascii_digit()));
    if !core_ok {
        return Err(Error::InvalidVersion {
            value: v.to_string(),
            reason: "expected MAJOR.MINOR.PATCH semver core".to_string(),
        });
    }
    Ok(())
}

fn repo_root() -> Result<PathBuf, Error> {
    let output = Command::new("git")
        .args(["rev-parse", "--show-toplevel"])
        .output()
        .map_err(Error::GitSpawn)?;
    if !output.status.success() {
        return Err(Error::GitCommand {
            command: "git rev-parse --show-toplevel".to_string(),
            stderr: String::from_utf8_lossy(&output.stderr).into_owned(),
        });
    }
    Ok(PathBuf::from(
        String::from_utf8_lossy(&output.stdout).trim(),
    ))
}

fn ensure_clean_tree(repo: &Path) -> Result<(), Error> {
    let output = Command::new("git")
        .current_dir(repo)
        .args(["status", "--porcelain"])
        .output()
        .map_err(Error::GitSpawn)?;
    if !output.status.success() {
        return Err(Error::GitCommand {
            command: "git status --porcelain".to_string(),
            stderr: String::from_utf8_lossy(&output.stderr).into_owned(),
        });
    }
    if !output.stdout.is_empty() {
        let lines = String::from_utf8_lossy(&output.stdout)
            .trim_end()
            .lines()
            .count();
        return Err(Error::DirtyTree { lines });
    }
    Ok(())
}

fn ensure_on_branch(repo: &Path, expected: &str, allow_mismatch: bool) -> Result<(), Error> {
    let output = Command::new("git")
        .current_dir(repo)
        .args(["rev-parse", "--abbrev-ref", "HEAD"])
        .output()
        .map_err(Error::GitSpawn)?;
    if !output.status.success() {
        return Err(Error::GitCommand {
            command: "git rev-parse --abbrev-ref HEAD".to_string(),
            stderr: String::from_utf8_lossy(&output.stderr).into_owned(),
        });
    }
    let actual = String::from_utf8_lossy(&output.stdout).trim().to_string();
    if actual != expected && !allow_mismatch {
        return Err(Error::WrongBranch {
            expected: expected.to_string(),
            actual,
        });
    }
    Ok(())
}

fn ensure_tag_absent(repo: &Path, tag: &str) -> Result<(), Error> {
    let status = Command::new("git")
        .current_dir(repo)
        .args(["rev-parse", "--verify", &format!("refs/tags/{tag}")])
        .stdout(Stdio::null())
        .stderr(Stdio::null())
        .status()
        .map_err(Error::GitSpawn)?;
    if status.success() {
        return Err(Error::TagExists(tag.to_string()));
    }
    Ok(())
}

fn git(repo: &Path, args: &[&str]) -> Result<(), Error> {
    let output = Command::new("git")
        .current_dir(repo)
        .args(args)
        .output()
        .map_err(Error::GitSpawn)?;
    if !output.status.success() {
        return Err(Error::GitCommand {
            command: format!("git {}", args.join(" ")),
            stderr: String::from_utf8_lossy(&output.stderr).into_owned(),
        });
    }
    Ok(())
}

fn read_workspace_version(path: &Path) -> Result<String, Error> {
    let content = read(path)?;
    let doc: DocumentMut = content.parse().map_err(|source| Error::TomlParse {
        path: path.to_path_buf(),
        source,
    })?;
    let v = doc
        .get("workspace")
        .and_then(|w| w.as_table())
        .and_then(|w| w.get("package"))
        .and_then(|p| p.as_table())
        .and_then(|p| p.get("version"))
        .and_then(|v| v.as_str())
        .ok_or_else(|| Error::MissingKey {
            path: path.to_path_buf(),
            key: "workspace.package.version".to_string(),
        })?;
    Ok(v.to_string())
}

fn rewrite_cargo_toml(path: &Path, new_version: &str) -> Result<String, Error> {
    let content = read(path)?;
    let mut doc: DocumentMut = content.parse().map_err(|source| Error::TomlParse {
        path: path.to_path_buf(),
        source,
    })?;
    let workspace = doc
        .get_mut("workspace")
        .and_then(|w| w.as_table_mut())
        .ok_or_else(|| Error::MissingKey {
            path: path.to_path_buf(),
            key: "workspace".to_string(),
        })?;
    let package = workspace
        .get_mut("package")
        .and_then(|p| p.as_table_mut())
        .ok_or_else(|| Error::MissingKey {
            path: path.to_path_buf(),
            key: "workspace.package".to_string(),
        })?;
    package["version"] = value(new_version);
    Ok(doc.to_string())
}

fn rewrite_cargo_lock(
    path: &Path,
    members: &[String],
    old: &str,
    new_version: &str,
) -> Result<String, Error> {
    let content = read(path)?;
    let mut doc: DocumentMut = content.parse().map_err(|source| Error::TomlParse {
        path: path.to_path_buf(),
        source,
    })?;
    let packages = doc
        .get_mut("package")
        .and_then(|p| p.as_array_of_tables_mut())
        .ok_or_else(|| Error::MissingKey {
            path: path.to_path_buf(),
            key: "package".to_string(),
        })?;
    let mut bumped = 0usize;
    for pkg in packages.iter_mut() {
        let name = pkg
            .get("name")
            .and_then(|n| n.as_str())
            .unwrap_or_default()
            .to_string();
        if !members.iter().any(|m| m == &name) {
            continue;
        }
        let current = pkg
            .get("version")
            .and_then(|v| v.as_str())
            .unwrap_or_default()
            .to_string();
        if current != old {
            return Err(Error::VersionMismatch {
                path: path.to_path_buf(),
                package: name,
                expected: old.to_string(),
                found: current,
            });
        }
        pkg["version"] = value(new_version);
        bumped += 1;
    }
    if bumped != members.len() {
        return Err(Error::WorkspaceCratesIncomplete {
            found: bumped,
            expected: members.len(),
        });
    }
    Ok(doc.to_string())
}

fn rewrite_package_json(path: &Path, old: &str, new_version: &str) -> Result<String, Error> {
    let content = read(path)?;
    let parsed: serde_json::Value =
        serde_json::from_str(&content).map_err(|source| Error::JsonParse {
            path: path.to_path_buf(),
            source,
        })?;
    let current = parsed
        .get("version")
        .and_then(|v| v.as_str())
        .unwrap_or_default();
    if current != old {
        return Err(Error::VersionMismatch {
            path: path.to_path_buf(),
            package: "frontend".to_string(),
            expected: old.to_string(),
            found: current.to_string(),
        });
    }
    let needle = format!("\"version\": \"{old}\"");
    let count = content.matches(&needle).count();
    if count != 1 {
        return Err(Error::UnexpectedMatchCount {
            path: path.to_path_buf(),
            needle,
            found: count,
            expected: 1,
        });
    }
    Ok(content.replacen(&needle, &format!("\"version\": \"{new_version}\""), 1))
}

fn rewrite_package_lock(path: &Path, old: &str, new_version: &str) -> Result<String, Error> {
    let content = read(path)?;
    // The project's own version appears twice near the top of the file
    // (top-level + `packages[""]`). Every other version field belongs to
    // a dependency under a `node_modules/...` key. Bounding the search to
    // the head keeps the substitution surgical without parsing the whole
    // lockfile (which would reorder keys without `preserve_order`).
    let cutoff = content.find("\"node_modules/").unwrap_or(content.len());
    let (head, tail) = content.split_at(cutoff);
    let needle = format!("\"version\": \"{old}\"");
    let count = head.matches(&needle).count();
    if count != 2 {
        return Err(Error::UnexpectedMatchCount {
            path: path.to_path_buf(),
            needle,
            found: count,
            expected: 2,
        });
    }
    let new_head = head.replace(&needle, &format!("\"version\": \"{new_version}\""));
    Ok(format!("{new_head}{tail}"))
}

fn read(path: &Path) -> Result<String, Error> {
    fs::read_to_string(path).map_err(|source| Error::Io {
        path: path.to_path_buf(),
        source,
    })
}

fn write(path: &Path, content: &str) -> Result<(), Error> {
    fs::write(path, content).map_err(|source| Error::Io {
        path: path.to_path_buf(),
        source,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn validate_version_accepts_semver_core() {
        assert!(validate_version("0.1.0").is_ok());
        assert!(validate_version("1.2.3").is_ok());
        assert!(validate_version("10.20.30").is_ok());
        assert!(validate_version("1.0.0-alpha.1").is_ok());
        assert!(validate_version("1.0.0+build.5").is_ok());
    }

    #[test]
    fn validate_version_rejects_leading_v() {
        let err = validate_version("v1.2.3").unwrap_err();
        assert!(matches!(err, Error::InvalidVersion { .. }));
    }

    #[test]
    fn validate_version_rejects_non_semver() {
        assert!(validate_version("1.2").is_err());
        assert!(validate_version("1.2.3.4").is_err());
        assert!(validate_version("1.x.0").is_err());
        assert!(validate_version("").is_err());
    }
}
