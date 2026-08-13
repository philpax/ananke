//! Regenerate `frontend/src/api/types.ts` from the OpenAPI spec, or check that
//! the committed copy is current.
//!
//! The generation itself belongs to the frontend's `gen-types` npm script —
//! this shells out to it rather than restating the `openapi-typescript` and
//! `prettier` invocation, so there is one command line to keep correct.
//!
//! `--check` runs the same script and compares, which means it writes the file
//! and then puts back what was there. Generating into a temporary path instead
//! would change what prettier resolves as its config root, and a check that
//! formats differently from the real generator is worse than useless.

use std::{
    fmt, fs,
    path::{Path, PathBuf},
    process::Command,
};

use clap::Args;

/// Path of the generated module, relative to the repository root.
const TYPES_PATH: &str = "frontend/src/api/types.ts";

#[derive(Args)]
pub struct GenApiTypesArgs {
    /// Compare the regenerated output against the committed
    /// `frontend/src/api/types.ts`; exit non-zero if different.
    #[arg(long)]
    pub check: bool,
}

pub fn run(args: GenApiTypesArgs) -> Result<(), Error> {
    let repo = repo_root()?;
    let types_path = repo.join(TYPES_PATH);
    let frontend = repo.join("frontend");

    if !args.check {
        run_gen_types(&frontend)?;
        println!("wrote {}", types_path.display());
        return Ok(());
    }

    let committed = read(&types_path)?;
    run_gen_types(&frontend)?;
    let generated = read(&types_path)?;
    if generated != committed {
        write(&types_path, &committed)?;
        eprintln!("{TYPES_PATH} is stale; run `npm run gen-types` in frontend/ to regenerate");
        return Err(Error::Stale);
    }
    println!("{TYPES_PATH} is up to date");
    Ok(())
}

fn run_gen_types(frontend: &Path) -> Result<(), Error> {
    let status = Command::new("npm")
        .args(["run", "gen-types"])
        .current_dir(frontend)
        .env("ANANKE_SKIP_FRONTEND_BUILD", "1")
        .status()
        .map_err(Error::NpmSpawn)?;
    if !status.success() {
        return Err(Error::GenTypes { status });
    }
    Ok(())
}

fn repo_root() -> Result<PathBuf, Error> {
    let metadata = cargo_metadata::MetadataCommand::new()
        .no_deps()
        .exec()
        .map_err(Error::CargoMetadata)?;
    Ok(metadata.workspace_root.as_std_path().to_path_buf())
}

fn read(path: &Path) -> Result<String, Error> {
    fs::read_to_string(path).map_err(|source| Error::Io {
        path: path.to_path_buf(),
        source,
    })
}

fn write(path: &Path, contents: &str) -> Result<(), Error> {
    fs::write(path, contents).map_err(|source| Error::Io {
        path: path.to_path_buf(),
        source,
    })
}

// ── error type ──────────────────────────────────────────────────────────

#[derive(Debug)]
pub enum Error {
    CargoMetadata(cargo_metadata::Error),
    NpmSpawn(std::io::Error),
    GenTypes {
        status: std::process::ExitStatus,
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
            Self::CargoMetadata(source) => write!(f, "failed to read cargo metadata: {source}"),
            Self::NpmSpawn(source) => write!(f, "failed to run npm: {source}"),
            Self::GenTypes { status } => {
                write!(
                    f,
                    "failed to generate API types: npm run gen-types {status}"
                )
            }
            Self::Io { path, source } => {
                write!(f, "failed to access {}: {source}", path.display())
            }
            Self::Stale => write!(f, "{TYPES_PATH} is out of date"),
        }
    }
}

impl std::error::Error for Error {}
