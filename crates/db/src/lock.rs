//! Linux-only: coordinates daemon and offline database operations with an adjacent advisory lock.

use std::{
    fs::{File, OpenOptions},
    io,
    path::{Path, PathBuf},
};

use nix::fcntl::{Flock, FlockArg};

#[derive(Debug)]
// The field is intentionally retained for its Drop-based lock lifetime.
#[expect(dead_code)]
pub(crate) struct DatabaseLock {
    file: Flock<File>,
}

impl DatabaseLock {
    pub(crate) fn acquire(database: &Path) -> io::Result<Self> {
        let file = OpenOptions::new()
            .create(true)
            .read(true)
            .write(true)
            .truncate(false)
            .open(database)?;
        let file = Flock::lock(file, FlockArg::LockExclusiveNonblock).map_err(|_| {
            io::Error::new(
                io::ErrorKind::WouldBlock,
                format!(
                    "database {} is held; stop the daemon before migrating",
                    database.display()
                ),
            )
        })?;
        Ok(Self { file })
    }
}

pub(crate) fn normalize_path(path: &Path, require_existing: bool) -> io::Result<PathBuf> {
    if require_existing {
        return std::fs::canonicalize(path);
    }
    if path.exists() {
        return std::fs::canonicalize(path);
    }
    let file_name = path.file_name().ok_or_else(|| {
        io::Error::new(io::ErrorKind::InvalidInput, "database path has no filename")
    })?;
    let parent = path.parent().unwrap_or_else(|| Path::new("."));
    Ok(std::fs::canonicalize(parent)?.join(file_name))
}
