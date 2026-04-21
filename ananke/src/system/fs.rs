//! Filesystem abstraction.
//!
//! Production code flows through [`LocalFs`], which delegates to `std::fs`.
//! Tests substitute [`InMemoryFs`] to avoid tempdir-based setup for any
//! path the daemon would otherwise touch: GGUF reads, orphan-reconcile
//! `/proc` lookups, config file loading, atomic config persist.
//!
//! Only the operations the daemon actually performs are exposed. Adding a
//! new callsite should add a method here rather than reaching for
//! `std::fs` directly.

use std::{
    collections::BTreeMap,
    io::{self, Cursor, Read, Seek},
    path::{Path, PathBuf},
    sync::Arc,
};

use parking_lot::RwLock;

/// Random-access reader. `Read + Seek` can't be combined into one trait
/// object directly in Rust, so this wrapper trait exists as the common
/// supertrait for `Box<dyn SeekRead + Send>`.
pub trait SeekRead: Read + Seek + Send {}
impl<T: Read + Seek + Send + ?Sized> SeekRead for T {}

/// Filesystem operations the daemon performs.
pub trait Fs: Send + Sync {
    /// Read an entire file into a byte buffer. For small config / `/proc`
    /// reads; prefer [`Fs::open`] for large files (GGUF weights).
    fn read(&self, path: &Path) -> io::Result<Vec<u8>>;

    /// Read an entire file into a `String`. UTF-8 required.
    fn read_to_string(&self, path: &Path) -> io::Result<String> {
        let bytes = self.read(path)?;
        String::from_utf8(bytes)
            .map_err(|e| io::Error::new(io::ErrorKind::InvalidData, e.to_string()))
    }

    /// Open a random-access reader. Used by the GGUF parser to walk
    /// multi-gigabyte files without materialising their contents.
    fn open(&self, path: &Path) -> io::Result<Box<dyn SeekRead>>;

    /// Write `bytes` to `path`, replacing any existing contents. Parent
    /// directories are created if missing.
    fn write(&self, path: &Path, bytes: &[u8]) -> io::Result<()>;

    /// Rename `from` to `to`, overwriting any existing `to`. Used by the
    /// config-persist path (write-tempfile-then-rename).
    fn rename(&self, from: &Path, to: &Path) -> io::Result<()>;

    /// Remove a single file. No-op if the path doesn't exist.
    fn remove_file(&self, path: &Path) -> io::Result<()>;

    fn exists(&self, path: &Path) -> bool;
}

/// Real filesystem. Every method forwards to `std::fs`.
#[derive(Default, Clone, Copy)]
pub struct LocalFs;

impl Fs for LocalFs {
    fn read(&self, path: &Path) -> io::Result<Vec<u8>> {
        std::fs::read(path)
    }

    fn open(&self, path: &Path) -> io::Result<Box<dyn SeekRead>> {
        let file = std::fs::File::open(path)?;
        Ok(Box::new(file))
    }

    fn write(&self, path: &Path, bytes: &[u8]) -> io::Result<()> {
        if let Some(parent) = path.parent()
            && !parent.as_os_str().is_empty()
        {
            std::fs::create_dir_all(parent)?;
        }
        std::fs::write(path, bytes)
    }

    fn rename(&self, from: &Path, to: &Path) -> io::Result<()> {
        std::fs::rename(from, to)
    }

    fn remove_file(&self, path: &Path) -> io::Result<()> {
        match std::fs::remove_file(path) {
            Ok(()) => Ok(()),
            Err(e) if e.kind() == io::ErrorKind::NotFound => Ok(()),
            Err(e) => Err(e),
        }
    }

    fn exists(&self, path: &Path) -> bool {
        path.exists()
    }
}

/// In-memory filesystem keyed by absolute path. Cloning shares the
/// underlying storage — tests can hold one handle and pass another into
/// daemon plumbing; writes through either are visible to the other.
#[derive(Default, Clone)]
pub struct InMemoryFs {
    inner: Arc<RwLock<BTreeMap<PathBuf, Vec<u8>>>>,
}

impl InMemoryFs {
    pub fn new() -> Self {
        Self::default()
    }

    /// Preload an entry. Convenience for `fs.with(path, bytes)` chaining.
    pub fn with(self, path: impl Into<PathBuf>, bytes: impl Into<Vec<u8>>) -> Self {
        self.inner.write().insert(path.into(), bytes.into());
        self
    }

    pub fn insert(&self, path: impl Into<PathBuf>, bytes: impl Into<Vec<u8>>) {
        self.inner.write().insert(path.into(), bytes.into());
    }

    /// Iterate every path currently stored. Useful for test assertions.
    pub fn paths(&self) -> Vec<PathBuf> {
        self.inner.read().keys().cloned().collect()
    }
}

impl Fs for InMemoryFs {
    fn read(&self, path: &Path) -> io::Result<Vec<u8>> {
        self.inner
            .read()
            .get(path)
            .cloned()
            .ok_or_else(|| io::Error::new(io::ErrorKind::NotFound, path.display().to_string()))
    }

    fn open(&self, path: &Path) -> io::Result<Box<dyn SeekRead>> {
        let bytes = self.read(path)?;
        Ok(Box::new(Cursor::new(bytes)))
    }

    fn write(&self, path: &Path, bytes: &[u8]) -> io::Result<()> {
        self.inner
            .write()
            .insert(path.to_path_buf(), bytes.to_vec());
        Ok(())
    }

    fn rename(&self, from: &Path, to: &Path) -> io::Result<()> {
        let mut guard = self.inner.write();
        let bytes = guard
            .remove(from)
            .ok_or_else(|| io::Error::new(io::ErrorKind::NotFound, from.display().to_string()))?;
        guard.insert(to.to_path_buf(), bytes);
        Ok(())
    }

    fn remove_file(&self, path: &Path) -> io::Result<()> {
        self.inner.write().remove(path);
        Ok(())
    }

    fn exists(&self, path: &Path) -> bool {
        self.inner.read().contains_key(path)
    }
}

/// Shared handle to a boxed [`Fs`]. Used by plumbing that needs to clone
/// the reference around (e.g. into task state).
pub type SharedFs = Arc<dyn Fs>;

/// Produce a [`SharedFs`] wrapping the real filesystem.
pub fn local_fs() -> SharedFs {
    Arc::new(LocalFs)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn in_memory_read_write_roundtrip() {
        let fs = InMemoryFs::new();
        fs.write(Path::new("/tmp/a"), b"hello").unwrap();
        assert_eq!(fs.read(Path::new("/tmp/a")).unwrap(), b"hello");
        assert!(fs.exists(Path::new("/tmp/a")));
        assert!(!fs.exists(Path::new("/tmp/missing")));
    }

    #[test]
    fn in_memory_rename_preserves_bytes() {
        let fs = InMemoryFs::new().with("/a", "payload");
        fs.rename(Path::new("/a"), Path::new("/b")).unwrap();
        assert!(!fs.exists(Path::new("/a")));
        assert_eq!(fs.read(Path::new("/b")).unwrap(), b"payload");
    }

    #[test]
    fn in_memory_open_is_seekable() {
        let fs = InMemoryFs::new().with("/x", "0123456789");
        let mut r = fs.open(Path::new("/x")).unwrap();
        let mut buf = [0u8; 4];
        r.read_exact(&mut buf).unwrap();
        assert_eq!(&buf, b"0123");
        r.seek(io::SeekFrom::Start(7)).unwrap();
        r.read_exact(&mut buf[..3]).unwrap();
        assert_eq!(&buf[..3], b"789");
    }

    #[test]
    fn in_memory_remove_file_is_idempotent() {
        let fs = InMemoryFs::new();
        fs.remove_file(Path::new("/missing")).unwrap();
        fs.write(Path::new("/present"), b"").unwrap();
        fs.remove_file(Path::new("/present")).unwrap();
        assert!(!fs.exists(Path::new("/present")));
    }

    #[test]
    fn shared_storage_across_clones() {
        let fs1 = InMemoryFs::new();
        let fs2 = fs1.clone();
        fs1.write(Path::new("/shared"), b"seen").unwrap();
        assert_eq!(fs2.read(Path::new("/shared")).unwrap(), b"seen");
    }
}
