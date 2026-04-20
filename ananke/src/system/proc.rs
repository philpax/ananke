//! Linux-only: `/proc` abstraction.
//!
//! `/proc` isn't really a filesystem — every read is a synthesised view
//! of kernel state — so routing it through `Fs` made every test stage
//! synthetic text, only to have the consumer parse it back into a
//! semantic value. This module models the daemon's actual reads as
//! typed trait methods. The production impl reads `/proc` directly; the
//! test impl takes pre-parsed values keyed by pid.
//!
//! Every `/proc` read the daemon performs should go through here. Adding
//! a new read should add a method rather than reaching for `std::fs`
//! directly.
//!
//! Linux-only: `LocalProcFs` assumes `/proc` exists and follows the
//! kernel conventions (NUL-separated cmdline, "VmRSS:" key in status,
//! etc.). Non-Linux hosts would need a different impl.

#[cfg(any(test, feature = "test-fakes"))]
use std::collections::BTreeMap;
use std::io;

#[cfg(any(test, feature = "test-fakes"))]
use parking_lot::RwLock;

/// Parsed `/proc/meminfo` values the scheduler cares about.
#[derive(Debug, Clone, Copy)]
pub struct Meminfo {
    pub total_bytes: u64,
    pub available_bytes: u64,
}

/// Semantic `/proc` reader. Production: [`LocalProcFs`]. Tests:
/// [`InMemoryProcFs`], which accepts pre-parsed values instead of
/// synthesised text.
pub trait ProcFs: Send + Sync {
    /// Parse `/proc/meminfo`, returning MemTotal + MemAvailable in bytes.
    /// `MemAvailable` (spec §4.2) is preferred over `MemFree` so page
    /// cache reclaim doesn't bias the scheduler.
    fn meminfo(&self) -> io::Result<Meminfo>;

    /// `VmRSS` from `/proc/<pid>/status` in bytes. `None` when the pid
    /// has exited or the status entry isn't fully populated yet.
    fn vm_rss(&self, pid: u32) -> Option<u64>;

    /// `/proc/<pid>/comm` as the raw command name (trimmed). `None` when
    /// the pid has exited.
    fn comm(&self, pid: u32) -> Option<String>;

    /// `/proc/<pid>/cmdline` with NULs replaced by spaces. `None` when
    /// the pid has exited. Used by orphan reconciliation to verify that
    /// a recorded pid still belongs to the recorded service.
    fn cmdline(&self, pid: i32) -> Option<String>;
}

/// Real `/proc` reader. Every method shells out to `std::fs`.
#[derive(Default, Clone, Copy)]
pub struct LocalProcFs;

impl ProcFs for LocalProcFs {
    fn meminfo(&self) -> io::Result<Meminfo> {
        let content = std::fs::read_to_string("/proc/meminfo")?;
        parse_meminfo(&content)
            .ok_or_else(|| io::Error::other("meminfo missing MemTotal or MemAvailable"))
    }

    fn vm_rss(&self, pid: u32) -> Option<u64> {
        let content = std::fs::read_to_string(format!("/proc/{pid}/status")).ok()?;
        parse_vm_rss(&content)
    }

    fn comm(&self, pid: u32) -> Option<String> {
        std::fs::read_to_string(format!("/proc/{pid}/comm"))
            .ok()
            .map(|s| s.trim().to_string())
    }

    fn cmdline(&self, pid: i32) -> Option<String> {
        let raw = std::fs::read(format!("/proc/{pid}/cmdline")).ok()?;
        Some(null_sep_to_space(&raw))
    }
}

/// Test impl keyed on pid. Callers pre-populate the values a test run
/// needs to see, including a single shared `meminfo`; reads that don't
/// match a registered value return the "exited" signal (`None` /
/// `NotFound` io error).
#[cfg(any(test, feature = "test-fakes"))]
#[derive(Default, Clone)]
pub struct InMemoryProcFs {
    inner: std::sync::Arc<RwLock<InMemoryProcFsState>>,
}

#[cfg(any(test, feature = "test-fakes"))]
#[derive(Default)]
struct InMemoryProcFsState {
    meminfo: Option<Meminfo>,
    vm_rss: BTreeMap<u32, u64>,
    comm: BTreeMap<u32, String>,
    cmdline: BTreeMap<i32, String>,
}

#[cfg(any(test, feature = "test-fakes"))]
impl InMemoryProcFs {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn set_meminfo(&self, m: Meminfo) {
        self.inner.write().meminfo = Some(m);
    }

    pub fn set_vm_rss(&self, pid: u32, bytes: u64) {
        self.inner.write().vm_rss.insert(pid, bytes);
    }

    pub fn set_comm(&self, pid: u32, comm: impl Into<String>) {
        self.inner.write().comm.insert(pid, comm.into());
    }

    /// Preload the `/proc/<pid>/cmdline` value the orphan reconciler
    /// would otherwise read off disk. Pass the command as a single
    /// space-separated string.
    pub fn set_cmdline(&self, pid: i32, cmdline: impl Into<String>) {
        self.inner.write().cmdline.insert(pid, cmdline.into());
    }
}

#[cfg(any(test, feature = "test-fakes"))]
impl ProcFs for InMemoryProcFs {
    fn meminfo(&self) -> io::Result<Meminfo> {
        self.inner
            .read()
            .meminfo
            .ok_or_else(|| io::Error::new(io::ErrorKind::NotFound, "meminfo not preloaded"))
    }

    fn vm_rss(&self, pid: u32) -> Option<u64> {
        self.inner.read().vm_rss.get(&pid).copied()
    }

    fn comm(&self, pid: u32) -> Option<String> {
        self.inner.read().comm.get(&pid).cloned()
    }

    fn cmdline(&self, pid: i32) -> Option<String> {
        self.inner.read().cmdline.get(&pid).cloned()
    }
}

fn parse_meminfo(content: &str) -> Option<Meminfo> {
    let mut total_kb = None;
    let mut avail_kb = None;
    for line in content.lines() {
        if let Some(rest) = line.strip_prefix("MemTotal:") {
            total_kb = parse_kb(rest);
        } else if let Some(rest) = line.strip_prefix("MemAvailable:") {
            avail_kb = parse_kb(rest);
        }
    }
    Some(Meminfo {
        total_bytes: total_kb? * 1024,
        available_bytes: avail_kb? * 1024,
    })
}

fn parse_vm_rss(content: &str) -> Option<u64> {
    for line in content.lines() {
        if let Some(rest) = line.strip_prefix("VmRSS:") {
            let kb = rest
                .trim()
                .trim_end_matches("kB")
                .trim()
                .parse::<u64>()
                .ok()?;
            return Some(kb * 1024);
        }
    }
    None
}

fn parse_kb(rest: &str) -> Option<u64> {
    let trimmed = rest.trim().trim_end_matches("kB").trim();
    trimmed.parse::<u64>().ok()
}

fn null_sep_to_space(bytes: &[u8]) -> String {
    let mut s: String = bytes
        .iter()
        .map(|b| if *b == 0 { ' ' } else { *b as char })
        .collect();
    // The kernel emits a trailing NUL after the last arg, which becomes a
    // trailing space here.
    s.truncate(s.trim_end().len());
    s
}

#[cfg(test)]
mod tests {
    use super::*;

    const SAMPLE_MEMINFO: &str = "\
MemTotal:       98765432 kB
MemFree:        12345678 kB
MemAvailable:   87654321 kB
Buffers:        1000000 kB
";

    const SAMPLE_STATUS: &str = "\
Name:\tllama-server
VmPeak:\t 1000000 kB
VmRSS:\t  524288 kB
";

    #[test]
    fn parses_meminfo() {
        let m = parse_meminfo(SAMPLE_MEMINFO).unwrap();
        assert_eq!(m.total_bytes, 98_765_432 * 1024);
        assert_eq!(m.available_bytes, 87_654_321 * 1024);
    }

    #[test]
    fn parses_vm_rss() {
        assert_eq!(parse_vm_rss(SAMPLE_STATUS), Some(524_288 * 1024));
        assert_eq!(parse_vm_rss("Name:\tfoo\n"), None);
    }

    #[test]
    fn null_sep_joins() {
        // `/proc/<pid>/cmdline` is NUL-separated with a trailing NUL.
        assert_eq!(
            null_sep_to_space(b"llama-server\0-m\0model.gguf\0"),
            "llama-server -m model.gguf"
        );
    }

    #[test]
    fn in_memory_round_trips_preloaded_values() {
        let proc = InMemoryProcFs::new();
        proc.set_meminfo(Meminfo {
            total_bytes: 1024,
            available_bytes: 512,
        });
        proc.set_vm_rss(4242, 8192);
        proc.set_comm(4242, "llama-server");
        proc.set_cmdline(4242, "llama-server -m model.gguf");

        assert_eq!(proc.meminfo().unwrap().available_bytes, 512);
        assert_eq!(proc.vm_rss(4242), Some(8192));
        assert_eq!(proc.comm(4242), Some("llama-server".into()));
        assert_eq!(
            proc.cmdline(4242).as_deref(),
            Some("llama-server -m model.gguf")
        );

        // Pids that weren't preloaded look exited.
        assert_eq!(proc.vm_rss(9999), None);
        assert_eq!(proc.comm(9999), None);
        assert_eq!(proc.cmdline(9999), None);
    }
}
