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
///
/// Cgroup methods assume cgroup v2 unified hierarchy (modern systemd /
/// NixOS). Hosts still on v1 will see `cgroup_path` return `None` for
/// every pid; pledge attribution then falls back to the descendants-only
/// view.
pub trait ProcFs: Send + Sync {
    /// Parse `/proc/meminfo`, returning MemTotal + MemAvailable in bytes.
    /// `MemAvailable` is preferred over `MemFree` so page cache reclaim
    /// doesn't bias the scheduler.
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

    /// Parent pid from `/proc/<pid>/stat` (field 4). `None` when the pid
    /// has exited or stat parsing fails.
    ///
    /// Field 2 (`comm`) is parenthesised and may itself contain spaces,
    /// parens, or other punctuation, so a naive whitespace split is wrong.
    /// The parser here scans for the **last** `)` before splitting the
    /// remainder.
    fn parent_pid(&self, pid: u32) -> Option<u32>;

    /// All numeric pid entries currently visible under `/proc`. Used by
    /// the snapshotter to build a parent map once per tick.
    fn all_pids(&self) -> Vec<u32>;

    /// Cgroup v2 path from `/proc/<pid>/cgroup` (the value after `0::`).
    /// `None` when the pid has exited, the entry doesn't exist, or the
    /// host is on cgroup v1 (no `0::` line).
    fn cgroup_path(&self, pid: u32) -> Option<String>;
}

/// Transitive descendants of `root` via `proc.parent_pid()` walks. Includes
/// the root itself. Cheap — bounded by the number of currently-running
/// pids and the depth of the parent chain (typically ≤ 10).
///
/// For workloads that fork children (wrapper scripts, multi-process
/// servers), this captures every child whose VRAM/RSS the snapshotter
/// should attribute to the registered root pid. Containerised workloads
/// are NOT covered — the container is reparented out of the daemon's
/// process tree; cgroup-based attribution closes that gap separately.
pub fn descendants(proc: &dyn ProcFs, root: u32) -> Vec<u32> {
    descendants_from_map(&parent_map(proc), root)
}

/// Build a `pid → parent_pid` map from a single `/proc` walk. The
/// snapshotter calls this once per tick and reuses the map for every
/// service's descendant computation.
pub fn parent_map(proc: &dyn ProcFs) -> std::collections::BTreeMap<u32, u32> {
    let mut map = std::collections::BTreeMap::new();
    for pid in proc.all_pids() {
        if let Some(ppid) = proc.parent_pid(pid) {
            map.insert(pid, ppid);
        }
    }
    map
}

/// Pure-data version of [`descendants`] that consumes a pre-built parent
/// map. Splitting the walk from the `/proc` read lets callers reuse the
/// map across many roots without paying for repeated directory scans.
pub fn descendants_from_map(parents: &std::collections::BTreeMap<u32, u32>, root: u32) -> Vec<u32> {
    use std::collections::BTreeSet;
    let mut out = vec![root];
    let mut seen: BTreeSet<u32> = [root].into_iter().collect();
    let mut frontier: Vec<u32> = vec![root];
    while let Some(parent) = frontier.pop() {
        for (&child, &ppid) in parents {
            if ppid == parent && seen.insert(child) {
                out.push(child);
                frontier.push(child);
            }
        }
    }
    out
}

/// Pids whose cgroup v2 path equals `parent` or sits anywhere inside its
/// subtree. The match is exact-or-followed-by-`/` so `foo.slice` doesn't
/// match a sibling `foo.slice-evil.scope`.
pub fn pids_in_cgroup_subtree(proc: &dyn ProcFs, parent: &str) -> Vec<u32> {
    proc.all_pids()
        .into_iter()
        .filter(|pid| match proc.cgroup_path(*pid) {
            Some(cg) => cg == parent || cg.starts_with(&format!("{parent}/")),
            None => false,
        })
        .collect()
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

    fn parent_pid(&self, pid: u32) -> Option<u32> {
        let content = std::fs::read_to_string(format!("/proc/{pid}/stat")).ok()?;
        parse_parent_pid(&content)
    }

    fn all_pids(&self) -> Vec<u32> {
        let dir = match std::fs::read_dir("/proc") {
            Ok(d) => d,
            Err(_) => return Vec::new(),
        };
        dir.filter_map(|entry| {
            let entry = entry.ok()?;
            entry.file_name().to_str()?.parse::<u32>().ok()
        })
        .collect()
    }

    fn cgroup_path(&self, pid: u32) -> Option<String> {
        let content = std::fs::read_to_string(format!("/proc/{pid}/cgroup")).ok()?;
        parse_cgroup_v2(&content)
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
    parent: BTreeMap<u32, u32>,
    cgroup: BTreeMap<u32, String>,
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

    /// Preload `/proc/<pid>/stat`'s parent-pid field. Used by the
    /// snapshotter's descendants walk; tests express the process tree by
    /// stating each child's parent.
    pub fn set_parent(&self, child: u32, parent: u32) {
        self.inner.write().parent.insert(child, parent);
    }

    /// Preload `/proc/<pid>/cgroup`'s v2 path (the value after `0::`).
    /// Tests use this to model containerised pids whose cgroup sits
    /// under a service's declared `cgroup_parent`.
    pub fn set_cgroup(&self, pid: u32, path: impl Into<String>) {
        self.inner.write().cgroup.insert(pid, path.into());
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

    fn parent_pid(&self, pid: u32) -> Option<u32> {
        self.inner.read().parent.get(&pid).copied()
    }

    fn all_pids(&self) -> Vec<u32> {
        let s = self.inner.read();
        let mut out: std::collections::BTreeSet<u32> = std::collections::BTreeSet::new();
        out.extend(s.vm_rss.keys());
        out.extend(s.comm.keys());
        out.extend(s.parent.keys());
        out.extend(s.parent.values());
        out.extend(s.cgroup.keys());
        out.into_iter().collect()
    }

    fn cgroup_path(&self, pid: u32) -> Option<String> {
        self.inner.read().cgroup.get(&pid).cloned()
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

/// Parse `/proc/<pid>/stat` field 4 (parent pid). The `comm` field at
/// index 1 is parenthesised and may itself contain spaces, parens, or
/// other punctuation, so a naive whitespace split misattributes the
/// later columns. The fix is to scan for the **last** `)` and split the
/// remainder; ppid is then the second whitespace-separated token (the
/// first being `state`).
fn parse_parent_pid(stat: &str) -> Option<u32> {
    let close = stat.rfind(')')?;
    let tail = stat.get(close + 1..)?;
    tail.split_whitespace().nth(1)?.parse::<u32>().ok()
}

/// Parse `/proc/<pid>/cgroup`. Returns the v2 unified-hierarchy path
/// (the value after `0::`); `None` when no `0::` line is present (cgroup
/// v1 hosts, or pid exited).
fn parse_cgroup_v2(content: &str) -> Option<String> {
    for line in content.lines() {
        if let Some(rest) = line.strip_prefix("0::") {
            return Some(rest.trim_end().to_string());
        }
    }
    None
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

    /// Field 2 (`comm`) is parenthesised; the only safe parser scans to
    /// the last `)`. The `kthreadd` worker name `(sd-pam)` and the
    /// llama-server pattern `llama-server` both round-trip correctly.
    #[test]
    fn parses_parent_pid_with_simple_comm() {
        let stat = "1234 (llama-server) S 4321 1234 1234 0 -1 4194304 ...";
        assert_eq!(parse_parent_pid(stat), Some(4321));
    }

    /// Pathological `comm` values: the kernel allows up to 15 bytes of any
    /// printable character including spaces, parens, and trailing `)`. The
    /// "scan to last `)`" rule must isolate the structural close.
    #[test]
    fn parses_parent_pid_with_parens_in_comm() {
        let stat = "42 ((sd-pam)) S 1 42 42 0 -1 ...";
        assert_eq!(parse_parent_pid(stat), Some(1));
    }

    #[test]
    fn parses_parent_pid_with_spaces_in_comm() {
        let stat = "99 (foo bar) S 7 99 99 0 -1 ...";
        assert_eq!(parse_parent_pid(stat), Some(7));
    }

    /// Cgroup v2 unified hierarchy: a single line `0::<path>`. v1 hosts
    /// emit `<n>:<controller>:<path>` for every controller; we ignore those.
    #[test]
    fn parses_cgroup_v2_path() {
        let content = "0::/system.slice/docker-abc.scope\n";
        assert_eq!(
            parse_cgroup_v2(content).as_deref(),
            Some("/system.slice/docker-abc.scope")
        );
    }

    #[test]
    fn parses_cgroup_v2_ignores_v1_lines() {
        let content = "12:cpu,cpuacct:/foo\n0::/system.slice/bar.scope\n";
        assert_eq!(
            parse_cgroup_v2(content).as_deref(),
            Some("/system.slice/bar.scope")
        );
    }

    #[test]
    fn parses_cgroup_v2_returns_none_on_v1_only() {
        let content = "12:cpu,cpuacct:/foo\n11:memory:/foo\n";
        assert_eq!(parse_cgroup_v2(content), None);
    }

    /// Descendants walk: parent map `2→1, 3→1, 4→2, 5→4` rooted at 1
    /// must include every transitive child plus the root itself.
    #[test]
    fn descendants_walks_full_subtree() {
        let mut parents = BTreeMap::new();
        parents.insert(2u32, 1u32);
        parents.insert(3u32, 1u32);
        parents.insert(4u32, 2u32);
        parents.insert(5u32, 4u32);
        parents.insert(99u32, 50u32); // unrelated tree
        let mut out = descendants_from_map(&parents, 1);
        out.sort();
        assert_eq!(out, vec![1, 2, 3, 4, 5]);
    }

    /// A root with no children returns just itself.
    #[test]
    fn descendants_of_leaf_is_singleton() {
        let parents: BTreeMap<u32, u32> = BTreeMap::new();
        assert_eq!(descendants_from_map(&parents, 42), vec![42]);
    }

    /// `pids_in_cgroup_subtree` must accept exact match and descendant
    /// paths, but reject sibling cgroups whose name starts with the same
    /// prefix (`foo.slice` vs. `foo.slice-evil.scope`).
    #[test]
    fn cgroup_subtree_matches_exact_and_children_only() {
        let proc = InMemoryProcFs::new();
        proc.set_cgroup(10, "/system.slice/ananke-comfyui.slice");
        proc.set_cgroup(11, "/system.slice/ananke-comfyui.slice/docker-abc.scope");
        proc.set_cgroup(12, "/system.slice/ananke-comfyui.slice-evil.scope");
        proc.set_cgroup(13, "/system.slice/other.scope");
        let mut pids = pids_in_cgroup_subtree(&proc, "/system.slice/ananke-comfyui.slice");
        pids.sort();
        assert_eq!(pids, vec![10, 11]);
    }

    #[test]
    fn cgroup_subtree_returns_empty_when_no_match() {
        let proc = InMemoryProcFs::new();
        proc.set_cgroup(1, "/system.slice/other.scope");
        let pids = pids_in_cgroup_subtree(&proc, "/system.slice/missing.slice");
        assert!(pids.is_empty());
    }
}
