//! Linux /proc/meminfo reader.
//!
//! `MemAvailable` is used (spec §4.2), not `MemFree`, because `MemFree` ignores
//! reclaimable page cache and misleads the scheduler about how much memory can
//! actually be allocated to a new process.

use std::path::Path;

#[derive(Debug, Clone, Copy)]
pub struct CpuMemory {
    pub total_bytes: u64,
    pub available_bytes: u64,
}

pub fn read() -> std::io::Result<CpuMemory> {
    read_from(Path::new("/proc/meminfo"))
}

pub fn read_from(path: &Path) -> std::io::Result<CpuMemory> {
    let content = std::fs::read_to_string(path)?;
    parse_meminfo(&content)
        .ok_or_else(|| std::io::Error::other("meminfo missing MemTotal or MemAvailable"))
}

pub(crate) fn parse_meminfo(content: &str) -> Option<CpuMemory> {
    let mut total_kb = None;
    let mut avail_kb = None;
    for line in content.lines() {
        if let Some(rest) = line.strip_prefix("MemTotal:") {
            total_kb = parse_kb(rest);
        } else if let Some(rest) = line.strip_prefix("MemAvailable:") {
            avail_kb = parse_kb(rest);
        }
    }
    Some(CpuMemory { total_bytes: total_kb? * 1024, available_bytes: avail_kb? * 1024 })
}

fn parse_kb(rest: &str) -> Option<u64> {
    let trimmed = rest.trim().trim_end_matches("kB").trim();
    trimmed.parse::<u64>().ok()
}

#[cfg(test)]
mod tests {
    use super::*;

    const SAMPLE: &str = "\
MemTotal:       98765432 kB
MemFree:        12345678 kB
MemAvailable:   87654321 kB
Buffers:        1000000 kB
";

    #[test]
    fn parses_meminfo_sample() {
        let m = parse_meminfo(SAMPLE).unwrap();
        assert_eq!(m.total_bytes, 98_765_432 * 1024);
        assert_eq!(m.available_bytes, 87_654_321 * 1024);
    }

    #[test]
    fn returns_none_when_missing() {
        assert!(parse_meminfo("MemFree: 100 kB").is_none());
    }
}
