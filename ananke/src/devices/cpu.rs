//! Linux-only: scheduler view of CPU memory.
//!
//! Thin shim over [`crate::system::ProcFs::meminfo`] that converts to the
//! `CpuMemory` shape the snapshotter + daemon logging expect. Parsing
//! and `/proc/meminfo` I/O live inside `ProcFs`; this module just
//! renames the fields.

use crate::system::ProcFs;

#[derive(Debug, Clone, Copy)]
pub struct CpuMemory {
    pub total_bytes: u64,
    pub available_bytes: u64,
}

pub fn read(proc: &dyn ProcFs) -> std::io::Result<CpuMemory> {
    let m = proc.meminfo()?;
    Ok(CpuMemory {
        total_bytes: m.total_bytes,
        available_bytes: m.available_bytes,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::system::{InMemoryProcFs, Meminfo};

    #[test]
    fn read_goes_through_procfs() {
        let proc = InMemoryProcFs::new();
        proc.set_meminfo(Meminfo {
            total_bytes: 98_765_432 * 1024,
            available_bytes: 87_654_321 * 1024,
        });
        let m = read(&proc).unwrap();
        assert_eq!(m.total_bytes, 98_765_432 * 1024);
        assert_eq!(m.available_bytes, 87_654_321 * 1024);
    }

    #[test]
    fn read_errors_when_meminfo_missing() {
        let proc = InMemoryProcFs::new();
        assert!(read(&proc).is_err());
    }
}
