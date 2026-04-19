//! GPU probe trait and associated data types.

/// Probe trait for querying GPU state.
pub trait GpuProbe: Send + Sync {
    fn list(&self) -> Vec<GpuInfo>;
    fn query(&self, id: u32) -> Option<GpuMemory>;
    fn processes(&self, id: u32) -> Vec<GpuProcess>;
}

/// Static information about a GPU device.
#[derive(Debug, Clone)]
pub struct GpuInfo {
    pub id: u32,
    pub name: String,
    pub total_bytes: u64,
}

/// Current memory usage of a GPU device.
#[derive(Debug, Clone)]
pub struct GpuMemory {
    pub total_bytes: u64,
    pub free_bytes: u64,
}

/// A process using GPU memory.
#[derive(Debug, Clone)]
pub struct GpuProcess {
    pub pid: u32,
    pub used_bytes: u64,
    pub name: String,
}
