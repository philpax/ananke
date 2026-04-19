//! Ephemeral port pool for oneshot services.
//!
//! Maintains a `BTreeSet` of available port numbers. Callers allocate a port
//! before spawning a oneshot supervisor and release it when the oneshot exits.

use std::{collections::BTreeSet, ops::Range};

/// Manages a bounded range of TCP ports for oneshot services.
pub struct PortPool {
    available: BTreeSet<u16>,
}

impl PortPool {
    /// Create a pool pre-populated with every port in `range`.
    pub fn new(range: Range<u16>) -> Self {
        Self {
            available: range.collect(),
        }
    }

    /// Allocate the smallest available port. Returns `None` when the pool is
    /// exhausted.
    pub fn allocate(&mut self) -> Option<u16> {
        let port = *self.available.iter().next()?;
        self.available.remove(&port);
        Some(port)
    }

    /// Return `port` to the pool. No-op if the port is already present.
    pub fn release(&mut self, port: u16) {
        self.available.insert(port);
    }

    /// Number of ports currently available.
    pub fn available(&self) -> usize {
        self.available.len()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn allocate_returns_smallest_first() {
        let mut pool = PortPool::new(18000..18003);
        assert_eq!(pool.allocate(), Some(18000));
        assert_eq!(pool.allocate(), Some(18001));
        assert_eq!(pool.allocate(), Some(18002));
        assert_eq!(pool.allocate(), None);
    }

    #[test]
    fn release_makes_port_available_again() {
        let mut pool = PortPool::new(18000..18002);
        let p1 = pool.allocate().unwrap();
        let p2 = pool.allocate().unwrap();
        assert_eq!(pool.available(), 0);
        pool.release(p1);
        assert_eq!(pool.available(), 1);
        pool.release(p2);
        assert_eq!(pool.available(), 2);
        // Re-allocating gives back the same ports.
        assert_eq!(pool.allocate(), Some(18000));
        assert_eq!(pool.allocate(), Some(18001));
    }
}
