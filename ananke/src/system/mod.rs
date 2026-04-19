//! System-boundary abstractions.
//!
//! Every place the daemon talks to the outside world — filesystem, `/proc`,
//! eventually child-process spawning — goes through a trait in this module
//! so tests can substitute an in-memory fake. The goal is: nothing in the
//! scheduler, allocator, or supervisor state machine should depend on real
//! disk/kernel state in a test context.
//!
//! Currently covers the filesystem ([`Fs`]). Future additions (process
//! snapshot, signal sender) land as siblings.

pub mod fs;

pub use fs::{Fs, InMemoryFs, LocalFs, SeekRead};
