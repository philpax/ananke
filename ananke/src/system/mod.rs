//! System-boundary abstractions.
//!
//! Every place the daemon talks to the outside world — filesystem,
//! child-process spawning + signalling, and (later) `/proc` readers —
//! goes through a trait in this module so tests can substitute an
//! in-memory fake. The goal is: nothing in the scheduler, allocator, or
//! supervisor state machine should depend on real disk/kernel state in
//! a test context.
//!
//! # The `SystemDeps` bundle
//!
//! Individual traits live in their own submodules (`fs`, `process`, and
//! eventually `clock`), but the daemon and tests pass a single
//! [`SystemDeps`] that composes them. Construct the production bundle
//! with [`SystemDeps::local`]; tests use [`SystemDeps::fake`] which also
//! hands back the concrete fakes so assertions can inspect their state.
//! New outside-world capabilities should be added as (a) a trait under
//! this module and (b) a new field on `SystemDeps`.

use std::sync::Arc;

pub mod fs;
pub mod proc;
pub mod process;

pub use fs::{Fs, InMemoryFs, LocalFs, SeekRead};
#[cfg(any(test, feature = "test-fakes"))]
pub use proc::InMemoryProcFs;
pub use proc::{LocalProcFs, Meminfo, ProcFs};
pub use process::{DynAsyncRead, LocalSpawner, ManagedChild, ProcessSpawner};
#[cfg(any(test, feature = "test-fakes"))]
pub use process::{FakeChildSnapshot, FakeProcessState, FakeSpawner};

/// Every capability the daemon needs from the outside world, bundled so
/// that supervisors, handlers, and tests receive one struct rather than a
/// drifting list of loose parameters.
///
/// Clone is cheap — every field is `Arc`-backed.
#[derive(Clone)]
pub struct SystemDeps {
    pub fs: Arc<dyn Fs>,
    pub proc: Arc<dyn ProcFs>,
    pub process_spawner: Arc<dyn ProcessSpawner>,
}

impl SystemDeps {
    /// Production bundle: real filesystem, real `/proc` reader, real
    /// process spawner with `PR_SET_PDEATHSIG`.
    pub fn local() -> Self {
        Self {
            fs: Arc::new(LocalFs),
            proc: Arc::new(LocalProcFs),
            process_spawner: Arc::new(LocalSpawner),
        }
    }
}

#[cfg(any(test, feature = "test-fakes"))]
impl SystemDeps {
    /// Fake bundle: in-memory filesystem, in-memory `/proc`, in-memory
    /// process spawner. Also returns the concrete handles so tests can
    /// preload GGUF bytes, stage `/proc` state, or inspect
    /// [`FakeChildSnapshot`] after a drain.
    pub fn fake() -> (Self, FakeBag) {
        let fs = InMemoryFs::new();
        let proc = InMemoryProcFs::new();
        let process_spawner = Arc::new(FakeSpawner::new());
        let deps = Self {
            fs: Arc::new(fs.clone()),
            proc: Arc::new(proc.clone()),
            process_spawner: process_spawner.clone(),
        };
        (
            deps,
            FakeBag {
                fs,
                proc,
                process_spawner,
            },
        )
    }
}

/// Test helper: the concrete fakes backing a `SystemDeps` produced by
/// [`SystemDeps::fake`].
#[cfg(any(test, feature = "test-fakes"))]
pub struct FakeBag {
    pub fs: InMemoryFs,
    pub proc: InMemoryProcFs,
    pub process_spawner: Arc<FakeSpawner>,
}
