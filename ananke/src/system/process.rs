//! Process spawn + lifecycle abstraction.
//!
//! The supervisor state machine never touches `tokio::process` or `nix`
//! directly. Instead it calls methods on a [`ProcessSpawner`] (to start a
//! new child) and on the returned [`ManagedChild`] (to read stdout/stderr,
//! wait for exit, or terminate). Two implementations live here:
//!
//! - [`LocalSpawner`]: wraps `tokio::process::Command` + `nix::sys::signal`
//!   for real children. Used by the daemon in production.
//! - [`FakeSpawner`]: produces purely in-memory [`FakeChild`] handles with
//!   virtual pids, no OS processes. Exported under the `test-fakes` feature
//!   so integration tests can assert "service X's child was terminated"
//!   without probing real pids or sleeping.
//!
//! This boundary is the single place in the crate that owns the child
//! process model. Keeping it here means a scheduler regression test can
//! observe the effects of a config reload on process state without waiting
//! on wall-clock or racing with OS cleanup.

use std::{io, pin::Pin, process::ExitStatus};

use async_trait::async_trait;
use tokio::io::AsyncRead;

use crate::{errors::ExpectedError, supervise::spawn::SpawnConfig};

/// Owned reader for a child's stdout or stderr.
///
/// Boxed trait object so production can hand out `tokio::process::ChildStdout`
/// and tests can substitute `tokio::io::empty()` without changing the consumer
/// signature. The log-pump code adds its own `BufReader` on top.
pub type DynAsyncRead = Pin<Box<dyn AsyncRead + Send + Unpin + 'static>>;

/// Spawn a child process described by [`SpawnConfig`]. Implementations are
/// expected to be cheap to clone (everything is `Arc`-backed or unit-sized)
/// and thread-safe — the daemon stores a single `Arc<dyn ProcessSpawner>`
/// inside `SupervisorDeps` and every supervisor borrows it.
#[async_trait]
pub trait ProcessSpawner: Send + Sync + 'static {
    async fn spawn(&self, cfg: &SpawnConfig) -> Result<Box<dyn ManagedChild>, ExpectedError>;
}

/// Handle to a running child process. Dropped handles are expected to
/// release OS resources; [`LocalChild`] achieves this via
/// `Command::kill_on_drop`.
#[async_trait]
pub trait ManagedChild: Send + 'static {
    /// Process identifier — an OS pid for the real impl, a virtual counter
    /// for the fake. `None` only if the child has already been reaped.
    fn id(&self) -> Option<u32>;

    /// Take ownership of the child's stdout reader, if not already consumed.
    fn take_stdout(&mut self) -> Option<DynAsyncRead>;

    /// Take ownership of the child's stderr reader.
    fn take_stderr(&mut self) -> Option<DynAsyncRead>;

    /// Await the child's exit.
    async fn wait(&mut self) -> io::Result<ExitStatus>;

    /// Request graceful termination (SIGTERM on Unix). Does not block on
    /// the child actually exiting — callers that need confirmation should
    /// combine with [`wait`](Self::wait) under a timeout.
    async fn sigterm(&mut self) -> io::Result<()>;

    /// Force-kill (SIGKILL on Unix). Idempotent.
    async fn sigkill(&mut self) -> io::Result<()>;
}

// ---------------------------------------------------------------------------
// Production impl: tokio::process + nix signals.
// ---------------------------------------------------------------------------

use std::ffi::OsString;

use nix::{
    sys::{prctl, signal::Signal as NixSignal},
    unistd::Pid,
};
use tokio::process::{ChildStderr, ChildStdout, Command};

/// Spawn real children via `tokio::process`. Sets `PR_SET_PDEATHSIG =
/// SIGTERM` in the child so a daemon crash takes its kids with it.
pub struct LocalSpawner;

#[async_trait]
impl ProcessSpawner for LocalSpawner {
    async fn spawn(&self, cfg: &SpawnConfig) -> Result<Box<dyn ManagedChild>, ExpectedError> {
        let mut cmd = Command::new(&cfg.binary);
        cmd.args(cfg.args.iter().map(OsString::from));
        cmd.env_clear();
        for (k, v) in &cfg.env {
            cmd.env(k, v);
        }
        cmd.stdin(std::process::Stdio::null());
        cmd.stdout(std::process::Stdio::piped());
        cmd.stderr(std::process::Stdio::piped());
        cmd.kill_on_drop(true);
        // SAFETY: pre_exec runs in the child after fork, before exec. Only
        // async-signal-safe operations are permitted here; prctl(2) is
        // async-signal-safe on Linux.
        unsafe {
            cmd.pre_exec(|| {
                prctl::set_pdeathsig(NixSignal::SIGTERM).map_err(io::Error::other)?;
                Ok(())
            });
        }
        let mut inner = cmd.spawn().map_err(|e| {
            ExpectedError::config_unparseable(
                std::path::PathBuf::from("<spawn>"),
                format!("spawn {}: {e}", cfg.binary),
            )
        })?;
        let pid = inner.id();
        let stdout: Option<DynAsyncRead> = inner
            .stdout
            .take()
            .map(|s: ChildStdout| Box::pin(s) as DynAsyncRead);
        let stderr: Option<DynAsyncRead> = inner
            .stderr
            .take()
            .map(|s: ChildStderr| Box::pin(s) as DynAsyncRead);
        Ok(Box::new(LocalChild {
            inner,
            pid,
            stdout,
            stderr,
        }))
    }
}

struct LocalChild {
    inner: tokio::process::Child,
    pid: Option<u32>,
    stdout: Option<DynAsyncRead>,
    stderr: Option<DynAsyncRead>,
}

#[async_trait]
impl ManagedChild for LocalChild {
    fn id(&self) -> Option<u32> {
        self.pid
    }

    fn take_stdout(&mut self) -> Option<DynAsyncRead> {
        self.stdout.take()
    }

    fn take_stderr(&mut self) -> Option<DynAsyncRead> {
        self.stderr.take()
    }

    async fn wait(&mut self) -> io::Result<ExitStatus> {
        self.inner.wait().await
    }

    async fn sigterm(&mut self) -> io::Result<()> {
        let Some(pid) = self.pid else {
            return Ok(());
        };
        match nix::sys::signal::kill(Pid::from_raw(pid as i32), NixSignal::SIGTERM) {
            Ok(()) => Ok(()),
            // The child may already have exited; SIGTERM on a reaped pid is
            // a no-op, not an error.
            Err(nix::errno::Errno::ESRCH) => Ok(()),
            Err(e) => Err(io::Error::other(e)),
        }
    }

    async fn sigkill(&mut self) -> io::Result<()> {
        self.inner.kill().await
    }
}

// ---------------------------------------------------------------------------
// Test impl: virtual children with no OS state.
// ---------------------------------------------------------------------------

#[cfg(any(test, feature = "test-fakes"))]
pub use fake::{FakeChildSnapshot, FakeProcessState, FakeSpawner};

#[cfg(any(test, feature = "test-fakes"))]
mod fake {
    use std::sync::Arc;

    use parking_lot::Mutex;
    use tokio::sync::watch;

    use super::*;

    /// Externally-visible state of a fake child. Tests snapshot the vector
    /// returned by [`FakeSpawner::children`] and assert on transitions.
    #[derive(Debug, Clone, PartialEq, Eq)]
    pub enum FakeProcessState {
        /// The supervisor spawned this child and hasn't called terminate.
        Running,
        /// The supervisor called `sigterm`.
        SigTerm,
        /// The supervisor called `sigkill` (either directly or after SIGTERM).
        SigKill,
        /// A test helper forced this child to self-exit.
        SelfExited { code: i32 },
    }

    #[derive(Debug, Clone)]
    pub struct FakeChildSnapshot {
        pub pid: u32,
        pub binary: String,
        pub args: Vec<String>,
        pub env: std::collections::BTreeMap<String, String>,
        pub state: FakeProcessState,
    }

    struct ChildSlot {
        pid: u32,
        binary: String,
        args: Vec<String>,
        env: std::collections::BTreeMap<String, String>,
        state: FakeProcessState,
        /// Published `true` exactly once when the child transitions out of
        /// `Running`. `watch` (rather than `oneshot`) so a supervisor that
        /// re-awaits `wait()` across `tokio::select!` cancellations doesn't
        /// observe a spurious "child exited" on the second call.
        exit_tx: watch::Sender<bool>,
    }

    struct Inner {
        next_pid: u32,
        slots: Vec<Arc<Mutex<ChildSlot>>>,
        ignore_sigterm: bool,
    }

    /// In-memory spawner. Every call to [`spawn`](ProcessSpawner::spawn)
    /// records a new child with a fresh virtual pid; tests inspect state
    /// via [`children`](Self::children).
    pub struct FakeSpawner {
        inner: Arc<Mutex<Inner>>,
    }

    impl FakeSpawner {
        pub fn new() -> Self {
            Self {
                inner: Arc::new(Mutex::new(Inner {
                    next_pid: 1000,
                    slots: Vec::new(),
                    ignore_sigterm: false,
                })),
            }
        }

        /// Variant whose children silently drop SIGTERM — used to exercise
        /// the SIGTERM→SIGKILL escalation in the drain pipeline without
        /// spawning real processes with `trap '' TERM`.
        pub fn ignoring_sigterm() -> Self {
            let me = Self::new();
            me.inner.lock().ignore_sigterm = true;
            me
        }

        /// Snapshot every child this spawner has ever produced, in order.
        pub fn children(&self) -> Vec<FakeChildSnapshot> {
            self.inner
                .lock()
                .slots
                .iter()
                .map(|slot| {
                    let s = slot.lock();
                    FakeChildSnapshot {
                        pid: s.pid,
                        binary: s.binary.clone(),
                        args: s.args.clone(),
                        env: s.env.clone(),
                        state: s.state.clone(),
                    }
                })
                .collect()
        }

        /// Resolve a child by its virtual pid. Returns `None` if no slot
        /// exists for `pid`.
        pub fn child(&self, pid: u32) -> Option<FakeChildSnapshot> {
            self.children().into_iter().find(|c| c.pid == pid)
        }

        /// Force the child with `pid` to self-exit. Tests use this to
        /// exercise the supervisor's "child exited unexpectedly" branches
        /// without any real process behaviour.
        pub fn exit(&self, pid: u32, code: i32) -> bool {
            let inner = self.inner.lock();
            let Some(slot) = inner.slots.iter().find(|s| s.lock().pid == pid) else {
                return false;
            };
            let mut s = slot.lock();
            if !matches!(s.state, FakeProcessState::Running) {
                return false;
            }
            s.state = FakeProcessState::SelfExited { code };
            let _ = s.exit_tx.send(true);
            true
        }
    }

    impl Default for FakeSpawner {
        fn default() -> Self {
            Self::new()
        }
    }

    #[async_trait]
    impl ProcessSpawner for FakeSpawner {
        async fn spawn(&self, cfg: &SpawnConfig) -> Result<Box<dyn ManagedChild>, ExpectedError> {
            let (tx, rx) = watch::channel(false);
            let (slot, ignore_sigterm) = {
                let mut inner = self.inner.lock();
                let pid = inner.next_pid;
                inner.next_pid += 1;
                let slot = Arc::new(Mutex::new(ChildSlot {
                    pid,
                    binary: cfg.binary.clone(),
                    args: cfg.args.clone(),
                    env: cfg.env.clone(),
                    state: FakeProcessState::Running,
                    exit_tx: tx,
                }));
                inner.slots.push(slot.clone());
                (slot, inner.ignore_sigterm)
            };
            Ok(Box::new(FakeChild {
                slot,
                exit_rx: rx,
                stdout: Some(Box::pin(tokio::io::empty()) as DynAsyncRead),
                stderr: Some(Box::pin(tokio::io::empty()) as DynAsyncRead),
                ignore_sigterm,
            }))
        }
    }

    struct FakeChild {
        slot: Arc<Mutex<ChildSlot>>,
        exit_rx: watch::Receiver<bool>,
        stdout: Option<DynAsyncRead>,
        stderr: Option<DynAsyncRead>,
        ignore_sigterm: bool,
    }

    #[async_trait]
    impl ManagedChild for FakeChild {
        fn id(&self) -> Option<u32> {
            Some(self.slot.lock().pid)
        }

        fn take_stdout(&mut self) -> Option<DynAsyncRead> {
            self.stdout.take()
        }

        fn take_stderr(&mut self) -> Option<DynAsyncRead> {
            self.stderr.take()
        }

        async fn wait(&mut self) -> io::Result<ExitStatus> {
            // Wait until the sender publishes `true`. Safe to re-await
            // across `tokio::select!` cancellations: the watch receiver
            // is not consumed, so a second call blocks if the first was
            // dropped before the child actually exited.
            if *self.exit_rx.borrow() {
                return Ok(exit_status(0));
            }
            let _ = self.exit_rx.wait_for(|v| *v).await;
            Ok(exit_status(0))
        }

        async fn sigterm(&mut self) -> io::Result<()> {
            if !self.ignore_sigterm {
                self.terminate(FakeProcessState::SigTerm);
            }
            Ok(())
        }

        async fn sigkill(&mut self) -> io::Result<()> {
            self.terminate(FakeProcessState::SigKill);
            Ok(())
        }
    }

    impl FakeChild {
        fn terminate(&self, new_state: FakeProcessState) {
            let mut slot = self.slot.lock();
            if !matches!(slot.state, FakeProcessState::Running) {
                return;
            }
            slot.state = new_state;
            let _ = slot.exit_tx.send(true);
        }
    }

    impl Drop for FakeChild {
        fn drop(&mut self) {
            // Mirror tokio's `kill_on_drop`: if the supervisor drops the
            // handle without explicit termination (e.g. a panic unwound
            // through its own state machine), pretend the OS killed it.
            let mut slot = self.slot.lock();
            if matches!(slot.state, FakeProcessState::Running) {
                slot.state = FakeProcessState::SigKill;
                let _ = slot.exit_tx.send(true);
            }
        }
    }

    fn exit_status(_code: i32) -> ExitStatus {
        // ExitStatus has no stable public constructor; round-trip through
        // ExitStatusExt (Unix) to synthesise one. Tests never read the
        // numeric code because the state enum carries the authoritative
        // outcome, so a zeroed success status is fine.
        use std::os::unix::process::ExitStatusExt;
        ExitStatus::from_raw(0)
    }
}
