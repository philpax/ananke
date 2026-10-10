//! Container-runtime trait and the types shared across implementations.

use std::{pin::Pin, sync::Arc};

use ananke_errors::ExpectedError;
use ananke_spawn::ContainerSpec;
use async_trait::async_trait;
use tokio::io::AsyncRead;

/// Owned async reader used for container log streams.
pub type DynAsyncRead = Pin<Box<dyn AsyncRead + Send + Unpin + 'static>>;

/// Two-phase container runtime seam. Production uses Docker/Podman CLIs;
/// tests use [`FakeContainerEngine`](super::FakeContainerEngine).
// Clippy 1.99 double-counts `async_trait`'s boxed futures; older stable Clippy does not emit this lint.
#[allow(clippy::double_must_use)]
#[async_trait]
pub trait ContainerEngine: Send + Sync {
    /// An engine that drives `executable` instead of this one's default.
    ///
    /// The id-keyed operations (`inspect`, `remove`, `list`) take no spec,
    /// so reconciliation — which may meet a container from any runtime the
    /// config has ever named — resolves the right binary from the
    /// persisted record and asks for a matching engine.
    fn for_executable(&self, executable: &str) -> std::sync::Arc<dyn ContainerEngine>;

    /// Create a container from the resolved spec. Returns a prepared handle
    /// that can be started or removed, but cannot wait or receive signals.
    async fn create(&self, spec: &ContainerSpec) -> Result<PreparedContainer, ExpectedError>;

    /// Start a prepared container and return a running handle.
    async fn start(
        &self,
        prepared: &PreparedContainer,
    ) -> Result<Box<dyn ManagedContainer>, ExpectedError>;

    /// Remove a prepared container idempotently (without starting it).
    async fn remove_prepared(&self, prepared: &PreparedContainer) -> Result<(), ExpectedError>;

    /// Inspect a container by ID.
    ///
    /// `Ok(None)` means the runtime answered and the container is not
    /// there. `Err` means the runtime could not be asked. Callers act very
    /// differently on those — one cleans a stale record, the other must
    /// preserve it — so the distinction is in the type rather than in the
    /// text of an error a caller would have to match on.
    async fn inspect(&self, id: &str) -> Result<Option<ContainerInspect>, ExpectedError>;

    /// Remove a container by ID, idempotently. Already-absent is success.
    /// Used by startup reconciliation, which has no running/prepared handle.
    async fn remove(&self, id: &str) -> Result<(), ExpectedError>;

    /// List containers matching the given `--filter` expressions (e.g.
    /// `"label=io.ananke.managed=true"`).
    async fn list(&self, filters: &[String]) -> Result<Vec<ContainerSummary>, ExpectedError>;
}

/// Output of a container inspection.
#[derive(Debug, Clone)]
pub struct ContainerInspect {
    /// Container ID, always the full 64-char hex.
    pub id: String,
    /// Generated name, with Docker's leading `/` stripped so it matches
    /// what Podman reports and what ananke generated.
    pub name: String,
    /// Current state: one of `created`, `running`, `exited`, `paused`, `dead`.
    pub state: String,
    /// Exit code when state is `"exited"`; `None` otherwise.
    pub exit_code: Option<i32>,
    /// Host PID of the main process (when running); `None` otherwise.
    pub host_pid: Option<u32>,
    /// Value of the `io.ananke.owner` label, the only one ananke reads.
    ///
    /// Asking the runtime for one named label rather than iterating them
    /// keeps this free of both the template divergence (Docker renders
    /// labels as a joined string, Podman as a map) and the ambiguity of
    /// picking a separator that an operator's own label value cannot
    /// contain.
    pub owner: Option<String>,
}

/// Summary of a container returned by list operations. Carries the same
/// fields as an inspection because it is built from one.
#[derive(Debug, Clone)]
pub struct ContainerSummary {
    /// Container ID, always the full 64-char hex.
    pub id: String,
    /// Name, normalised as in [`ContainerInspect::name`].
    pub name: String,
    /// State.
    pub state: String,
    /// Value of the `io.ananke.owner` label.
    pub owner: Option<String>,
}

/// A prepared container: created but not yet started. Can be started or
/// removed idempotently. Cannot wait or receive signals.
pub struct PreparedContainer {
    /// Container ID.
    pub id: String,
    /// Generated name.
    pub name: String,
    /// Runtime executable used for this container.
    pub runtime_executable: String,
    /// Runtime used (Docker or Podman), for lifecycle-command differences.
    pub runtime: ananke_spawn::ContainerRuntime,
    /// Underlying engine shared by the prepared/running handles.
    pub(crate) engine: Arc<dyn ContainerEngine>,
}

impl PreparedContainer {
    /// Start the container and return a running handle.
    ///
    /// The caller must have already persisted the ownership row before
    /// calling this (the supervisor enforces that ordering).
    pub async fn start(&self) -> Result<Box<dyn ManagedContainer>, ExpectedError> {
        self.engine.start(self).await
    }

    /// Remove the container idempotently (without starting it first).
    pub async fn remove(&self) -> Result<(), ExpectedError> {
        self.engine.remove_prepared(self).await
    }
}

/// A running container handle implementing the common managed-workload
/// operations.
// Clippy 1.99 double-counts `async_trait`'s boxed futures; older stable Clippy does not emit this lint.
#[allow(clippy::double_must_use)]
#[async_trait]
pub trait ManagedContainer: Send + Sync {
    /// Return the container ID.
    fn id(&self) -> &str;

    /// Return the container name.
    fn name(&self) -> &str;

    /// Return the runtime executable.
    fn runtime_executable(&self) -> &str;

    /// Return the inspected host PID when available.
    fn host_pid(&self) -> Option<u32>;

    /// Spawn a log-follower and return its output readers.
    ///
    /// Both runtimes route the container's stdout to the follower's stdout
    /// and its stderr to the follower's stderr, so both are needed: llama.cpp
    /// and vLLM write almost everything to stderr. They are returned
    /// separately rather than merged because the caller tags every line
    /// `combined` regardless, and interleaving bytes from two pipes would
    /// risk splicing partial lines together.
    fn logs(&self) -> Vec<DynAsyncRead>;

    /// Wait for the container to exit. Returns the exit code.
    async fn wait(&self) -> Result<i32, ExpectedError>;

    /// Send SIGTERM to the container's main process.
    async fn terminate(&self) -> Result<(), ExpectedError>;

    /// Send SIGKILL to the container's main process.
    async fn kill(&self) -> Result<(), ExpectedError>;

    /// Remove the container idempotently. Already-absent is success.
    async fn remove(&self) -> Result<(), ExpectedError>;
}
