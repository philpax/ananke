//! Hand out the private loopback ports supervised children bind to, from the
//! bounded range configured by `daemon.private_port_start` / `_end`.

use smol_str::SmolStr;

use crate::{
    fields,
    validate::{
        ConfigDiagnostic, DEFAULT_PRIVATE_PORT_END, DEFAULT_PRIVATE_PORT_START,
        ValidationErrorCode, ValueDiagnosticDetail,
    },
};

// `DEFAULT_PRIVATE_PORT_START` / `DEFAULT_PRIVATE_PORT_END` are re-exported
// from `ananke_config` (see the `pub use` at the top of this module).

/// Inclusive `start..=end` range of loopback ports assigned to supervised
/// children. Derived from `daemon.private_port_start` / `_end` or the
/// compiled-in default.
#[derive(Debug, Clone, Copy)]
pub(crate) struct PrivatePortRange {
    pub(crate) start: u16,
    pub(crate) end: u16,
}

impl PrivatePortRange {
    pub(crate) fn width(self) -> u32 {
        (self.end as u32) - (self.start as u32) + 1
    }

    pub(crate) fn from_config(
        start: Option<u16>,
        end: Option<u16>,
    ) -> Result<Self, ConfigDiagnostic> {
        let start = start.unwrap_or(DEFAULT_PRIVATE_PORT_START);
        let end = end.unwrap_or(DEFAULT_PRIVATE_PORT_END);
        if end <= start {
            return Err(ConfigDiagnostic::value_with_detail(
                ValidationErrorCode::PrivatePortRangeInvalid,
                ValueDiagnosticDetail::PrivatePortRangeInvalid,
                fields::daemon::PRIVATE_PORT_END,
                format!("{start}..={end}"),
                Some("private_port_end greater than private_port_start".into()),
            ));
        }
        Ok(Self { start, end })
    }
}

/// Hand out unique private ports from a bounded range. First-come
/// first-served from `range.start` upward. External-process collisions are
/// detected at spawn time by llama-server's bind failure, not by this
/// allocator — probing here would only narrow a race window that the
/// supervisor already surfaces as a `StartFailure`.
#[derive(Clone)]
pub(crate) struct PrivatePortAllocator {
    pub(crate) range: PrivatePortRange,
    next: u32,
}

impl PrivatePortAllocator {
    pub(crate) fn new(range: PrivatePortRange) -> Self {
        Self {
            range,
            next: range.start as u32,
        }
    }

    pub(crate) fn allocate(&mut self, svc_name: &SmolStr) -> Result<u16, ConfigDiagnostic> {
        if self.next > self.range.end as u32 {
            return Err(ConfigDiagnostic::value(
                ValidationErrorCode::PrivatePortExhausted,
                fields::daemon::PRIVATE_PORT_END,
                format!(
                    "service {svc_name}: private_port_range [{}, {}] exhausted ({} slots) — widen the range or reduce service count",
                    self.range.start,
                    self.range.end,
                    self.range.width()
                ),
                Some("an available private port".into()),
            ));
        }
        let port = self.next as u16;
        self.next += 1;
        Ok(port)
    }

    /// `true` when `port` is within the allocator's range (and would be
    /// a candidate for auto-assignment). Used to warn operators whose
    /// `private_port` override happens to overlap the auto-pool.
    pub(crate) fn contains(&self, port: u16) -> bool {
        port >= self.range.start && port <= self.range.end
    }
}

#[cfg(test)]
mod tests {
    use crate::validate::{test_fixtures::parse_and_merge, validate};

    #[test]
    fn private_port_range_is_configurable_and_exhausts_cleanly() {
        // A two-port window must fit exactly two services; the third triggers
        // the exhausted-range error with the requested bounds echoed back so
        // the operator knows which knobs to widen. The default 40_000–59_999
        // window is deliberately large, so this guard only matters on hosts
        // that shrink it to dodge a port collision.
        let cfg = parse_and_merge(
            r#"
[daemon]
private_port_start = 50000
private_port_end = 50001

[[service]]
name = "a"
template = "llama-cpp"
model = "/m/x.gguf"
port = 11000
lifecycle = "persistent"
devices.placement_override = { "gpu:0" = 1000 }

[[service]]
name = "b"
template = "llama-cpp"
model = "/m/x.gguf"
port = 11001
lifecycle = "persistent"
devices.placement_override = { "gpu:0" = 1000 }

[[service]]
name = "c"
template = "llama-cpp"
model = "/m/x.gguf"
port = 11002
lifecycle = "persistent"
devices.placement_override = { "gpu:0" = 1000 }
"#,
        );
        let err = validate(&cfg).unwrap_err();
        let msg = err.to_string();
        assert!(
            msg.contains("exhausted") && msg.contains("50000") && msg.contains("50001"),
            "expected range-exhausted error naming [50000, 50001]; got: {msg}"
        );
    }

    #[test]
    fn private_port_range_assigns_in_order_from_start() {
        // Two services in a custom window should get start, start+1 — not the
        // 40000-base default, and not duplicates. Deriving the private port
        // from the public one (`40_000 + (port - 11_000)`) ignores the window
        // and wraps to 65535 for every service.
        let cfg = parse_and_merge(
            r#"
[daemon]
private_port_start = 45000
private_port_end = 45099

[[service]]
name = "a"
template = "llama-cpp"
model = "/m/x.gguf"
port = 11000
lifecycle = "persistent"
devices.placement_override = { "gpu:0" = 1000 }

[[service]]
name = "b"
template = "llama-cpp"
model = "/m/x.gguf"
port = 11001
lifecycle = "persistent"
devices.placement_override = { "gpu:0" = 1000 }
"#,
        );
        let ec = validate(&cfg).unwrap();
        let ports: Vec<u16> = ec.services.iter().map(|s| s.private_port).collect();
        assert_eq!(ports, vec![45000, 45001]);
    }

    #[test]
    fn private_port_range_rejects_inverted_bounds() {
        let cfg = parse_and_merge(
            r#"
[daemon]
private_port_start = 50000
private_port_end = 49999

[[service]]
name = "a"
template = "llama-cpp"
model = "/m/x.gguf"
port = 11000
lifecycle = "persistent"
devices.placement_override = { "gpu:0" = 1000 }
"#,
        );
        let err = validate(&cfg).unwrap_err();
        let msg = err.to_string();
        assert!(
            msg.contains("private_port_end") && msg.contains("must exceed"),
            "expected inverted-bounds error; got: {msg}"
        );
    }
    #[test]
    fn private_port_override_outside_pool_does_not_warn() {
        // Smoke-test the code path; we don't capture tracing output,
        // but this at least exercises the branch.
        let cfg = parse_and_merge(
            r#"
[daemon]
private_port_start = 40000
private_port_end = 40100

[[service]]
name = "ext"
template = "command"
command = ["/bin/true"]
port = 8500
private_port = 18188
allocation.mode = "static"
allocation.reserve_gb = 1
"#,
        );
        assert!(validate(&cfg).is_ok());
    }
}
