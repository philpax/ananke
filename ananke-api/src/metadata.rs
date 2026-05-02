//! Shared type alias for the per-service `ananke_metadata` passthrough.

use std::collections::BTreeMap;

/// Arbitrary passthrough entries attached to a service via its
/// `[[service]] metadata.*` keys.
///
/// Serialises as a plain JSON object on the wire. Entries are opaque to
/// ananke and exist purely for consumers (UIs, bots, dashboards) to
/// read back through `/v1/models` and `/api/services`.
pub type AnankeMetadata = BTreeMap<String, serde_json::Value>;
