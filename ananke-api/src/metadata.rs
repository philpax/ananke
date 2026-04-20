//! Shared type alias for the per-service `ananke_metadata` passthrough.

use std::collections::BTreeMap;

/// Arbitrary passthrough entries attached to a service via its
/// `[[service]] metadata.*` keys.
///
/// Serialises as a plain JSON object on the wire. One well-known key,
/// `openai_compat`, is special-cased by the daemon's OpenAI proxy (it
/// gates whether the service shows up under `/v1/models`); every other
/// entry is opaque to ananke and exists purely for consumers (UIs,
/// bots, dashboards) to read back through `/v1/models` and
/// `/api/services`.
pub type AnankeMetadata = BTreeMap<String, serde_json::Value>;
