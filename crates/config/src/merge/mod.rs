//! Resolve `extends` inheritance and `*_append` concatenation, and
//! `migrate_from` rename chains, before validation.

mod field_merge;
mod migrations;
mod resolve;
#[cfg(test)]
mod test_support;

pub use migrations::{Migration, resolve_migrations};
pub use resolve::resolve_inheritance;

/// Format a merge/inheritance diagnostic's message: `service {s} extends
/// {p}: {reason}` when both the service and its parent are known, dropping
/// whichever half isn't. The full sentence — including the "service" text —
/// is baked in here rather than left to the diagnostic's shared `service`
/// prefix, since the `extends {parent}` clause sits between the two in a
/// shape that prefix can't produce.
pub(crate) fn merge_message(service: Option<&str>, parent: Option<&str>, reason: &str) -> String {
    match (service, parent) {
        (Some(service), Some(parent)) => format!("service {service} extends {parent}: {reason}"),
        (Some(service), None) => format!("service {service}: {reason}"),
        _ => reason.to_string(),
    }
}
