use ananke::oneshot::{OneshotRecord, OneshotRegistry};
use smol_str::SmolStr;

fn make_record(id: &str) -> OneshotRecord {
    OneshotRecord {
        id: SmolStr::new(id),
        service_name: SmolStr::new(id),
        port: 18000,
        ttl_ms: 60_000,
        started_at_ms: 0,
        ended_at_ms: None,
        exit_code: None,
    }
}

#[test]
fn registry_insert_and_get() {
    let r = OneshotRegistry::new();
    r.insert(make_record("os_1"));
    assert!(r.get("os_1").is_some());
    r.remove("os_1");
    assert!(r.get("os_1").is_none());
}

/// Regression for the 404-after-TTL scenario: `mark_ended` must leave the
/// record in place with `ended_at_ms` set, so callers polling
/// `GET /api/oneshot/:id` across the TTL boundary still get 200. A plain
/// `remove` on TTL expiry turned poll #2 into a 404 from the Python
/// scenario (but not from a single pre-TTL curl).
#[test]
fn mark_ended_keeps_record_visible_with_terminal_fields() {
    let r = OneshotRegistry::new();
    r.insert(make_record("os_1"));

    let updated = r.mark_ended("os_1", 12345, Some(0)).expect("record present");
    assert_eq!(updated.ended_at_ms, Some(12345));
    assert_eq!(updated.exit_code, Some(0));

    // The record must still be retrievable after mark_ended — the whole
    // point is to preserve it as a tombstone for status polls.
    let fetched = r.get("os_1").expect("record must survive mark_ended");
    assert_eq!(fetched.ended_at_ms, Some(12345));
    assert_eq!(fetched.exit_code, Some(0));

    // Explicit remove still evicts it (DELETE /api/oneshot/:id path).
    r.remove("os_1");
    assert!(r.get("os_1").is_none());
}

#[test]
fn mark_ended_is_noop_when_record_absent() {
    let r = OneshotRegistry::new();
    assert!(r.mark_ended("missing", 42, None).is_none());
}
