use ananke::oneshot::{OneshotRecord, OneshotRegistry};
use smol_str::SmolStr;

#[test]
fn registry_insert_and_get() {
    let r = OneshotRegistry::new();
    let rec = OneshotRecord {
        id: SmolStr::new("os_1"),
        service_name: SmolStr::new("os_1"),
        port: 18000,
        ttl_ms: 60_000,
        started_at_ms: 0,
    };
    r.insert(rec.clone());
    assert!(r.get("os_1").is_some());
    r.remove("os_1");
    assert!(r.get("os_1").is_none());
}
