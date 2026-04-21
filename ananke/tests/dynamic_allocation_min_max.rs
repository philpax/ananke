#![cfg(feature = "test-fakes")]
use ananke::config::{AllocationMode, parse_toml, resolve_inheritance, validate};

#[test]
fn dynamic_parses_min_max_and_runtime() {
    let mut cfg = parse_toml(
        r#"
[[service]]
name = "comfy"
template = "command"
command = ["python"]
port = 8188
allocation.mode = "dynamic"
allocation.min_vram_gb = 4
allocation.max_vram_gb = 20
allocation.min_borrower_runtime = "90s"
"#,
        std::path::Path::new("/t"),
    )
    .unwrap();
    resolve_inheritance(&mut cfg).unwrap();
    let ec = validate(&cfg).unwrap();
    let svc = &ec.services[0];
    if let AllocationMode::Dynamic {
        min_mb,
        max_mb,
        min_borrower_runtime_ms,
    } = svc.allocation_mode
    {
        assert_eq!(min_mb, 4096);
        assert_eq!(max_mb, 20480);
        assert_eq!(min_borrower_runtime_ms, 90_000);
    } else {
        panic!("expected Dynamic, got {:?}", svc.allocation_mode);
    }
}
