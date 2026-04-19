// Smoke test: --help runs without error. Richer integration tests are added
// per-subcommand as those commands are implemented.

use std::process::Command;

#[test]
fn help_works() {
    let output = Command::new(env!("CARGO_BIN_EXE_anankectl"))
        .arg("--help")
        .output()
        .expect("spawn");
    assert!(output.status.success(), "--help exit: {:?}", output.status);
    let stdout = String::from_utf8_lossy(&output.stdout);
    assert!(stdout.contains("devices"));
    assert!(stdout.contains("services"));
}
