use std::process::Command;

#[test]
fn config_show_help_runs() {
    let output = Command::new(env!("CARGO_BIN_EXE_anankectl"))
        .args(["config", "--help"])
        .output()
        .expect("spawn");
    assert!(output.status.success(), "{:?}", output);
    let stdout = String::from_utf8_lossy(&output.stdout);
    assert!(stdout.contains("show"));
    assert!(stdout.contains("validate"));
    assert!(stdout.contains("reload"));
}
