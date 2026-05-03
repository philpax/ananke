use std::process::Command;

use tempfile::TempDir;

/// `anankectl server-config --help` exposes the daemon-config flow.
#[test]
fn server_config_help_lists_subcommands() {
    let output = Command::new(env!("CARGO_BIN_EXE_anankectl"))
        .args(["server-config", "--help"])
        .output()
        .expect("spawn");
    assert!(output.status.success(), "{:?}", output);
    let stdout = String::from_utf8_lossy(&output.stdout);
    assert!(stdout.contains("show"), "{stdout}");
    assert!(stdout.contains("validate"), "{stdout}");
    assert!(stdout.contains("reload"), "{stdout}");
}

/// `anankectl config --help` exposes the client-config flow.
#[test]
fn client_config_help_lists_subcommands() {
    let output = Command::new(env!("CARGO_BIN_EXE_anankectl"))
        .args(["config", "--help"])
        .output()
        .expect("spawn");
    assert!(output.status.success(), "{:?}", output);
    let stdout = String::from_utf8_lossy(&output.stdout);
    for sub in ["get", "set", "unset", "list", "path", "edit"] {
        assert!(stdout.contains(sub), "missing `{sub}` in: {stdout}");
    }
}

/// A `set` → `get` → `unset` round-trip against a tempdir XDG config
/// home. Avoids touching the user's real config and exercises the same
/// path resolution + TOML round-trip used in production.
#[test]
fn client_config_set_get_unset_roundtrip() {
    let xdg = TempDir::new().expect("tempdir");

    let set = Command::new(env!("CARGO_BIN_EXE_anankectl"))
        .env("XDG_CONFIG_HOME", xdg.path())
        .args(["config", "set", "endpoint", "http://127.0.0.1:9999"])
        .output()
        .expect("spawn set");
    assert!(set.status.success(), "{:?}", set);

    let get = Command::new(env!("CARGO_BIN_EXE_anankectl"))
        .env("XDG_CONFIG_HOME", xdg.path())
        .args(["config", "get", "endpoint"])
        .output()
        .expect("spawn get");
    assert!(get.status.success(), "{:?}", get);
    let stdout = String::from_utf8_lossy(&get.stdout);
    assert!(
        stdout.trim() == "http://127.0.0.1:9999",
        "unexpected get output: {stdout:?}",
    );

    let unset = Command::new(env!("CARGO_BIN_EXE_anankectl"))
        .env("XDG_CONFIG_HOME", xdg.path())
        .args(["config", "unset", "endpoint"])
        .output()
        .expect("spawn unset");
    assert!(unset.status.success(), "{:?}", unset);

    let after = Command::new(env!("CARGO_BIN_EXE_anankectl"))
        .env("XDG_CONFIG_HOME", xdg.path())
        .args(["config", "get", "endpoint"])
        .output()
        .expect("spawn get after unset");
    assert!(after.status.success(), "{:?}", after);
    assert!(
        String::from_utf8_lossy(&after.stdout).trim().is_empty(),
        "expected unset endpoint to print nothing"
    );
}

/// Setting an obviously-invalid URL is rejected up-front rather than
/// silently persisted; same for unknown keys.
#[test]
fn client_config_rejects_invalid_inputs() {
    let xdg = TempDir::new().expect("tempdir");

    let bad_url = Command::new(env!("CARGO_BIN_EXE_anankectl"))
        .env("XDG_CONFIG_HOME", xdg.path())
        .args(["config", "set", "endpoint", "not-a-url"])
        .output()
        .expect("spawn");
    assert!(
        !bad_url.status.success(),
        "expected invalid URL to fail: {:?}",
        bad_url
    );

    let unknown_key = Command::new(env!("CARGO_BIN_EXE_anankectl"))
        .env("XDG_CONFIG_HOME", xdg.path())
        .args(["config", "set", "bogus", "value"])
        .output()
        .expect("spawn");
    assert!(
        !unknown_key.status.success(),
        "expected unknown key to fail: {:?}",
        unknown_key
    );
}
