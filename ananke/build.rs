//! Build-time glue that produces `../frontend/dist` before the main crate
//! compiles so `rust-embed` can fold the built assets into the binary.
//!
//! The flow is deliberately conservative: cargo only re-runs this script
//! when one of the listed frontend inputs changes, and we run `npm ci`
//! only on the first build (when `node_modules` is missing). A `npm run
//! build` on a clean tree is ~1 second — cheap enough not to feature-
//! flag.
//!
//! Set `ANANKE_SKIP_FRONTEND_BUILD=1` to skip the npm invocation entirely
//! — useful for Rust-only dev loops or for packaging that builds the
//! frontend separately and drops the assets into `../frontend/dist`
//! out-of-band.

use std::{
    path::{Path, PathBuf},
    process::Command,
};

fn main() {
    if std::env::var_os("ANANKE_SKIP_FRONTEND_BUILD").is_some() {
        return;
    }

    let manifest_dir = PathBuf::from(std::env::var("CARGO_MANIFEST_DIR").unwrap());
    let frontend = manifest_dir.parent().unwrap().join("frontend");

    // Re-run when any frontend input changes. `cargo:rerun-if-changed` on a
    // directory does *not* reliably detect an in-place edit of a nested file
    // (the intermediate directory mtimes don't change), so a change to e.g.
    // `src/locales/en/translation.json` would leave the embedded bundle stale.
    // Emit one `rerun-if-changed` per file by walking the tree instead.
    let watched = [
        "src",
        "index.html",
        "package.json",
        "package-lock.json",
        "vite.config.ts",
        "tsconfig.json",
        "tsconfig.app.json",
        "tsconfig.node.json",
        "eslint.config.js",
    ];
    for rel in watched {
        watch_recursively(&frontend.join(rel));
    }
    println!("cargo:rerun-if-env-changed=ANANKE_SKIP_FRONTEND_BUILD");

    // `npm ci` only when `node_modules` is absent; routine builds skip it.
    if !frontend.join("node_modules").exists() {
        run("npm", &["ci", "--legacy-peer-deps"], &frontend, "npm ci");
    }

    run("npm", &["run", "build"], &frontend, "npm run build");

    let index_html = frontend.join("dist").join("index.html");
    assert!(
        index_html.exists(),
        "frontend build did not produce {}",
        index_html.display()
    );
}

/// Emit `cargo:rerun-if-changed` for `path` and, if it is a directory, every
/// file beneath it — watching each file individually so an in-place edit of a
/// nested source triggers a rebuild. Also watches the directories themselves,
/// so adding or removing a file is caught. Skips `node_modules` and `dist`,
/// which are build inputs/outputs, not sources.
fn watch_recursively(path: &Path) {
    println!("cargo:rerun-if-changed={}", path.display());
    if !path.is_dir() {
        return;
    }
    if matches!(
        path.file_name().and_then(|n| n.to_str()),
        Some("node_modules") | Some("dist")
    ) {
        return;
    }
    let Ok(entries) = std::fs::read_dir(path) else {
        return;
    };
    for entry in entries.flatten() {
        watch_recursively(&entry.path());
    }
}

fn run(cmd: &str, args: &[&str], cwd: &std::path::Path, label: &str) {
    let status = Command::new(cmd).args(args).current_dir(cwd).status();
    match status {
        Ok(s) if s.success() => {}
        Ok(s) => panic!("{label} failed with exit status {s}"),
        Err(e) => panic!(
            "{label} failed to spawn: {e}. Set ANANKE_SKIP_FRONTEND_BUILD=1 \
             to build without the embedded frontend."
        ),
    }
}
