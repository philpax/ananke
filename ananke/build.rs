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

use std::{path::PathBuf, process::Command};

fn main() {
    if std::env::var_os("ANANKE_SKIP_FRONTEND_BUILD").is_some() {
        return;
    }

    let manifest_dir = PathBuf::from(std::env::var("CARGO_MANIFEST_DIR").unwrap());
    let frontend = manifest_dir.parent().unwrap().join("frontend");

    // Re-run when any frontend input changes. `cargo:rerun-if-changed`
    // watches the named path recursively for directories, so `src` alone
    // covers every component.
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
        println!("cargo:rerun-if-changed={}", frontend.join(rel).display());
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
