//! Builds the default-enabled, self-contained dashboard from locked dependencies.

use std::{
    fs,
    path::{Path, PathBuf},
    process::Command,
};

fn main() {
    if std::env::var_os("CARGO_FEATURE_FRONTEND").is_none() {
        return;
    }
    let manifest = PathBuf::from(std::env::var_os("CARGO_MANIFEST_DIR").unwrap_or_default());
    let frontend = manifest.join("../frontend");
    for input in [
        "src",
        "public",
        "scripts",
        "e2e",
        "index.html",
        "package.json",
        "package-lock.json",
        "vite.config.ts",
        "vitest.config.ts",
        "playwright.config.ts",
        "tsconfig.json",
        "tsconfig.app.json",
        "tsconfig.node.json",
    ] {
        watch(&frontend.join(input));
    }

    // A lockfile change invalidates the installed dependency tree, including
    // trees left by another checkout/toolchain. Outputs are never watched.
    let lock = fs::read(frontend.join("package-lock.json"))
        .unwrap_or_else(|e| panic!("frontend: cannot read package-lock.json: {e}"));
    let receipt = frontend.join("node_modules/.ananke-lock");
    if fs::read(&receipt).ok().as_ref() != Some(&lock) {
        run(&frontend, &["ci", "--no-audit", "--no-fund"]);
        fs::write(receipt, lock)
            .unwrap_or_else(|e| panic!("frontend: cannot record installed dependencies: {e}"));
    }
    run(&frontend, &["run", "build"]);
    assert!(
        frontend.join("dist/index.html").is_file(),
        "frontend: npm build did not produce dist/index.html"
    );
}

fn watch(path: &Path) {
    println!("cargo:rerun-if-changed={}", path.display());
    if path.is_dir() {
        let entries = fs::read_dir(path)
            .unwrap_or_else(|e| panic!("frontend: cannot watch {}: {e}", path.display()));
        for entry in entries {
            let entry = entry.unwrap_or_else(|e| panic!("frontend: cannot watch input: {e}"));
            watch(&entry.path());
        }
    }
}

fn run(frontend: &Path, args: &[&str]) {
    match Command::new("npm")
        .args(args)
        .current_dir(frontend)
        .status()
    {
        Ok(status) if status.success() => {}
        result => panic!(
            "frontend: npm {} failed ({result:?}). Install Node.js >=22.12 and npm, or use --no-default-features for a backend-only build.",
            args.join(" ")
        ),
    }
}
