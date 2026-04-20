# Development shell for the ananke repo.
#
# Provides the toolchain plus a LD_LIBRARY_PATH that exposes
# `libnvidia-ml.so`, which `nvml-wrapper` needs to dlopen at runtime.
# On NixOS the driver libraries live under /run/opengl-driver/lib rather
# than a distro-standard /usr/lib path, so a plain `cargo run` against
# `target/release/ananke` fails NVML init without this shell. Enter with
# `nix-shell` from the repo root.
#
# Deployment (systemd unit, package, etc.) lives in a separate NixOS
# module — this shell is strictly for local dev + calibration runs.

{ pkgs ? import <nixpkgs> { config.allowUnfree = true; } }:

pkgs.mkShell {
  # We deliberately don't pin the Rust toolchain here — nixpkgs stable
  # tends to lag a few minors behind what the workspace's dependency tree
  # requires (toasty currently needs ≥ 1.94). Bring your own rustc via
  # the system channel or rustup; this shell supplies everything else.
  nativeBuildInputs = with pkgs; [
    pkg-config
  ];

  buildInputs = with pkgs; [
    openssl
  ];

  # Stress + calibration scripts run through uv.
  #
  # We deliberately don't pull in llama.cpp here — on NixOS it's installed
  # system-wide (e.g. via the graphics module), and the daemon + calibration
  # harness spawn whichever `llama-server` is first on PATH. Adding a pinned
  # copy from nixpkgs would often diverge from the system GPU driver's CUDA
  # version, which is worse than "use what the host has".
  packages = with pkgs; [
    uv
    python312
  ];

  shellHook = ''
    # NVIDIA driver libraries. /run/opengl-driver/lib is populated by the
    # hardware.nvidia NixOS module; if you're on a non-NixOS host the dir
    # may not exist, in which case the daemon falls back to CPU-only
    # (which is fine for unit tests but useless for calibration).
    if [ -e /run/opengl-driver/lib ]; then
      export LD_LIBRARY_PATH="/run/opengl-driver/lib''${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
    fi

    # Colourful backtraces during cargo test runs.
    export RUST_BACKTRACE=1

    echo "ananke dev shell ready"
    echo "  rustc:      $(rustc --version 2>/dev/null || echo 'missing — install via rustup')"
    echo "  cargo:      $(cargo --version 2>/dev/null || echo 'missing — install via rustup')"
    echo "  uv:         $(uv --version 2>/dev/null || echo missing)"
    echo "  nvml libs:  ''${LD_LIBRARY_PATH:-<not set>}"
  '';
}
