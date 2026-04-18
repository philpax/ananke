default:
    @just --list

# Run all linters and tests across Rust and TypeScript.
lint: lint-rust lint-frontend

lint-rust:
    cargo fmt --all -- --check
    cargo clippy --all-targets --all-features -- -D warnings
    cargo clippy --all-targets --no-default-features -- -D warnings
    cargo test --workspace
    cargo test --workspace --no-default-features

lint-frontend:
    cd frontend && npm run lint

# Format Rust and TypeScript sources in place.
fmt:
    cargo fmt --all
    cd frontend && npm run format
