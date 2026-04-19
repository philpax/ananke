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
# Uses nightly rustfmt because rustfmt.toml sets nightly-only options
# (imports_granularity, group_imports) that enforce import style.
fmt:
    cargo +nightly fmt --all
    cd frontend && npm run format

# Regenerate the toasty migration SQL from the models.
# Wraps `toasty-cli migrate generate`. Requires toasty-cli installed:
#   cargo install toasty-cli
db-migrate:
    toasty-cli migrate generate
