# Configuration Guide

ananke is configured via a single TOML file discovered in this order:

1. `ANANKE_CONFIG` environment variable.
2. `--config` CLI argument.
3. `$XDG_CONFIG_HOME/ananke/config.toml`
4. `~/.config/ananke/config.toml`
5. `/etc/ananke/config.toml`

The file is hot-reloaded on save: ananke validates the new config, spawns added services and drains removed ones, and ignores failed reloads so the previous valid config stays in effect.
