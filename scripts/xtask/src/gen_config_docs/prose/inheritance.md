Services can inherit configuration from other services using `extends`. This is useful for sharing common settings across related models:

```toml
# Base template: shared settings for all Gemma 4 models
[[service]]
name = "gemma4-base"
template = "llama-cpp"
flash_attn = true
cache_type_k = "q8_0"
cache_type_v = "q8_0"
context = 262144

# Child: inherits flash_attn, cache types, and context; overrides name, port, and model
[[service]]
name = "gemma-4-31b"
template = "llama-cpp"
extends = "gemma4-base"
port = 8200
model = "/models/gemma-4-31B.gguf"
```

Merge rules:

- Scalars: child overrides parent.
- Sub-tables: deep-merged field-by-field.
- Arrays: child replaces parent outright.
- `*_append` fields (e.g., `extra_args_append`): concatenated with parent's list.
- `name`, `port`, `extends`, and `template` must be overridden in the child.
- Cross-template inheritance is an error.

## Service Migration

When renaming a service, use `migrate_from` to preserve database history:

```toml
[[service]]
name = "gemma-4-31b"
template = "llama-cpp"
migrate_from = "old-gemma-31b"
port = 8200
model = "/models/gemma-4-31B.gguf"
```
