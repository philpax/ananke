Arbitrary key-value pairs exposed through `/v1/models` and `/api/services`:

```toml
[service.metadata]
discord_visible = true
```

These are opaque to the daemon - they exist only to be echoed back to clients (Discord rotation, residence flags, …).
