-- 0002_metrics: per-request metrics and periodic device sampling.
--
-- request_metrics: one row per proxied OpenAI API request. Written by
-- the proxy handler after the response stream completes (streaming) or
-- the body is received (non-streaming). Retention pruned by the
-- existing retention loop to a configurable window (default 7 days).
--
-- device_samples: periodic VRAM/RAM snapshots written by the
-- snapshotter. Retention pruned to 24h. Used by the devices view and
-- the Prometheus /metrics endpoint.

CREATE TABLE request_metrics (
  metric_id         INTEGER PRIMARY KEY AUTOINCREMENT,
  service_id        INTEGER NOT NULL,
  run_id            INTEGER,
  timestamp_ms      INTEGER NOT NULL,
  endpoint          TEXT NOT NULL,
  model             TEXT NOT NULL,
  prompt_tokens     INTEGER,
  completion_tokens INTEGER,
  duration_ms       INTEGER,
  ttft_ms           INTEGER,
  status_code       INTEGER NOT NULL,
  FOREIGN KEY (service_id) REFERENCES services(service_id)
);
CREATE INDEX request_metrics_ts ON request_metrics(service_id, timestamp_ms);
CREATE INDEX request_metrics_run ON request_metrics(run_id);

CREATE TABLE device_samples (
  sample_id    INTEGER PRIMARY KEY AUTOINCREMENT,
  device       TEXT NOT NULL,
  timestamp_ms INTEGER NOT NULL,
  total_bytes  INTEGER NOT NULL,
  free_bytes   INTEGER NOT NULL,
  used_bytes   INTEGER NOT NULL
);
CREATE INDEX device_samples_ts ON device_samples(device, timestamp_ms);
