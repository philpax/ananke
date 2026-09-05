-- 0012_strict: enforce type checking on every application table.
--
-- SQLite cannot add STRICT to an existing table. Recreate the schema in one
-- migration transaction, copying rows before dropping the old tables. The
-- migration runner supplies the transaction; foreign_keys stays enabled so
-- copied rows and the final schema are checked rather than merely assumed.
-- AUTOINCREMENT sequence values are saved explicitly because recreating a
-- table otherwise resets a high-water mark after a history has been pruned.

CREATE TEMP TABLE migration_0012_sequences (
    name TEXT PRIMARY KEY,
    seq  INTEGER NOT NULL
) STRICT;

INSERT INTO migration_0012_sequences (name, seq)
SELECT name, seq
FROM sqlite_sequence
WHERE name IN (
    'services',
    'allocation_events',
    'request_metrics',
    'device_samples',
    'service_restarts',
    'container_launch_intents'
);

-- Explicit indexes must be removed before the replacement tables are created.
-- Their definitions are recreated below with the same names and expressions.
DROP INDEX service_logs_ts;
DROP INDEX allocation_events_service;
DROP INDEX oneshots_service;
DROP INDEX request_metrics_ts;
DROP INDEX request_metrics_run;
DROP INDEX idx_service_restarts_service;
DROP INDEX idx_service_restarts_at;
DROP INDEX idx_running_services_service_id;
DROP INDEX device_samples_ts;

-- Keep all old foreign keys valid while the replacement tables are populated.
-- Renaming a referenced table updates the old child definitions, so the old
-- graph remains self-contained until its child tables are dropped below.
ALTER TABLE service_config_versions RENAME TO migration_0012_old_service_config_versions;
ALTER TABLE running_services RENAME TO migration_0012_old_running_services;
ALTER TABLE service_logs RENAME TO migration_0012_old_service_logs;
ALTER TABLE allocation_events RENAME TO migration_0012_old_allocation_events;
ALTER TABLE oneshots RENAME TO migration_0012_old_oneshots;
ALTER TABLE request_metrics RENAME TO migration_0012_old_request_metrics;
ALTER TABLE device_samples RENAME TO migration_0012_old_device_samples;
ALTER TABLE service_restarts RENAME TO migration_0012_old_service_restarts;
ALTER TABLE service_restart_counts RENAME TO migration_0012_old_service_restart_counts;
ALTER TABLE installation_metadata RENAME TO migration_0012_old_installation_metadata;
ALTER TABLE container_launch_intents RENAME TO migration_0012_old_container_launch_intents;
ALTER TABLE services RENAME TO migration_0012_old_services;

CREATE TABLE services (
  service_id INTEGER PRIMARY KEY AUTOINCREMENT,
  name       TEXT NOT NULL UNIQUE,
  created_at INTEGER NOT NULL,
  deleted_at INTEGER
) STRICT;

CREATE TABLE service_config_versions (
  service_id       INTEGER NOT NULL,
  version          INTEGER NOT NULL,
  effective_config TEXT NOT NULL,
  recorded_at      INTEGER NOT NULL,
  PRIMARY KEY (service_id, version),
  FOREIGN KEY (service_id) REFERENCES services(service_id)
) STRICT;

CREATE TABLE running_services (
  service_id   INTEGER NOT NULL,
  run_id       INTEGER NOT NULL,
  pid          INTEGER,
  spawned_at   INTEGER NOT NULL,
  command_line TEXT NOT NULL,
  allocation   TEXT NOT NULL,
  state        TEXT NOT NULL,
  workload_kind   TEXT,
  runtime         TEXT,
  container_name  TEXT,
  container_id    TEXT,
  runtime_executable TEXT,
  PRIMARY KEY (service_id, run_id),
  FOREIGN KEY (service_id) REFERENCES services(service_id)
) STRICT;

CREATE TABLE service_logs (
  service_id   INTEGER NOT NULL,
  run_id       INTEGER NOT NULL,
  timestamp_ms INTEGER NOT NULL,
  seq          INTEGER NOT NULL,
  stream       TEXT NOT NULL,
  line         TEXT NOT NULL,
  PRIMARY KEY (service_id, run_id, seq),
  FOREIGN KEY (service_id) REFERENCES services(service_id)
) STRICT;

CREATE TABLE allocation_events (
  event_id   INTEGER PRIMARY KEY AUTOINCREMENT,
  service_id INTEGER NOT NULL,
  run_id     INTEGER NOT NULL,
  event_type TEXT NOT NULL,
  device     TEXT NOT NULL,
  bytes      INTEGER NOT NULL,
  at         INTEGER NOT NULL,
  FOREIGN KEY (service_id) REFERENCES services(service_id)
) STRICT;

CREATE TABLE oneshots (
  id           TEXT PRIMARY KEY,
  service_id   INTEGER NOT NULL,
  submitted_at INTEGER NOT NULL,
  started_at   INTEGER,
  ended_at     INTEGER,
  exit_code    INTEGER,
  ttl_ms       INTEGER NOT NULL,
  FOREIGN KEY (service_id) REFERENCES services(service_id)
) STRICT;

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
  prompt_ms         INTEGER,
  predicted_ms      INTEGER,
  prompt_eval_tokens INTEGER,
  draft_tokens      INTEGER,
  draft_tokens_accepted INTEGER,
  FOREIGN KEY (service_id) REFERENCES services(service_id)
) STRICT;

CREATE TABLE device_samples (
  sample_id    INTEGER PRIMARY KEY AUTOINCREMENT,
  device       TEXT NOT NULL,
  timestamp_ms INTEGER NOT NULL,
  total_bytes  INTEGER NOT NULL,
  free_bytes   INTEGER NOT NULL,
  used_bytes   INTEGER NOT NULL
) STRICT;

CREATE TABLE service_restarts (
  restart_id   INTEGER NOT NULL PRIMARY KEY AUTOINCREMENT,
  service_id   INTEGER NOT NULL REFERENCES services(service_id),
  run_id       INTEGER,
  at_ms        INTEGER NOT NULL,
  trigger_name TEXT NOT NULL,
  detail       TEXT NOT NULL
) STRICT;

CREATE TABLE service_restart_counts (
  service_id   INTEGER NOT NULL REFERENCES services(service_id),
  trigger_name TEXT NOT NULL,
  count        INTEGER NOT NULL,
  PRIMARY KEY (service_id, trigger_name)
) STRICT;

CREATE TABLE installation_metadata (
    version    INTEGER PRIMARY KEY,
    owner_uuid TEXT NOT NULL UNIQUE
) STRICT;

CREATE TABLE container_launch_intents (
    intent_id       INTEGER PRIMARY KEY AUTOINCREMENT,
    service_id      INTEGER NOT NULL,
    run_id          INTEGER NOT NULL,
    owner_uuid      TEXT NOT NULL,
    workload_kind   TEXT NOT NULL,
    runtime         TEXT NOT NULL,
    runtime_executable TEXT NOT NULL,
    container_name  TEXT NOT NULL,
    labels_json     TEXT NOT NULL,
    spec_json       TEXT NOT NULL,
    container_id    TEXT,
    state           TEXT NOT NULL DEFAULT 'intent',
    created_at      INTEGER NOT NULL
) STRICT;

-- Copy parents before children so the new foreign-key graph accepts every row.
INSERT INTO services
    (service_id, name, created_at, deleted_at)
SELECT service_id, name, created_at, deleted_at
FROM migration_0012_old_services;

INSERT INTO service_config_versions
    (service_id, version, effective_config, recorded_at)
SELECT service_id, version, effective_config, recorded_at
FROM migration_0012_old_service_config_versions;

INSERT INTO running_services
    (service_id, run_id, pid, spawned_at, command_line, allocation, state,
     workload_kind, runtime, container_name, container_id, runtime_executable)
SELECT service_id, run_id, pid, spawned_at, command_line, allocation, state,
       workload_kind, runtime, container_name, container_id, runtime_executable
FROM migration_0012_old_running_services;

INSERT INTO service_logs
    (service_id, run_id, timestamp_ms, seq, stream, line)
SELECT service_id, run_id, timestamp_ms, seq, stream, line
FROM migration_0012_old_service_logs;

INSERT INTO allocation_events
    (event_id, service_id, run_id, event_type, device, bytes, at)
SELECT event_id, service_id, run_id, event_type, device, bytes, at
FROM migration_0012_old_allocation_events;

INSERT INTO oneshots
    (id, service_id, submitted_at, started_at, ended_at, exit_code, ttl_ms)
SELECT id, service_id, submitted_at, started_at, ended_at, exit_code, ttl_ms
FROM migration_0012_old_oneshots;

INSERT INTO request_metrics
    (metric_id, service_id, run_id, timestamp_ms, endpoint, model,
     prompt_tokens, completion_tokens, duration_ms, ttft_ms, status_code,
     prompt_ms, predicted_ms, prompt_eval_tokens, draft_tokens,
     draft_tokens_accepted)
SELECT metric_id, service_id, run_id, timestamp_ms, endpoint, model,
       prompt_tokens, completion_tokens, duration_ms, ttft_ms, status_code,
       prompt_ms, predicted_ms, prompt_eval_tokens, draft_tokens,
       draft_tokens_accepted
FROM migration_0012_old_request_metrics;

INSERT INTO device_samples
    (sample_id, device, timestamp_ms, total_bytes, free_bytes, used_bytes)
SELECT sample_id, device, timestamp_ms, total_bytes, free_bytes, used_bytes
FROM migration_0012_old_device_samples;

INSERT INTO service_restarts
    (restart_id, service_id, run_id, at_ms, trigger_name, detail)
SELECT restart_id, service_id, run_id, at_ms, trigger_name, detail
FROM migration_0012_old_service_restarts;

INSERT INTO service_restart_counts
    (service_id, trigger_name, count)
SELECT service_id, trigger_name, count
FROM migration_0012_old_service_restart_counts;

INSERT INTO installation_metadata
    (version, owner_uuid)
SELECT version, owner_uuid
FROM migration_0012_old_installation_metadata;

INSERT INTO container_launch_intents
    (intent_id, service_id, run_id, owner_uuid, workload_kind, runtime,
     runtime_executable, container_name, labels_json, spec_json, container_id,
     state, created_at)
SELECT intent_id, service_id, run_id, owner_uuid, workload_kind, runtime,
       runtime_executable, container_name, labels_json, spec_json, container_id,
       state, created_at
FROM migration_0012_old_container_launch_intents;

-- Drop old children before the old services parent. The old definitions still
-- reference the migration_0012_old_services name after the rename above.
DROP TABLE migration_0012_old_service_config_versions;
DROP TABLE migration_0012_old_running_services;
DROP TABLE migration_0012_old_service_logs;
DROP TABLE migration_0012_old_allocation_events;
DROP TABLE migration_0012_old_oneshots;
DROP TABLE migration_0012_old_request_metrics;
DROP TABLE migration_0012_old_device_samples;
DROP TABLE migration_0012_old_service_restarts;
DROP TABLE migration_0012_old_service_restart_counts;
DROP TABLE migration_0012_old_installation_metadata;
DROP TABLE migration_0012_old_container_launch_intents;
DROP TABLE migration_0012_old_services;

CREATE INDEX service_logs_ts ON service_logs(service_id, run_id, timestamp_ms);
CREATE INDEX allocation_events_service ON allocation_events(service_id);
CREATE INDEX oneshots_service ON oneshots(service_id);
CREATE INDEX request_metrics_ts ON request_metrics(service_id, timestamp_ms);
CREATE INDEX request_metrics_run ON request_metrics(run_id);
CREATE INDEX idx_service_restarts_service ON service_restarts(service_id, at_ms);
CREATE INDEX idx_service_restarts_at ON service_restarts(at_ms);
CREATE INDEX idx_running_services_service_id ON running_services(service_id);
CREATE INDEX device_samples_ts ON device_samples(device, timestamp_ms);

-- A copied table advances sqlite_sequence only to its largest copied row. Put
-- back the pre-migration values, including values above the current max rowid.
DELETE FROM sqlite_sequence
WHERE name IN (
    'services',
    'allocation_events',
    'request_metrics',
    'device_samples',
    'service_restarts',
    'container_launch_intents'
);

INSERT INTO sqlite_sequence (name, seq)
SELECT name, seq
FROM migration_0012_sequences;

DROP TABLE migration_0012_sequences;
