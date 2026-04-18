//! SQLite schema migrations.

pub const MIGRATION_0001: &str = r#"
PRAGMA auto_vacuum = INCREMENTAL;
PRAGMA journal_mode = WAL;
PRAGMA synchronous = NORMAL;
PRAGMA foreign_keys = ON;

CREATE TABLE IF NOT EXISTS schema_version (
  version INTEGER PRIMARY KEY
);

CREATE TABLE IF NOT EXISTS services (
  service_id INTEGER PRIMARY KEY,
  name       TEXT NOT NULL UNIQUE,
  created_at INTEGER NOT NULL,
  deleted_at INTEGER
);

CREATE TABLE IF NOT EXISTS service_config_versions (
  service_id       INTEGER NOT NULL,
  version          INTEGER NOT NULL,
  effective_config TEXT NOT NULL,
  recorded_at      INTEGER NOT NULL,
  PRIMARY KEY (service_id, version),
  FOREIGN KEY (service_id) REFERENCES services(service_id)
);

CREATE TABLE IF NOT EXISTS running_services (
  service_id   INTEGER NOT NULL,
  run_id       INTEGER NOT NULL,
  pid          INTEGER NOT NULL,
  spawned_at   INTEGER NOT NULL,
  command_line TEXT NOT NULL,
  allocation   TEXT NOT NULL,
  state        TEXT NOT NULL,
  PRIMARY KEY (service_id, run_id),
  FOREIGN KEY (service_id) REFERENCES services(service_id)
);

CREATE TABLE IF NOT EXISTS service_logs (
  service_id   INTEGER NOT NULL,
  run_id       INTEGER NOT NULL,
  timestamp_ms INTEGER NOT NULL,
  seq          INTEGER NOT NULL,
  stream       TEXT NOT NULL,
  line         TEXT NOT NULL,
  PRIMARY KEY (service_id, run_id, seq),
  FOREIGN KEY (service_id) REFERENCES services(service_id)
);
CREATE INDEX IF NOT EXISTS service_logs_ts ON service_logs(service_id, run_id, timestamp_ms);

CREATE TABLE IF NOT EXISTS allocation_events (
  event_id   INTEGER PRIMARY KEY,
  service_id INTEGER NOT NULL,
  run_id     INTEGER NOT NULL,
  event_type TEXT NOT NULL,
  device     TEXT NOT NULL,
  bytes      INTEGER NOT NULL,
  at         INTEGER NOT NULL,
  FOREIGN KEY (service_id) REFERENCES services(service_id)
);

CREATE TABLE IF NOT EXISTS oneshots (
  id           TEXT PRIMARY KEY,
  service_id   INTEGER NOT NULL,
  submitted_at INTEGER NOT NULL,
  started_at   INTEGER,
  ended_at     INTEGER,
  exit_code    INTEGER,
  ttl_ms       INTEGER NOT NULL,
  FOREIGN KEY (service_id) REFERENCES services(service_id)
);

INSERT OR IGNORE INTO schema_version(version) VALUES (1);
"#;
