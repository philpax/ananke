-- 0001_initial_schema: the tables ananke has shipped with.
--
-- Migrations are applied by `db::migrations::apply_pending` inside a
-- transaction alongside a row insert into `schema_version`, so a
-- mid-migration failure leaves the DB in its pre-migration state.
-- File-persistent pragmas (`journal_mode`, `auto_vacuum`, etc.) and
-- the `schema_version` table itself are bootstrapped outside the
-- migration chain; this file is therefore pure DDL for the app
-- tables.

CREATE TABLE services (
  service_id INTEGER PRIMARY KEY AUTOINCREMENT,
  name       TEXT NOT NULL UNIQUE,
  created_at INTEGER NOT NULL,
  deleted_at INTEGER
);

CREATE TABLE service_config_versions (
  service_id       INTEGER NOT NULL,
  version          INTEGER NOT NULL,
  effective_config TEXT NOT NULL,
  recorded_at      INTEGER NOT NULL,
  PRIMARY KEY (service_id, version),
  FOREIGN KEY (service_id) REFERENCES services(service_id)
);

CREATE TABLE running_services (
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

CREATE TABLE service_logs (
  service_id   INTEGER NOT NULL,
  run_id       INTEGER NOT NULL,
  timestamp_ms INTEGER NOT NULL,
  seq          INTEGER NOT NULL,
  stream       TEXT NOT NULL,
  line         TEXT NOT NULL,
  PRIMARY KEY (service_id, run_id, seq),
  FOREIGN KEY (service_id) REFERENCES services(service_id)
);
CREATE INDEX service_logs_ts ON service_logs(service_id, run_id, timestamp_ms);

CREATE TABLE allocation_events (
  event_id   INTEGER PRIMARY KEY AUTOINCREMENT,
  service_id INTEGER NOT NULL,
  run_id     INTEGER NOT NULL,
  event_type TEXT NOT NULL,
  device     TEXT NOT NULL,
  bytes      INTEGER NOT NULL,
  at         INTEGER NOT NULL,
  FOREIGN KEY (service_id) REFERENCES services(service_id)
);
CREATE INDEX allocation_events_service ON allocation_events(service_id);

CREATE TABLE oneshots (
  id           TEXT PRIMARY KEY,
  service_id   INTEGER NOT NULL,
  submitted_at INTEGER NOT NULL,
  started_at   INTEGER,
  ended_at     INTEGER,
  exit_code    INTEGER,
  ttl_ms       INTEGER NOT NULL,
  FOREIGN KEY (service_id) REFERENCES services(service_id)
);
CREATE INDEX oneshots_service ON oneshots(service_id);
