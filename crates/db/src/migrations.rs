//! Versioned schema migrations.
//!
//! Each [`Migration`] is an immutable (version, name, SQL) triple. At open
//! time, [`apply_pending`] bootstraps a `schema_version` table, reads which
//! versions are recorded there, and applies any unrecorded migration in
//! version order. Each migration runs inside a transaction alongside its
//! `schema_version` insert, so a mid-migration failure leaves the database
//! in its pre-migration state.
//!
//! Adding a migration:
//! 1. Drop a `NNNN_description.sql` file in `src/db/migrations/`.
//! 2. Append a [`Migration`] entry to [`MIGRATIONS`] with the next
//!    version number; version numbers are monotonic (gaps allowed but
//!    discouraged).
//! 3. Never mutate an existing entry — live databases have already
//!    recorded it as applied.

use rusqlite::{Connection, params};
use sha2::{Digest, Sha256};
use tracing::info;

/// One immutable schema change. Applied exactly once per database.
pub struct Migration {
    pub version: u32,
    pub name: &'static str,
    pub source_path: &'static str,
    pub digest: &'static str,
    pub sql: &'static str,
}

/// Chain of migrations in application order. Append-only.
pub const MIGRATIONS: &[Migration] = &[
    Migration {
        version: 1,
        name: "initial_schema",
        source_path: "migrations/0001_initial_schema.sql",
        digest: "ac5f291612e0f6f36b103d7b0b5d21400820a6af80404e8c37a4eebe8f567b36",
        sql: include_str!("migrations/0001_initial_schema.sql"),
    },
    Migration {
        version: 2,
        name: "metrics",
        source_path: "migrations/0002_metrics.sql",
        digest: "cb3bc2b587b36736bf2929c04b4e495b7270e00f1884fc40115a44ee1a911276",
        sql: include_str!("migrations/0002_metrics.sql"),
    },
    Migration {
        version: 3,
        name: "engine_timings",
        source_path: "migrations/0003_engine_timings.sql",
        digest: "fb3e441a20b12d2dfaa85e9d6f93dde8da80afdad8905bdc7112d339755aa7f3",
        sql: include_str!("migrations/0003_engine_timings.sql"),
    },
    Migration {
        version: 4,
        name: "prompt_eval_tokens",
        source_path: "migrations/0004_prompt_eval_tokens.sql",
        digest: "18e3bfd9a9aab7e9f6b96907f370462552b1bbb8a52eb647f3b9aa7f6aad0085",
        sql: include_str!("migrations/0004_prompt_eval_tokens.sql"),
    },
    Migration {
        version: 5,
        name: "draft_tokens",
        source_path: "migrations/0005_draft_tokens.sql",
        digest: "b0ea1b4d088559a8ef9d48616af41c463948fb87a0e6073cbbfc291592ee89cf",
        sql: include_str!("migrations/0005_draft_tokens.sql"),
    },
    Migration {
        version: 6,
        name: "service_restarts",
        source_path: "migrations/0006_service_restarts.sql",
        digest: "44f70de1a0ed8f09cb1d7c41e010a750c4742e7918d8cc48ffffed2cd4163cc2",
        sql: include_str!("migrations/0006_service_restarts.sql"),
    },
    Migration {
        version: 7,
        name: "service_restart_counts",
        source_path: "migrations/0007_service_restart_counts.sql",
        digest: "1a7b5a1a1b89378fdea13cea0432afb5d6a5eab5856a828450a4f955c6b61b23",
        sql: include_str!("migrations/0007_service_restart_counts.sql"),
    },
    Migration {
        version: 8,
        name: "installation_metadata",
        source_path: "migrations/0008_installation_metadata.sql",
        digest: "e77770990cd4cec5ac46c14401578d5410138faa118fe06ec51d6059fc6bc64d",
        sql: include_str!("migrations/0008_installation_metadata.sql"),
    },
    Migration {
        version: 9,
        name: "container_launch_intents",
        source_path: "migrations/0009_container_launch_intents.sql",
        digest: "bdff994a345e5a2191d3936e19acef6a80002e9e868197583d6cf080f38ecf02",
        sql: include_str!("migrations/0009_container_launch_intents.sql"),
    },
    Migration {
        version: 10,
        name: "managed_workloads",
        source_path: "migrations/0010_managed_workloads.sql",
        digest: "755df758d12126208bc467b8cad80a10b9759701974cc9da1f9438dbd8f4285c",
        sql: include_str!("migrations/0010_managed_workloads.sql"),
    },
    Migration {
        version: 11,
        name: "running_runtime_executable",
        source_path: "migrations/0011_running_runtime_executable.sql",
        digest: "c74319bcb04a127d0b1cce4339256baca645b27619d7b9cbc4f161f7eaa6f161",
        sql: include_str!("migrations/0011_running_runtime_executable.sql"),
    },
    Migration {
        version: 12,
        name: "strict",
        source_path: "migrations/0012_strict_tables.sql",
        digest: "9f663d483b3e2d33a6c6037857fe53ff73484ec501a60d81615668a7274c6bef",
        sql: include_str!("migrations/0012_strict_tables.sql"),
    },
];

fn validate_integrity(conn: &Connection) -> Result<(), rusqlite::Error> {
    let mut foreign_keys = conn.prepare("PRAGMA foreign_key_check")?;
    if foreign_keys.query_map([], |_| Ok(()))?.next().is_some() {
        return Err(rusqlite::Error::InvalidQuery);
    }
    let quick: String = conn.query_row("PRAGMA quick_check", [], |row| row.get(0))?;
    if quick == "ok" {
        Ok(())
    } else {
        Err(rusqlite::Error::InvalidQuery)
    }
}

/// Ensure the version tracker exists and apply pending migrations through `target`.
/// Migrations above the target are retained, never downgraded, and remain part
/// of the validated history.
pub fn apply_pending(conn: &mut Connection, now_ms: i64) -> Result<Vec<u32>, rusqlite::Error> {
    apply_pending_to(conn, now_ms, 12)
}

pub fn apply_pending_to(
    conn: &mut Connection,
    now_ms: i64,
    target: u32,
) -> Result<Vec<u32>, rusqlite::Error> {
    conn.execute_batch(
        "CREATE TABLE IF NOT EXISTS schema_version (\n         version INTEGER NOT NULL PRIMARY KEY,\n         name TEXT NOT NULL,\n         applied_at INTEGER NOT NULL,\n         digest TEXT\n     );",
    )?;
    let has_digest: bool = conn
        .prepare("PRAGMA table_info(schema_version)")?
        .query_map([], |row| row.get::<_, String>(1))?
        .collect::<rusqlite::Result<Vec<_>>>()?
        .iter()
        .any(|name| name == "digest");
    let history: Vec<(u32, String, Option<String>)> = if has_digest {
        let mut stmt =
            conn.prepare("SELECT version, name, digest FROM schema_version ORDER BY version")?;
        let rows = stmt.query_map([], |r| Ok((r.get(0)?, r.get(1)?, r.get(2)?)))?;
        rows.collect::<rusqlite::Result<_>>()?
    } else {
        let mut stmt = conn.prepare("SELECT version, name FROM schema_version ORDER BY version")?;
        let rows = stmt.query_map([], |r| Ok((r.get(0)?, r.get(1)?, None)))?;
        rows.collect::<rusqlite::Result<_>>()?
    };
    for (index, (version, name, digest)) in history.iter().enumerate() {
        let Some(migration) = MIGRATIONS.get(index) else {
            return Err(rusqlite::Error::InvalidQuery);
        };
        if *version != migration.version || name != migration.name {
            return Err(rusqlite::Error::InvalidQuery);
        }
        let expected_digest = format!("{:x}", Sha256::digest(migration.sql.as_bytes()));
        if expected_digest != migration.digest {
            return Err(rusqlite::Error::InvalidQuery);
        }
        if let Some(digest) = digest
            && !digest.is_empty()
            && digest != migration.digest
        {
            return Err(rusqlite::Error::InvalidQuery);
        }
    }

    let applied: std::collections::HashMap<u32, ()> =
        history.iter().map(|(v, _, _)| (*v, ())).collect();
    let mut applied_now = Vec::new();
    for migration in MIGRATIONS.iter().filter(|m| m.version <= target) {
        if applied.contains_key(&migration.version) {
            continue;
        }
        let tx = conn.transaction()?;
        tx.execute_batch(migration.sql)?;
        if !has_digest && migration.version == 12 {
            tx.execute_batch("ALTER TABLE schema_version ADD COLUMN digest TEXT")?;
        }
        tx.execute(
            "INSERT INTO schema_version(version, name, applied_at, digest) VALUES (?1, ?2, ?3, ?4)",
            params![migration.version, migration.name, now_ms, migration.digest],
        )?;
        if target >= 12 && migration.version == 12 {
            validate_integrity(&tx)?;
        }
        tx.commit()?;
        info!(
            version = migration.version,
            name = migration.name,
            "applied migration"
        );
        applied_now.push(migration.version);
    }
    Ok(applied_now)
}

#[cfg(test)]
mod tests {
    use std::collections::HashSet;

    use super::*;

    #[test]
    fn migrations_apply_once_and_are_idempotent() {
        let mut conn = Connection::open_in_memory().unwrap();

        let first = apply_pending(&mut conn, 1_000).unwrap();
        assert!(!first.is_empty(), "fresh DB must apply at least one");

        let second = apply_pending(&mut conn, 2_000).unwrap();
        assert!(
            second.is_empty(),
            "re-open on an up-to-date DB must apply nothing, got {second:?}"
        );
    }

    #[test]
    fn versions_are_strictly_monotonic_and_unique() {
        let mut seen = HashSet::new();
        let mut last = 0;
        for m in MIGRATIONS {
            assert!(
                m.version > last,
                "migration {} ({}): versions must strictly increase",
                m.version,
                m.name
            );
            assert!(
                seen.insert(m.version),
                "duplicate version {} in MIGRATIONS",
                m.version
            );
            last = m.version;
        }
    }

    #[test]
    fn strict_migration_preserves_schema_data_indexes_and_sequences() {
        let mut conn = Connection::open_in_memory().unwrap();
        apply_migrations_through(&mut conn, 11);
        seed_v11_database(&conn);

        let applied = apply_pending_to(&mut conn, 2_000, 12).unwrap();
        assert_eq!(applied, vec![12]);

        let strict_tables: Vec<String> = conn
            .prepare(
                "SELECT name FROM sqlite_schema
                 WHERE type = 'table' AND name NOT LIKE 'sqlite_%'
                   AND sql LIKE '% STRICT' ORDER BY name",
            )
            .unwrap()
            .query_map([], |row| row.get(0))
            .unwrap()
            .collect::<rusqlite::Result<_>>()
            .unwrap();
        assert_eq!(
            strict_tables,
            vec![
                "allocation_events",
                "container_launch_intents",
                "device_samples",
                "installation_metadata",
                "oneshots",
                "request_metrics",
                "running_services",
                "service_config_versions",
                "service_logs",
                "service_restart_counts",
                "service_restarts",
                "services",
            ]
        );

        let expected_columns = [
            ("services", "service_id,name,created_at,deleted_at"),
            (
                "service_config_versions",
                "service_id,version,effective_config,recorded_at",
            ),
            (
                "running_services",
                "service_id,run_id,pid,spawned_at,command_line,allocation,state,workload_kind,runtime,container_name,container_id,runtime_executable",
            ),
            (
                "service_logs",
                "service_id,run_id,timestamp_ms,seq,stream,line",
            ),
            (
                "allocation_events",
                "event_id,service_id,run_id,event_type,device,bytes,at",
            ),
            (
                "oneshots",
                "id,service_id,submitted_at,started_at,ended_at,exit_code,ttl_ms",
            ),
            (
                "request_metrics",
                "metric_id,service_id,run_id,timestamp_ms,endpoint,model,prompt_tokens,completion_tokens,duration_ms,ttft_ms,status_code,prompt_ms,predicted_ms,prompt_eval_tokens,draft_tokens,draft_tokens_accepted",
            ),
            (
                "device_samples",
                "sample_id,device,timestamp_ms,total_bytes,free_bytes,used_bytes",
            ),
            (
                "service_restarts",
                "restart_id,service_id,run_id,at_ms,trigger_name,detail",
            ),
            ("service_restart_counts", "service_id,trigger_name,count"),
            ("installation_metadata", "version,owner_uuid"),
            (
                "container_launch_intents",
                "intent_id,service_id,run_id,owner_uuid,workload_kind,runtime,runtime_executable,container_name,labels_json,spec_json,container_id,state,created_at",
            ),
        ];
        for (table, expected) in expected_columns {
            let actual: Vec<String> = conn
                .prepare(&format!("PRAGMA table_info({table})"))
                .unwrap()
                .query_map([], |row| row.get(1))
                .unwrap()
                .collect::<rusqlite::Result<_>>()
                .unwrap();
            assert_eq!(actual.join(","), expected, "columns changed for {table}");
        }

        let expected_indexes = [
            (
                "service_logs",
                "service_logs_ts",
                "service_id,run_id,timestamp_ms",
            ),
            (
                "allocation_events",
                "allocation_events_service",
                "service_id",
            ),
            ("oneshots", "oneshots_service", "service_id"),
            (
                "request_metrics",
                "request_metrics_ts",
                "service_id,timestamp_ms",
            ),
            ("request_metrics", "request_metrics_run", "run_id"),
            ("device_samples", "device_samples_ts", "device,timestamp_ms"),
            (
                "service_restarts",
                "idx_service_restarts_service",
                "service_id,at_ms",
            ),
            ("service_restarts", "idx_service_restarts_at", "at_ms"),
            (
                "running_services",
                "idx_running_services_service_id",
                "service_id",
            ),
        ];
        for (table, index, expected) in expected_indexes {
            let actual: Vec<String> = conn
                .prepare(&format!("PRAGMA index_info({index})"))
                .unwrap()
                .query_map([], |row| row.get(2))
                .unwrap()
                .collect::<rusqlite::Result<_>>()
                .unwrap();
            assert_eq!(actual.join(","), expected, "index changed: {index}");
            let index_table: String = conn
                .query_row(
                    "SELECT tbl_name FROM sqlite_schema WHERE type = 'index' AND name = ?1",
                    [index],
                    |row| row.get(0),
                )
                .unwrap();
            assert_eq!(index_table, table, "index moved: {index}");
        }

        let expected_foreign_keys = [
            ("service_config_versions", "service_id", "services"),
            ("running_services", "service_id", "services"),
            ("service_logs", "service_id", "services"),
            ("allocation_events", "service_id", "services"),
            ("oneshots", "service_id", "services"),
            ("service_restarts", "service_id", "services"),
            ("service_restart_counts", "service_id", "services"),
        ];
        for (table, from, parent) in expected_foreign_keys {
            let actual: Vec<(String, String)> = conn
                .prepare(&format!("PRAGMA foreign_key_list({table})"))
                .unwrap()
                .query_map([], |row| Ok((row.get(2)?, row.get(3)?)))
                .unwrap()
                .collect::<rusqlite::Result<_>>()
                .unwrap();
            assert_eq!(actual, vec![(parent.to_string(), from.to_string())]);
        }
        let fk_violations: Vec<String> = conn
            .prepare("PRAGMA foreign_key_check")
            .unwrap()
            .query_map([], |row| row.get(0))
            .unwrap()
            .collect::<rusqlite::Result<_>>()
            .unwrap();
        assert!(fk_violations.is_empty());

        let preserved_rows: Vec<(String, i64)> = conn
            .prepare("SELECT name, created_at FROM services")
            .unwrap()
            .query_map([], |row| Ok((row.get(0)?, row.get(1)?)))
            .unwrap()
            .collect::<rusqlite::Result<_>>()
            .unwrap();
        assert_eq!(preserved_rows, vec![("demo".to_string(), 100)]);
        let log_line: String = conn
            .query_row(
                "SELECT line FROM service_logs WHERE service_id = 1",
                [],
                |row| row.get(0),
            )
            .unwrap();
        assert_eq!(log_line, "hello");
        let intent_name: String = conn
            .query_row(
                "SELECT container_name FROM container_launch_intents WHERE intent_id = 1",
                [],
                |row| row.get(0),
            )
            .unwrap();
        assert_eq!(intent_name, "demo-7");

        for (name, expected) in [
            ("services", 1_000),
            ("allocation_events", 2_000),
            ("request_metrics", 3_000),
            ("device_samples", 4_000),
            ("service_restarts", 5_000),
            ("container_launch_intents", 6_000),
        ] {
            let actual: i64 = conn
                .query_row(
                    "SELECT seq FROM sqlite_sequence WHERE name = ?1",
                    [name],
                    |row| row.get(0),
                )
                .unwrap();
            assert_eq!(actual, expected, "sequence changed for {name}");
        }

        let error = conn
            .execute(
                "INSERT INTO services(name, created_at) VALUES ('wrong-type', 'not-an-integer')",
                [],
            )
            .unwrap_err();
        assert!(
            error
                .to_string()
                .contains("cannot store TEXT value in INTEGER column"),
            "STRICT type error should identify the rejected value: {error}"
        );
        let error = conn
            .execute(
                "INSERT INTO running_services
                 (service_id, run_id, pid, spawned_at, command_line, allocation, state)
                 VALUES (999, 1, 1, 1, 'cmd', '{}', 'running')",
                [],
            )
            .unwrap_err();
        assert!(
            error.to_string().contains("FOREIGN KEY constraint failed"),
            "foreign-key rejection should remain enabled: {error}"
        );
    }

    #[test]
    fn strict_migration_failure_rolls_back_every_change() {
        let mut conn = Connection::open_in_memory().unwrap();
        apply_migrations_through(&mut conn, 11);
        conn.execute(
            "INSERT INTO services(name, created_at) VALUES ('legacy-bad-row', 'not-an-integer')",
            [],
        )
        .unwrap();

        let error = apply_pending_to(&mut conn, 2_000, 12).unwrap_err();
        assert!(
            error
                .to_string()
                .contains("cannot store TEXT value in INTEGER column"),
            "the strict copy should be the failing statement: {error}"
        );
        let latest_version: u32 = conn
            .query_row("SELECT MAX(version) FROM schema_version", [], |row| {
                row.get(0)
            })
            .unwrap();
        assert_eq!(latest_version, 11);

        let services_sql: String = conn
            .query_row(
                "SELECT sql FROM sqlite_schema WHERE type = 'table' AND name = 'services'",
                [],
                |row| row.get(0),
            )
            .unwrap();
        assert!(!services_sql.ends_with(" STRICT"));
        let created_at: String = conn
            .query_row(
                "SELECT created_at FROM services WHERE name = 'legacy-bad-row'",
                [],
                |row| row.get(0),
            )
            .unwrap();
        assert_eq!(created_at, "not-an-integer");
        let index_count: i64 = conn
            .query_row(
                "SELECT COUNT(*) FROM sqlite_schema
                 WHERE type = 'index' AND name = 'service_logs_ts'",
                [],
                |row| row.get(0),
            )
            .unwrap();
        assert_eq!(index_count, 1);
        let temporary_objects: i64 = conn
            .query_row(
                "SELECT COUNT(*) FROM sqlite_temp_schema
                 WHERE name LIKE 'migration_0012_%'",
                [],
                |row| row.get(0),
            )
            .unwrap();
        assert_eq!(temporary_objects, 0);
    }

    fn apply_migrations_through(conn: &mut Connection, version: u32) {
        conn.execute_batch(
            "PRAGMA foreign_keys = ON;
             CREATE TABLE IF NOT EXISTS schema_version (
                 version INTEGER NOT NULL PRIMARY KEY,
                 name TEXT NOT NULL,
                 applied_at INTEGER NOT NULL,
                 digest TEXT
             );",
        )
        .unwrap();
        for migration in MIGRATIONS.iter().take(version as usize) {
            let tx = conn.transaction().unwrap();
            tx.execute_batch(migration.sql).unwrap();
            tx.execute(
                "INSERT INTO schema_version(version, name, applied_at, digest) VALUES (?1, ?2, ?3, ?4)",
                params![migration.version, migration.name, 1_000, migration.digest],
            )
            .unwrap();
            tx.commit().unwrap();
        }
    }

    fn seed_v11_database(conn: &Connection) {
        conn.execute_batch(
            "INSERT INTO services(service_id, name, created_at) VALUES (1, 'demo', 100);
             INSERT INTO service_config_versions(service_id, version, effective_config, recorded_at)
                 VALUES (1, 1, '{}', 101);
             INSERT INTO running_services
                 (service_id, run_id, pid, spawned_at, command_line, allocation, state,
                  workload_kind, runtime, container_name, container_id, runtime_executable)
                 VALUES (1, 7, 1234, 100, 'cmd', '{}', 'running', 'native', NULL, NULL, NULL, NULL);
             INSERT INTO service_logs(service_id, run_id, timestamp_ms, seq, stream, line)
                 VALUES (1, 7, 102, 1, 'stdout', 'hello');
             INSERT INTO allocation_events(event_id, service_id, run_id, event_type, device, bytes, at)
                 VALUES (1, 1, 7, 'allocated', 'cpu', 10, 103);
             INSERT INTO oneshots(id, service_id, submitted_at, started_at, ended_at, exit_code, ttl_ms)
                 VALUES ('one', 1, 104, 105, 106, 0, 1000);
             INSERT INTO request_metrics
                 (metric_id, service_id, run_id, timestamp_ms, endpoint, model, prompt_tokens,
                  completion_tokens, duration_ms, ttft_ms, status_code, prompt_ms, predicted_ms,
                  prompt_eval_tokens, draft_tokens, draft_tokens_accepted)
                 VALUES (1, 1, 7, 107, '/v1/chat/completions', 'demo', 2, 3, 4, 5, 200, 6, 7, 8, 9, 10);
             INSERT INTO device_samples(sample_id, device, timestamp_ms, total_bytes, free_bytes, used_bytes)
                 VALUES (1, 'gpu:0', 108, 100, 90, 10);
             INSERT INTO service_restarts(restart_id, service_id, run_id, at_ms, trigger_name, detail)
                 VALUES (1, 1, 7, 109, 'periodic', 'test');
             INSERT INTO service_restart_counts(service_id, trigger_name, count)
                 VALUES (1, 'periodic', 1);
             INSERT INTO installation_metadata(version, owner_uuid)
                 VALUES (1, '00000000-0000-0000-0000-000000000001');
             INSERT INTO container_launch_intents
                 (intent_id, service_id, run_id, owner_uuid, workload_kind, runtime,
                  runtime_executable, container_name, labels_json, spec_json, container_id,
                  state, created_at)
                 VALUES (1, 1, 7, 'owner', 'container', 'docker', '/bin/docker',
                         'demo-7', '{}', '{}', NULL, 'intent', 110);
             UPDATE sqlite_sequence SET seq = CASE name
                 WHEN 'services' THEN 1000
                 WHEN 'allocation_events' THEN 2000
                 WHEN 'request_metrics' THEN 3000
                 WHEN 'device_samples' THEN 4000
                 WHEN 'service_restarts' THEN 5000
                 WHEN 'container_launch_intents' THEN 6000
             END
             WHERE name IN ('services', 'allocation_events', 'request_metrics',
                            'device_samples', 'service_restarts', 'container_launch_intents');",
        )
        .unwrap();
    }
}
