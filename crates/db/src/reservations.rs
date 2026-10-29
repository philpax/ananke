//! Persistence for manual reservations: operator-placed holds on device
//! memory that outlive daemon restarts until removed.

use std::collections::BTreeMap;

use ananke_errors::ExpectedError;
use rusqlite::params;

use crate::{Database, models::ManualReservation};

impl Database {
    /// Store a new reservation and its per-device amounts in one
    /// transaction, returning its id.
    pub async fn insert_manual_reservation(
        &self,
        label: Option<&str>,
        created_at_ms: i64,
        devices: &BTreeMap<String, i64>,
    ) -> Result<i64, ExpectedError> {
        let mut conn = self.conn.lock();
        let tx = conn.transaction().map_err(|e| self.db_err(e))?;
        tx.execute(
            "INSERT INTO manual_reservations(label, created_at) VALUES (?1, ?2)",
            params![label, created_at_ms],
        )
        .map_err(|e| self.db_err(e))?;
        let id = tx.last_insert_rowid();
        for (device, mb) in devices {
            tx.execute(
                "INSERT INTO manual_reservation_devices(reservation_id, device, reserved_mb)
                 VALUES (?1, ?2, ?3)",
                params![id, device, mb],
            )
            .map_err(|e| self.db_err(e))?;
        }
        tx.commit().map_err(|e| self.db_err(e))?;
        Ok(id)
    }

    /// Remove a reservation. Returns whether a row existed.
    pub async fn delete_manual_reservation(&self, id: i64) -> Result<bool, ExpectedError> {
        let conn = self.conn.lock();
        let n = conn
            .execute(
                "DELETE FROM manual_reservations WHERE reservation_id = ?1",
                params![id],
            )
            .map_err(|e| self.db_err(e))?;
        Ok(n > 0)
    }

    /// Every stored reservation, oldest first.
    pub async fn list_manual_reservations(&self) -> Result<Vec<ManualReservation>, ExpectedError> {
        let conn = self.conn.lock();
        let mut out: BTreeMap<i64, ManualReservation> = BTreeMap::new();
        let mut stmt = conn
            .prepare(
                "SELECT reservation_id, label, created_at
                 FROM manual_reservations ORDER BY reservation_id",
            )
            .map_err(|e| self.db_err(e))?;
        let rows = stmt
            .query_map([], |r| {
                Ok(ManualReservation {
                    reservation_id: r.get(0)?,
                    label: r.get(1)?,
                    created_at_ms: r.get(2)?,
                    devices: BTreeMap::new(),
                })
            })
            .map_err(|e| self.db_err(e))?;
        for row in rows {
            let row = row.map_err(|e| self.db_err(e))?;
            out.insert(row.reservation_id, row);
        }
        let mut stmt = conn
            .prepare("SELECT reservation_id, device, reserved_mb FROM manual_reservation_devices")
            .map_err(|e| self.db_err(e))?;
        let rows = stmt
            .query_map([], |r| {
                Ok((
                    r.get::<_, i64>(0)?,
                    r.get::<_, String>(1)?,
                    r.get::<_, i64>(2)?,
                ))
            })
            .map_err(|e| self.db_err(e))?;
        for row in rows {
            let (id, device, mb) = row.map_err(|e| self.db_err(e))?;
            if let Some(res) = out.get_mut(&id) {
                res.devices.insert(device, mb);
            }
        }
        Ok(out.into_values().collect())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn reservations_round_trip_and_cascade_on_delete() {
        let dir = tempfile::tempdir().unwrap();
        let db = Database::open(&dir.path().join("a.sqlite")).await.unwrap();
        let devices = BTreeMap::from([("gpu:0".to_string(), 18_432), ("gpu:1".to_string(), 1_024)]);
        let id = db
            .insert_manual_reservation(Some("train"), 1_000, &devices)
            .await
            .unwrap();
        let other = db
            .insert_manual_reservation(None, 2_000, &BTreeMap::from([("cpu".to_string(), 4_096)]))
            .await
            .unwrap();

        let listed = db.list_manual_reservations().await.unwrap();
        assert_eq!(listed.len(), 2);
        assert_eq!(listed[0].reservation_id, id);
        assert_eq!(listed[0].label.as_deref(), Some("train"));
        assert_eq!(listed[0].devices, devices);
        assert_eq!(listed[1].label, None);

        assert!(db.delete_manual_reservation(id).await.unwrap());
        assert!(!db.delete_manual_reservation(id).await.unwrap());
        let listed = db.list_manual_reservations().await.unwrap();
        assert_eq!(listed.len(), 1);
        assert_eq!(listed[0].reservation_id, other);
        let orphans: i64 = db
            .conn
            .lock()
            .query_row(
                "SELECT COUNT(*) FROM manual_reservation_devices WHERE reservation_id = ?1",
                params![id],
                |r| r.get(0),
            )
            .unwrap();
        assert_eq!(orphans, 0);
    }
}
