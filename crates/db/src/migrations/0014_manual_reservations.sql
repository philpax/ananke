-- 0014_manual_reservations: operator-placed capacity reservations.
--
-- A manual reservation holds memory on one or more devices that no service
-- may be placed into. Each row survives daemon restarts until an operator
-- removes it. The per-device amounts live in a child table so a reservation
-- can span several devices with a different amount on each.

CREATE TABLE manual_reservations (
    reservation_id INTEGER PRIMARY KEY AUTOINCREMENT,
    label          TEXT,
    created_at     INTEGER NOT NULL
) STRICT;

CREATE TABLE manual_reservation_devices (
    reservation_id INTEGER NOT NULL,
    device         TEXT NOT NULL,
    reserved_mb    INTEGER NOT NULL CHECK (reserved_mb > 0),
    PRIMARY KEY (reservation_id, device),
    FOREIGN KEY (reservation_id) REFERENCES manual_reservations(reservation_id)
        ON DELETE CASCADE
) STRICT;
