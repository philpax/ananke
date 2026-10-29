-- 0014_disabled_services: operator-disabled services.
--
-- A row marks a service the operator disabled, so it stays disabled across
-- daemon and host restarts until the operator enables it again. Automatic
-- disables (OOM, crash loops, restart caps) are not recorded here; they clear
-- on restart. Keyed by `service_id`, so a rename carries the mark along.

CREATE TABLE disabled_services (
    service_id  INTEGER PRIMARY KEY,
    disabled_at INTEGER NOT NULL,
    FOREIGN KEY (service_id) REFERENCES services(service_id)
) STRICT;
