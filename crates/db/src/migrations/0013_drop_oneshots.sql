-- 0013_drop_oneshots: remove the oneshot submission table.
--
-- The oneshot API is gone, so nothing reads or writes these rows any more.
-- Dropping the table also drops its `oneshots_service` index, and with it the
-- table's foreign key into `services`.

DROP TABLE oneshots;
