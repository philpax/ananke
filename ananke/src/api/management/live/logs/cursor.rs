//! A log follow's functional core: which lines it has delivered, and so
//! which of a broadcast it should send.
//!
//! A follow reads the store after subscribing to the broadcast, so lines
//! committed between the two arrive on both paths. Within a run, the batcher
//! assigns `seq` in commit order, so a broadcast line at or below the highest
//! seq already delivered for its run was committed before the read and has
//! been sent, or was at or below the client's position. Anything above it is
//! new. Tracking the high-water mark per run rather than one position lets a
//! line from an older run that lands after a newer run's lines through.

use std::collections::BTreeMap;

use ananke_api::internal::log_line::{LogLine, LogPosition};
use ananke_db::{LogFilter, LogsAfter};

/// What a follow has delivered.
#[derive(Debug)]
pub(crate) struct FollowCursor {
    filter: LogFilter,
    /// The client's position. Nothing at or below it is sent.
    after: Option<LogPosition>,
    /// Run id to the highest seq delivered, from backfill or broadcast.
    high_water: BTreeMap<i64, i64>,
}

impl FollowCursor {
    pub(crate) fn new(after: Option<LogPosition>, filter: LogFilter) -> Self {
        Self {
            filter,
            after,
            high_water: BTreeMap::new(),
        }
    }

    pub(crate) fn filter(&self) -> &LogFilter {
        &self.filter
    }

    /// Whether to send a broadcast line, recording it as delivered if so.
    pub(crate) fn admit(&mut self, line: &LogLine) -> bool {
        if !self.filter.matches(line) || self.after.is_some_and(|a| line.position() <= a) {
            return false;
        }
        let high = self.high_water.entry(line.run_id).or_insert(0);
        if line.seq <= *high {
            return false;
        }
        *high = line.seq;
        true
    }

    /// Record a backfill page as delivered. The page holds every matching
    /// row after the resume point up to the read, so its maximum per run is
    /// the true high-water mark even when the cap dropped older rows.
    pub(crate) fn absorb_backfill(&mut self, rows: &[LogLine]) {
        for line in rows {
            let high = self.high_water.entry(line.run_id).or_insert(0);
            *high = (*high).max(line.seq);
        }
    }

    /// Where a fresh read of the store should start: the newest position
    /// delivered, or the client's position if nothing has been.
    pub(crate) fn resume_point(&self) -> Option<LogPosition> {
        let delivered = self
            .high_water
            .iter()
            .next_back()
            .map(|(&run_id, &seq)| LogPosition { run_id, seq });
        delivered.max(self.after)
    }
}

/// Most lines in one `log_lines` frame.
pub(crate) const LINES_PER_FRAME: usize = 256;

/// Rough byte budget for one `log_lines` frame. Clients cap message sizes
/// (tungstenite at 16 MiB by default), and a few long lines can reach that
/// well inside the line limit.
pub(crate) const FRAME_BYTES: usize = 1024 * 1024;

/// Split `lines` into frames of at most [`LINES_PER_FRAME`] lines and about
/// [`FRAME_BYTES`] bytes. A single line over the budget gets a frame alone.
pub(crate) fn frames(lines: &[LogLine]) -> Vec<&[LogLine]> {
    let mut out = Vec::new();
    let mut start = 0;
    let mut bytes = 0;
    for (i, line) in lines.iter().enumerate() {
        // The JSON wrapper around each line is under 128 bytes.
        let size = line.line.len() + line.stream.len() + 128;
        if i > start && (i - start == LINES_PER_FRAME || bytes + size > FRAME_BYTES) {
            out.push(&lines[start..i]);
            start = i;
            bytes = 0;
        }
        bytes += size;
    }
    if start < lines.len() {
        out.push(&lines[start..]);
    }
    out
}

/// The gap a capped backfill leaves: the first position sent after it, and
/// how many matching lines it holds.
pub(crate) fn gap(page: &LogsAfter) -> Option<(LogPosition, u64)> {
    match page.rows.first() {
        Some(first) if page.skipped > 0 => Some((first.position(), page.skipped)),
        _ => None,
    }
}

#[cfg(test)]
mod tests {
    use ananke_api::internal::log_line::LogStream;

    use super::*;

    fn line(run_id: i64, seq: i64) -> LogLine {
        line_on(run_id, seq, "stdout")
    }

    fn line_on(run_id: i64, seq: i64, stream: &str) -> LogLine {
        LogLine {
            timestamp_ms: 0,
            stream: stream.into(),
            line: format!("{run_id}:{seq}"),
            run_id,
            seq,
        }
    }

    fn admitted(cursor: &mut FollowCursor, lines: &[LogLine]) -> Vec<(i64, i64)> {
        lines
            .iter()
            .filter(|l| cursor.admit(l))
            .map(|l| (l.run_id, l.seq))
            .collect()
    }

    #[test]
    fn a_broadcast_overlapping_the_backfill_is_deduplicated() {
        let mut cursor = FollowCursor::new(None, LogFilter::default());
        let backfill: Vec<_> = (1..=5).map(|seq| line(1, seq)).collect();
        cursor.absorb_backfill(&backfill);
        let replay: Vec<_> = (4..=6).map(|seq| line(1, seq)).collect();
        assert_eq!(admitted(&mut cursor, &replay), [(1, 6)]);
        assert_eq!(admitted(&mut cursor, &replay), []);
    }

    #[test]
    fn a_new_run_is_admitted_from_its_first_line() {
        let mut cursor = FollowCursor::new(None, LogFilter::default());
        cursor.absorb_backfill(&[line(1, 40)]);
        assert_eq!(
            admitted(&mut cursor, &[line(2, 1), line(2, 2)]),
            [(2, 1), (2, 2)]
        );
    }

    #[test]
    fn interleaved_runs_are_tracked_separately() {
        let mut cursor = FollowCursor::new(None, LogFilter::default());
        cursor.absorb_backfill(&[line(1, 3), line(2, 1)]);
        // A late line from the old run is still above that run's mark.
        assert_eq!(
            admitted(
                &mut cursor,
                &[line(1, 3), line(1, 4), line(2, 1), line(2, 2)]
            ),
            [(1, 4), (2, 2)]
        );
    }

    #[test]
    fn nothing_at_or_below_the_clients_position_is_sent() {
        let after = LogPosition { run_id: 2, seq: 5 };
        let mut cursor = FollowCursor::new(Some(after), LogFilter::default());
        assert_eq!(
            admitted(
                &mut cursor,
                &[line(1, 9), line(2, 5), line(2, 6), line(3, 1)]
            ),
            [(2, 6), (3, 1)]
        );
    }

    #[test]
    fn filters_apply_and_do_not_advance_the_mark() {
        let filter = LogFilter {
            run: Some(2),
            stream: Some(LogStream::Stderr),
        };
        let mut cursor = FollowCursor::new(None, filter);
        let lines = [
            line_on(1, 1, "stderr"),
            line_on(2, 1, "stdout"),
            line_on(2, 2, "stderr"),
        ];
        assert_eq!(admitted(&mut cursor, &lines), [(2, 2)]);
        assert_eq!(
            cursor.resume_point(),
            Some(LogPosition { run_id: 2, seq: 2 })
        );
    }

    #[test]
    fn the_resume_point_is_the_newest_delivered_or_the_clients() {
        let after = LogPosition { run_id: 3, seq: 7 };
        let mut cursor = FollowCursor::new(Some(after), LogFilter::default());
        assert_eq!(cursor.resume_point(), Some(after));
        cursor.admit(&line(3, 8));
        cursor.admit(&line(4, 1));
        assert_eq!(
            cursor.resume_point(),
            Some(LogPosition { run_id: 4, seq: 1 })
        );
        assert_eq!(
            FollowCursor::new(None, LogFilter::default()).resume_point(),
            None
        );
    }

    #[test]
    fn frames_are_bounded_by_count_and_bytes() {
        let short: Vec<_> = (1..=600).map(|seq| line(1, seq)).collect();
        let sizes: Vec<usize> = frames(&short).iter().map(|f| f.len()).collect();
        assert_eq!(sizes, [256, 256, 88]);
        assert!(frames(&[]).is_empty());

        let long = |seq| LogLine {
            line: "x".repeat(FRAME_BYTES / 2),
            ..line(1, seq)
        };
        let lines = [long(1), long(2), long(3)];
        let sizes: Vec<usize> = frames(&lines).iter().map(|f| f.len()).collect();
        assert_eq!(sizes, [1, 1, 1]);
        let huge = [LogLine {
            line: "x".repeat(FRAME_BYTES * 2),
            ..line(1, 1)
        }];
        assert_eq!(frames(&huge).len(), 1);
    }

    #[test]
    fn a_capped_backfill_reports_its_gap() {
        let page = LogsAfter {
            rows: vec![line(2, 501), line(2, 502)],
            skipped: 500,
        };
        assert_eq!(
            gap(&page),
            Some((
                LogPosition {
                    run_id: 2,
                    seq: 501
                },
                500
            ))
        );
        let whole = LogsAfter {
            rows: vec![line(2, 1)],
            skipped: 0,
        };
        assert_eq!(gap(&whole), None);
    }
}
