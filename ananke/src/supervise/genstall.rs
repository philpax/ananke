//! Generation-stall detection via the child's Prometheus `/metrics` counters.
//!
//! A wedged llama.cpp child (SWA/hybrid cache-invalidation bugs, e.g.
//! ggml-org/llama.cpp#22450) accepts a request and then never advances —
//! neither prompt processing nor token generation — while staying alive and
//! healthy-looking. A non-streaming client gets no signal the proxy can
//! observe, so the TTFT stall watchdog (which watches response frames) never
//! arms. This module instead polls the child's own progress counters:
//! `llamacpp:prompt_tokens_total` (prefill) and
//! `llamacpp:tokens_predicted_total` (decode). Their sum advances every batch
//! on a healthy run, whatever the phase; flat counters with at least one
//! request in flight for the full timeout is the wedge signature.
//!
//! The decision logic is pure ([`GenStallState`]); the supervisor's Running
//! loop owns the poll cadence and performs the HTTP fetch via
//! [`fetch_progress`].

use std::time::Duration;

use tokio::time::Instant;

/// Pure stall-decision state: the last observed progress value and when it
/// last advanced. Timestamps use [`tokio::time::Instant`] so tests under
/// `start_paused = true` drive the clock deterministically.
pub struct GenStallState {
    last_value: Option<u64>,
    last_advance: Instant,
}

impl GenStallState {
    /// A fresh state anchored at `now`; the first observation seeds the
    /// counter baseline rather than firing.
    pub fn new(now: Instant) -> Self {
        Self {
            last_value: None,
            last_advance: now,
        }
    }

    /// Feed one poll observation; returns `true` if the run should be
    /// restarted. `progress` is the summed counter value (`None` when the
    /// fetch failed or the body had no recognisable counters — treated as
    /// "cannot detect", never as a stall). `inflight` is the daemon's own
    /// in-flight request count for the service.
    ///
    /// A counter regression (child restarted underneath us, counters reset to
    /// zero) reseeds the baseline instead of firing. An idle service
    /// (`inflight == 0`) is never considered stalled: flat counters with no
    /// requests is simply quiescence.
    pub fn observe(&mut self, progress: Option<u64>, inflight: u64, timeout: Duration) -> bool {
        let now = Instant::now();
        match (progress, self.last_value) {
            (Some(v), Some(prev)) if v == prev => {}
            (Some(v), _) => {
                // First observation, an advance, or a regression: (re)seed.
                self.last_value = Some(v);
                self.last_advance = now;
            }
            (None, _) => {
                // No signal this tick. Don't advance the baseline — a wedge
                // that also breaks /metrics should still eventually fire off
                // prior flat observations — but don't fire on fetch failure
                // alone either (last_value stays whatever it was).
            }
        }
        if inflight == 0 {
            self.last_advance = now;
            return false;
        }
        progress.is_some() && now.duration_since(self.last_advance) >= timeout
    }
}

/// Accepted spellings of the prompt-processing counter, oldest first.
/// llama.cpp has renamed its Prometheus counters across versions; at most one
/// spelling per family is counted (first match wins) so a build that ever
/// emitted both could not double-count progress.
const PROMPT_COUNTERS: [&str; 2] = [
    "llamacpp:prompt_tokens_total",
    "llamacpp:n_prompt_tokens_processed_total",
];

/// Accepted spellings of the token-generation counter, oldest first.
const PREDICTED_COUNTERS: [&str; 2] = [
    "llamacpp:tokens_predicted_total",
    "llamacpp:n_tokens_predicted_total",
];

/// Extract and sum the llama.cpp progress counters (prompt + predicted
/// tokens) from a Prometheus text-exposition body. Tolerant of either family
/// being absent (llama.cpp versions differ in what they expose); `None` only
/// when neither is present, which distinguishes "not a llama.cpp `/metrics`"
/// from a zero counter.
pub fn parse_progress_counters(body: &str) -> Option<u64> {
    let prompt = find_counter(body, &PROMPT_COUNTERS);
    let predicted = find_counter(body, &PREDICTED_COUNTERS);
    match (prompt, predicted) {
        (None, None) => None,
        (a, b) => Some(a.unwrap_or(0) + b.unwrap_or(0)),
    }
}

/// The first counter from `names` present in `body`, if any.
fn find_counter(body: &str, names: &[&str]) -> Option<u64> {
    for name in names {
        for line in body.lines() {
            let line = line.trim();
            if line.starts_with('#') {
                continue;
            }
            // Guard against a shared prefix matching a longer metric name:
            // the name must be followed by whitespace, then the value
            // (llama.cpp emits no labels on these counters).
            if let Some(rest) = line.strip_prefix(name)
                && rest.starts_with(|c: char| c.is_ascii_whitespace())
                && let Some(Ok(v)) = rest.split_whitespace().next().map(str::parse::<f64>)
            {
                return Some(v as u64);
            }
        }
    }
    None
}

/// Fetch the child's `/metrics` and parse the progress counters. Any failure
/// — connection refused, timeout, non-2xx, unparseable body — collapses to
/// `None`; the caller treats that as "cannot detect", not as a stall.
pub async fn fetch_progress(client: &reqwest::Client, url: &str) -> Option<u64> {
    let resp = client.get(url).send().await.ok()?;
    if !resp.status().is_success() {
        return None;
    }
    let body = resp.text().await.ok()?;
    parse_progress_counters(&body)
}

#[cfg(test)]
mod tests {
    use super::*;

    const TIMEOUT: Duration = Duration::from_secs(300);

    fn body(prompt: u64, predicted: u64) -> String {
        format!(
            "# HELP llamacpp:prompt_tokens_total Number of prompt tokens processed.\n\
             # TYPE llamacpp:prompt_tokens_total counter\n\
             llamacpp:prompt_tokens_total {prompt}\n\
             # HELP llamacpp:tokens_predicted_total Number of generation tokens processed.\n\
             # TYPE llamacpp:tokens_predicted_total counter\n\
             llamacpp:tokens_predicted_total {predicted}\n\
             llamacpp:n_busy_slots_per_decode 1\n"
        )
    }

    #[test]
    fn parses_both_counters() {
        assert_eq!(parse_progress_counters(&body(11050, 465)), Some(11515));
    }

    #[test]
    fn parses_scientific_notation() {
        // Prometheus clients may emit large counters in float notation.
        assert_eq!(
            parse_progress_counters("llamacpp:tokens_predicted_total 1.2e3\n"),
            Some(1200)
        );
    }

    #[test]
    fn tolerates_missing_one_counter() {
        assert_eq!(
            parse_progress_counters("llamacpp:tokens_predicted_total 42\n"),
            Some(42)
        );
        assert_eq!(
            parse_progress_counters("llamacpp:prompt_tokens_total 7\n"),
            Some(7)
        );
    }

    #[test]
    fn missing_both_counters_is_none() {
        assert_eq!(parse_progress_counters("some_other_metric 1\n"), None);
        assert_eq!(parse_progress_counters(""), None);
        assert_eq!(
            parse_progress_counters("# HELP llamacpp:prompt_tokens_total x\n"),
            None
        );
    }

    #[test]
    fn parses_renamed_counters() {
        // Newer llama.cpp builds ship `n_`-prefixed counter names.
        assert_eq!(
            parse_progress_counters(
                "llamacpp:n_prompt_tokens_processed_total 100\n\
                 llamacpp:n_tokens_predicted_total 50\n"
            ),
            Some(150)
        );
    }

    #[test]
    fn both_spellings_do_not_double_count() {
        // If a build ever emitted old and new names, only one per family
        // counts.
        assert_eq!(
            parse_progress_counters(
                "llamacpp:tokens_predicted_total 50\n\
                 llamacpp:n_tokens_predicted_total 50\n"
            ),
            Some(50)
        );
    }

    #[test]
    fn prefix_only_match_is_rejected() {
        // A hypothetical longer metric sharing the prefix must not count.
        assert_eq!(
            parse_progress_counters("llamacpp:prompt_tokens_total_bytes 999\n"),
            None
        );
    }

    #[tokio::test(start_paused = true)]
    async fn flat_counter_past_timeout_fires() {
        let mut st = GenStallState::new(Instant::now());
        assert!(!st.observe(Some(100), 1, TIMEOUT));
        tokio::time::advance(TIMEOUT).await;
        assert!(st.observe(Some(100), 1, TIMEOUT));
    }

    #[tokio::test(start_paused = true)]
    async fn advancing_counter_never_fires() {
        let mut st = GenStallState::new(Instant::now());
        let mut v = 100;
        for _ in 0..10 {
            assert!(!st.observe(Some(v), 1, TIMEOUT));
            tokio::time::advance(TIMEOUT / 2).await;
            v += 50;
        }
    }

    #[tokio::test(start_paused = true)]
    async fn idle_service_never_fires() {
        let mut st = GenStallState::new(Instant::now());
        assert!(!st.observe(Some(100), 1, TIMEOUT));
        tokio::time::advance(TIMEOUT * 2).await;
        // Counter is flat but nothing is in flight: quiescent, not wedged.
        assert!(!st.observe(Some(100), 0, TIMEOUT));
        // And the idle tick reset the baseline: load arriving now starts a
        // fresh window rather than firing instantly.
        assert!(!st.observe(Some(100), 1, TIMEOUT));
    }

    #[tokio::test(start_paused = true)]
    async fn counter_regression_reseeds() {
        let mut st = GenStallState::new(Instant::now());
        assert!(!st.observe(Some(5000), 1, TIMEOUT));
        tokio::time::advance(TIMEOUT).await;
        // The child restarted underneath us; counters reset near zero.
        assert!(!st.observe(Some(3), 1, TIMEOUT));
        tokio::time::advance(TIMEOUT / 2).await;
        assert!(!st.observe(Some(3), 1, TIMEOUT));
        tokio::time::advance(TIMEOUT / 2).await;
        assert!(st.observe(Some(3), 1, TIMEOUT));
    }

    #[tokio::test(start_paused = true)]
    async fn fetch_failure_never_fires() {
        let mut st = GenStallState::new(Instant::now());
        assert!(!st.observe(Some(100), 1, TIMEOUT));
        tokio::time::advance(TIMEOUT * 2).await;
        // /metrics unreachable: cannot detect, so no verdict either way.
        assert!(!st.observe(None, 1, TIMEOUT));
    }
}
