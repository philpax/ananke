//! Small helpers shared across the validation passes: unit conversion,
//! duration parsing, the vocabulary-table lookups, and the config-error
//! constructor.

pub use crate::{
    placement::{flag_variant, variant_flag},
    units::gib_to_mib,
};

/// Parse a duration string (`"10m"`, `"30s"`, `"500ms"`, `"2h"`) into
/// milliseconds. Returns a rendered error message on failure.
pub fn parse_duration_ms(s: &str) -> Result<u64, String> {
    // Accepts "10m", "30s", "500ms", "2h". Returns milliseconds.
    let s = s.trim();
    if let Some(rest) = s.strip_suffix("ms") {
        return rest
            .parse::<u64>()
            .map_err(|_| format!("invalid number in `{s}`"));
    }
    if let Some(rest) = s.strip_suffix('s') {
        return rest
            .parse::<u64>()
            .map(|n| n * 1000)
            .map_err(|_| format!("invalid number in `{s}`"));
    }
    if let Some(rest) = s.strip_suffix('m') {
        return rest
            .parse::<u64>()
            .map(|n| n * 60_000)
            .map_err(|_| format!("invalid number in `{s}`"));
    }
    if let Some(rest) = s.strip_suffix('h') {
        return rest
            .parse::<u64>()
            .map(|n| n * 3_600_000)
            .map_err(|_| format!("invalid number in `{s}`"));
    }
    Err(format!("unrecognised duration: {s}"))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn duration_parser() {
        assert_eq!(parse_duration_ms("500ms").unwrap(), 500);
        assert_eq!(parse_duration_ms("30s").unwrap(), 30_000);
        assert_eq!(parse_duration_ms("10m").unwrap(), 600_000);
        assert_eq!(parse_duration_ms("2h").unwrap(), 7_200_000);
        assert!(parse_duration_ms("bogus").is_err());
    }
}
