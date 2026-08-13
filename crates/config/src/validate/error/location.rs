//! Source positions for diagnostics that carry one.

use std::ops::Range;

/// A source location in the original configuration text.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DiagnosticLocation {
    /// Zero-based byte offset of the beginning of the span.
    pub start: usize,
    /// Exclusive zero-based byte offset of the end of the span.
    pub end: usize,
    /// One-based line containing the beginning of the span.
    pub line: u32,
    /// One-based column containing the beginning of the span.
    pub column: u32,
}

impl DiagnosticLocation {
    /// Construct a location from a byte range and source text.
    pub fn from_range(source: &str, range: Range<usize>) -> Self {
        let (line, column) = byte_offset_to_line_column(source, range.start);
        Self {
            start: range.start,
            end: range.end,
            line,
            column,
        }
    }
}

/// Convert a byte offset into a one-based line and column.
///
/// Offsets in the middle of a UTF-8 code point are clamped to the beginning of
/// that code point. End-of-file is a valid position.
pub fn byte_offset_to_line_column(source: &str, offset: usize) -> (u32, u32) {
    let offset = offset.min(source.len());
    let offset = (0..=offset)
        .rev()
        .find(|candidate| source.is_char_boundary(*candidate))
        .unwrap_or(0);
    let prefix = &source[..offset];
    let line = prefix.bytes().filter(|b| *b == b'\n').count() as u32 + 1;
    let column = prefix
        .rsplit_once('\n')
        .map_or(prefix.chars().count(), |(_, rest)| rest.chars().count()) as u32
        + 1;
    (line, column)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn byte_offsets_handle_multiline_utf8_and_eof() {
        let source = "α = 1\nname = \"é\"";
        assert_eq!(byte_offset_to_line_column(source, 0), (1, 1));
        assert_eq!(byte_offset_to_line_column(source, 7), (2, 1));
        assert_eq!(byte_offset_to_line_column(source, source.len()), (2, 11));
    }
}
