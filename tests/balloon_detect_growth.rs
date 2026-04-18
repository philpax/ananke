use std::collections::VecDeque;

use ananke::balloon::detect_growth;

#[test]
fn growing_window_with_floor_detected() {
    let window: VecDeque<u64> = vec![10, 12, 14, 16, 18, 20].into();
    assert!(detect_growth(&window, 0));
}

#[test]
fn floor_blocks_detection() {
    // Gently growing, but the floor is set absurdly high so the
    // slope-projected next sample stays below it.
    let window: VecDeque<u64> = vec![10, 11, 12, 13, 14, 15].into();
    assert!(!detect_growth(&window, 1_000_000));
}
