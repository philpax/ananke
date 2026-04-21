//! KV cache bytes-per-element table.

use crate::gguf::GgufType;

/// Approximate bytes per element for llama.cpp's accepted
/// `--cache-type-k` / `--cache-type-v` values.
pub fn kv_bytes_per_element(cache_type: &str) -> f64 {
    match cache_type {
        "f32" => 4.0,
        "f16" | "bf16" => 2.0,
        "q8_0" => 1.0625,            // 34 bytes / 32 elements
        "q5_1" => 0.75,              // 24/32
        "q5_0" => 0.6875,            // 22/32
        "q4_1" => 0.625,             // 20/32
        "q4_0" | "iq4_nl" => 0.5625, // 18/32
        _ => 2.0,                    // unknown → fall back to f16 equivalent
    }
}

/// Convenience: bytes per element for a declared GgufType, for KV.
pub fn kv_bytes_for_type(t: GgufType) -> f64 {
    match t {
        GgufType::F32 => 4.0,
        GgufType::F16 | GgufType::BF16 => 2.0,
        GgufType::Q8_0 => 1.0625,
        GgufType::Q5_1 => 0.75,
        GgufType::Q5_0 => 0.6875,
        GgufType::Q4_1 => 0.625,
        GgufType::Q4_0 | GgufType::IQ4_NL => 0.5625,
        _ => 2.0,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn q8_0_is_around_1_point_06() {
        assert!((kv_bytes_per_element("q8_0") - 1.0625).abs() < 1e-6);
    }

    #[test]
    fn unknown_falls_back_to_f16() {
        assert_eq!(kv_bytes_per_element("this-is-bogus"), 2.0);
    }
}
