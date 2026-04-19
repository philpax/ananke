//! `override_tensor` rule application (spec §8.2.4).
//!
//! The user declares `override_tensor = ["<regex>=<device>", ...]`; llama.cpp
//! takes the same rules via `-ot`. We must mirror the placement accounting so
//! the allocator and placement walker see the correct per-device budgets:
//! any tensor matching a rule is attributed to the declared device rather
//! than following layer placement.
//!
//! Rules apply in array order; first match wins. Matched tensor bytes are
//! subtracted from per-layer / non-layer accounting (so the layer walker
//! packs only the residual) and accumulated into `Estimate.override_tensor_bytes`.

use std::collections::BTreeMap;

use regex::Regex;
use tracing::warn;

use super::llama::layer_index;
use super::types::Estimate;
use crate::config::DeviceSlot;
use crate::gguf::GgufSummary;

#[derive(Debug)]
pub struct OverrideRule {
    pub regex: Regex,
    pub target: DeviceSlot,
}

#[derive(Debug)]
pub struct ParseError(pub String);

impl std::fmt::Display for ParseError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.0)
    }
}

impl std::error::Error for ParseError {}

/// Parse a user-supplied `override_tensor` array into compiled rules.
pub fn parse_rules(rules: &[String]) -> Result<Vec<OverrideRule>, ParseError> {
    let mut out = Vec::with_capacity(rules.len());
    for rule in rules {
        let (pattern, device_str) = rule
            .rsplit_once('=')
            .ok_or_else(|| ParseError(format!("missing '=' in rule `{rule}`")))?;
        let regex =
            Regex::new(pattern).map_err(|e| ParseError(format!("regex `{pattern}`: {e}")))?;
        let target = parse_device(device_str.trim())
            .ok_or_else(|| ParseError(format!("unknown device `{device_str}` in rule `{rule}`")))?;
        out.push(OverrideRule { regex, target });
    }
    Ok(out)
}

fn parse_device(s: &str) -> Option<DeviceSlot> {
    let upper = s.to_ascii_uppercase();
    if upper == "CPU" {
        return Some(DeviceSlot::Cpu);
    }
    if let Some(tail) = upper.strip_prefix("GPU")
        && let Ok(n) = tail.parse::<u32>()
    {
        return Some(DeviceSlot::Gpu(n));
    }
    None
}

/// Apply `rules` to `summary`, moving matched tensor bytes out of
/// `estimate.per_layer_bytes` / `non_layer` into `override_tensor_bytes`.
///
/// The placement walker will then use the reduced per-layer totals and the
/// pre-seeded per-device map (via `override_tensor_bytes`) to produce a
/// consistent allocation.
pub fn apply(estimate: &mut Estimate, summary: &GgufSummary, rules: &[OverrideRule]) {
    if rules.is_empty() {
        return;
    }

    let mut override_bytes: BTreeMap<DeviceSlot, u64> = BTreeMap::new();

    for tensor in summary.tensors.values() {
        let Some(rule) = rules
            .iter()
            .find(|r| r.regex.is_match(tensor.name.as_str()))
        else {
            continue;
        };
        *override_bytes.entry(rule.target.clone()).or_default() += tensor.byte_size;

        if let Some(idx) = layer_index(tensor.name.as_str()) {
            if let Some(per_layer) = estimate.per_layer_bytes.as_mut()
                && (idx as usize) < per_layer.len()
            {
                per_layer[idx as usize] = per_layer[idx as usize].saturating_sub(tensor.byte_size);
            }
            if let Some(existing) = estimate.expert_layer_cpu_bytes.get_mut(&idx) {
                *existing = existing.saturating_sub(tensor.byte_size);
            }
        } else {
            match tensor.name.as_str() {
                "output.weight" => {
                    estimate.non_layer.output_head_bytes = estimate
                        .non_layer
                        .output_head_bytes
                        .saturating_sub(tensor.byte_size)
                }
                "token_embd.weight" => {
                    estimate.non_layer.token_embd_bytes = estimate
                        .non_layer
                        .token_embd_bytes
                        .saturating_sub(tensor.byte_size)
                }
                _ => {
                    estimate.non_layer.other_bytes = estimate
                        .non_layer
                        .other_bytes
                        .saturating_sub(tensor.byte_size)
                }
            }
        }
    }

    estimate.override_tensor_bytes = override_bytes;

    // Recompute the weights total to reflect the reduced per-layer/non-layer sums.
    let per_layer_sum = estimate
        .per_layer_bytes
        .as_ref()
        .map(|p| p.iter().sum::<u64>())
        .unwrap_or(0);
    estimate.weights_bytes = per_layer_sum
        + estimate.non_layer.output_head_bytes
        + estimate.non_layer.token_embd_bytes
        + estimate.non_layer.other_bytes;
}

/// Convenience: parse and apply in one call; errors are logged (not returned),
/// consistent with how the mmproj integration handles soft failures.
pub fn parse_and_apply(estimate: &mut Estimate, summary: &GgufSummary, rules: &[String]) {
    match parse_rules(rules) {
        Ok(parsed) => apply(estimate, summary, &parsed),
        Err(e) => warn!(error = %e, "override_tensor parse failed; running without overrides"),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::DeviceSlot;
    use crate::estimator::types::{Estimate, NonLayer};
    use crate::gguf::types::{GgufSummary, GgufTensor, GgufType};
    use smol_str::SmolStr;

    fn tensor(name: &str, bytes: u64) -> GgufTensor {
        GgufTensor {
            name: SmolStr::new(name),
            dtype: GgufType::F16,
            shape: vec![bytes / 2],
            byte_size: bytes,
            shard_idx: 0,
            offset: 0,
        }
    }

    fn base_estimate(per_layer: Vec<u64>) -> Estimate {
        let weights = per_layer.iter().sum::<u64>();
        Estimate {
            weights_bytes: weights,
            kv_per_token: 0,
            compute_buffer_mb: 400,
            per_layer_bytes: Some(per_layer),
            attention_layers: None,
            non_layer: NonLayer::default(),
            override_tensor_bytes: BTreeMap::new(),
            expert_layers: Vec::new(),
            expert_layer_cpu_bytes: BTreeMap::new(),
            context: 4096,
            architecture: SmolStr::new("qwen3moe"),
        }
    }

    fn summary_with(tensors: Vec<GgufTensor>) -> GgufSummary {
        let mut map = std::collections::BTreeMap::new();
        let mut total = 0;
        for t in tensors {
            total += t.byte_size;
            map.insert(t.name.clone(), t);
        }
        GgufSummary {
            path: "/fake".into(),
            total_tensor_bytes: total,
            tensors: map,
            metadata: Default::default(),
            block_count: Some(2),
            architecture: SmolStr::new("qwen3moe"),
            shards: vec!["/fake".into()],
        }
    }

    #[test]
    fn parses_single_rule() {
        let rules = parse_rules(&[".ffn_(up|down)_exps.=CPU".into()]).unwrap();
        assert_eq!(rules.len(), 1);
        assert_eq!(rules[0].target, DeviceSlot::Cpu);
    }

    #[test]
    fn rejects_unknown_device() {
        let err = parse_rules(&["foo=BLAH".into()]).unwrap_err();
        assert!(format!("{err}").contains("BLAH"));
    }

    #[test]
    fn rejects_bad_regex() {
        let err = parse_rules(&["(unclosed=CPU".into()]).unwrap_err();
        assert!(format!("{err}").contains("regex"));
    }

    #[test]
    fn parses_gpu_index() {
        let rules = parse_rules(&["output=GPU0".into(), "foo=GPU1".into()]).unwrap();
        assert_eq!(rules[0].target, DeviceSlot::Gpu(0));
        assert_eq!(rules[1].target, DeviceSlot::Gpu(1));
    }

    #[test]
    fn moves_expert_tensors_to_cpu() {
        // 2 layers, each with an attn tensor (1 MiB) and an expert tensor (10 MiB).
        let tensors = vec![
            tensor("blk.0.attn_q.weight", 1024 * 1024),
            tensor("blk.0.ffn_up_exps.weight", 10 * 1024 * 1024),
            tensor("blk.1.attn_q.weight", 1024 * 1024),
            tensor("blk.1.ffn_down_exps.weight", 10 * 1024 * 1024),
        ];
        let summary = summary_with(tensors);
        // Per-layer starts at 11 MiB each (sum of both tensors on the layer).
        let mut est = base_estimate(vec![11 * 1024 * 1024, 11 * 1024 * 1024]);
        let rules = parse_rules(&[".ffn_(up|down)_exps.=CPU".into()]).unwrap();

        apply(&mut est, &summary, &rules);

        // Each layer drops to 1 MiB (just the attn tensor).
        let per_layer = est.per_layer_bytes.unwrap();
        assert_eq!(per_layer, vec![1024 * 1024, 1024 * 1024]);
        // CPU gets 20 MiB total.
        assert_eq!(
            est.override_tensor_bytes.get(&DeviceSlot::Cpu).copied(),
            Some(20 * 1024 * 1024)
        );
        // weights_bytes reflects the reduced per-layer + non-layer.
        assert_eq!(est.weights_bytes, 2 * 1024 * 1024);
    }

    #[test]
    fn first_match_wins() {
        let tensors = vec![tensor("blk.0.ffn_up_exps.weight", 10 * 1024 * 1024)];
        let summary = summary_with(tensors);
        let mut est = base_estimate(vec![10 * 1024 * 1024, 0]);
        // GPU1 comes first; CPU second. Expert should land on GPU1.
        let rules =
            parse_rules(&["ffn_up=GPU1".into(), ".ffn_(up|down)_exps.=CPU".into()]).unwrap();

        apply(&mut est, &summary, &rules);

        assert_eq!(
            est.override_tensor_bytes.get(&DeviceSlot::Gpu(1)).copied(),
            Some(10 * 1024 * 1024)
        );
        assert!(!est.override_tensor_bytes.contains_key(&DeviceSlot::Cpu));
    }
}
