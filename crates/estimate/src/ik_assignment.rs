//! Conservative inputs for IK's runtime layer assignment, separate from VRAM estimates.

use ananke_gguf::{Architecture, GgufSummary, GgufType, GgufValue, keys};

use crate::{EstimatorInputs, IkAssignment, Layout, Speculation};

pub(crate) fn assignment(
    summary: &GgufSummary,
    inputs: &EstimatorInputs<'_>,
    layout: &Layout,
) -> Option<IkAssignment> {
    let arch = &summary.architecture;
    if !inputs.fork.is_ik()
        || !inputs.ik_assignment_options_supported
        || !matches!(
            arch,
            Architecture::Mixtral | Architecture::Qwen3Moe | Architecture::Glm4Moe
        )
        || inputs.mmproj.is_some()
        || !inputs.override_tensor.is_empty()
        || !matches!(inputs.speculation, Speculation::None)
        || summary
            .meta_u32(&keys::nextn_predict_layers(arch))
            .unwrap_or(0)
            != 0
        || inputs.context == 0
    {
        return None;
    }
    // These metadata extensions change the runtime cache layout. The whitelist
    // covers ordinary GQA only, even if a future model reuses an architecture.
    for key in [
        keys::full_attention_interval(arch),
        keys::attention_key_length_swa(arch),
        keys::attention_value_length_swa(arch),
        keys::attention_shared_kv_layers(arch),
        keys::attention_compress_ratios(arch),
        keys::ssm_state_size(arch),
    ] {
        if summary.metadata.contains_key(&key) {
            return None;
        }
    }
    let n_layers = summary.block_count? as usize;
    if n_layers == 0 || layout.per_layer_bytes.as_ref()?.len() != n_layers {
        return None;
    }
    let embedding = u64::from(summary.meta_u32(&keys::embedding_length(arch))?);
    let heads = u64::from(summary.meta_u32(&keys::attention_head_count(arch))?);
    let experts_used = u64::from(summary.meta_u32(&keys::expert_used_count(arch))?);
    if embedding == 0 || heads == 0 || experts_used == 0 || !embedding.is_multiple_of(heads) {
        return None;
    }
    let key_length = u64::from(
        summary
            .meta_u32(&keys::attention_key_length(arch))
            .unwrap_or((embedding / heads) as u32),
    );
    let value_length = u64::from(
        summary
            .meta_u32(&keys::attention_value_length(arch))
            .unwrap_or((embedding / heads) as u32),
    );
    let kv_heads = summary
        .metadata
        .get(&keys::attention_head_count_kv(arch))
        .or_else(|| summary.metadata.get(&keys::attention_head_count(arch)))?
        .as_u32_array()?;
    if kv_heads.len() != 1 && kv_heads.len() != n_layers {
        return None;
    }
    // The loader clamps ubatch to the logical batch and context. Omitting the
    // logical-batch clamp only enlarges the scratch and logits upper bounds.
    let ubatch = u64::from(inputs.ubatch.unwrap_or(512).min(inputs.context));
    if ubatch == 0 || key_length == 0 || value_length == 0 {
        return None;
    }
    let cache_rows = u64::from(inputs.context).div_ceil(256).checked_mul(256)?;
    let mut cache_upper = Vec::with_capacity(n_layers);
    for i in 0..n_layers {
        let n_head_kv = u64::from(kv_heads[if kv_heads.len() == 1 { 0 } else { i }]);
        if n_head_kv == 0 {
            return None;
        }
        let k = row_bytes(inputs.cache_type_k.unwrap_or(GgufType::F16), key_length)?
            .checked_mul(n_head_kv)?;
        let v = row_bytes(
            inputs.cache_type_v.unwrap_or(GgufType::F16),
            value_length.checked_mul(n_head_kv)?,
        )?;
        cache_upper.push(k.checked_add(v)?.checked_mul(cache_rows)?);
    }
    let mut layer_weights = vec![0u64; n_layers];
    let mut scratch_upper = embedding.checked_mul(ubatch)?.checked_mul(8)?;
    let mut output_weights = 0u64;
    let mut output_misc = 0u64;
    let mut embedding_weights = 0u64;
    for (name, tensor) in &summary.tensors {
        if matches!(tensor.dtype, GgufType::Unknown(_)) {
            return None;
        }
        match name.as_str() {
            "token_embd.weight" => embedding_weights = tensor.byte_size,
            "output.weight" => output_weights = output_weights.checked_add(tensor.byte_size)?,
            "token_embd_norm.weight" | "token_embd_norm.bias" => {
                output_misc = output_misc.checked_add(tensor.byte_size)?
            }
            "output_norm.weight" | "rope_freqs.weight" => {}
            _ if name.starts_with("output_hc_") || name.starts_with("hc_head_") => {
                output_weights = output_weights.checked_add(tensor.byte_size)?;
            }
            _ if name.starts_with("blk.") => {
                let (index, suffix) = name.strip_prefix("blk.")?.split_once('.')?;
                let index: usize = index.parse().ok()?;
                if suffix.contains("_exps.") && !suffix.ends_with(".weight") {
                    return None;
                }
                // MLA creates extra tensors and can repair GGUF dimensions. A
                // generic layer sum is not the runtime weight cost in that case.
                if suffix.contains("attn_k_b")
                    || suffix.contains("attn_v_b")
                    || suffix.contains("attn_kv_b")
                {
                    return None;
                }
                let weight = layer_weights.get_mut(index)?;
                *weight = weight.checked_add(tensor.byte_size)?;
                if tensor.shape.is_empty() || tensor.shape.contains(&0) {
                    return None;
                }
                let width = tensor.shape.iter().take(2).copied().max()?;
                // Summing all layer tensors dominates the runtime's maximum of
                // the expert, shared-expert, and dense FFN activation sums.
                scratch_upper = scratch_upper.checked_add(
                    width
                        .checked_mul(ubatch)?
                        .checked_mul(experts_used)?
                        .checked_mul(4)?,
                )?;
            }
            _ => {}
        }
    }
    if layer_weights.contains(&0) {
        return None;
    }
    if output_weights == 0 {
        output_weights = embedding_weights;
    }
    if output_weights == 0 {
        return None;
    }
    let vocab = summary
        .meta_u64(&keys::scoped(arch, "vocab_size"))
        .or_else(|| match summary.metadata.get("tokenizer.ggml.tokens") {
            Some(GgufValue::Array(tokens)) => Some(tokens.len() as u64),
            _ => None,
        })
        .or_else(|| {
            summary
                .tensors
                .get("output.weight")
                .or_else(|| summary.tensors.get("token_embd.weight"))?
                .shape
                .get(1)
                .copied()
        })?;
    if vocab == 0 {
        return None;
    }
    Some(IkAssignment {
        layer_weights,
        cache_lower: vec![0; n_layers],
        cache_upper,
        output_weights: output_weights.checked_add(output_misc)?,
        logits_bytes: vocab
            .checked_mul(u64::from(
                inputs
                    .worst_graph_tokens
                    .filter(|&n| n > 0)
                    .unwrap_or(ubatch as u32),
            ))?
            .checked_mul(4)?,
        scratch_upper,
    })
}

fn row_bytes(dtype: GgufType, width: u64) -> Option<u64> {
    let (block, bytes) = match dtype {
        GgufType::F32 => (1, 4),
        GgufType::F16 | GgufType::BF16 => (1, 2),
        GgufType::Q8_0 => (32, 34),
        GgufType::Q5_1 => (32, 24),
        GgufType::Q5_0 => (32, 22),
        GgufType::Q4_1 => (32, 20),
        GgufType::Q4_0 | GgufType::IQ4_NL => (32, 18),
        GgufType::Q6_0 => (32, 26),
        GgufType::Q8_KV => (32, 32),
        _ => return None,
    };
    if !width.is_multiple_of(block) {
        return None;
    }
    (width / block).checked_mul(bytes)
}

#[cfg(test)]
mod tests;
