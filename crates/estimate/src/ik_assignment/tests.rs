use std::{collections::BTreeMap, path::Path};

use ananke_gguf::{GgufTensor, GgufValue};

use super::*;
use crate::{Fork, dispatch};

fn summary(arch: Architecture) -> GgufSummary {
    let mut summary = GgufSummary {
        path: "/fake".into(),
        total_tensor_bytes: 0,
        tensors: BTreeMap::new(),
        metadata: BTreeMap::new(),
        block_count: Some(2),
        architecture: arch,
        shards: vec![],
    };
    for (key, value) in [
        (keys::embedding_length(&summary.architecture), 256),
        (keys::attention_head_count(&summary.architecture), 4),
        (keys::attention_head_count_kv(&summary.architecture), 2),
        (keys::expert_used_count(&summary.architecture), 2),
    ] {
        summary.metadata.insert(key, GgufValue::U32(value));
    }
    for (name, shape, bytes) in [
        ("blk.0.attn_norm.weight", vec![256], 512),
        ("blk.0.ffn_up_exps.weight", vec![256, 512, 8], 2_097_152),
        ("blk.1.attn_norm.weight", vec![256], 512),
        ("blk.1.ffn_up_exps.weight", vec![256, 512, 8], 2_097_152),
        ("token_embd.weight", vec![256, 1000], 512_000),
        ("output.weight", vec![256, 1000], 512_000),
        ("output_norm.weight", vec![256], 512),
    ] {
        add_tensor(&mut summary, name, shape, bytes);
    }
    summary
}

fn add_tensor(summary: &mut GgufSummary, name: &str, shape: Vec<u64>, byte_size: u64) {
    summary.tensors.insert(
        name.into(),
        GgufTensor {
            name: name.into(),
            dtype: GgufType::F16,
            shape,
            byte_size,
            shard_idx: 0,
            offset: 0,
        },
    );
    summary.total_tensor_bytes += byte_size;
}

fn inputs() -> EstimatorInputs<'static> {
    EstimatorInputs {
        context: 4096,
        ubatch: Some(512),
        fork: Fork::Ik { dsa: false },
        ..EstimatorInputs::empty(Path::new("/fake"))
    }
}

#[test]
fn dispatch_populates_runtime_accounting_for_the_whitelist() {
    for arch in [
        Architecture::Mixtral,
        Architecture::Qwen3Moe,
        Architecture::Glm4Moe,
    ] {
        let s = summary(arch);
        let e = dispatch(&s, &inputs()).unwrap();
        let assignment = e.layout.ik_assignment.unwrap();
        assert_eq!(assignment.layer_weights, vec![2_097_664; 2]);
        assert_eq!(
            assignment.layer_weights.len(),
            e.layout.per_layer_bytes.unwrap().len()
        );
        assert_eq!(assignment.cache_lower, vec![0; 2]);
        assert_eq!(assignment.cache_upper, vec![2 * (64 + 64) * 2 * 4096; 2]);
        assert_eq!(assignment.output_weights, 512_000);
        assert_eq!(assignment.logits_bytes, 1000 * 512 * 4);
        let inp = 256 * 512 * 8;
        let tensor_activation_sum = 2 * (256 + 512) * 512 * 2 * 4;
        assert_eq!(assignment.scratch_upper, inp + tensor_activation_sum);
    }
}

#[test]
fn output_norm_is_excluded_and_tied_output_and_misc_follow_the_runtime() {
    let mut s = summary(Architecture::Qwen3Moe);
    s.tensors.remove("output.weight");
    add_tensor(&mut s, "token_embd_norm.bias", vec![256], 512);
    let e = dispatch(&s, &inputs()).unwrap();
    assert_eq!(e.layout.ik_assignment.unwrap().output_weights, 512_512);
}

#[test]
fn raw_logits_use_vocab_metadata_and_context_clamped_ubatch() {
    let mut s = summary(Architecture::Qwen3Moe);
    s.metadata.insert(
        keys::scoped(&s.architecture, "vocab_size"),
        GgufValue::U32(1234),
    );
    let i = EstimatorInputs {
        context: 32,
        ..inputs()
    };
    let assignment = dispatch(&s, &i).unwrap().layout.ik_assignment.unwrap();
    assert_eq!(assignment.logits_bytes, 1234 * 32 * 4);
}

#[test]
fn explicit_worst_graph_tokens_can_exceed_ubatch() {
    let s = summary(Architecture::Qwen3Moe);
    let i = EstimatorInputs {
        worst_graph_tokens: Some(2048),
        ..inputs()
    };
    assert_eq!(
        dispatch(&s, &i)
            .unwrap()
            .layout
            .ik_assignment
            .unwrap()
            .logits_bytes,
        1000 * 2048 * 4
    );
}

#[test]
fn cache_upper_includes_context_padding() {
    let s = summary(Architecture::Qwen3Moe);
    let i = EstimatorInputs {
        context: 1025,
        ..inputs()
    };
    let a = dispatch(&s, &i).unwrap().layout.ik_assignment.unwrap();
    assert_eq!(a.cache_upper, vec![2 * (64 + 64) * 2 * 1280; 2]);
}

#[test]
fn quantized_cache_uses_ggml_rows_and_per_layer_heads() {
    let mut s = summary(Architecture::Qwen3Moe);
    s.metadata.insert(
        keys::attention_head_count_kv(&s.architecture),
        GgufValue::Array(vec![GgufValue::U32(2), GgufValue::U32(4)]),
    );
    let i = EstimatorInputs {
        cache_type_k: Some(GgufType::Q8_0),
        cache_type_v: Some(GgufType::Q4_0),
        ..inputs()
    };
    let a = dispatch(&s, &i).unwrap().layout.ik_assignment.unwrap();
    assert_eq!(
        a.cache_upper,
        vec![(68 * 2 + 72) * 4096, (68 * 4 + 144) * 4096]
    );
}

#[test]
fn calibrated_compute_does_not_change_assignment_scratch() {
    let s = summary(Architecture::Qwen3Moe);
    let base = dispatch(&s, &inputs()).unwrap();
    let overridden = dispatch(
        &s,
        &EstimatorInputs {
            compute_buffer_mb: Some(9999),
            ..inputs()
        },
    )
    .unwrap();
    assert_eq!(base.layout.ik_assignment, overridden.layout.ik_assignment);
}

#[test]
fn unsupported_forks_architectures_and_mtp_have_no_metadata() {
    for arch in [
        Architecture::DeepSeek2,
        Architecture::GlmDsa,
        Architecture::DeepSeek4,
        Architecture::Qwen35Moe,
        Architecture::GptOss,
        Architecture::Llama4,
    ] {
        assert!(
            dispatch(&summary(arch), &inputs())
                .unwrap()
                .layout
                .ik_assignment
                .is_none()
        );
    }
    let mut s = summary(Architecture::Qwen3Moe);
    assert!(
        dispatch(
            &s,
            &EstimatorInputs {
                fork: Fork::Mainline,
                ..inputs()
            }
        )
        .unwrap()
        .layout
        .ik_assignment
        .is_none()
    );
    s.metadata.insert(
        keys::nextn_predict_layers(&s.architecture),
        GgufValue::U32(1),
    );
    assert!(
        dispatch(&s, &inputs())
            .unwrap()
            .layout
            .ik_assignment
            .is_none()
    );
}

#[test]
fn missing_dimensions_and_unsupported_cache_are_not_guessed() {
    let mut s = summary(Architecture::Qwen3Moe);
    let unsupported = EstimatorInputs {
        cache_type_k: Some(GgufType::Unknown(999)),
        ..inputs()
    };
    assert!(
        dispatch(&s, &unsupported)
            .unwrap()
            .layout
            .ik_assignment
            .is_none()
    );
    s.metadata.remove(&keys::expert_used_count(&s.architecture));
    assert!(
        dispatch(&s, &inputs())
            .unwrap()
            .layout
            .ik_assignment
            .is_none()
    );
}

#[test]
fn mla_tensors_and_unrepresentable_scratch_fail_closed() {
    let mut s = summary(Architecture::Qwen3Moe);
    add_tensor(&mut s, "blk.0.attn_kv_b.weight", vec![64, 64], 8192);
    assert!(
        dispatch(&s, &inputs())
            .unwrap()
            .layout
            .ik_assignment
            .is_none()
    );
    s.tensors.remove("blk.0.attn_kv_b.weight");
    add_tensor(
        &mut s,
        "blk.0.ffn_down_exps.weight",
        vec![64, u64::MAX],
        8192,
    );
    assert!(
        dispatch(&s, &inputs())
            .unwrap()
            .layout
            .ik_assignment
            .is_none()
    );
}

#[test]
fn auxiliary_expert_tensors_fail_closed() {
    let mut s = summary(Architecture::Qwen3Moe);
    add_tensor(&mut s, "blk.0.ffn_up_exps.scale", vec![128], 256);
    assert!(
        dispatch(&s, &inputs())
            .unwrap()
            .layout
            .ik_assignment
            .is_none()
    );
}

#[test]
fn non_gqa_cache_metadata_fails_closed() {
    let mut s = summary(Architecture::Qwen3Moe);
    s.metadata.insert(
        keys::attention_key_length_swa(&s.architecture),
        GgufValue::U32(256),
    );
    assert!(
        dispatch(&s, &inputs())
            .unwrap()
            .layout
            .ik_assignment
            .is_none()
    );
}

#[test]
fn overrides_projection_and_speculation_fail_closed() {
    let s = summary(Architecture::Qwen3Moe);
    let overrides = ["ffn=CPU".into()];
    for i in [
        EstimatorInputs {
            ik_assignment_options_supported: false,
            ..inputs()
        },
        EstimatorInputs {
            override_tensor: &overrides,
            ..inputs()
        },
        EstimatorInputs {
            mmproj: Some(Path::new("/projection")),
            ..inputs()
        },
        EstimatorInputs {
            speculation: Speculation::EmbeddedMtp,
            ..inputs()
        },
    ] {
        assert!(dispatch(&s, &i).unwrap().layout.ik_assignment.is_none());
    }
}
