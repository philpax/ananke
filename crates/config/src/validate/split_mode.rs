//! The `--split-mode` vocabulary: how a multi-GPU llama.cpp service divides
//! a model across the GPUs it spans.

/// How a multi-GPU llama.cpp service divides the model across the GPUs it
/// spans. Orthogonal to [`PlacementPolicy`], which decides CPU-vs-GPU and
/// whether CPU spill is allowed; this decides the *inter-GPU* strategy and
/// maps straight onto llama.cpp's `--split-mode`.
///
/// - `Layer` (default): pipeline — each GPU holds whole layers and the
///   first-fit packer fills one GPU before spilling to the next. Minimal
///   inter-GPU traffic, but only one GPU computes at a time for a single
///   request.
/// - `Row` / `Tensor`: tensor parallelism — every layer is sharded across
///   all spanned GPUs, which compute in parallel and reduce per layer.
///   `tensor` is llama.cpp's newer, faster implementation; `row` is the
///   older one, kept for parity. Both require [`PlacementPolicy::GpuOnly`]
///   (no CPU spill) and a llama-cpp service.
pub use crate::placement::SplitMode;

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        fields,
        validate::{ValidationErrorCode, test_fixtures::parse_and_merge, validate},
    };

    #[test]
    fn parses_tensor_split_mode() {
        let cfg = parse_and_merge(
            r#"
[[service]]
name = "demo"
template = "llama-cpp"
model = "/m/x.gguf"
port = 11435
context = 4096
devices.placement = "gpu-only"
devices.split = "tensor"
lifecycle = "persistent"
"#,
        );
        let ec = validate(&cfg).unwrap();
        assert_eq!(ec.services[0].split_mode, SplitMode::Tensor);
    }

    #[test]
    fn defaults_split_mode_to_layer() {
        let cfg = parse_and_merge(
            r#"
[[service]]
name = "demo"
template = "llama-cpp"
model = "/m/x.gguf"
port = 11435
context = 4096
lifecycle = "persistent"
"#,
        );
        let ec = validate(&cfg).unwrap();
        assert_eq!(ec.services[0].split_mode, SplitMode::Layer);
    }

    #[test]
    fn rejects_unknown_split_mode() {
        let cfg = parse_and_merge(
            r#"
[[service]]
name = "demo"
template = "llama-cpp"
model = "/m/x.gguf"
port = 11435
devices.split = "diagonal"
lifecycle = "persistent"
"#,
        );
        let err = validate(&cfg).unwrap_err();
        let diag = &err.as_slice()[0];
        assert_eq!(diag.fields(), [fields::devices::SPLIT]);
        assert!(diag.to_string().contains("unknown devices.split"));
        assert!(diag.to_string().contains(&SplitMode::valid_values()));
    }

    #[test]
    fn rejects_tensor_split_with_cpu_spill() {
        let cfg = parse_and_merge(
            r#"
[[service]]
name = "demo"
template = "llama-cpp"
model = "/m/x.gguf"
port = 11435
context = 4096
devices.placement = "hybrid"
devices.split = "tensor"
lifecycle = "persistent"
"#,
        );
        let err = validate(&cfg).unwrap_err();
        let diag = &err.as_slice()[0];
        assert_eq!(diag.fields(), [fields::devices::SPLIT]);
        assert!(diag.to_string().contains("requires placement=gpu-only"));
    }

    #[test]
    fn rejects_tensor_split_on_command_service() {
        let cfg = parse_and_merge(
            r#"
[[service]]
name = "demo"
template = "command"
command = ["/bin/true"]
port = 11435
allocation.mode = "static"
allocation.reserve_gb = 4
devices.placement = "gpu-only"
devices.split = "row"
lifecycle = "persistent"
"#,
        );
        let err = validate(&cfg).unwrap_err();
        let diag = &err.as_slice()[0];
        assert_eq!(diag.fields(), [fields::devices::SPLIT]);
        assert!(
            diag.to_string()
                .contains("is only valid for llama-cpp services")
        );
    }

    #[test]
    fn parses_tensor_split_weights() {
        let cfg = parse_and_merge(
            r#"
[[service]]
name = "demo"
template = "llama-cpp"
model = "/m/x.gguf"
port = 11435
context = 4096
devices.placement = "gpu-only"
devices.split = "tensor"
devices.gpu_allow = [0, 1]
devices.tensor_split_weights = [2.6, 1.0]
lifecycle = "persistent"
"#,
        );
        let ec = validate(&cfg).unwrap();
        assert_eq!(
            ec.services[0].tensor_split_weights.as_deref(),
            Some(&[2.6f32, 1.0f32][..])
        );
    }

    #[test]
    fn rejects_tensor_split_weights_wrong_count() {
        let cfg = parse_and_merge(
            r#"
[[service]]
name = "demo"
template = "llama-cpp"
model = "/m/x.gguf"
port = 11435
context = 4096
devices.placement = "gpu-only"
devices.split = "tensor"
devices.gpu_allow = [0, 1]
devices.tensor_split_weights = [2.6, 1.0, 1.0]
lifecycle = "persistent"
"#,
        );
        let err = validate(&cfg).unwrap_err();
        let diag = &err.as_slice()[0];
        assert_eq!(diag.code(), ValidationErrorCode::TensorSplitWeightsCount);
    }

    #[test]
    fn rejects_tensor_split_weights_non_positive() {
        let cfg = parse_and_merge(
            r#"
[[service]]
name = "demo"
template = "llama-cpp"
model = "/m/x.gguf"
port = 11435
context = 4096
devices.placement = "gpu-only"
devices.split = "tensor"
devices.gpu_allow = [0, 1]
devices.tensor_split_weights = [2.6, 0.0]
lifecycle = "persistent"
"#,
        );
        let err = validate(&cfg).unwrap_err();
        let diag = &err.as_slice()[0];
        assert_eq!(diag.code(), ValidationErrorCode::TensorSplitWeightInvalid);
    }

    #[test]
    fn rejects_tensor_split_weights_on_non_sharded_split() {
        let cfg = parse_and_merge(
            r#"
[[service]]
name = "demo"
template = "llama-cpp"
model = "/m/x.gguf"
port = 11435
context = 4096
devices.placement = "gpu-only"
devices.gpu_allow = [0, 1]
devices.tensor_split_weights = [2.6, 1.0]
lifecycle = "persistent"
"#,
        );
        let err = validate(&cfg).unwrap_err();
        let diag = &err.as_slice()[0];
        assert_eq!(diag.fields(), [fields::devices::SPLIT]);
        assert!(
            diag.to_string()
                .contains("devices.tensor_split_weights is only valid")
        );
    }

    #[test]
    fn rejects_tensor_split_weights_on_hybrid_placement() {
        // Sharded splits already require gpu-only, so this fails on the split
        // constraint before it reaches the weight check.
        let cfg = parse_and_merge(
            r#"
[[service]]
name = "demo"
template = "llama-cpp"
model = "/m/x.gguf"
port = 11435
context = 4096
devices.placement = "hybrid"
devices.split = "tensor"
devices.gpu_allow = [0, 1]
devices.tensor_split_weights = [2.6, 1.0]
lifecycle = "persistent"
"#,
        );
        let err = validate(&cfg).unwrap_err();
        let diag = &err.as_slice()[0];
        assert_eq!(diag.fields(), [fields::devices::SPLIT]);
        assert!(diag.to_string().contains("requires placement=gpu-only"));
    }

    #[test]
    fn rejects_tensor_split_weights_with_unsorted_gpu_allow() {
        let cfg = parse_and_merge(
            r#"
[[service]]
name = "demo"
template = "llama-cpp"
model = "/m/x.gguf"
port = 11435
context = 4096
devices.placement = "gpu-only"
devices.split = "tensor"
devices.gpu_allow = [1, 0]
devices.tensor_split_weights = [2.6, 1.0]
lifecycle = "persistent"
"#,
        );
        let err = validate(&cfg).unwrap_err();
        let diag = &err.as_slice()[0];
        assert_eq!(diag.code(), ValidationErrorCode::GpuAllowUnsorted);
    }

    #[test]
    fn rejects_tensor_split_weights_with_duplicate_gpu_allow() {
        let cfg = parse_and_merge(
            r#"
[[service]]
name = "demo"
template = "llama-cpp"
model = "/m/x.gguf"
port = 11435
context = 4096
devices.placement = "gpu-only"
devices.split = "tensor"
devices.gpu_allow = [0, 0]
devices.tensor_split_weights = [2.6, 1.0]
lifecycle = "persistent"
"#,
        );
        let err = validate(&cfg).unwrap_err();
        let diag = &err.as_slice()[0];
        assert_eq!(diag.code(), ValidationErrorCode::GpuAllowDuplicate);
    }
}
