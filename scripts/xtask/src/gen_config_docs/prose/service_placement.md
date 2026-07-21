Placement policies:

- `gpu-only` (default): Service must reside entirely on GPU.
- `cpu-only`: Service resides entirely on CPU. `n_gpu_layers` must be `0` (or unset), otherwise config validation rejects it.
- `hybrid`: Allows a mix of GPU and CPU. The packer fills the GPUs first and spills the remainder to CPU. For MoE models with `expert_offload` enabled it spills *expert tensors* before whole layers, keeping every layer's attention and KV cache on the GPU (see [MoE Expert Offload](#moe-expert-offload)). Manual `override_tensor` rules also work here for hand-picked CPU offloading.

#### Multi-GPU split modes

When a llama.cpp service spans more than one GPU, `devices.split` selects how llama.cpp divides the model across them. It maps directly to llama.cpp's `--split-mode`:

```toml
[service.devices]
placement = "gpu-only"
split = "tensor"   # "layer" (default), "row", or "tensor"
```

- `layer` (default): pipeline parallelism - each GPU holds a contiguous range of whole layers. ananke estimates each layer's footprint and packs them across the allowed GPUs first-fit, so the split ratio follows the per-GPU layer counts. Lowest interconnect demand; the right default when the cards have no fast peer link.
- `row`: the older tensor-parallel mode (`--split-mode row`). Splits individual tensors by row. Without NVLink/P2P it is typically *slower* than `layer` because every token incurs cross-GPU traffic over PCIe; prefer `tensor` over `row` on such hosts.
- `tensor`: the newer tensor-parallel mode (`--split-mode tensor`). Shards each tensor across the GPUs and emits a balanced `--tensor-split` with `--main-gpu` set to the lowest allowed GPU. On dual identical cards this measures meaningfully faster decode than `layer` even without P2P, at the cost of a larger compute buffer and constant cross-GPU communication.

`row` and `tensor` are sharded modes and carry extra constraints, rejected at config validation:

- The service must use `placement = "gpu-only"` - a sharded model cannot spill to CPU.
- Only valid for `llama-cpp` services, not `command` services.
- Cannot be combined with `override_tensor` (manual tensor placement), since the sharded modes manage tensor placement themselves.

With a sharded mode, ananke reserves an equal share of the model weights, KV cache, and compute buffer on each allowed GPU by default, placing the non-layer remainder (output tensor, MTP overhead, …) on the main GPU. The pledge book reflects this per-GPU split, so a co-tenant (e.g. an embedding service) sees the true free capacity on each card.

For heterogeneous GPUs, set `devices.tensor_split_weights` to a weight per allowed GPU in ascending GPU-id order. The weights scale the per-GPU share and the emitted `--tensor-split` ratio. For example, an RTX 3090 paired with an RTX 3060, where the 3090 has roughly 2.6 times the memory bandwidth, can be weighted `[2.6, 1.0]` to give the faster card ~2.6 times the tensors instead of the historical equal split:

```toml
[service.devices]
placement = "gpu-only"
split = "tensor"
gpu_allow = [0, 1]
tensor_split_weights = [2.6, 1.0]
```

The weights are normalised by their sum, so only the ratio matters. The number of weights must match the number of allowed GPUs, and weights must be positive and finite.
