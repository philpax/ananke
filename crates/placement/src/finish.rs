//! Materialising the finished `Packed` result: deriving `-ngl`,
//! `--tensor-split`, `--split-mode`/`--main-gpu`, and `-ot` from the packer's
//! accumulated per-device state.

use ananke_config::placement::DeviceSlot;

use crate::{
    devices::{Allocation, DeviceId},
    entry::{NGL_CPU_ONLY, NGL_OFFLOAD_ALL},
    packer::Packer,
    types::{CommandArgs, Packed},
};

impl<'a> Packer<'a> {
    /// Step 6: materialise the final `Packed` — derive -ngl, --tensor-split,
    /// -ot, and convert the per_device map into an `Allocation`.
    pub(crate) fn finish(self) -> Packed {
        // Sharded (tensor/row) split: every layer is offloaded and divided
        // across all spanned GPUs by the configured proportions (equal by
        // default, or weighted via tensor_split_weights), so emit `-ngl 999`,
        // the plan's `--tensor-split` ratio, the `--split-mode`, and
        // `--main-gpu` (visible index 0 — the lowest-id GPU, which cuda_env
        // renders first).
        let (ngl, tensor_split, split_mode, main_gpu) = if let Some(plan) = &self.sharded {
            (
                Some(NGL_OFFLOAD_ALL),
                Some(plan.tensor_split.clone()),
                Some(plan.mode),
                Some(0),
            )
        } else {
            let total_on_gpus: u32 = self.layers_per_gpu.values().sum();
            let ngl = if self.allowed_gpus.is_empty() {
                Some(NGL_CPU_ONLY)
            } else if self.fallback_on_gpu {
                Some(NGL_OFFLOAD_ALL)
            } else {
                Some(total_on_gpus)
            };

            // Ratios in CUDA_VISIBLE_DEVICES-remapped order: must be in
            // ascending GPU-id order to match CUDA device numbering, regardless
            // of the placement sort order.
            let tensor_split = if self.allowed_gpus.len() > 1 && total_on_gpus > 0 {
                let mut gpus_by_id = self.allowed_gpus.clone();
                gpus_by_id.sort_unstable();
                if let Some(split) = &self.expert_tensor_split {
                    Some(split.clone())
                } else {
                    Some(
                        gpus_by_id
                            .iter()
                            .map(|g| self.layers_per_gpu.get(g).copied().unwrap_or(0))
                            .collect(),
                    )
                }
            } else {
                None
            };
            (ngl, tensor_split, None, None)
        };

        // Operator-declared `-ot` rules pass straight through; the packer
        // synthesises none of its own (whole-layer expert offload rides on
        // `--n-cpu-moe`, emitted above).
        let override_tensor = self.placement.override_tensor.clone();

        let expert_offload_bytes = self.expert_offload_cpu_bytes;
        let expert_offload_layers = self.expert_offload_cpu_layers.len() as u32;
        let rolling = self.rolling_inputs();

        let allocation = Allocation {
            bytes: self
                .per_device
                .into_iter()
                .map(|(slot, bytes)| {
                    let id = match slot {
                        DeviceSlot::Cpu => DeviceId::Cpu,
                        DeviceSlot::Gpu(n) => DeviceId::Gpu(n),
                    };
                    (id, bytes)
                })
                .collect(),
        };

        Packed {
            allocation,
            args: CommandArgs {
                ngl,
                tensor_split,
                override_tensor,
                split_mode,
                main_gpu,
                n_cpu_moe: self.n_cpu_moe,
            },
            expert_offload_bytes,
            expert_offload_layers,
            rolling,
        }
    }
}
