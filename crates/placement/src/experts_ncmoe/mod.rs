//! Whole-layer expert placement derived from the emitted command.
//! Multi-GPU ik reservations bound possible contiguous ranges using runtime
//! assignment metadata rather than calibrated compute-buffer estimates.

mod plan;

#[cfg(test)]
mod tests;

use ananke_config::placement::{DeviceSlot, OffloadMode, PlacementInputs};

use crate::{
    entry::PackMode,
    packer::{Charge, Packer},
    types::PackError,
};

pub(crate) enum Ncmoe {
    /// Mainline: offload blocks below this index.
    LeadingBlocksBelow(u32),
    /// ik_llama: offload this many trailing expert layers.
    TrailingLayers(u32),
}

impl Ncmoe {
    /// Is this service served by the fork?
    pub(crate) fn is_ik(placement: &PlacementInputs) -> bool {
        placement.ik_llama
    }

    /// Global expert offload for the sharded path.
    pub(crate) fn for_runtime(
        placement: &PlacementInputs,
        n: u32,
        layers: &[u32],
        _gpus: usize,
        _mtp_head_layers: u32,
    ) -> Self {
        if Self::is_ik(placement) {
            Self::TrailingLayers(n.min(layers.len() as u32))
        } else {
            Self::LeadingBlocksBelow(n)
        }
    }

    /// The plan that retains `kept` expert layers, from whichever end this
    /// runtime keeps them.
    pub(crate) fn keeping(placement: &PlacementInputs, kept: u32, layers: &[u32]) -> Self {
        let total = layers.len() as u32;
        if Self::is_ik(placement) {
            Self::TrailingLayers(total.saturating_sub(kept))
        } else {
            // Mainline keeps a trailing suffix, so the bound is the lowest block
            // it retains — or one past the highest when it retains none.
            Self::LeadingBlocksBelow(match layers.len().checked_sub(kept as usize) {
                Some(i) if i < layers.len() => layers[i],
                _ => layers.last().map(|l| l + 1).unwrap_or(0),
            })
        }
    }

    pub(crate) fn keeps(&self, layer: u32, layers: &[u32]) -> bool {
        match *self {
            Self::LeadingBlocksBelow(bound) => layer >= bound,
            Self::TrailingLayers(n) => {
                let kept = layers.len().saturating_sub(n as usize);
                layers
                    .iter()
                    .position(|&l| l == layer)
                    .is_some_and(|i| i < kept)
            }
        }
    }

    /// The value to pass on the command line.
    pub(crate) fn flag(&self) -> u32 {
        match *self {
            Self::LeadingBlocksBelow(bound) => bound,
            Self::TrailingLayers(n) => n,
        }
    }
}

impl<'a> Packer<'a> {
    pub(crate) fn distribute_experts_ncmoe(&mut self) -> Result<(), PackError> {
        let mut gpus = self.allowed_gpus.clone();
        gpus.sort_unstable();
        if gpus.is_empty() {
            if !matches!(self.mode, PackMode::Demand) {
                return Err(PackError::LayerDoesNotFit {
                    layer_index: 0,
                    bytes: self.effective_layer_avg(),
                    shortfalls: Vec::new(),
                });
            }
            for (layer, bytes) in self.per_layer.clone().into_iter().enumerate() {
                self.charge(DeviceSlot::Cpu, bytes, Charge::Weights);
                self.spilled_layers.insert(layer as u32);
                self.layers_on_cpu += 1;
            }
            return Ok(());
        }
        let output = self.estimate.layout.non_layer.output_head_bytes
            + self.estimate.layout.non_layer.tied_head_bytes;
        if let (Some(&first), Some(&last)) = (gpus.first(), gpus.last())
            && first != last
            && output > 0
        {
            let slot = DeviceSlot::Gpu(first);
            let corrected = self.vram_cost(output);
            if let Some(bytes) = self.per_device.get_mut(&slot) {
                *bytes = bytes.saturating_sub(corrected);
            }
            if let Some(bytes) = self.raw_per_device.get_mut(&slot) {
                *bytes = bytes.saturating_sub(output);
            }
            if let Some(bytes) = self.raw_weight_per_device.get_mut(&slot) {
                *bytes = bytes.saturating_sub(output);
            }
            self.charge(DeviceSlot::Gpu(last), output, Charge::Weights);
            self.initialise_gpu_remaining();
        }
        let n = self.per_layer.len();
        let experts: Vec<u64> = (0..n)
            .map(|l| {
                self.expert_bytes_by_layer
                    .get(&(l as u32))
                    .copied()
                    .unwrap_or(0)
            })
            .collect();
        let budgets: Vec<u64> = gpus
            .iter()
            .map(|g| self.gpu_remaining.get(g).copied().unwrap_or(0))
            .collect();
        let raw_budgets: Vec<u64> = budgets
            .iter()
            .map(|&b| (b as f64 / self.corrections.vram) as u64)
            .collect();
        let kv = self
            .estimate
            .kv_per_token
            .saturating_mul(self.estimate.context as u64);
        let mut n_cpu = match self.offload_mode {
            OffloadMode::Layers(n) => n,
            _ => 0,
        };
        let conservative_ik =
            self.placement.ik_llama && gpus.len() > 1 && !matches!(self.mode, PackMode::Demand);
        let mut expert_sizes = experts.clone();
        expert_sizes.sort_unstable_by(|a, b| b.cmp(a));
        let nonexpert_total: u64 = self
            .per_layer
            .iter()
            .zip(&experts)
            .map(|(&w, &e)| w.saturating_sub(e))
            .sum();
        let selected = loop {
            let input = plan::Inputs {
                weights: &self.per_layer,
                experts: &experts,
                budgets: &raw_budgets,
                kv,
                compute: self.estimate.buffers.compute_mb as u64 * 1024 * 1024,
                output: self.estimate.layout.non_layer.output_head_bytes
                    + self.estimate.layout.non_layer.tied_head_bytes,
                ik: self.placement.ik_llama,
                n_cpu,
                assignment: self
                    .estimate
                    .layout
                    .ik_assignment
                    .as_ref()
                    .filter(|_| n <= 4096 && gpus.len() <= 64),
            };
            let mut candidate = plan::plan(&input);
            if conservative_ik && input.assignment.is_none() {
                let retained_bound: u64 = expert_sizes
                    .iter()
                    .take(n.saturating_sub(n_cpu as usize))
                    .sum();
                candidate.costs.fill(
                    nonexpert_total
                        .saturating_add(kv)
                        .saturating_add(retained_bound)
                        .saturating_add(output),
                );
            }
            let shortfalls: Vec<_> = candidate
                .costs
                .iter()
                .zip(&budgets)
                .zip(&gpus)
                .filter_map(|((&cost, &available), &gpu)| {
                    let requested = self.vram_cost(cost);
                    (requested > available).then_some(crate::types::DeviceShortfall {
                        device: crate::devices::DeviceId::Gpu(gpu),
                        requested_bytes: requested,
                        available_bytes: available,
                    })
                })
                .collect();
            if shortfalls.is_empty()
                || matches!(self.mode, PackMode::Demand)
                    && (n_cpu >= n as u32 || !matches!(self.offload_mode, OffloadMode::Auto))
            {
                break candidate;
            }
            if !matches!(self.offload_mode, OffloadMode::Auto) {
                return Err(PackError::ManualExpertsDoNotFit {
                    needed: shortfalls.iter().map(|s| s.requested_bytes).sum(),
                    available: shortfalls.iter().map(|s| s.available_bytes).sum(),
                    shortfalls,
                });
            }
            if n_cpu >= n as u32 {
                return Err(PackError::LayerDoesNotFit {
                    layer_index: 0,
                    bytes: self.effective_layer_avg(),
                    shortfalls,
                });
            }
            n_cpu += 1;
        };
        let conservative_costs = selected.costs.clone();
        self.expert_tensor_split = Some(selected.split);
        self.fallback_on_gpu = true;
        if n_cpu > 0 {
            self.n_cpu_moe = Some(n_cpu);
        }
        let before = self.per_device.clone();
        let mut remaining = budgets;
        for (layer, &home) in selected.homes.iter().enumerate() {
            let gpu = gpus[home];
            let offloaded = if selected.cpu.contains(&(layer as u32)) {
                experts[layer]
            } else {
                0
            };
            let retained = self.per_layer[layer].saturating_sub(offloaded);
            let cost =
                self.vram_cost(retained.saturating_add(kv.checked_div(n as u64).unwrap_or(0)));
            if matches!(self.mode, PackMode::Demand) && cost > remaining[home] {
                self.charge(DeviceSlot::Cpu, retained, Charge::Weights);
                self.layers_on_cpu += 1;
                self.spilled_layers.insert(layer as u32);
            } else {
                self.charge(DeviceSlot::Gpu(gpu), retained, Charge::Weights);
                remaining[home] = remaining[home].saturating_sub(cost);
                *self.layers_per_gpu.entry(gpu).or_default() += 1;
                self.layer_home.insert(layer as u32, gpu);
            }
            if offloaded > 0 {
                self.charge(DeviceSlot::Cpu, offloaded, Charge::Weights);
                self.expert_offload_cpu_bytes += offloaded;
                self.expert_offload_cpu_layers.insert(layer as u32);
            }
        }
        if conservative_ik {
            for (&gpu, &bound) in gpus.iter().zip(&conservative_costs) {
                let slot = DeviceSlot::Gpu(gpu);
                let weight_charge = self
                    .per_device
                    .get(&slot)
                    .copied()
                    .unwrap_or(0)
                    .saturating_sub(before.get(&slot).copied().unwrap_or(0));
                let kv_share = kv * u64::from(self.layers_per_gpu.get(&gpu).copied().unwrap_or(0))
                    / n.max(1) as u64;
                let reserved = self
                    .vram_cost(bound)
                    .saturating_sub(self.vram_cost(kv_share));
                if reserved > weight_charge {
                    // Add the corrected difference directly to avoid inverse-scaling
                    // truncation. Range uncertainty is slop, not observed usage.
                    *self.per_device.entry(slot).or_default() += reserved - weight_charge;
                }
            }
            let cpu_bound: u64 = expert_sizes.iter().take(n_cpu as usize).sum();
            self.charge(
                DeviceSlot::Cpu,
                cpu_bound.saturating_sub(self.expert_offload_cpu_bytes),
                Charge::Slop,
            );
        }
        Ok(())
    }
}
