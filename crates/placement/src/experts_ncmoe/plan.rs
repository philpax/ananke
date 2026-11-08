use std::collections::BTreeSet;

use ananke_estimate::IkAssignment;

pub(super) struct Inputs<'a> {
    pub weights: &'a [u64],
    pub experts: &'a [u64],
    pub budgets: &'a [u64],
    pub kv: u64,
    pub compute: u64,
    pub output: u64,
    pub ik: bool,
    pub n_cpu: u32,
    pub assignment: Option<&'a IkAssignment>,
}

pub(super) struct Plan {
    pub homes: Vec<usize>,
    pub cpu: BTreeSet<u32>,
    pub split: Vec<u32>,
    pub costs: Vec<u64>,
}

pub(super) fn plan(input: &Inputs<'_>) -> Plan {
    let n = input.weights.len();
    let g = input.budgets.len();
    if input.ik
        && g > 1
        && let Some(assignment) = input.assignment
    {
        return bounded_plan(input, assignment);
    }
    let mut cuts = vec![0; g + 1];
    cuts[g] = n;
    let sum: u64 = input.budgets.iter().sum();
    let mut cumulative = 0u64;
    for (i, &budget) in input.budgets.iter().enumerate().take(g - 1) {
        cumulative = cumulative.saturating_add(budget);
        cuts[i + 1] = if sum == 0 {
            n * (i + 1) / g
        } else {
            (n as u128 * cumulative as u128 / sum as u128) as usize
        };
    }
    let mut best = evaluate(input, &cuts);
    for _ in 0..g * 2 + 2 {
        let mut changed = false;
        for i in 1..g {
            let mut chosen = cuts[i];
            let mut score = pressure(&best, input.budgets);
            for cut in cuts[i - 1]..=cuts[i + 1] {
                cuts[i] = cut;
                let candidate = evaluate(input, &cuts);
                let candidate_score = pressure(&candidate, input.budgets);
                if candidate_score < score {
                    score = candidate_score;
                    chosen = cut;
                    best = candidate;
                    changed = true;
                }
            }
            cuts[i] = chosen;
        }
        if !changed {
            break;
        }
    }
    best
}

fn bounded_plan(input: &Inputs<'_>, assignment: &IkAssignment) -> Plan {
    let g = input.budgets.len();
    let total: u64 = input.budgets.iter().sum();
    let proportional: Vec<u32> = input
        .budgets
        .iter()
        .map(|&b| ((b as u128 * 1_000_000 / total.max(1) as u128) as u32).max(1))
        .collect();
    let mut best = bounded_evaluate(input, assignment, vec![1; g]);
    let candidate = bounded_evaluate(input, assignment, proportional);
    if pressure(&candidate, input.budgets) < pressure(&best, input.budgets) {
        best = candidate;
    }
    best
}

fn bounded_evaluate(input: &Inputs<'_>, assignment: &IkAssignment, split: Vec<u32>) -> Plan {
    let n = input.weights.len();
    let g = split.len();
    let bounds = cut_bounds(assignment, &split);
    let mut homes = vec![g - 1; n];
    let mut costs = vec![0u64; g];
    let mut full_prefix = vec![0u64];
    let mut expert_prefix = vec![0u64];
    for layer in 0..n {
        full_prefix.push(
            full_prefix[layer]
                .saturating_add(input.weights[layer].max(assignment.layer_weights[layer]))
                .saturating_add(assignment.cache_upper[layer]),
        );
        expert_prefix.push(expert_prefix[layer].saturating_add(input.experts[layer]));
    }
    let mut previous = 0;
    for id in 0..g {
        let end = ((bounds[id + 1].0 + bounds[id + 1].1) / 2).max(previous);
        homes[previous..end].fill(id);
        previous = end;
        for start in bounds[id].0..=bounds[id].1 {
            for end in bounds[id + 1].0.max(start)..=bounds[id + 1].1 {
                let count = end - start;
                // Initial quotas round within half a layer. At most ceil(g/2)
                // total corrections can remove slots from any one device.
                let quota = if input.n_cpu as usize >= n {
                    count
                } else {
                    let round_error = 4.0 * f64::from(f32::EPSILON) * f64::from(input.n_cpu);
                    let corrections =
                        (g as f64 * (0.5 + round_error) + round_error).ceil() as usize + 1;
                    (count * input.n_cpu as usize / n.max(1)).saturating_sub(corrections)
                };
                let cost = (full_prefix[end] - full_prefix[start])
                    .saturating_sub(expert_prefix[end] - expert_prefix[end - quota])
                    .saturating_add(input.output.max(assignment.output_weights));
                costs[id] = costs[id].max(cost);
            }
        }
    }
    let cpu = offloaded(input, &homes);
    Plan {
        homes,
        cpu,
        split,
        costs,
    }
}

fn cut_bounds(assignment: &IkAssignment, split: &[u32]) -> Vec<(usize, usize)> {
    let n = assignment.layer_weights.len();
    let g = split.len();
    let mut sum = 0.0f32;
    let mut points = Vec::with_capacity(g);
    for &weight in split {
        sum += weight as f32;
        points.push(sum);
    }
    let weights = assignment
        .layer_weights
        .iter()
        .map(|&w| w as f64)
        .sum::<f64>();
    let maximum_total = weights
        + assignment
            .cache_upper
            .iter()
            .map(|&c| c as f64)
            .sum::<f64>()
        + assignment.output_weights as f64
        + assignment.logits_bytes as f64
        + g as f64 * assignment.scratch_upper as f64;
    // Covers f32 cumulative normalization, target conversion, loaded sums, and
    // both rounded subtractions in the nearest-layer comparison (39fc3128).
    let operations = 8.0 * (n + g + 4) as f64;
    let unit = f32::EPSILON as f64;
    let gamma = operations * unit / (1.0 - operations * unit).max(unit);
    let error = 4.0 * gamma * maximum_total + 8.0;
    let mut bounds = vec![(0, 0)];
    for (device, &point) in points.iter().enumerate().take(g - 1) {
        let p = f64::from(point / sum);
        let i = (device + 1) as f64;
        let mut low = 0;
        let mut high = 0;
        let mut prefix = 0.0;
        for layer in 0..n {
            let midpoint = prefix + assignment.layer_weights[layer] as f64 / 2.0;
            let mut lower = f64::INFINITY;
            let mut upper = f64::NEG_INFINITY;
            for scratch in [
                0,
                assignment.scratch_upper,
                assignment.logits_bytes.min(assignment.scratch_upper),
            ] {
                let c = scratch as f64;
                let threshold =
                    p * (weights + assignment.output_weights as f64) + (p * g as f64 - i) * c;
                lower = lower.min(threshold - midpoint);
                upper = upper.max(
                    threshold + p * assignment.logits_bytes.saturating_sub(scratch) as f64
                        - midpoint,
                );
            }
            for k in 0..n {
                let coefficient = p - if k < layer {
                    1.0
                } else if k == layer {
                    0.5
                } else {
                    0.0
                };
                let a = coefficient * assignment.cache_lower[k] as f64;
                let b = coefficient * assignment.cache_upper[k] as f64;
                lower += a.min(b);
                upper += a.max(b);
            }
            if lower > error {
                low = layer + 1;
            }
            if upper >= -error {
                high = layer + 1;
            }
            prefix += assignment.layer_weights[layer] as f64;
        }
        let &(previous_low, previous_high) = bounds.last().unwrap_or(&(0, 0));
        bounds.push((low.max(previous_low), high.max(previous_high)));
    }
    bounds.push((n, n));
    bounds
}

fn pressure(plan: &Plan, budgets: &[u64]) -> (u64, u64) {
    let mut maximum = 0;
    let mut total = 0u64;
    for (&cost, &budget) in plan.costs.iter().zip(budgets) {
        let ratio = (cost as u128 * 1_000_000 / budget.max(1) as u128).min(u64::MAX as u128) as u64;
        maximum = maximum.max(ratio);
        total = total.saturating_add(ratio);
    }
    (maximum, total)
}

fn evaluate(input: &Inputs<'_>, cuts: &[usize]) -> Plan {
    let n = input.weights.len();
    let g = input.budgets.len();
    let mut homes = vec![g - 1; n];
    for id in 0..g {
        homes[cuts[id]..cuts[id + 1]].fill(id);
    }
    let split = if input.ik {
        // Without runtime assignment metadata this is a nominal attribution only;
        // admission uses the whole-model bound in the caller.
        let mut sizes = input.weights.to_vec();
        for size in &mut sizes {
            *size = size.saturating_add(input.kv / n.max(1) as u64);
        }
        let total = sizes
            .iter()
            .sum::<u64>()
            .saturating_add(input.output)
            .saturating_add(input.compute.saturating_mul(g as u64));
        let mut points = vec![0.0f32];
        for id in 1..g {
            let prefix: u64 = sizes[..cuts[id]].iter().sum();
            points.push(
                (prefix.saturating_add(input.compute * id as u64)) as f32 / total.max(1) as f32,
            );
        }
        points.push(1.0);
        let split: Vec<u32> = points
            .windows(2)
            .map(|p| ((p[1] - p[0]) * 1_000_000.0).round().max(0.0) as u32)
            .collect();
        homes.fill(g - 1);
        let sum = split.iter().map(|&x| x as f32).sum::<f32>();
        let mut cumulative = 0.0f32;
        let mut loaded = 0.0f32;
        let mut last = 0;
        for (id, &weight) in split.iter().enumerate() {
            cumulative += weight as f32;
            loaded += input.compute as f32;
            let target = cumulative / sum * total as f32;
            while last < n {
                let size = sizes[last] as f32;
                if loaded + size <= target || loaded + size - target < target - loaded {
                    homes[last] = id;
                    loaded += size;
                    last += 1;
                } else {
                    break;
                }
            }
        }
        split
    } else {
        let mut points = vec![0u32];
        for &cut in &cuts[1..g] {
            points.push(if cut == 0 { 0 } else { 2 * cut as u32 - 1 });
        }
        points.push(2 * (n as u32 + 1));
        let split: Vec<u32> = points.windows(2).map(|p| p[1] - p[0]).collect();
        let sum = split.iter().sum::<u32>() as f32;
        let mut cumulative = 0.0f32;
        let points: Vec<f32> = split
            .iter()
            .map(|&x| {
                cumulative += x as f32;
                cumulative / sum
            })
            .collect();
        for (layer, home) in homes.iter_mut().enumerate() {
            let fraction = layer as f32 / (n + 1) as f32;
            *home = points.iter().position(|&p| p > fraction).unwrap_or(g - 1);
        }
        split
    };
    let cpu = offloaded(input, &homes);
    let mut costs = vec![0u64; g];
    for (layer, &home) in homes.iter().enumerate() {
        costs[home] = costs[home]
            .saturating_add(
                input.weights[layer].saturating_sub(if cpu.contains(&(layer as u32)) {
                    input.experts[layer]
                } else {
                    0
                }),
            )
            .saturating_add(input.kv / n.max(1) as u64);
    }
    Plan {
        homes,
        cpu,
        split,
        costs,
    }
}

fn offloaded(input: &Inputs<'_>, homes: &[usize]) -> BTreeSet<u32> {
    let n = homes.len();
    let g = input.budgets.len();
    if !input.ik {
        return (0..n)
            .filter(|&l| l < input.n_cpu as usize && input.experts[l] > 0)
            .map(|l| l as u32)
            .collect();
    }
    if g == 1 || input.n_cpu as usize >= n {
        return (0..n)
            .rev()
            .filter(|&l| input.experts[l] > 0)
            .take(input.n_cpu as usize)
            .map(|l| l as u32)
            .collect();
    }
    let counts: Vec<usize> = (0..g)
        .map(|id| homes.iter().filter(|&&home| home == id).count())
        .collect();
    let wanted: Vec<f32> = counts
        .iter()
        .map(|&c| c as f32 * input.n_cpu as f32 / n as f32)
        .collect();
    let mut quotas: Vec<usize> = wanted.iter().map(|x| x.round() as usize).collect();
    while quotas.iter().sum::<usize>() > input.n_cpu as usize {
        let mut best = None;
        let mut error = f32::NEG_INFINITY;
        for id in 0..g {
            let e = quotas[id] as f32 - 1.0 - wanted[id];
            if quotas[id] > 0 && e > error {
                best = Some(id);
                error = e;
            }
        }
        let Some(id) = best else {
            break;
        };
        quotas[id] -= 1;
    }
    while quotas.iter().sum::<usize>() < input.n_cpu as usize {
        let mut best = None;
        let mut error = f32::INFINITY;
        for id in 0..g {
            let e = quotas[id] as f32 + 1.0 - wanted[id];
            if quotas[id] < counts[id] && e < error {
                best = Some(id);
                error = e;
            }
        }
        let Some(id) = best else {
            break;
        };
        quotas[id] += 1;
    }
    let mut cpu = BTreeSet::new();
    for layer in (0..n).rev() {
        let id = homes[layer];
        if quotas[id] > 0 {
            quotas[id] -= 1;
            if input.experts[layer] > 0 {
                cpu.insert(layer as u32);
            }
        }
    }
    cpu
}
