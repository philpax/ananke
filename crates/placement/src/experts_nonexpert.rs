//! Capacity tests for command-derived whole-layer MoE placement.

#[cfg(test)]
mod tests {
    use ananke_config::placement::OffloadMode;

    use crate::{
        AllocationTable,
        entry::pack,
        test_support::{moe_estimate, moe_svc, snapshot},
        types::PackError,
    };

    #[test]
    fn expert_offload_nonexpert_gpu_only_fails_instead_of_cpu_spilling() {
        let e = moe_estimate(20, 100, 700);
        let snap = snapshot(&[1]);
        let alloc = AllocationTable::new();
        let err = pack(&e, &moe_svc(OffloadMode::Auto), &snap, &alloc)
            .expect_err("non-expert weight is GPU-only; a tight card must fail, not CPU-spill");
        assert!(
            matches!(err, PackError::LayerDoesNotFit { .. }),
            "got {err:?}"
        );
    }

    #[test]
    fn expert_offload_headroom_forces_more_offload() {
        let e = moe_estimate(12, 100, 300);
        let snap = snapshot(&[6]);
        let alloc = AllocationTable::new();
        let tight = pack(&e, &moe_svc(OffloadMode::Auto), &snap, &alloc).unwrap();
        let mut svc = moe_svc(OffloadMode::Auto);
        svc.gpu_headroom_mb = 2048;
        let with_headroom = pack(&e, &svc, &snap, &alloc).unwrap();
        assert!(with_headroom.expert_offload_bytes > tight.expert_offload_bytes);
    }
}
