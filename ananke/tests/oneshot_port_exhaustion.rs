use ananke::oneshot::PortPool;

#[test]
fn pool_exhausts_returns_none() {
    let mut pool = PortPool::new(18000..18002);
    assert!(pool.allocate().is_some());
    assert!(pool.allocate().is_some());
    assert!(pool.allocate().is_none());
}

#[test]
fn release_restores_availability() {
    let mut pool = PortPool::new(18000..18001);
    let p = pool.allocate().unwrap();
    assert!(pool.allocate().is_none());
    pool.release(p);
    assert!(pool.allocate().is_some());
}
