//! Read-only management endpoints.

mod devices;
mod estimate;
mod services;

use axum::routing::{Router, get};
pub(crate) use devices::devices_snapshot;
// `utoipa::path` generates a hidden `__path_<fn>` sibling item next to each
// handler; the `OpenApi` derive in `api::openapi` resolves both through
// this re-export, so it must travel alongside the handler function.
pub use devices::{__path_list_devices, list_devices};
pub(crate) use estimate::{model_estimate_entry, placement_preview, read_current_allocation};
pub use services::{
    __path_list_services, __path_service_command, __path_service_detail, list_services,
    service_command, service_detail,
};
pub(crate) use services::{service_detail_snapshot, services_snapshot};

use crate::daemon::app_state::AppState;

pub fn register(router: Router, state: AppState) -> Router {
    // Build the typed router against AppState, collapse to Router<()> via
    // with_state, then merge into the caller's router.
    let mgmt: Router = Router::new()
        .route("/api/services", get(list_services))
        .route("/api/services/:name", get(service_detail))
        .route("/api/services/:name/command", get(service_command))
        .route("/api/devices", get(list_devices))
        .with_state(state);
    router.merge(mgmt)
}
