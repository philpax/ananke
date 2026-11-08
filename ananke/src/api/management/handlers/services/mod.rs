//! Service list, detail, and launch-command-preview handlers.

mod command;
mod config_view;
mod detail;
mod list;

pub use command::{__path_service_command, service_command};
pub(crate) use detail::service_detail_snapshot;
pub use detail::{__path_service_detail, service_detail};
pub(crate) use list::services_snapshot;
pub use list::{__path_list_services, list_services};
