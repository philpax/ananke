//! `anankectl status`: combined daemon health, devices, and services snapshot.

use ananke_api::{
    devices::list::DeviceSummary,
    services::list::{ServiceSummary, ServicesResponse},
    shared::{DisableReason, ServiceStateKind},
};
use serde::Serialize;

use crate::{
    client::{ApiClient, ApiClientError},
    output,
};

pub async fn run(client: &ApiClient, json: bool) -> Result<(), ApiClientError> {
    let (services, devices) = tokio::try_join!(
        client.get_json::<ServicesResponse>("/api/services"),
        client.get_json::<Vec<DeviceSummary>>("/api/devices"),
    )?;

    if json {
        output::print_json(&StatusReport {
            endpoint: client.endpoint.as_str(),
            openai_api_port: services.openai_api_port,
            services: &services.services,
            devices: &devices,
        });
        return Ok(());
    }

    let running = services
        .services
        .iter()
        .filter(|s| s.state == ServiceStateKind::Running)
        .count();
    let idle = services
        .services
        .iter()
        .filter(|s| s.state == ServiceStateKind::Idle)
        .count();
    let disabled = services
        .services
        .iter()
        .filter(|s| s.state == ServiceStateKind::Disabled)
        .count();
    let faulted = services
        .services
        .iter()
        .filter(|s| s.disabled_reason.is_some_and(DisableReason::is_fault))
        .count();
    let total = services.services.len();

    let dot = if running > 0 { "●" } else { "○" };
    let faulted = if faulted > 0 {
        format!(" ({faulted} faulted)")
    } else {
        String::new()
    };
    println!(
        "{dot} {total} services · {running} running · {idle} idle · {disabled} disabled{faulted}"
    );
    println!(
        "  endpoint    {}  (openai port {})",
        client.endpoint.as_str(),
        services.openai_api_port
    );
    println!();

    println!("DEVICES");
    output::print_devices_table(&devices);
    println!();

    println!("SERVICES");
    // Hide disabled here — `anankectl services --all` is the explicit way
    // to see them, and the disabled count above is enough for "is anything
    // missing?".
    output::print_services_table(&services.services, false);

    Ok(())
}

/// `status --json`: the two API responses joined under the endpoint they
/// came from. Borrows both, since it is serialised and dropped.
#[derive(Serialize)]
struct StatusReport<'a> {
    endpoint: &'a str,
    openai_api_port: u16,
    services: &'a [ServiceSummary],
    devices: &'a [DeviceSummary],
}
