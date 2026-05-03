//! `anankectl status`: combined daemon health, devices, and services snapshot.

use ananke_api::{DeviceSummary, ServicesResponse};

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
        output::print_json(&serde_json::json!({
            "endpoint": client.endpoint.as_str(),
            "openai_api_port": services.openai_api_port,
            "services": services.services,
            "devices": devices,
        }));
        return Ok(());
    }

    let running = services
        .services
        .iter()
        .filter(|s| s.state == "running")
        .count();
    let idle = services
        .services
        .iter()
        .filter(|s| s.state == "idle")
        .count();
    let disabled = services
        .services
        .iter()
        .filter(|s| s.state.starts_with("disabled"))
        .count();
    let total = services.services.len();

    println!("daemon");
    println!("  endpoint:        {}", client.endpoint.as_str());
    println!("  openai api port: {}", services.openai_api_port);
    println!(
        "  services:        {total} total · {running} running · {idle} idle · {disabled} disabled"
    );
    println!();

    println!("devices");
    output::print_devices_table(&devices);
    println!();

    println!("services");
    // Hide disabled here — `anankectl services --all` is the explicit way
    // to see them, and the disabled count above is enough for "is anything
    // missing?".
    output::print_services_table(&services.services, false);

    Ok(())
}
