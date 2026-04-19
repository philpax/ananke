use ananke_api::DeviceSummary;

use crate::{
    client::{ApiClient, ApiClientError},
    output,
};

pub async fn run(client: &ApiClient, json: bool) -> Result<(), ApiClientError> {
    let devices: Vec<DeviceSummary> = client.get_json("/api/devices").await?;
    if json {
        output::print_json(&devices);
    } else {
        output::print_devices_table(&devices);
    }
    Ok(())
}
