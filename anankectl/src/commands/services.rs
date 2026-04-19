use ananke_api::ServiceSummary;

use crate::{
    client::{ApiClient, ApiClientError},
    output,
};

pub async fn run(client: &ApiClient, json: bool, all: bool) -> Result<(), ApiClientError> {
    let services: Vec<ServiceSummary> = client.get_json("/api/services").await?;
    if json {
        output::print_json(&services);
    } else {
        output::print_services_table(&services, all);
    }
    Ok(())
}
