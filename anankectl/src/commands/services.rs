use ananke_api::ServicesResponse;

use crate::{
    client::{ApiClient, ApiClientError},
    output,
};

pub async fn run(client: &ApiClient, json: bool, all: bool) -> Result<(), ApiClientError> {
    let resp: ServicesResponse = client.get_json("/api/services").await?;
    if json {
        output::print_json(&resp);
    } else {
        output::print_services_table(&resp.services, all);
    }
    Ok(())
}
