use ananke_api::ServiceDetail;

use crate::{
    client::{ApiClient, ApiClientError},
    output,
};

pub async fn run(client: &ApiClient, json: bool, name: &str) -> Result<(), ApiClientError> {
    let path = format!("/api/services/{}", name);
    let detail: ServiceDetail = client.get_json(&path).await?;
    if json {
        output::print_json(&detail);
    } else {
        output::print_service_detail(&detail);
    }
    Ok(())
}
