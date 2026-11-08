use ananke_api::services::{detail::ServiceDetail, logs::LogsResponse};

use crate::{
    client::{ApiClient, ApiClientError},
    output,
};

/// How many of the newest log lines the human-readable view shows.
const RECENT_LOG_LINES: u32 = 10;

pub async fn run(client: &ApiClient, json: bool, name: &str) -> Result<(), ApiClientError> {
    let path = format!("/api/services/{}", name);
    let detail: ServiceDetail = client.get_json(&path).await?;
    if json {
        output::print_json(&detail);
    } else {
        let logs_path = format!("/api/services/{name}/logs?limit={RECENT_LOG_LINES}");
        let recent: LogsResponse = client.get_json(&logs_path).await?;
        output::print_service_detail(&detail, &recent.logs);
    }
    Ok(())
}
