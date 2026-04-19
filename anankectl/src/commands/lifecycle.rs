//! Service lifecycle command handlers.

use ananke_api::{DisableResponse, EnableResponse, StartResponse, StopResponse};

use crate::{
    client::{ApiClient, ApiClientError},
    output,
};

pub async fn start(client: &ApiClient, json: bool, name: &str) -> Result<(), ApiClientError> {
    let resp: StartResponse = client
        .post_empty(&format!("/api/services/{name}/start"))
        .await?;
    report_start(&resp, json);
    Ok(())
}

pub async fn stop(client: &ApiClient, json: bool, name: &str) -> Result<(), ApiClientError> {
    let resp: StopResponse = client
        .post_empty(&format!("/api/services/{name}/stop"))
        .await?;
    if json {
        output::print_json(&resp);
    } else {
        match resp {
            StopResponse::Drained => println!("ok: stopped '{name}'"),
            StopResponse::NotRunning => println!("noop: '{name}' was not running"),
        }
    }
    Ok(())
}

pub async fn restart(client: &ApiClient, json: bool, name: &str) -> Result<(), ApiClientError> {
    let resp: StartResponse = client
        .post_empty(&format!("/api/services/{name}/restart"))
        .await?;
    report_start(&resp, json);
    Ok(())
}

pub async fn enable(client: &ApiClient, json: bool, name: &str) -> Result<(), ApiClientError> {
    let resp: EnableResponse = client
        .post_empty(&format!("/api/services/{name}/enable"))
        .await?;
    if json {
        output::print_json(&resp);
    } else {
        match resp {
            EnableResponse::Enabled => println!("ok: enabled '{name}'"),
            EnableResponse::NotDisabled | EnableResponse::AlreadyEnabled => {
                println!("noop: '{name}' was not disabled");
            }
        }
    }
    Ok(())
}

pub async fn disable(client: &ApiClient, json: bool, name: &str) -> Result<(), ApiClientError> {
    let resp: DisableResponse = client
        .post_empty(&format!("/api/services/{name}/disable"))
        .await?;
    if json {
        output::print_json(&resp);
    } else {
        match resp {
            DisableResponse::Disabled => println!("ok: disabled '{name}'"),
            DisableResponse::AlreadyDisabled => println!("noop: '{name}' was already disabled"),
        }
    }
    Ok(())
}

pub async fn retry(client: &ApiClient, json: bool, name: &str) -> Result<(), ApiClientError> {
    // Best-effort enable (idempotent if not disabled), then start.
    let _ = client
        .post_empty::<EnableResponse>(&format!("/api/services/{name}/enable"))
        .await;
    start(client, json, name).await
}

fn report_start(resp: &StartResponse, json: bool) {
    if json {
        output::print_json(resp);
        return;
    }
    match resp {
        StartResponse::AlreadyRunning => println!("ok: already running"),
        StartResponse::Started { run_id } => println!("ok: started (run_id={run_id})"),
        StartResponse::QueueFull => println!("error: start queue full"),
        StartResponse::Unavailable { reason } => println!("error: unavailable ({reason})"),
    }
}
