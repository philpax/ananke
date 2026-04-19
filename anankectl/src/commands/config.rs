//! Configuration management command handlers.

use std::{io::Read, path::Path};

use ananke_api::{ConfigResponse, ConfigValidateRequest, ConfigValidateResponse};

use crate::{
    client::{ApiClient, ApiClientError},
    output,
};

pub async fn show(client: &ApiClient, json: bool) -> Result<(), ApiClientError> {
    let resp: ConfigResponse = client.get_json("/api/config").await?;
    if json {
        output::print_json(&resp);
    } else {
        println!("{}", resp.content);
    }
    Ok(())
}

pub async fn validate(
    client: &ApiClient,
    json: bool,
    file: Option<&Path>,
) -> Result<(), ApiClientError> {
    let content = match file {
        Some(p) => std::fs::read_to_string(p)
            .map_err(|e| ApiClientError::Usage(format!("read {}: {e}", p.display())))?,
        None => {
            let mut s = String::new();
            std::io::stdin().read_to_string(&mut s).ok();
            s
        }
    };
    let req = ConfigValidateRequest { content };
    let resp: ConfigValidateResponse = client.post_json("/api/config/validate", &req).await?;
    if json {
        output::print_json(&resp);
    } else if resp.valid {
        println!("ok: config is valid");
    } else {
        println!("error: config is invalid");
        for err in &resp.errors {
            println!("  line {}:{} {}", err.line, err.column, err.message);
        }
    }
    Ok(())
}

pub async fn reload(client: &ApiClient, _json: bool) -> Result<(), ApiClientError> {
    // Force-reload by PUTting the current file back to the server.
    // Read the file's current content via GET, then PUT it unchanged
    // with the matching If-Match hash.
    let resp: ConfigResponse = client.get_json("/api/config").await?;
    client
        .put_body("/api/config", resp.content, Some(&resp.hash))
        .await?;
    println!("ok: config reload requested");
    Ok(())
}
