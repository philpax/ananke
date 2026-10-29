//! Manual reservation commands: list, place, and remove.

use std::collections::BTreeMap;

use ananke_api::reservations::{
    CreateReservationRequest, CreateReservationResponse, ReservationsResponse,
};
use ananke_config::units::{GIB, MIB};

use crate::{
    client::{ApiClient, ApiClientError},
    output,
};

pub async fn list(client: &ApiClient, json: bool) -> Result<(), ApiClientError> {
    let resp: ReservationsResponse = client.get_json("/api/reservations").await?;
    if json {
        output::print_json(&resp);
    } else {
        output::print_reservations(&resp);
    }
    Ok(())
}

pub async fn reserve(
    client: &ApiClient,
    json: bool,
    devices: &[String],
    label: Option<String>,
    no_wait: bool,
) -> Result<(), ApiClientError> {
    let devices = devices
        .iter()
        .map(|arg| parse_device_amount(arg).map_err(ApiClientError::Usage))
        .collect::<Result<BTreeMap<_, _>, _>>()?;
    let req = CreateReservationRequest {
        devices,
        label,
        wait: Some(!no_wait),
    };
    let resp: CreateReservationResponse = client.post_json("/api/reservations", &req).await?;
    if json {
        output::print_json(&resp);
    } else {
        let r = &resp.reservation;
        let held = r
            .devices
            .iter()
            .map(|(id, bytes)| format!("{id} {}", output::format_bytes(*bytes)))
            .collect::<Vec<_>>()
            .join(", ");
        println!("ok: placed reservation {} ({held})", r.id);
        if !resp.evicted.is_empty() {
            let verb = if no_wait { "evicting" } else { "evicted" };
            println!("{verb}: {}", resp.evicted.join(", "));
        }
    }
    Ok(())
}

pub async fn unreserve(client: &ApiClient, json: bool, id: i64) -> Result<(), ApiClientError> {
    client.delete(&format!("/api/reservations/{id}")).await?;
    if json {
        output::print_json(&serde_json::json!({ "removed": id }));
    } else {
        println!("ok: removed reservation {id}");
    }
    Ok(())
}

/// Parse `DEVICE=SIZE`, such as `gpu:0=18G` or `cpu=512MiB`, into the
/// device id and a byte count. Sizes are binary and need a unit (`M`, `MiB`,
/// `G`, or `GiB`); fractions are allowed.
fn parse_device_amount(arg: &str) -> Result<(String, u64), String> {
    let (device, size) = arg
        .split_once('=')
        .ok_or_else(|| format!("`{arg}`: expected DEVICE=SIZE, such as gpu:0=18G"))?;
    let device = device.trim();
    if device != "cpu"
        && device
            .strip_prefix("gpu:")
            .and_then(|n| n.parse::<u32>().ok())
            .is_none()
    {
        return Err(format!("`{device}`: expected `cpu` or `gpu:N`"));
    }
    let size = size.trim();
    let split = size
        .find(|c: char| !(c.is_ascii_digit() || c == '.'))
        .ok_or_else(|| format!("`{size}`: add a unit, such as 18G or 512M"))?;
    let (number, unit) = size.split_at(split);
    let number: f64 = number
        .parse()
        .map_err(|_| format!("`{size}`: not a number"))?;
    let unit_bytes = match unit.trim() {
        "M" | "MiB" | "m" => MIB,
        "G" | "GiB" | "g" => GIB,
        other => return Err(format!("`{other}`: unknown unit; use M, MiB, G, or GiB")),
    };
    let bytes = (number * unit_bytes as f64).ceil();
    if bytes < 1.0 {
        return Err(format!("`{size}`: the size must be positive"));
    }
    Ok((device.to_string(), bytes as u64))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_device_amounts() {
        assert_eq!(
            parse_device_amount("gpu:0=18G"),
            Ok(("gpu:0".into(), 18 * GIB))
        );
        assert_eq!(
            parse_device_amount("cpu=512MiB"),
            Ok(("cpu".into(), 512 * MIB))
        );
        assert_eq!(
            parse_device_amount("gpu:1=1.5GiB"),
            Ok(("gpu:1".into(), 3 * GIB / 2))
        );
        assert!(parse_device_amount("gpu:0=18").is_err());
        assert!(parse_device_amount("gpu:0=0G").is_err());
        assert!(parse_device_amount("tpu=1G").is_err());
        assert!(parse_device_amount("gpu:0").is_err());
        assert!(parse_device_amount("gpu:0=5T").is_err());
    }
}
