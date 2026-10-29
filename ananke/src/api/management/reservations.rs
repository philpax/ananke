//! `/api/reservations` — list, place, and remove manual reservations.

use ananke_api::{
    reservations::{CreateReservationRequest, CreateReservationResponse, ReservationsResponse},
    shared::errors::ApiError,
};
use axum::{
    Json,
    extract::{Path, State, rejection::JsonRejection},
    http::StatusCode,
    response::{IntoResponse, Response},
};

use crate::{api::errors::ApiErrorCode, daemon::app_state::AppState, reservations};

#[utoipa::path(
    summary = "List reservations",
    get,
    path = "/api/reservations",
    responses((status = 200, body = ReservationsResponse))
)]
pub async fn get_reservations(State(state): State<AppState>) -> Response {
    (
        StatusCode::OK,
        Json(reservations::reservations_snapshot(&state)),
    )
        .into_response()
}

#[utoipa::path(
    summary = "Place a manual reservation",
    description = "Reserve memory on one or more devices so ananke never places a service \
                   into it. A reservation outranks every service: any service in the way \
                   is evicted, busy and persistent ones included. By default the response \
                   is sent once they have stopped; with `wait: false` it is sent as soon as \
                   their evictions begin.",
    post,
    path = "/api/reservations",
    request_body = CreateReservationRequest,
    responses(
        (status = 201, body = CreateReservationResponse),
        (status = 400, body = ApiError, description = "invalid_request_error"),
        (status = 409, body = ApiError, description = "reservation_no_fit"),
        (status = 500, body = ApiError, description = "persist_failed")
    )
)]
pub async fn post_reservation(
    State(state): State<AppState>,
    body: Result<Json<CreateReservationRequest>, JsonRejection>,
) -> Response {
    let Json(req) = match body {
        Ok(body) => body,
        Err(rejection) => {
            return ApiErrorCode::InvalidRequest {
                reason: rejection.body_text(),
            }
            .into_response();
        }
    };
    match reservations::create(&state, req).await {
        Ok(created) => (StatusCode::CREATED, Json(created)).into_response(),
        Err(code) => code.into_response(),
    }
}

#[utoipa::path(
    summary = "Remove a manual reservation",
    delete,
    path = "/api/reservations/{id}",
    params(("id" = i64, Path, description = "Reservation id")),
    responses(
        (status = 204),
        (status = 404, body = ApiError, description = "reservation_not_found"),
        (status = 500, body = ApiError, description = "persist_failed")
    )
)]
pub async fn delete_reservation(State(state): State<AppState>, Path(id): Path<i64>) -> Response {
    match reservations::remove(&state, id).await {
        Ok(()) => StatusCode::NO_CONTENT.into_response(),
        Err(code) => code.into_response(),
    }
}
