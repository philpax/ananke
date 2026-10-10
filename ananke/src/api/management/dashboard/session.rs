//! Connection-local subscriptions and bounded concurrent operations; no dashboard polling loop.

use std::collections::{BTreeSet, HashMap};

use ananke_api::dashboard::{DashboardMessage, DashboardQuery, DashboardRequest};
use ananke_db::DatabaseChange;
use axum::{
    Router,
    extract::ws::{Message, WebSocket},
};
use futures::{SinkExt, StreamExt};
use serde_json::json;
use tokio::{
    sync::{broadcast::error::RecvError, mpsc},
    task::{AbortHandle, JoinSet},
};

use crate::{
    api::management::{self, dashboard::query},
    daemon::app_state::AppState,
};

const MAX_SUBSCRIPTIONS: usize = 64;
const MAX_OPERATIONS: usize = 32;

struct Subscription {
    query: DashboardQuery,
    service_id: Option<i64>,
}

enum Change {
    Event,
    Snapshot,
    Database(DatabaseChange),
    Logs(BTreeSet<i64>),
    All,
}

pub async fn serve(socket: WebSocket, state: AppState) {
    // Subscribe before any reads, so writes during initial backfill are replayed.
    let mut events = state.events.subscribe();
    let mut snapshots = state.snapshot.subscribe();
    let mut database = state.db.subscribe();
    let mut logs = state.batcher.subscribe();
    let router = management::register(Router::new(), state.clone());
    let (mut sink, mut incoming) = socket.split();
    let (out, mut outgoing) = mpsc::channel::<DashboardMessage>(64);
    let mut subscriptions = HashMap::<String, Subscription>::new();
    let mut operations = HashMap::<String, AbortHandle>::new();
    let mut tasks = JoinSet::new();
    let mut heartbeat = tokio::time::interval(std::time::Duration::from_secs(25));
    heartbeat.set_missed_tick_behavior(tokio::time::MissedTickBehavior::Delay);
    heartbeat.tick().await;
    loop {
        tokio::select! {
            frame = incoming.next() => {
                let Some(Ok(frame)) = frame else { break };
                let text = match frame {
                    Message::Text(text) => text,
                    Message::Close(_) => break,
                    Message::Ping(bytes) => { if sink.send(Message::Pong(bytes)).await.is_err() { break } continue; }
                    _ => continue,
                };
                let request = match serde_json::from_str::<DashboardRequest>(&text) {
                    Ok(request) => request,
                    Err(error) => {
                        let message = error_message("", 400, format!("dashboard: invalid request: {error}"));
                        if send(&mut sink, message).await.is_err() { break };
                        continue;
                    }
                };
                if let DashboardRequest::Cancel { id } = request {
                    subscriptions.remove(&id);
                    if let Some(task) = operations.remove(&id) { task.abort(); }
                    continue;
                }
                let id = request_id(&request).to_string();
                if id.is_empty() || id.len() > 128 || subscriptions.contains_key(&id) || operations.contains_key(&id) {
                    if send(&mut sink, error_message(&id, 409, "dashboard: invalid or duplicate operation id".into())).await.is_err() { break };
                    continue;
                }
                if let DashboardRequest::Read { query, subscribe, .. } = request {
                    if subscribe && subscriptions.len() >= MAX_SUBSCRIPTIONS {
                        if send(&mut sink, error_message(&id, 429, "dashboard: subscription limit reached".into())).await.is_err() { break };
                        continue;
                    }
                    let message = read(&id, &query, &router, &state).await;
                    if subscribe {
                        let service_id = match &query { DashboardQuery::Logs { name, .. } => state.db.resolve_service_id(name).await.ok().flatten(), _ => None };
                        subscriptions.insert(id, Subscription { query, service_id });
                    }
                    if send(&mut sink, message).await.is_err() { break };
                } else {
                    if operations.len() >= MAX_OPERATIONS {
                        if send(&mut sink, error_message(&id, 429, "dashboard: operation limit reached".into())).await.is_err() { break };
                        continue;
                    }
                    let router = router.clone();
                    let out = out.clone();
                    let task_id = id.clone();
                    let task = tasks.spawn(async move {
                        if let Err(error) = execute(&task_id, request, router, &out).await {
                            let _ = out.send(error_message(&task_id, 502, error)).await;
                        }
                        task_id
                    });
                    operations.insert(id, task);
                }
            }
            Some(message) = outgoing.recv() => {
                if send(&mut sink, message).await.is_err() { break };
            }
            Some(result) = tasks.join_next(), if !tasks.is_empty() => {
                if let Ok(id) = result { operations.remove(&id); }
            }
            changed = snapshots.changed() => {
                if changed.is_err() { break };
                if refresh(&mut sink, &subscriptions, Change::Snapshot, &router, &state).await.is_err() { break };
            }
            event = events.recv() => {
                let change = match event { Ok(_) => Change::Event, Err(RecvError::Lagged(_)) => Change::All, Err(RecvError::Closed) => break };
                while events.try_recv().is_ok() {}
                if refresh(&mut sink, &subscriptions, change, &router, &state).await.is_err() { break };
            }
            changed = database.recv() => {
                let change = match changed { Ok(change) => Change::Database(change), Err(RecvError::Lagged(_)) => Change::All, Err(RecvError::Closed) => break };
                if refresh(&mut sink, &subscriptions, change, &router, &state).await.is_err() { break };
            }
            changed = logs.recv() => {
                let change = match changed {
                    Ok((service_id, _)) => {
                        let mut services = BTreeSet::from([service_id]);
                        while let Ok((service_id, _)) = logs.try_recv() { services.insert(service_id); }
                        Change::Logs(services)
                    }
                    Err(RecvError::Lagged(_)) => Change::All,
                    Err(RecvError::Closed) => break,
                };
                if refresh(&mut sink, &subscriptions, change, &router, &state).await.is_err() { break };
            }
            _ = heartbeat.tick() => {
                if sink.send(Message::Ping(Vec::new())).await.is_err() { break };
            }
        }
    }
    tasks.abort_all();
}

async fn read(
    id: &str,
    resource: &DashboardQuery,
    router: &Router,
    state: &AppState,
) -> DashboardMessage {
    let result = if matches!(resource, DashboardQuery::Events) {
        serde_json::to_value(state.events.history())
            .map(|body| (200, body))
            .map_err(|e| format!("dashboard: encode captured events: {e}"))
    } else {
        match query::read(resource) {
            Ok(request) => match query::response(router.clone(), request).await {
                Ok(response) => query::json_response(response).await,
                Err(error) => Err(error),
            },
            Err(error) => Err(error),
        }
    };
    match result {
        Ok((status, body)) => DashboardMessage::Response {
            id: id.into(),
            status,
            body,
        },
        Err(error) => error_message(id, 500, error),
    }
}

async fn refresh<S>(
    sink: &mut S,
    subscriptions: &HashMap<String, Subscription>,
    change: Change,
    router: &Router,
    state: &AppState,
) -> Result<(), ()>
where
    S: futures::Sink<Message> + Unpin,
{
    for (id, subscription) in subscriptions {
        if affected(subscription, &change) {
            send(sink, read(id, &subscription.query, router, state).await).await?;
        }
    }
    Ok(())
}

fn affected(subscription: &Subscription, change: &Change) -> bool {
    match change {
        Change::All => true,
        Change::Snapshot => matches!(
            subscription.query,
            DashboardQuery::Services
                | DashboardQuery::Service { .. }
                | DashboardQuery::Devices
                | DashboardQuery::Info
        ),
        Change::Event => !matches!(
            subscription.query,
            DashboardQuery::Metrics { .. }
                | DashboardQuery::Samples { .. }
                | DashboardQuery::Logs { .. }
        ),
        Change::Database(DatabaseChange::Metrics) => matches!(
            subscription.query,
            DashboardQuery::Metrics { .. }
                | DashboardQuery::Services
                | DashboardQuery::Service { .. }
        ),
        Change::Database(DatabaseChange::Samples) => matches!(
            subscription.query,
            DashboardQuery::Samples { .. }
                | DashboardQuery::Metrics {
                    range: ananke_api::dashboard::DashboardRange {
                        live_ms: Some(_),
                        ..
                    },
                    ..
                }
                | DashboardQuery::Restarts {
                    range: ananke_api::dashboard::DashboardRange {
                        live_ms: Some(_),
                        ..
                    },
                    ..
                }
        ),
        Change::Database(DatabaseChange::Restarts) => matches!(
            subscription.query,
            DashboardQuery::Restarts { .. } | DashboardQuery::Service { .. }
        ),
        Change::Logs(services) => {
            matches!(&subscription.query, DashboardQuery::Logs { range, before: None, .. } if range.live_ms.is_some() && subscription.service_id.is_some_and(|id| services.contains(&id)))
        }
    }
}

async fn execute(
    id: &str,
    request: DashboardRequest,
    router: Router,
    out: &mpsc::Sender<DashboardMessage>,
) -> Result<(), String> {
    let chat = matches!(request, DashboardRequest::Chat { .. });
    let response = query::response(router, query::command(&request)?).await?;
    if !chat || !response.status().is_success() {
        let (status, body) = query::json_response(response).await?;
        return out
            .send(DashboardMessage::Response {
                id: id.into(),
                status,
                body,
            })
            .await
            .map_err(|_| "dashboard: connection closed".into());
    }
    let mut stream = response.into_body().into_data_stream();
    let mut pending = Vec::new();
    while let Some(bytes) = stream.next().await {
        pending.extend_from_slice(&bytes.map_err(|e| format!("chat: read upstream stream: {e}"))?);
        let valid = match std::str::from_utf8(&pending) {
            Ok(_) => pending.len(),
            Err(error) if error.error_len().is_none() => error.valid_up_to(),
            Err(error) => {
                return Err(format!(
                    "chat: upstream stream contains invalid UTF-8: {error}"
                ));
            }
        };
        if valid > 0 {
            let text = std::str::from_utf8(&pending[..valid])
                .map_err(|e| format!("chat: decode stream: {e}"))?
                .to_string();
            out.send(DashboardMessage::Stream {
                id: id.into(),
                text,
            })
            .await
            .map_err(|_| "dashboard: connection closed")?;
            pending.drain(..valid);
        }
    }
    if !pending.is_empty() {
        return Err("chat: upstream stream ends within a UTF-8 character".into());
    }
    out.send(DashboardMessage::End { id: id.into() })
        .await
        .map_err(|_| "dashboard: connection closed".into())
}

fn request_id(request: &DashboardRequest) -> &str {
    match request {
        DashboardRequest::Read { id, .. }
        | DashboardRequest::Cancel { id }
        | DashboardRequest::Action { id, .. }
        | DashboardRequest::Validate { id, .. }
        | DashboardRequest::Save { id, .. }
        | DashboardRequest::Chat { id, .. } => id,
    }
}

fn error_message(id: &str, status: u16, error: String) -> DashboardMessage {
    DashboardMessage::Response {
        id: id.into(),
        status,
        body: json!({"error": error}),
    }
}

async fn send<S>(sink: &mut S, message: DashboardMessage) -> Result<(), ()>
where
    S: futures::Sink<Message> + Unpin,
{
    let text = serde_json::to_string(&message).map_err(|_| ())?;
    tokio::time::timeout(
        std::time::Duration::from_secs(10),
        sink.send(Message::Text(text)),
    )
    .await
    .map_err(|_| ())?
    .map_err(|_| ())
}
