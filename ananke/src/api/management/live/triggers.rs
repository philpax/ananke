//! Turns a resource's [`Triggers`] into one awaitable stream of "something
//! this resource reads may have changed".

use std::future::pending;

use ananke_api::{events::Event, live::LiveResource};
use tokio::sync::{
    broadcast::{
        self,
        error::{RecvError, TryRecvError},
    },
    watch,
};

use crate::{
    api::management::live::resource::{Triggers, wants_event},
    daemon::app_state::AppState,
};

/// A change source shut down; the daemon is going away.
#[derive(Debug)]
pub(crate) struct Closed;

/// The receivers for one subscription. Build it before taking the first
/// snapshot, so a change landing between the two still wakes it.
pub(crate) struct TriggerStream {
    resource: LiveResource,
    events: Option<broadcast::Receiver<Event>>,
    device_snapshot: Option<watch::Receiver<()>>,
    inflight: Option<watch::Receiver<u64>>,
    activity: Option<watch::Receiver<u64>>,
    mirror: Option<watch::Receiver<u64>>,
}

impl TriggerStream {
    pub(crate) fn new(state: &AppState, resource: LiveResource, triggers: Triggers) -> Self {
        Self {
            resource,
            events: triggers.events.then(|| state.events.subscribe()),
            device_snapshot: triggers.device_snapshot.then(|| state.snapshot.subscribe()),
            inflight: triggers.inflight.then(|| state.inflight.changes()),
            activity: triggers.activity.then(|| state.activity.changes()),
            mirror: triggers.mirror.then(|| state.mirror_changed.subscribe()),
        }
    }

    /// Wait for the next change. A lagged event receiver counts as a
    /// change: the next snapshot is complete, so nothing missed matters.
    pub(crate) async fn changed(&mut self) -> Result<(), Closed> {
        let resource = &self.resource;
        tokio::select! {
            r = next_event(self.events.as_mut(), resource) => r,
            r = watch_changed(self.device_snapshot.as_mut()) => r,
            r = watch_changed(self.inflight.as_mut()) => r,
            r = watch_changed(self.activity.as_mut()) => r,
            r = watch_changed(self.mirror.as_mut()) => r,
        }
    }

    /// Mark everything pending as seen, so changes already covered by the
    /// snapshot about to be taken do not wake the stream again.
    pub(crate) fn drain(&mut self) {
        if let Some(rx) = self.events.as_mut() {
            while let Ok(_) | Err(TryRecvError::Lagged(_)) = rx.try_recv() {}
        }
        for rx in [&mut self.inflight, &mut self.activity, &mut self.mirror]
            .into_iter()
            .flatten()
        {
            rx.borrow_and_update();
        }
        if let Some(rx) = self.device_snapshot.as_mut() {
            rx.borrow_and_update();
        }
    }
}

async fn next_event(
    rx: Option<&mut broadcast::Receiver<Event>>,
    resource: &LiveResource,
) -> Result<(), Closed> {
    let Some(rx) = rx else {
        return pending().await;
    };
    loop {
        match rx.recv().await {
            Ok(event) if wants_event(resource, &event) => return Ok(()),
            Ok(_) => {}
            Err(RecvError::Lagged(_)) => return Ok(()),
            Err(RecvError::Closed) => return Err(Closed),
        }
    }
}

async fn watch_changed<T>(rx: Option<&mut watch::Receiver<T>>) -> Result<(), Closed> {
    match rx {
        Some(rx) => rx.changed().await.map_err(|_| Closed),
        None => pending().await,
    }
}
