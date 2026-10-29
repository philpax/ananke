//! Per-resource declarations for the live socket: which inputs can change a
//! resource's body (pure), and how to compute that body (a thin shell over
//! the same functions the HTTP handlers call).
//!
//! A trigger missing from [`triggers`] is a resource that silently goes
//! stale, which is what issue #45 was. When a snapshot function starts
//! reading a new input, add the input's signal here.

use ananke_api::{
    events::Event,
    live::{LiveData, LiveResource},
};

use crate::{
    api::{
        errors::ApiErrorCode,
        management::{
            events_history::events_history,
            handlers::{devices_snapshot, service_detail_snapshot, services_snapshot},
            info::info_snapshot,
        },
    },
    daemon::app_state::AppState,
    reservations::reservations_snapshot,
};

/// The change sources a resource's body depends on.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct Triggers {
    /// Wake on bus events that [`wants_event`] accepts.
    pub(crate) events: bool,
    /// Wake when the device sampler writes a new snapshot.
    pub(crate) device_snapshot: bool,
    /// Wake when any service's in-flight count changes.
    pub(crate) inflight: bool,
    /// Wake when any service's last-used stamp changes.
    pub(crate) activity: bool,
    /// Wake when any supervisor's state, run id, or pid changes.
    pub(crate) mirror: bool,
}

pub(crate) fn triggers(resource: &LiveResource) -> Triggers {
    match resource {
        // Fit verdicts read free memory and the allocation table; rows show
        // in-flight counts, last use, state, and pid.
        LiveResource::Services | LiveResource::Service { .. } => Triggers {
            events: true,
            device_snapshot: true,
            inflight: true,
            activity: true,
            mirror: true,
        },
        LiveResource::Devices => Triggers {
            events: true,
            device_snapshot: true,
            inflight: false,
            activity: false,
            mirror: false,
        },
        // Rows show each service's state and in-flight count beside the
        // allocation table.
        LiveResource::Reservations => Triggers {
            events: true,
            device_snapshot: false,
            inflight: true,
            activity: false,
            mirror: true,
        },
        LiveResource::Events | LiveResource::Info => Triggers {
            events: true,
            device_snapshot: false,
            inflight: false,
            activity: false,
            mirror: false,
        },
    }
}

/// Whether `event` can change `resource`'s body.
pub(crate) fn wants_event(resource: &LiveResource, event: &Event) -> bool {
    match resource {
        LiveResource::Services | LiveResource::Events => true,
        LiveResource::Devices | LiveResource::Reservations => matches!(
            event,
            Event::AllocationChanged { .. }
                | Event::ConfigReloaded { .. }
                | Event::ReservationPlaced { .. }
                | Event::ReservationRemoved { .. }
        ),
        LiveResource::Info => matches!(event, Event::ConfigReloaded { .. }),
        LiveResource::Service { name } => match event {
            Event::StateChanged { service, .. }
            | Event::AllocationChanged { service, .. }
            | Event::EstimatorDrift { service, .. }
            | Event::AutoRestarted { service, .. } => service.as_str() == name,
            // A placement preview reads every row of the allocation table.
            Event::ConfigReloaded { .. }
            | Event::ReservationPlaced { .. }
            | Event::ReservationRemoved { .. } => true,
        },
    }
}

/// Compute the resource's current body.
pub(crate) async fn snapshot(
    state: &AppState,
    resource: &LiveResource,
) -> Result<LiveData, ApiErrorCode> {
    Ok(match resource {
        LiveResource::Services => LiveData::Services(services_snapshot(state)),
        LiveResource::Devices => LiveData::Devices(devices_snapshot(state)),
        LiveResource::Events => LiveData::Events(events_history(state)),
        LiveResource::Info => LiveData::Info(info_snapshot(state)),
        LiveResource::Reservations => LiveData::Reservations(reservations_snapshot(state)),
        LiveResource::Service { name } => {
            LiveData::Service(Box::new(service_detail_snapshot(state, name).await?))
        }
    })
}

#[cfg(test)]
mod tests {
    use ananke_api::shared::ServiceStateKind;
    use smol_str::SmolStr;

    use super::*;

    #[test]
    fn service_views_wake_on_every_input_without_an_event() {
        for resource in [
            LiveResource::Services,
            LiveResource::Service {
                name: "demo".into(),
            },
        ] {
            let t = triggers(&resource);
            assert!(
                t.inflight && t.activity && t.mirror && t.device_snapshot,
                "{resource:?}"
            );
        }
    }

    #[test]
    fn devices_wake_on_samples_but_not_requests() {
        let t = triggers(&LiveResource::Devices);
        assert!(t.device_snapshot && t.events);
        assert!(!t.inflight && !t.activity && !t.mirror);
    }

    #[test]
    fn service_events_are_filtered_by_name() {
        let resource = LiveResource::Service {
            name: "demo".into(),
        };
        let state_changed = |service: &str| Event::StateChanged {
            service: SmolStr::new(service),
            from: ServiceStateKind::Idle,
            to: ServiceStateKind::Starting,
            disabled_reason: None,
            at_ms: 0,
        };
        assert!(wants_event(&resource, &state_changed("demo")));
        assert!(!wants_event(&resource, &state_changed("other")));
        assert!(wants_event(
            &resource,
            &Event::ConfigReloaded {
                at_ms: 0,
                changed_services: vec![],
            }
        ));
    }

    #[test]
    fn info_only_wakes_on_config_reloads() {
        assert!(!wants_event(
            &LiveResource::Info,
            &Event::EstimatorDrift {
                service: SmolStr::new("demo"),
                class: "vram".into(),
                rolling_mean: 1.0,
                at_ms: 0,
            }
        ));
    }

    #[test]
    fn reservation_changes_wake_every_view_that_reads_the_table() {
        let placed = Event::ReservationPlaced {
            id: 1,
            label: None,
            devices: Default::default(),
            evicted: vec![],
            at_ms: 0,
        };
        let removed = Event::ReservationRemoved {
            id: 1,
            label: None,
            at_ms: 0,
        };
        for resource in [
            LiveResource::Reservations,
            LiveResource::Devices,
            LiveResource::Services,
            LiveResource::Service {
                name: "demo".into(),
            },
        ] {
            assert!(wants_event(&resource, &placed), "{resource:?}");
            assert!(wants_event(&resource, &removed), "{resource:?}");
        }
        assert!(!wants_event(&LiveResource::Info, &placed));
        let t = triggers(&LiveResource::Reservations);
        assert!(t.events && t.inflight && t.mirror);
    }
}
