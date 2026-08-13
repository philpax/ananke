//! Snapshotter attribution hints (`[[service]].tracking`) and their
//! validation.

use smol_str::SmolStr;

use crate::{
    fields,
    validate::{ConfigDiagnostic, ValidationErrorCode},
};

/// Snapshotter attribution hints. Empty (`Default::default()`) by default,
/// in which case the snapshotter falls back to "registered pid +
/// transitive descendants" attribution. Set when the service runs in a
/// container (or otherwise out of the daemon's process tree).
#[derive(Debug, Clone, Default)]
pub struct TrackingSettings {
    /// Cgroup v2 path that contains every pid the service's workload
    /// runs in. The snapshotter unions in any pid whose cgroup matches
    /// this path or sits inside its subtree.
    pub cgroup_parent: Option<SmolStr>,
}

pub(crate) fn validate_tracking(
    _name: &SmolStr,
    raw: Option<&crate::parse::RawTracking>,
) -> Result<TrackingSettings, ConfigDiagnostic> {
    let Some(raw) = raw else {
        return Ok(TrackingSettings::default());
    };
    let cgroup_parent = match &raw.cgroup_parent {
        Some(s) if s.is_empty() => {
            return Err(ConfigDiagnostic::value(
                ValidationErrorCode::TrackingConstraint,
                fields::tracking::CGROUP_PARENT,
                "tracking.cgroup_parent is empty — omit the field or supply a non-empty cgroup path",
            ));
        }
        Some(s) if !s.starts_with('/') => {
            return Err(ConfigDiagnostic::value(
                ValidationErrorCode::TrackingConstraint,
                fields::tracking::CGROUP_PARENT,
                format!(
                    "tracking.cgroup_parent must be an absolute cgroup v2 path starting with `/` (got `{s}`)"
                ),
            ));
        }
        Some(s) if s.ends_with('/') => {
            return Err(ConfigDiagnostic::value(
                ValidationErrorCode::TrackingConstraint,
                fields::tracking::CGROUP_PARENT,
                format!("tracking.cgroup_parent must not end with `/` (got `{s}`)"),
            ));
        }
        Some(s)
            if !s
                .chars()
                .all(|c| c.is_ascii_alphanumeric() || matches!(c, '.' | '_' | '/' | '-')) =>
        {
            return Err(ConfigDiagnostic::value(
                ValidationErrorCode::TrackingConstraint,
                fields::tracking::CGROUP_PARENT,
                format!(
                    "tracking.cgroup_parent may only contain alphanumeric, `.`, `_`, `/`, and `-` characters (got `{s}`)"
                ),
            ));
        }
        Some(s) => Some(s.clone()),
        None => None,
    };
    Ok(TrackingSettings { cgroup_parent })
}

#[cfg(test)]
mod tests {
    use crate::validate::{ValidationErrorCode, test_fixtures::parse_and_merge, validate};

    #[test]
    fn tracking_cgroup_parent_accepted_when_well_formed() {
        let cfg = parse_and_merge(
            r#"
[[service]]
name = "comfy"
template = "command"
command = ["python"]
port = 8188
lifecycle = "on_demand"
allocation.mode = "dynamic"
allocation.min_reserve_gb = 2
allocation.max_reserve_gb = 20
tracking.cgroup_parent = "/system.slice/ananke-comfyui.slice"
"#,
        );
        let ec = validate(&cfg).unwrap();
        assert_eq!(
            ec.services[0].tracking.cgroup_parent.as_deref(),
            Some("/system.slice/ananke-comfyui.slice"),
        );
    }

    #[test]
    fn tracking_rejects_relative_cgroup_path() {
        let cfg = parse_and_merge(
            r#"
[[service]]
name = "comfy"
template = "command"
command = ["python"]
port = 8188
lifecycle = "on_demand"
allocation.mode = "static"
allocation.reserve_gb = 4
tracking.cgroup_parent = "ananke-comfyui.slice"
"#,
        );
        let err = validate(&cfg).unwrap_err();
        let diag = &err.as_slice()[0];
        assert_eq!(diag.code(), ValidationErrorCode::TrackingConstraint);
        assert!(diag.to_string().contains("must be an absolute cgroup"));
    }

    #[test]
    fn tracking_rejects_trailing_slash() {
        let cfg = parse_and_merge(
            r#"
[[service]]
name = "comfy"
template = "command"
command = ["python"]
port = 8188
lifecycle = "on_demand"
allocation.mode = "static"
allocation.reserve_gb = 4
tracking.cgroup_parent = "/system.slice/ananke-comfyui.slice/"
"#,
        );
        let err = validate(&cfg).unwrap_err();
        let diag = &err.as_slice()[0];
        assert_eq!(diag.code(), ValidationErrorCode::TrackingConstraint);
        assert!(diag.to_string().contains("must not end with"));
    }

    #[test]
    fn tracking_rejects_unsupported_characters() {
        let cfg = parse_and_merge(
            r#"
[[service]]
name = "comfy"
template = "command"
command = ["python"]
port = 8188
lifecycle = "on_demand"
allocation.mode = "static"
allocation.reserve_gb = 4
tracking.cgroup_parent = "/system.slice/ananke comfyui!.slice"
"#,
        );
        let err = validate(&cfg).unwrap_err();
        let diag = &err.as_slice()[0];
        assert_eq!(diag.code(), ValidationErrorCode::TrackingConstraint);
        // The dedicated message must render, not fall through to the generic
        // "invalid value" wording.
        assert!(diag.to_string().contains("may only contain alphanumeric"));
    }
}
