//! Apply `strip_params` and `set_params` to a JSON body.

use serde_json::Value;

use crate::config::Filters;

/// Apply filters in place. Strip first, then set.
pub fn apply(body: &mut Value, filters: &Filters) {
    if let Some(obj) = body.as_object_mut() {
        for key in &filters.strip_params {
            obj.remove(key);
        }
        for (key, value) in &filters.set_params {
            obj.insert(key.clone(), value.clone());
        }
    }
}

#[cfg(test)]
mod tests {
    use std::collections::BTreeMap;

    use serde_json::json;

    use super::*;

    fn filters_with(strip: &[&str], set: &[(&str, Value)]) -> Filters {
        let mut set_params = BTreeMap::new();
        for (k, v) in set {
            set_params.insert(k.to_string(), v.clone());
        }
        Filters {
            strip_params: strip.iter().map(|s| s.to_string()).collect(),
            set_params,
        }
    }

    #[test]
    fn strips_keys() {
        let mut body = json!({"model":"m","temperature":0.7});
        apply(&mut body, &filters_with(&["temperature"], &[]));
        assert_eq!(body, json!({"model":"m"}));
    }

    #[test]
    fn sets_keys() {
        let mut body = json!({"model":"m"});
        apply(
            &mut body,
            &filters_with(&[], &[("max_tokens", json!(4096))]),
        );
        assert_eq!(body["max_tokens"], json!(4096));
    }

    #[test]
    fn strip_then_set_order() {
        let mut body = json!({"model":"m","temperature":0.7});
        apply(
            &mut body,
            &filters_with(&["temperature"], &[("temperature", json!(0.3))]),
        );
        assert_eq!(body["temperature"], json!(0.3));
    }

    #[test]
    fn no_object_is_noop() {
        let mut body = json!([1, 2, 3]);
        apply(&mut body, &filters_with(&["temperature"], &[]));
        assert_eq!(body, json!([1, 2, 3]));
    }
}
