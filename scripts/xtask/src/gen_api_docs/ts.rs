//! TypeScript type generation from OpenAPI schemas.
//!
//! All `$ref`s are expanded inline. A `visited` set prevents infinite
//! recursion on circular `$ref` chains.

use std::collections::{BTreeMap, BTreeSet};

use serde_json::Value;

/// Maximum recursion depth for inline type expansion.
pub const MAX_TS_DEPTH: usize = 8;

/// Resolve a `$ref` string (e.g. `#/components/schemas/Foo`) within `spec`.
pub fn resolve_ref<'a>(spec: &'a Value, ref_str: &str) -> Option<&'a Value> {
    let path = ref_str.strip_prefix("#/")?;
    let mut current = spec;
    for segment in path.split('/') {
        current = current.get(segment)?;
    }
    Some(current)
}

/// Check whether a `oneOf` variant is a `{"type": "null"}` sentinel.
fn is_null_type(v: &Value) -> bool {
    match v.get("type") {
        Some(Value::String(s)) => s == "null",
        Some(Value::Array(arr)) => arr.iter().all(|t| t.as_str() == Some("null")),
        _ => false,
    }
}

/// Extract a human-readable label for a `oneOf` variant.
///
/// Looks for a required property whose schema has a single-value `enum`
/// (the discriminator). Falls back to `"variant_N"`.
fn oneof_variant_label(variant: &Value, index: usize) -> String {
    if let Some(props) = variant.get("properties").and_then(|p| p.as_object()) {
        let required: Vec<&str> = variant
            .get("required")
            .and_then(|r| r.as_array())
            .map(|arr| arr.iter().filter_map(|v| v.as_str()).collect())
            .unwrap_or_default();
        for field in &required {
            if let Some(fs) = props.get(*field) {
                if let Some(enums) = fs.get("enum").and_then(|e| e.as_array()) {
                    if enums.len() == 1 {
                        if let Some(val) = enums.first().and_then(|v| v.as_str()) {
                            return val.to_string();
                        }
                    }
                }
            }
        }
    }
    format!("variant_{}", index)
}

/// Generate a TypeScript type string from a schema.
///
/// All `$ref`s are expanded inline. `visited` prevents infinite recursion
/// on circular references — when a schema is revisited, its name is used
/// as a placeholder instead of expanding again.
///
/// Objects are rendered multi-line with proper indentation.
pub fn ts_type(
    schema: &Value,
    spec: &Value,
    visited: &mut BTreeSet<String>,
    indent: usize,
) -> String {
    if indent > MAX_TS_DEPTH {
        return "any".to_string();
    }

    // $ref — always expand inline, unless we've already visited it (circular).
    if let Some(ref_str) = schema.get("$ref").and_then(|r| r.as_str()) {
        let name = ref_str
            .strip_prefix("#/components/schemas/")
            .unwrap_or(ref_str);
        if visited.contains(name) {
            return name.to_string();
        }
        if let Some(resolved) = resolve_ref(spec, ref_str) {
            visited.insert(name.to_string());
            let result = ts_type(resolved, spec, visited, indent);
            visited.remove(name);
            return result;
        }
        return name.to_string();
    }

    // allOf — merge properties from all entries.
    if let Some(all_of) = schema.get("allOf").and_then(|a| a.as_array()) {
        let mut merged: BTreeMap<String, &Value> = BTreeMap::new();
        let mut required: BTreeSet<String> = BTreeSet::new();
        let mut has_passthrough = false;
        for sub in all_of {
            if let Some(props) = sub.get("properties").and_then(|p| p.as_object()) {
                for (k, v) in props {
                    merged.insert(k.clone(), v);
                }
            }
            if let Some(req) = sub.get("required").and_then(|r| r.as_array()) {
                for r in req {
                    if let Some(s) = r.as_str() {
                        required.insert(s.to_string());
                    }
                }
            }
            if sub.get("additionalProperties").and_then(|v| v.as_bool()) == Some(true) {
                has_passthrough = true;
            }
        }
        if merged.is_empty() {
            return "Record<string, any>".to_string();
        }
        let mut obj = format_ts_object(&merged, &required, spec, visited, indent);
        if has_passthrough {
            obj = obj.replace("}", "  ...any\n}");
        }
        return obj;
    }

    // oneOf / anyOf.
    for keyword in ["oneOf", "anyOf"] {
        if let Some(variants) = schema.get(keyword).and_then(|o| o.as_array()) {
            let non_null: Vec<&Value> = variants.iter().filter(|v| !is_null_type(v)).collect();
            let has_null = non_null.len() < variants.len();

            if non_null.is_empty() {
                return "null".to_string();
            }

            // Nullable wrapper: Type | null.
            if non_null.len() == 1 && has_null {
                let inner = ts_type(non_null[0], spec, visited, indent);
                return format!("{} | null", inner);
            }

            // Tagged union: show each variant with a comment label.
            let mut parts: Vec<String> = Vec::new();
            for (i, variant) in non_null.iter().enumerate() {
                let label = oneof_variant_label(variant, i);
                let ts = ts_inline(variant, spec, visited);
                if parts.is_empty() {
                    parts.push(format!("// {}\n{}", label, ts));
                } else {
                    parts.push(format!("\n// {}\n{}", label, ts));
                }
            }
            return parts.join("");
        }
    }

    // Determine primary type (handle nullable type arrays).
    let (types, has_null) = get_types(schema);
    let non_null_types: Vec<&str> = types
        .iter()
        .map(|s| s.as_str())
        .filter(|s| *s != "null")
        .collect();
    let primary = non_null_types.first().copied();

    let base = match primary {
        Some("object") => {
            if let Some(props) = schema.get("properties").and_then(|p| p.as_object()) {
                if !props.is_empty() {
                    let required: BTreeSet<String> = schema
                        .get("required")
                        .and_then(|r| r.as_array())
                        .map(|arr| {
                            arr.iter()
                                .filter_map(|v| v.as_str().map(String::from))
                                .collect()
                        })
                        .unwrap_or_default();
                    let props_map: BTreeMap<String, &Value> =
                        props.iter().map(|(k, v)| (k.clone(), v)).collect();
                    format_ts_object(&props_map, &required, spec, visited, indent)
                } else {
                    format_additional_properties(schema, spec, visited, indent)
                }
            } else {
                format_additional_properties(schema, spec, visited, indent)
            }
        }
        Some("array") => {
            if let Some(items) = schema.get("items") {
                let item_type = ts_type(items, spec, visited, indent);
                format!("{}[]", item_type)
            } else {
                "any[]".to_string()
            }
        }
        Some("string") => {
            if let Some(enums) = schema.get("enum").and_then(|e| e.as_array()) {
                let vals: Vec<String> = enums
                    .iter()
                    .map(|v| format!("\"{}\"", v.as_str().unwrap_or("")))
                    .collect();
                vals.join(" | ")
            } else {
                "string".to_string()
            }
        }
        Some("integer") | Some("number") => "number".to_string(),
        Some("boolean") => "boolean".to_string(),
        Some(other) => other.to_string(),
        None => {
            if let Some(enums) = schema.get("enum").and_then(|e| e.as_array()) {
                let vals: Vec<String> = enums
                    .iter()
                    .map(|v| format!("\"{}\"", v.as_str().unwrap_or("")))
                    .collect();
                vals.join(" | ")
            } else {
                "any".to_string()
            }
        }
    };

    if has_null {
        format!("{} | null", base)
    } else {
        base
    }
}

/// Generate a compact single-line TypeScript type (for `oneOf` variants and
/// other inline contexts). Never expands `$ref`s — always uses the type
/// name (to avoid duplicating large schemas in union members).
pub fn ts_inline(schema: &Value, spec: &Value, visited: &mut BTreeSet<String>) -> String {
    // $ref → type name only.
    if let Some(ref_str) = schema.get("$ref").and_then(|r| r.as_str()) {
        return ref_str
            .strip_prefix("#/components/schemas/")
            .unwrap_or(ref_str)
            .to_string();
    }

    // allOf → merge and format inline.
    if let Some(all_of) = schema.get("allOf").and_then(|a| a.as_array()) {
        let mut merged: BTreeMap<String, &Value> = BTreeMap::new();
        let mut required: BTreeSet<String> = BTreeSet::new();
        let mut has_passthrough = false;
        for sub in all_of {
            if let Some(props) = sub.get("properties").and_then(|p| p.as_object()) {
                for (k, v) in props {
                    merged.insert(k.clone(), v);
                }
            }
            if let Some(req) = sub.get("required").and_then(|r| r.as_array()) {
                for r in req {
                    if let Some(s) = r.as_str() {
                        required.insert(s.to_string());
                    }
                }
            }
            if sub.get("additionalProperties").and_then(|v| v.as_bool()) == Some(true) {
                has_passthrough = true;
            }
        }
        if merged.is_empty() {
            return "Record<string, any>".to_string();
        }
        let mut obj = format_ts_inline_object(&merged, &required, spec, visited);
        if has_passthrough {
            obj = obj.replace(" }", ", ...any }");
        }
        return obj;
    }

    // oneOf / anyOf.
    for keyword in ["oneOf", "anyOf"] {
        if let Some(variants) = schema.get(keyword).and_then(|o| o.as_array()) {
            let non_null: Vec<&Value> = variants.iter().filter(|v| !is_null_type(v)).collect();
            let has_null = non_null.len() < variants.len();
            if non_null.is_empty() {
                return "null".to_string();
            }
            if non_null.len() == 1 && has_null {
                let inner = ts_inline(non_null[0], spec, visited);
                return format!("{} | null", inner);
            }
            let parts: Vec<String> = non_null
                .iter()
                .map(|v| ts_inline(v, spec, visited))
                .collect();
            return parts.join(" | ");
        }
    }

    let (types, has_null) = get_types(schema);
    let non_null_types: Vec<&str> = types
        .iter()
        .map(|s| s.as_str())
        .filter(|s| *s != "null")
        .collect();
    let primary = non_null_types.first().copied();

    let base = match primary {
        Some("object") => {
            if let Some(props) = schema.get("properties").and_then(|p| p.as_object()) {
                if !props.is_empty() {
                    let required: BTreeSet<String> = schema
                        .get("required")
                        .and_then(|r| r.as_array())
                        .map(|arr| {
                            arr.iter()
                                .filter_map(|v| v.as_str().map(String::from))
                                .collect()
                        })
                        .unwrap_or_default();
                    let props_map: BTreeMap<String, &Value> =
                        props.iter().map(|(k, v)| (k.clone(), v)).collect();
                    format_ts_inline_object(&props_map, &required, spec, visited)
                } else {
                    "Record<string, any>".to_string()
                }
            } else {
                "Record<string, any>".to_string()
            }
        }
        Some("array") => {
            if let Some(items) = schema.get("items") {
                let item_type = ts_inline(items, spec, visited);
                format!("{}[]", item_type)
            } else {
                "any[]".to_string()
            }
        }
        Some("string") => {
            if let Some(enums) = schema.get("enum").and_then(|e| e.as_array()) {
                let vals: Vec<String> = enums
                    .iter()
                    .map(|v| format!("\"{}\"", v.as_str().unwrap_or("")))
                    .collect();
                vals.join(" | ")
            } else {
                "string".to_string()
            }
        }
        Some("integer") | Some("number") => "number".to_string(),
        Some("boolean") => "boolean".to_string(),
        Some(other) => other.to_string(),
        None => {
            if let Some(enums) = schema.get("enum").and_then(|e| e.as_array()) {
                let vals: Vec<String> = enums
                    .iter()
                    .map(|v| format!("\"{}\"", v.as_str().unwrap_or("")))
                    .collect();
                vals.join(" | ")
            } else {
                "any".to_string()
            }
        }
    };

    if has_null {
        format!("{} | null", base)
    } else {
        base
    }
}

/// Format an object as a multi-line TypeScript block.
fn format_ts_object(
    props: &BTreeMap<String, &Value>,
    required: &BTreeSet<String>,
    spec: &Value,
    visited: &mut BTreeSet<String>,
    indent: usize,
) -> String {
    if props.is_empty() {
        return "{}".to_string();
    }

    let pad = "  ".repeat(indent);
    let inner_pad = "  ".repeat(indent + 1);

    let mut lines: Vec<String> = vec!["{".to_string()];
    let mut sorted: Vec<_> = props.iter().collect();
    sorted.sort_by(|a, b| a.0.cmp(b.0));

    for (field, field_schema) in sorted {
        let field_type = ts_type(field_schema, spec, visited, indent + 1);
        let is_required = required.contains(field.as_str());
        let field_name = if is_required {
            field.to_string()
        } else {
            format!("{}?", field)
        };

        if field_type.contains('\n') {
            let mut type_lines = field_type.lines();
            let first = type_lines.next().unwrap_or("");
            lines.push(format!("{}{}: {}", inner_pad, field_name, first));
            for line in type_lines {
                lines.push(line.to_string());
            }
        } else {
            lines.push(format!("{}{}: {}", inner_pad, field_name, field_type));
        }
    }
    lines.push(format!("{}}}", pad));
    lines.join("\n")
}

/// Format an object as a single-line TypeScript literal: `{ a: T, b: U }`.
fn format_ts_inline_object(
    props: &BTreeMap<String, &Value>,
    required: &BTreeSet<String>,
    spec: &Value,
    visited: &mut BTreeSet<String>,
) -> String {
    let mut parts: Vec<String> = Vec::new();
    let mut sorted: Vec<_> = props.iter().collect();
    sorted.sort_by(|a, b| a.0.cmp(b.0));
    for (field, field_schema) in sorted {
        let field_type = ts_inline(field_schema, spec, visited);
        let is_required = required.contains(field.as_str());
        let field_name = if is_required {
            field.to_string()
        } else {
            format!("{}?", field)
        };
        parts.push(format!("{}: {}", field_name, field_type));
    }
    format!("{{ {} }}", parts.join(", "))
}

/// Handle `additionalProperties` (map type).
fn format_additional_properties(
    schema: &Value,
    spec: &Value,
    visited: &mut BTreeSet<String>,
    indent: usize,
) -> String {
    if let Some(addl) = schema.get("additionalProperties") {
        match addl {
            Value::Bool(true) => "Record<string, any>".to_string(),
            Value::Bool(false) => "{}".to_string(),
            _ => {
                let val_type = ts_type(addl, spec, visited, indent);
                format!("Record<string, {}>", val_type)
            }
        }
    } else {
        "Record<string, any>".to_string()
    }
}

/// Extract the `type` field as a list of type strings, and whether `null`
/// is among them.
fn get_types(schema: &Value) -> (Vec<String>, bool) {
    let types: Vec<String> = match schema.get("type") {
        Some(Value::String(s)) => vec![s.clone()],
        Some(Value::Array(arr)) => arr
            .iter()
            .filter_map(|v| v.as_str().map(String::from))
            .collect(),
        _ => vec![],
    };
    let has_null = types.iter().any(|t| t == "null");
    (types, has_null)
}

/// Extract a human-readable, TS-style type name from a schema. Used in
/// field tables and response columns. Never expands `$ref`s.
pub fn schema_ref_name(schema: &Value) -> String {
    // $ref
    if let Some(r#ref) = schema.get("$ref").and_then(|r| r.as_str()) {
        return r#ref
            .strip_prefix("#/components/schemas/")
            .unwrap_or(r#ref)
            .to_string();
    }

    // oneOf / anyOf — nullable ref or tagged union.
    for keyword in ["oneOf", "anyOf"] {
        if let Some(variants) = schema.get(keyword).and_then(|o| o.as_array()) {
            let refs: Vec<_> = variants
                .iter()
                .filter_map(|v| v.get("$ref").and_then(|r| r.as_str()))
                .map(|r| {
                    r.strip_prefix("#/components/schemas/")
                        .unwrap_or(r)
                        .to_string()
                })
                .collect();
            let has_null = variants
                .iter()
                .any(|v| v.get("type").and_then(|t| t.as_str()) == Some("null"));
            if refs.len() == 1 && has_null {
                return format!("{} | null", refs[0]);
            }
            if !refs.is_empty() {
                return refs.join(" | ");
            }
            let non_null: Vec<_> = variants
                .iter()
                .filter(|v| v.get("type").and_then(|t| t.as_str()) != Some("null"))
                .collect();
            if non_null.len() == 1 {
                return format!("{} | null", schema_ref_name(non_null[0]));
            }
            return "oneOf".to_string();
        }
    }

    // allOf
    if schema.get("allOf").is_some() {
        return "object".to_string();
    }

    // type as string
    if let Some(ty) = schema.get("type").and_then(|t| t.as_str()) {
        return ts_primitive_name(ty, schema);
    }

    // type as array (nullable) — e.g. ["integer", "null"]
    if let Some(types) = schema.get("type").and_then(|t| t.as_array()) {
        let non_null: Vec<_> = types
            .iter()
            .filter_map(|v| v.as_str())
            .filter(|s| *s != "null")
            .collect();
        let has_null = types.iter().any(|t| t.as_str() == Some("null"));
        if let Some(first) = non_null.first() {
            let base = ts_primitive_name(first, schema);
            return if has_null {
                format!("{} | null", base)
            } else {
                base
            };
        }
        return "null".to_string();
    }

    // No type — maybe an enum.
    if schema.get("enum").is_some() {
        return "string".to_string();
    }

    "any".to_string()
}

/// Map an OpenAPI primitive type to a TypeScript type name.
fn ts_primitive_name(ty: &str, schema: &Value) -> String {
    match ty {
        "array" => {
            if let Some(items) = schema.get("items") {
                format!("{}[]", schema_ref_name(items))
            } else {
                "any[]".to_string()
            }
        }
        "object" => "object".to_string(),
        "string" => {
            if let Some(enums) = schema.get("enum").and_then(|e| e.as_array()) {
                let vals: Vec<String> = enums
                    .iter()
                    .map(|v| format!("\"{}\"", v.as_str().unwrap_or("")))
                    .collect();
                vals.join(" | ")
            } else if let Some(format) = schema.get("format").and_then(|f| f.as_str()) {
                format.to_string()
            } else {
                "string".to_string()
            }
        }
        "integer" | "number" => "number".to_string(),
        "boolean" => "boolean".to_string(),
        other => other.to_string(),
    }
}
