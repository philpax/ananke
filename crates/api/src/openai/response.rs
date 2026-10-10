//! What ananke reads back out of an upstream engine's OpenAI-compatible
//! response body.
//!
//! The daemon proxies these bodies through untouched; it only *reads* a
//! handful of fields, for request metrics and for `anankectl`'s chat
//! surfaces. So this is deliberately a partial view: unknown keys are
//! ignored, and every field is both optional and read through
//! [`lenient`], so a field ananke cannot make sense of costs it that one
//! field and nothing else.
//!
//! That leniency is load-bearing rather than defensive. The engine
//! decides what it emits; llama.cpp has renamed counters and changed
//! their types between versions, and these bodies reach us through
//! whatever proxies sit in front of the engine. A failed parse discards
//! the whole chunk — which for the metrics recorder means silently
//! losing every counter in it, including the ones that were perfectly
//! well-formed. Read a field strictly here and that is what happens.
//!
//! One type covers both the streaming and non-streaming shapes, which
//! differ only in whether a choice carries `delta` or `message`.

use serde::{Deserialize, Deserializer};
use serde_json::Value;
use utoipa::ToSchema;

/// One `data:` chunk of a streamed completion, or a whole non-streamed
/// response body.
#[derive(Debug, Clone, Default, Deserialize, ToSchema)]
pub struct ChatCompletionChunk {
    /// Absent on chunks that carry only `usage`.
    #[serde(default, deserialize_with = "lenient")]
    pub choices: Option<Vec<Choice>>,
    /// Token counts, when the engine reported them.
    #[serde(default, deserialize_with = "lenient")]
    pub usage: Option<Usage>,
    /// llama.cpp's non-standard per-request timing block.
    #[serde(default, deserialize_with = "lenient")]
    pub timings: Option<Timings>,
}

impl ChatCompletionChunk {
    /// The first choice, if the engine sent any. ananke never requests
    /// `n > 1`, so no caller here looks past index 0.
    pub fn first_choice(&self) -> Option<&Choice> {
        self.choices.as_ref()?.first()
    }

    /// The first choice's incremental text, for a streamed response.
    pub fn delta(&self) -> Option<&MessageBody> {
        self.first_choice()?.delta.as_ref()
    }

    /// The first choice's complete text, for a non-streamed response.
    pub fn message(&self) -> Option<&MessageBody> {
        self.first_choice()?.message.as_ref()
    }
}

/// One completion candidate. Streaming responses fill `delta`,
/// non-streaming ones fill `message`.
#[derive(Debug, Clone, Default, Deserialize, ToSchema)]
pub struct Choice {
    /// Set on a streamed chunk.
    #[serde(default, deserialize_with = "lenient")]
    pub delta: Option<MessageBody>,
    /// Set on a non-streamed body.
    #[serde(default, deserialize_with = "lenient")]
    pub message: Option<MessageBody>,
}

/// The text of a choice, whether incremental or complete.
#[derive(Debug, Clone, Default, Deserialize, ToSchema)]
pub struct MessageBody {
    /// The answer text itself. `None` when the engine sent structured
    /// content parts rather than a plain string — ananke has no use for
    /// those, and reading them is not worth failing the chunk over.
    #[serde(default, deserialize_with = "lenient")]
    pub content: Option<String>,
    /// DeepSeek-style reasoning text.
    #[serde(default, deserialize_with = "lenient")]
    pub reasoning_content: Option<String>,
    /// Reasoning text under the plainer key some engines use instead.
    #[serde(default, deserialize_with = "lenient")]
    pub reasoning: Option<String>,
}

impl MessageBody {
    /// Reasoning text under whichever of the two keys the engine chose.
    pub fn reasoning_text(&self) -> Option<&str> {
        self.reasoning_content
            .as_deref()
            .or(self.reasoning.as_deref())
    }

    /// Any text at all — content or reasoning. What "the model has
    /// started answering" means for a time-to-first-token measurement.
    pub fn any_text(&self) -> Option<&str> {
        self.content.as_deref().or_else(|| self.reasoning_text())
    }
}

/// Token counts. Emitted in the final chunk of a stream when the client
/// asked for `stream_options.include_usage`, and always on a
/// non-streamed body.
#[derive(Debug, Clone, Default, Deserialize, ToSchema)]
pub struct Usage {
    /// Prompt tokens billed, cached prefix included.
    #[serde(default, deserialize_with = "lenient")]
    pub prompt_tokens: Option<i64>,
    /// Tokens the model generated.
    #[serde(default, deserialize_with = "lenient")]
    pub completion_tokens: Option<i64>,
}

/// llama.cpp's engine-reported phase timings, which sit next to `usage`.
/// Absent for engines that do not emit them.
#[derive(Debug, Clone, Default, Deserialize, ToSchema)]
pub struct Timings {
    /// Prefill duration, in floating-point milliseconds.
    #[serde(default, deserialize_with = "lenient")]
    pub prompt_ms: Option<f64>,
    /// Decode duration, in floating-point milliseconds.
    #[serde(default, deserialize_with = "lenient")]
    pub predicted_ms: Option<f64>,
    /// Prompt tokens actually evaluated — cache hits excluded, which is
    /// what makes this the right prefill-throughput numerator rather
    /// than `usage.prompt_tokens`.
    #[serde(default, deserialize_with = "lenient")]
    pub prompt_n: Option<i64>,
    /// Speculative draft tokens proposed. Present only when the engine
    /// ran speculative decoding.
    #[serde(default, deserialize_with = "lenient")]
    pub draft_n: Option<i64>,
    /// Draft tokens the target accepted.
    #[serde(default, deserialize_with = "lenient")]
    pub draft_n_accepted: Option<i64>,
}

/// Read a field as `T`, or as `None` if it is anything else.
///
/// Every field in this module goes through here, which is what keeps one
/// unexpected value from costing the caller the whole body. Buffering
/// through [`Value`] first is required rather than incidental: attempting
/// `T` directly against the caller's deserializer and then swallowing the
/// error would leave it stopped partway through the field's tokens.
fn lenient<'de, D, T>(deserializer: D) -> Result<Option<T>, D::Error>
where
    D: Deserializer<'de>,
    T: serde::de::DeserializeOwned,
{
    let value = Value::deserialize(deserializer)?;
    Ok(T::deserialize(value).ok())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_chunk_missing_every_optional_field_still_parses() {
        let chunk: ChatCompletionChunk =
            serde_json::from_str(r#"{"id":"x","object":"y"}"#).unwrap();
        assert!(chunk.first_choice().is_none());
        assert!(chunk.usage.is_none());
        assert!(chunk.timings.is_none());
    }

    #[test]
    fn reasoning_falls_back_to_the_plain_key() {
        let chunk: ChatCompletionChunk =
            serde_json::from_str(r#"{"choices":[{"delta":{"reasoning":"hm"}}]}"#).unwrap();
        assert_eq!(chunk.delta().unwrap().reasoning_text(), Some("hm"));
    }

    #[test]
    fn an_unknown_timings_counter_does_not_discard_the_known_ones() {
        let chunk: ChatCompletionChunk =
            serde_json::from_str(r#"{"timings":{"prompt_n":12,"future_counter":3}}"#).unwrap();
        let timings = chunk.timings.unwrap();
        assert_eq!(timings.prompt_n, Some(12));
        assert_eq!(timings.draft_n, None);
    }

    #[test]
    fn a_null_field_is_read_as_absent() {
        let chunk: ChatCompletionChunk =
            serde_json::from_str(r#"{"choices":null,"usage":null,"timings":null}"#).unwrap();
        assert!(chunk.first_choice().is_none());
        assert!(chunk.usage.is_none());
        assert!(chunk.timings.is_none());
    }

    /// The guarantee this module exists for: one field of an unexpected
    /// *type* costs that field, never its well-formed neighbours. Each
    /// case here is a body the metrics recorder would otherwise write to
    /// the database with every column null.
    #[test]
    fn a_wrongly_typed_field_does_not_poison_the_rest_of_the_body() {
        let cases = [
            r#"{"timings":"n/a","usage":{"prompt_tokens":7,"completion_tokens":2}}"#,
            r#"{"timings":123,"usage":{"prompt_tokens":7,"completion_tokens":2}}"#,
            r#"{"choices":"nope","usage":{"prompt_tokens":7,"completion_tokens":2}}"#,
            r#"{"choices":[{"delta":"hi"}],"usage":{"prompt_tokens":7,"completion_tokens":2}}"#,
            r#"{"usage":{"prompt_tokens":7,"completion_tokens":2,"extra":{"nested":[1]}}}"#,
        ];
        for case in cases {
            let chunk: ChatCompletionChunk = serde_json::from_str(case).unwrap();
            let usage = chunk
                .usage
                .unwrap_or_else(|| panic!("usage lost by: {case}"));
            assert_eq!(usage.prompt_tokens, Some(7), "{case}");
            assert_eq!(usage.completion_tokens, Some(2), "{case}");
        }
    }

    /// A counter whose type changed costs that counter alone. Reading a
    /// float `prompt_tokens` as `None` matches what the untyped
    /// `as_i64()` walk this replaced did with the same input.
    #[test]
    fn a_wrongly_typed_counter_costs_only_itself() {
        let chunk: ChatCompletionChunk = serde_json::from_str(
            r#"{"usage":{"prompt_tokens":10.5,"completion_tokens":3},
                "timings":{"prompt_ms":50.0,"prompt_n":"eight"}}"#,
        )
        .unwrap();
        let usage = chunk.usage.unwrap();
        assert_eq!(usage.prompt_tokens, None);
        assert_eq!(usage.completion_tokens, Some(3));
        let timings = chunk.timings.unwrap();
        assert_eq!(timings.prompt_ms, Some(50.0));
        assert_eq!(timings.prompt_n, None);
    }

    /// Structured content parts — what some proxies put in `content` —
    /// leave the text unread but must not cost the chunk its usage.
    #[test]
    fn structured_content_parts_leave_the_rest_intact() {
        let chunk: ChatCompletionChunk = serde_json::from_str(
            r#"{"choices":[{"delta":{"content":[{"type":"text","text":"hi"}]}}],
                "usage":{"prompt_tokens":1,"completion_tokens":1}}"#,
        )
        .unwrap();
        assert_eq!(chunk.delta().unwrap().content, None);
        assert_eq!(chunk.usage.unwrap().prompt_tokens, Some(1));
    }
}
