-- 0004_prompt_eval_tokens: engine-reported evaluated prompt token count.
--
-- llama.cpp's `timings` object carries `prompt_n`, the number of prompt
-- tokens actually evaluated during prefill. This EXCLUDES tokens served from
-- the KV cache, whereas `usage.prompt_tokens` counts the full billed prompt
-- including the cached prefix. Dividing the billed count by the (cache-aware)
-- `prompt_ms` massively overstates prefill throughput, so the input and
-- aggregate TPS numerators use this evaluated count when present.
--
-- Nullable — engines other than llama.cpp (and rows recorded before this
-- migration) leave it null and fall back to `prompt_tokens` at query time.

ALTER TABLE request_metrics ADD COLUMN prompt_eval_tokens INTEGER;
