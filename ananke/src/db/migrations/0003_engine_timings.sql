-- 0003_engine_timings: engine-reported phase timings on request metrics.
--
-- llama.cpp attaches a `timings` object alongside `usage` in both
-- streaming and non-streaming responses, carrying the engine-measured
-- prefill and decode durations. These are strictly better than the
-- proxy-observed `ttft_ms` (which folds in queue and network latency)
-- and, crucially, are present even for non-streaming responses where no
-- TTFT boundary can be observed.
--
-- prompt_ms:    engine prefill time (`timings.prompt_ms`), input phase.
-- predicted_ms: engine decode time (`timings.predicted_ms`), output phase.
--
-- Both are nullable — engines other than llama.cpp do not emit `timings`,
-- and those rows fall back to the TTFT-based (streaming) or aggregate
-- (non-streaming) tiers at query time.

ALTER TABLE request_metrics ADD COLUMN prompt_ms INTEGER;
ALTER TABLE request_metrics ADD COLUMN predicted_ms INTEGER;
