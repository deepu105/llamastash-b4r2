---
title: "feat: R14 (v0.7.0) batch plans"
type: feat
status: active
date: 2026-10-05
origin: R14 checklist in TODO.md; feature review 2026-10-05
---

# feat: R14 (v0.7.0) batch plans

**Status:** active. Six mini plans, one per batch of the R14 checklist in
[`TODO.md`](../../TODO.md). The item details live here; `TODO.md` keeps one
line per item.

Batches group items by the code and docs a session has to load. One batch per
session, in order. Tick an item here and in `TODO.md` in the same change.

## Batch 1: vision detection bug

**Context:** `src/discovery/scanner.rs`, `src/discovery/metadata_cache.rs`. The cause is not known: reproduce on the real cache and add the regression test before the fix.

**Result (2026-10-08):** Cause was neither suspect listed below. `mmproj-BF16.gguf` /
`mmproj-F16.gguf` strip to a bare dtype, `canonical_base` counts `bf16`/`f16` as a quant
(`QUANT_PATTERN`), so both land in the nameless tier, which accepted only a lone catch-all.
An empty cache dir reproduced it, and the daemon already logged the ambiguity: `2 mmproj
candidates found but none match …`. `pick`'s nameless tier now ranks by
`scanner::companion_precision_rank` — the order `pull` already used in
`download::pick_one_companion`, so pull and discovery agree on the file. Cross-directory
pairing takes the same rank, which is what the shard-subdir rows needed.

Follow-up (same day): the two picks now share one comparator, `scanner::companion_order`,
ranked on the basename so a precision token in a directory name cannot pull them apart.
A delete now takes the unpaired precisions with the model instead of leaving them behind.

### Plan

1. Reproduce with the working tree against the real cache, in an isolated state dir: `list --json` for the two unsloth rows and for the gufo row that works.
2. Check the first suspect. `find_mmproj` ([`src/discovery/scanner.rs`](../../src/discovery/scanner.rs)) returns `None` when several projectors are equally plausible, and both failing snapshots hold `mmproj-BF16.gguf` and `mmproj-F16.gguf`. Rule the stale metadata cache in or out by rescanning with an empty cache dir.
3. Add the failing tests first: a fixture dir with two projectors beside the model, and one with the projectors one level above a shard subfolder.
4. Fix. If the tie is the cause, decide between a tie-break (pick one projector by a fixed order) and reporting `vision: true` while leaving the projector choice to `--mmproj`.
5. Re-run `integrations pi` into an isolated config dir and confirm the image input is kept. Update `docs/usage.md` if the pairing rule changes.

### Items

- [x] **Vision not detected on unsloth's Qwen3.8 rows, so `integrations` drops their image input.** On 0.6.0 (2026-10-02), `list --json` shows `multimodal: null` for `Qwen3.8-27B-UD-Q6_K.gguf` and `Qwen3.8-Flash-Next-UD-Q4_K_XL.gguf`, though both snapshots hold `mmproj-BF16.gguf` and `mmproj-F16.gguf` (27B beside the model, Flash-Next one level above its `UD-Q4_K_XL/` shards). `vmlinux/Qwen3.8-Flash-Next-Uncensored-Gufo-Q4Mix-GGUF`, with one `mmproj-BF16.gguf` beside the model, shows `vision: true`. Re-running `integrations pi` replaced the 4 pi entries and dropped their hand-added `"input": ["text", "image"]`. Cause not found yet; suspects are two projectors in one folder and a stale metadata cache.

## Batch 2: proxy model resolution and listing

**Context:** `src/proxy/route.rs`, `src/discovery/catalog.rs`, `src/proxy/openai.rs`, `src/proxy/ui.rs`, `resolve_model_id_and_arch` in `src/ipc/methods.rs`; `docs/usage.md` § Model ids on the proxy and § Web UI. Do R-08 first, so aliases build on the new catalog read.

### Plan

1. R-08: `ModelCatalog::snapshot()` ([`src/discovery/catalog.rs`](../../src/discovery/catalog.rs)) clones the whole `Vec`, and `src/proxy/route.rs` calls it at four sites. Hand out a shared `Arc` instead. An `Arc<Vec<..>>` swapped on write needs no new dependency (`arc-swap` is not in `Cargo.toml`). No behaviour change; the existing route tests are the gate.
2. Aliases: settle the three open questions in the item and write the answers into `docs/usage.md` § Model ids on the proxy. Then add `proxy.aliases` to `ProxyConfig` ([`src/config/loader.rs`](../../src/config/loader.rs)) and `config.example.yaml`, and resolve an alias before the normal id lookup.

   Settled 2026-10-05: aliases are **not** listed on `/v1/models` or `/api/tags`; a **real id wins**, so the alias is consulted only after the name the client sent matches no model (the plan's earlier "before the normal id lookup" ordering would have let an alias hide a real model, which is what the decision rules out); and an alias **names a model only**, so a target cannot pin a preset or a launch name. The first shadowed alias logs one warning.
3. `/ui`: `chooser_html` ([`src/proxy/ui.rs`](../../src/proxy/ui.rs)) renders the running launches. Carry a has-UI flag on `RunningEntry` from the backend's `serves_web_ui` and colour the rows by it. A row without a UI is not a link.
4. R-12: wrap the header read in `resolve_model_id_and_arch` (`src/ipc/methods.rs`) in `spawn_blocking`, the way `proxy::launch::canonical_id_for_row` already does.
5. E2E on an isolated daemon: a request by alias reaches the model, `/v1/models` shows what was decided, and the `/ui` chooser shows both colours. One commit per item.

### Items

- [x] **Proxy perf (R-08)**: Replace the per-request `Vec<CatalogRow>` clone in `proxy::route::decide` with an `ArcSwap<Vec<CatalogRow>>` pre-built by the discovery task. Today every inbound `/v1/...` request walks the catalog snapshot and allocates a fresh `CatalogRow` per row before handing it to the resolver. Origin: PR #7 ce-review (R-08), deferred from this PR's scope. Needs catalog publish-side wiring (`ModelCatalog::publish_view()` → `ArcSwap` slot read by the proxy). Done as `ModelCatalog::shared_rows()` / `shared_view()`: an `Arc<Vec<…>>` rebuilt under the catalog's write lock, no new dependency.
- [x] **Model aliases on the proxy.** A `proxy.aliases` map in `config.yaml` (`gpt-4o-mini: <model-ref>`) so a tool with a hard-coded model name reaches a local model. Today the resolver accepts every spelling of a real id ([`docs/usage.md` § Model ids on the proxy](../usage.md#model-ids-on-the-proxy)) but no arbitrary name. Decide: whether aliases show on `/v1/models` and `/api/tags`, what happens when an alias equals a real id, and whether an alias can pin a preset.
- [x] On the /ui endpoint show accessible models in green and models with no UI in red. Done in the chooser: the row name is green when the running model's backend serves a web UI, red when it does not (that row stays a non-link), with the legend in the page copy.
- [x] **Proxy stability (R-12)**: Move the GGUF header read inside `ipc::methods::resolve_model_id_and_arch` onto `spawn_blocking`. Today the call is invoked from async IPC handlers but does up to ~16 MiB of synchronous file I/O on the tokio worker, which can stall a worker thread under concurrent IPC load. The proxy-side call site was already fixed in this PR (`proxy::launch::canonical_id_for_row` via `spawn_blocking`); the IPC site is the remaining gap. Origin: PR #7 ce-review (R-12 partial). Done by making `resolve_model_id_and_arch` itself `async` and blocking-threading the read, so no future async caller can reintroduce the stall.

## Batch 3: proxy request log and speed stats

**Context:** `src/proxy/forward.rs`, `src/proxy/router.rs`, `src/proxy/state.rs`, `src/ipc/methods.rs`; for the tab, `src/tui/tabs/mod.rs` (`tabs_for_mode`), `src/tui/app.rs` (`available_right_tabs_uncached`), `src/tui/right_pane.rs`, `src/tui/list_pane.rs` (`Column` / `layout_columns`), `src/tui/events.rs` (`spawn_logs_poller`); `docs/architecture.md` § IPC surface and § Right pane tabs. Request log first, then the tab, then the body tap, then speed stats. Run after batch 2: both are likely to change `ProxyState`.

### Plan

1. Request log with clock fields only: a bounded ring buffer, filled where the proxy dispatches a request and where `GuardedBody` drops ([`src/proxy/forward.rs`](../../src/proxy/forward.rs)). No body parsing yet. Each row carries the resolved model path and the launch id when one exists, so readers can filter by model. The log sits on `MethodContext`, not `ProxyState`, because the IPC handlers read it and only see the context.
2. Expose it: `requests_tail` in the `src/ipc/methods.rs` dispatch table and `llamastash requests [<model>]` with `--json`. The method takes an optional `model_path` and returns `{ summary, requests }`, with the summary computed in the daemon. Document both in `docs/usage.md` and `docs/architecture.md` § IPC surface.
3. TUI `Requests` tab with the clock fields, per the design below. Token fields render `—` until step 5.
4. Before the body tap, send one streamed and one non-streamed request to llama.cpp, Halogen and gufo and record which of `timings` / `usage` the last chunk carries. Done for llama.cpp only, see Result.
5. Body tap: `GuardedBody` keeps the head and the tail of the response and reads `timings` / `usage` / `metrics` by field name at end of stream ([`src/proxy/usage_tap.rs`](../../src/proxy/usage_tap.rs)). Token counts and tok/s land on the log rows; the tab's token columns and tok/s summary fill in.
6. Speed stats: add the per-launch summary to `status` as `request_stats`, with the same fields as the tab's top strip. The scopes differ: the tab totals a model's requests since the daemon started, `status` totals one launch's and drops them when the launch stops. The numbers match while a model has had one launch and no fallback traffic.
7. Measure proxy overhead with and without the tap using the proxy bench in `scripts/bench/`. Not run, see Follow-ups. Check the tab with `--render` and the pty driver and update the golden snapshots.

### Requests tab

- **Placement.** A `Requests` variant in `RightTab` (`src/tui/tabs/mod.rs`), last in `tabs_for_mode`: `Settings, Logs, Chat|Embed|Rerank, Requests`. A delegated multiplexer model keeps it (only `Settings` is filtered out there). `Launching`, `Loading`, `Error` and `Stopped` rows get the tab too: a failed auto-start leaves the row in `Error` and its 503 rows are what the log is for. Tabs are reached by a Shift-letter jump, and `R` is Rerank's, so Requests underlines the `q` of its label and takes `Shift+Q`.
- **Filter.** Only the focused model's requests, matched on the model's catalog path, not launch id, so a request that never got a launch (a 503 at auto-start) still shows. No Model column.
- **Data.** Poll `requests_tail` with the focused model through the same loop the Logs tab uses (`spawn_tab_poller`), and only while the tab is open. One call returns the summary and the rows.
- **Top strip: summary.** Requests, errors (4xx / 5xx), average total time, average time to first byte, tok/s (average and last), prompt tokens, generated tokens, auto-starts and evictions triggered. The TUI renders the daemon's summary and does not recompute it from the visible rows. The summary counts finished requests since the daemon started, so it keeps counting after a row leaves the ring.
- **Table.** Below the strip, newest first, scrolled with the existing right-pane scroll actions. Key labels come from `KeyMap`. Empty state: one line saying the model has no requests yet.
- **Ranked columns.** Each column has a fixed width and a rank; lower rank stays longer as the pane narrows. Lift the picker out of `src/tui/list_pane.rs` (`Column { label, width, rank }` plus the strict rank-tail drop in `layout_columns`) into a shared helper and use it for both tables. Declaration order is display order. `Note` is a ranked column like the others; when it shows it also takes the leftover width, like `Name` in the list pane.

  | Column | Width | Rank | Content |
  |---|---|---|---|
  | Time | 8 | 10 | `HH:MM:SS`, local |
  | Code | 4 | 20 | HTTP status, coloured by class |
  | Tok/s | 6 | 30 | generation speed |
  | Total | 7 | 40 | total time |
  | In | 6 | 50 | prompt tokens |
  | Out | 6 | 60 | generated tokens |
  | TTFB | 7 | 70 | time to first byte |
  | Route | 16 | 80 | the route without `/v1/`, e.g. `chat/completions` |
  | Client | 21 | 100 | client address (`ip:port`) |
  | Note | 12 + leftover | 90 | auto-start, eviction, or the 503 cause |

  These are the shipped values, re-ranked on 2026-10-06. Display order is the keep order, so columns go from the right as the pane narrows. `Client` ranks below `Note` and goes first, but sits before it on screen so `Note` stays last and can flex.
- **Tests.** `tabs_for_mode` and `available_right_tabs` order, the column picker at three widths, and the golden snapshots.
- **Docs.** `docs/usage.md` and `docs/architecture.md` § Right pane tabs. That table lists `Logs, Chat` for a ready chat model and omits `Settings`, which the code returns first; fix it in the same change.

### Items

- [x] **Proxy request log.** Keep the last N proxy requests in memory (time, client address, route, model, launch, HTTP status, latency, time to first byte, the auto-start or eviction it triggered) and show them in a per-model `Requests` tab in the TUI (design under Requests tab above) and a CLI command with `--json`. It answers "why did my agent get a 503" without reading daemon logs. No request log exists today. Token counts and tok/s per request need the response body: llama-server puts a `timings` object in it, and the proxy streams bytes through untouched ([`src/proxy/forward.rs`](../../src/proxy/forward.rs) `GuardedBody`), so that part is a tap on the body wrapper and can ship second. Store no prompt or response text. Nothing request-shaped exists today to build on: no counters or log ring in the proxy, no `/metrics` route, and `DaemonState` (`src/daemon/state_store.rs`) has no request field, so an in-memory ring owned by `ProxyState` is the starting shape. Hook points for one shared tap: `GuardedBody` in `src/proxy/forward.rs` (status, latency, time to first byte, later the `usage` object), `route::decide` for the routing decision, and `eviction.rs` `make_room` plus `failure_tracker.rs` for the 503 causes, which are the answer to the "why did my agent get a 503" question this exists for. Ship metadata first, tokens second, per the speed-stats entry above.
- [x] **Live speed stats per model.** Show tok/s, time to first token and prompt/generated token counts for running models in the summary strip of the TUI `Requests` tab and in `status --json`. Read them from the response as it passes through the proxy, the same body tap the proxy request log needs: (1) the proxy's clock for time to first byte and total time, (2) the OpenAI `usage` object for token counts, (3) the server's own speed fields when present: `timings` (llama.cpp, Halogen), `usage.*_tokens_per_second` (gufo), `metrics` (vLLM with `--enable-per-request-metrics`). Match on field names, not backend ids. Open: whether Halogen and gufo put `timings` / `usage` in the last streamed chunk (only non-streaming is checked, in `scripts/bench/qwen38-flash-speed/bench.py`). A streamed OpenAI response carries `usage` only when the client sends `stream_options.include_usage: true`, and the proxy should not add it by default. Optional second source for requests that skip the proxy: poll `/metrics` (llama.cpp needs `--metrics`; Halogen 0.16.2 serves the same `llamacpp:*` names; vLLM `vllm:*`; SGLang `sglang:*` with `--enable-metrics`). Checked 2026-10-05. Confirmed absent today, so the tap has no prior art in-tree: no read of `usage` / `prompt_tokens` / `completion_tokens` anywhere in `src/proxy`, and no `/metrics` route on either server. Two notes for whoever picks it up: the same tap is what the **Proxy request log** entry below needs, so design the `GuardedBody` hook once for both; and `daemon.metrics_interval_secs` is the host sampler (CPU/GPU/temp), so name this surface something else (`launch_stats`, `speed`) to keep the two apart. Nothing request-shaped is persisted: `DaemonState` (`src/daemon/state_store.rs`) holds favorites, `last_params`, `running` and `schema_version` only, so totals that survive a restart need a new field there or they die with the daemon.

### Result (2026-10-05)

Shipped: the log, `requests_tail`, `llamastash requests`, the `Requests` tab, the body tap, and `request_stats` on `status`.

Response shapes, recorded from llama-server b11390 (upstream was at b11417) with `Llama-3.2-1B-Instruct-Q4_K_M` on CPU:

| Route | Streamed | `usage` | `timings` |
|---|---|---|---|
| `/v1/chat/completions`, `/v1/completions` | no | yes | yes |
| `/v1/chat/completions` | yes | only with `stream_options.include_usage` | yes, in the last chunk |
| `/v1/messages` | no | yes (`input_tokens` excludes `cache_read_input_tokens`) | no |
| `/v1/messages` | yes | `message_start` has the input tokens, `message_delta` the output tokens | no |
| `/v1/responses` | no | yes (`input_tokens` includes cached tokens) | no |
| `/v1/responses` | yes | yes, in `response.completed` | yes, in `response.completed` |

So on a stream without `usage` the token counts come from `timings` (`prompt_n + cache_n`, `predicted_n`), and the tap keeps the head of the response as well as the tail because an Anthropic stream reports its input tokens in the first event.

The vLLM field name (`metrics.tokens_per_second`, sent with `--enable-per-request-metrics`, on a stream only in the final usage chunk) is from the upstream source on 2026-10-05 (`vllm/entrypoints/generate/base/protocol.py`, release v0.31.0), not from a live server. gufo's `usage.completion_tokens_per_second` is from `scripts/bench/qwen38-flash-speed/bench.py`, non-streamed only.

`proxy.request_log_file: true` (added 2026-10-05, off by default) also appends each finished row to `<log dir>/requests.jsonl`. The in-memory log is not read back from it.

Halogen 0.16.2, read from `tools/serve_api.py` in the image of a running container on 2026-10-06:

- A streamed chat or completions response puts `timings` on the finish chunk whether or not usage was asked for. With `stream_options.include_usage` one more chunk follows, carrying `usage` and `timings`, then `[DONE]`.
- Non-streamed chat, `/v1/messages` and `/v1/responses` carry both `usage` and `timings`.
- `timings` has llama-server's shape and meaning: `prompt_n` is what the engine processed, `cache_n` the reused prefix, `predicted_per_second` the engine's own decode rate. `cache_n` is only present when the engine reports a cached count.

So the tap needs nothing Halogen-specific. A live Halogen run through the proxy the same day showed it: 69 streamed `chat/completions` rows with server-reported tok/s (41.6 to 46.4, no `~`) and both token counts.

gufo, the local build from the 0.7.1 checkout (`96a4647`; upstream was at v0.8.1), probed on its own port with `Qwen3.8-27B-UD-Q6_K` on 2026-10-06:

- A streamed chat or completions response puts `timings` on the last chunk, with or without `stream_options.include_usage`. With it, one more chunk carries `usage`, which has `completion_tokens_per_second`.
- Non-streamed chat, `/v1/messages` and `/v1/responses` carry `usage` and `timings`. A streamed `/v1/responses` has both in `response.completed`.
- `timings` uses llama-server's field names (`prompt_n`, `cache_n`, `predicted_n`, `predicted_per_second`).
- `/v1/messages` rejects `stream: true` with a 400, and `/v1/messages/count_tokens` is not implemented.
- On `/v1/messages` it reported a fully cached 58-token prompt as `input_tokens: 58` and `cache_read_input_tokens: 58`. llama.cpp reports the same case as `input_tokens: 1` and `cache_read_input_tokens: 40` for 41. Adding the cache fields to `input_tokens` therefore double-counted on gufo. The tap now prefers `timings` (`prompt_n + cache_n`) over `input_tokens` for the prompt size.

The build tested is two releases behind. The upstream source on 2026-10-06 (`src/cli/serve/http_server.cpp`, `openai_chat.cpp`) still sets the `/v1/messages` `input_tokens` to the whole prompt beside `cache_read_input_tokens`, and still puts `timings` on a stream's last chunk whether or not usage was asked for. The stream and `count_tokens` refusals were only seen on the local build.

Proxy overhead, 2026-10-06, on `deepu-flowz13-arch` (AC power, `performance` profile, nothing else on the GPU). Two release builds on isolated daemons: `main` at `a5ad9beb` (no request log) and this branch at `acf89d02`. Model `gemma-4-E2B-it-Q4_K_M` on llama-server b11390, three rounds per build, alternating builds.

| Measure | `main` | this branch |
|---|---|---|
| TTFT, proxy minus direct (Suite C, 20 reps a round) | +1.34, +3.42, -0.47 ms | +3.12, -0.92, +0.73 ms |
| Decode tok/s lost through the proxy (Suite C) | -0.01, -1.18, +1.37 % | +1.63, -2.31, +0.89 % |
| Daemon CPU per request (240 streamed chats, 8 at a time, 64 tokens each) | 3.92, 4.00, 3.96 ms | 4.21, 4.33, 4.38 ms |

Suite C shows no difference between the builds: both sit inside its noise (TTFT varied by 12 to 16 % of about 39 ms within a round, decode by 3 to 4 %). The request log and tap cost about 0.35 ms of daemon CPU per request, roughly 9 % more than before, on requests that took about 3.1 s each. The raw reports are under `target/bench-batch3/out/` and were not added to `docs/benchmarks/`.

### Follow-ups

- [x] Run the proxy overhead bench with and without the tap (plan step 7). Run 2026-10-06, see below.
- [x] Check which of `timings` / `usage` Halogen puts in the last streamed chunk. Checked 2026-10-06 against Halogen 0.16.2, see below.
- [x] Check the same for gufo. Checked 2026-10-06, see below.
- [x] Estimate tok/s from the proxy clock where the server reports none (`/v1/messages`, non-streamed `/v1/responses` on llama.cpp). Decided 2026-10-05: estimate, for any backend. The row carries `tokens_per_second_estimated` and the tables mark the value with `~`. The clock starts at the first response byte for a stream and at the upstream send for a response that arrives whole.

## Batch 4: faster reloads

**Context:** `src/daemon/supervisor.rs`, `src/daemon/launch_service.rs`, `src/daemon/preload.rs`, `src/proxy/eviction.rs`; `docs/architecture.md` § Model lifecycle. Both items need timed loads of a large model on real hardware, so measure both in one loaded-model session before writing code. Largest batch: one session per item.

### Plan

1. Check host load, then run one measurement session on a large model: cold load time, load time with the weights in page cache, prompt reprocess time at about 100k tokens, slot save time, restore time and save file size. Write the numbers to a dated page in `docs/spikes/`.
2. Decide go or no-go per item from the numbers. The prompt cache is worth building only if save plus restore beats reprocessing.
3. Warm: `posix_fadvise(POSIX_FADV_WILLNEED)` through the existing `libc` dependency on Linux, `PrefetchVirtualMemory` through `windows-sys` on Windows, a bounded read thread on macOS. Ship `llamastash warm <model> --json` first, then `daemon.preload_warm`, with the guards the item lists.
4. Prompt cache: keep `--slot-save-path` and the save and restore calls in `src/backend/llama_cpp/`, behind trait hooks (before stop, after ready) that default to no-op, so `make_room` and `sweep_once` in `src/proxy/eviction.rs` stay backend-neutral. Decide file location, size cap, cleanup, and how a restored slot is matched to the returning conversation.
5. Docs: `docs/usage.md`, `docs/architecture.md` § Model lifecycle and `config.example.yaml`. Real-hardware check per `docs/testing/hardware-uat.md`.

### Items

- [x] **Keep a model's prompt cache across an unload.** When make-room or the idle sweep stops a llama.cpp launch mid-conversation, the next request reprocesses the whole prompt. llama-server can write a slot's KV cache to disk and read it back: `--slot-save-path PATH` plus `POST /slots/{id_slot}?action=save|restore` (present on local build 11390 and in upstream `tools/server/README.md`, checked 2026-10-05). Nothing in `src/` uses it. Spike first: measure save time, restore time and file size at a large context (100k tokens) against plain reprocessing. Then decide when to save (before an eviction only), where the files live, how they are capped and cleaned up, and how a restored slot is matched to the returning conversation. Done 2026-10-10 as `backend.llamacpp.slot_cache` (off by default): spike in `docs/spikes/2026-10-10-prompt-cache-slot-save.md` (1.2 s save+restore vs 2,881 s reprocess at 100k tokens → go); saves idle slots from `eviction::stop_launch` through a `prepare_to_stop` trait hook, restores from the probe's `on_launch_ready` hook before `Ready`, files under `cache_dir/slot-cache/<model-hash>/` capped by `slot_cache_max_bytes` (LRU by mtime) and `slot_cache_ttl_days`; slot matching is the engine's own LCP prompt-similarity routing.
- [ ] **Warm the page cache before a load (`llamastash warm <model>`).** Weight reads dominate reload time: `docs/usage.md:855` measured 93-105 s cold against about 6 s with the weights already in page cache, and the loading keepalive entry in `TODO.md` has a cold 104 GB GGUF at roughly 130 s. Ask the kernel to read a model's shards and mmproj ahead of the launch: `posix_fadvise(POSIX_FADV_WILLNEED)` on Linux (and FreeBSD), `PrefetchVirtualMemory` on Windows, and a bounded buffered read thread on macOS, which has no `posix_fadvise`. Nothing in `src/` calls `posix_fadvise` or `readahead` today. Page cache is reclaimable, so warmed bytes need no admission reservation; the guard is to refuse rather than warm past the free pool, and to skip while a launch is loading, or the read evicts what a running launch holds resident. Sits beside the KV item above rather than duplicating it: that keeps the prompt cache, this covers re-reading weights, and a `--slot-save-path` file wants the same warming on restore. Surfaces: `llamastash warm <model> --json`, and `daemon.preload_warm:` as the RAM-only sibling of `daemon.preload` (no server spawned).

## Batch 5: CLI setup commands

**Context:** `src/cli/cli_args.rs`, `src/init/doctor.rs`, `src/util/file_security.rs`, `src/daemon/lockfile.rs`; `docs/usage.md` § Setup subcommands.

### Plan

1. Completions: add `clap_complete` and confirm it builds with clap's `default-features = false`. `llamastash completions <bash|zsh|fish>` prints the script. Static completion only in this pass; model refs stay uncompleted. Decide whether `--json` applies to a command that prints a script.
2. Add the install snippets to `docs/usage.md` and `README.md`, and try each generated script in a real bash, zsh and fish.
3. `doctor --fix`: give a finding an optional fix action in `src/init/doctor.rs`. Start with the two repairs that need no new check: `chmod 0600` on the config (`check_config_mode_drift`), and removing `runtime.json` / `daemon.pid` when `src/daemon/lockfile.rs` shows no live holder.
4. `--dry-run` prints the same list without acting, `--json` reports each fix and its result, and `doctor` keeps exiting `0`.
5. The third repair (a tool config pointing at a dead proxy port) needs a new check first. Do it last or split it out.
6. Tests with `unique_temp_dir`, then E2E on an isolated state dir with a wrong-mode config and a stale pidfile. Update `docs/usage.md` § `llamastash doctor` and the read-only wording on `Command::Doctor`.

### Items

- [x] **Shell completions.** No generator anywhere and `clap_complete` is not a dependency, so nothing completes `llamastash` in any shell. `llamastash completions <bash|zsh|fish>` plus the eval snippets in `docs/usage.md`. Static subcommands, flags and enum values (install methods, modes, backends) come free from the clap spec; model refs are dynamic, so either leave them uncompleted in the first pass or wire clap's dynamic completion to `list --json`, which needs a live daemon.
- [x] **`doctor --fix` for findings with a safe mechanical repair.** `doctor` is documented read-only (`Command::Doctor` in `src/cli/cli_args.rs`). It already detects what a fix would act on: `check_config_mode_drift` in `src/init/doctor.rs` flags a non-`0600` config and the directory swap surface through `util::file_security::dir_swap_surface`, and `src/daemon/lockfile.rs` already distinguishes a live holder from a stale pidfile. Fixable without touching a running daemon: `chmod 0600` the config, remove `runtime.json` / `daemon.pid` when no flock holder exists, and (needs a new check first) re-point a tool config whose proxy URL names a port the proxy no longer listens on. Print every change, take `--dry-run`, never delete a model or rewrite live daemon state.

## Batch 6: small standalone items

Independent of each other and of the batches above; each fits a short session. Context for the bell: `src/cli/pull.rs`, `src/tui/hf_pull.rs`. The login-service item is docs only (no cargo runs); it needs a live systemd run.

### Plan

1. Bell: add one config key (name and default to decide) to `Config` (`src/config/loader.rs`) and `config.example.yaml`.
2. Bell, CLI: write `\x07` to stderr when `pull` ends (`src/cli/pull.rs`), only on a TTY and never with `--json`.
3. Bell, TUI: ring when a pull finishes or fails (`src/tui/hf_pull.rs`) and when a launch the TUI started turns ready or failed. Test the trigger logic, then listen in a real terminal.
4. Login service: write a systemd user unit around `llamastash daemon start --foreground`. Cover `Environment=` for `HF_HOME` and `PATH` (a user unit does not read the shell profile), `TimeoutStopSec` above the longest child stop grace, and `loginctl enable-linger` for start at boot.
5. Login service: run the unit live with an isolated `LLAMASTASH_STATE_DIR` and non-default ports. Confirm `systemctl --user stop` unloads the models: the daemon handles SIGTERM (`src/daemon/shutdown.rs`), and systemd's default `KillMode=control-group` kills whatever is left in the unit.
6. Login service: write the launchd agent from Apple's `launchd.plist` docs and mark it untested until it has run on a Mac.

### Items

- [x] **Terminal bell when a pull or a long model load finishes or fails.** Bell only (`\x07`), behind a config key. No escape-sequence or OS notifications. Pulls run in the CLI/TUI process (`src/cli/pull.rs`, `src/tui/hf_pull.rs`), so they ring directly. Loads run in the daemon, so the TUI rings when it sees the launch turn ready or failed.
- [ ] **Document running the daemon as a login service.** A `docs/usage.md` section with a systemd user unit (Linux) and a launchd agent (macOS) around `daemon start --foreground`, so the daemon and `preload` models come up at login. Docs only, no `daemon install` subcommand. Run both units live before writing them down, including a clean stop (models unloaded) on `systemctl --user stop`. **The section shipped in #104; the launchd half of the run-it-live clause is unmet.** The systemd unit ran on systemd 262 (preload ready, clean stop, restart after `kill -9`). The plist is transcribed from `launchd.plist(5)` and marked untested in the doc, and still needs a Mac; tracked in `TODO.md`.
