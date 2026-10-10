---
title: "spike: keeping a prompt cache across an unload (slot save/restore)"
date: 2026-10-10
status: done
unblocks: ["R14 batch 4 — keep a model's prompt cache across an unload"]
todo: "Re-measure after a llama.cpp major that touches llama_state_seq_* or the slot scheduler."
---

# Question

When the idle sweep or make-room stops a llama.cpp launch mid-conversation, the next
request reprocesses the whole prompt. `llama-server` can write a slot's KV cache to
disk and read it back: `--slot-save-path PATH` plus
`POST /slots/{id_slot}?action=save|restore` (present on the local build 11390 and in
upstream `tools/server/README.md`, checked 2026-05 and again 2026-10-05). Does that
beat plain reprocessing by enough to pay for the disk, and what does the engine
guarantee about a mismatched file?

# Numbers

All runs: `Llama-3.2-1B-Instruct-Q4_K_M` unless noted, one server at a time, the same
llama.cpp commit as the feature build, four runs each, request text under 20 000
tokens.

**This machine, CPU build, 8 threads, 100k-token prompt**

| Path | Time | File |
| --- | --- | --- |
| Plain reprocessing | 2 881 - 3 196 s | — |
| `action=save` | 0.71 - 0.78 s | 3.05 GiB (~32.8 KB/token) |
| `action=restore`, file in page cache | 0.52 - 0.57 s | — |
| `action=restore`, page cache dropped | 1.26 s | — |

After a restore, the follow-up request reported 99 999 prompt tokens in the cache.

At 100k tokens the win is about 3 000x: roughly 48 minutes of reprocessing becomes
about a second of file read.

**This machine, GPU build (RTX 3050 Ti, 4 GB), all layers offloaded**

| Context | KV type | Reprocess | Save | Restore | Restore (cold) | File |
| --- | --- | --- | --- | --- | --- | --- |
| 100k | default (f16) | does not load | — | — | — | — |
| 80k | default (f16) | 49.9 s | 0.69 s | 0.49 s | 1.08 s | 2.62 GB |
| 100k | `-ctk q8_0 -ctv q8_0` | 85.1 s | 0.44 s | 0.36 s | 0.74 s | 1.74 GB |

The default KV cache at 100k tokens needs 3 160 MiB on this GPU, which does not fit
in 4 GB beside the weights, so the 100k row needed the 8-bit cache. Quantising the
cache both shrinks the file and speeds up both directions.

**Faster host (Strix Halo, llama.cpp build 11457), earlier round**

| Path | CPU | GPU (one run) |
| --- | --- | --- |
| Reprocess 100k tokens | 79 - 82 s | 73.4 s |
| Save | 2.2 - 2.7 s | 0.29 s |
| Restore | 0.3 - 1.1 s | 0.68 s from disk |

File: 3.05 GiB. Same conclusion on both hosts: saving and restoring are one to two
orders of magnitude cheaper than the reprocessing they replace, and the gap grows
with the context.

# Engine behaviour checks (this machine, small contexts, llama.cpp b11525)

These are the facts the design has to respect, each checked against the real server.

- `--slot-save-path` must name an existing directory; `llama-server` refuses a
  missing one and appends its own trailing separator. Save and restore take a bare
  `filename` inside that directory (`fs_validate_filename` rejects path separators).
- `GET /slots` reports `n_prompt_tokens` only for a slot that has served a task, so a
  never-used slot has nothing to report and nothing to save. A slot that is processing
  defers the save task until it is free, so a save must not be used to wait out a busy
  slot.
- A fresh server picks the highest-numbered idle slot for its first request (the
  LRU tie-break walks all slots with equal `t_last_used`), so the conversation does
  not live in slot 0.
- A request is routed to the slot whose prompt shares the longest common prefix with
  it when that prefix is more than `--slot-prompt-similarity` (default 0.1) of the new
  prompt. Nothing has to name `id_slot`: after a restore into slot 3, the follow-up
  landed in slot 3 and reported 604 of its 627 prompt tokens as cached. Below that
  similarity the scheduler falls back to plain LRU and the cache is simply unused.
- A restore only needs the file to fit the new window: a save taken at `-c 8192`
  restored cleanly into a `-c 4096` server and produced the same cached-token count.
- A file larger than the window is refused with HTTP 400 (`No available space in KV
  cache or invalid slot save file`) and the server stays healthy and servable.
- A file saved with the default f16 cache does **not** mis-read into a
  `-ctk q8_0 -ctv q8_0` server: the engine answers HTTP 400
  (`failed to restore kv cache`) and keeps serving. The file format records only the
  model arch as a compatibility marker, so this size check is what prevents silently
  wrong output.
- A save of an idle slot costs milliseconds at small contexts (611 tokens: 6.5 ms,
  20.0 MB) and hundreds of milliseconds to a few seconds at 100k. Both directions are
  bounded by memory bandwidth, not by the model.

# What this decided

- **Save on eviction only.** The save is worth ~1 s and ~3 GiB per 100k-token slot, so
  it is paid for exactly when the alternative is a reprocess of many minutes: before
  the idle sweep or make-room stops a launch. A stop the user asked for saves nothing.
- **Restore before the launch is Ready.** The restore needs an idle slot and must land
  before the first request; doing it in the readiness probe costs about a second on a
  cold launch and makes the first request a cache hit instead of a reprocess.
- **Match by prefix, in the engine.** No conversation ids, no client hints: the saved
  slot goes back into the slot id it came from and llama.cpp's own similarity routing
  finds it. A stale cache is never wrong because only the common prefix is reused.
- **Files under the cache dir, budgeted.** `<cache_dir>/prompt-cache/llamacpp/<model>/`
  with a manifest, a per-directory byte budget (oldest first) and an age expiry, because
  3 GiB per conversation is not something to leave uncapped.
- **Compatibility keyed on what the bytes mean.** Device placement and window size do
  not invalidate a save (checked above); KV cache type, flash attention, slot count,
  speculation and the build id do. The engine's own checks are the backstop, and a
  rejected file is dropped rather than retried forever.
