---
title: "spike: llama.cpp slot save/restore vs prompt reprocess at 100k tokens"
date: 2026-10-10
status: complete
unblocks: ["R14 batch 4: keep a model's prompt cache across an unload"]
---

# Finding

**Save plus restore beats reprocessing by roughly three orders of magnitude at 100k tokens: 0.71 s to save and 0.52 s to restore versus 2,881 s to reprocess the same prompt on CPU. A restored server answers a returning 100k-token prompt identically to one that never lost the cache (same `cache_n`, same request time). Verdict: go.**

## Setup

- llama.cpp `0.6.0-dev (build 11525, commit 609290be6)`, CPU-only build, `-t 8`, one server at a time. The plan item was written against local build 11390 (checked 2026-10-05); the slot save/restore API is unchanged on the current build.
- Model: `Llama-3.2-1B-Instruct-Q4_K_M.gguf` (1.2B Q4_K_M), `-c 100000`. The engine auto-sized 4 slots, each `n_ctx` 100096 on a unified KV pool.
- Prompt: a 100,000-token document (tokenized exactly against this model's `/tokenize`) sent as the system message, 8 generated tokens, `temperature: 0`.
- `--slot-save-path` on a real filesystem (NVMe, ext4), never tmpfs. Server B is a fresh process of the same command line.
- Method, in order, one driver run: server A answers the full prompt cold → the identical prompt is sent again to the same server (in-process control) → `POST /slots/{id}?action=save` for all 4 slots → server A is stopped (SIGTERM, exits in 0.06 s) → server B starts → `action=restore` per slot file → the same prompt is sent as the follow-up, then an 8-token-longer variant. Every timing below comes from the server's own `timings` object or wall clock around the request; no client-side timeouts were set (a 1800 s read timeout cancelled the first attempt at 80% mid-run).

## Numbers

| Step | Result |
| --- | --- |
| Cold prompt process, 100,008 tokens, fresh slot | 2,881.2 s (`prompt_ms` 2,881,234), `cache_n` 0, average 34.7 tok/s; rate decays from about 77 tok/s at short contexts to about 34 tok/s near 100k |
| Same prompt again, same server (in-process control) | wall 8.3 s, `cache_n` 100,007, `prompt_n` 1 |
| `action=save` on the busy slot | 709.7 ms, `n_saved` 100,015 tokens, `n_written` 3,278,892,196 bytes (3.05 GiB) |
| `action=save` on the 3 empty slots | 0.2-0.3 ms, `n_saved` 0, 36-byte file each |
| Stop server A (SIGTERM) | 0.06 s |
| Server B boot to healthy | 1.5 s |
| `action=restore` on slot 3 | 516.4 ms, `n_restored` 100,015 |
| Follow-up, same prompt, restored server | wall 8.3 s, `cache_n` 100,007, `prompt_n` 1: byte-for-byte the in-process control |
| Follow-up, prompt extended by 8 tokens | `cache_n` 100,008, `prompt_n` 8 (3.8 s), the engine's LCP routing matched the restored slot (`selected slot by LCP similarity`) |

Whole cycle for a 100k-token conversation: **save 0.71 s before the stop + restore 0.52 s on the next launch = 1.2 s**, against **2,881 s** of reprocess without it. The restored server reproduces the never-evicted behavior exactly; only the 8 tokens beyond the saved prefix are reprocessed on the extended follow-up.

## Sizing

The save file is the KV state only (plus a 36-byte header and per-slot checkpoint bookkeeping): 3,278,892,196 bytes / 100,015 tokens = 32,783 B/token, matching the model's geometry exactly (16 layers x 8 KV heads x 64 head-dim x K+V x f16 = 32,768 B/token). Budget 32 KB per cached token per slot when sizing the cap.

## Caveats

- The restore read its 3.05 GiB file with it warm in page cache (written 1 s earlier, same boot), which is the eviction scenario: evict, restart within minutes. A cold NVMe read of the same file would add roughly 1-2 s; three orders of magnitude below the reprocess it replaces.
- llama.cpp cancels an in-flight task when the HTTP client disconnects (observed directly: a client timeout at 79,872/100,000 tokens produced `cancel task` server-side). Long prompt calls must not carry client-side read timeouts.
- A released slot keeps its KV between requests on a live server, so a "cold" rerun against a server that already saw the prompt is not cold. The driver wipes the save dir and restarts the server between every phase; reruns need to.
- Save and restore only touch idle slots; a busy slot defers the action (upstream behavior). llamastash saves only idle slots, and restores before publishing `Ready`, while slots are necessarily idle.

## Decision

Go. Built as `backend.llamacpp.slot_cache` (off by default), saving idle slots on eviction stops and restoring before `Ready`. See `docs/architecture.md` § Model lifecycle and `docs/usage.md` § Prompt cache across unloads.
