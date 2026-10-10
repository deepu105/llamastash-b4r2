# Spike: slot prompt-cache persistence across unloads (R14)

Date: 2026-10-10
Status: complete — feeds plan `2026-05-27-003` (TODO R14 "prompt cache survives unload")

## Question

When the idle sweep or make-room stops a llama.cpp launch mid-conversation, the
next request reprocesses the whole prompt. llama-server can dump and reload a
slot's KV state (`--slot-save-path` + `POST /slots/{id}?action=save|restore`).
How much time does that actually save, what does it cost on disk, and what wire
behavior does the implementation need to know about?

## Setup

- llama.cpp `b11524` (`609290be`, `build-cpu` tree at `/mnt/work/Workspace/llms/llama.cpp`), `llama-server --version` output recorded at run time.
- Model: `Llama-3.2-1B-Instruct-Q4_K_M.gguf` (16 layers, 8 KV heads, head dim 64, K/V fp16 → 32,768 B/token of KV).
- `llama-server -c 102400 -t 8 --slot-save-path <dir>` — CPU only, 8 threads, one server at a time, ~54 GB free RAM.
- Prompt: lorem-ipsum sentence blocks tokenized up to the token budget, sent as one `/v1/chat/completions` request; `usage.prompt_tokens` and `timings` read from the response.

The driver script ran two server lifetimes: boot #1 → send the long prompt
(full reprocess) → `action=save` → stop → boot #2 → `action=restore` into the
same slot id → send a 3-token follow-up and read `usage.prompt_tokens_details.cached_tokens`.

## Numbers (100k context)

| Step | Value |
| --- | --- |
| Prompt sent | 99,998 tokens (287,281 chars) |
| Cold reprocess (server-side prompt eval) | **3,195.9 s** (avg 31.3 t/s; starts ~55 t/s, decays to ~31 t/s at 100k) |
| `action=save` wall | **0.79 s** (server `save_ms` 784 ms) |
| Save file | **3,278,859,412 bytes** (3.28 GB) = **32.8 KB/token** |
| Boot #2 (model reload, mmap) | 1.5 s |
| `action=restore` wall | **0.56 s** (server `restore_ms` 558 ms; file in page cache) |
| Follow-up request (3 new tokens) | `prompt_tokens` 3, **`cached_tokens` 99,999**, prompt eval 1,384 ms |

Reprocess avoided: **3,195.9 s → 1.4 s** (~2,300×). Save + restore together cost
1.3 s on the server side (file writes go through the page cache; even a cold
NVMe re-read of 3.28 GB adds ~1–2 s, still orders of magnitude below reprocess).

A 3k-token smoke run showed the same shape at small scale: reprocess 8.0 s,
save 128 ms (98.7 MB), restore 27 ms, follow-up 47 ms with 3,001/3,002 cached.

## Findings that shaped the design

1. **File size scales with model geometry**: 32.8 KB/token here (KV only). A
   Llama-3-8B-class model (32 layers × 8 KV heads × 128 dim) is 4× that —
   128 KB/token, ~13 GB at 100k. A cap + TTL is mandatory, not optional.
2. **`--slot-save-path` is required at boot even for a restore-only launch**:
   the server returns 400 (`all slots are disabled ... specify --slot-save-path`)
   for both `save` and `restore` without it. The flag must be composed up
   front, which is why capability-gating (below) matters.
3. **Restore slot ids wrap silently**: `POST /slots/{id}` does
   `id % slots.size()` — restoring file `5` into a 4-slot server writes into
   slot 1, clobbering nothing but reading the wrong file. Restore must iterate
   *live* slot ids from `GET /slots`, not filenames on disk.
4. **Slot-to-conversation matching is free**: llama.cpp's own scheduler picks a
   restored slot for a new prompt by longest-common-prefix similarity
   (`slot_prompt_similarity` default 0.1). E2E log: `selected slot by LCP
   similarity, f_sim_best = 1.000`. No llamastash-side matching needed; N-slot
   fan-out (N saves) is what makes at least one restored slot available.
5. **`GET /slots` reports 0-token ghosts**: a slot can show `n_prompt_tokens: 0`
   after its state moved to `prompt_cache` during generation; saving those is a
   no-op write, so the saver skips them (only saves `n_prompt_tokens > 0`).
6. **`usage.prompt_tokens_details.cached_tokens`** is the observable the E2E
   gate reads; after a restored launch it reports the restored prefix, so the
   proxy path proves the same thing the timing lines do.

## End-to-end proof (implemented behavior)

Isolated daemon (`LLAMASTASH_STATE_DIR`/`CONFIG_DIR`/`CACHE_DIR` overrides,
proxy `:11835`, `idle_ttl_secs: 40`), this repo's `target/debug/llamastash`,
same model and binary.

| Step | Result |
| --- | --- |
| R1 cold, 20,305-token prompt through the proxy | 148.6 s wall; `prompt_tokens` 20,305, `cached_tokens` 0 |
| R2 same conversation, same launch | 2.5 s; `cached_tokens` 20,300 (in-process prompt cache) |
| Idle sweep evicts | daemon saved `slot-3.bin` (665,712,340 B ≈ 20,305 × 32 KB) into the per-model cache dir before stop |
| R3 follow-up after eviction | auto-start → restore before Ready → **1.6 s wall**; server re-evaluated **7 of 20,307 tokens** (`prompt eval time = 636 ms / 7 tokens`), LCP 1.000 onto the restored slot |

148.6 s of reprocess became 1.6 s of launch + restore + 7-token prefill.

## Conclusion

Save-before-eviction / restore-before-ready is a large, low-risk win: sub-second
server-side cost at 100k, ~32 KB/token per model on disk, and llama.cpp itself
handles slot matching. Ship it gated on `--help` advertising
`--slot-save-path` (so older binaries never see the flag), with a per-model
size cap and TTL cleanup since files are large.
