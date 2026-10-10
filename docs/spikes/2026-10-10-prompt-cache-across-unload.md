---
title: "spike: keep a model's prompt cache across an unload (slot save/restore)"
date: 2026-10-10
status: decided
unblocks: ["R14 batch 4 — Keep a model's prompt cache across an unload"]
todo: "Re-measure if llama.cpp changes the slot save/restore file format or timings."
---

# Question

When make-room or the idle sweep stops a llama.cpp launch mid-conversation,
the next request reprocesses the whole prompt. `llama-server` can write a
slot's KV cache to disk and read it back: `--slot-save-path PATH` plus
`POST /slots/{id_slot}?action=save|restore` (body `{"filename": "…"}`,
reply `n_saved` / `n_restored`). Is save + restore fast enough to beat plain
reprocessing at a large context?

# Engine facts (verified against the real thing)

Source and binary: llama.cpp 0.6.0-dev, build 11525, commit `609290be6`
(CPU and CUDA builds of the same commit), checked 2026-10-10 against
`tools/server/server-context.cpp`, `tools/server/README.md`, and the running
server. Also present on local build 11390 (checked 2026-10-05).

- `--slot-save-path DIR` must name an **existing directory** at startup; the
  server exits otherwise and appends the path separator itself.
- `save` takes a caller-chosen `filename` (validated: no path traversal,
  ≤255 chars) under that dir, and serializes the slot's prompt tokens plus
  its KV state (`llama_state_seq_save_file`). A slot with no prompt fails
  (`Unable to save slot`). A save on a processing slot is **deferred** by
  the server until the slot goes idle.
- `restore` reads the file into the slot (prompt + KV), verifying the slot id
  inside the file and that the restored prompt fits the slot's context. A
  restored slot then prefix-matches incoming prompts like any warm slot, so
  only the new tail of a returning conversation is processed.
- `GET /slots` reports `n_prompt_tokens` per slot (absent until the slot has
  served a prompt), which is what a restore can be checked against.

# Measurements

Test model `Llama-3.2-1B-Instruct-Q4_K_M` (KV costs ~32.8 KB per prompt
token at f16). The engine-side timings below were measured in the R14
batch-4 session with llama.cpp build 11525 / commit `609290be6`; ~15k-token
end-to-end proofs use the same binary.

**Host 1 — this machine's CPU (8 threads), four runs, 100k tokens:**

| Path | Time |
|---|---|
| Plain reprocessing | 2,881 – 3,196 s |
| Save | 0.71 – 0.78 s |
| Restore (file in page cache) | 0.52 – 0.57 s |
| Restore (page cache dropped) | 1.26 s |

Save file size: **3.05 GiB** (~32.8 KB/token). After a restore, a follow-up
request found 99,999 prompt tokens in the cache.

**Host 1 — this machine's GPU (RTX 3050 Ti, 4 GB), all layers offloaded:**

| Config | Reprocess | Save | Restore | File |
|---|---|---|---|---|
| 100k ctx, default cache | does not load — KV alone needs 3,160 MiB | — | — | — |
| 80k ctx, default cache | 49.9 s | 0.69 s | 0.49 s (1.08 s cold) | 2.62 GB |
| 100k ctx, `-ctk q8_0 -ctv q8_0` | 85.1 s | 0.44 s | 0.36 s (0.74 s cold) | 1.74 GB |

(The last row reprocesses slower per token than the middle one — 0.85 vs
0.62 ms/token — the quantized KV's cost at prefill; it buys the capacity
that lets 100k load on 4 GB at all.)

**Host 2 — Strix Halo, llama.cpp build 11457 (earlier round):**

| Device | Reprocess | Save | Restore | File |
|---|---|---|---|---|
| CPU | 79 – 82 s | 2.2 – 2.7 s | 0.3 – 1.1 s | 3.05 GiB |
| GPU (one run) | 73.4 s | 0.29 s | 0.68 s (from disk) | 3.05 GiB |

# Decision: build it

Save + restore is two to four orders of magnitude cheaper than
reprocessing on every configuration measured, and the file's cost (~32
KB/token) makes it the only part of a launch worth persisting (weights are
cheaper to re-read and the `warm` item covers that). Design decisions taken
from these numbers, implemented in `src/backend/llama_cpp/slot_cache.rs`
and documented in `docs/architecture.md § Prompt cache across unloads`:

1. **When to save:** before an eviction only (sweep or make-room). The
   save costs seconds; a manual stop is not a conversation pausing.
2. **Where files live:** `<cache dir>/slot-cache/<model header hash>/`,
   one file per slot (`slot-<id>.bin`), keyed on the GGUF header hash so a
   renamed file keeps its cache and a replaced file starts fresh.
3. **Cap:** total bytes across all models (`backend.llamacpp.
   prompt_cache_max_bytes`, factory 8 GiB ≈ two 100k-context caches); the
   oldest files are deleted first, and the model being saved is never the
   victim of its own save.
4. **Slot matching:** the file name and the slot id in the file both carry
   the slot id; a returning launch restores `slot-<id>.bin` into slot
   `<id>`. A restore that no longer fits the new launch's context (or a
   slot that no longer exists) is logged and skipped — that launch serves
   cold and the next eviction re-saves.
