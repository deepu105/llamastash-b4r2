---
title: "spike: keeping the prompt cache across an unload (slot save/restore)"
date: 2026-10-10
status: done
unblocks: ["R14 batch 4 — Keep a model's prompt cache across an unload"]
todo: "Re-measure if upstream changes the slot save format or file size scaling."
---

# Question

When the idle sweep or make-room stops a llama.cpp launch mid-conversation, the
next request reprocesses the whole prompt. `llama-server` can write a slot's KV
cache to disk (`--slot-save-path` plus `POST /slots/{id_slot}?action=save|restore`,
`tools/server/README.md`, checked on local build 11390 on 2026-10-05). Is
save-then-restore fast enough to beat plain reprocessing, at a large context,
on the kinds of hosts llamastash runs on?

# API (verified 2026-10-10 against llama.cpp master `609290be6` / build 11525)

- Launch flag `--slot-save-path PATH`: PATH must be an existing directory;
  llama.cpp appends a trailing separator and joins the filename given per call.
  Without the flag every `/slots` action answers
  `"This server does not support slots action. Start it with \`--slot-save-path\`"`.
- `POST /slots/{id_slot}?action=save` with body `{"filename": "<name>.bin"}` →
  `{"id_slot", "filename", "n_saved", "n_written", "timings": {"save_ms"}}`.
  The name is validated flat (no path separators, ≤255 bytes). A busy slot
  defers the action rather than failing it.
- `POST /slots/{id_slot}?action=restore` with the same body →
  `{"n_restored", "n_read", "timings": {"restore_ms"}}`. The file carries the
  state plus its token list; loading into a slot whose layout differs (other
  ctx, other cache type) fails closed with an error response, never corrupt
  state.
- `GET /slots` lists every slot regardless of the flag: `id`, `is_processing`,
  and after its first task `n_prompt_tokens` / `n_prompt_tokens_cache`.
  A slot that never served a request omits the token fields.
- After a restore, a follow-up request prefix-matched 99,999 of 100,000 prompt
  tokens (1B test model, CPU run) — the engine's own prefix matching does the
  conversation re-join; nothing needs to route the request to the slot.

# Numbers

Test model Llama-3.2-1B-Instruct-Q4_K_M (≈32.8 KB per token of dump at the
default cache), four runs per figure.

## Dell XPS 15 9510 (`deepu-xps15-arch`), 100k-token prompt

CPU build of the same llama.cpp commit, 8 threads:

| Path | Time |
|---|---|
| Plain reprocessing | 2,881 – 3,196 s |
| Save | 0.71 – 0.78 s (file 3.05 GiB) |
| Restore | 0.52 – 0.57 s warm page cache, 1.26 s cold |

GPU (RTX 3050 Ti, 4 GB, CUDA build, all layers offloaded). The default cache at
100k does not fit — the KV cache alone needs 3,160 MiB — so the GPU rows run at
80k tokens, and at 100k with `-ctk q8_0 -ctv q8_0`:

| Config | Reprocess | Save | Restore (cold) | File |
|---|---|---|---|---|
| 80k, default cache | 49.9 s | 0.69 s | 0.49 s (1.08 s) | 2.62 GB |
| 100k, q8_0 cache | 85.1 s | 0.44 s | 0.36 s (0.74 s) | 1.74 GB |

## Strix Halo (128 GB unified), earlier round, llama.cpp build 11457

CPU: reprocess 79 – 82 s, save 2.2 – 2.7 s, restore 0.3 – 1.1 s.
GPU (one run): reprocess 73.4 s, save 0.29 s, restore 0.68 s. File 3.05 GiB.

# Verdict

**Build it.** Save + restore lands in 1 – 4 s across every host and placement
measured, against 50 – 3,200 s of reprocessing — two to four orders of
magnitude, and the restore time barely grows with context size. The dump is
large (≈33 KB/token), so the pile needs a byte cap and an age expiry; the
layout check the engine already does on load makes a wrong-sized or
wrong-shape file a clean miss, which simplifies everything on our side.

Design decisions landed in
[`docs/plans/2026-10-05-001-feat-r14-batches-plan.md`](../plans/2026-10-05-001-feat-r14-batches-plan.md)
§ Batch 4 and § Implementation notes; the code lives in
[`src/backend/llama_cpp/slots.rs`](../../src/backend/llama_cpp/slots.rs).
