---
title: "spike: llama.cpp slot KV save/restore at 100k tokens"
date: 2026-10-10
status: done
unblocks: ["R14 batch 4 — keep a model's prompt cache across an unload"]
todo: "Remeasure on GPU builds before making save-on-eviction default-on there."
---

# Question

Is `--slot-save-path` + `POST /slots/{id}?action=save|restore` fast and small enough
at a 100k-token context that saving a slot's KV cache before an eviction beats letting
the next launch reprocess the whole prompt?

# Setup

- llama.cpp **b11525** (commit `609290be6`), built CPU-only; `llama-server -c 131072 -np 1 -t 8`, `kv_unified = false`.
- Model: `Llama-3.2-1B-Instruct-Q4_K_M.gguf`.
- Prompt: 100,760 tokens (repeating English sentences; exact count via the live `POST /tokenize`, 1.78 MB of text at 5.6 chars/token for this generator).
- Machine: 16-core CPU box, 8 threads per llama.cpp, one server at a time. Observed prompt-processing throughput varied 20-124 t/s with background load (the cold run averaged 31.4 t/s); save/restore timings were insensitive to this.

# Numbers at 100k tokens

| Operation | Value |
| --- | --- |
| Reprocess cold (fresh server, full 100,761-token prompt) | **3,205.4 s** (`timings.prompt_ms` 3,205,422, `cache_n` 0, 31.4 t/s avg) |
| `action=save` | **778.8 / 791.2 ms** wall (two samples; `timings.save_ms`) for 100,761 tokens |
| Save file size | **3,303,349,060 bytes (3.15 GiB)**, ≈32 KiB/token |
| `action=restore` | **546.2 / 559.0 ms** wall (two samples; `timings.restore_ms`, `n_restored` 100,761) |
| First follow-up on the restored slot | **58.8 ms** `prompt_ms` for a new 16-token turn (15 tokens processed, 0 of the 100k reprocessed) vs 31.8 ms/token cold |

Reprocess is ~5 orders of magnitude more expensive than save + restore at this size:
a ~1.4 s pair of HTTP calls replaces a 53-minute prompt processing at this machine's
load, on a 1B model where weight reload itself is only seconds. The win grows
with model size (reload-dominated) and with prompt length (processing is ~linear).

# Behavioral findings (from the live server and `tools/server/`)

- Save/restore response shapes: save returns `{id_slot, filename, n_saved, n_written, timings.save_ms}`, restore returns `{id_slot, filename, n_restored, n_read, timings.restore_ms}`. `n_written`/`n_read` are bytes and equal the on-disk file size.
- Filenames are caller-supplied and flat: `filepath = slot_save_path + filename`. The `--slot-save-path` directory must already exist at launch; llama.cpp appends the trailing separator. Nothing is written automatically on exit.
- Restore targets a slot id on the running server and validates the file (rejects bad tokens or a prompt larger than the slot context with `ERROR_TYPE_INVALID_REQUEST`). A rejected restore leaves the slot usable.
- A restored slot behaves as if the conversation were still live: the official test (`tools/server/tests/unit/test_slot_save.py`) restores and then sends only a new turn; llama.cpp truncates to the common prefix and processes only the delta (`prompt_n` = new tokens). Re-sending the full history also works as long as the history alone still fits the context: the launch-time size check counts the incoming request only, not restored + incoming (measured: a 320k-token request against a slot restored with 100k was rejected as `320613 >= 131072`, i.e. incoming counted once).
- llama.cpp keeps prompt processing to completion even if the client disconnects mid-prompt (observed: killed clients, server finished the task and logged full timings). The reverse also holds for cancellation on some paths; long client timeouts are required for 100k prompts.
- KV cells for a released slot stay in the pool when `kv_unified` is on and prefix reuse can hit them without any save file; with `kv_unified = false` (this setup) the save file is the only way across a process restart.
- `GET /slots` rows for never-used slots can omit every field except `id` (observed on a freshly launched server). Anything matching on slot rows must not require `n_prompt_tokens`/`is_processing` there; requiring them makes a brand-new server look slotless.
- File size scales linearly with prompt tokens and model KV size (≈32 KiB/token here: 1B, 16 layers, f16 KV). A 13B-class model at 100k lands in the tens of GiB — the cap and the save-before-eviction-only policy in the implementation are what keep that bounded.

# Verdict

Ship it, gated on eviction only (see `docs/architecture.md` § Prompt cache across an
unload). Save + restore at 100k costs about 1.3 s of wall time against tens of minutes
of reprocessing on CPU, and degrades gracefully (a failed or timed-out restore just
reprocesses).
