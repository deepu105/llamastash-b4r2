---
title: "spike: keep the prompt cache across an unload (slot save/restore vs reprocessing)"
date: 2026-10-10
status: done
unblocks: ["R14 batch 4 - keep a model's prompt cache across an unload"]
todo: "Remeasure if llama.cpp changes the slot save/restore format."
---

# Question

When make-room or the idle sweep stops a llama.cpp launch mid-conversation, the next
request reprocesses the whole prompt. `llama-server` can write a slot's prompt cache to
disk (`--slot-save-path` + `POST /slots/{id}?action=save|restore`). Is save + restore to
disk faster than reprocessing a large prompt, and does the next request actually reuse a
restored slot?

# Result

**Go.** Save + restore is 1-2 orders of magnitude faster than reprocessing at 100k tokens
on every host measured, the restored slot is matched automatically by llama.cpp's own
prompt prefix matching (after restore, a repeat request found 99,999 of 100k tokens
cached), and a restore that cannot be honored (wrong KV quantization, prompt larger than
the new window) answers an HTTP error instead of crashing the server.

# Numbers

Test model on this machine (RTX 3050 Ti 4 GB + CPU), Llama-3.2-1B-Instruct-Q4_K_M.
CPU runs: CPU build of the same llama.cpp commit as the GPU build, 8 threads, four runs
each. The GPU has 4 GB: with the default (f16) cache a context over about 85,000 tokens
does not load, so the 100k GPU rows use an 8-bit cache.

## CPU, 100k tokens

| Measure | Value |
|---|---|
| Plain reprocessing | 2,881 - 3,196 s |
| Save | 0.71 - 0.78 s |
| Save file size | 3.05 GiB (about 32.8 KB per token) |
| Restore (file in page cache) | 0.52 - 0.57 s |
| Restore (page cache dropped) | 1.26 s |
| Cached tokens found after restore | 99,999 of 100k |

## GPU (RTX 3050 Ti, all layers offloaded)

| Config | Reprocess | Save | Restore | Restore (cache dropped) | File |
|---|---|---|---|---|---|
| 80k tokens, f16 cache | 49.9 s | 0.69 s | 0.49 s | 1.08 s | 2.62 GB |
| 100k tokens, `-ctk q8_0 -ctv q8_0` | 85.1 s | 0.44 s | 0.36 s | 0.74 s | 1.74 GB |

## Second host (Strix Halo, llama.cpp build 11457, 100k tokens)

| Arm | Reprocess | Save | Restore | File |
|---|---|---|---|---|
| CPU | 79 - 82 s | 2.2 - 2.7 s | 0.3 - 1.1 s | 3.05 GiB |
| GPU (one run) | 73.4 s | 0.29 s | 0.68 s (from disk) | 3.05 GiB |

# API observations (llama.cpp b11525, re-checked live 2026-10-10)

- `POST /slots/{id}?action=save` takes a JSON body `{"filename": "..."}`; the file is
  written under the `--slot-save-path` directory. Response:
  `{"id_slot", "filename", "n_saved", "n_written", "timings.save_ms"}`.
- `POST /slots/{id}?action=restore` with the same body shape; response carries
  `n_restored`, `n_read`, `restore_ms`.
- Save and restore are queued on the server's task loop and deferred while the slot is
  processing, so the HTTP call completes once the I/O is done and never races a
  generation.
- A save file is not bound to the slot id it came from: saving slot 3 and restoring into
  slot 0 works, so per-slot files only have to keep the slot index in range of the new
  launch's slot count.
- Mismatched files are rejected, not fatal: restoring an f16-KV save file into a server
  started with `-ctk q8_0 -ctv q8_0` answers
  `400 "Unable to restore slot: No available space in KV cache or invalid slot save file"`
  and the server stays healthy. Restoring into a smaller window is the same error.
- The returning conversation does not need to be matched by us: after a restore, the next
  request lands on the restored slot through llama.cpp's regular prompt prefix matching.
  Live check: a 3,174-token prompt took 546 ms cold; after stop + relaunch + restore, the
  same prompt reported `usage.prompt_tokens_details.cached_tokens = 3173` and
  `prompt_ms = 8.4`.
- `GET /slots` reports each slot's `id` and (when it has held a prompt)
  `n_prompt_tokens`; empty slots omit the count. `GET /props` carries `total_slots`.

# Decision

Save only before an eviction stop (idle sweep or make-room), never on an explicit user
stop. Files live under `<state dir>/slot-cache/<entry>/` keyed by the model path + size +
mtime, are guarded by a `meta.json` fingerprint, and are pruned by age (default 7 days)
and total size (default 16 GiB) at save time. Restore runs after the engine is reachable
and before the launch is declared Ready; every failure along the way is best-effort and
only ever costs the cache benefit, never the launch.
