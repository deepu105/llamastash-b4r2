---
title: "spike: llama-server slot save and restore against prompt reprocessing"
date: 2026-10-10
status: done
unblocks: ["R14 batch 4: keep a model's prompt cache across an unload"]
---

# Finding

**Go.** Saving a slot and reading it back takes about 1 to 3 s at 100k tokens. Reprocessing the same prompt takes 50 s to 53 min, depending on the hardware. The cost is disk: about 32.8 KB per token on a 1B model, 3.05 GiB at 100k tokens.

The feature that came out of this is `backend.llamacpp.slot_cache` (`src/backend/llama_cpp/slot_cache.rs`, `docs/architecture.md` § Model lifecycle).

# What was measured

`llama-server --slot-save-path DIR` enables `POST /slots/{id_slot}?action=save` and `?action=restore`, each taking `{"filename": "..."}`. Save writes the slot's KV cache and token list to `DIR/filename`; restore reads it into a slot of a server started later.

- Model: `Llama-3.2-1B-Instruct-Q4_K_M.gguf`.
- "Reprocess" is the prompt evaluation time of the same prompt sent to a fresh server with an empty cache.
- "Restore, dropped" is a restore after the file was dropped from the page cache, so it is read from disk.

## Dell XPS 15 9510 (i9-11900H, RTX 3050 Ti 4 GB), llama.cpp build 11525 (`609290be6`)

CPU build, 8 threads, 100k tokens, four runs:

| | Time |
|---|---|
| Reprocess | 2,881 to 3,196 s |
| Save | 0.71 to 0.78 s |
| Restore, file in page cache | 0.52 to 0.57 s |
| Restore, dropped | 1.26 s |

File size: 3.05 GiB, about 32.8 KB per token. After a restore, a follow-up request found 99,999 prompt tokens in the cache.

CUDA build, all layers offloaded. 100k tokens do not load with the default cache on this 4 GB card (the KV cache alone needs 3,160 MiB), so the default-cache run is at 80k tokens and the 100k run uses an 8-bit cache (`-ctk q8_0 -ctv q8_0`):

| | 80k tokens, default cache | 100k tokens, 8-bit cache |
|---|---|---|
| Reprocess | 49.9 s | 85.1 s |
| Save | 0.69 s | 0.44 s |
| Restore, file in page cache | 0.49 s | 0.36 s |
| Restore, dropped | 1.08 s | 0.74 s |
| File size | 2.62 GB | 1.74 GB |

## Strix Halo, llama.cpp build 11457 (earlier round)

100k tokens. File size 3.05 GiB.

| | CPU | GPU (one run) |
|---|---|---|
| Reprocess | 79 to 82 s | 73.4 s |
| Save | 2.2 to 2.7 s | 0.29 s |
| Restore | 0.3 to 1.1 s | 0.68 s from disk |

# Behaviour checked while building the feature

Checked on build 11525 (CUDA, 2026-10-10). Upstream was at `b11541` that day, 16 builds ahead; its `tools/server/README.md` documents the same flag, endpoints and reply fields.

- The server exits at startup with `not a directory` when the `--slot-save-path` target does not exist.
- A save of a slot that holds nothing answers `200` with `n_saved: 0` and writes a 36-byte file.
- A restore of a missing file answers `400` (`Unable to restore slot: No available space in KV cache or invalid slot save file`).
- With the default slot count (4, unified KV), a request after a restore is routed to the restored slot by prompt prefix: the log reads `selected slot by LCP similarity, f_sim_best = 1.000 (> 0.100 thold)`. Nothing has to tell the server which slot a conversation belongs to.
- `GET /slots` does not report a token count for a slot that has not run a task since startup, so it cannot be used to tell an empty slot from a restored one. `/props` reports `total_slots`.
- `--cache-idle-slots` (default on) moves an idle slot into the server's in-RAM prompt cache when another conversation takes a slot. That cache is not written by a slot save.

# End to end through the daemon

Isolated daemon, same machine and CUDA build, context pinned to 24,576, `proxy.idle_ttl_secs: 10`. One conversation of about 15,350 prompt tokens; the idle sweep unloads the launch between the two turns.

| Follow-up after the unload | Cached prompt tokens | Prompt time | Request time |
|---|---|---|---|
| `slot_cache.enabled: false` | 0 of 15,370 | 4,222 ms | 5.52 s |
| `slot_cache.enabled: true` | 15,353 of 15,369 | 20.2 ms | 1.31 s |

The request time includes the model load. The save took 128 ms and the restore 104 ms for a 503 MB file. The first turn of the same run, with nothing cached, spent 3,445 ms on the prompt.

# Not measured

- A model larger than 1B. The file grows with the model's KV size per token, so both the size and the save time will be higher.
- A slow disk (SATA SSD or spinning).
