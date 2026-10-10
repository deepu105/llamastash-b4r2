---
title: "spike: llama-server slot save / restore against prompt reprocessing"
date: 2026-10-10
status: done
unblocks: ["R14 batch 4: keep a model's prompt cache across an unload"]
todo: "Remeasure on a GPU build with a large model (KV cache in VRAM) before trusting the 60 s budgets near the 32 GiB cap."
---

# Finding

**Go. At 100,000 tokens, saving the slot took 0.72 s and restoring it 0.57 s (1.26 s when the file had to be read from disk), against 2,910 s (48.5 minutes) to reprocess the prompt. The file is 3.05 GiB, 32,784 bytes per token.**

| At 100,000 tokens | Time | Notes |
|---|---|---|
| Reprocess the prompt from an empty slot | 2,910.1 s | `timings.prompt_ms`, 34.4 tokens/s average |
| Save the slot (`POST /slots/0?action=save`) | 0.717 s | server's `save_ms`; returns once the bytes are in page cache |
| Flush the save file to disk (`fsync`) | 0.99 s | not on the eviction path; the kernel does this in the background |
| Restore, save file in page cache | 0.568 s | server's `restore_ms` |
| Restore, save file read from disk | 1.260 s | file evicted with `posix_fadvise(DONTNEED)` first, 0 bytes cached |
| Follow-up request after a restore | 14.8 to 15.5 s | 100,000 tokens from cache, 32 processed. The same follow-up in the server that had processed the prompt took 16.0 s |
| File size | 3,278,400,436 bytes (3.05 GiB) | `n_written` and `stat` agree |

Save plus a restore from disk is 1.98 s, 0.07% of the reprocess time. The plan's bar was "worth building only if save plus restore beats reprocessing".

## Setup

- Host: Dell XPS 15 9510, Intel i9-11900H (8 cores, 16 threads), 62 GiB RAM, on AC, `platform_profile` = `performance`. Linux 7.2.9, ext4 on an SK hynix PC711 1 TB NVMe.
- `llama-server` build 11525 (commit `609290be6`, 2026-10-09), CPU-only build. The latest release that day was `b11540`, 15 commits newer. It was not run, but its `tools/server/server-context.cpp` carries the same slot save, restore, `/slots` and slot-selection code and the same error strings (read through `gh api` on 2026-10-10).
- Model: `Llama-3.2-1B-Instruct-Q4_K_M.gguf` (770 MiB; 16 layers, 8 KV heads of 64 dims, f16 KV cache).
- Flags: `-c 102048 -np 1 -t 8 -tb 8 --slot-save-path <dir> --no-webui`. Everything else at the build's defaults.
- Prompt: 100,000 token ids from seeded random English words, sent to `/completion` as a token array with `n_predict: 1`, `cache_prompt: true`, `id_slot: 0`.
- Script: [`scripts/bench/slot_save_restore.py`](../../scripts/bench/slot_save_restore.py). One server at a time; each step after the save runs in a freshly started server.

## Raw results

```json
{
  "tokens": 100000,
  "ctx": 102048,
  "threads": 8,
  "model": "Llama-3.2-1B-Instruct-Q4_K_M.gguf",
  "server_version": "version: 0.6.0-dev (build 11525, commit 609290be6)\nbuilt with GNU 16.2.1 for Linux x86_64",
  "load_s": 1.318,
  "reprocess": {
    "wall_s": 2910.11,
    "prompt_n": 100000,
    "cache_n": 0,
    "prompt_ms": 2910077.703,
    "prompt_per_second": 34.3633435962586
  },
  "warm_same_process": {
    "wall_s": 16.01,
    "prompt_n": 32,
    "cache_n": 100000,
    "prompt_ms": 15971.335,
    "prompt_per_second": 2.003589555913767
  },
  "save": {
    "id_slot": 0,
    "filename": "spike.bin",
    "n_saved": 100000,
    "n_written": 3278400436,
    "timings": {
      "save_ms": 716.969
    },
    "wall_s": 0.718,
    "file_bytes": 3278400436,
    "fsync_s": 0.988
  },
  "restore_page_cache": {
    "file_cached_bytes_before": 3278401536,
    "load_s": 1.413,
    "restore": {
      "id_slot": 0,
      "filename": "spike.bin",
      "n_restored": 100000,
      "n_read": 3278400436,
      "timings": {
        "restore_ms": 568.04
      },
      "wall_s": 0.569
    },
    "followup": {
      "wall_s": 15.495,
      "prompt_n": 32,
      "cache_n": 100000,
      "prompt_ms": 15461.913,
      "prompt_per_second": 2.0696016075113084
    }
  },
  "restore_cold_file": {
    "file_cached_bytes_before": 0,
    "load_s": 1.212,
    "restore": {
      "id_slot": 0,
      "filename": "spike.bin",
      "n_restored": 100000,
      "n_read": 3278400436,
      "timings": {
        "restore_ms": 1260.153
      },
      "wall_s": 1.261
    },
    "followup": {
      "wall_s": 14.796,
      "prompt_n": 32,
      "cache_n": 100000,
      "prompt_ms": 14761.942,
      "prompt_per_second": 2.1677364671938153
    }
  }
}
```

## What the numbers mean

- **Reprocessing is slow because CPU attention cost grows with the context.** The log shows 388 tokens/s for the first 4,096 tokens and 18.5 tokens/s for the last batch. Time per 2,048-token batch rose linearly with the tokens already in context (3.1 s plus 1.11 s per 1,000 tokens of context), so total time grows with the square of the prompt length. A GPU changes the constants, not the shape.
- **The host ran `cargo` builds during part of the reprocess pass.** Seven of the 48 batches (between 49k and 78k tokens) took 3 to 17 s longer than a straight-line fit of the rest, 59 s in total, which is 2% of the pass. The save and restore steps ran on a quiet host.
- **The reprocess figure is one sample.** The script's second reprocess pass (a fresh server, no restore) was stopped after it started, to free the host. It would have repeated the same measurement.
- **32 new tokens took 15 s.** At 100,000 tokens of context this CPU evaluates about 2 tokens/s, restored slot or not. That is the cost of the context length, not of the restore: the same request in the server that had built the cache took 16.0 s.
- **The save is a memory copy plus a buffered write.** The server answers when the bytes are in page cache. Writing them to the NVMe took another 0.99 s. An eviction does not wait for that.
- **The file is the KV cache.** 2 (K and V) x 16 layers x 8 KV heads x 64 dims x 2 bytes (f16) = 32,768 bytes per token, plus 16 bytes per token of token ids and cell metadata. `gguf::memory::kv_bytes` gives the first term from the GGUF header, so the daemon can size a slot before it writes one. The same formula gives 131,072 bytes per token for an 8B Llama (32 layers, 8 KV heads of 128 dims): about 12.2 GiB at 100,000 tokens. Not measured here.
- **A restored slot is used without any hint from the client.** The follow-up sent the same 100,000 tokens plus 32 new ones and the server reported `cache_n: 100000`. llama-server gives a request to the slot with the longest common token prefix (`get_available_slot` in `tools/server/server-context.cpp`).

## What the server does, read from the source

Checked in `tools/server/server-context.cpp` and `common/arg.cpp` at commit `609290be6`:

- `--slot-save-path PATH` must name an existing directory or the server exits with `not a directory`. A trailing separator is added if missing.
- `POST /slots/{id}?action=save|restore|erase` answers `501` unless the flag was given. The body is `{"filename": "<name>"}`; `fs_validate_filename` rejects a name with a path in it. The file is `PATH + filename`.
- Save writes `llama_state_seq_save_file` for that slot's sequence, then appends the slot's context checkpoints. Restore reads both back and replaces the slot's prompt. A failed restore clears the slot.
- A save or restore of a slot that is processing a request is deferred until the slot is free.
- Restore checks the token count against the slot's context and validates the token ids. It does not check which weights wrote the file.
- `GET /slots` lists `id`, `is_processing`, and `n_prompt_tokens` for a slot that has run a task. The prompt text is left out unless `LLAMA_SERVER_SLOTS_DEBUG` is set in the server's environment.
- With the default unified KV cache and `--cache-ram` on, starting a task moves every other idle slot into the in-memory prompt cache and clears it. So at eviction time usually one slot holds a prompt.

## Through the daemon

The same build and model behind an isolated llamastash daemon, driven through the proxy by [`scripts/bench/prompt_cache_e2e.py`](../../scripts/bench/prompt_cache_e2e.py): `ctx-size: 32768`, `threads: 8`, the default 4 slots with unified KV, `proxy.idle_ttl_secs: 20`. One conversation with a 15,046-token prompt; each follow-up adds one short turn.

| Request | Prompt cache on: cached / processed tokens, prompt time | Prompt cache off: cached / processed tokens, prompt time |
|---|---|---|
| 1. First request (auto-start) | 0 / 15,046, 87.0 s | 0 / 15,046, 91.0 s |
| 2. Follow-up, model still loaded | 15,047 / 17, 1.1 s | 15,047 / 17, 1.1 s |
| Idle sweep unloads the model | slot 3 saved: 15,065 tokens, 471.0 MiB, 108 ms | nothing saved |
| 3. Follow-up after the unload (auto-start) | 15,065 / 17, 1.0 s | 0 / 15,082, 92.0 s |

Request 3 took 2.4 s end to end with the cache on (model load, a 67 ms restore, 1.0 s of prompt work) and 93.3 s with it off. The new server's log shows how the conversation found its cache: `selected slot by LCP similarity, f_sim_best = 0.999 (> 0.100 thold), f_keep = 1.000` on slot 3. The slot file (493,891,396 bytes) was on disk only while the model was unloaded.

## Decisions taken from this

- Save before an eviction only, restore before `Ready`, delete the file after the restore. Details: [`docs/plans/2026-10-05-001-feat-r14-batches-plan.md`](../plans/2026-10-05-001-feat-r14-batches-plan.md#result-prompt-cache-2026-10-10).
- Key the file to the launch (argv plus model file size and mtime), because the server does not check the weights.
- Budget 60 s for the saves of one eviction and 60 s for the restores of one start. At the measured rates a file near the 32 GiB cap needs about 8 s into page cache (18 s if the kernel has to write it through at the measured 1.8 GiB/s) to save and 13 s to restore from a cold file on this disk. Those are extrapolations from one 3 GiB file on CPU; the GPU remeasure in `TODO.md` is there to check them.

## Not measured

- A GPU build. Saving reads the KV cache back from VRAM and restoring writes it there.
- A larger model. Bytes per token are computed above, not measured.
- A slower disk. Restore time from a cold file is bounded by sequential read speed.
- More than one slot.
