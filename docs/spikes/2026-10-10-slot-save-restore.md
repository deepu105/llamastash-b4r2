---
title: "spike: slot save and restore against prompt reprocessing"
date: 2026-10-10
status: done
unblocks: ["R14 batch 4: keep a model's prompt cache across an unload"]
---

# Finding

**Go.** Writing a 100k-token slot to disk and reading it back takes about one second each way. Reprocessing the same prompt takes 50 s to 53 minutes, depending on the machine. The file is large: 3.05 GiB for 100k tokens on a 1B model, and it grows with the model.

Test model for every row: `Llama-3.2-1B-Instruct-Q4_K_M.gguf`.

## Numbers

### Dell XPS 15 9510, CPU (i9-11900H, 8 threads, four runs)

CPU build of the llama.cpp commit used for the GPU rows below.

| At 100k tokens | Result |
|---|---|
| Plain reprocessing | 2,881 to 3,196 s |
| Save | 0.71 to 0.78 s |
| Restore, file in the page cache | 0.52 to 0.57 s |
| Restore, page cache dropped | 1.26 s |
| File size | 3.05 GiB (about 32.8 KB per token) |
| Prompt tokens found in the cache by a follow-up after a restore | 99,999 |

### Dell XPS 15 9510, GPU (RTX 3050 Ti, 4 GB, CUDA build, all layers offloaded)

100k tokens do not load with the default cache type on this card: the KV cache alone needs 3,160 MiB.

| | 80k tokens, default cache | 100k tokens, `-ctk q8_0 -ctv q8_0` |
|---|---|---|
| Plain reprocessing | 49.9 s | 85.1 s |
| Save | 0.69 s | 0.44 s |
| Restore, file in the page cache | 0.49 s | 0.36 s |
| Restore, page cache dropped | 1.08 s | 0.74 s |
| File size | 2.62 GB | 1.74 GB |

### Strix Halo, earlier round (llama.cpp build 11457)

| At 100k tokens | CPU | GPU (one run) |
|---|---|---|
| Plain reprocessing | 79 to 82 s | 73.4 s |
| Save | 2.2 to 2.7 s | 0.29 s |
| Restore | 0.3 to 1.1 s | 0.68 s from disk |
| File size | 3.05 GiB | 3.05 GiB |

### Save plus restore against reprocessing

Slowest save plus slowest restore, against the fastest reprocess on the same row.

| Machine | Save + restore | Reprocess | Ratio |
|---|---|---|---|
| XPS CPU, 100k | 2.04 s | 2,881 s | about 1,400x |
| XPS GPU, 80k | 1.77 s | 49.9 s | 28x |
| XPS GPU, 100k q8_0 | 1.18 s | 85.1 s | 72x |
| Strix Halo CPU, 100k | 3.8 s | 79 s | 21x |
| Strix Halo GPU, 100k | 0.97 s | 73.4 s | 76x |

## File size

The file holds the KV cache for the tokens in the slot, so its size is the per-token KV size times the token count. For this model that is 16 layers x 8 KV heads x 64 dims x 2 (K and V) x 2 bytes (f16) = 32,768 bytes per token, which matches the measured 32.8 KB. `gguf::memory::kv_bytes` computes the same figure from the GGUF header, and the save uses it to refuse a file that would not fit before writing it.

The same arithmetic for a larger model, not measured: 64 layers with 8 KV heads of 128 dims is 262 KB per token, or 24 GiB at 100k tokens with an f16 cache. This is why saved files have a size cap.

## What llama-server does

Checked against build 11525 (commit `609290be6`), which is what `build-cuda/bin/llama-server` on the XPS reports. The latest release on 2026-10-10 was `b11541`, 16 commits ahead. The save and restore handlers, the `/slots` routes, the slot picker and the reply shapes are byte-identical between the two (compared from `tools/server/server-context.cpp` and `server-task.cpp`).

- `--slot-save-path DIR` turns the slot actions on. The server exits at startup with `not a directory` when `DIR` does not exist. Without the flag, `POST /slots/...` answers 501.
- `POST /slots/{id}?action=save` with `{"filename": "x.bin"}` writes `DIR/x.bin` and answers `{"id_slot", "filename", "n_saved", "n_written", "timings": {"save_ms"}}`. The filename may not contain a path separator.
- `POST /slots/{id}?action=restore` reads it back and answers `n_restored`, `n_read` and `timings.restore_ms`.
- Saving an empty slot answers 200 with `n_saved: 0` and writes a 36-byte file.
- A restore of a missing file, or of a file larger than the slot's context, answers 400 (`Unable to restore slot: ...`). The server keeps serving.
- A slot file loads into the slot id it came from or into another one. A follow-up found the same 3,571 cached tokens either way.
- `GET /slots` reports `n_prompt_tokens` only for a slot that has run a task. A restored slot reports none until it serves a request, so `n_restored` in the restore reply is the only confirmation.
- The server picks the slot for a request by the longest common prefix between the request's tokens and each slot's cached tokens (`--slot-prompt-similarity`, default 0.10). With no match it takes the least recently used slot.
- By default the server has 4 slots sharing one KV cache. When a request starts in one slot, the other idle slots are copied to the in-memory prompt cache (`--cache-ram`) and cleared, so normally one slot holds tokens. The in-memory prompt cache is not part of a slot save.
- The server writes the files with mode `0644`.

A 3,571-token probe on the XPS GPU, same build: first request 0.77 s with 0 cached tokens; server restarted and the follow-up sent without a restore, 0.77 s and 0 cached; server restarted, slot restored (21.8 ms, 117 MB), follow-up 0.18 s with 3,571 of 3,587 prompt tokens cached.

The reply after a restore is the same text as after a full reprocess and as on a server that was never restarted (`temperature: 0`, same 3.6k-token conversation).

## Speculative decoding

A slot file holds the target model's context only. The draft context starts empty after a restore.

Measured with `--spec-type draft-simple` and the test model as its own draft, so that an in-sync draft is accepted every time: the reply after a restore was again identical, but 38 of 57 drafted tokens were accepted, against 42 of 42 on a server that was never restarted and 42 of 42 after a full reprocess. The turn after that accepted 7 of 9.

MTP (`draft-mtp`) keeps its own per-sequence state in a second context too, and no MTP model was available for this spike. A launch that runs speculative decoding is therefore left out of slot saving until MTP is measured.

## Decisions taken from this

Recorded in `docs/architecture.md` § Model lifecycle and implemented in `src/backend/llama_cpp/slot_save.rs`:

- Save before an eviction only (idle sweep, make-room). At one second for 100k tokens the save is cheap enough to do on every eviction, and too large to do after every request.
- Not for a launch that runs speculative decoding, per the section above.
- Restore before the launch is reported Ready, so the request that triggered the reload already hits the cache.
- No conversation ids. The server's own prefix match finds the restored slot.
- Files go in the cache directory, one directory per model, capped by total size and by age.

## End-to-end check of the implementation

Isolated daemon (own state, config and cache directories, proxy on port 11735), XPS GPU, `ctx-size: 32768`, `n-gpu-layers: 99`, `proxy.idle_ttl_secs: 20`. One conversation with a 14,048-token first prompt, sent through the proxy. The third turn is sent after the idle sweep has stopped the launch.

| Request | `slot_save` on | `slot_save` off |
|---|---|---|
| First turn, cold start | 14,048 prompt tokens, 0 cached, prompt 3,173 ms, 4.55 s total | 14,048 prompt tokens, 0 cached, prompt 3,344 ms, 4.72 s total |
| Second turn, model still loaded | 14,071 of 14,087 cached, prompt 17 ms, 0.19 s total | 14,071 of 14,087 cached, prompt 16 ms, 0.19 s total |
| Third turn, after the eviction | **14,103 of 14,118 cached, prompt 21 ms, 1.42 s total** | **0 of 14,118 cached, prompt 3,254 ms, 4.74 s total** |

Total time for the first and third turns includes loading the model. With `slot_save` off the daemon removed the `slots/` directory at start. Daemon log for the run with it on:

```
proxy eviction: stopping L1 (...Llama-3.2-1B-Instruct-Q4_K_M.gguf) — idle Some(33.4s) >= ttl 20s
slot save: Llama-3.2-1B-Instruct-Q4_K_M.gguf saved 14103 tokens in 1 slot(s), 0.4 GiB in 128 ms (estimated 0.4 GiB)
slot save: Llama-3.2-1B-Instruct-Q4_K_M.gguf restored 14103 tokens in 1 slot(s), 0.4 GiB in 85 ms
```

## Not measured

- A model larger than 1B, where the file is several times larger.
- MTP speculative decoding, on a model with an MTP head.
- A launch with a multimodal projector. llama-server serializes media chunks into the slot file and refuses to restore them without an mmproj; neither path was run.
- A slow disk. Every run here wrote to NVMe.
