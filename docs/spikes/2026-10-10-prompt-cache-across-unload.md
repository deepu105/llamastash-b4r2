---
title: "spike: carrying a llama.cpp prompt cache across an unload"
date: 2026-10-10
status: done
unblocks: ["R14 Batch 4 (faster reloads)"]
---

# Finding

**Go.** Saving the KV of a slot before an eviction and restoring it into the next
launch replaces a reprocess that takes minutes with a file copy that takes about a
second. At a 99 998-token prompt on a 1B model on CPU, reprefilling took
**2 979 s**; `action=save` plus `action=restore` took **1.42 s** and wrote a
**3.05 GiB** file. That is a ~2 100x difference on the part the feature controls,
on a laptop-class CPU with a model small enough that this is close to the worst
case for the file size and the best case for reprocessing speed.

The cost is disk. The KV is 32 784 bytes per token for this model, so a long
conversation is gigabytes before it is a line item — which is why the feature
needs a size cap and an age limit from day one.

## Setup

Measured on this host (16 cores, 62 GiB RAM, no GPU in the path), one
`llama-server` at a time, nothing else loading the CPU.

- `llama-server` b11525 (commit `609290be6`), CPU build, `-t 8`, KV in f16, flash-attn off (what a CPU launch resolves to)
- model `Llama-3.2-1B-Instruct-Q4_K_M.gguf` (16 layers, 8 KV heads, head dim 64)
- driver: `scripts/spikes/kv_slot_spike.py`, JSON in `/mnt/work/b4r2/scratch/b4r2-piqfnxhg-1/spike-100k.json`
- context `-c 102400`, `-np 1`, `--slot-save-path <dir>`

The prompt is 99 998 tokens of repeated synthetic tokens (`alphaNNNN`, 504 127
characters) — no semantic content on purpose, so nothing about the text can help
or hurt the measurement.

## Numbers

| Phase | Measured |
|---|---|
| Cold prefill, 99 998 tokens | **2 979 s** wall (`timings.prompt_ms` 2 978 974), 33.5 tokens/s, `cached_tokens` 0 |
| Same request again, same process (KV already warm) | `prompt_ms` 1 059.6, `cache_n` 99 997, `prompt_n` 1 |
| `POST /slots/0?action=save` | **862.6 ms** server-reported (0.864 s wall), 99 998 tokens, 3 278 334 868 bytes written |
| Server stopped, new server, `POST /slots/0?action=restore` | **553.4 ms** (0.554 s wall), 99 998 tokens, 3 278 334 868 bytes read |
| First request after the restore | `prompt_ms` 1 055.0, `cache_n` 99 997, `prompt_n` 1, 1.096 s wall |
| File size | 3 278 334 868 bytes = 3.05 GiB = 32 784 bytes per token |

Reading of the last two rows: the restore puts the prompt back, and the next
request then still spends ~1 s of `prompt_ms`. That is not the cache failing —
`prompt_n` is 1 and `cache_n` is 99 997, so a single token is being evaluated
against a 100k-token context, and the same 1 s shows up on a warm in-process
repeat. The part that save+restore removes is the 2 979 s.

32 784 bytes per token is the model's KV geometry plus the slot index: 16 layers
x 8 KV heads x 64 dims x 2 bytes (f16) x 2 (K and V) = 32 768, plus 16 bytes per
token of bookkeeping. A bigger model scales linearly with layers and KV heads, so
budget roughly 4.7 GiB per 100k tokens for a 48-layer, 8-head model in f16.

## What the engine will and will not accept

`scripts/spikes/kv_slot_compat.py`, 1 937-token prompt, llama-server b11525:

| Case | Result |
|---|---|
| Save the slot holding a prompt (slot 1 of `-np 2`) | 200, `n_saved` 1937 |
| Restore that file into a launch with a **bigger** context (8 192 to 16 384) | 200, `n_restored` 1937; next request `cache_n` 1936, `prompt_n` 1, 26.6 ms (cold prefill was 4 353.7 ms) |
| Unrelated 17-token prompt after a restore | normal, 52.5 ms, correct answer; the restored slot is left alone |
| Restore into a **different slot id** (saved slot 1, restored slot 0) | 200; the next request is routed to slot 0 and hits the cache (`cache_n` 1936, 29.3 ms) |
| Restore a file saved with f16 KV into a launch using `--cache-type-k q8_0 --cache-type-v q8_0` | **400** `Unable to restore slot: No available space in KV cache or invalid slot save file`; the request that follows produced text byte-identical to the same launch without any restore |

Three consequences, all of which shaped the implementation:

1. **A file is not bound to the slot id or the context size it was written from.**
   A launch whose context came out bigger (the normal case when `--fit` finds more
   free memory) still takes the cache. So the file name must not be keyed on
   context size or slot count, or the cache would be thrown away on the launches
   where it is most likely to help.
2. **llama.cpp does the conversation matching.** Its default
   `--slot-prompt-similarity` (0.1) makes the router pick the slot whose prompt
   shares the longest common prefix with the request, so a returning conversation
   finds its slot without any id being carried by LlamaStash. An unrelated prompt
   costs the wasted read and nothing else.
3. **A file the server cannot use is refused, not half-read.** The mismatch case
   returns a 400 and the launch behaves exactly as if nothing had been restored.
   That makes "delete the file when the restore fails" safe, which is what the
   implementation does.

## End-to-end through the daemon

`scripts/spikes/prompt_cache_e2e.py`, its own `LLAMASTASH_STATE_DIR` /
`CONFIG_DIR` / `CACHE_DIR`, proxy on 11661, launch ports 41500+,
`proxy.idle_ttl_secs: 15` so the sweep is what stops the launch. Same 4 212-token
prompt and same follow-up in both runs; ctx pinned to 8 192 and 8 threads by a
preset so the two runs are comparable.

| | feature on | feature off |
|---|---|---|
| Request 1 (4 212 tokens) | `prompt_ms` 12 519.4, `cached_tokens` 0 | `prompt_ms` 12 397.3, `cached_tokens` 0 |
| Eviction | `proxy eviction: stopping L1 ... idle 28.8 s >= ttl 15s`, then `prompt cache: saved 1 idle slot(s) (138 316 132 bytes)` | same stop, nothing written |
| Files after the stop | `llamastash-pc-Llama-3.2-1B-Instruct-Q4_K_M-ead050eb572879e8-slot3.bin`, 138 316 132 bytes | none |
| Request 2 (4 226 tokens) | **`cached_tokens` 4 212**, `prompt_n` 14, **`prompt_ms` 141.8**, 1.651 s wall | `cached_tokens` 0, `prompt_n` 4 226, `prompt_ms` 12 720.4, 14.150 s wall |
| Engine's own view after request 2 | slot 3 holds 4 233 prompt tokens | slot holds 4 226 after reprocessing |

The follow-up is 89x faster in prefill and 8.6x faster end to end; the remaining
1.6 s is model load plus the restore, which the feature does not remove.

Note where the evidence comes from: `timings` and
`usage.prompt_tokens_details.cached_tokens` on the proxy's `/v1/completions`
response, and `GET /slots?debug=1` on the launch port for the engine-side view.

## What this decided

- Save on **eviction only**. Both eviction callers (`sweep_once`, `make_room`)
  prove the launch is idle first, which is also when llama.cpp accepts a save.
- Restore **before** the launch is declared Ready, not after. Restoring after
  `Ready` lets a request route against an empty cache and then have the restore
  overwrite the prompt that request just built.
- Key the file on the **model file** (path, size, mtime) and the slot number, and
  let the engine reject what does not fit — because the compat matrix shows the
  engine accepts more than the launcher could predict and refuses cleanly what it
  cannot use.
- Cap by total size and by age, because 32 784 bytes per token is 3.05 GiB at
  100k tokens, and nothing here is worth filling a disk for.
