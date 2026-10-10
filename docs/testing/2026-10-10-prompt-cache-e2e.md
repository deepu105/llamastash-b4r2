# Prompt cache across an unload — E2E run (2026-10-10)

Real-hardware check for the R14 batch 4 item, run on the RTX 3050 Ti 4 GB host
(`deepu-xps15-arch`, Arch Linux) against llama.cpp build 11525 (CUDA, all layers
offloaded — `--fit` placed them all and sized the window at 57,600 tokens,
`kv_unified` with 4 slots). Test model: `Llama-3.2-1B-Instruct-Q4_K_M.gguf`
(770 MB), prompt 3,174 tokens. Isolated daemon under its own
`LLAMASTASH_STATE_DIR` / `CONFIG_DIR` / `CACHE_DIR` and `--proxy-port 18135`;
`proxy.idle_ttl_secs: 15`.

## Sequence

1. `POST /v1/chat/completions` through the proxy (model not loaded) → proxy
   auto-starts the llama.cpp launch; the launch argv carries
   `--slot-save-path <state>/slot-cache/78d7d7c17039aeef` (read back from
   `/proc/<pid>/cmdline`).
2. Idle sweep stops the launch after TTL (`proxy eviction: stopping L1 … idle
   Some(27 s) >= ttl 15s`), preceded by
   `prompt cache: saved 1 slot(s), 104089636 bytes in 36 ms`. Entry:
   `slot-3.bin` (104,089,636 B, mode 0600) + `meta.json` (0600) under the
   0700 entry dir.
3. The same request goes through the proxy again → auto-start →
   `prompt cache: restored 1/1 slot(s), 3175 tokens in 20 ms into port 45100`
   before Ready → response served from the restored slot.

## Numbers (same 3,174-token prompt before and after the eviction)

| Arm | `usage.prompt_tokens_details.cached_tokens` | `timings.cache_n` | `timings.prompt_n` | `timings.prompt_ms` | wall |
|---|---|---|---|---|---|
| First request (cold server) | 0 | 0 | 3,174 | 545.0 | 2.2 s |
| After eviction, feature **off** (control) | 0 | 0 | 3,174 | 544.7 | 2.4 s |
| After eviction, feature **on** | 3,173 | 3,173 | 1 | 7.4 | 1.3 s |

The control (config `prompt_cache: false`) proves the cached 3,173 is the slot
file being restored, not the live cache surviving — the sweep stopped the server
in both arms. The restore slot was slot 3 and the follow-up request landed on it
through llama.cpp's own prefix matching (no matching logic on our side).

`--slot-save-path` reaching the argv, the save line, the restore line, and the
cached-token counts above were all read from the isolated daemon's own logs and
responses; the run's scratch dirs live under the run's `LLAMASTASH_STATE_DIR`.
