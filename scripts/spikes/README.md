# Spike drivers

One-off measurement scripts, kept because the numbers they produced are cited in
[`docs/spikes/`](../../docs/spikes) and someone has to be able to re-measure them.
They talk to a real `llama-server`; none of them are part of `make test`.

Each takes `--server-bin` and `--model`, writes its own scratch under the directory
you pass, and refuses to share a port with anything already listening. Run one at a
time.

| Script | Answers | Cited in |
|---|---|---|
| `kv_slot_spike.py` | how long a cold prefill, a slot `save`, and a slot `restore` take at a given prompt size, and how big the file is | [`2026-10-10-prompt-cache-across-unload.md`](../../docs/spikes/2026-10-10-prompt-cache-across-unload.md) |
| `kv_slot_compat.py` | what a saved file may be restored into: a bigger context, a different slot, an unrelated prompt, a mismatched KV layout | same |
| `prompt_cache_e2e.py` | the whole feature through a real daemon: request, idle-TTL eviction, follow-up — with `--prompt-cache true` and `false` as the control pair | same |

```bash
cargo build --bin llamastash
python3 scripts/spikes/kv_slot_spike.py \
  --server-bin /path/to/llama-server --model /path/to/model.gguf \
  --tokens 100000 --ctx 102400 --save-dir ./scratch/slots
python3 scripts/spikes/kv_slot_compat.py \
  --server-bin /path/to/llama-server --model /path/to/model.gguf --save-dir ./scratch/slots
python3 scripts/spikes/prompt_cache_e2e.py \
  --scratch ./scratch --server-bin /path/to/llama-server --model /path/to/model.gguf \
  --words 1500 --prompt-cache true
```

`prompt_cache_e2e.py` starts its own daemon with its own state, config and cache
directories and non-default ports, so it never touches a daemon you are using. Pass
`--prompt-cache false` for the control run: same prompt, same eviction, no save.
