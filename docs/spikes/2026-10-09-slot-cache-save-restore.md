---
title: "spike: llama-server slot save / restore against prompt reprocessing"
date: 2026-10-09
status: done
unblocks: ["R14 batch 4: keep a model's prompt cache across an unload"]
todo: "Repeat on a GPU with a large model before turning the feature on by default."
---

# Finding

**Go.** At 100k tokens, saving the slot and restoring it takes about 1.3 s in
total (2 s with the file out of page cache). Reprocessing the same prompt takes
53 minutes on this host. The cost is disk: 3.05 GiB for this model.

| Prompt | Reprocess | Save | Restore (file in page cache) | Restore (page cache dropped) | File size | Follow-up after restore |
|---|---|---|---|---|---|---|
| 10,000 tokens | 42.8 s | 72 ms | 49 ms | 138 ms | 327,840,436 B (313 MiB) | 114 ms, 4 tokens evaluated, 10,000 cached |
| 100,000 tokens | 3,175.2 s (52 min 55 s) | 771 ms | 569 ms | 1,257 ms | 3,278,400,436 B (3.05 GiB) | 2,065 ms, 4 tokens evaluated, 100,000 cached |

Save and restore times are llama-server's own `timings.save_ms` /
`timings.restore_ms`; the wall time of the HTTP call was within 2 ms of each.
Reprocess and follow-up are `timings.prompt_ms` of `/completion`.

## Setup

- Host: Dell XPS 15 9510, i9-11900H (8 cores / 16 threads), 62 GiB RAM, on AC at
  100% battery, `platform_profile` = `performance`. Slot files on NVMe ext4.
- `llama-server` build 11525 (commit `609290be6`), a CPU-only build. Upstream's
  latest release on 2026-10-09 was `b11538`, 13 builds newer; not tested.
- Model: `Llama-3.2-1B-Instruct-Q4_K_M.gguf`.
- Flags: `-c <tokens + 1024> -np 1 -t 8 -tb 8 -ngl 0 --slot-save-path <dir>/`.
- Script: `scripts/bench/slot_cache_spike.py --tokens 10000,100000 --threads 8`.
  One run per size. Per size it starts a server, sends the prompt as a token
  array with `n_predict: 1`, saves slot 0, restarts the server, restores, sends
  the prompt plus 4 follow-up tokens, restarts again, drops the slot file from
  page cache (`posix_fadvise(POSIX_FADV_DONTNEED)` after `fsync`) and restores
  once more.

## What the numbers do not cover

- **CPU only, one small model.** On a GPU the reprocess time is far lower and
  the file is larger. The file is the KV cache at F16: 32,784 bytes per token
  here, which matches 16 layers x (512 K + 512 V) x 2 bytes. A model with more
  layers or a wider KV writes proportionally more. Tracked in `TODO.md`.
- **The 100k reprocess figure is slightly high.** Two `cargo` builds ran during
  its first 6 minutes (the second took 1 minute; the first was not timed). Save,
  restore and file size were measured on an idle machine.
- **"Page cache dropped" covers the slot file only.** The model weights stayed
  cached, and the drop is a request to the kernel that this run did not verify.
- **Single run per size.** No variance figure.

## Behaviour checked on the real server

Checked by hand on the same build with default slot settings (`total_slots: 4`,
`kv_unified = true`), two 1,690-token conversations A then B:

- Saving a slot that holds nothing succeeds with `n_saved: 0` and writes a
  36-byte file. The daemon drops any save under `min_tokens`.
- Only B's slot held tokens at save time (`n_saved` 1695 in slot 2, 0 in the
  other three). With a unified KV buffer llama-server clears an idle slot when
  another slot takes a task and moves its content to the in-memory prompt cache
  (`--cache-ram`). That cache has no save API, so A was not saved.
- After a restart and a restore into the same slot ids, B's follow-up evaluated
  1 token with 1,691 cached (30 ms). A's follow-up evaluated all 1,684 tokens
  (4.6 s).
- `--slot-save-path` must name an existing directory or the server exits at
  startup with `not a directory`. The directory is passed with a trailing
  separator: the server joins path and file name with no separator.
- The file name in the save / restore body must be a bare file name.

## Decisions taken from this

- Save only before an eviction (idle sweep, make-room). A save costs under a
  second per 3 GiB here, so it is cheap next to a reload, but it is wasted on a
  user's `stop`.
- One directory per launch under `<cache dir>/slot-cache`, keyed on the model
  file and the launch argv, one `slot-<id>.bin` per slot.
- Size cap (32 GiB default), age limit (24 h default), delete on restore.
- No conversation tracking in llamastash: each file goes back to its slot id,
  and llama-server's own prefix matching picks the slot.
- Off by default until the GPU measurement is done.
