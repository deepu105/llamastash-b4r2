#!/usr/bin/env python3
"""Measure llama.cpp slot KV save/restore against plain prompt reprocessing.

Spikes it for (R14 batch 4, prompt cache across unload): a fresh llama-server
process throws away a slot's KV cache, so the next request reprocesses the whole
prompt. llama-server can instead write that KV to disk (`--slot-save-path` plus
`POST /slots/{id}?action=save|restore`). This script measures both sides on one
real server:

  1. cold reprocess of an N-token prompt in a fresh process
  2. warm reprocess of the same prompt in the same process (the in-process cache)
  3. `action=save` time and file size
  4. `action=restore` time in a fresh process, then the first request's timings

Numbers go to stdout as a table plus a JSON blob (optionally --json-out).

Usage (the spike behind docs/spikes/2026-10-10-prompt-cache-across-unload.md):

  python3 scripts/spikes/kv_slot_spike.py \
    --server-bin /path/to/llama-server \
    --model /path/to/model.gguf \
    --save-dir /path/to/scratch/slots \
    --tokens 100000 --threads 8 --ctx 102400 --port 11599

One server runs at a time; each phase stops its server before starting the next.
"""

import argparse
import json
import random
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request
from pathlib import Path

WORDS = (
    "the kernel parses a tensor header while the scheduler queues requests and "
    "the cache keeps only the prefix it can prove identical; a slot holds tokens, "
    "timings, sampler state, and a pointer into shared memory, so reloading means "
    "reading weights again, re-running prefill, then decoding one token at a time "
    "under a clock that never lies about the cost of a cold start on this host"
).split()


def http(method, url, body=None, timeout=3600):
  data = None
  headers = {"Content-Type": "application/json"}
  if body is not None:
    data = json.dumps(body).encode()
  req = urllib.request.Request(url, data=data, headers=headers, method=method)
  with urllib.request.urlopen(req, timeout=timeout) as resp:
    return json.loads(resp.read().decode())


class Server:
  """One llama-server process; used as a context manager so phases never overlap."""

  def __init__(self, args, log_path):
    self.args = args
    self.log_path = log_path
    self.proc = None
    self.reader = None
    self.log = []

  def __enter__(self):
    argv = [
      self.args.server_bin,
      "-m", self.args.model,
      "--host", "127.0.0.1",
      "--port", str(self.args.port),
      "-t", str(self.args.threads),
      "-c", str(self.args.ctx),
      "-np", str(self.args.np),
      "--slot-save-path", f"{self.args.save_dir}/",
    ]
    fh = open(self.log_path, "w")
    self.proc = subprocess.Popen(argv, stdout=fh, stderr=subprocess.STDOUT)
    self._wait_ready()
    return self

  def __exit__(self, *exc):
    self.proc.terminate()
    self.proc.wait(timeout=60)
    return False

  def _wait_ready(self):
    deadline = time.time() + self.args.startup_timeout
    while time.time() < deadline:
      try:
        if http("GET", f"{self.url}/health").get("status") == "ok":
          return
      except (urllib.error.URLError, ConnectionError, TimeoutError):
        time.sleep(0.25)
    raise RuntimeError("server did not become healthy in time")

  @property
  def url(self):
    return f"http://127.0.0.1:{self.args.port}"

  def completion(self, prompt, max_tokens=1):
    t0 = time.perf_counter()
    out = http("POST", f"{self.url}/v1/completions", {
      "prompt": prompt,
      "max_tokens": max_tokens,
      "temperature": 0,
    })
    wall = time.perf_counter() - t0
    timings = out["timings"]
    return {
      "wall_s": round(wall, 3),
      "prompt_tokens": out["usage"]["prompt_tokens"],
      "cached_tokens": out["usage"].get("prompt_tokens_details", {}).get("cached_tokens", 0),
      "cache_n": timings["cache_n"],
      "prompt_n": timings["prompt_n"],
      "prompt_ms": round(timings["prompt_ms"], 1),
    }

  def slot_action(self, action, slot, filename):
    t0 = time.perf_counter()
    out = http("POST", f"{self.url}/slots/{slot}?action={action}", {"filename": filename})
    wall = time.perf_counter() - t0
    return out, round(wall, 3)

  def slots(self):
    return http("GET", f"{self.url}/slots?debug=1")


def build_prompt(target_tokens, url, seed=7):
  """Word soup trimmed to `target_tokens` with the server's own tokenizer."""
  rng = random.Random(seed)
  words = max(target_tokens, 1)
  parts = []
  while len(parts) < words:
    parts.append(WORDS[rng.randrange(len(WORDS))])
  text = " ".join(parts)
  for _ in range(40):
    n = len(http("POST", f"{url}/tokenize", {"content": text, "add_special": True})["tokens"])
    if n <= target_tokens:
      return text, n
    text = " ".join(text.split()[: max(1, int(len(text.split()) * target_tokens / n))])
  raise RuntimeError("could not trim prompt to target token count")


def phase(label, value):
  print(f"[{label}] {json.dumps(value)}", flush=True)


def server_version(binary):
  """`llama-server --version` prints its banner on stderr."""
  proc = subprocess.run([binary, "--version"], capture_output=True, text=True)
  text = (proc.stdout + proc.stderr).strip().splitlines()
  return text[0] if text else "unknown"


def main():
  ap = argparse.ArgumentParser(description=__doc__)
  ap.add_argument("--server-bin", required=True)
  ap.add_argument("--model", required=True)
  ap.add_argument("--save-dir", required=True)
  ap.add_argument("--tokens", type=int, default=100_000)
  ap.add_argument("--threads", type=int, default=8)
  ap.add_argument("--ctx", type=int, default=102_400)
  ap.add_argument("--np", type=int, default=1)
  ap.add_argument("--port", type=int, default=11599)
  ap.add_argument("--slot", type=int, default=0)
  ap.add_argument("--startup-timeout", type=float, default=600)
  ap.add_argument("--log-dir", default=None, help="default: --save-dir/logs")
  ap.add_argument("--json-out", default=None)
  args = ap.parse_args()

  save_dir = Path(args.save_dir)
  save_dir.mkdir(parents=True, exist_ok=True)
  log_dir = Path(args.log_dir or save_dir) / "logs" if not args.log_dir else Path(args.log_dir)
  log_dir.mkdir(parents=True, exist_ok=True)

  filename = f"spike-{args.tokens}.bin"
  result = {
    "model": str(Path(args.model).name),
    "server": server_version(args.server_bin),
    "ctx": args.ctx,
    "np": args.np,
    "threads": args.threads,
    "target_tokens": args.tokens,
  }

  # Phase 0 sizes the prompt with a throwaway short-context server, then stops it.
  short = argparse.Namespace(**vars(args))
  short.ctx = 4096
  with Server(short, log_dir / "phase0-tokenize.log") as srv:
    prompt, n_tokens = build_prompt(args.tokens, srv.url)
  result["prompt_tokens"] = n_tokens
  result["prompt_chars"] = len(prompt)
  phase("prompt", {"tokens": n_tokens, "chars": len(prompt)})

  # Phase 1: cold process, first request reprocesses the whole prompt.
  with Server(args, log_dir / "phase1-cold.log") as srv:
    cold = srv.completion(prompt)
    warm = srv.completion(prompt)
    save, save_wall = srv.slot_action("save", args.slot, filename)
    size = (save_dir / filename).stat().st_size
  result["cold_reprocess"] = cold
  result["warm_in_process"] = warm
  result["save"] = {"save_ms": round(save["timings"]["save_ms"], 1),
                    "wall_s": save_wall,
                    "n_saved": save["n_saved"],
                    "n_written": save["n_written"],
                    "file_bytes": size}
  phase("cold_reprocess", cold)
  phase("warm_in_process", warm)
  phase("save", result["save"])

  # Phase 2: fresh process, restore, then the same request.
  with Server(args, log_dir / "phase2-restore.log") as srv:
    restore, restore_wall = srv.slot_action("restore", args.slot, filename)
    after = srv.completion(prompt)
  result["restore"] = {"restore_ms": round(restore["timings"]["restore_ms"], 1),
                       "wall_s": restore_wall,
                       "n_restored": restore["n_restored"],
                       "n_read": restore["n_read"]}
  result["after_restore"] = after
  phase("restore", result["restore"])
  phase("after_restore", after)

  cold_ms = cold["prompt_ms"]
  save_restore_ms = result["save"]["save_ms"] + result["restore"]["restore_ms"]
  result["verdict"] = {
    "reprocess_ms": cold_ms,
    "save_plus_restore_ms": round(save_restore_ms, 1),
    "speedup_x": round(cold_ms / max(save_restore_ms + after["prompt_ms"], 0.001), 1),
    "file_bytes_per_token": round(size / max(save["n_saved"], 1), 1),
  }
  phase("verdict", result["verdict"])

  if args.json_out:
    Path(args.json_out).write_text(json.dumps(result, indent=2) + "\n")
  return 0


if __name__ == "__main__":
  sys.exit(main())
