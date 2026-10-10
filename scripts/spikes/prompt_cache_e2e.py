#!/usr/bin/env python3
"""End-to-end proof that a conversation survives an eviction.

Drives a real daemon, on its own state/config/cache dirs and its own ports,
through the sequence the feature exists for:

  request 1            -> the proxy auto-starts the model and prefills the prompt
  (idle past the TTL)  -> the sweeper evicts the launch, saving the slot's KV
  request 2            -> the proxy auto-starts it again, restores the KV, and the
                          same prompt comes back as cached instead of reprefilled

The before/after evidence is in the responses themselves: `prompt_tokens_details`
`.cached_tokens` and `timings.prompt_ms` (the prefill). The same run with
`--prompt-cache false` is the control - same prompt, same eviction, no save.

Everything is under one scratch root; the daemon the human is running is never
touched.

  python3 scripts/spikes/prompt_cache_e2e.py \
    --repo . --scratch /mnt/work/b4r2/scratch/b4r2-piqfnxhg-1 \
    --server-bin /path/to/llama-server --model /path/to/model.gguf
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

PROXY_PORT = 11661
MODEL_REF = "Llama-3.2-1B"
LAUNCH_PORT_START = 41500
LAUNCH_PORT_END = 41599


def log(*a):
  print(*a, flush=True)


def ls(repo: Path, binary: str, args, env, capture=False):
  return subprocess.run(
    [str(repo / binary)] + args,
    env=env,
    capture_output=capture,
    text=True,
    check=False,
  )


def argv(repo: Path, binary: str, args):
  return [str(repo / binary)] + args


def wait_daemon_up(repo, binary, env, timeout=90):
  deadline = time.time() + timeout
  while time.time() < deadline:
    proc = ls(repo, binary, ["status", "--json", "--no-colors"], env, capture=True)
    if proc.returncode == 0:
      try:
        body = json.loads(proc.stdout)
        if (body.get("daemon") or {}).get("pid"):
          return body
      except json.JSONDecodeError:
        pass
    time.sleep(1)
  raise SystemExit("daemon did not come up")


def running_launches(repo, binary, env):
  proc = ls(repo, binary, ["status", "--json", "--no-colors"], env, capture=True)
  try:
    return json.loads(proc.stdout).get("models") or []
  except json.JSONDecodeError:
    return [{"state": "unparseable"}]


def slots(port):
  """Engine-side view of one launch: which slot holds how many tokens, and how
  many of them are cached. This is the evidence the cache is really in RAM."""
  with urllib.request.urlopen(f"http://127.0.0.1:{port}/slots?debug=1", timeout=30) as resp:
    return json.loads(resp.read())


def completions(prompt, max_tokens, timeout=900):
  body = json.dumps({
      "model": MODEL_REF,
      "prompt": prompt,
      "max_tokens": max_tokens,
      "temperature": 0.0,
  }).encode()
  req = urllib.request.Request(
      f"http://127.0.0.1:{PROXY_PORT}/v1/completions",
      data=body,
      headers={"content-type": "application/json"},
  )
  t0 = time.time()
  with urllib.request.urlopen(req, timeout=timeout) as resp:
    out = json.loads(resp.read())
  t = out.get("timings") or {}
  details = (out.get("usage") or {}).get("prompt_tokens_details") or {}
  return {
      "wall_s": round(time.time() - t0, 3),
      "prompt_tokens": (out.get("usage") or {}).get("prompt_tokens"),
      "cached_tokens": details.get("cached_tokens"),
      "cache_n": t.get("cache_n"),
      "prompt_n": t.get("prompt_n"),
      "prefill_ms": round(t.get("prompt_ms", 0.0), 1),
  }


def compact_slots(rows):
  keep = ("id", "is_processing", "n_prompt_tokens", "n_prompt_tokens_cache")
  return [{k: r.get(k) for k in keep} for r in rows]


def main():
  ap = argparse.ArgumentParser()
  ap.add_argument("--repo", default=".")
  ap.add_argument("--binary", default="target/debug/llamastash")
  ap.add_argument("--scratch", required=True)
  ap.add_argument("--server-bin", required=True)
  ap.add_argument("--model", required=True)
  ap.add_argument("--tokens", type=int, default=2000)
  ap.add_argument("--words", type=int)
  ap.add_argument("--idle-ttl", type=int, default=15)
  ap.add_argument("--prompt-cache", default="true", choices=["true", "false"])
  args = ap.parse_args()

  repo = Path(args.repo).resolve()
  root = Path(args.scratch).resolve()
  model = Path(args.model).resolve()
  state, conf, cache = root / "e2e-state", root / "e2e-config", root / "e2e-cache"
  logs = root / "e2e-logs"
  for d in (state, conf, cache, logs):
    shutil.rmtree(d, ignore_errors=True)
    d.mkdir(parents=True)

  # ctx-size and threads are pinned by a preset so two runs of this script are
  # comparable; idle_ttl_secs is short so the sweep is what does the eviction.
  stem = model.stem
  (conf / "config.yaml").write_text(
      f"""
model_paths:
  - {model.parent}
daemon:
  port_range:
    start: {LAUNCH_PORT_START}
    end: {LAUNCH_PORT_END}
proxy:
  idle_ttl_secs: {args.idle_ttl}
backend:
  llamacpp:
    prompt_cache: {args.prompt_cache}
presets:
  {stem}.gguf:
    default: e2e
    entries:
      e2e:
        knobs:
          ctx-size: 8192
          threads: 8
"""
  )

  env = dict(os.environ)
  env.update({
      "LLAMASTASH_STATE_DIR": str(state),
      "LLAMASTASH_CONFIG_DIR": str(conf),
      "LLAMASTASH_CACHE_DIR": str(cache),
      "LLAMASTASH_LLAMA_SERVER": args.server_bin,
      "LLAMASTASH_NO_ANALYTICS": "1",
  })

  daemon_log = logs / "daemon.log"
  handle = daemon_log.open("w")
  daemon = subprocess.Popen(
      argv(repo, args.binary, ["daemon", "start", "--foreground", "--proxy-port",
                               str(PROXY_PORT), "--verbose"]),
      env=env,
      stdout=handle,
      stderr=subprocess.STDOUT,
  )
  result = {"prompt_cache": args.prompt_cache, "idle_ttl_secs": args.idle_ttl}
  try:
    wait_daemon_up(repo, args.binary, env)
    n_words = args.words or args.tokens
    prompt = " ".join(f"alpha{((i * 7919) % 5237)}" for i in range(n_words))
    follow_up = prompt + "\nQuestion: what was the first token of this document? Answer:"

    log("[request 1] prefilling a fresh launch")
    result["request_1"] = completions(prompt, 8)
    log(f"  {result['request_1']}")
    live = running_launches(repo, args.binary, env)
    result["launch_after_1"] = [{"port": r.get("port"), "state": r.get("state")} for r in live]
    if live:
      result["slots_after_1"] = compact_slots(slots(live[0]["port"]))

    # The sweeper stops the launch; wait for it rather than guessing a sleep.
    log("[evict] waiting for the idle sweep")
    gone = False
    deadline = time.time() + 4 * args.idle_ttl + 90
    while time.time() < deadline:
      time.sleep(2)
      if not running_launches(repo, args.binary, env):
        gone = True
        break
    result["evicted"] = gone
    cache_dir = cache / "prompt-cache"
    files = sorted(p.name for p in cache_dir.glob("*.bin")) if cache_dir.is_dir() else []
    result["files_after_evict"] = [
        {"name": p, "bytes": (cache_dir / p).stat().st_size} for p in files
    ]
    text = daemon_log.read_text()
    result["daemon_lines_of_interest"] = [
        line
        for line in text.splitlines()
        if any(k in line for k in ("prompt cache", "eviction", "stopping"))
    ][-12:]
    log(f"[evict] gone={gone} files={result['files_after_evict']}")

    log("[request 2] same conversation after the eviction")
    result["request_2"] = completions(follow_up, 8)
    log(f"  {result['request_2']}")
    live = running_launches(repo, args.binary, env)
    if live:
      result["slots_after_2"] = compact_slots(slots(live[0]["port"]))
    live = running_launches(repo, args.binary, env)
    if live:
      result["slots_after_2"] = compact_slots(slots(live[0]["port"]))

    first, second = result["request_1"], result["request_2"]
    result["prefill_saved_ms"] = round(first["prefill_ms"] - second["prefill_ms"], 1)
    if args.prompt_cache == "true":
      if not second.get("cached_tokens"):
        raise SystemExit("the follow-up hit no cache - the feature did not carry it")
      if second["prefill_ms"] >= first["prefill_ms"]:
        raise SystemExit("the follow-up was not faster than the cold prefill")
  finally:
    ls(repo, args.binary, ["daemon", "stop"], env, capture=True)
    try:
      daemon.wait(timeout=30)
    except subprocess.TimeoutExpired:
      daemon.kill()
    handle.close()
  print(json.dumps(result, indent=2))
  return 0


if __name__ == "__main__":
  sys.exit(main())
