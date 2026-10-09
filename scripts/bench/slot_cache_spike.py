#!/usr/bin/env python3
"""Measure llama-server slot save / restore against plain prompt reprocessing.

For each prompt size it starts one llama-server, processes a prompt of that
many tokens, saves slot 0 with `POST /slots/0?action=save`, restarts the
server, restores the file and sends the same prompt plus a short follow-up.
It reports reprocess time, save time, restore time (file in page cache and
dropped from it) and file size.

Stdlib only. One server at a time. Example:

  scripts/bench/slot_cache_spike.py \
    --server /path/to/llama-server --model /path/to/model.gguf \
    --tokens 10000,100000 --threads 8 --work-dir /path/off/tmpfs \
    --out spike.json
"""

import argparse
import json
import os
import random
import signal
import subprocess
import sys
import time
import urllib.error
import urllib.request

WORDS = (
  "river stone window copper morning ledger harbor signal thread orchard "
  "winter marble lantern engine garden paper silver meadow anchor bridge "
  "candle forest valley mirror market thunder velvet pocket shadow summer"
).split()


def http(port, path, body=None, timeout=14400):
  url = f"http://127.0.0.1:{port}{path}"
  data = None if body is None else json.dumps(body).encode()
  req = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json"})
  with urllib.request.urlopen(req, timeout=timeout) as resp:
    return json.loads(resp.read())


class Server:
  def __init__(self, args, ctx, save_dir, log_path):
    self.port = args.port
    argv = [
      args.server, "-m", args.model, "-c", str(ctx), "-np", "1",
      "-t", str(args.threads), "-tb", str(args.threads), "-ngl", "0",
      "--host", "127.0.0.1", "--port", str(args.port),
      "--slot-save-path", save_dir + os.sep, "--no-webui",
    ]
    self.log = open(log_path, "ab")
    started = time.monotonic()
    self.proc = subprocess.Popen(argv, stdout=self.log, stderr=subprocess.STDOUT)
    while True:
      if self.proc.poll() is not None:
        raise RuntimeError(f"llama-server exited early, see {log_path}")
      try:
        http(self.port, "/health", timeout=2)
        break
      except (urllib.error.URLError, ConnectionError, OSError):
        time.sleep(0.2)
    self.load_s = time.monotonic() - started

  def stop(self):
    self.proc.send_signal(signal.SIGTERM)
    self.proc.wait(timeout=120)
    self.log.close()


def timed(fn):
  started = time.monotonic()
  out = fn()
  return out, time.monotonic() - started


def build_tokens(port, n):
  rng = random.Random(1234)
  tokens = []
  while len(tokens) < n:
    text = " ".join(rng.choice(WORDS) for _ in range(4000))
    tokens += http(port, "/tokenize", {"content": text, "add_special": not tokens})["tokens"]
  return tokens[:n]


def complete(port, tokens):
  body = {"prompt": tokens, "n_predict": 1, "cache_prompt": True, "id_slot": 0}
  resp, wall = timed(lambda: http(port, "/completion", body))
  t = resp["timings"]
  return {
    "wall_s": round(wall, 3),
    "prompt_n": t["prompt_n"],
    "cache_n": t.get("cache_n"),
    "prompt_ms": round(t["prompt_ms"], 1),
  }


def drop_page_cache(path):
  fd = os.open(path, os.O_RDONLY)
  try:
    os.fsync(fd)
    os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_DONTNEED)
  finally:
    os.close(fd)


def restore(port, name):
  resp, wall = timed(lambda: http(port, "/slots/0?action=restore", {"filename": name}))
  return {
    "wall_s": round(wall, 3),
    "restore_ms": round(resp["timings"]["restore_ms"], 1),
    "n_restored": resp["n_restored"],
    "n_read": resp["n_read"],
  }


def run_size(args, n):
  save_dir = os.path.join(args.work_dir, "slots")
  os.makedirs(save_dir, exist_ok=True)
  log_path = os.path.join(args.work_dir, f"server-{n}.log")
  name = f"spike-{n}.bin"
  path = os.path.join(save_dir, name)
  ctx = n + 1024
  row = {"tokens": n, "ctx": ctx}

  srv = Server(args, ctx, save_dir, log_path)
  try:
    tokens = build_tokens(srv.port, n)
    follow_up = http(srv.port, "/tokenize", {"content": " What comes next?"})["tokens"]
    row["reprocess"] = complete(srv.port, tokens)
    resp, wall = timed(lambda: http(srv.port, "/slots/0?action=save", {"filename": name}))
    os.sync()
    row["save"] = {
      "wall_s": round(wall, 3),
      "save_ms": round(resp["timings"]["save_ms"], 1),
      "n_saved": resp["n_saved"],
      "n_written": resp["n_written"],
      "file_bytes": os.path.getsize(path),
    }
  finally:
    srv.stop()

  srv = Server(args, ctx, save_dir, log_path)
  try:
    row["model_load_s"] = round(srv.load_s, 3)
    row["restore_warm"] = restore(srv.port, name)
    row["follow_up_after_restore"] = complete(srv.port, tokens + follow_up)
  finally:
    srv.stop()

  srv = Server(args, ctx, save_dir, log_path)
  try:
    drop_page_cache(path)
    row["restore_cold"] = restore(srv.port, name)
  finally:
    srv.stop()

  if not args.keep:
    os.remove(path)
  return row


def main():
  ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
  ap.add_argument("--server", required=True)
  ap.add_argument("--model", required=True)
  ap.add_argument("--tokens", default="10000,100000", help="comma-separated prompt sizes")
  ap.add_argument("--threads", type=int, default=8)
  ap.add_argument("--port", type=int, default=18451)
  ap.add_argument("--work-dir", required=True, help="slot files and server logs; keep it off tmpfs")
  ap.add_argument("--out", help="write the JSON result here as well as stdout")
  ap.add_argument("--keep", action="store_true", help="keep the slot files")
  args = ap.parse_args()

  os.makedirs(args.work_dir, exist_ok=True)
  version = subprocess.run([args.server, "--version"], capture_output=True, text=True)
  result = {
    "server_version": (version.stderr + version.stdout).strip().splitlines()[0],
    "model": os.path.basename(args.model),
    "threads": args.threads,
    "rows": [],
  }
  for n in (int(x) for x in args.tokens.split(",")):
    result["rows"].append(run_size(args, n))
    print(json.dumps(result["rows"][-1]), file=sys.stderr, flush=True)
  text = json.dumps(result, indent=2)
  if args.out:
    with open(args.out, "w") as f:
      f.write(text + "\n")
  print(text)


if __name__ == "__main__":
  main()
