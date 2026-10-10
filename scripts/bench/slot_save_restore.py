#!/usr/bin/env python3
"""Measure llama-server slot save / restore against plain prompt reprocessing.

Drives one raw `llama-server` (no llamastash) through:

  1. process an N-token prompt from an empty slot        -> reprocess time
  2. POST /slots/0?action=save                           -> save time, file size
  3. restart, POST /slots/0?action=restore, send the
     same prompt plus a short follow-up                  -> restore time, cached tokens
     (once with the save file in page cache, once after evicting it)
  4. restart, send the same prompt with no restore       -> reprocess time again

Stdlib only. One server at a time. Writes results.json into --work-dir.

  scripts/bench/slot_save_restore.py \
      --server /path/to/llama-server --model /path/to/model.gguf \
      --work-dir /path/on/a/real/disk --tokens 100000 --threads 8
"""

import argparse
import json
import os
import random
import shutil
import signal
import subprocess
import sys
import time
import urllib.error
import urllib.request

WORDS = (
  "the of and to in is that for it as was with be by on not he this are or his from at which "
  "but have an had they you were their one all we can her has there been if more when will "
  "would who so no out up said what its about than into them only some could time these two "
  "may then do first any my now such like our over man me even most made after also did many "
  "before must through years where much your way well down should because each just those "
  "people how too little state good very make world still own see men work long get here "
  "between both life being under never day same another know while last might us great old "
  "year off come since against go came right used take three model cache token prompt slot"
).split()


def http(method, url, body=None, timeout=7200):
  data = None if body is None else json.dumps(body).encode()
  req = urllib.request.Request(url, data=data, method=method)
  if data is not None:
    req.add_header("Content-Type", "application/json")
  with urllib.request.urlopen(req, timeout=timeout) as resp:
    return json.loads(resp.read())


class Server:
  def __init__(self, args, log_name):
    self.args = args
    self.base = f"http://127.0.0.1:{args.port}"
    self.log_path = os.path.join(args.work_dir, log_name)
    self.proc = None

  def start(self):
    argv = [
      self.args.server,
      "-m", self.args.model,
      "-c", str(self.args.ctx),
      "-np", "1",
      "-t", str(self.args.threads),
      "-tb", str(self.args.threads),
      "--host", "127.0.0.1",
      "--port", str(self.args.port),
      "--slot-save-path", self.args.slot_dir,
      "--no-webui",
    ] + self.args.server_arg
    t0 = time.monotonic()
    self.proc = subprocess.Popen(argv, stdout=open(self.log_path, "w"), stderr=subprocess.STDOUT)
    while True:
      if self.proc.poll() is not None:
        sys.exit(f"llama-server exited early, see {self.log_path}")
      try:
        if http("GET", self.base + "/health", timeout=2).get("status") == "ok":
          break
      except (urllib.error.URLError, OSError, ValueError):
        pass
      time.sleep(0.1)
    return time.monotonic() - t0

  def stop(self):
    self.proc.send_signal(signal.SIGTERM)
    self.proc.wait(timeout=60)
    self.proc = None

  def complete(self, tokens):
    t0 = time.monotonic()
    out = http("POST", self.base + "/completion", {
      "prompt": tokens,
      "n_predict": 1,
      "cache_prompt": True,
      "id_slot": 0,
      "temperature": 0,
    })
    wall = time.monotonic() - t0
    timings = out.get("timings", {})
    return {
      "wall_s": round(wall, 3),
      "prompt_n": timings.get("prompt_n"),
      "cache_n": timings.get("cache_n"),
      "prompt_ms": timings.get("prompt_ms"),
      "prompt_per_second": timings.get("prompt_per_second"),
    }

  def slot_action(self, action, filename):
    t0 = time.monotonic()
    out = http("POST", f"{self.base}/slots/0?action={action}", {"filename": filename})
    out["wall_s"] = round(time.monotonic() - t0, 3)
    return out


def build_tokens(server, n):
  rng = random.Random(1234)
  tokens = []
  while len(tokens) < n:
    text = " ".join(rng.choice(WORDS) for _ in range(20000))
    tokens += http("POST", server.base + "/tokenize", {"content": text, "add_special": not tokens})["tokens"]
  return tokens[:n]


def evict_from_page_cache(path):
  fd = os.open(path, os.O_RDONLY)
  try:
    os.fsync(fd)
    os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_DONTNEED)
  finally:
    os.close(fd)


def cached_bytes(path):
  if not shutil.which("fincore"):
    return None
  out = subprocess.run(["fincore", "--bytes", "--noheadings", "--output", "RES", path],
                       capture_output=True, text=True)
  return int(out.stdout.split()[0]) if out.returncode == 0 and out.stdout.split() else None


def main():
  ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
  ap.add_argument("--server", required=True)
  ap.add_argument("--model", required=True)
  ap.add_argument("--work-dir", required=True, help="on a real disk, not tmpfs: the save file is ~33 KiB/token for a 1B model")
  ap.add_argument("--tokens", type=int, default=100000)
  ap.add_argument("--ctx", type=int, default=0, help="default: tokens + 2048")
  ap.add_argument("--threads", type=int, default=8)
  ap.add_argument("--port", type=int, default=18431)
  ap.add_argument("--followup-tokens", type=int, default=32)
  ap.add_argument("--skip-second-reprocess", action="store_true")
  ap.add_argument("--server-arg", action="append", default=[], help="extra llama-server argument, repeatable")
  args = ap.parse_args()
  args.ctx = args.ctx or args.tokens + 2048
  args.slot_dir = os.path.join(args.work_dir, "slots")
  os.makedirs(args.slot_dir, exist_ok=True)
  save_name = "spike.bin"
  save_path = os.path.join(args.slot_dir, save_name)
  results = {"tokens": args.tokens, "ctx": args.ctx, "threads": args.threads,
             "model": os.path.basename(args.model),
             "server_version": subprocess.run([args.server, "--version"], capture_output=True, text=True).stderr.strip()}

  def checkpoint():
    with open(os.path.join(args.work_dir, "results.json"), "w") as f:
      json.dump(results, f, indent=2)

  srv = Server(args, "server-1-process-save.log")
  results["load_s"] = round(srv.start(), 3)
  tokens = build_tokens(srv, args.tokens)
  followup = tokens + tokens[1:1 + args.followup_tokens]
  results["reprocess"] = srv.complete(tokens)
  checkpoint()
  results["warm_same_process"] = srv.complete(followup)
  # Put the slot back to exactly the N-token prompt before saving.
  srv.complete(tokens)
  results["save"] = srv.slot_action("save", save_name)
  results["save"]["file_bytes"] = os.path.getsize(save_path)
  # The save returns once the bytes are in page cache; this is the time to reach the disk.
  t0 = time.monotonic()
  fd = os.open(save_path, os.O_RDONLY)
  os.fsync(fd)
  os.close(fd)
  results["save"]["fsync_s"] = round(time.monotonic() - t0, 3)
  srv.stop()
  checkpoint()

  for label, evict in (("restore_page_cache", False), ("restore_cold_file", True)):
    if evict:
      evict_from_page_cache(save_path)
    entry = {"file_cached_bytes_before": cached_bytes(save_path)}
    srv = Server(args, f"server-{label}.log")
    entry["load_s"] = round(srv.start(), 3)
    entry["restore"] = srv.slot_action("restore", save_name)
    entry["followup"] = srv.complete(followup)
    srv.stop()
    results[label] = entry
    checkpoint()

  if not args.skip_second_reprocess:
    srv = Server(args, "server-4-reprocess.log")
    entry = {"load_s": round(srv.start(), 3)}
    entry["followup"] = srv.complete(followup)
    srv.stop()
    results["reprocess_after_restart"] = entry
    checkpoint()

  print(json.dumps(results, indent=2))


if __name__ == "__main__":
  main()
