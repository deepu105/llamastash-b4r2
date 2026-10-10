#!/usr/bin/env python3
"""What a saved llama.cpp slot cache will and will not restore into.

Companion to `kv_slot_spike.py`, answering the compatibility questions that
decide how a saved file is matched to a returning launch. Each phase starts its
own server (one at a time) with a small prompt so a run takes a couple of minutes:

  A  save a slot's cache at one context size, restore it into a launch with a
     larger context. Answers whether the file may be reused across a `--fit`
     launch that resolved a different `n_ctx`.
  B  restore a saved slot's file into a *different* slot id, then send the prompt
     it was saved from, and read `cache_n`. Answers whether the KV is bound to
     the slot it came from.
  C  after a restore, send an unrelated prompt to the same slot. A stale restore
     must cost time, never correctness.

  python3 scripts/spikes/kv_slot_compat.py --server-bin ... --model ... \
      --save-dir /scratch/slots --port 11599 --tokens 1500
"""

import argparse
import json
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

WORDS = (
  "the kernel parses a tensor header while the scheduler queues requests and the "
  "cache keeps only the prefix it can prove identical; a slot holds tokens and "
  "timings and sampler state so a reload means reading weights again and "
  "re-running prefill before decoding"
).split()


def http(method, url, body=None, timeout=600):
  data = json.dumps(body).encode() if body is not None else None
  req = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json"},
                               method=method)
  with urllib.request.urlopen(req, timeout=timeout) as resp:
    return resp.status, json.loads(resp.read().decode())


class Server:
  def __init__(self, args, ctx, np, log, extra=None):
    self.argv = [
      args.server_bin, "-m", args.model, "--host", "127.0.0.1", "--port", str(args.port),
      "-t", str(args.threads), "-c", str(ctx), "-np", str(np),
      "--slot-save-path", f"{args.save_dir}/",
    ] + (extra or [])
    self.log = log
    self.fh = None
    self.proc = None

  def __enter__(self):
    self.fh = open(self.log, "w")
    self.proc = subprocess.Popen(self.argv, stdout=self.fh, stderr=subprocess.STDOUT)
    deadline = time.time() + 300
    while time.time() < deadline:
      try:
        if http("GET", f"{self.url}/health")[1].get("status") == "ok":
          return self
      except (urllib.error.URLError, ConnectionError, TimeoutError):
        time.sleep(0.25)
    raise RuntimeError("server did not come up")

  def __exit__(self, *exc):
    self.proc.terminate()
    self.proc.wait(timeout=60)
    self.fh.close()
    return False

  @property
  def url(self):
    return f"http://127.0.0.1:{self.argv[self.argv.index('--port') + 1]}"

  def slots(self):
    return http("GET", f"{self.url}/slots?debug=1")[1]

  def completion(self, prompt, max_tokens=1):
    """Native `/completion`, which (unlike the OpenAI surface) names the slot."""
    _, out = http("POST", f"{self.url}/completion", {
      "prompt": prompt, "n_predict": max_tokens, "temperature": 0, "cache_prompt": True})
    t = out["timings"]
    return {"id_slot": out.get("id_slot"), "tokens_evaluated": out.get("tokens_evaluated"),
            "tokens_cached": out.get("tokens_cached"), "cache_n": t["cache_n"],
            "prompt_n": t["prompt_n"], "prompt_ms": round(t["prompt_ms"], 1),
            "text": out["content"]}

  def action(self, action, slot, filename):
    try:
      status, body = http("POST", f"{self.url}/slots/{slot}?action={action}",
                          {"filename": filename})
      return status, body
    except urllib.error.HTTPError as e:
      raw = e.read().decode()
      try:
        body = json.loads(raw)
      except json.JSONDecodeError:
        body = {"raw": raw[:200]}
      return e.code, body


def busy_slot(slots):
  for s in slots:
    if s.get("n_prompt_tokens", 0) > 0:
      return s["id"]
  return None


def main():
  ap = argparse.ArgumentParser(description=__doc__)
  ap.add_argument("--server-bin", required=True)
  ap.add_argument("--model", required=True)
  ap.add_argument("--save-dir", required=True)
  ap.add_argument("--port", type=int, default=11599)
  ap.add_argument("--threads", type=int, default=8)
  ap.add_argument("--tokens", type=int, default=1500)
  ap.add_argument("--log-dir", default=None)
  args = ap.parse_args()

  Path(args.save_dir).mkdir(parents=True, exist_ok=True)
  logs = Path(args.log_dir or args.save_dir) / "logs"
  logs.mkdir(parents=True, exist_ok=True)
  rng = __import__("random").Random(11)
  prompt = " ".join(rng.choice(WORDS) for _ in range(int(args.tokens * 1.2)))
  other = "Unrelated question: what is the capital of France? Answer in one word."
  out = {}

  # Phase A: save at ctx 8192 on a 2-slot server.
  with Server(args, 8192, 2, logs / "compat-a.log") as srv:
    before = srv.completion(prompt)
    saved_slot = busy_slot(srv.slots()) or 0
    status, body = srv.action("save", saved_slot, "compat-8k.bin")
    out["A_saved"] = {"slot": saved_slot, "request": before, "status": status,
                      "n_saved": body.get("n_saved")}
    print(f"A: {out['A_saved']}", flush=True)

  # Phase B: bigger context, same slot layout. Does the file still load?
  with Server(args, 16384, 2, logs / "compat-b.log") as srv:
    status, body = srv.action("restore", saved_slot, "compat-8k.bin")
    out["B_restore_into_bigger_ctx"] = {"status": status, "n_restored": body.get("n_restored"),
                                        "error": body.get("error", {}).get("message")
                                        if isinstance(body.get("error"), dict) else body.get("error")}
    after = srv.completion(prompt)
    out["B_restore_into_bigger_ctx"]["first_request"] = after
    print(f"B: {out['B_restore_into_bigger_ctx']}", flush=True)
    # C in the same process: an unrelated prompt on a slot that holds a stale cache.
    stale = srv.completion(other)
    out["C_unrelated_prompt_after_restore"] = stale
    print(f"C: {stale}", flush=True)

  # D: same layout, restore into the other slot, then ask for the original prompt.
  other_slot = 1 - saved_slot
  with Server(args, 8192, 2, logs / "compat-d.log") as srv:
    status, body = srv.action("restore", other_slot, "compat-8k.bin")
    out["D_restore_into_other_slot"] = {"target_slot": other_slot, "status": status,
                                        "n_restored": body.get("n_restored")}
    req = srv.completion(prompt)
    out["D_restore_into_other_slot"]["first_request"] = req
    print(f"D: {out['D_restore_into_other_slot']}", flush=True)

  # E/F: a cache written with one KV layout, read back by a launch using another
  # (q8_0 KV instead of f16). Either it is refused, or the answer it produces has
  # to be the answer that launch would have produced anyway - silently wrong KV
  # would show up as a different completion.
  q8 = ["--cache-type-k", "q8_0", "--cache-type-v", "q8_0"]
  with Server(args, 8192, 1, logs / "compat-e.log", extra=q8) as srv:
    status, body = srv.action("restore", 0, "compat-8k.bin")
    e = srv.completion(prompt, 8)
    out["E_mismatched_kv_layout"] = {"restore_status": status,
                                     "restore_error": body.get("error"), "request": e}
    print(f"E: {out['E_mismatched_kv_layout']}", flush=True)
  with Server(args, 8192, 1, logs / "compat-f.log", extra=q8) as srv:
    out["F_ground_truth_same_layout"] = srv.completion(prompt, 8)
    print(f"F: {out['F_ground_truth_same_layout']}", flush=True)
  out["E_matches_F"] = out["E_mismatched_kv_layout"]["request"]["text"] == \
    out["F_ground_truth_same_layout"]["text"]
  print(f"E text == F text: {out['E_matches_F']}", flush=True)

  print(json.dumps(out, indent=2))
  return 0

if __name__ == "__main__":
  sys.exit(main())
