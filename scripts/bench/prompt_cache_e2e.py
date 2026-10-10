#!/usr/bin/env python3
"""End-to-end check of the prompt cache across a proxy eviction.

Starts an isolated llamastash daemon (own state / config / cache dirs, own
proxy port), then through the proxy:

  1. sends a long prompt (auto-start, prompt processed from nothing)
  2. sends the follow-up while the model is still loaded
  3. waits for the idle sweep to unload the model
  4. sends the next follow-up (auto-start again)

and reports, for each request, how many prompt tokens llama-server took from
its cache and how long prompt processing took. With the prompt cache on,
request 4 should read almost everything from the cache; with `--disable` it
reprocesses the whole prompt, which is the comparison.

Stdlib only. Writes result.json and the daemon's log lines into --work-dir.

  scripts/bench/prompt_cache_e2e.py --llamastash target/debug/llamastash \
      --llama-server /path/to/llama-server --model /path/to/model.gguf \
      --work-dir /path/on/a/real/disk [--disable]
"""

import argparse
import json
import os
import random
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

CONFIG = """\
model_paths:
  - {model_dir}
disable_default_cache_paths:
  huggingface: true
  ollama: true
  lm_studio: true
backend:
  llamacpp:
    servers:
      - binary: {llama_server}
    prompt_cache:
      enabled: {enabled}
      min_tokens: 1024
  lemonade:
    enabled: false
  vllm:
    enabled: false
  sglang:
    enabled: false
daemon:
  port_range:
    start: {launch_port}
    end: {launch_port_end}
proxy:
  idle_ttl_secs: {idle_ttl}
presets:
  {model_file}:
    default: e2e
    entries:
      e2e:
        knobs:
          ctx-size: {ctx}
          threads: {threads}
"""


def http_json(method, url, body=None, timeout=3600):
  data = None if body is None else json.dumps(body).encode()
  req = urllib.request.Request(url, data=data, method=method)
  if data is not None:
    req.add_header("Content-Type", "application/json")
  with urllib.request.urlopen(req, timeout=timeout) as resp:
    return json.loads(resp.read())


def main():
  ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
  ap.add_argument("--llamastash", required=True, help="binary built from the working tree")
  ap.add_argument("--llama-server", required=True)
  ap.add_argument("--model", required=True)
  ap.add_argument("--work-dir", required=True, help="on a real disk, not tmpfs")
  ap.add_argument("--prompt-words", type=int, default=15000, help="about one token per word")
  ap.add_argument("--ctx", type=int, default=32768)
  ap.add_argument("--threads", type=int, default=8)
  ap.add_argument("--idle-ttl", type=int, default=20)
  ap.add_argument("--proxy-port", type=int, default=21435)
  ap.add_argument("--launch-port", type=int, default=43100)
  ap.add_argument("--disable", action="store_true", help="run with prompt_cache.enabled: false")
  args = ap.parse_args()

  work = os.path.abspath(args.work_dir)
  dirs = {name: os.path.join(work, name) for name in ("state", "config", "cache", "hf")}
  for d in dirs.values():
    os.makedirs(d, exist_ok=True)
  model = os.path.abspath(args.model)
  with open(os.path.join(dirs["config"], "config.yaml"), "w") as f:
    f.write(CONFIG.format(
      model_dir=os.path.dirname(model),
      model_file=os.path.basename(model),
      llama_server=os.path.abspath(args.llama_server),
      enabled="false" if args.disable else "true",
      launch_port=args.launch_port,
      launch_port_end=args.launch_port + 20,
      idle_ttl=args.idle_ttl,
      ctx=args.ctx,
      threads=args.threads,
    ))
  env = {k: v for k, v in os.environ.items() if not k.startswith("LLAMASTASH_")}
  env.update(
    LLAMASTASH_STATE_DIR=dirs["state"],
    LLAMASTASH_CONFIG_DIR=dirs["config"],
    LLAMASTASH_CACHE_DIR=dirs["cache"],
    HF_HOME=dirs["hf"],
  )
  cli = os.path.abspath(args.llamastash)
  proxy = f"http://127.0.0.1:{args.proxy_port}"
  slot_dir = os.path.join(dirs["cache"], "slot-cache")
  log_path = os.path.join(dirs["cache"], "logs", "llamastash.log")

  def run(*argv):
    return subprocess.run([cli, *argv], env=env, capture_output=True, text=True)

  def launches():
    # `status.models` lists running launches only.
    out = run("status", "--json")
    if out.returncode != 0:
      sys.exit(f"status failed: {out.stderr}")
    return json.loads(out.stdout).get("models", [])

  def slot_files():
    if not os.path.isdir(slot_dir):
      return []
    return sorted((n, os.path.getsize(os.path.join(slot_dir, n))) for n in os.listdir(slot_dir))

  def chat(messages):
    t0 = time.monotonic()
    out = http_json("POST", proxy + "/v1/chat/completions", {
      "model": model_id,
      "messages": messages,
      "max_tokens": 8,
      "temperature": 0,
    })
    wall = time.monotonic() - t0
    timings = out.get("timings", {})
    reply = out["choices"][0]["message"].get("content") or ""
    return reply, {
      "wall_s": round(wall, 2),
      "prompt_tokens": out.get("usage", {}).get("prompt_tokens"),
      "cached_tokens": timings.get("cache_n"),
      "processed_tokens": timings.get("prompt_n"),
      "prompt_ms": timings.get("prompt_ms"),
    }

  started = run("daemon", "start", "--proxy-port", str(args.proxy_port))
  if started.returncode != 0:
    sys.exit(f"daemon start failed: {started.stderr}")
  result = {"prompt_cache_enabled": not args.disable, "idle_ttl_secs": args.idle_ttl}
  try:
    deadline = time.monotonic() + 60
    model_id = None
    stem = os.path.basename(model).removesuffix(".gguf")
    while model_id is None:
      try:
        ids = [m["id"] for m in http_json("GET", proxy + "/v1/models", timeout=5)["data"]]
        model_id = next((i for i in ids if stem in i), None)
      except (urllib.error.URLError, OSError, ValueError, KeyError):
        pass
      if model_id is None:
        if time.monotonic() > deadline:
          sys.exit("the proxy never listed the model")
        time.sleep(0.5)
    result["model"] = model_id

    rng = random.Random(1234)
    document = " ".join(rng.choice(WORDS) for _ in range(args.prompt_words))
    messages = [
      {"role": "system", "content": "Reference text follows.\n" + document},
      {"role": "user", "content": "Reply with the single word: one"},
    ]
    reply, result["1_first_request"] = chat(messages)
    messages += [{"role": "assistant", "content": reply},
                 {"role": "user", "content": "Reply with the single word: two"}]
    reply, result["2_followup_still_loaded"] = chat(messages)

    deadline = time.monotonic() + args.idle_ttl + 180
    while launches():
      if time.monotonic() > deadline:
        sys.exit("the idle sweep never unloaded the model")
      time.sleep(1)
    result["slot_files_while_unloaded"] = slot_files()

    messages += [{"role": "assistant", "content": reply},
                 {"role": "user", "content": "Reply with the single word: three"}]
    reply, result["3_followup_after_unload"] = chat(messages)
    result["slot_files_after_reload"] = slot_files()
    requests = run("requests", "--json")
    if requests.returncode == 0:
      result["requests"] = json.loads(requests.stdout)
  finally:
    run("daemon", "stop")
  if os.path.exists(log_path):
    with open(log_path) as f:
      result["daemon_log"] = [line.rstrip() for line in f if "prompt cache:" in line or "proxy eviction:" in line]
  with open(os.path.join(work, "result.json"), "w") as f:
    json.dump(result, f, indent=2)
  print(json.dumps(result, indent=2))


if __name__ == "__main__":
  main()
