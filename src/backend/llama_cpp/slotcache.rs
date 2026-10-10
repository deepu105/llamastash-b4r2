//! Cross-unload KV prompt cache: save each idle slot's prompt cache to disk
//! before an eviction stop and load it back into the same slot ids after the
//! next launch of the same model is servable but before it turns `Ready`.
//!
//! llama-server writes a slot's state through `--slot-save-path` +
//! `POST /slots/{id}?action=save|restore`. The engine validates a restored
//! file against the live model (tokens, KV shape) and rejects mismatches, so
//! every call here is best-effort: failures are logged, the launch proceeds,
//! and a rejected file is dropped so it can't fail again.
//!
//! Files live under `cache_dir/slot-cache/<hash-of-model-path>/slot-<id>.bin`.
//! The directory is derived from the model path alone, so a stop and the next
//! launch of the same file always agree on where the files are. Matching a
//! returning conversation to the right slot is left to the engine's own
//! prompt-similarity slot routing — llamastash just repopulates every slot
//! that had cache worth keeping.
//!
//! Sizing is bounded by `backend.llamacpp.slot_cache_max_bytes` (LRU by file
//! mtime across all model dirs, enforced after each save) and
//! `backend.llamacpp.slot_cache_ttl_days` (stale files are deleted at save
//! time, and skipped-and-deleted at restore time).

use std::path::{Path, PathBuf};
use std::time::{Duration, SystemTime, UNIX_EPOCH};

use serde_json::Value;
use sha2::{Digest, Sha256};
use tokio::io::{AsyncReadExt, AsyncWriteExt};
use tokio::net::TcpStream;

use super::LlamaCppConfig;
use crate::launch::mode::LaunchMode;
use crate::launch::params::LaunchParams;

/// `launch_config` key carrying the per-model slot-save directory the launch
/// was given (absent when the feature is off for this launch). `compose` turns
/// it into `--slot-save-path`; the stop/restore hooks read it back off the
/// persisted params.
pub const LLAMACPP_KNOB_SLOT_SAVE_PATH: &str = "slot_save_path";
/// `launch_config` keys carrying the size-cleanup policy so the stop/restore
/// hooks need no config access of their own.
pub const LLAMACPP_KNOB_SLOT_CACHE_TTL_DAYS: &str = "slot_cache_ttl_days";
pub const LLAMACPP_KNOB_SLOT_CACHE_MAX_BYTES: &str = "slot_cache_max_bytes";

/// Prompts shorter than this are not worth a save round-trip: reprocessing
/// them costs about as much as writing and reading the file back.
const MIN_PROMPT_TOKENS: u64 = 1024;
/// Hard ceilings on the eviction stop path (the sweeper waits on this) and on
/// the pre-Ready restore (the launch's readiness waits on it).
/// Both far exceed a healthy multi-GB save/restore on NVMe while bounding a
/// wedged engine from stalling every stop forever.
const SAVE_BUDGET: Duration = Duration::from_secs(30);
const RESTORE_BUDGET: Duration = Duration::from_secs(120);
const SLOTS_FETCH_TIMEOUT: Duration = Duration::from_secs(3);
const SLOT_OP_TIMEOUT: Duration = Duration::from_secs(60);

/// Project the config onto `launch_config`, fresh each launch (the
/// config-projection channel `jinja` / `strict_fit` use). When the feature is
/// off, or the directory cannot be created, every key is removed so an
/// inherited value can never survive.
pub fn seed(cfg: &LlamaCppConfig, params: &mut LaunchParams) {
  seed_in(slot_cache_root().as_deref(), cfg, params);
}

/// The root-resolved half of [`seed`], split out so tests drive it with a
/// temp root instead of the env-var-overridden cache dir.
fn seed_in(root: Option<&Path>, cfg: &LlamaCppConfig, params: &mut LaunchParams) {
  // Embedding / rerank launches carry no conversation to keep warm.
  if !cfg.slot_cache || params.mode != LaunchMode::Chat {
    clear(params);
    return;
  }
  let Some(dir) = root.map(|root| model_slot_dir(root, &params.model_path)) else {
    clear(params);
    return;
  };
  if let Err(e) = std::fs::create_dir_all(&dir) {
    log::warn!(
      "slot-cache: cannot create {}: {e}; launching without prompt cache",
      dir.display()
    );
    clear(params);
    return;
  }
  params.launch_config.insert(
    LLAMACPP_KNOB_SLOT_SAVE_PATH.into(),
    dir.display().to_string(),
  );
  params.launch_config.insert(
    LLAMACPP_KNOB_SLOT_CACHE_TTL_DAYS.into(),
    cfg.slot_cache_ttl_days.to_string(),
  );
  params.launch_config.insert(
    LLAMACPP_KNOB_SLOT_CACHE_MAX_BYTES.into(),
    cfg.slot_cache_max_bytes.to_string(),
  );
}

fn clear(params: &mut LaunchParams) {
  params.launch_config.remove(LLAMACPP_KNOB_SLOT_SAVE_PATH);
  params
    .launch_config
    .remove(LLAMACPP_KNOB_SLOT_CACHE_TTL_DAYS);
  params
    .launch_config
    .remove(LLAMACPP_KNOB_SLOT_CACHE_MAX_BYTES);
}

fn slot_cache_root() -> Option<PathBuf> {
  crate::util::paths::cache_dir().map(|c| c.join("slot-cache"))
}

/// The per-model save directory: `<root>/<sha256(model path)[..16]>`. Pure, so
/// the stop and next-launch derivations cannot drift.
pub fn model_slot_dir(root: &Path, model_path: &Path) -> PathBuf {
  let mut hasher = Sha256::new();
  hasher.update(model_path.to_string_lossy().as_bytes());
  let digest = hasher.finalize();
  let hex: String = digest[..8].iter().map(|b| format!("{b:02x}")).collect();
  root.join(hex)
}

fn dir_of(params: &LaunchParams) -> Option<PathBuf> {
  params
    .launch_config
    .get(LLAMACPP_KNOB_SLOT_SAVE_PATH)
    .map(PathBuf::from)
}

fn knob_u64(params: &LaunchParams, key: &str) -> Option<u64> {
  params
    .launch_config
    .get(key)
    .and_then(|s| s.parse::<u64>().ok())
}

/// Save every idle slot's prompt cache to `params`' slot directory, then
/// enforce the ttl / size-cap cleanup. Never fails: any transport or file
/// error is logged and the stop proceeds.
pub async fn prepare_to_stop(port: u16, params: &LaunchParams) {
  let Some(dir) = dir_of(params) else {
    return;
  };
  let slots = match fetch_slots(port).await {
    Ok(s) => s,
    Err(e) => {
      log::debug!("slot-cache: /slots fetch failed on stop (port {port}): {e}");
      return;
    }
  };
  let outcome = tokio::time::timeout(SAVE_BUDGET, save_idle_slots(port, &dir, &slots)).await;
  if outcome.is_err() {
    log::warn!("slot-cache: save budget ({SAVE_BUDGET:?}) exceeded on port {port}");
  }
  if let (Some(ttl_days), Some(max_bytes), Some(root)) = (
    knob_u64(params, LLAMACPP_KNOB_SLOT_CACHE_TTL_DAYS),
    knob_u64(params, LLAMACPP_KNOB_SLOT_CACHE_MAX_BYTES),
    dir.parent(),
  ) {
    cleanup(root, ttl_days, max_bytes);
  }
}

async fn save_idle_slots(port: u16, dir: &Path, slots: &[Value]) {
  let mut saved_tokens = 0u64;
  let mut saved_slots = 0usize;
  for slot in slots {
    let Some(id) = slot.get("id").and_then(Value::as_u64) else {
      continue;
    };
    if slot.get("is_processing").and_then(Value::as_bool) == Some(true) {
      log::debug!("slot-cache: slot {id} busy at stop, skipping save");
      continue;
    }
    let filename = format!("slot-{id}.bin");
    match slot_action(port, id, "save", &filename).await {
      Ok(resp) => {
        let n = resp.get("n_saved").and_then(Value::as_u64).unwrap_or(0);
        if n < MIN_PROMPT_TOKENS {
          // Not worth keeping — drop the file the engine just wrote.
          let _ = std::fs::remove_file(dir.join(&filename));
          continue;
        }
        saved_tokens += n;
        saved_slots += 1;
        log::debug!(
          "slot-cache: saved slot {id} ({n} tokens, {} bytes, {:.0} ms)",
          resp.get("n_written").and_then(Value::as_u64).unwrap_or(0),
          timing_ms(&resp, "save_ms"),
        );
      }
      Err(e) => log::debug!("slot-cache: save slot {id} failed: {e}"),
    }
  }
  if saved_slots > 0 {
    log::info!(
      "slot-cache: saved {saved_tokens} prompt tokens across {saved_slots} slot(s) for restart"
    );
  }
}

/// Restore this launch's saved slot files into the same slot ids of the
/// freshly-ready server. Every candidate file is consumed (deleted) either
/// way: a success now lives in the engine's cache, a failure is invalid for
/// this launch and can never restore.
pub async fn on_launch_ready(port: u16, params: &LaunchParams) {
  let Some(dir) = dir_of(params) else {
    return;
  };
  let ttl_days = knob_u64(params, LLAMACPP_KNOB_SLOT_CACHE_TTL_DAYS);
  let Some(mut pending) = slot_files(&dir, ttl_days) else {
    return;
  };
  if pending.is_empty() {
    return;
  }
  let slots = match fetch_slots(port).await {
    Ok(s) => s,
    Err(e) => {
      log::debug!("slot-cache: /slots fetch failed on ready (port {port}): {e}");
      return;
    }
  };
  // A restore lands only in a slot that exists in this launch (parallelism can
  // differ between the two launches) and is not mid-generation.
  pending.retain(|(id, _)| {
    slots.iter().any(|s| {
      s.get("id").and_then(Value::as_u64) == Some(*id)
        && s.get("is_processing").and_then(Value::as_bool) != Some(true)
    })
  });
  if pending.is_empty() {
    return;
  }
  let budget = tokio::time::timeout(RESTORE_BUDGET, async {
    let mut restored_tokens = 0u64;
    let mut restored_slots = 0usize;
    for (id, filename) in pending {
      match slot_action(port, id, "restore", &filename).await {
        Ok(resp) => {
          restored_tokens += resp.get("n_restored").and_then(Value::as_u64).unwrap_or(0);
          restored_slots += 1;
          log::debug!(
            "slot-cache: restored slot {id} ({} tokens, {:.0} ms)",
            resp.get("n_restored").and_then(Value::as_u64).unwrap_or(0),
            timing_ms(&resp, "restore_ms")
          );
        }
        Err(e) => {
          // Invalid for this model/config (another model's file, a different
          // KV layout, a stale file after a model swap) or a wedged engine.
          log::warn!("slot-cache: restore slot {id} failed, dropping file: {e}");
        }
      }
      let _ = std::fs::remove_file(dir.join(&filename));
    }
    if restored_slots > 0 {
      log::info!(
        "slot-cache: restored {restored_tokens} prompt tokens into {restored_slots} slot(s) \
         before Ready"
      );
    }
  })
  .await;
  if budget.is_err() {
    log::warn!("slot-cache: restore budget ({RESTORE_BUDGET:?}) exceeded on port {port}");
  }
}

/// `GET /slots` → the JSON array of slot objects.
async fn fetch_slots(port: u16) -> Result<Vec<Value>, String> {
  let body = http_exchange(port, "GET", "/slots", None, SLOTS_FETCH_TIMEOUT).await?;
  serde_json::from_str(&body).map_err(|e| format!("slots parse: {e}"))
}

/// `POST /slots/{id}?action={action}` with the `filename` body.
async fn slot_action(port: u16, id: u64, action: &str, filename: &str) -> Result<Value, String> {
  let body = format!("{{\"filename\":\"{filename}\"}}");
  let text = http_exchange(
    port,
    "POST",
    &format!("/slots/{id}?action={action}"),
    Some(&body),
    SLOT_OP_TIMEOUT,
  )
  .await?;
  serde_json::from_str(&text).map_err(|e| format!("{action} parse: {e}"))
}

/// Minimal one-shot HTTP over loopback TCP — the same no-extra-dep stance as
/// [`super::actuals`]. Returns the body on 2xx; any other status is an error
/// carrying the leading response text (llama-server's error JSON).
async fn http_exchange(
  port: u16,
  method: &str,
  path: &str,
  body: Option<&str>,
  timeout: Duration,
) -> Result<String, String> {
  let fut = async {
    let mut sock = TcpStream::connect(("127.0.0.1", port))
      .await
      .map_err(|e| format!("connect {port}: {e}"))?;
    let mut req =
      format!("{method} {path} HTTP/1.1\r\nHost: 127.0.0.1:{port}\r\nConnection: close\r\n");
    if body.is_some() {
      req.push_str("Content-Type: application/json\r\n");
    }
    if let Some(b) = body {
      req.push_str(&format!("Content-Length: {}\r\n", b.len()));
    }
    req.push_str("\r\n");
    sock
      .write_all(req.as_bytes())
      .await
      .map_err(|e| format!("write: {e}"))?;
    if let Some(b) = body {
      sock
        .write_all(b.as_bytes())
        .await
        .map_err(|e| format!("write: {e}"))?;
    }
    let mut buf = Vec::with_capacity(4096);
    let mut chunk = [0u8; 8192];
    loop {
      let n = sock
        .read(&mut chunk)
        .await
        .map_err(|e| format!("read: {e}"))?;
      if n == 0 {
        break;
      }
      buf.extend_from_slice(&chunk[..n]);
      if buf.len() > 1024 * 1024 {
        break;
      }
    }
    Ok::<_, String>(buf)
  };
  let raw = tokio::time::timeout(timeout, fut)
    .await
    .map_err(|_| format!("{method} {path} timed out"))??;
  let text = String::from_utf8_lossy(&raw).into_owned();
  let (head, resp_body) = text
    .split_once("\r\n\r\n")
    .ok_or_else(|| format!("{method} {path}: malformed response"))?;
  let status_ok = head.lines().next().is_some_and(|l| {
    l.split_whitespace()
      .nth(1)
      .is_some_and(|s| s.starts_with('2'))
  });
  if status_ok {
    Ok(resp_body.to_string())
  } else {
    let lead = resp_body.chars().take(200).collect::<String>();
    Err(format!("{method} {path}: {}", lead.trim()))
  }
}

/// The engine's own timing field (`save_ms` / `restore_ms`), 0 when absent.
fn timing_ms(resp: &Value, key: &str) -> f64 {
  resp
    .get("timings")
    .and_then(|t| t.get(key))
    .and_then(Value::as_f64)
    .unwrap_or(0.0)
}

/// `(slot id, filename)` for each `slot-<id>.bin` in `dir`, dropping
/// (and deleting) files past the ttl. `None` when the dir is unreadable.
fn slot_files(dir: &Path, ttl_days: Option<u64>) -> Option<Vec<(u64, String)>> {
  let max_age = ttl_days.map(|d| Duration::from_secs(d.saturating_mul(86_400)));
  let now = SystemTime::now();
  let mut out = Vec::new();
  for entry in std::fs::read_dir(dir).ok()?.flatten() {
    let name = entry.file_name().to_string_lossy().to_string();
    let Some(id) = slot_id_of(&name) else {
      continue;
    };
    if let Some(max_age) = max_age {
      let stale = entry
        .metadata()
        .and_then(|m| m.modified())
        .map(|m| now.duration_since(m).unwrap_or(Duration::ZERO) > max_age)
        .unwrap_or(false);
      if stale {
        let _ = std::fs::remove_file(entry.path());
        continue;
      }
    }
    out.push((id, name));
  }
  out.sort();
  Some(out)
}

fn slot_id_of(name: &str) -> Option<u64> {
  name
    .strip_prefix("slot-")?
    .strip_suffix(".bin")
    .and_then(|s| s.parse().ok())
}

/// Delete stale files everywhere under `root`, then drop the least-recently
/// saved files until the total fits `max_bytes`. Best-effort per entry.
pub fn cleanup(root: &Path, ttl_days: u64, max_bytes: u64) {
  let now = SystemTime::now();
  let max_age = Duration::from_secs(ttl_days.saturating_mul(86_400));
  let mut files: Vec<(PathBuf, SystemTime, u64)> = Vec::new();
  let Ok(model_dirs) = std::fs::read_dir(root) else {
    return;
  };
  for model_dir in model_dirs.flatten() {
    let Ok(entries) = std::fs::read_dir(model_dir.path()) else {
      continue;
    };
    for entry in entries.flatten() {
      let Ok(meta) = entry.metadata() else { continue };
      if !meta.is_file() {
        continue;
      }
      let mtime = meta.modified().unwrap_or(UNIX_EPOCH);
      if now.duration_since(mtime).unwrap_or(Duration::ZERO) > max_age {
        let _ = std::fs::remove_file(entry.path());
        continue;
      }
      files.push((entry.path(), mtime, meta.len()));
    }
  }
  let mut total: u64 = files.iter().map(|(_, _, len)| len).sum();
  if total <= max_bytes {
    return;
  }
  files.sort_by_key(|(_, mtime, _)| *mtime);
  for (path, _, len) in files {
    if total <= max_bytes {
      break;
    }
    match std::fs::remove_file(&path) {
      Ok(()) => {
        total -= len;
        log::debug!(
          "slot-cache: evicted {} ({len} bytes) over the size cap",
          path.display()
        );
      }
      Err(e) => log::debug!("slot-cache: cap delete {} failed: {e}", path.display()),
    }
  }
}

#[cfg(test)]
mod tests {
  use super::*;

  fn temp_dir(label: &str) -> PathBuf {
    crate::util::test_temp::unique_temp_dir(&format!("slotcache-{label}"))
  }

  fn write_file(path: &Path, len: u64, mtime: SystemTime) {
    std::fs::write(path, vec![b'x'; len as usize]).unwrap();
    let f = std::fs::OpenOptions::new().append(true).open(path).unwrap();
    f.set_modified(mtime).unwrap();
  }

  #[test]
  fn model_slot_dir_is_stable_and_path_specific() {
    let root = Path::new("/root");
    let a1 = model_slot_dir(root, Path::new("/models/a.gguf"));
    let a2 = model_slot_dir(root, Path::new("/models/a.gguf"));
    let b = model_slot_dir(root, Path::new("/models/b.gguf"));
    assert_eq!(a1, a2);
    assert_ne!(a1, b);
    assert!(a1.starts_with(root));
    assert_eq!(a1.file_name().unwrap().to_string_lossy().len(), 16);
  }

  #[test]
  fn slot_id_parses_only_our_filenames() {
    assert_eq!(slot_id_of("slot-0.bin"), Some(0));
    assert_eq!(slot_id_of("slot-12.bin"), Some(12));
    assert_eq!(slot_id_of("slot-x.bin"), None);
    assert_eq!(slot_id_of("slot-1.txt"), None);
    assert_eq!(slot_id_of("other.bin"), None);
  }

  #[test]
  fn slot_files_skips_other_names_and_drops_stale() {
    let dir = temp_dir("files");
    let old = SystemTime::now() - Duration::from_secs(10 * 86_400);
    write_file(&dir.join("slot-0.bin"), 10, SystemTime::now());
    write_file(&dir.join("slot-1.bin"), 10, old);
    write_file(&dir.join("junk.bin"), 10, SystemTime::now());
    let found = slot_files(&dir, Some(7)).unwrap();
    assert_eq!(found, vec![(0, "slot-0.bin".to_string())]);
    assert!(!dir.join("slot-1.bin").exists(), "stale file was deleted");
    assert!(dir.join("junk.bin").exists(), "foreign file untouched");
    // Without a ttl the old file is kept too.
    write_file(&dir.join("slot-2.bin"), 10, old);
    let all = slot_files(&dir, None).unwrap();
    assert!(all.contains(&(2, "slot-2.bin".to_string())));
    std::fs::remove_dir_all(&dir).unwrap();
  }

  #[test]
  fn cleanup_prunes_by_age_then_lru_over_cap() {
    let root = temp_dir("cleanup");
    let m1 = root.join("m1");
    let m2 = root.join("m2");
    std::fs::create_dir_all(&m1).unwrap();
    std::fs::create_dir_all(&m2).unwrap();
    let now = SystemTime::now();
    write_file(
      &m1.join("slot-0.bin"),
      100,
      now - Duration::from_secs(9 * 86_400),
    ); // stale
    write_file(
      &m1.join("slot-1.bin"),
      100,
      now - Duration::from_secs(86_400),
    ); // older fresh
    write_file(&m2.join("slot-0.bin"), 50, now); // newest
                                                 // ttl drops the stale file; cap 120 < 250 evicts oldest-first.
    cleanup(&root, 7, 120);
    assert!(!m1.join("slot-0.bin").exists(), "ttl file deleted");
    assert!(
      !m1.join("slot-1.bin").exists(),
      "oldest over cap evicted first"
    );
    assert!(m2.join("slot-0.bin").exists(), "newest survives");
    // Under the cap nothing else goes.
    cleanup(&root, 7, 1_000_000);
    assert!(m2.join("slot-0.bin").exists());
    std::fs::remove_dir_all(&root).unwrap();
  }

  #[test]
  fn seed_projects_and_clears_with_config() {
    let root = temp_dir("seed");
    let mut params = LaunchParams::new(PathBuf::from("/models/seed-test.gguf"), LaunchMode::Chat);
    let cfg = LlamaCppConfig {
      slot_cache: true,
      ..LlamaCppConfig::default()
    };
    seed_in(Some(&root), &cfg, &mut params);
    assert!(params
      .launch_config
      .contains_key(LLAMACPP_KNOB_SLOT_SAVE_PATH));
    assert!(params
      .launch_config
      .contains_key(LLAMACPP_KNOB_SLOT_CACHE_TTL_DAYS));
    assert!(params
      .launch_config
      .contains_key(LLAMACPP_KNOB_SLOT_CACHE_MAX_BYTES));
    assert_eq!(
      params
        .launch_config
        .get(LLAMACPP_KNOB_SLOT_CACHE_TTL_DAYS)
        .map(String::as_str),
      Some(cfg.slot_cache_ttl_days.to_string().as_str())
    );
    let dir = PathBuf::from(
      params
        .launch_config
        .get(LLAMACPP_KNOB_SLOT_SAVE_PATH)
        .unwrap(),
    );
    assert!(dir.is_dir());
    assert_eq!(dir, model_slot_dir(&root, &params.model_path));
    // Config off clears everything, even an inherited value.
    seed(&LlamaCppConfig::default(), &mut params);
    assert!(!params
      .launch_config
      .contains_key(LLAMACPP_KNOB_SLOT_SAVE_PATH));
    assert!(!params
      .launch_config
      .contains_key(LLAMACPP_KNOB_SLOT_CACHE_TTL_DAYS));
    // Embedding launches never carry the keys.
    params.mode = LaunchMode::Embedding;
    seed_in(Some(&root), &cfg, &mut params);
    assert!(!params
      .launch_config
      .contains_key(LLAMACPP_KNOB_SLOT_SAVE_PATH));
    // No resolvable cache dir ⇒ the knob is not set, not dangling.
    params.mode = LaunchMode::Chat;
    params.launch_config.insert(
      LLAMACPP_KNOB_SLOT_SAVE_PATH.into(),
      "inherited-value".into(),
    );
    seed_in(None, &cfg, &mut params);
    assert!(!params
      .launch_config
      .contains_key(LLAMACPP_KNOB_SLOT_SAVE_PATH));
    let _ = std::fs::remove_dir_all(root);
  }

  #[tokio::test]
  async fn prepare_and_ready_noop_without_knob() {
    let params = LaunchParams::new(PathBuf::from("/models/x.gguf"), LaunchMode::Chat);
    // No knob ⇒ no network attempt at all (nothing listens on port 1).
    prepare_to_stop(1, &params).await;
    on_launch_ready(1, &params).await;
  }

  /// Canned-response TCP stub: answers one request, returns the request line
  /// and body it saw. Exercises the raw-HTTP transport and status handling
  /// without a real engine.
  async fn stub_http(response: String) -> (u16, tokio::sync::oneshot::Receiver<String>) {
    let listener = tokio::net::TcpListener::bind(("127.0.0.1", 0))
      .await
      .unwrap();
    let port = listener.local_addr().unwrap().port();
    let (tx, rx) = tokio::sync::oneshot::channel();
    tokio::spawn(async move {
      let (mut sock, _) = listener.accept().await.unwrap();
      let mut buf = vec![0u8; 4096];
      let n = sock.read(&mut buf).await.unwrap();
      let _ = tx.send(String::from_utf8_lossy(&buf[..n]).into_owned());
      let _ = sock.write_all(response.as_bytes()).await;
    });
    (port, rx)
  }

  fn http_stub(status: u16, body: &str) -> String {
    format!(
      "HTTP/1.1 {status} X\r\nContent-Type: application/json\r\nContent-Length: {}\r\n\r\n{body}",
      body.len()
    )
  }

  #[tokio::test]
  async fn http_exchange_returns_body_on_2xx() {
    let (port, _rx) = stub_http(http_stub(200, "[{\"id\":0}]")).await;
    let body = fetch_slots(port).await.unwrap();
    assert_eq!(body[0]["id"].as_u64(), Some(0));
  }

  #[tokio::test]
  async fn http_exchange_errors_carry_response_text() {
    let err = slot_action(
      {
        let (port, _rx) = stub_http(http_stub(
          500,
          "{\"error\":\"Unable to restore slot: bad file\"}",
        ))
        .await;
        port
      },
      0,
      "restore",
      "slot-0.bin",
    )
    .await
    .unwrap_err();
    assert!(err.contains("Unable to restore slot"), "err: {err}");
  }

  #[tokio::test]
  async fn slot_action_shapes_the_request() {
    let (port, rx) = stub_http(http_stub(200, "{\"n_saved\":5}")).await;
    let resp = slot_action(port, 3, "save", "slot-3.bin").await.unwrap();
    assert_eq!(resp["n_saved"].as_u64(), Some(5));
    let seen = rx.await.unwrap();
    assert!(
      seen.starts_with("POST /slots/3?action=save HTTP/1.1"),
      "request line: {seen}"
    );
    assert!(seen.contains("{\"filename\":\"slot-3.bin\"}"));
    assert!(seen.contains("Content-Length: 25"));
  }
}
