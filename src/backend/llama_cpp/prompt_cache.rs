//! Prompt cache across an unload: save a llama-server's slot caches to disk before an
//! eviction stop, restore them into the next launch of the same model.
//!
//! A stopped `llama-server` takes its KV cache with it, so the next request to a
//! returning conversation reprocesses the whole prompt. llama-server can write a
//! slot's cached prompt to a file (`--slot-save-path` +
//! `POST /slots/{id}?action=save|restore`), and at 100k tokens that round trip is
//! orders of magnitude faster than reprocessing (see
//! `docs/spikes/2026-10-10-prompt-cache-across-unload.md`). A returning conversation
//! needs no matching on our side: llama.cpp routes a new request to the slot whose
//! cached prompt shares the longest prefix, so a restored slot is picked up by the
//! request that resumes the conversation.
//!
//! Every step here is best-effort. A failed save, a stale file, or a rejected restore
//! costs at most the cache benefit — never the launch. llama.cpp validates restore
//! files itself and answers an HTTP error (server stays healthy) when a file cannot be
//! loaded into the current engine, so the meta fingerprint below is an optimization
//! and a cleanup signal, not the safety line.
//!
//! Lifetimes: files live under `<state dir>/slot-cache/<entry>/`, one entry per model
//! file (path + size + mtime keyed). `meta.json` records the fingerprint; a mismatch
//! (the model file was replaced) deletes the entry. Entries age out (default 7 days)
//! and the cache root is capped by total size (default 16 GiB), both pruned at save
//! time. Files hold conversation KV — plaintext prompt data — so the entry dir is
//! 0700 and files 0600, inside the already-private state dir.

use std::path::{Path, PathBuf};
use std::time::{Duration, SystemTime, UNIX_EPOCH};

use serde::{Deserialize, Serialize};

use crate::launch::params::LaunchParams;

/// `launch_config` key carrying this launch's cache entry directory. Present only
/// when the feature is enabled for the launch; every entry point below is a no-op
/// without it. Rides on `launch_config` like `jinja` / `fit_ctx_floor` so the
/// neutral launch path never learns the feature exists.
pub const CFG_DIR: &str = "slot_cache_dir";
/// `launch_config` key: the resolved `llama-server` binary advertises
/// `--slot-save-path` (probed from its `--help` like the load-mode dialect).
pub const CFG_SUPPORTED: &str = "slot_cache_supported";
/// `launch_config` keys carrying the pruning limits, stashed at seed so the save
/// hook can read them off the launch's own params.
pub const CFG_MAX_BYTES: &str = "slot_cache_max_bytes";
pub const CFG_MAX_AGE_DAYS: &str = "slot_cache_max_age_days";

/// Directory under the state dir holding every entry.
const ROOT_DIR_NAME: &str = "slot-cache";
/// Per-entry fingerprint file (written by us, never touched by the engine).
const META_FILE: &str = "meta.json";
/// Prefix llama.cpp slot save files are written under, one per slot:
/// `slot-<id>.bin`. The engine writes them into the `--slot-save-path` directory.
const SLOT_FILE_PREFIX: &str = "slot-";
const SLOT_FILE_SUFFIX: &str = ".bin";

/// HTTP budget for the small control calls (`/slots`, `/props`).
const QUERY_TIMEOUT: Duration = Duration::from_secs(5);
/// HTTP budget for a save: the engine queues the action on the slot's task loop
/// and the response lands after the file write, about a second per 100k tokens
/// on local storage (2.7 s worst in the spike). The save runs on the eviction
/// path, where a wedged engine must not stall the sweep or make-room for long.
const SAVE_TIMEOUT: Duration = Duration::from_secs(30);
/// HTTP budget for a restore: same queue, but reading a multi-GB file back on a
/// cold drive is slower than writing it, and the wait only delays a launch's
/// Ready by that much (bounded per slot).
const RESTORE_TIMEOUT: Duration = Duration::from_secs(120);

/// The per-entry fingerprint written next to the slot files after a successful save.
#[derive(Debug, Clone, Serialize, Deserialize)]
struct Meta {
  /// The model file the saves belong to, verbatim from the launch params.
  model: String,
  /// Model file size and mtime at save time. A mismatch means the file behind the
  /// path is not the file the saves were made against.
  bytes: u64,
  mtime: u64,
  /// Context window the saving server reported (`/props`). Informational: a restore
  /// that cannot fit the saved prompt is rejected by the engine, and `--fit` may
  /// size two launches of one model differently, so a mismatch is not grounds for
  /// deleting the entry.
  #[serde(default)]
  n_ctx: Option<u64>,
  saved_at: u64,
}

/// `(size bytes, mtime seconds)` of a model file, or `None` when unreadable.
fn file_fingerprint(path: &Path) -> Option<(u64, u64)> {
  let md = std::fs::metadata(path).ok()?;
  let mtime = md
    .modified()
    .ok()
    .and_then(|t| t.duration_since(UNIX_EPOCH).ok())
    .map(|d| d.as_secs())
    .unwrap_or(0);
  Some((md.len(), mtime))
}

/// The cache root: `<state dir>/slot-cache`. `None` when no state dir resolves.
pub fn root_dir() -> Option<PathBuf> {
  crate::util::paths::state_dir().map(|d| d.join(ROOT_DIR_NAME))
}

/// The stable cache-entry name for a model file: a hash over the canonical path plus
/// the file's size and mtime, so a replaced model never lands in the same entry as
/// the file it replaced. Pure (no filesystem access) so tests drive it directly.
pub fn entry_name(canonical_path: &Path, bytes: u64, mtime: u64) -> String {
  let mut hasher = blake3::Hasher::new();
  hasher.update(canonical_path.as_os_str().as_encoded_bytes());
  hasher.update(&[0]);
  hasher.update(bytes.to_le_bytes().as_slice());
  hasher.update(mtime.to_le_bytes().as_slice());
  hasher.finalize().to_hex()[..16].to_string()
}

/// The entry directory a launch of `model_path` should use, or `None` when the model
/// file or the state dir cannot be resolved. Does not create anything.
pub fn entry_dir(model_path: &Path) -> Option<PathBuf> {
  let canonical =
    crate::util::paths::canonicalize(model_path).unwrap_or_else(|_| model_path.to_path_buf());
  let (bytes, mtime) = file_fingerprint(&canonical)?;
  Some(root_dir()?.join(entry_name(&canonical, bytes, mtime)))
}

/// Seed one launch: create the entry directory and hand back the `launch_config`
/// entries that carry it. Called from the backend's launch seeding; an empty
/// return means this launch gets no prompt cache (disabled, bench parity, missing
/// state dir, unreadable model file) and every later hook is skipped.
pub fn seed(model_path: &Path, max_bytes: &str, max_age_days: u64) -> Vec<(String, String)> {
  let Some(dir) = entry_dir(model_path) else {
    return Vec::new();
  };
  if let Err(e) = std::fs::create_dir_all(&dir) {
    log::warn!(
      "prompt cache: cannot create {}: {e}; disabled for this launch",
      dir.display()
    );
    return Vec::new();
  }
  #[cfg(unix)]
  {
    use std::os::unix::fs::PermissionsExt;
    let _ = std::fs::set_permissions(&dir, std::fs::Permissions::from_mode(0o700));
  }
  vec![
    (CFG_DIR.to_string(), dir.display().to_string()),
    (CFG_MAX_BYTES.to_string(), max_bytes.to_string()),
    (CFG_MAX_AGE_DAYS.to_string(), max_age_days.to_string()),
  ]
}

/// The entry directory when this launch has the cache on, honoring the binary
/// capability flag alongside the seed keys.
fn enabled_dir(params: &LaunchParams) -> Option<PathBuf> {
  if params.launch_config.get(CFG_SUPPORTED).map(String::as_str) != Some("true") {
    return None;
  }
  Some(PathBuf::from(params.launch_config.get(CFG_DIR)?))
}

/// One summary of a save pass.
struct SaveOutcome {
  slots: usize,
  bytes: u64,
  ms: f64,
}

/// Save every slot that holds cached prompt tokens, then (re)write the entry
/// fingerprint and prune the cache root. Best-effort: errors are logged and leave
/// whatever succeeded on disk.
pub async fn save_before_eviction(port: u16, params: &LaunchParams) {
  let Some(dir) = enabled_dir(params) else {
    return;
  };
  let slots = match prompt_slots(port).await {
    Ok(slots) => slots,
    Err(e) => {
      log::warn!("prompt cache: slot listing on port {port} failed: {e}; skipping save");
      return;
    }
  };
  let mut outcome = SaveOutcome {
    slots: 0,
    bytes: 0,
    ms: 0.0,
  };
  for id in slot_ids_with_prompt(&slots) {
    let filename = slot_file_name(id);
    match save_slot(port, id, &filename).await {
      Ok((n_written, ms)) => {
        outcome.slots += 1;
        outcome.bytes = outcome.bytes.saturating_add(n_written);
        outcome.ms += ms;
        // The engine writes the file itself and takes no guidance on mode; the
        // bytes are prompt-derived plaintext, so tighten it right after.
        #[cfg(unix)]
        {
          use std::os::unix::fs::PermissionsExt;
          let _ = std::fs::set_permissions(
            dir.join(&filename),
            std::fs::Permissions::from_mode(0o600),
          );
        }
      }
      Err(e) => log::warn!(
        "prompt cache: save of slot {id} on port {port} failed: {e}; keeping whatever earlier slots saved"
      ),
    }
  }
  if outcome.slots == 0 {
    log::debug!(
      "prompt cache: no slot on port {port} held a prompt; nothing saved to {}",
      dir.display()
    );
    return;
  }
  let n_ctx = props_n_ctx(port).await;
  write_meta(&dir, params, n_ctx);
  log::info!(
    "prompt cache: saved {} slot(s), {} bytes in {:.0} ms to {}",
    outcome.slots,
    outcome.bytes,
    outcome.ms,
    dir.display()
  );
  if let Some(root) = root_dir() {
    prune_at(&root, limits_from(params), &dir);
  }
}

/// Restore the entry's slot files into a freshly-ready server, before the launch is
/// declared Ready. Guarded by the meta fingerprint; every failure path deletes or
/// ignores the entry rather than failing the launch.
pub async fn restore_after_ready(port: u16, params: &LaunchParams) {
  let Some(dir) = enabled_dir(params) else {
    return;
  };
  if !dir.is_dir() {
    return;
  }
  let meta = match load_meta(&dir) {
    Some(meta) => meta,
    None => {
      log::debug!(
        "prompt cache: {} has no readable {}, nothing to restore",
        dir.display(),
        META_FILE
      );
      return;
    }
  };
  let Some((bytes, mtime)) = file_fingerprint(&params.model_path) else {
    log::warn!(
      "prompt cache: model {} unreadable now; dropping stale entry {}",
      params.model_path.display(),
      dir.display()
    );
    remove_entry(&dir);
    return;
  };
  if meta.model != params.model_path.display().to_string()
    || meta.bytes != bytes
    || meta.mtime != mtime
  {
    log::info!(
      "prompt cache: model {} changed since the saves; dropping entry {}",
      params.model_path.display(),
      dir.display()
    );
    remove_entry(&dir);
    return;
  }
  let files = slot_files(&dir);
  if files.is_empty() {
    return;
  }
  let total_slots = props_total_slots(port).await;
  let mut attempted = 0usize;
  let mut restored = 0usize;
  let mut rejected = 0usize;
  let mut tokens = 0u64;
  let mut ms = 0f64;
  let mut last_err = String::new();
  for (id, filename) in files {
    if let Some(total) = total_slots {
      if u64::from(id) >= total {
        // The new launch has fewer slots than the saves were taken from; the file
        // stays for a later launch that runs enough slots.
        continue;
      }
    }
    attempted += 1;
    match restore_slot(port, id, &filename).await {
      Ok((n_restored, t)) => {
        restored += 1;
        tokens += n_restored;
        ms += t;
      }
      Err(e) => {
        rejected += e.rejected() as usize;
        last_err = e.to_string();
        log::warn!("prompt cache: restore of slot {id} on port {port} failed: {e}");
      }
    }
  }
  if restored > 0 {
    log::info!(
      "prompt cache: restored {restored}/{attempted} slot(s), {tokens} tokens in {ms:.0} ms into port {port}"
    );
  } else if rejected > 0 && rejected == attempted {
    // The engine answered and refused every file (wrong engine, ragged file).
    // The entry cannot serve this launch; delete it so the next one does not
    // pay the same dead round trip. Transport failures are transient and keep
    // the entry for a later launch.
    log::info!(
      "prompt cache: all {attempted} restore attempt(s) into port {port} were rejected ({last_err}); dropping entry {}",
      dir.display()
    );
    remove_entry(&dir);
  }
}

/// Prune the cache root at `root`: entries older than the age limit, then
/// oldest-first while the root exceeds its size cap. `keep` (the entry just
/// written) is never removed.
fn prune_at(root: &Path, limits: Limits, keep: &Path) {
  let Ok(entries) = std::fs::read_dir(root) else {
    return;
  };
  let now = SystemTime::now()
    .duration_since(UNIX_EPOCH)
    .map(|d| d.as_secs())
    .unwrap_or(0);
  let mut dirs: Vec<(PathBuf, u64, u64)> = Vec::new();
  for entry in entries.flatten() {
    let path = entry.path();
    let Ok(md) = std::fs::metadata(&path) else {
      continue;
    };
    if !md.is_dir() {
      continue;
    }
    let mtime = md
      .modified()
      .ok()
      .and_then(|t| t.duration_since(UNIX_EPOCH).ok())
      .map(|d| d.as_secs())
      .unwrap_or(0);
    let bytes = dir_size(&path);
    if path != keep && limits.max_age_secs > 0 && now.saturating_sub(mtime) > limits.max_age_secs {
      remove_entry(&path);
      continue;
    }
    dirs.push((path, bytes, mtime));
  }
  let total: u64 = dirs.iter().map(|(_, b, _)| *b).fold(0, u64::saturating_add);
  if limits.max_bytes == 0 || total <= limits.max_bytes {
    return;
  }
  dirs.sort_by_key(|(_, _, mtime)| *mtime);
  let mut freed = 0u64;
  for (path, bytes, _) in dirs {
    if total.saturating_sub(freed) <= limits.max_bytes {
      break;
    }
    if path == keep {
      continue;
    }
    log::info!(
      "prompt cache: cache over the {} byte cap; dropped oldest entry {}",
      limits.max_bytes,
      path.display()
    );
    remove_entry(&path);
    freed = freed.saturating_add(bytes);
  }
}

/// The pruning limits, read back off a launch's params (seeded from config).
struct Limits {
  max_bytes: u64,
  max_age_secs: u64,
}

fn limits_from(params: &LaunchParams) -> Limits {
  let max_bytes = params
    .launch_config
    .get(CFG_MAX_BYTES)
    .and_then(|s| crate::launch::admission::parse_size_bytes(s))
    .unwrap_or(16 * 1024 * 1024 * 1024);
  let max_age_days = params
    .launch_config
    .get(CFG_MAX_AGE_DAYS)
    .and_then(|s| s.parse::<u64>().ok())
    .unwrap_or(7);
  Limits {
    max_bytes,
    max_age_secs: max_age_days * 86_400,
  }
}

/// Files named `slot-<digits>.bin` in an entry, keyed by slot id.
fn slot_files(dir: &Path) -> Vec<(u32, String)> {
  let Ok(entries) = std::fs::read_dir(dir) else {
    return Vec::new();
  };
  let mut out: Vec<(u32, String)> = entries
    .flatten()
    .filter_map(|e| {
      let name = e.file_name().to_string_lossy().into_owned();
      slot_file_id(&name).map(|id| (id, name))
    })
    .collect();
  out.sort_by_key(|(id, _)| *id);
  out
}

/// The slot id a save filename carries, or `None` when the name is not a slot file.
/// Pure so tests can cover the name grammar.
pub fn slot_file_id(name: &str) -> Option<u32> {
  let rest = name
    .strip_prefix(SLOT_FILE_PREFIX)?
    .strip_suffix(SLOT_FILE_SUFFIX)?;
  if rest.is_empty() || !rest.bytes().all(|b| b.is_ascii_digit()) {
    return None;
  }
  rest.parse().ok()
}

/// The filename a slot's cache is saved under.
fn slot_file_name(id: u32) -> String {
  format!("{SLOT_FILE_PREFIX}{id}{SLOT_FILE_SUFFIX}")
}

/// A slot listing entry: `(id, prompt tokens held)`, with the count taken from
/// whichever field the build answers (`n_prompt_tokens` on current builds,
/// `n_prompt` on older ones; an absent count means the slot holds nothing).
fn slot_ids_with_prompt(slots: &serde_json::Value) -> Vec<u32> {
  slots
    .as_array()
    .map(|arr| {
      arr
        .iter()
        .filter_map(|s| {
          let id = s.get("id").or_else(|| s.get("id_slot"))?.as_u64()?;
          let n = s
            .get("n_prompt_tokens")
            .or_else(|| s.get("n_prompt"))
            .and_then(serde_json::Value::as_u64)
            .unwrap_or(0);
          (n > 0).then_some(id as u32)
        })
        .collect()
    })
    .unwrap_or_default()
}

fn dir_size(dir: &Path) -> u64 {
  let Ok(entries) = std::fs::read_dir(dir) else {
    return 0;
  };
  entries
    .flatten()
    .filter_map(|e| std::fs::metadata(e.path()).ok())
    .filter(|md| md.is_file())
    .map(|md| md.len())
    .fold(0u64, u64::saturating_add)
}

fn remove_entry(dir: &Path) {
  if let Err(e) = std::fs::remove_dir_all(dir) {
    if e.kind() != std::io::ErrorKind::NotFound {
      log::warn!("prompt cache: could not remove {}: {e}", dir.display());
    }
  }
}

fn write_meta(dir: &Path, params: &LaunchParams, n_ctx: Option<u64>) {
  let (bytes, mtime) = file_fingerprint(&params.model_path).unwrap_or((0, 0));
  let meta = Meta {
    model: params.model_path.display().to_string(),
    bytes,
    mtime,
    n_ctx,
    saved_at: SystemTime::now()
      .duration_since(UNIX_EPOCH)
      .map(|d| d.as_secs())
      .unwrap_or(0),
  };
  let body = match serde_json::to_vec_pretty(&meta) {
    Ok(b) => b,
    Err(e) => {
      log::warn!("prompt cache: meta serialization failed: {e}");
      return;
    }
  };
  let final_path = dir.join(META_FILE);
  if let Err(e) =
    crate::util::atomic_write::write_secure(dir, "meta.json.tmp.", &final_path, &body, Some(0o600))
  {
    log::warn!(
      "prompt cache: could not write {}: {e}",
      final_path.display()
    );
  }
}

fn load_meta(dir: &Path) -> Option<Meta> {
  let body = std::fs::read(dir.join(META_FILE)).ok()?;
  serde_json::from_slice(&body).ok()
}

// ---------------------------------------------------------------------------
// Engine HTTP calls. Same no-dependency raw-TCP stance as the `/props` fetch in
// [`super::actuals`]: these are loopback calls to a server we spawned.
// ---------------------------------------------------------------------------

/// One HTTP exchange outcome error against the launch's server. `status` is the
/// engine's own HTTP code when it answered at all: a rejection (Some) means the
/// engine saw the request and refused it, a transport failure (None) is possibly
/// transient, and the two get different cleanup treatment upstream.
#[derive(Debug)]
struct HttpErr {
  status: Option<u16>,
  msg: String,
}

impl HttpErr {
  fn rejected(&self) -> bool {
    self.status.is_some()
  }
}

impl std::fmt::Display for HttpErr {
  fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
    f.write_str(&self.msg)
  }
}

/// One HTTP exchange against the launch's server. `Connection: close` so the
/// read-to-EOF terminates without keep-alive bookkeeping. Non-2xx answers `Err`
/// with the status and a body excerpt; the slot-action error shapes all carry
/// JSON with a `message`, which is what the log lines want.
async fn http(
  port: u16,
  method: &str,
  path: &str,
  body: Option<&str>,
  timeout: Duration,
) -> Result<serde_json::Value, HttpErr> {
  use tokio::io::{AsyncReadExt, AsyncWriteExt};
  use tokio::net::TcpStream;

  let mut request = format!(
    "{method} {path} HTTP/1.1\r\nHost: 127.0.0.1:{port}\r\nConnection: close\r\nAccept: application/json\r\n"
  );
  if let Some(b) = body {
    request.push_str("Content-Type: application/json\r\n");
    request.push_str(&format!("Content-Length: {}\r\n", b.len()));
  }
  request.push_str("\r\n");
  let fut = async {
    let mut sock = TcpStream::connect(("127.0.0.1", port)).await?;
    sock.write_all(request.as_bytes()).await?;
    if let Some(b) = body {
      sock.write_all(b.as_bytes()).await?;
    }
    let mut buf = Vec::with_capacity(4096);
    let mut chunk = [0u8; 8192];
    loop {
      let n = sock.read(&mut chunk).await?;
      if n == 0 {
        break;
      }
      buf.extend_from_slice(&chunk[..n]);
      if buf.len() > 8 * 1024 * 1024 {
        break;
      }
    }
    Ok::<_, std::io::Error>(buf)
  };
  let resp = tokio::time::timeout(timeout, fut)
    .await
    .map_err(|e| HttpErr {
      status: None,
      msg: format!("{method} {path} timed out: {e}"),
    })?
    .map_err(|e| HttpErr {
      status: None,
      msg: format!("{method} {path}: {e}"),
    })?;
  let Some(header_end) = resp.windows(4).position(|w| w == b"\r\n\r\n") else {
    return Err(HttpErr {
      status: None,
      msg: format!("{method} {path}: response had no header end"),
    });
  };
  let head = String::from_utf8_lossy(&resp[..header_end]);
  let status = head
    .split_whitespace()
    .nth(1)
    .and_then(|s| s.parse::<u16>().ok())
    .unwrap_or(0);
  let body_bytes = &resp[header_end + 4..];
  let text = String::from_utf8_lossy(body_bytes).into_owned();
  if !(200..300).contains(&status) {
    let excerpt: String = text.chars().take(300).collect();
    return Err(HttpErr {
      status: Some(status),
      msg: format!("{method} {path} -> {status}: {excerpt}"),
    });
  }
  serde_json::from_slice(body_bytes).map_err(|e| HttpErr {
    status: Some(status),
    msg: format!("{method} {path}: unparseable JSON response ({e})"),
  })
}

async fn prompt_slots(port: u16) -> Result<serde_json::Value, HttpErr> {
  http(port, "GET", "/slots", None, QUERY_TIMEOUT).await
}

async fn props_n_ctx(port: u16) -> Option<u64> {
  let v = http(port, "GET", "/props", None, QUERY_TIMEOUT)
    .await
    .ok()?;
  v.get("default_generation_settings")
    .and_then(|g| g.get("n_ctx"))
    .and_then(serde_json::Value::as_u64)
    .or_else(|| v.get("n_ctx").and_then(serde_json::Value::as_u64))
}

async fn props_total_slots(port: u16) -> Option<u64> {
  let v = http(port, "GET", "/props", None, QUERY_TIMEOUT)
    .await
    .ok()?;
  v.get("total_slots").and_then(serde_json::Value::as_u64)
}

/// `POST /slots/{id}?action=save`; returns `(bytes written, save ms)`.
async fn save_slot(port: u16, id: u32, filename: &str) -> Result<(u64, f64), HttpErr> {
  let body = serde_json::json!({ "filename": filename }).to_string();
  let v = http(
    port,
    "POST",
    &format!("/slots/{id}?action=save"),
    Some(&body),
    SAVE_TIMEOUT,
  )
  .await?;
  let n_written = v.get("n_written").and_then(serde_json::Value::as_u64);
  let ms = v
    .get("timings")
    .and_then(|t| t.get("save_ms"))
    .and_then(serde_json::Value::as_f64)
    .unwrap_or(0.0);
  Ok((n_written.unwrap_or(0), ms))
}

/// `POST /slots/{id}?action=restore`; returns `(tokens restored, restore ms)`.
async fn restore_slot(port: u16, id: u32, filename: &str) -> Result<(u64, f64), HttpErr> {
  let body = serde_json::json!({ "filename": filename }).to_string();
  let v = http(
    port,
    "POST",
    &format!("/slots/{id}?action=restore"),
    Some(&body),
    RESTORE_TIMEOUT,
  )
  .await?;
  let n_restored = v
    .get("n_restored")
    .and_then(serde_json::Value::as_u64)
    .unwrap_or(0);
  let ms = v
    .get("timings")
    .and_then(|t| t.get("restore_ms"))
    .and_then(serde_json::Value::as_f64)
    .unwrap_or(0.0);
  Ok((n_restored, ms))
}

#[cfg(test)]
mod tests {
  use super::*;
  use crate::launch::mode::LaunchMode;
  use std::path::PathBuf;

  fn params_with_cache(dir: &Path) -> LaunchParams {
    let mut p = LaunchParams::new(PathBuf::from("/m/model.gguf"), LaunchMode::Chat);
    p.launch_config
      .insert(CFG_DIR.to_string(), dir.display().to_string());
    p.launch_config
      .insert(CFG_SUPPORTED.to_string(), "true".into());
    p
  }

  #[test]
  fn entry_name_is_stable_and_tracks_the_file_fingerprint() {
    let a = entry_name(Path::new("/m/model.gguf"), 100, 5);
    assert_eq!(a, entry_name(Path::new("/m/model.gguf"), 100, 5));
    assert_ne!(a, entry_name(Path::new("/m/model.gguf"), 101, 5), "size");
    assert_ne!(a, entry_name(Path::new("/m/model.gguf"), 100, 6), "mtime");
    assert_ne!(a, entry_name(Path::new("/m/other.gguf"), 100, 5), "path");
    assert_eq!(a.len(), 16);
  }

  #[test]
  fn slot_file_names_round_trip_through_the_id_grammar() {
    assert_eq!(slot_file_id("slot-0.bin"), Some(0));
    assert_eq!(slot_file_id("slot-12.bin"), Some(12));
    assert_eq!(slot_file_id("slot-.bin"), None);
    assert_eq!(slot_file_id("slot-x.bin"), None);
    assert_eq!(slot_file_id("meta.json"), None);
    assert_eq!(slot_file_id("slot-3.txt"), None);
    assert_eq!(slot_file_name(7), "slot-7.bin");
  }

  #[test]
  fn slot_listing_keeps_only_slots_holding_a_prompt() {
    let v = serde_json::json!([
      { "id": 0 },
      { "id": 1, "n_prompt_tokens": 3175 },
      { "id": 2, "n_prompt_tokens": 0 },
      { "id_slot": 3, "n_prompt": 42 }
    ]);
    assert_eq!(slot_ids_with_prompt(&v), vec![1, 3]);
    assert!(slot_ids_with_prompt(&serde_json::json!({})).is_empty());
  }

  #[test]
  fn enabled_requires_both_the_directory_and_the_capability() {
    let dir = Path::new("/tmp/unused-entry");
    let mut p = params_with_cache(dir);
    assert_eq!(enabled_dir(&p), Some(dir.to_path_buf()));
    p.launch_config
      .insert(CFG_SUPPORTED.to_string(), "false".into());
    assert_eq!(enabled_dir(&p), None);
    p.launch_config.remove(CFG_SUPPORTED);
    assert_eq!(enabled_dir(&p), None);
    let bare = LaunchParams::new(PathBuf::from("/m/model.gguf"), LaunchMode::Chat);
    assert_eq!(enabled_dir(&bare), None);
  }

  #[test]
  fn limits_fall_back_when_the_stashed_strings_are_absent_or_bogus() {
    let p = LaunchParams::new(PathBuf::from("/m/model.gguf"), LaunchMode::Chat);
    let l = limits_from(&p);
    assert_eq!(l.max_bytes, 16 * 1024 * 1024 * 1024);
    assert_eq!(l.max_age_secs, 7 * 86_400);

    let mut p2 = params_with_cache(Path::new("/tmp/x"));
    p2.launch_config
      .insert(CFG_MAX_BYTES.to_string(), "512M".into());
    p2.launch_config
      .insert(CFG_MAX_AGE_DAYS.to_string(), "2".into());
    let l2 = limits_from(&p2);
    assert_eq!(l2.max_bytes, 512 * 1024 * 1024);
    assert_eq!(l2.max_age_secs, 2 * 86_400);
  }

  #[test]
  fn meta_round_trips_and_rejects_ragged_json() {
    let dir = crate::util::test_temp::unique_temp_dir("pc-meta");
    std::fs::create_dir_all(&dir).unwrap();
    let p = params_with_cache(&dir);
    // The model path does not exist here, so the fingerprint records the 0/0
    // fallback rather than failing the write.
    write_meta(&dir, &p, Some(4096));
    let meta = load_meta(&dir).expect("meta written");
    assert_eq!(meta.model, p.model_path.display().to_string());
    assert_eq!(meta.n_ctx, Some(4096));
    assert_eq!((meta.bytes, meta.mtime), (0, 0));
    std::fs::write(dir.join(META_FILE), b"{").unwrap();
    assert!(load_meta(&dir).is_none());
    let _ = std::fs::remove_dir_all(&dir);
  }

  /// Prune deletes age-expired entries, then oldest-first under the size cap, and
  /// never the entry it was just asked to keep.
  #[test]
  fn prune_enforces_age_then_size_and_keeps_the_live_entry() {
    let root = crate::util::test_temp::unique_temp_dir("pc-prune");
    std::fs::create_dir_all(&root).unwrap();
    let mk = |name: &str, bytes: u64, age_secs: u64| {
      let d = root.join(name);
      std::fs::create_dir_all(&d).unwrap();
      std::fs::write(d.join("slot-0.bin"), vec![7u8; bytes as usize]).unwrap();
      let old = std::time::SystemTime::now() - Duration::from_secs(age_secs);
      filetime_set(&d, old);
      d
    };
    let day = 86_400;
    let old = mk("old", 10, 9 * day);
    let mid = mk("mid", 100, 2 * day);
    let young = mk("young", 1000, day);

    // Age pass only: size cap generous. `old` (> 7d) goes even though `young` is bigger.
    prune_at(
      &root,
      Limits {
        max_bytes: 0, // 0 = no size cap
        max_age_secs: 7 * day,
      },
      &young,
    );
    assert!(!old.exists(), "age-expired entry must be pruned");
    assert!(mid.exists());
    assert!(young.exists());

    // Size pass: cap of 500 bytes forces `mid` out; `young` is the live keep.
    prune_at(
      &root,
      Limits {
        max_bytes: 500,
        max_age_secs: 0,
      },
      &young,
    );
    assert!(!mid.exists(), "oldest entry over the cap must be pruned");
    assert!(young.exists(), "the live entry is never pruned");

    // With no live entry to keep and a tiny cap, everything goes oldest-first.
    let big = mk("big", 4000, day);
    let small = mk("small", 10, 3 * day);
    prune_at(
      &root,
      Limits {
        max_bytes: 100,
        max_age_secs: 0,
      },
      &big,
    );
    assert!(!small.exists());
    assert!(big.exists());
    let _ = std::fs::remove_dir_all(&root);
  }

  #[cfg(unix)]
  fn filetime_set(path: &Path, t: std::time::SystemTime) {
    // Directories cannot be opened for writing, so the mtime goes through
    // `utimensat` (AT_EMPTY_PATH is not portable for dirs).
    let c = std::ffi::CString::new(path.as_os_str().as_encoded_bytes()).unwrap();
    let secs = t.duration_since(std::time::UNIX_EPOCH).unwrap().as_secs() as i64;
    let ts = [libc::timespec {
      tv_sec: secs,
      tv_nsec: 0,
    }; 2];
    let rc = unsafe {
      libc::utimensat(
        libc::AT_FDCWD,
        c.as_ptr(),
        ts.as_ptr(),
        libc::AT_SYMLINK_NOFOLLOW,
      )
    };
    assert_eq!(rc, 0, "utimensat failed for {}", path.display());
  }

  #[cfg(not(unix))]
  fn filetime_set(_path: &Path, _t: std::time::SystemTime) {}

  // A canned-answer listener stands in for llama-server so the HTTP client layer
  // (status handling, JSON bodies, error passthrough) is covered without an engine.
  async fn canned_listener(responses: Vec<(&'static str, &'static str)>) -> u16 {
    let listener = tokio::net::TcpListener::bind(("127.0.0.1", 0))
      .await
      .unwrap();
    let port = listener.local_addr().unwrap().port();
    let mut responder = tokio::sync::Mutex::new(responses.into_iter());
    tokio::spawn(async move {
      loop {
        let Ok((mut sock, _)) = listener.accept().await else {
          return;
        };
        use tokio::io::AsyncReadExt;
        let mut buf = [0u8; 4096];
        let _ = sock.read(&mut buf).await;
        let Some((status, body)) = responder.get_mut().next() else {
          break;
        };
        let resp = format!(
          "HTTP/1.1 {status}\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{body}",
          body.len()
        );
        let _ = tokio::io::AsyncWriteExt::write_all(&mut sock, resp.as_bytes()).await;
      }
    });
    port
  }

  #[tokio::test]
  async fn http_parses_success_bodies_and_surfaces_engine_errors() {
    let port = canned_listener(vec![
      ("200 OK", r#"[{"id":0,"n_prompt_tokens":10}]"#),
      (
        "400 Bad Request",
        r#"{"error":{"message":"Unable to restore slot"}}"#,
      ),
      ("200 OK", "not json"),
    ])
    .await;
    let v = http(port, "GET", "/slots", None, QUERY_TIMEOUT)
      .await
      .expect("ok response");
    assert_eq!(v[0]["n_prompt_tokens"], 10);
    let err = http(
      port,
      "POST",
      "/slots/0?action=save",
      Some("{}"),
      QUERY_TIMEOUT,
    )
    .await
    .expect_err("400 is an error");
    assert!(err.rejected(), "{err:?}");
    assert!(err.to_string().contains("400"), "{err}");
    assert!(err.to_string().contains("Unable to restore slot"), "{err}");
    let err2 = http(port, "GET", "/props", None, QUERY_TIMEOUT)
      .await
      .expect_err("ragged body is an error");
    assert!(err2.to_string().contains("unparseable"), "{err2}");
  }

  #[tokio::test]
  async fn save_slot_reads_written_bytes_and_timing() {
    let port = canned_listener(vec![
      (
        "200 OK",
        r#"{"id_slot":0,"filename":"slot-0.bin","n_saved":3175,"n_written":104089636,"timings":{"save_ms":36.5}}"#,
      ),
      (
        "200 OK",
        r#"{"id_slot":0,"filename":"slot-0.bin","n_restored":3175,"n_read":104089636,"timings":{"restore_ms":20.1}}"#,
      ),
    ])
    .await;
    let (bytes, save_ms) = save_slot(port, 0, "slot-0.bin").await.unwrap();
    assert_eq!(bytes, 104089636);
    assert!((save_ms - 36.5).abs() < 0.01);
    let (tokens, restore_ms) = restore_slot(port, 0, "slot-0.bin").await.unwrap();
    assert_eq!(tokens, 3175);
    assert!((restore_ms - 20.1).abs() < 0.01);
  }

  #[tokio::test]
  async fn restore_drops_the_entry_when_the_model_file_no_longer_matches() {
    let dir = crate::util::test_temp::unique_temp_dir("pc-restore-stale");
    std::fs::create_dir_all(&dir).unwrap();
    let model_dir = crate::util::test_temp::unique_temp_dir("pc-restore-model");
    let model = model_dir.join("m.gguf");
    std::fs::write(&model, vec![1u8; 10]).unwrap();
    let mut p = params_with_cache(&dir);
    p.model_path = model.clone();
    write_meta(&dir, &p, None);
    std::fs::write(dir.join("slot-0.bin"), vec![9u8; 5]).unwrap();

    // The model file grew: the saved KV belongs to a different file.
    std::fs::write(&model, vec![1u8; 20]).unwrap();
    // Nothing listens on this port, which is fine: the fingerprint guard must
    // bail before any HTTP is attempted.
    restore_after_ready(1, &p).await;
    assert!(
      !dir.exists(),
      "stale entry must be dropped, no restore attempted"
    );
    let _ = std::fs::remove_dir_all(&model_dir);
  }

  /// A server that refuses every file deletes the entry; one that never answers
  /// (a transport failure, possibly transient) leaves it for a later launch.
  #[tokio::test]
  async fn rejected_restores_drop_the_entry_but_transport_failures_do_not() {
    let model_dir = crate::util::test_temp::unique_temp_dir("pc-reject-model");
    let model = model_dir.join("m.gguf");
    std::fs::write(&model, vec![1u8; 10]).unwrap();

    let dir = crate::util::test_temp::unique_temp_dir("pc-reject");
    std::fs::create_dir_all(&dir).unwrap();
    let mut p = params_with_cache(&dir);
    p.model_path = model.clone();
    write_meta(&dir, &p, None);
    std::fs::write(dir.join("slot-0.bin"), vec![9u8; 5]).unwrap();

    // props answers, the restore is refused (400): the entry is dropped.
    let port = canned_listener(vec![
      ("200 OK", r#"{"total_slots": 4}"#),
      (
        "400 Bad Request",
        r#"{"error":{"message":"invalid slot save file"}}"#,
      ),
    ])
    .await;
    restore_after_ready(port, &p).await;
    assert!(!dir.exists(), "engine rejections drop the entry");

    // Same entry, nothing listening: transport failures keep it.
    std::fs::create_dir_all(&dir).unwrap();
    write_meta(&dir, &p, None);
    std::fs::write(dir.join("slot-0.bin"), vec![9u8; 5]).unwrap();
    restore_after_ready(1, &p).await;
    assert!(dir.exists(), "transient failures must not drop the entry");

    let _ = std::fs::remove_dir_all(&dir);
    let _ = std::fs::remove_dir_all(&model_dir);
  }

  #[tokio::test]
  async fn save_without_slot_listing_writes_nothing() {
    // Port 1 refuses instantly; the save must log and return, touching nothing.
    let dir = crate::util::test_temp::unique_temp_dir("pc-save-dark");
    std::fs::create_dir_all(&dir).unwrap();
    let mut p = params_with_cache(&dir);
    p.model_path = dir.join("missing.gguf");
    save_before_eviction(1, &p).await;
    assert!(
      slot_files(&dir).is_empty(),
      "no listing means no files written"
    );
    assert!(
      load_meta(&dir).is_none(),
      "no successful save means no meta"
    );
    let _ = std::fs::remove_dir_all(&dir);
  }
}
