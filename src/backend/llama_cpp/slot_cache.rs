//! Keep a launch's prompt cache across an unload (`backend.llamacpp.slot_cache`).
//!
//! `llama-server --slot-save-path DIR` exposes
//! `POST /slots/{id}?action=save|restore`, which writes a slot's KV cache and
//! token list to a file under `DIR` and reads it back. The daemon saves every
//! slot right before the idle sweep or make-room unloads a launch, and restores
//! the files when a launch with the same argv turns ready, before it takes
//! traffic. Numbers: `docs/spikes/2026-10-10-slot-save-restore.md`.
//!
//! - **Where.** `<root>/<key>-<model-stem>.slot-<id>.bin`. `key` hashes the
//!   launch argv (port excluded) and the model file's size and mtime, so a file
//!   is only ever read back by a launch that would have written the same one.
//! - **Matching.** Each file goes back into the slot id it came from.
//!   llama-server picks the slot for a request by longest common prefix
//!   (`--slot-prompt-similarity`), so the returning conversation lands on the
//!   restored slot with no help from the proxy.
//! - **Cleanup.** A file is deleted once it has been restored (or failed to).
//!   After every save, files older than `max_age_secs` go, then the oldest files
//!   go until the root is under `max_gib`.
//!
//! Everything here is best-effort: a failed save or restore is logged and the
//! unload or launch carries on.

use std::ffi::OsString;
use std::path::{Path, PathBuf};
use std::time::{Duration, SystemTime};

use serde::{Deserialize, Serialize};
use tokio::io::{AsyncReadExt, AsyncWriteExt};
use tokio::net::TcpStream;

use crate::launch::mode::LaunchMode;
use crate::launch::params::LaunchParams;

/// `launch_config` keys carrying the slot-cache settings for one launch.
/// Seeded by [`seed`], read by `compose` and by the save and restore hooks,
/// which run without the daemon config in reach.
const KNOB_ROOT: &str = "slot_cache_root";
const KNOB_STAMP: &str = "slot_cache_stamp";
const KNOB_MAX_BYTES: &str = "slot_cache_max_bytes";
const KNOB_MAX_AGE: &str = "slot_cache_max_age_secs";

/// The engine flag that names the save directory and enables the endpoints.
const SAVE_PATH_FLAG: &str = "--slot-save-path";

/// One save or one restore call. The spike measured under 3 s for a 3 GiB file;
/// a KV cache several times that size on a slow disk still fits.
const CALL_TIMEOUT: Duration = Duration::from_secs(120);
const PROPS_TIMEOUT: Duration = Duration::from_secs(5);
const GIB: u64 = 1024 * 1024 * 1024;

/// `backend.llamacpp.slot_cache` in `config.yaml`.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields, rename_all = "snake_case")]
pub struct SlotCacheConfig {
  /// Save slots before an eviction and restore them on the next launch.
  /// Factory `false`: one save is about 33 KB per token on a 1B model and
  /// grows with the model, so it is opt-in.
  pub enabled: bool,
  /// Where the files go. Unset: `<cache dir>/slot-cache`.
  pub dir: Option<PathBuf>,
  /// Cap on the total size of all saved files, in GiB.
  pub max_gib: u64,
  /// A file older than this is deleted instead of restored.
  pub max_age_secs: u64,
}

impl Default for SlotCacheConfig {
  fn default() -> Self {
    Self {
      enabled: false,
      dir: None,
      max_gib: 16,
      max_age_secs: 24 * 60 * 60,
    }
  }
}

impl SlotCacheConfig {
  fn root(&self) -> Option<PathBuf> {
    self
      .dir
      .clone()
      .or_else(|| crate::util::paths::cache_dir().map(|d| d.join("slot-cache")))
  }
}

/// Project the config onto `params.launch_config`, or clear the keys when the
/// feature does not apply to this launch (off, not a chat launch, or the user
/// passes their own `--slot-save-path`).
pub(super) fn seed(cfg: &SlotCacheConfig, params: &mut LaunchParams) {
  for key in [KNOB_ROOT, KNOB_STAMP, KNOB_MAX_BYTES, KNOB_MAX_AGE] {
    params.launch_config.remove(key);
  }
  if !cfg.enabled || params.mode != LaunchMode::Chat || user_sets_save_path(&params.extras) {
    return;
  }
  let Some(root) = cfg.root() else {
    return;
  };
  // llama-server exits at startup when the directory is missing.
  if let Err(e) = create_private_dir(&root) {
    log::warn!(
      "slot cache: cannot create {}: {e}; off for this launch",
      root.display()
    );
    return;
  }
  let mut put = |key: &str, value: String| {
    params.launch_config.insert(key.to_string(), value);
  };
  put(KNOB_ROOT, root.display().to_string());
  put(KNOB_STAMP, model_stamp(&params.model_path));
  put(KNOB_MAX_BYTES, cfg.max_gib.saturating_mul(GIB).to_string());
  put(KNOB_MAX_AGE, cfg.max_age_secs.to_string());
}

fn user_sets_save_path(extras: &[OsString]) -> bool {
  extras.iter().any(|a| {
    let a = a.to_string_lossy();
    a == SAVE_PATH_FLAG || a.starts_with("--slot-save-path=")
  })
}

/// Size and mtime of the model file. A re-pulled or replaced GGUF gets a new
/// key, so a cache written by the old weights is never read into the new ones.
fn model_stamp(model: &Path) -> String {
  let Ok(meta) = std::fs::metadata(model) else {
    return "unknown".to_string();
  };
  let mtime = meta
    .modified()
    .ok()
    .and_then(|t| t.duration_since(SystemTime::UNIX_EPOCH).ok())
    .map_or(0, |d| d.as_secs());
  format!("{}-{mtime}", meta.len())
}

/// The slot-cache settings one launch runs with, read back off its params.
#[derive(Debug, Clone, PartialEq, Eq)]
struct Settings {
  root: PathBuf,
  /// `<key>-<model-stem>.`, the start of every file this launch shape owns.
  prefix: String,
  max_bytes: u64,
  max_age: Duration,
}

fn settings(params: &LaunchParams) -> Option<Settings> {
  let get = |key: &str| params.launch_config.get(key);
  let root = PathBuf::from(get(KNOB_ROOT)?);
  // Composed for port `0`, so two launches that differ only by port share
  // their files.
  let mut hasher = blake3::Hasher::new();
  for arg in super::compose::compose(params, 0) {
    let bytes = arg.as_encoded_bytes();
    hasher.update(&(bytes.len() as u64).to_le_bytes());
    hasher.update(bytes);
  }
  hasher.update(get(KNOB_STAMP).map_or("", String::as_str).as_bytes());
  let key = &hasher.finalize().to_hex()[..16];
  // The name goes to the engine, which rejects path separators and odd
  // characters, so keep the readable part to a safe set.
  let stem: String = params
    .model_path
    .file_stem()
    .map(|s| s.to_string_lossy().into_owned())
    .unwrap_or_default()
    .chars()
    .map(|c| {
      if c.is_ascii_alphanumeric() || matches!(c, '-' | '_') {
        c
      } else {
        '_'
      }
    })
    .take(64)
    .collect();
  Some(Settings {
    root,
    prefix: format!("{key}-{stem}."),
    max_bytes: get(KNOB_MAX_BYTES).and_then(|s| s.parse().ok())?,
    max_age: Duration::from_secs(get(KNOB_MAX_AGE).and_then(|s| s.parse().ok())?),
  })
}

impl Settings {
  fn slot_file(&self, id: u32) -> String {
    format!("{}slot-{id}.bin", self.prefix)
  }

  /// `<prefix>slot-<id>.bin` → `id`. A `.part` file or another launch shape's
  /// file is not this launch's finished save.
  fn slot_id(&self, name: &str) -> Option<u32> {
    name
      .strip_prefix(&self.prefix)?
      .strip_prefix("slot-")?
      .strip_suffix(".bin")?
      .parse()
      .ok()
  }

  /// This launch shape's files: finished saves with their slot id, `.part`
  /// leftovers with `None`.
  async fn files(&self) -> Vec<(PathBuf, Option<u32>)> {
    let mut out = Vec::new();
    let Ok(mut entries) = tokio::fs::read_dir(&self.root).await else {
      return out;
    };
    while let Ok(Some(entry)) = entries.next_entry().await {
      let name = entry.file_name();
      let Some(name) = name.to_str().filter(|n| n.starts_with(&self.prefix)) else {
        continue;
      };
      out.push((entry.path(), self.slot_id(name)));
    }
    out.sort_by_key(|(_, id)| *id);
    out
  }

  async fn clear(&self) {
    for (path, _) in self.files().await {
      let _ = tokio::fs::remove_file(path).await;
    }
  }
}

/// The `--slot-save-path DIR` pair for this launch, empty when the feature is
/// off for it.
pub(super) fn argv(params: &LaunchParams) -> Vec<OsString> {
  match params.launch_config.get(KNOB_ROOT) {
    Some(root) => vec![SAVE_PATH_FLAG.into(), root.into()],
    None => Vec::new(),
  }
}

/// What one save wrote.
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq)]
pub(crate) struct Saved {
  pub slots: u32,
  pub tokens: u64,
  pub bytes: u64,
}

/// What one restore read back.
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq)]
pub(crate) struct Restored {
  pub slots: u32,
  pub tokens: u64,
}

/// Write every non-empty slot of the server on `port` to the root, then
/// enforce the age limit and the size cap over it.
pub(super) async fn save(params: &LaunchParams, port: u16) -> Saved {
  let Some(s) = settings(params) else {
    return Saved::default();
  };
  let started = std::time::Instant::now();
  if let Err(e) = create_private_dir(&s.root) {
    log::warn!("slot cache: cannot create {}: {e}", s.root.display());
    return Saved::default();
  }
  // The size of a save is not known until it is written, so require room for
  // a whole cap's worth up front rather than fill the disk and prune after.
  if let Some(free) = crate::init::download::available_bytes(&s.root) {
    if free < s.max_bytes {
      log::warn!(
        "slot cache: {} free on {} is under the {} cap; not saving",
        crate::launch::admission::human_gib(free),
        s.root.display(),
        crate::launch::admission::human_gib(s.max_bytes),
      );
      return Saved::default();
    }
  }
  // Leftovers belong to an earlier unload of this same launch shape that was
  // never restored; the slots being saved now replace them.
  s.clear().await;
  let Some(total_slots) = total_slots(port).await else {
    log::warn!("slot cache: /props on port {port} gave no slot count; not saving");
    return Saved::default();
  };
  let mut out = Saved::default();
  for id in 0..total_slots {
    let part = format!("{}.part", s.slot_file(id));
    let body = serde_json::json!({ "filename": part }).to_string();
    let reply = http(
      port,
      "POST",
      &format!("/slots/{id}?action=save"),
      Some(&body),
      CALL_TIMEOUT,
    )
    .await;
    let part_path = s.root.join(&part);
    let tokens = match reply {
      Ok((200, v)) => v.get("n_saved").and_then(serde_json::Value::as_u64),
      Ok((status, v)) => {
        log::warn!("slot cache: save of slot {id} on port {port} answered {status}: {v}");
        None
      }
      Err(e) => {
        // The server is gone or stuck; asking the other slots would only wait
        // out the same timeout again.
        log::warn!("slot cache: save of slot {id} on port {port} failed: {e}");
        let _ = tokio::fs::remove_file(&part_path).await;
        break;
      }
    };
    // An empty slot still writes a header-only file; it restores nothing.
    let Some(tokens) = tokens.filter(|n| *n > 0) else {
      let _ = tokio::fs::remove_file(&part_path).await;
      continue;
    };
    // The engine writes in place, so a save cut short by a crash would leave a
    // truncated file under the final name. Rename only once it answered 200.
    let final_path = s.root.join(s.slot_file(id));
    if let Err(e) = tokio::fs::rename(&part_path, &final_path).await {
      log::warn!("slot cache: cannot rename {}: {e}", part_path.display());
      let _ = tokio::fs::remove_file(&part_path).await;
      continue;
    }
    out.slots += 1;
    out.tokens += tokens;
    out.bytes += tokio::fs::metadata(&final_path)
      .await
      .map_or(0, |m| m.len());
  }
  if out.slots > 0 {
    log::info!(
      "slot cache: saved {} slot(s), {} tokens, {} to {} in {} ms",
      out.slots,
      out.tokens,
      crate::launch::admission::human_gib(out.bytes),
      s.root.display(),
      started.elapsed().as_millis(),
    );
  }
  let root = s.root.clone();
  let (max_bytes, max_age) = (s.max_bytes, s.max_age);
  let _ =
    tokio::task::spawn_blocking(move || prune(&root, max_bytes, max_age, SystemTime::now())).await;
  out
}

/// Read this launch's saved slots back into the server on `port`, each into the
/// slot id it was saved from, and delete the files.
pub(super) async fn restore(params: &LaunchParams, port: u16) -> Restored {
  let Some(s) = settings(params) else {
    return Restored::default();
  };
  let started = std::time::Instant::now();
  let mut out = Restored::default();
  let now = SystemTime::now();
  for (path, id) in s.files().await {
    let Some(id) = id else {
      let _ = tokio::fs::remove_file(&path).await;
      continue;
    };
    let fresh = tokio::fs::metadata(&path)
      .await
      .ok()
      .and_then(|m| m.modified().ok())
      .is_some_and(|t| now.duration_since(t).unwrap_or_default() <= s.max_age);
    if fresh {
      let body = serde_json::json!({ "filename": s.slot_file(id) }).to_string();
      let reply = http(
        port,
        "POST",
        &format!("/slots/{id}?action=restore"),
        Some(&body),
        CALL_TIMEOUT,
      )
      .await;
      match reply {
        Ok((200, v)) => {
          out.slots += 1;
          out.tokens += v
            .get("n_restored")
            .and_then(serde_json::Value::as_u64)
            .unwrap_or(0);
        }
        Ok((status, v)) => {
          log::warn!("slot cache: restore of slot {id} on port {port} answered {status}: {v}");
        }
        Err(e) => log::warn!("slot cache: restore of slot {id} on port {port} failed: {e}"),
      }
    }
    // The server now holds the cache (or cannot use the file); the next unload
    // writes a current one.
    let _ = tokio::fs::remove_file(&path).await;
  }
  if out.slots > 0 {
    log::info!(
      "slot cache: restored {} slot(s), {} tokens from {} in {} ms",
      out.slots,
      out.tokens,
      s.root.display(),
      started.elapsed().as_millis(),
    );
  }
  out
}

/// A saved slot holds the conversation's tokens, so keep the root owner-only.
fn create_private_dir(dir: &Path) -> std::io::Result<()> {
  std::fs::create_dir_all(dir)?;
  #[cfg(unix)]
  {
    use std::os::unix::fs::PermissionsExt;
    std::fs::set_permissions(dir, std::fs::Permissions::from_mode(0o700))?;
  }
  Ok(())
}

/// Delete files older than `max_age`, then the oldest files until the root
/// holds at most `max_bytes`. Returns the bytes deleted.
///
/// Oldest first means the save that just ran goes last. When it alone is over
/// the cap it is deleted too.
fn prune(root: &Path, max_bytes: u64, max_age: Duration, now: SystemTime) -> u64 {
  let mut files: Vec<(SystemTime, u64, PathBuf)> = Vec::new();
  let Ok(entries) = std::fs::read_dir(root) else {
    return 0;
  };
  for entry in entries.flatten() {
    let Ok(meta) = entry.metadata() else {
      continue;
    };
    if meta.is_file() {
      let mtime = meta.modified().unwrap_or(SystemTime::UNIX_EPOCH);
      files.push((mtime, meta.len(), entry.path()));
    }
  }
  files.sort();
  let mut total: u64 = files.iter().map(|(_, len, _)| len).sum();
  let mut deleted = 0u64;
  for (mtime, len, path) in &files {
    let expired = now.duration_since(*mtime).unwrap_or_default() > max_age;
    if !expired && total <= max_bytes {
      continue;
    }
    if std::fs::remove_file(path).is_ok() {
      log::info!(
        "slot cache: deleted {} ({}, {})",
        path.display(),
        crate::launch::admission::human_gib(*len),
        if expired { "expired" } else { "over the cap" },
      );
      total -= len;
      deleted += len;
    }
  }
  deleted
}

async fn total_slots(port: u16) -> Option<u32> {
  let (status, v) = http(port, "GET", "/props", None, PROPS_TIMEOUT)
    .await
    .ok()?;
  if status != 200 {
    return None;
  }
  v.get("total_slots")
    .and_then(serde_json::Value::as_u64)
    .map(|n| n as u32)
}

/// One request to the child on loopback, answered as `(status, JSON body)`.
/// Raw TCP with `Connection: close`, the same no-dependency transport as the
/// `/props` fetch in `actuals`.
async fn http(
  port: u16,
  method: &str,
  path: &str,
  body: Option<&str>,
  timeout: Duration,
) -> std::io::Result<(u16, serde_json::Value)> {
  let body = body.unwrap_or("");
  let request = format!(
    "{method} {path} HTTP/1.1\r\nHost: 127.0.0.1:{port}\r\nConnection: close\r\n\
     Accept: application/json\r\nContent-Type: application/json\r\nContent-Length: {}\r\n\r\n{body}",
    body.len()
  );
  let fut = async {
    let mut sock = TcpStream::connect(("127.0.0.1", port)).await?;
    sock.write_all(request.as_bytes()).await?;
    let mut buf = Vec::with_capacity(4096);
    let mut chunk = [0u8; 4096];
    loop {
      let n = sock.read(&mut chunk).await?;
      if n == 0 {
        break;
      }
      buf.extend_from_slice(&chunk[..n]);
      if buf.len() > 256 * 1024 {
        break;
      }
    }
    Ok::<_, std::io::Error>(buf)
  };
  let response = tokio::time::timeout(timeout, fut)
    .await
    .map_err(|_| std::io::Error::new(std::io::ErrorKind::TimedOut, "slot request timeout"))??;
  parse_response(&response)
    .ok_or_else(|| std::io::Error::new(std::io::ErrorKind::InvalidData, "unreadable response"))
}

fn parse_response(response: &[u8]) -> Option<(u16, serde_json::Value)> {
  let split = response.windows(4).position(|w| w == b"\r\n\r\n")?;
  let head = std::str::from_utf8(&response[..split]).ok()?;
  let status: u16 = head.split_whitespace().nth(1)?.parse().ok()?;
  let body = &response[split + 4..];
  // A chunked body wraps the JSON in size lines; the object itself is still
  // everything between the first `{` and the last `}`.
  let start = body.iter().position(|b| *b == b'{')?;
  let end = body.iter().rposition(|b| *b == b'}')?;
  let json = serde_json::from_slice(body.get(start..=end)?).ok()?;
  Some((status, json))
}

#[cfg(test)]
mod tests {
  use super::*;
  use std::sync::{Arc, Mutex};
  use tokio::net::TcpListener;

  fn temp(label: &str) -> PathBuf {
    crate::test_support::unique_temp_dir("ls-slot", label)
  }

  fn enabled(root: &Path) -> SlotCacheConfig {
    SlotCacheConfig {
      enabled: true,
      dir: Some(root.to_path_buf()),
      // A save needs a cap's worth of free disk; keep that small for CI.
      max_gib: 1,
      ..SlotCacheConfig::default()
    }
  }

  fn seeded(root: &Path, model: &str) -> LaunchParams {
    let mut params = LaunchParams::new(PathBuf::from(model), LaunchMode::Chat);
    seed(&enabled(root), &mut params);
    params
  }

  fn names(root: &Path) -> Vec<String> {
    let mut names: Vec<String> = std::fs::read_dir(root)
      .unwrap()
      .map(|e| e.unwrap().file_name().into_string().unwrap())
      .collect();
    names.sort();
    names
  }

  /// A stand-in for the slot endpoints: `/props` reports `tokens.len()` slots,
  /// a save writes `tokens[id]` as the file body, a restore reads it back.
  /// Every request line is recorded.
  async fn slot_server(dir: PathBuf, tokens: Vec<u64>) -> (u16, Arc<Mutex<Vec<String>>>) {
    let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let port = listener.local_addr().unwrap().port();
    let seen = Arc::new(Mutex::new(Vec::new()));
    let log = Arc::clone(&seen);
    tokio::spawn(async move {
      loop {
        let Ok((mut sock, _)) = listener.accept().await else {
          return;
        };
        let mut buf = vec![0u8; 8192];
        let read = sock.read(&mut buf).await.unwrap_or(0);
        let text = String::from_utf8_lossy(&buf[..read]).into_owned();
        let line = text.lines().next().unwrap_or("").to_string();
        log.lock().unwrap().push(line.clone());
        let filename = text
          .split("\r\n\r\n")
          .nth(1)
          .and_then(|b| serde_json::from_str::<serde_json::Value>(b).ok())
          .and_then(|v| v.get("filename")?.as_str().map(str::to_string))
          .unwrap_or_default();
        let id: usize = line
          .split("/slots/")
          .nth(1)
          .and_then(|r| r.split('?').next())
          .and_then(|s| s.parse().ok())
          .unwrap_or(0);
        let (status, body) = if line.starts_with("GET /props") {
          (
            200,
            serde_json::json!({ "total_slots": tokens.len() }).to_string(),
          )
        } else if line.contains("action=save") {
          std::fs::write(dir.join(&filename), tokens[id].to_string()).unwrap();
          (
            200,
            serde_json::json!({ "n_saved": tokens[id] }).to_string(),
          )
        } else if line.contains("action=restore") {
          match std::fs::read_to_string(dir.join(&filename))
            .ok()
            .and_then(|s| s.parse::<u64>().ok())
          {
            Some(n) => (200, serde_json::json!({ "n_restored": n }).to_string()),
            None => (
              400,
              r#"{"error":{"message":"invalid slot save file"}}"#.into(),
            ),
          }
        } else {
          (404, "{}".to_string())
        };
        let reply = format!(
          "HTTP/1.1 {status} X\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{body}",
          body.len()
        );
        let _ = sock.write_all(reply.as_bytes()).await;
      }
    });
    (port, seen)
  }

  #[test]
  fn off_by_default_and_seeds_nothing() {
    let mut params = LaunchParams::new(PathBuf::from("/m/a.gguf"), LaunchMode::Chat);
    seed(&SlotCacheConfig::default(), &mut params);
    assert!(params.launch_config.is_empty());
    assert!(settings(&params).is_none());
    assert!(argv(&params).is_empty());
  }

  #[test]
  fn turning_it_off_clears_keys_an_earlier_launch_left() {
    let root = temp("clear");
    let mut params = seeded(&root, "/m/a.gguf");
    assert!(params.launch_config.contains_key(KNOB_ROOT));
    seed(&SlotCacheConfig::default(), &mut params);
    assert!(params.launch_config.is_empty());
    std::fs::remove_dir_all(&root).ok();
  }

  #[test]
  fn only_chat_launches_without_their_own_save_path_are_seeded() {
    let base = temp("gate");
    let root = base.join("slot-cache");
    let mut embed = LaunchParams::new(PathBuf::from("/m/a.gguf"), LaunchMode::Embedding);
    seed(&enabled(&root), &mut embed);
    assert!(embed.launch_config.is_empty());

    for extras in [vec!["--slot-save-path", "/x"], vec!["--slot-save-path=/x"]] {
      let mut own = LaunchParams::new(PathBuf::from("/m/a.gguf"), LaunchMode::Chat);
      own.extras = extras.into_iter().map(OsString::from).collect();
      seed(&enabled(&root), &mut own);
      assert!(own.launch_config.is_empty());
    }
    assert!(!root.exists(), "nothing is created for a launch it skips");
    std::fs::remove_dir_all(&base).ok();
  }

  /// Regression: llama-server exits at startup with "not a directory" when the
  /// `--slot-save-path` target is missing, so the launch that names it must
  /// find it there.
  #[test]
  fn seeding_creates_the_directory_the_launch_argv_names() {
    let root = temp("mkdir").join("nested/slot-cache");
    let params = seeded(&root, "/m/a.gguf");
    let composed = super::super::compose::compose(&params, 4242);
    let flag = composed.iter().position(|x| x == SAVE_PATH_FLAG).unwrap();
    assert_eq!(PathBuf::from(&composed[flag + 1]), root);
    assert!(root.is_dir());
    #[cfg(unix)]
    {
      use std::os::unix::fs::PermissionsExt;
      let mode = std::fs::metadata(&root).unwrap().permissions().mode();
      assert_eq!(mode & 0o777, 0o700);
    }
    std::fs::remove_dir_all(&root).ok();
  }

  #[test]
  fn the_file_prefix_ignores_the_port_and_follows_the_argv_and_the_model_file() {
    let root = temp("key");
    std::fs::create_dir_all(&root).unwrap();
    let model = root.join("My Model.v2.gguf");
    std::fs::write(&model, b"one").unwrap();
    let model_str = model.to_str().unwrap();

    let a = settings(&seeded(&root, model_str)).unwrap();
    assert!(a.prefix.ends_with("-My_Model_v2."), "{}", a.prefix);
    assert_eq!(a.prefix.len(), 16 + "-My_Model_v2.".len());
    assert_eq!(settings(&seeded(&root, model_str)).unwrap(), a);

    let mut other_ctx = seeded(&root, model_str);
    other_ctx.ctx = Some(8192);
    assert_ne!(settings(&other_ctx).unwrap().prefix, a.prefix);

    std::fs::write(&model, b"replaced with other weights").unwrap();
    assert_ne!(
      settings(&seeded(&root, model_str)).unwrap().prefix,
      a.prefix
    );
    std::fs::remove_dir_all(&root).ok();
  }

  #[tokio::test]
  async fn save_keeps_only_slots_that_hold_tokens() {
    let root = temp("save");
    let params = seeded(&root, "/m/a.gguf");
    let s = settings(&params).unwrap();
    std::fs::write(root.join(s.slot_file(7)), b"stale").unwrap();
    std::fs::write(root.join("other-shape.slot-0.bin"), b"1").unwrap();
    let (port, seen) = slot_server(root.clone(), vec![0, 4295, 0, 12]).await;

    let saved = save(&params, port).await;

    assert_eq!(saved.slots, 2);
    assert_eq!(saved.tokens, 4307);
    let mut want = vec![
      s.slot_file(1),
      s.slot_file(3),
      "other-shape.slot-0.bin".to_string(),
    ];
    want.sort();
    assert_eq!(names(&root), want);
    assert_eq!(
      seen.lock().unwrap().len(),
      5,
      "/props plus one save per slot"
    );
    std::fs::remove_dir_all(&root).ok();
  }

  #[tokio::test]
  async fn save_is_skipped_when_the_server_does_not_answer() {
    let root = temp("dead");
    let params = seeded(&root, "/m/a.gguf");
    let port = {
      let l = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
      l.local_addr().unwrap().port()
    };
    assert_eq!(save(&params, port).await, Saved::default());
    assert!(names(&root).is_empty());
    std::fs::remove_dir_all(&root).ok();
  }

  #[tokio::test]
  async fn restore_puts_each_file_back_in_its_slot_and_deletes_it() {
    let root = temp("restore");
    let params = seeded(&root, "/m/a.gguf");
    let s = settings(&params).unwrap();
    std::fs::write(root.join(s.slot_file(1)), b"4295").unwrap();
    std::fs::write(root.join(s.slot_file(3)), b"12").unwrap();
    std::fs::write(root.join(format!("{}.part", s.slot_file(2))), b"cut").unwrap();
    std::fs::write(root.join("other-shape.slot-0.bin"), b"1").unwrap();
    let (port, seen) = slot_server(root.clone(), vec![0; 4]).await;

    let restored = restore(&params, port).await;

    assert_eq!(
      restored,
      Restored {
        slots: 2,
        tokens: 4307
      }
    );
    let lines = seen.lock().unwrap().clone();
    assert_eq!(lines.len(), 2, "{lines:?}");
    assert!(lines[0].starts_with("POST /slots/1?action=restore"));
    assert!(lines[1].starts_with("POST /slots/3?action=restore"));
    assert_eq!(
      names(&root),
      ["other-shape.slot-0.bin"],
      "its own files and leftovers go, another launch shape's file stays"
    );
    std::fs::remove_dir_all(&root).ok();
  }

  #[tokio::test]
  async fn a_file_the_server_rejects_is_deleted_not_retried() {
    let root = temp("reject");
    let params = seeded(&root, "/m/a.gguf");
    let s = settings(&params).unwrap();
    std::fs::write(root.join(s.slot_file(0)), b"not a number").unwrap();
    let (port, seen) = slot_server(root.clone(), vec![0]).await;

    assert_eq!(restore(&params, port).await, Restored::default());
    assert_eq!(seen.lock().unwrap().len(), 1);
    assert!(names(&root).is_empty());
    std::fs::remove_dir_all(&root).ok();
  }

  #[tokio::test]
  async fn an_expired_file_is_deleted_without_a_restore_call() {
    let root = temp("expired");
    let mut params = seeded(&root, "/m/a.gguf");
    params
      .launch_config
      .insert(KNOB_MAX_AGE.to_string(), "0".to_string());
    let s = settings(&params).unwrap();
    std::fs::write(root.join(s.slot_file(0)), b"9").unwrap();
    tokio::time::sleep(Duration::from_millis(20)).await;
    let (port, seen) = slot_server(root.clone(), vec![0]).await;

    assert_eq!(restore(&params, port).await, Restored::default());
    assert!(seen.lock().unwrap().is_empty());
    assert!(names(&root).is_empty());
    std::fs::remove_dir_all(&root).ok();
  }

  #[tokio::test]
  async fn restore_without_a_save_makes_no_call() {
    let root = temp("nothing");
    let params = seeded(&root, "/m/a.gguf");
    let (port, seen) = slot_server(root.clone(), vec![0]).await;
    assert_eq!(restore(&params, port).await, Restored::default());
    assert!(seen.lock().unwrap().is_empty());
    std::fs::remove_dir_all(&root).ok();
  }

  fn put(root: &Path, name: &str, len: usize, age_secs: u64, now: SystemTime) {
    std::fs::create_dir_all(root).unwrap();
    let path = root.join(name);
    std::fs::write(&path, vec![0u8; len]).unwrap();
    let f = std::fs::File::options().write(true).open(&path).unwrap();
    f.set_modified(now - Duration::from_secs(age_secs)).unwrap();
  }

  #[test]
  fn prune_deletes_expired_files_then_the_oldest_over_the_cap() {
    let root = temp("prune");
    let now = SystemTime::now();
    put(&root, "a.slot-0.bin", 100, 5000, now);
    put(&root, "b.slot-0.bin", 100, 300, now);
    put(&root, "b.slot-1.bin", 100, 200, now);
    put(&root, "c.slot-0.bin", 100, 100, now);

    // `a` is past the age limit; of the 300 bytes left, the oldest goes to
    // get under the 250-byte cap.
    let deleted = prune(&root, 250, Duration::from_secs(1000), now);

    assert_eq!(deleted, 200);
    assert_eq!(names(&root), ["b.slot-1.bin", "c.slot-0.bin"]);
    std::fs::remove_dir_all(&root).ok();
  }

  #[test]
  fn prune_deletes_a_save_that_alone_is_over_the_cap() {
    let root = temp("prune-big");
    let now = SystemTime::now();
    put(&root, "a.slot-0.bin", 500, 1, now);
    assert_eq!(prune(&root, 100, Duration::from_secs(1000), now), 500);
    assert!(names(&root).is_empty());
    std::fs::remove_dir_all(&root).ok();
  }

  #[test]
  fn prune_leaves_a_root_under_both_limits_alone() {
    let root = temp("prune-ok");
    let now = SystemTime::now();
    put(&root, "a.slot-0.bin", 50, 1, now);
    assert_eq!(prune(&root, 100, Duration::from_secs(1000), now), 0);
    assert_eq!(names(&root), ["a.slot-0.bin"]);
    assert_eq!(prune(&root.join("missing"), 100, Duration::ZERO, now), 0);
    std::fs::remove_dir_all(&root).ok();
  }

  #[test]
  fn parses_plain_and_chunked_replies() {
    let plain = b"HTTP/1.1 200 OK\r\nContent-Length: 13\r\n\r\n{\"n_saved\":7}";
    assert_eq!(
      parse_response(plain),
      Some((200, serde_json::json!({ "n_saved": 7 })))
    );
    let chunked =
      b"HTTP/1.1 400 Bad Request\r\nTransfer-Encoding: chunked\r\n\r\nd\r\n{\"error\":\"x\"}\r\n0\r\n\r\n";
    assert_eq!(
      parse_response(chunked),
      Some((400, serde_json::json!({ "error": "x" })))
    );
    assert_eq!(parse_response(b"garbage"), None);
  }
}
