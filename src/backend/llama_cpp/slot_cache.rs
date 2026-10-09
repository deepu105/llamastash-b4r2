//! Keep a launch's prompt cache across an eviction.
//!
//! `llama-server --slot-save-path DIR` exposes `POST /slots/{id}?action=save`
//! and `action=restore`, which write a slot's KV cache and its tokens to a file
//! and read them back. [`save`] runs before the idle sweep or make-room stops a
//! launch; [`restore`] runs when the next launch with the same argv turns
//! ready, before it takes traffic. llama-server picks a slot for a request by
//! prompt-prefix similarity, so restoring each file into the slot it came from
//! is enough for a returning conversation to find its cache.
//!
//! Files live in `<base>/<model-stem>-<key>/slot-<id>.bin`. The key hashes the
//! model file's size and mtime plus the launch argv, so a changed model or a
//! changed knob never reads a cache made under other settings. A file is
//! deleted once restored; [`prune`] removes what is too old or over the size
//! cap.

use std::ffi::OsString;
use std::path::{Path, PathBuf};
use std::time::{Duration, SystemTime};

use serde::{Deserialize, Serialize};
use tokio::io::{AsyncReadExt, AsyncWriteExt};
use tokio::net::TcpStream;

use crate::launch::mode::LaunchMode;
use crate::launch::params::LaunchParams;

/// `launch_config` keys, seeded per launch by [`seed`]. `DIR` present means the
/// cache is on for that launch.
const KNOB_DIR: &str = "slot_cache_dir";
const KNOB_MAX_BYTES: &str = "slot_cache_max_bytes";
const KNOB_MAX_AGE_SECS: &str = "slot_cache_max_age_secs";
const KNOB_MIN_TOKENS: &str = "slot_cache_min_tokens";

const SAVE_PATH_FLAG: &str = "--slot-save-path";
const BASE_DIR_NAME: &str = "slot-cache";
const FILE_PREFIX: &str = "slot-";
const FILE_SUFFIX: &str = ".bin";

/// One save or restore call. A 100k-token slot is a few GB; this bounds a hung
/// server, not a slow disk.
const ACTION_TIMEOUT: Duration = Duration::from_secs(120);
const PROPS_TIMEOUT: Duration = Duration::from_secs(5);

const GIB: u64 = 1024 * 1024 * 1024;

/// `backend.llamacpp.slot_cache` in `config.yaml`.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields, rename_all = "snake_case")]
pub struct SlotCacheConfig {
  /// Save slots before an eviction and restore them on the next launch.
  /// Factory `false`: a save writes GBs to disk and delays the eviction.
  pub enabled: bool,
  /// Where the files go. Unset: `<cache_dir>/slot-cache`.
  pub dir: Option<PathBuf>,
  /// Total size cap across every launch's files. Oldest files go first.
  pub max_gb: u64,
  /// A file older than this is deleted unread.
  pub max_age_hours: u64,
  /// A slot holding fewer tokens is not kept: reprocessing it is cheap.
  pub min_tokens: u32,
}

impl Default for SlotCacheConfig {
  fn default() -> Self {
    Self {
      enabled: false,
      dir: None,
      max_gb: 32,
      max_age_hours: 24,
      min_tokens: 1024,
    }
  }
}

impl SlotCacheConfig {
  fn base_dir(&self) -> Option<PathBuf> {
    self
      .dir
      .clone()
      .or_else(|| crate::util::paths::cache_dir().map(|d| d.join(BASE_DIR_NAME)))
  }
}

/// Project the config onto `params.launch_config`, or clear it. Cleared when
/// the cache is off, the launch is not a chat launch, the bench parity switch
/// is set, or the user passes their own `--slot-save-path` and so owns it.
pub(super) fn seed(cfg: &SlotCacheConfig, params: &mut LaunchParams) {
  for key in [KNOB_DIR, KNOB_MAX_BYTES, KNOB_MAX_AGE_SECS, KNOB_MIN_TOKENS] {
    params.launch_config.remove(key);
  }
  let user_owned = params.extras.iter().any(|a| {
    a.to_str()
      .is_some_and(|s| s == SAVE_PATH_FLAG || s.starts_with("--slot-save-path="))
  });
  if !cfg.enabled
    || user_owned
    || params.mode != LaunchMode::Chat
    || crate::launch::params::bench_disable_defaults_from_env()
  {
    return;
  }
  let Some(base) = cfg.base_dir() else {
    return;
  };
  let mut put = |key: &str, value: String| {
    params.launch_config.insert(key.to_string(), value);
  };
  put(KNOB_DIR, base.to_string_lossy().into_owned());
  put(KNOB_MAX_BYTES, cfg.max_gb.saturating_mul(GIB).to_string());
  put(
    KNOB_MAX_AGE_SECS,
    cfg.max_age_hours.saturating_mul(3600).to_string(),
  );
  put(KNOB_MIN_TOKENS, cfg.min_tokens.to_string());
}

/// The per-launch settings [`seed`] wrote, read back by the hooks.
#[derive(Debug, Clone, PartialEq, Eq)]
struct Settings {
  base: PathBuf,
  max_bytes: u64,
  max_age: Duration,
  min_tokens: u64,
}

fn settings(params: &LaunchParams) -> Option<Settings> {
  let cfg = &params.launch_config;
  let num = |key: &str| cfg.get(key).and_then(|s| s.parse::<u64>().ok());
  Some(Settings {
    base: PathBuf::from(cfg.get(KNOB_DIR)?),
    max_bytes: num(KNOB_MAX_BYTES)?,
    max_age: Duration::from_secs(num(KNOB_MAX_AGE_SECS)?),
    min_tokens: num(KNOB_MIN_TOKENS)?,
  })
}

/// This launch's directory, or `None` when the cache is off for it.
/// `argv` is the composed argv without the save-path flag.
fn launch_dir(params: &LaunchParams, argv: &[OsString]) -> Option<PathBuf> {
  let base = PathBuf::from(params.launch_config.get(KNOB_DIR)?);
  let meta = std::fs::metadata(&params.model_path).ok();
  let size = meta.as_ref().map_or(0, std::fs::Metadata::len);
  let mtime = meta
    .and_then(|m| m.modified().ok())
    .and_then(|t| t.duration_since(SystemTime::UNIX_EPOCH).ok())
    .map_or(0, |d| d.as_nanos());
  let mut hasher = blake3::Hasher::new();
  hasher.update(&size.to_le_bytes());
  hasher.update(&mtime.to_le_bytes());
  let mut after_port = false;
  for arg in argv {
    // The port is allocated per launch and has no bearing on the KV cache.
    if !std::mem::replace(&mut after_port, arg == "--port") {
      hasher.update(arg.as_encoded_bytes());
    }
    hasher.update(&[0]);
  }
  let key = hasher.finalize().to_hex();
  let stem: String = params
    .model_path
    .file_stem()
    .map(|s| s.to_string_lossy().into_owned())
    .unwrap_or_default()
    .chars()
    .map(|c| {
      if c.is_ascii_alphanumeric() || matches!(c, '-' | '_' | '.') {
        c
      } else {
        '_'
      }
    })
    .take(64)
    .collect();
  Some(base.join(format!("{stem}-{}", &key[..16])))
}

/// The `--slot-save-path` argv pair for `dir`. llama-server joins the path and
/// the file name with no separator, so the path ends in one.
fn save_path_argv(dir: &Path) -> [OsString; 2] {
  let mut path = dir.as_os_str().to_os_string();
  path.push(std::path::MAIN_SEPARATOR_STR);
  [SAVE_PATH_FLAG.into(), path]
}

/// The argv to append to `argv` for this launch: the save-path pair, or
/// nothing when the cache is off. Creates the directory, because llama-server
/// exits at startup when the path is not one; if that fails the launch goes
/// ahead without the cache.
pub(super) fn launch_argv(params: &LaunchParams, argv: &[OsString]) -> Vec<OsString> {
  let Some(dir) = launch_dir(params, argv) else {
    return Vec::new();
  };
  match create_private_dir(&dir) {
    Ok(()) => save_path_argv(&dir).into(),
    Err(e) => {
      log::warn!("slot cache: cannot create {}: {e}", dir.display());
      Vec::new()
    }
  }
}

fn slot_file_name(id: u64) -> String {
  format!("{FILE_PREFIX}{id}{FILE_SUFFIX}")
}

fn slot_id_of(name: &str) -> Option<u64> {
  name
    .strip_prefix(FILE_PREFIX)?
    .strip_suffix(FILE_SUFFIX)?
    .parse()
    .ok()
}

/// Slot files hold the conversation's tokens, so the directories are private.
fn create_private_dir(dir: &Path) -> std::io::Result<()> {
  let mut builder = std::fs::DirBuilder::new();
  builder.recursive(true);
  #[cfg(unix)]
  {
    use std::os::unix::fs::DirBuilderExt;
    builder.mode(0o700);
  }
  builder.create(dir)
}

/// Save every slot of the server on `port` that holds at least `min_tokens`.
/// Best-effort: a failure logs and leaves the eviction to go ahead.
pub(super) async fn save(port: u16, params: &LaunchParams) {
  let Some(cfg) = settings(params) else {
    return;
  };
  let Some(dir) = launch_dir(params, &super::compose(params, 0)) else {
    return;
  };
  let Some(slots) = total_slots(port).await else {
    log::warn!("slot cache: no slot count from port {port}, nothing saved");
    return;
  };
  if let Err(e) = create_private_dir(&dir) {
    log::warn!("slot cache: cannot create {}: {e}", dir.display());
    return;
  }
  for id in 0..slots {
    let name = slot_file_name(id);
    let path = dir.join(&name);
    match slot_action(port, id, "save", &name).await {
      Ok(reply) => {
        let tokens = reply.get("n_saved").and_then(|v| v.as_u64()).unwrap_or(0);
        if tokens < cfg.min_tokens {
          let _ = std::fs::remove_file(&path);
          continue;
        }
        log::info!(
          "slot cache: saved slot {id} of port {port}: {tokens} tokens, {} MiB in {:.0} ms",
          reply_mib(&reply, "n_written"),
          action_ms(&reply, "save_ms"),
        );
      }
      Err(e) => {
        // A failed save can leave a partial file; never restore it.
        let _ = std::fs::remove_file(&path);
        log::warn!("slot cache: saving slot {id} of port {port} failed: {e}");
      }
    }
  }
  prune_blocking(cfg).await;
}

/// Restore every file saved for this launch into the slot it came from, then
/// delete it. Runs before the launch is declared ready.
pub(super) async fn restore(port: u16, params: &LaunchParams) {
  let Some(cfg) = settings(params) else {
    return;
  };
  let Some(dir) = launch_dir(params, &super::compose(params, 0)) else {
    return;
  };
  prune_blocking(cfg).await;
  let files = slot_files(&dir);
  if files.is_empty() {
    return;
  }
  // A slot id past the new server's count has nowhere to go.
  let slots = total_slots(port).await.unwrap_or(0);
  for (id, path) in files {
    if id < slots {
      match slot_action(port, id, "restore", &slot_file_name(id)).await {
        Ok(reply) => log::info!(
          "slot cache: restored slot {id} on port {port}: {} tokens, {} MiB in {:.0} ms",
          reply
            .get("n_restored")
            .and_then(|v| v.as_u64())
            .unwrap_or(0),
          reply_mib(&reply, "n_read"),
          action_ms(&reply, "restore_ms"),
        ),
        Err(e) => log::warn!("slot cache: restoring slot {id} on port {port} failed: {e}"),
      }
    }
    let _ = std::fs::remove_file(&path);
  }
  let _ = std::fs::remove_dir(&dir);
}

fn reply_mib(reply: &serde_json::Value, field: &str) -> u64 {
  reply.get(field).and_then(|v| v.as_u64()).unwrap_or(0) / (1024 * 1024)
}

fn action_ms(reply: &serde_json::Value, field: &str) -> f64 {
  reply
    .pointer(&format!("/timings/{field}"))
    .and_then(|v| v.as_f64())
    .unwrap_or(0.0)
}

/// `(slot id, path)` for each slot file in `dir`, by slot id.
fn slot_files(dir: &Path) -> Vec<(u64, PathBuf)> {
  let Ok(entries) = std::fs::read_dir(dir) else {
    return Vec::new();
  };
  let mut files: Vec<(u64, PathBuf)> = entries
    .flatten()
    .filter_map(|e| Some((slot_id_of(e.file_name().to_str()?)?, e.path())))
    .collect();
  files.sort();
  files
}

async fn prune_blocking(cfg: Settings) {
  let _ = tokio::task::spawn_blocking(move || {
    prune(&cfg.base, cfg.max_bytes, cfg.max_age, SystemTime::now())
  })
  .await;
}

/// Delete slot files older than `max_age`, then the oldest ones until the
/// total is within `max_bytes`, then any launch directory left empty. Returns
/// the bytes kept.
fn prune(base: &Path, max_bytes: u64, max_age: Duration, now: SystemTime) -> u64 {
  let Ok(launch_dirs) = std::fs::read_dir(base) else {
    return 0;
  };
  let mut dirs: Vec<PathBuf> = Vec::new();
  let mut files: Vec<(SystemTime, u64, PathBuf)> = Vec::new();
  for dir in launch_dirs.flatten().map(|e| e.path()) {
    for (_, path) in slot_files(&dir) {
      let Ok(meta) = std::fs::metadata(&path) else {
        continue;
      };
      let modified = meta.modified().unwrap_or(SystemTime::UNIX_EPOCH);
      files.push((modified, meta.len(), path));
    }
    dirs.push(dir);
  }
  files.sort();
  let mut total: u64 = files.iter().map(|f| f.1).sum();
  for (modified, len, path) in &files {
    let expired = now.duration_since(*modified).is_ok_and(|age| age > max_age);
    if (expired || total > max_bytes) && std::fs::remove_file(path).is_ok() {
      log::info!(
        "slot cache: removed {} ({})",
        path.display(),
        if expired {
          "too old"
        } else {
          "over the size cap"
        }
      );
      total -= len;
    }
  }
  for dir in dirs {
    // Fails on a non-empty directory, which is the check.
    let _ = std::fs::remove_dir(dir);
  }
  total
}

async fn total_slots(port: u16) -> Option<u64> {
  let raw = super::actuals::fetch_props_body(port, PROPS_TIMEOUT)
    .await
    .ok()?;
  let (status, body) = split_response(&raw)?;
  if status != 200 {
    return None;
  }
  serde_json::from_slice::<serde_json::Value>(body)
    .ok()?
    .get("total_slots")?
    .as_u64()
}

/// `POST /slots/{id}?action={action}` with the file name, returning the JSON
/// reply of a 200.
async fn slot_action(
  port: u16,
  id: u64,
  action: &str,
  filename: &str,
) -> Result<serde_json::Value, String> {
  let body = serde_json::json!({ "filename": filename }).to_string();
  let request = format!(
    "POST /slots/{id}?action={action} HTTP/1.1\r\nHost: 127.0.0.1:{port}\r\nConnection: \
     close\r\nContent-Type: application/json\r\nContent-Length: {}\r\n\r\n{body}",
    body.len()
  );
  let exchange = async {
    let mut sock = TcpStream::connect(("127.0.0.1", port)).await?;
    sock.write_all(request.as_bytes()).await?;
    let mut raw = Vec::with_capacity(1024);
    sock.take(256 * 1024).read_to_end(&mut raw).await?;
    Ok::<_, std::io::Error>(raw)
  };
  let raw = tokio::time::timeout(ACTION_TIMEOUT, exchange)
    .await
    .map_err(|_| format!("no reply in {} s", ACTION_TIMEOUT.as_secs()))?
    .map_err(|e| e.to_string())?;
  let (status, reply) = split_response(&raw).ok_or("malformed HTTP reply")?;
  if status != 200 {
    return Err(format!(
      "HTTP {status}: {}",
      String::from_utf8_lossy(reply).trim()
    ));
  }
  serde_json::from_slice(reply).map_err(|e| e.to_string())
}

/// `(status, body)` of a `Connection: close` HTTP/1.1 response.
fn split_response(raw: &[u8]) -> Option<(u16, &[u8])> {
  let split = raw.windows(4).position(|w| w == b"\r\n\r\n")?;
  let status = std::str::from_utf8(&raw[..split])
    .ok()?
    .split_whitespace()
    .nth(1)?
    .parse()
    .ok()?;
  Some((status, &raw[split + 4..]))
}

#[cfg(test)]
mod tests {
  use super::*;
  use std::sync::{Arc, Mutex};
  use tokio::net::TcpListener;

  fn enabled_cfg(base: &Path) -> SlotCacheConfig {
    SlotCacheConfig {
      enabled: true,
      dir: Some(base.to_path_buf()),
      min_tokens: 100,
      ..SlotCacheConfig::default()
    }
  }

  fn seeded(base: &Path) -> LaunchParams {
    let mut params = LaunchParams::new(base.join("model.gguf"), LaunchMode::Chat);
    seed(&enabled_cfg(base), &mut params);
    params
  }

  fn dir_of(params: &LaunchParams) -> PathBuf {
    launch_dir(params, &crate::backend::llama_cpp::compose(params, 0)).unwrap()
  }

  /// A llama-server stand-in: `/props` reports `slots` slots; a save writes
  /// `n_saved[id]` tokens' worth of bytes into `dir`; a restore of an existing
  /// file answers 200 and anything else 400. Every request line is recorded.
  async fn fake_server(
    dir: PathBuf,
    slots: u64,
    n_saved: Vec<u64>,
  ) -> (u16, Arc<Mutex<Vec<String>>>) {
    let listener = TcpListener::bind(("127.0.0.1", 0)).await.unwrap();
    let port = listener.local_addr().unwrap().port();
    let seen = Arc::new(Mutex::new(Vec::new()));
    let log = Arc::clone(&seen);
    tokio::spawn(async move {
      loop {
        let Ok((mut sock, _)) = listener.accept().await else {
          return;
        };
        let mut buf = vec![0u8; 8192];
        let n = sock.read(&mut buf).await.unwrap_or(0);
        let text = String::from_utf8_lossy(&buf[..n]).into_owned();
        let line = text.lines().next().unwrap_or_default().to_string();
        log.lock().unwrap().push(line.clone());
        let filename = text
          .rsplit("\r\n\r\n")
          .next()
          .and_then(|b| serde_json::from_str::<serde_json::Value>(b).ok())
          .and_then(|v| v.get("filename")?.as_str().map(str::to_string))
          .unwrap_or_default();
        let id = slot_id_of(&filename).unwrap_or(0);
        let (status, reply) = if line.starts_with("GET /props") {
          (200, serde_json::json!({ "total_slots": slots }))
        } else if line.contains("action=save") {
          let tokens = n_saved.get(id as usize).copied().unwrap_or(0);
          std::fs::write(dir.join(&filename), vec![0u8; tokens as usize]).unwrap();
          (
            200,
            serde_json::json!({ "n_saved": tokens, "n_written": tokens, "timings": { "save_ms": 1.5 } }),
          )
        } else if line.contains("action=restore") && dir.join(&filename).exists() {
          (
            200,
            serde_json::json!({ "n_restored": 7, "n_read": 7, "timings": { "restore_ms": 1.5 } }),
          )
        } else {
          (400, serde_json::json!({ "error": { "message": "nope" } }))
        };
        let payload = reply.to_string();
        let response = format!(
          "HTTP/1.1 {status} X\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{payload}",
          payload.len()
        );
        let _ = sock.write_all(response.as_bytes()).await;
      }
    });
    (port, seen)
  }

  #[test]
  fn config_is_off_by_default_and_parses_every_field() {
    let cfg = SlotCacheConfig::default();
    assert!(!cfg.enabled);
    assert_eq!(
      (cfg.max_gb, cfg.max_age_hours, cfg.min_tokens),
      (32, 24, 1024)
    );
    let parsed: SlotCacheConfig = yaml_serde::from_str(
      "enabled: true\ndir: /var/slots\nmax_gb: 4\nmax_age_hours: 2\nmin_tokens: 50\n",
    )
    .unwrap();
    assert_eq!(
      parsed,
      SlotCacheConfig {
        enabled: true,
        dir: Some(PathBuf::from("/var/slots")),
        max_gb: 4,
        max_age_hours: 2,
        min_tokens: 50,
      }
    );
    assert!(yaml_serde::from_str::<SlotCacheConfig>("enable: true\n").is_err());
  }

  #[test]
  fn seed_writes_the_settings_and_clears_them_when_off() {
    let base = PathBuf::from("/var/slots");
    let mut params = seeded(&base);
    assert_eq!(
      settings(&params),
      Some(Settings {
        base: base.clone(),
        max_bytes: 32 * GIB,
        max_age: Duration::from_secs(24 * 3600),
        min_tokens: 100,
      })
    );
    // An inherited launch_config must not keep the cache on.
    seed(&SlotCacheConfig::default(), &mut params);
    assert_eq!(settings(&params), None);
    assert!(params.launch_config.is_empty());
  }

  #[test]
  fn seed_skips_non_chat_launches_and_a_user_owned_save_path() {
    let base = PathBuf::from("/var/slots");
    let mut embedding = LaunchParams::new(base.join("model.gguf"), LaunchMode::Chat);
    embedding.mode = LaunchMode::Embedding;
    seed(&enabled_cfg(&base), &mut embedding);
    assert_eq!(settings(&embedding), None);

    for extras in [
      vec!["--slot-save-path", "/mine"],
      vec!["--slot-save-path=/mine"],
    ] {
      let mut params = LaunchParams::new(base.join("model.gguf"), LaunchMode::Chat);
      params.extras = extras.into_iter().map(OsString::from).collect();
      seed(&enabled_cfg(&base), &mut params);
      assert_eq!(settings(&params), None);
    }
  }

  #[test]
  fn launch_dir_ignores_the_port_and_follows_the_argv_and_the_model_file() {
    let base = crate::util::test_temp::unique_temp_dir("slot-cache-key");
    let model = base.join("My Model.Q4.gguf");
    std::fs::write(&model, b"one").unwrap();
    let mut params = LaunchParams::new(model.clone(), LaunchMode::Chat);
    seed(&enabled_cfg(&base), &mut params);
    let compose = crate::backend::llama_cpp::compose;

    let dir = launch_dir(&params, &compose(&params, 1111)).unwrap();
    assert_eq!(dir.parent(), Some(base.as_path()));
    let name = dir.file_name().unwrap().to_str().unwrap();
    assert!(name.starts_with("My_Model.Q4-"), "{name}");
    assert_eq!(name.len(), "My_Model.Q4-".len() + 16);
    assert_eq!(launch_dir(&params, &compose(&params, 2222)).unwrap(), dir);

    let mut other_ctx = params.clone();
    other_ctx.ctx = Some(4096);
    assert_ne!(
      launch_dir(&other_ctx, &compose(&other_ctx, 1111)).unwrap(),
      dir
    );

    std::fs::write(&model, b"a different file").unwrap();
    assert_ne!(launch_dir(&params, &compose(&params, 1111)).unwrap(), dir);

    params.launch_config.clear();
    assert_eq!(launch_dir(&params, &compose(&params, 1111)), None);
    let _ = std::fs::remove_dir_all(&base);
  }

  #[test]
  fn launch_argv_creates_the_directory_it_names() {
    let base = crate::util::test_temp::unique_temp_dir("slot-cache-argv");
    let params = seeded(&base);
    let argv = crate::backend::llama_cpp::compose(&params, 1234);
    let extra = launch_argv(&params, &argv);
    let dir = dir_of(&params);
    assert_eq!(extra, save_path_argv(&dir));
    assert!(
      dir.is_dir(),
      "llama-server exits when the path is not a dir"
    );

    // Off: no flag and nothing on disk.
    let off = LaunchParams::new(base.join("other.gguf"), LaunchMode::Chat);
    assert!(launch_argv(&off, &argv).is_empty());

    // A base that cannot be created: the launch goes ahead without the flag.
    std::fs::write(base.join("file"), b"x").unwrap();
    let mut blocked = params.clone();
    blocked.launch_config.insert(
      KNOB_DIR.to_string(),
      base.join("file").to_string_lossy().into_owned(),
    );
    assert!(launch_argv(&blocked, &argv).is_empty());
    let _ = std::fs::remove_dir_all(&base);
  }

  #[test]
  fn save_path_argv_ends_in_a_separator() {
    let [flag, path] = save_path_argv(Path::new("/var/slots/m-abc"));
    assert_eq!(flag, "--slot-save-path");
    assert_eq!(path, "/var/slots/m-abc/");
  }

  #[test]
  fn slot_file_names_round_trip_and_reject_other_files() {
    assert_eq!(slot_id_of(&slot_file_name(3)), Some(3));
    for name in [
      "slot-.bin",
      "slot-x.bin",
      "slot-1.tmp",
      "other-1.bin",
      "slot-1",
    ] {
      assert_eq!(slot_id_of(name), None, "{name}");
    }
  }

  #[test]
  fn split_response_reads_status_and_body() {
    assert_eq!(
      split_response(b"HTTP/1.1 200 OK\r\nA: b\r\n\r\n{}"),
      Some((200, &b"{}"[..]))
    );
    assert_eq!(
      split_response(b"HTTP/1.1 501 Not Implemented\r\n\r\n"),
      Some((501, &b""[..]))
    );
    assert_eq!(split_response(b"garbage"), None);
  }

  #[test]
  fn prune_drops_old_files_then_the_oldest_over_the_cap() {
    let base = crate::util::test_temp::unique_temp_dir("slot-cache-prune");
    let now = SystemTime::now();
    let file = |launch: &str, id: u64, len: usize, age_secs: u64| {
      let dir = base.join(launch);
      std::fs::create_dir_all(&dir).unwrap();
      let path = dir.join(slot_file_name(id));
      std::fs::write(&path, vec![0u8; len]).unwrap();
      std::fs::File::options()
        .write(true)
        .open(&path)
        .unwrap()
        .set_modified(now - Duration::from_secs(age_secs))
        .unwrap();
      path
    };
    let expired = file("a", 0, 10, 5000);
    let oldest = file("a", 1, 40, 300);
    let middle = file("b", 0, 40, 200);
    let newest = file("c", 0, 40, 100);
    let stranger = base.join("c").join("notes.txt");
    std::fs::write(&stranger, b"keep").unwrap();

    // 130 bytes on disk, cap 100: the expired file goes for its age, then the
    // oldest live one for size.
    let kept = prune(&base, 100, Duration::from_secs(3600), now);
    assert_eq!(kept, 80);
    assert!(!expired.exists() && !oldest.exists());
    assert!(middle.exists() && newest.exists() && stranger.exists());
    assert!(!base.join("a").exists(), "an emptied launch dir is removed");

    // A cap below one file removes everything, newest included.
    assert_eq!(prune(&base, 10, Duration::from_secs(3600), now), 0);
    assert!(!middle.exists() && !newest.exists());
    assert!(!base.join("b").exists());
    assert!(
      stranger.exists(),
      "files that are not slot files are left alone"
    );
    assert_eq!(prune(&base.join("missing"), 10, Duration::ZERO, now), 0);
    let _ = std::fs::remove_dir_all(&base);
  }

  #[tokio::test]
  async fn save_keeps_slots_over_the_token_floor_and_restore_consumes_them() {
    let base = crate::util::test_temp::unique_temp_dir("slot-cache-roundtrip");
    let params = seeded(&base);
    let dir = dir_of(&params);

    // Slot 0 is over the 100-token floor, slot 1 under it, slot 2 over.
    let (port, seen) = fake_server(dir.clone(), 3, vec![500, 20, 300]).await;
    save(port, &params).await;
    assert_eq!(
      slot_files(&dir)
        .into_iter()
        .map(|f| f.0)
        .collect::<Vec<_>>(),
      vec![0, 2]
    );
    #[cfg(unix)]
    {
      use std::os::unix::fs::PermissionsExt;
      let mode = std::fs::metadata(&dir).unwrap().permissions().mode();
      assert_eq!(mode & 0o777, 0o700);
    }

    // The next server has two slots: slot 0 is restored, slot 2 has no slot to
    // go to, and both files are gone afterwards.
    let (port2, seen2) = fake_server(dir.clone(), 2, Vec::new()).await;
    restore(port2, &params).await;
    assert!(!dir.exists());
    let restores: Vec<String> = seen2
      .lock()
      .unwrap()
      .iter()
      .filter(|l| l.starts_with("POST"))
      .cloned()
      .collect();
    assert_eq!(restores, vec!["POST /slots/0?action=restore HTTP/1.1"]);
    assert_eq!(
      seen
        .lock()
        .unwrap()
        .iter()
        .filter(|l| l.contains("action=save"))
        .count(),
      3
    );
    let _ = std::fs::remove_dir_all(&base);
  }

  #[tokio::test]
  async fn a_failed_restore_still_deletes_the_file() {
    let base = crate::util::test_temp::unique_temp_dir("slot-cache-bad-restore");
    let params = seeded(&base);
    let dir = dir_of(&params);
    std::fs::create_dir_all(&dir).unwrap();
    std::fs::write(dir.join(slot_file_name(0)), b"x").unwrap();
    // The server looks in another directory, so the restore answers 400.
    let (port, seen) = fake_server(base.join("elsewhere"), 1, Vec::new()).await;
    restore(port, &params).await;
    assert!(!dir.exists());
    assert_eq!(seen.lock().unwrap().len(), 2);
    let _ = std::fs::remove_dir_all(&base);
  }

  #[tokio::test]
  async fn hooks_do_nothing_when_the_cache_is_off_or_the_server_is_gone() {
    let base = crate::util::test_temp::unique_temp_dir("slot-cache-off");
    let off = LaunchParams::new(base.join("model.gguf"), LaunchMode::Chat);
    let (port, seen) = fake_server(base.clone(), 1, vec![500]).await;
    save(port, &off).await;
    restore(port, &off).await;
    assert!(seen.lock().unwrap().is_empty());

    // Enabled, but nothing listens: no directory is left behind.
    let params = seeded(&base);
    let dead = {
      let l = TcpListener::bind(("127.0.0.1", 0)).await.unwrap();
      l.local_addr().unwrap().port()
    };
    save(dead, &params).await;
    restore(dead, &params).await;
    assert!(!dir_of(&params).exists());
    let _ = std::fs::remove_dir_all(&base);
  }

  #[tokio::test]
  async fn save_enforces_the_size_cap() {
    let base = crate::util::test_temp::unique_temp_dir("slot-cache-cap");
    let mut params = seeded(&base);
    params
      .launch_config
      .insert(KNOB_MAX_BYTES.to_string(), "600".to_string());
    let dir = dir_of(&params);
    let (port, _seen) = fake_server(dir.clone(), 2, vec![500, 400]).await;
    save(port, &params).await;
    // 900 bytes written against a 600-byte cap: the older file goes.
    assert_eq!(slot_files(&dir).len(), 1);
    let _ = std::fs::remove_dir_all(&base);
  }
}
