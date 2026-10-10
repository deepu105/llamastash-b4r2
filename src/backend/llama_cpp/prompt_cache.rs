//! Keep a launch's prompt cache across a proxy eviction.
//!
//! When the idle sweep or make-room stops a launch, the next request for that
//! model starts a fresh `llama-server` and reprocesses its whole prompt.
//! llama-server can write one slot's KV cache to a file and read it back
//! (`--slot-save-path DIR`, `POST /slots/{id}?action=save|restore`), so:
//!
//! - every chat launch is started with `--slot-save-path <cache_dir>/slot-cache`
//!   (`seed`);
//! - an eviction saves each slot that holds a prompt before the stop (`save`,
//!   behind [`Backend::before_evict`](crate::backend::Backend::before_evict));
//! - the next launch of the same model with the same argv reads those files
//!   back before it is published `Ready`, then deletes them (`restore`, behind
//!   [`Backend::after_ready`](crate::backend::Backend::after_ready)).
//!
//! Nothing here matches a restored slot to a conversation. llama-server gives
//! an incoming prompt to the slot whose cached tokens share the longest prefix
//! with it, so the returning conversation lands on its restored slot and any
//! other request overwrites one.
//!
//! Measured save / restore / reprocess numbers:
//! `docs/spikes/2026-10-10-slot-save-restore.md`.

use std::path::{Path, PathBuf};
use std::time::{Duration, Instant, SystemTime};

use serde::{Deserialize, Serialize};

use crate::gguf::header::{GgufHeader, HeaderReadOptions};
use crate::gguf::memory::{kv_bytes, parse_cache_type, CacheType, EstimateOptions};
use crate::launch::knobs::Concept;
use crate::launch::mode::LaunchMode;
use crate::launch::params::LaunchParams;

/// `launch_config` keys carrying the slot directory and its limits. The save
/// and restore hooks run without a daemon context, so everything they need
/// rides on the launch's own params.
pub const LLAMACPP_KNOB_SLOT_SAVE_DIR: &str = "slot_save_dir";
const KNOB_MAX_BYTES: &str = "slot_save_max_bytes";
const KNOB_MAX_AGE_SECS: &str = "slot_save_max_age_secs";
const KNOB_MIN_TOKENS: &str = "slot_save_min_tokens";

/// [`super::caps`] probe key: whether the build accepts `--slot-save-path`.
pub const CAP_SLOT_SAVE: &str = "slot_save";

/// Directory under the cache dir. One flat directory for every launch:
/// llama-server takes a bare filename, never a path.
const DIR_NAME: &str = "slot-cache";

/// Hex characters of the launch key in a file name.
const KEY_LEN: usize = 16;

/// Time allowed for all of one eviction's saves. A make-room eviction has a
/// client waiting behind it, so a save that cannot finish in this window is
/// dropped and the stop goes ahead.
const SAVE_BUDGET: Duration = Duration::from_secs(60);

/// Time the readiness path waits for all of one launch's restores before it
/// publishes `Ready` anyway.
const RESTORE_BUDGET: Duration = Duration::from_secs(60);

/// `GET /slots` is answered from memory.
const LIST_TIMEOUT: Duration = Duration::from_secs(5);

/// Free space a save must leave on the cache filesystem.
const DISK_HEADROOM_BYTES: u64 = 2 * 1024 * 1024 * 1024;

const GIB: u64 = 1024 * 1024 * 1024;

/// `backend.llamacpp.prompt_cache` in `config.yaml`.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields, rename_all = "snake_case")]
pub struct PromptCacheConfig {
  /// Save a launch's slots when the proxy evicts it and restore them on its
  /// next start. Factory `true`.
  pub enabled: bool,
  /// Size cap for the whole slot directory, in GiB. A slot estimated above it
  /// is not saved; past it the oldest files are deleted first.
  pub max_gb: u64,
  /// A saved file older than this is deleted unrestored. `0` keeps files
  /// until the size cap removes them.
  pub max_age_hours: u64,
  /// Slots holding fewer prompt tokens than this are not saved.
  pub min_tokens: u32,
}

impl Default for PromptCacheConfig {
  fn default() -> Self {
    Self {
      enabled: true,
      max_gb: 32,
      max_age_hours: 24,
      min_tokens: 1024,
    }
  }
}

/// Whether a build's `--help` lists `--slot-save-path`.
pub fn supported_in_help(help: &str) -> bool {
  help.contains("--slot-save-path")
}

/// Project the config onto one launch: set the slot directory and limits, or
/// clear them so a value inherited from `last_params` cannot outlive a config
/// flip. Touches no disk; [`confirm`] finishes the job once the launch's
/// binary is known.
pub(super) fn seed(cfg: &PromptCacheConfig, params: &mut LaunchParams) {
  seed_keys(cfg, crate::util::paths::cache_dir(), params);
}

fn seed_keys(cfg: &PromptCacheConfig, cache_dir: Option<PathBuf>, params: &mut LaunchParams) {
  clear(params);
  // Embedding and rerank launches hold no conversation. A hand-passed
  // `--slot-save-path` means the user manages slot files themselves.
  if !cfg.enabled
    || params.mode != LaunchMode::Chat
    || crate::launch::params::bench_disable_defaults_from_env()
    || crate::launch::params::extras_have_flag(&params.extras, "--slot-save-path")
  {
    return;
  }
  let Some(dir) = cache_dir.map(|d| d.join(DIR_NAME)) else {
    return;
  };
  let Some(dir_str) = dir.to_str() else {
    return;
  };
  let config = &mut params.launch_config;
  config.insert(LLAMACPP_KNOB_SLOT_SAVE_DIR.into(), dir_str.to_string());
  config.insert(
    KNOB_MAX_BYTES.into(),
    cfg.max_gb.saturating_mul(GIB).to_string(),
  );
  config.insert(
    KNOB_MAX_AGE_SECS.into(),
    cfg.max_age_hours.saturating_mul(3600).to_string(),
  );
  config.insert(KNOB_MIN_TOKENS.into(), cfg.min_tokens.to_string());
}

/// Keep the seeded keys only if this launch can use them: the build accepts
/// `--slot-save-path` and the directory exists. llama-server refuses to start
/// on either miss, so both drop the keys instead.
pub(super) fn confirm(params: &mut LaunchParams, build_supports_flag: bool) {
  let Some(dir) = params
    .launch_config
    .get(LLAMACPP_KNOB_SLOT_SAVE_DIR)
    .map(PathBuf::from)
  else {
    return;
  };
  if !build_supports_flag {
    clear(params);
    return;
  }
  if let Err(e) = create_private_dir(&dir) {
    log::warn!(
      "prompt cache: cannot create {}: {e}; launching without it",
      dir.display()
    );
    clear(params);
  }
}

/// `seed` and `confirm` in one call, with the cache directory passed in.
/// Public so a test can point a launch at its own directory without setting
/// `LLAMASTASH_CACHE_DIR`.
pub fn seed_into(cfg: &PromptCacheConfig, cache_dir: Option<PathBuf>, params: &mut LaunchParams) {
  seed_keys(cfg, cache_dir, params);
  confirm(params, true);
}

/// Drop the slot keys, so `compose` emits no `--slot-save-path` and both hooks
/// do nothing.
fn clear(params: &mut LaunchParams) {
  for key in [
    LLAMACPP_KNOB_SLOT_SAVE_DIR,
    KNOB_MAX_BYTES,
    KNOB_MAX_AGE_SECS,
    KNOB_MIN_TOKENS,
  ] {
    params.launch_config.remove(key);
  }
}

/// Saved files hold prompt text as tokens, so the directory is owner-only.
fn create_private_dir(dir: &Path) -> std::io::Result<()> {
  std::fs::create_dir_all(dir)?;
  #[cfg(unix)]
  {
    use std::os::unix::fs::PermissionsExt;
    std::fs::set_permissions(dir, std::fs::Permissions::from_mode(0o700))?;
  }
  Ok(())
}

/// The slot directory and limits one launch was started with.
#[derive(Debug, Clone, PartialEq, Eq)]
struct Settings {
  dir: PathBuf,
  max_bytes: u64,
  max_age: Duration,
  min_tokens: u64,
}

impl Settings {
  fn from_launch(params: &LaunchParams) -> Option<Self> {
    let get = |key: &str| params.launch_config.get(key);
    Some(Self {
      dir: PathBuf::from(get(LLAMACPP_KNOB_SLOT_SAVE_DIR)?),
      max_bytes: get(KNOB_MAX_BYTES)?.parse().ok()?,
      max_age: Duration::from_secs(get(KNOB_MAX_AGE_SECS)?.parse().ok()?),
      min_tokens: get(KNOB_MIN_TOKENS)?.parse().ok()?,
    })
  }
}

/// Names the files of one launch configuration: the composed argv (without
/// the port and the slot directory) plus the size and mtime of the model and
/// projector files.
///
/// llama-server checks a restored file's KV layout but not which weights
/// wrote it, so a file from other weights, another LoRA or another context
/// size must not be offered at all. Any argv change gives a new key, which
/// costs one reprocess and can never restore a cache that does not belong.
fn launch_key(params: &LaunchParams) -> String {
  let mut bare = params.clone();
  clear(&mut bare);
  let mut hasher = blake3::Hasher::new();
  for arg in super::compose::compose(&bare, 0) {
    hasher.update(arg.as_encoded_bytes());
    hasher.update(&[0]);
  }
  for path in [Some(&params.model_path), params.mmproj_path.as_ref()]
    .into_iter()
    .flatten()
  {
    if let Ok(meta) = std::fs::metadata(path) {
      hasher.update(&meta.len().to_le_bytes());
      let mtime = meta
        .modified()
        .ok()
        .and_then(|t| t.duration_since(SystemTime::UNIX_EPOCH).ok())
        .map_or(0, |d| d.as_nanos());
      hasher.update(&mtime.to_le_bytes());
    }
  }
  let mut key = hasher.finalize().to_hex().to_string();
  key.truncate(KEY_LEN);
  key
}

fn slot_file_name(key: &str, slot: u32) -> String {
  format!("{key}-slot{slot}.bin")
}

/// `(key, slot)` when `name` is a file this module named.
fn parse_slot_file(name: &str) -> Option<(&str, u32)> {
  let (key, slot) = name.strip_suffix(".bin")?.split_once("-slot")?;
  if key.len() != KEY_LEN || !key.bytes().all(|b| b.is_ascii_hexdigit()) {
    return None;
  }
  Some((key, slot.parse().ok()?))
}

#[derive(Debug)]
struct CacheFile {
  path: PathBuf,
  /// `(key, slot)` for a file this module named; `None` for anything else.
  owner: Option<(String, u32)>,
  len: u64,
  modified: SystemTime,
}

/// Every regular file in the slot directory. The directory belongs to this
/// module, and llama-server writes whatever filename a `/slots` caller hands
/// it, so the cap and the age limit cover files this module did not name too.
fn cache_files(dir: &Path) -> Vec<CacheFile> {
  let Ok(entries) = std::fs::read_dir(dir) else {
    return Vec::new();
  };
  entries
    .flatten()
    .filter_map(|entry| {
      let meta = entry.metadata().ok()?;
      let name = entry.file_name();
      meta.is_file().then(|| CacheFile {
        path: entry.path(),
        owner: name
          .to_str()
          .and_then(parse_slot_file)
          .map(|(key, slot)| (key.to_string(), slot)),
        len: meta.len(),
        modified: meta.modified().unwrap_or(SystemTime::UNIX_EPOCH),
      })
    })
    .collect()
}

/// `(slot, path)` of the files saved for `key`, lowest slot first.
fn key_files(dir: &Path, key: &str) -> Vec<(u32, PathBuf)> {
  let mut files: Vec<(u32, PathBuf)> = cache_files(dir)
    .into_iter()
    .filter_map(|f| match f.owner {
      Some((owner, slot)) if owner == key => Some((slot, f.path)),
      _ => None,
    })
    .collect();
  files.sort();
  files
}

fn remove_key_files(dir: &Path, key: &str) {
  for (_, path) in key_files(dir, key) {
    let _ = std::fs::remove_file(path);
  }
}

/// Delete files past `max_age`, then the oldest files until the directory is
/// within `max_bytes`. Returns `(files, bytes)` removed.
fn prune(dir: &Path, max_bytes: u64, max_age: Duration, now: SystemTime) -> (usize, u64) {
  let mut files = cache_files(dir);
  files.sort_by_key(|f| f.modified);
  let mut total: u64 = files.iter().map(|f| f.len).sum();
  let mut removed = (0usize, 0u64);
  for file in files {
    let expired = !max_age.is_zero()
      && now
        .duration_since(file.modified)
        .is_ok_and(|age| age > max_age);
    if !expired && total <= max_bytes {
      continue;
    }
    if std::fs::remove_file(&file.path).is_ok() {
      total = total.saturating_sub(file.len);
      removed = (removed.0 + 1, removed.1.saturating_add(file.len));
    }
  }
  removed
}

fn log_pruned((files, bytes): (usize, u64)) {
  if files > 0 {
    log::info!(
      "prompt cache: deleted {files} old slot file(s), {}",
      mib(bytes)
    );
  }
}

/// What a slot's KV cache will weigh on disk, from the model's own geometry.
struct KvGeometry {
  header: GgufHeader,
  arch: Option<String>,
  cache_type_k: CacheType,
  cache_type_v: CacheType,
}

impl KvGeometry {
  /// `None` when the header cannot be read; the save then goes ahead on the
  /// time budget and the size cap alone.
  fn read(params: &LaunchParams) -> Option<Self> {
    let header = crate::gguf::header::read_path(&params.model_path, HeaderReadOptions::default())
      .ok()?
      .header;
    let concept = |c| {
      parse_cache_type(
        params
          .knobs
          .str_by_concept(crate::backend::DEFAULT_BACKEND_ID, c),
      )
    };
    Some(Self {
      arch: header.string(&["general.architecture"]).map(str::to_string),
      cache_type_k: concept(Concept::KvCacheKType),
      cache_type_v: concept(Concept::KvCacheVType),
      header,
    })
  }

  /// `0` for an architecture the estimator has no geometry for.
  fn bytes(&self, tokens: u64) -> u64 {
    kv_bytes(
      &self.header,
      self.arch.as_deref(),
      EstimateOptions {
        ctx_len: tokens,
        cache_type_k: self.cache_type_k,
        cache_type_v: self.cache_type_v,
        n_gpu_layers: None,
      },
    )
  }
}

#[derive(Debug, Deserialize)]
struct SlotInfo {
  id: u32,
  #[serde(default)]
  is_processing: bool,
  /// Absent until the slot has run a task, and on builds older than
  /// 2025-09 for every slot.
  #[serde(default)]
  n_prompt_tokens: Option<u64>,
}

#[derive(Debug, Deserialize)]
struct SaveReply {
  n_saved: u64,
  n_written: u64,
}

#[derive(Debug, Deserialize)]
struct RestoreReply {
  n_restored: u64,
  n_read: u64,
}

fn client() -> Option<reqwest::Client> {
  reqwest::Client::builder()
    .no_proxy()
    .pool_max_idle_per_host(0)
    .build()
    .ok()
}

async fn list_slots(client: &reqwest::Client, port: u16) -> Result<Vec<SlotInfo>, String> {
  let resp = client
    .get(format!("http://127.0.0.1:{port}/slots"))
    .timeout(LIST_TIMEOUT)
    .send()
    .await
    .map_err(|e| e.to_string())?;
  if !resp.status().is_success() {
    return Err(format!("GET /slots answered {}", resp.status().as_u16()));
  }
  resp.json().await.map_err(|e| e.to_string())
}

async fn slot_action<T: serde::de::DeserializeOwned>(
  client: &reqwest::Client,
  port: u16,
  slot: u32,
  action: &str,
  filename: &str,
  timeout: Duration,
) -> Result<T, String> {
  let resp = client
    .post(format!(
      "http://127.0.0.1:{port}/slots/{slot}?action={action}"
    ))
    .timeout(timeout)
    .json(&serde_json::json!({ "filename": filename }))
    .send()
    .await
    .map_err(|e| e.to_string())?;
  let status = resp.status();
  if !status.is_success() {
    let body = resp.text().await.unwrap_or_default();
    let detail: String = body.chars().take(200).collect();
    return Err(format!("{} {detail}", status.as_u16()));
  }
  resp.json().await.map_err(|e| e.to_string())
}

fn model_name(params: &LaunchParams) -> String {
  params
    .model_path
    .file_name()
    .map(|n| n.to_string_lossy().into_owned())
    .unwrap_or_default()
}

fn mib(bytes: u64) -> String {
  format!("{:.1} MiB", bytes as f64 / (1024.0 * 1024.0))
}

/// Save every idle slot of the launch on `port` that holds a prompt worth
/// keeping. Best-effort: any failure is logged and the eviction goes ahead.
pub(super) async fn save(params: &LaunchParams, port: u16) {
  if let Some(settings) = Settings::from_launch(params) {
    save_with(&settings, params, port, SAVE_BUDGET).await;
  }
}

async fn save_with(settings: &Settings, params: &LaunchParams, port: u16, budget: Duration) {
  let deadline = Instant::now() + budget;
  let Some(client) = client() else {
    return;
  };
  let model = model_name(params);
  let slots = match list_slots(&client, port).await {
    Ok(slots) => slots,
    Err(e) => {
      log::debug!("prompt cache: no slot list from {model} on port {port}: {e}");
      return;
    }
  };
  let blocking_params = params.clone();
  let blocking_dir = settings.dir.clone();
  let Ok((key, geometry)) = tokio::task::spawn_blocking(move || {
    let key = launch_key(&blocking_params);
    // Whatever an earlier eviction left for this launch is older than what the
    // slots hold now.
    remove_key_files(&blocking_dir, &key);
    (key, KvGeometry::read(&blocking_params))
  })
  .await
  else {
    return;
  };
  let min_tokens = settings.min_tokens.max(1);
  for slot in slots {
    // A slot that reports its token count is sized before anything is written.
    // One that does not is saved and judged by what the save reports.
    if slot.is_processing || slot.n_prompt_tokens.is_some_and(|n| n < min_tokens) {
      continue;
    }
    let estimate = slot
      .n_prompt_tokens
      .zip(geometry.as_ref())
      .map_or(0, |(tokens, g)| g.bytes(tokens));
    if estimate > settings.max_bytes {
      log::info!(
        "prompt cache: slot {} of {model} holds {} tokens, about {}, over the {} cap; not saved",
        slot.id,
        slot.n_prompt_tokens.unwrap_or_default(),
        mib(estimate),
        mib(settings.max_bytes),
      );
      continue;
    }
    let free = crate::init::download::available_bytes(&settings.dir);
    if free.is_some_and(|free| free < estimate.saturating_add(DISK_HEADROOM_BYTES)) {
      log::warn!(
        "prompt cache: {} free under {}, slot {} of {model} needs about {}; not saved",
        mib(free.unwrap_or_default()),
        settings.dir.display(),
        slot.id,
        mib(estimate),
      );
      continue;
    }
    let remaining = deadline.saturating_duration_since(Instant::now());
    if remaining.is_zero() {
      log::warn!(
        "prompt cache: save budget used up, slot {} of {model} not saved",
        slot.id
      );
      continue;
    }
    let filename = slot_file_name(&key, slot.id);
    let path = settings.dir.join(&filename);
    let started = Instant::now();
    match slot_action::<SaveReply>(&client, port, slot.id, "save", &filename, remaining).await {
      Ok(reply) if reply.n_saved >= min_tokens => log::info!(
        "prompt cache: saved slot {} of {model}: {} tokens, {}, {} ms",
        slot.id,
        reply.n_saved,
        mib(reply.n_written),
        started.elapsed().as_millis(),
      ),
      // An empty or small slot that could not be sized up front.
      Ok(_) => {
        let _ = tokio::fs::remove_file(&path).await;
      }
      Err(e) => {
        // A timed-out save leaves a partial file behind.
        let _ = tokio::fs::remove_file(&path).await;
        log::warn!(
          "prompt cache: saving slot {} of {model} failed: {e}",
          slot.id
        );
      }
    }
  }
  prune_blocking(settings).await;
}

async fn prune_blocking(settings: &Settings) {
  let settings = settings.clone();
  let pruned = tokio::task::spawn_blocking(move || {
    prune(
      &settings.dir,
      settings.max_bytes,
      settings.max_age,
      SystemTime::now(),
    )
  })
  .await;
  if let Ok(pruned) = pruned {
    log_pruned(pruned);
  }
}

/// Read back the files an eviction saved for this launch configuration, then
/// delete them. Runs before the launch is published `Ready`, so no request can
/// take a slot first. Best-effort: a failed restore costs one reprocess.
pub(super) async fn restore(params: &LaunchParams, port: u16) {
  if let Some(settings) = Settings::from_launch(params) {
    restore_with(&settings, params, port, RESTORE_BUDGET).await;
  }
}

async fn restore_with(settings: &Settings, params: &LaunchParams, port: u16, budget: Duration) {
  let deadline = Instant::now() + budget;
  prune_blocking(settings).await;
  let blocking_params = params.clone();
  let blocking_dir = settings.dir.clone();
  let Ok((key, files)) = tokio::task::spawn_blocking(move || {
    let key = launch_key(&blocking_params);
    let files = key_files(&blocking_dir, &key);
    (key, files)
  })
  .await
  else {
    return;
  };
  if files.is_empty() {
    return;
  }
  let Some(client) = client() else {
    return;
  };
  let model = model_name(params);
  for (slot, path) in files {
    let filename = slot_file_name(&key, slot);
    let remaining = deadline.saturating_duration_since(Instant::now());
    let started = Instant::now();
    let outcome = if remaining.is_zero() {
      Err("restore budget used up".to_string())
    } else {
      slot_action::<RestoreReply>(&client, port, slot, "restore", &filename, remaining).await
    };
    match outcome {
      Ok(reply) => log::info!(
        "prompt cache: restored slot {slot} of {model}: {} tokens, {}, {} ms",
        reply.n_restored,
        mib(reply.n_read),
        started.elapsed().as_millis(),
      ),
      Err(e) => log::warn!("prompt cache: restoring slot {slot} of {model} failed: {e}"),
    }
    // Used or unusable, the file is done: the live slot moves on from here and
    // the next eviction writes a fresh one.
    let _ = tokio::fs::remove_file(&path).await;
  }
}

#[cfg(test)]
mod tests {
  use super::*;
  use std::sync::{Arc, Mutex};
  use tokio::io::{AsyncReadExt, AsyncWriteExt};

  fn chat_params() -> LaunchParams {
    LaunchParams::new(PathBuf::from("/m/model.gguf"), LaunchMode::Chat)
  }

  fn temp_dir(label: &str) -> PathBuf {
    crate::util::test_temp::unique_temp_dir(label)
  }

  fn settings(dir: &Path) -> Settings {
    Settings {
      dir: dir.to_path_buf(),
      max_bytes: 10 * GIB,
      max_age: Duration::from_secs(3600),
      min_tokens: 100,
    }
  }

  fn seeded(dir: &Path) -> LaunchParams {
    let mut params = chat_params();
    seed_into(
      &PromptCacheConfig::default(),
      Some(dir.to_path_buf()),
      &mut params,
    );
    params
  }

  #[test]
  fn config_defaults_and_yaml_round_trip() {
    let cfg = PromptCacheConfig::default();
    assert!(cfg.enabled);
    assert_eq!(
      (cfg.max_gb, cfg.max_age_hours, cfg.min_tokens),
      (32, 24, 1024)
    );
    let partial: PromptCacheConfig = yaml_serde::from_str("max_gb: 8\n").unwrap();
    assert_eq!(partial.max_gb, 8);
    assert!(partial.enabled, "an omitted key keeps its default");
    assert!(yaml_serde::from_str::<PromptCacheConfig>("max_gib: 8\n").is_err());
  }

  #[test]
  fn help_probe_keys_on_the_flag() {
    assert!(supported_in_help(
      "--slot-save-path PATH                   path to save slot kv cache (default: disabled)"
    ));
    assert!(!supported_in_help(
      "--slots, --no-slots   expose slots monitoring endpoint"
    ));
  }

  #[test]
  fn seed_sets_the_directory_and_limits_for_a_chat_launch() {
    let cache = temp_dir("pc-seed");
    let params = seeded(&cache);
    let dir = cache.join(DIR_NAME);
    assert!(dir.is_dir(), "llama-server refuses a missing directory");
    assert_eq!(
      Settings::from_launch(&params),
      Some(Settings {
        dir: dir.clone(),
        max_bytes: 32 * GIB,
        max_age: Duration::from_secs(24 * 3600),
        min_tokens: 1024,
      })
    );
    #[cfg(unix)]
    {
      use std::os::unix::fs::PermissionsExt;
      let mode = std::fs::metadata(&dir).unwrap().permissions().mode();
      assert_eq!(mode & 0o777, 0o700);
    }
    let _ = std::fs::remove_dir_all(&cache);
  }

  #[test]
  fn the_directory_is_created_only_for_a_build_that_takes_the_flag() {
    let cache = temp_dir("pc-confirm");
    let dir = cache.join(DIR_NAME);
    let mut params = chat_params();
    seed_keys(
      &PromptCacheConfig::default(),
      Some(cache.clone()),
      &mut params,
    );
    assert!(Settings::from_launch(&params).is_some());
    assert!(!dir.exists(), "seeding touches no disk");

    let mut unsupported = params.clone();
    confirm(&mut unsupported, false);
    assert_eq!(Settings::from_launch(&unsupported), None);
    assert!(!dir.exists());

    // A file where the directory should be: the launch goes ahead without
    // the flag instead of failing in llama-server.
    std::fs::write(&dir, b"in the way").unwrap();
    let mut blocked = params.clone();
    confirm(&mut blocked, true);
    assert_eq!(Settings::from_launch(&blocked), None);
    std::fs::remove_file(&dir).unwrap();

    confirm(&mut params, true);
    assert!(Settings::from_launch(&params).is_some());
    assert!(dir.is_dir());
    let _ = std::fs::remove_dir_all(&cache);
  }

  #[test]
  fn seed_clears_inherited_keys_when_the_launch_does_not_qualify() {
    let cache = temp_dir("pc-seed-off");
    let off = PromptCacheConfig {
      enabled: false,
      ..PromptCacheConfig::default()
    };
    let mut disabled = seeded(&cache);
    seed_into(&off, Some(cache.clone()), &mut disabled);
    assert_eq!(Settings::from_launch(&disabled), None);

    for mode in [LaunchMode::Embedding, LaunchMode::Rerank] {
      let mut params = seeded(&cache);
      params.mode = mode;
      seed_into(
        &PromptCacheConfig::default(),
        Some(cache.clone()),
        &mut params,
      );
      assert_eq!(Settings::from_launch(&params), None, "{mode:?}");
    }

    let mut own_path = seeded(&cache);
    own_path.extras = vec!["--slot-save-path".into(), "/elsewhere".into()];
    seed_into(
      &PromptCacheConfig::default(),
      Some(cache.clone()),
      &mut own_path,
    );
    assert_eq!(Settings::from_launch(&own_path), None);

    let mut no_cache_dir = seeded(&cache);
    seed_into(&PromptCacheConfig::default(), None, &mut no_cache_dir);
    assert_eq!(Settings::from_launch(&no_cache_dir), None);
    let _ = std::fs::remove_dir_all(&cache);
  }

  #[test]
  fn launch_key_ignores_port_and_slot_dir_but_not_the_launch() {
    let dir = temp_dir("pc-key");
    let model = dir.join("model.gguf");
    std::fs::write(&model, b"weights-v1").unwrap();
    let mut params = LaunchParams::new(model.clone(), LaunchMode::Chat);
    let bare = launch_key(&params);
    assert_eq!(bare.len(), KEY_LEN);
    assert!(parse_slot_file(&slot_file_name(&bare, 3)).is_some());

    params.port = Some(41000);
    seed_into(
      &PromptCacheConfig::default(),
      Some(dir.clone()),
      &mut params,
    );
    assert_eq!(
      launch_key(&params),
      bare,
      "port and slot dir are not identity"
    );

    let mut other_ctx = params.clone();
    other_ctx.ctx = Some(8192);
    assert_ne!(launch_key(&other_ctx), bare);

    let mut other_extras = params.clone();
    other_extras.extras = vec!["--lora".into(), "/a.gguf".into()];
    assert_ne!(launch_key(&other_extras), bare);

    std::fs::write(&model, b"weights-v2-longer").unwrap();
    assert_ne!(launch_key(&params), bare, "replaced weights must not match");
    let _ = std::fs::remove_dir_all(&dir);
  }

  #[test]
  fn slot_file_names_round_trip_and_reject_foreign_files() {
    let key = "0123456789abcdef";
    assert_eq!(slot_file_name(key, 2), "0123456789abcdef-slot2.bin");
    assert_eq!(
      parse_slot_file("0123456789abcdef-slot2.bin"),
      Some((key, 2))
    );
    for foreign in [
      "notes.txt",
      "0123456789abcdef-slot2.tmp",
      "0123456789abcdef-slotx.bin",
      "short-slot0.bin",
      "0123456789abcdeg-slot0.bin",
    ] {
      assert_eq!(parse_slot_file(foreign), None, "{foreign}");
    }
  }

  fn write_aged(dir: &Path, name: &str, len: usize, age: Duration, now: SystemTime) -> PathBuf {
    let path = dir.join(name);
    std::fs::write(&path, vec![0u8; len]).unwrap();
    std::fs::File::options()
      .write(true)
      .open(&path)
      .unwrap()
      .set_modified(now - age)
      .unwrap();
    path
  }

  #[test]
  fn prune_deletes_expired_files_then_the_oldest_past_the_cap() {
    let dir = temp_dir("pc-prune");
    let now = SystemTime::now();
    let secs = Duration::from_secs;
    let expired = write_aged(&dir, "aaaaaaaaaaaaaaaa-slot0.bin", 10, secs(7200), now);
    let oldest = write_aged(&dir, "bbbbbbbbbbbbbbbb-slot0.bin", 40, secs(300), now);
    let middle = write_aged(&dir, "cccccccccccccccc-slot0.bin", 40, secs(200), now);
    let newest = write_aged(&dir, "dddddddddddddddd-slot1.bin", 40, secs(100), now);
    // Not a name this module writes, but llama-server saves under whatever
    // filename a `/slots` caller sends, so it is capped and aged like the rest.
    let foreign = write_aged(&dir, "someone-elses.bin", 500, secs(99999), now);

    assert_eq!(prune(&dir, 80, secs(3600), now), (3, 550));
    assert!(!expired.exists() && !oldest.exists() && !foreign.exists());
    assert!(middle.exists() && newest.exists());

    assert_eq!(prune(&dir, 80, secs(3600), now), (0, 0), "within the cap");
    assert_eq!(
      prune(&dir, 0, Duration::ZERO, now),
      (2, 80),
      "a zero cap keeps nothing"
    );
    let _ = std::fs::remove_dir_all(&dir);
  }

  #[test]
  fn prune_with_no_age_limit_keeps_old_files_under_the_cap() {
    let dir = temp_dir("pc-prune-age0");
    let now = SystemTime::now();
    let old = write_aged(
      &dir,
      "aaaaaaaaaaaaaaaa-slot0.bin",
      10,
      Duration::from_secs(90 * 24 * 3600),
      now,
    );
    assert_eq!(prune(&dir, 100, Duration::ZERO, now), (0, 0));
    assert!(old.exists());
    assert_eq!(
      prune(&dir.join("missing"), 100, Duration::ZERO, now),
      (0, 0)
    );
    let _ = std::fs::remove_dir_all(&dir);
  }

  #[test]
  fn kv_geometry_estimates_from_the_model_header() {
    let dir = temp_dir("pc-geometry");
    let model = dir.join("m.gguf");
    // Llama-3.2-1B's attention shape: 16 layers, 8 KV heads of 64 dims.
    let bytes = crate::gguf::test_fixtures::FixtureBuilder::new()
      .with_arch("llama")
      .with_block_count(16)
      .with_head_count(32)
      .with_head_count_kv(8)
      .with_embedding_length(2048)
      .build();
    std::fs::write(&model, bytes).unwrap();
    let geometry = KvGeometry::read(&LaunchParams::new(model, LaunchMode::Chat)).unwrap();
    // K + V, f16: 2 * 16 * 8 * 64 * 2 bytes per token. The real file measured
    // 32 784 bytes per token (the rest is token ids and cell metadata).
    assert_eq!(geometry.bytes(1), 32_768);
    assert_eq!(geometry.bytes(100_000), 3_276_800_000);

    let missing = LaunchParams::new(dir.join("gone.gguf"), LaunchMode::Chat);
    assert!(KvGeometry::read(&missing).is_none());
    let _ = std::fs::remove_dir_all(&dir);
  }

  /// What the stub answers and what it saw.
  #[derive(Default)]
  struct Stub {
    slots: String,
    /// `(slot, action, filename)` in arrival order.
    calls: Vec<(u32, String, String)>,
    /// Slot ids whose action answers 400.
    failing: Vec<u32>,
    /// Slot ids whose save reports 50 tokens, under every test's threshold.
    small: Vec<u32>,
    dir: PathBuf,
  }

  /// A stand-in for the three llama-server routes this module calls. `save`
  /// writes a 64-byte file the way the real server writes into its
  /// `--slot-save-path`.
  async fn spawn_stub(stub: Arc<Mutex<Stub>>) -> u16 {
    let listener = tokio::net::TcpListener::bind(("127.0.0.1", 0))
      .await
      .unwrap();
    let port = listener.local_addr().unwrap().port();
    tokio::spawn(async move {
      loop {
        let Ok((mut sock, _)) = listener.accept().await else {
          return;
        };
        let stub = Arc::clone(&stub);
        tokio::spawn(async move {
          let mut buf = Vec::new();
          let mut chunk = [0u8; 4096];
          let (head, body) = loop {
            let n = sock.read(&mut chunk).await.unwrap_or(0);
            if n == 0 {
              return;
            }
            buf.extend_from_slice(&chunk[..n]);
            let text = String::from_utf8_lossy(&buf).into_owned();
            let Some((head, body)) = text.split_once("\r\n\r\n") else {
              continue;
            };
            let want = head
              .lines()
              .find_map(|l| {
                l.to_ascii_lowercase()
                  .strip_prefix("content-length:")
                  .map(str::trim)
                  .map(str::to_string)
              })
              .and_then(|v| v.parse::<usize>().ok())
              .unwrap_or(0);
            if body.len() >= want {
              break (head.to_string(), body.to_string());
            }
          };
          let target = head
            .split_whitespace()
            .nth(1)
            .unwrap_or_default()
            .to_string();
          let (status, reply) = stub_reply(&stub, &target, &body);
          let response = format!(
            "HTTP/1.1 {status}\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{reply}",
            reply.len()
          );
          let _ = sock.write_all(response.as_bytes()).await;
        });
      }
    });
    port
  }

  fn stub_reply(stub: &Mutex<Stub>, target: &str, body: &str) -> (&'static str, String) {
    let mut stub = stub.lock().unwrap();
    if target == "/slots" {
      return ("200 OK", stub.slots.clone());
    }
    let (path, query) = target.split_once('?').unwrap_or((target, ""));
    let slot: u32 = path.trim_start_matches("/slots/").parse().unwrap();
    let action = query.trim_start_matches("action=").to_string();
    let filename = serde_json::from_str::<serde_json::Value>(body).unwrap()["filename"]
      .as_str()
      .unwrap()
      .to_string();
    stub.calls.push((slot, action.clone(), filename.clone()));
    if stub.failing.contains(&slot) {
      return (
        "400 Bad Request",
        r#"{"error":{"code":400,"message":"Unable to restore slot","type":"invalid_request_error"}}"#.into(),
      );
    }
    if action == "save" {
      std::fs::write(stub.dir.join(&filename), [7u8; 64]).unwrap();
      let n_saved = if stub.small.contains(&slot) { 50 } else { 5000 };
      return (
        "200 OK",
        format!(
          r#"{{"id_slot":{slot},"filename":"{filename}","n_saved":{n_saved},"n_written":64,"timings":{{"save_ms":1.5}}}}"#
        ),
      );
    }
    (
      "200 OK",
      format!(
        r#"{{"id_slot":{slot},"filename":"{filename}","n_restored":5000,"n_read":64,"timings":{{"restore_ms":1.5}}}}"#
      ),
    )
  }

  fn stub_in(dir: &Path, slots: &str) -> Arc<Mutex<Stub>> {
    Arc::new(Mutex::new(Stub {
      slots: slots.to_string(),
      dir: dir.to_path_buf(),
      ..Stub::default()
    }))
  }

  fn names_in(dir: &Path) -> Vec<String> {
    let mut names: Vec<String> = std::fs::read_dir(dir)
      .unwrap()
      .flatten()
      .map(|e| e.file_name().to_string_lossy().into_owned())
      .collect();
    names.sort();
    names
  }

  #[tokio::test]
  async fn save_writes_one_file_per_slot_worth_keeping() {
    let dir = temp_dir("pc-save");
    let params = chat_params();
    let key = launch_key(&params);
    // An earlier eviction's file for the same launch is replaced, not kept.
    std::fs::write(dir.join(slot_file_name(&key, 9)), b"stale").unwrap();
    let stub = stub_in(
      &dir,
      r#"[
        {"id":0,"n_ctx":8192,"is_processing":false,"n_prompt_tokens":5000},
        {"id":1,"n_ctx":8192,"is_processing":false},
        {"id":2,"n_ctx":8192,"is_processing":true,"n_prompt_tokens":9000},
        {"id":3,"n_ctx":8192,"is_processing":false,"n_prompt_tokens":99},
        {"id":4,"n_ctx":8192,"is_processing":false}
      ]"#,
    );
    stub.lock().unwrap().small = vec![4];
    let port = spawn_stub(Arc::clone(&stub)).await;

    save_with(&settings(&dir), &params, port, Duration::from_secs(10)).await;

    let calls = stub.lock().unwrap().calls.clone();
    assert_eq!(
      calls,
      vec![
        (0, "save".to_string(), slot_file_name(&key, 0)),
        (1, "save".to_string(), slot_file_name(&key, 1)),
        (4, "save".to_string(), slot_file_name(&key, 4)),
      ],
      "busy and known-small slots are skipped; a slot with no count is tried"
    );
    assert_eq!(
      names_in(&dir),
      vec![slot_file_name(&key, 0), slot_file_name(&key, 1)],
      "a save under the threshold leaves no file, and the stale one is gone"
    );
    let _ = std::fs::remove_dir_all(&dir);
  }

  #[tokio::test]
  async fn save_skips_a_slot_estimated_over_the_cap() {
    let dir = temp_dir("pc-save-cap");
    let model = dir.join("m.gguf");
    let bytes = crate::gguf::test_fixtures::FixtureBuilder::new()
      .with_arch("llama")
      .with_block_count(16)
      .with_head_count(32)
      .with_head_count_kv(8)
      .with_embedding_length(2048)
      .build();
    std::fs::write(&model, bytes).unwrap();
    let params = LaunchParams::new(model, LaunchMode::Chat);
    let stub = stub_in(
      &dir,
      r#"[{"id":0,"is_processing":false,"n_prompt_tokens":5000},
          {"id":1,"is_processing":false,"n_prompt_tokens":200}]"#,
    );
    let port = spawn_stub(Arc::clone(&stub)).await;
    // 5000 tokens estimate at 164 MB, 200 tokens at 6.5 MB.
    let capped = Settings {
      max_bytes: 100 * 1024 * 1024,
      ..settings(&dir)
    };

    save_with(&capped, &params, port, Duration::from_secs(10)).await;

    let saved: Vec<u32> = stub.lock().unwrap().calls.iter().map(|c| c.0).collect();
    assert_eq!(saved, vec![1], "the over-cap slot is never written");
    let _ = std::fs::remove_dir_all(&dir);
  }

  #[tokio::test]
  async fn a_failed_or_out_of_budget_save_leaves_no_file() {
    let dir = temp_dir("pc-save-fail");
    let params = chat_params();
    let stub = stub_in(
      &dir,
      r#"[{"id":0,"is_processing":false,"n_prompt_tokens":5000}]"#,
    );
    stub.lock().unwrap().failing = vec![0];
    let port = spawn_stub(Arc::clone(&stub)).await;
    save_with(&settings(&dir), &params, port, Duration::from_secs(10)).await;
    assert_eq!(stub.lock().unwrap().calls.len(), 1);
    assert!(names_in(&dir).is_empty());

    stub.lock().unwrap().failing.clear();
    save_with(&settings(&dir), &params, port, Duration::ZERO).await;
    assert_eq!(stub.lock().unwrap().calls.len(), 1, "no budget, no request");
    assert!(names_in(&dir).is_empty());

    // Nothing listening: the slot list fails and the eviction just proceeds.
    save_with(&settings(&dir), &params, 1, Duration::from_secs(1)).await;
    let _ = std::fs::remove_dir_all(&dir);
  }

  #[tokio::test]
  async fn restore_reads_back_this_launch_only_and_deletes_what_it_used() {
    let dir = temp_dir("pc-restore");
    let params = chat_params();
    let key = launch_key(&params);
    let other = "ffffffffffffffff-slot0.bin";
    for name in [
      slot_file_name(&key, 1),
      slot_file_name(&key, 0),
      other.to_string(),
    ] {
      std::fs::write(dir.join(name), [1u8; 32]).unwrap();
    }
    let stub = stub_in(&dir, "[]");
    stub.lock().unwrap().failing = vec![1];
    let port = spawn_stub(Arc::clone(&stub)).await;

    restore_with(&settings(&dir), &params, port, Duration::from_secs(10)).await;

    let calls = stub.lock().unwrap().calls.clone();
    assert_eq!(
      calls,
      vec![
        (0, "restore".to_string(), slot_file_name(&key, 0)),
        (1, "restore".to_string(), slot_file_name(&key, 1)),
      ]
    );
    assert_eq!(
      names_in(&dir),
      vec![other.to_string()],
      "restored and unusable files are both deleted; another launch's file stays"
    );
    let _ = std::fs::remove_dir_all(&dir);
  }

  #[tokio::test]
  async fn restore_does_nothing_without_files_and_drops_expired_ones() {
    let dir = temp_dir("pc-restore-none");
    let params = chat_params();
    let stub = stub_in(&dir, "[]");
    let port = spawn_stub(Arc::clone(&stub)).await;
    restore_with(&settings(&dir), &params, port, Duration::from_secs(10)).await;
    assert!(stub.lock().unwrap().calls.is_empty());

    let name = slot_file_name(&launch_key(&params), 0);
    write_aged(
      &dir,
      &name,
      16,
      Duration::from_secs(7200),
      SystemTime::now(),
    );
    restore_with(&settings(&dir), &params, port, Duration::from_secs(10)).await;
    assert!(
      stub.lock().unwrap().calls.is_empty(),
      "a file past max_age is not restored"
    );
    assert!(names_in(&dir).is_empty());
    let _ = std::fs::remove_dir_all(&dir);
  }

  #[tokio::test]
  async fn hooks_do_nothing_for_a_launch_without_the_slot_keys() {
    let dir = temp_dir("pc-hooks-off");
    let stub = stub_in(
      &dir,
      r#"[{"id":0,"is_processing":false,"n_prompt_tokens":5000}]"#,
    );
    let port = spawn_stub(Arc::clone(&stub)).await;
    save(&chat_params(), port).await;
    restore(&chat_params(), port).await;
    assert!(stub.lock().unwrap().calls.is_empty());
    let _ = std::fs::remove_dir_all(&dir);
  }
}
