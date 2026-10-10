//! Keep a launch's prompt cache across an eviction.
//!
//! `llama-server --slot-save-path DIR` turns on `POST /slots/{id}?action=save`
//! and `action=restore`, which write one slot's KV cache to a file in `DIR` and
//! read it back. The idle sweep and make-room stop a launch mid-conversation;
//! without this the next request reprocesses the whole prompt.
//!
//! - **Save** runs before an eviction only (`save`), for every slot that
//!   holds tokens. A manual stop, a daemon shutdown and a crash save nothing.
//! - **Restore** runs once the next launch answers its readiness probe, before
//!   it is reported Ready (`restore`), so the request that started the launch
//!   already finds the cache. A restored save is deleted: the launch holds the
//!   cache in memory again and saves it again at its next eviction.
//! - A restored slot is not pinned to a conversation. llama-server picks the
//!   slot for a request by the longest common prefix against each slot's cached
//!   tokens (`--slot-prompt-similarity`), so the returning conversation lands on
//!   the restored slot and any other conversation goes elsewhere.
//!
//! Files sit in one directory per model under `<cache dir>/slots`. A save is
//! named by a fingerprint of the launch (the model file's size and mtime plus
//! the argv), so only a launch that would rebuild the same cache reads it back.
//!
//! Measured numbers: `docs/spikes/2026-10-10-slot-save-restore.md`.

use std::path::{Path, PathBuf};
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

use serde::{Deserialize, Serialize};
use tokio::io::{AsyncReadExt, AsyncWriteExt};
use tokio::net::TcpStream;

use crate::launch::mode::LaunchMode;
use crate::launch::params::LaunchParams;

/// `launch_config` key carrying the directory passed as `--slot-save-path`.
/// Absent means the launch saves and restores nothing.
pub(super) const DIR_KEY: &str = "slot_save_dir";
/// `launch_config` key carrying the oldest save a restore accepts, in seconds.
/// The restore runs in the supervisor, which has the launch params but no
/// daemon config.
const MAX_AGE_KEY: &str = "slot_save_max_age_secs";
const FLAG: &str = "--slot-save-path";
/// Hand-passed flags that switch speculative decoding on.
const SPECULATION_FLAGS: &[&str] = &["--spec-type", "--spec-draft-model", "--model-draft", "-md"];

const LIST_TIMEOUT: Duration = Duration::from_secs(5);
/// One slot's save or restore call. A 100k-token slot took under 3 s on every
/// machine measured; the bound is for a slow disk, not the normal case.
const CALL_TIMEOUT: Duration = Duration::from_secs(120);
const GIB: u64 = 1024 * 1024 * 1024;
/// Free space a save must leave on the filesystem.
const DISK_RESERVE_BYTES: u64 = 2 * GIB;
/// A slot file with no manifest is a save still being written, or one that
/// was interrupted. It is only removed once it is older than any save in
/// flight can be.
const ORPHAN_GRACE: Duration = Duration::from_secs(15 * 60);
const MANIFEST_VERSION: u32 = 1;
const FINGERPRINT_LEN: usize = 16;

/// `backend.llamacpp.slot_save` in `config.yaml`.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields, rename_all = "snake_case")]
pub struct SlotSaveConfig {
  /// Save a launch's slots before the idle sweep or make-room stops it, and
  /// restore them when it next starts. Factory `true`.
  pub enabled: bool,
  /// Where the files go. Unset means `<cache dir>/slots`.
  pub dir: Option<PathBuf>,
  /// Cap on all saved files together, in GiB. The least recently saved go
  /// first, and a save larger than the cap is not kept. `0` is no cap.
  pub max_gib: u64,
  /// A save older than this is deleted instead of restored. `0` is no limit.
  pub max_age_hours: u64,
}

impl Default for SlotSaveConfig {
  fn default() -> Self {
    Self {
      enabled: true,
      dir: None,
      max_gib: 32,
      max_age_hours: 24,
    }
  }
}

impl SlotSaveConfig {
  fn max_bytes(&self) -> Option<u64> {
    (self.max_gib > 0).then(|| self.max_gib.saturating_mul(GIB))
  }

  fn max_age(&self) -> Option<Duration> {
    (self.max_age_hours > 0).then(|| Duration::from_secs(self.max_age_hours.saturating_mul(3600)))
  }

  /// The directory holding one subdirectory per model, or `None` when neither
  /// config nor the daemon names one.
  fn root(&self, cache_dir: Option<&Path>) -> Option<PathBuf> {
    match &self.dir {
      Some(dir) => Some(crate::util::paths::expand_user_path(dir)),
      None => cache_dir.map(|c| c.join("slots")),
    }
  }
}

/// Project `cfg` onto a launch: set [`DIR_KEY`] when this launch should save
/// and restore its slots, and clear it otherwise so a value inherited from
/// `last_params` cannot outlive a config change.
///
/// Creates the model's directory, because llama-server refuses to start when
/// the `--slot-save-path` directory does not exist.
pub(super) fn seed(cfg: &SlotSaveConfig, cache_dir: Option<&Path>, params: &mut LaunchParams) {
  params.launch_config.remove(DIR_KEY);
  params.launch_config.remove(MAX_AGE_KEY);
  // Bench parity: `start` must produce the same argv as raw llama-server.
  if !cfg.enabled || crate::launch::params::bench_disable_defaults_from_env() {
    return;
  }
  // Embedding and rerank launches have no conversation to keep. A hand-passed
  // flag means the user manages slot files themselves.
  if params.mode != LaunchMode::Chat
    || crate::launch::params::extras_have_flag(&params.extras, FLAG)
    || speculates(params)
  {
    return;
  }
  let Some(root) = cfg.root(cache_dir) else {
    return;
  };
  let dir = root.join(model_dir_name(&params.model_path));
  let Some(dir_str) = dir.to_str() else {
    log::warn!(
      "slot save: {} is not UTF-8; not saving slots",
      dir.display()
    );
    return;
  };
  if let Err(e) = create_private_dir(&dir) {
    log::warn!(
      "slot save: cannot create {}: {e}; not saving slots",
      dir.display()
    );
    return;
  }
  params
    .launch_config
    .insert(DIR_KEY.to_string(), dir_str.to_string());
  params.launch_config.insert(
    MAX_AGE_KEY.to_string(),
    cfg.max_age().map_or(0, |d| d.as_secs()).to_string(),
  );
}

/// Whether the launch runs speculative decoding.
///
/// A slot file holds the target model's cache and nothing of the draft
/// context. Measured with `draft-simple`: the reply after a restore was the
/// same, but 38 of 57 drafted tokens were accepted against 42 of 42 before the
/// unload. MTP has not been measured, so these launches are left out until it
/// is.
fn speculates(params: &LaunchParams) -> bool {
  params.mtp_directive.is_some()
    || SPECULATION_FLAGS
      .iter()
      .any(|flag| crate::launch::params::extras_have_flag(&params.extras, flag))
}

/// The directory a launch saves its slots to, or `None` when it saves none:
/// [`seed`] left it out, or it speculates. The MTP directive is resolved after
/// the seed runs, so that half of the check has to happen here.
pub(super) fn dir_for(params: &LaunchParams) -> Option<PathBuf> {
  if speculates(params) {
    return None;
  }
  params.launch_config.get(DIR_KEY).map(PathBuf::from)
}

/// A saved slot holds the conversation's tokens, so the directory is the
/// owner's alone. llama-server writes the files with the process umask.
fn create_private_dir(dir: &Path) -> std::io::Result<()> {
  std::fs::create_dir_all(dir)?;
  #[cfg(unix)]
  {
    use std::os::unix::fs::PermissionsExt;
    std::fs::set_permissions(dir, std::fs::Permissions::from_mode(0o700))?;
  }
  crate::util::file_security::set_owner_only_dacl(dir);
  Ok(())
}

/// `<file stem>-<hash of the full path>`: readable in a directory listing, and
/// distinct for two files that share a name.
fn model_dir_name(model_path: &Path) -> String {
  let stem: String = crate::util::paths::model_display_name(model_path)
    .chars()
    .map(|c| {
      if c.is_ascii_alphanumeric() || matches!(c, '.' | '-' | '_') {
        c
      } else {
        '_'
      }
    })
    .take(80)
    .collect();
  let hash = blake3::hash(model_path.as_os_str().as_encoded_bytes()).to_hex();
  format!("{stem}-{}", &hash[..12])
}

/// Identifies the cache a launch builds: the model file as it is on disk, and
/// every flag the server is started with. Two launches with the same
/// fingerprint produce interchangeable slot files; anything else (another
/// quant at the same path, a different cache type or context size) gets its
/// own save and never reads this one.
fn fingerprint(params: &LaunchParams) -> String {
  let mut bare = params.clone();
  bare.launch_config.remove(DIR_KEY);
  bare.launch_config.remove(MAX_AGE_KEY);
  let mut hasher = blake3::Hasher::new();
  if let Ok(meta) = std::fs::metadata(&params.model_path) {
    hasher.update(&meta.len().to_le_bytes());
    let mtime = meta
      .modified()
      .ok()
      .and_then(|t| t.duration_since(UNIX_EPOCH).ok())
      .map_or(0, |d| d.as_nanos());
    hasher.update(&mtime.to_le_bytes());
  }
  // Port 0: the port is allocated per launch and has no effect on the cache.
  for arg in super::compose::compose(&bare, 0) {
    hasher.update(arg.as_encoded_bytes());
    hasher.update(&[0]);
  }
  hasher.finalize().to_hex()[..FINGERPRINT_LEN].to_string()
}

/// What one save wrote, kept beside the slot files as `<fingerprint>.json`.
/// Written last, so a set without a manifest is incomplete and never restored.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
struct Manifest {
  version: u32,
  model_path: PathBuf,
  /// Unix seconds.
  saved_at: u64,
  slots: Vec<SavedSlot>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
struct SavedSlot {
  id: u32,
  file: String,
  n_tokens: u64,
  bytes: u64,
}

impl Manifest {
  fn bytes(&self) -> u64 {
    self.slots.iter().map(|s| s.bytes).sum()
  }

  fn tokens(&self) -> u64 {
    self.slots.iter().map(|s| s.n_tokens).sum()
  }
}

fn manifest_name(fp: &str) -> String {
  format!("{fp}.json")
}

fn slot_file_name(fp: &str, slot: u32) -> String {
  format!("{fp}.slot{slot}.bin")
}

/// The fingerprint a file name starts with, when the name is one this module
/// writes. Everything that deletes goes through here, so a `dir` pointed at a
/// directory holding other files loses none of them.
fn owned_fingerprint(name: &str) -> Option<&str> {
  let (fp, rest) = name.split_once('.')?;
  if fp.len() != FINGERPRINT_LEN || !fp.bytes().all(|b| b.is_ascii_hexdigit()) {
    return None;
  }
  let slot_file = rest
    .strip_prefix("slot")
    .and_then(|r| r.strip_suffix(".bin"))
    .is_some_and(|n| !n.is_empty() && n.bytes().all(|b| b.is_ascii_digit()));
  (slot_file || rest == "json" || rest == "json.tmp").then_some(fp)
}

fn read_manifest(path: &Path) -> Option<Manifest> {
  let manifest: Manifest = serde_json::from_slice(&std::fs::read(path).ok()?).ok()?;
  (manifest.version == MANIFEST_VERSION).then_some(manifest)
}

fn write_manifest(dir: &Path, fp: &str, manifest: &Manifest) -> std::io::Result<()> {
  let tmp = dir.join(format!("{}.tmp", manifest_name(fp)));
  std::fs::write(&tmp, serde_json::to_vec_pretty(manifest)?)?;
  std::fs::rename(&tmp, dir.join(manifest_name(fp)))
}

/// Delete every file of the save named `fp` in `dir`. Returns the bytes freed.
fn remove_set(dir: &Path, fp: &str) -> u64 {
  let Ok(entries) = std::fs::read_dir(dir) else {
    return 0;
  };
  let mut freed = 0u64;
  for entry in entries.flatten() {
    let name = entry.file_name();
    if name.to_str().and_then(owned_fingerprint) != Some(fp) {
      continue;
    }
    let size = entry.metadata().map_or(0, |m| m.len());
    if std::fs::remove_file(entry.path()).is_ok() {
      freed = freed.saturating_add(size);
    }
  }
  freed
}

#[derive(Debug, Default, PartialEq, Eq)]
struct Pruned {
  sets: usize,
  bytes: u64,
}

/// Enforce the age limit and the size cap over every model directory in
/// `root`, and remove slot files no manifest accounts for.
///
/// The cap drops the least recently saved first. A save larger than the cap
/// goes before anything else: keeping it would cost every other save.
fn prune(
  root: &Path,
  max_bytes: Option<u64>,
  max_age: Option<Duration>,
  now: SystemTime,
) -> Pruned {
  let mut pruned = Pruned::default();
  let mut drop_set = |dir: &Path, fp: &str| {
    pruned.sets += 1;
    pruned.bytes = pruned.bytes.saturating_add(remove_set(dir, fp));
  };
  let now_secs = now.duration_since(UNIX_EPOCH).map_or(0, |d| d.as_secs());
  // (saved_at, bytes, dir, fingerprint) of every save that survives the age
  // and orphan passes.
  let mut live: Vec<(u64, u64, PathBuf, String)> = Vec::new();
  for dir in model_dirs(root) {
    for (fp, newest) in sets_in(&dir) {
      match read_manifest(&dir.join(manifest_name(&fp))) {
        Some(manifest) => {
          let age = Duration::from_secs(now_secs.saturating_sub(manifest.saved_at));
          if max_age.is_some_and(|limit| age > limit) {
            drop_set(&dir, &fp);
          } else {
            live.push((manifest.saved_at, manifest.bytes(), dir.clone(), fp));
          }
        }
        None => {
          let settled = newest
            .and_then(|t| now.duration_since(t).ok())
            .is_none_or(|age| age > ORPHAN_GRACE);
          if settled {
            drop_set(&dir, &fp);
          }
        }
      }
    }
  }
  if let Some(cap) = max_bytes {
    live.retain(|(_, bytes, dir, fp)| {
      let fits = *bytes <= cap;
      if !fits {
        drop_set(dir, fp);
      }
      fits
    });
    live.sort_by_key(|(saved_at, ..)| *saved_at);
    let mut total: u64 = live.iter().map(|(_, bytes, ..)| *bytes).sum();
    for (_, bytes, dir, fp) in &live {
      if total <= cap {
        break;
      }
      drop_set(dir, fp);
      total = total.saturating_sub(*bytes);
    }
  }
  pruned
}

fn model_dirs(root: &Path) -> Vec<PathBuf> {
  std::fs::read_dir(root)
    .map(|entries| {
      entries
        .flatten()
        .map(|e| e.path())
        .filter(|p| p.is_dir())
        .collect()
    })
    .unwrap_or_default()
}

/// The saves in one model directory: each fingerprint with the mtime of its
/// newest file.
fn sets_in(dir: &Path) -> Vec<(String, Option<SystemTime>)> {
  let mut sets: Vec<(String, Option<SystemTime>)> = Vec::new();
  let Ok(entries) = std::fs::read_dir(dir) else {
    return sets;
  };
  for entry in entries.flatten() {
    let name = entry.file_name();
    let Some(fp) = name.to_str().and_then(owned_fingerprint) else {
      continue;
    };
    let mtime = entry.metadata().ok().and_then(|m| m.modified().ok());
    match sets.iter_mut().find(|(f, _)| f == fp) {
      Some((_, newest)) => *newest = (*newest).max(mtime),
      None => sets.push((fp.to_string(), mtime)),
    }
  }
  sets
}

/// Delete every save under `root`, then the directories left empty.
fn clear(root: &Path) -> Pruned {
  let mut pruned = Pruned::default();
  for dir in model_dirs(root) {
    for (fp, _) in sets_in(&dir) {
      pruned.sets += 1;
      pruned.bytes = pruned.bytes.saturating_add(remove_set(&dir, &fp));
    }
    // Fails on a directory that still holds something, which is the point.
    let _ = std::fs::remove_dir(&dir);
  }
  let _ = std::fs::remove_dir(root);
  pruned
}

/// Boot-time cleanup, off the startup path: a config change only takes effect
/// at a daemon start, so this is where a disabled feature gives its disk back
/// and where saves that aged out while the daemon was down are dropped.
pub(super) fn prune_at_boot(cfg: &SlotSaveConfig, cache_dir: Option<&Path>) {
  let Some(root) = cfg.root(cache_dir) else {
    return;
  };
  let owned = cfg.clone();
  tokio::task::spawn_blocking(move || {
    let pruned = boot_cleanup(&owned, &root);
    if pruned.sets > 0 {
      log::info!(
        "slot save: removed {} saved prompt cache(s), {} freed",
        pruned.sets,
        crate::launch::admission::human_gib(pruned.bytes),
      );
    }
  });
}

fn boot_cleanup(cfg: &SlotSaveConfig, root: &Path) -> Pruned {
  if !root.is_dir() {
    Pruned::default()
  } else if cfg.enabled {
    prune(root, cfg.max_bytes(), cfg.max_age(), SystemTime::now())
  } else {
    clear(root)
  }
}

/// What [`save`] wrote.
#[derive(Debug, PartialEq, Eq)]
pub(super) struct Saved {
  pub slots: usize,
  pub tokens: u64,
  pub bytes: u64,
}

/// Save every slot of the launch on `port` that holds tokens. Called while the
/// launch is still serving, right before an eviction stops it.
///
/// Best-effort: any failure leaves no manifest behind, and the eviction goes
/// ahead either way.
pub(super) async fn save(cfg: &SlotSaveConfig, params: &LaunchParams, port: u16) -> Option<Saved> {
  let dir = dir_for(params)?;
  let label = crate::util::paths::model_file_label(&params.model_path);
  let slots: Vec<SlotState> = list_slots(port)
    .await?
    .into_iter()
    .filter(|s| s.tokens > 0 && !s.processing)
    .collect();
  if slots.is_empty() {
    return None;
  }
  let tokens: u64 = slots.iter().map(|s| s.tokens).sum();

  // Refuse before writing rather than after: the write is the slow part, and
  // a request may be waiting on this eviction.
  let estimate = estimate_bytes(params, tokens).await;
  if let Some(estimate) = estimate {
    if cfg.max_bytes().is_some_and(|cap| estimate > cap) {
      log::info!(
        "slot save: {label} not saved: about {} for {tokens} tokens is over the {} GiB cap",
        crate::launch::admission::human_gib(estimate),
        cfg.max_gib,
      );
      return None;
    }
    let free = crate::init::download::available_bytes(&dir).unwrap_or(u64::MAX);
    if estimate.saturating_add(DISK_RESERVE_BYTES) > free {
      log::warn!(
        "slot save: {label} not saved: about {} needed, {} free on {}",
        crate::launch::admission::human_gib(estimate),
        crate::launch::admission::human_gib(free),
        dir.display(),
      );
      return None;
    }
  }

  let fp = fingerprint(params);
  let started = Instant::now();
  // The directory can have been deleted by hand since the launch started. The
  // previous save of this launch is replaced, not kept beside the new one.
  let prepared = {
    let (dir, fp) = (dir.clone(), fp.clone());
    tokio::task::spawn_blocking(move || {
      create_private_dir(&dir)?;
      remove_set(&dir, &fp);
      Ok::<_, std::io::Error>(())
    })
    .await
  };
  if !matches!(prepared, Ok(Ok(()))) {
    log::warn!(
      "slot save: cannot write to {}; {label} not saved",
      dir.display()
    );
    return None;
  }

  let mut saved: Vec<SavedSlot> = Vec::new();
  for slot in &slots {
    let file = slot_file_name(&fp, slot.id);
    match slot_action(port, slot.id, "save", &file).await {
      Ok(reply) => {
        let path = dir.join(&file);
        #[cfg(unix)]
        {
          use std::os::unix::fs::PermissionsExt;
          let _ = std::fs::set_permissions(&path, std::fs::Permissions::from_mode(0o600));
        }
        saved.push(SavedSlot {
          id: slot.id,
          n_tokens: reply.tokens,
          bytes: std::fs::metadata(&path).map_or(reply.bytes, |m| m.len()),
          file,
        });
      }
      Err(e) => {
        log::warn!("slot save: {label} slot {} not saved: {e}", slot.id);
        // A timed-out save can leave a partial file; the server is about to
        // be stopped, so nothing completes it.
        let _ = std::fs::remove_file(dir.join(&file));
      }
    }
  }
  if saved.is_empty() {
    return None;
  }
  let manifest = Manifest {
    version: MANIFEST_VERSION,
    model_path: params.model_path.clone(),
    saved_at: SystemTime::now()
      .duration_since(UNIX_EPOCH)
      .map_or(0, |d| d.as_secs()),
    slots: saved,
  };
  if let Err(e) = write_manifest(&dir, &fp, &manifest) {
    log::warn!("slot save: {label}: cannot write the manifest: {e}");
    remove_set(&dir, &fp);
    return None;
  }
  let out = Saved {
    slots: manifest.slots.len(),
    tokens: manifest.tokens(),
    bytes: manifest.bytes(),
  };
  log::info!(
    "slot save: {label} saved {} tokens in {} slot(s), {} in {} ms{}",
    out.tokens,
    out.slots,
    crate::launch::admission::human_gib(out.bytes),
    started.elapsed().as_millis(),
    estimate.map_or(String::new(), |e| format!(
      " (estimated {})",
      crate::launch::admission::human_gib(e)
    )),
  );

  if let Some(root) = dir.parent().map(Path::to_path_buf) {
    let (max_bytes, max_age) = (cfg.max_bytes(), cfg.max_age());
    let pruned =
      tokio::task::spawn_blocking(move || prune(&root, max_bytes, max_age, SystemTime::now()))
        .await
        .unwrap_or_default();
    if pruned.sets > 0 {
      log::info!(
        "slot save: removed {} older saved prompt cache(s), {} freed",
        pruned.sets,
        crate::launch::admission::human_gib(pruned.bytes),
      );
    }
  }
  // The cap may have taken the save just written, when it alone is over it.
  dir.join(manifest_name(&fp)).is_file().then_some(out)
}

/// What [`restore`] loaded.
#[derive(Debug, PartialEq, Eq)]
pub(super) struct Restored {
  pub slots: usize,
  pub tokens: u64,
}

/// Load the save a launch with these params left behind, into the server on
/// `port`. Each slot goes back to the id it was saved from. The save is
/// deleted afterwards whether or not it loaded: a file the server rejects
/// (another build's format, a context too small for it) will not load later
/// either.
pub(super) async fn restore(params: &LaunchParams, port: u16) -> Option<Restored> {
  let dir = dir_for(params)?;
  let fp = fingerprint(params);
  let manifest = read_manifest(&dir.join(manifest_name(&fp)))?;
  let label = crate::util::paths::model_file_label(&params.model_path);
  let max_age_secs = params
    .launch_config
    .get(MAX_AGE_KEY)
    .and_then(|s| s.parse::<u64>().ok())
    .unwrap_or(0);
  let age_secs = SystemTime::now()
    .duration_since(UNIX_EPOCH)
    .map_or(0, |d| d.as_secs())
    .saturating_sub(manifest.saved_at);

  let mut out = Restored {
    slots: 0,
    tokens: 0,
  };
  if max_age_secs > 0 && age_secs > max_age_secs {
    log::info!("slot save: {label}: save is {age_secs} s old, over the limit; not restored");
  } else {
    let started = Instant::now();
    for slot in &manifest.slots {
      match slot_action(port, slot.id, "restore", &slot.file).await {
        Ok(reply) => {
          out.slots += 1;
          out.tokens = out.tokens.saturating_add(reply.tokens);
        }
        Err(e) => log::warn!("slot save: {label} slot {} not restored: {e}", slot.id),
      }
    }
    if out.slots > 0 {
      log::info!(
        "slot save: {label} restored {} tokens in {} slot(s), {} in {} ms",
        out.tokens,
        out.slots,
        crate::launch::admission::human_gib(manifest.bytes()),
        started.elapsed().as_millis(),
      );
    }
  }
  let _ = tokio::task::spawn_blocking(move || remove_set(&dir, &fp)).await;
  (out.slots > 0).then_some(out)
}

/// The size of a slot file holding `tokens` tokens: the KV cache for that many
/// tokens. `None` when the header cannot be read or carries no attention
/// geometry, in which case the save proceeds and the cap is applied to what it
/// wrote.
async fn estimate_bytes(params: &LaunchParams, tokens: u64) -> Option<u64> {
  use crate::gguf::memory::{kv_bytes, parse_cache_type, EstimateOptions};
  use crate::launch::knobs::Concept;
  let path = params.model_path.clone();
  let knobs = params.knobs.clone();
  tokio::task::spawn_blocking(move || {
    let header = crate::gguf::header::read_path(&path, Default::default())
      .ok()?
      .header;
    let arch = header.string(&["general.architecture"]).map(str::to_string);
    let id = crate::backend::DEFAULT_BACKEND_ID;
    let bytes = kv_bytes(
      &header,
      arch.as_deref(),
      EstimateOptions {
        ctx_len: tokens,
        cache_type_k: parse_cache_type(knobs.str_by_concept(id, Concept::KvCacheKType)),
        cache_type_v: parse_cache_type(knobs.str_by_concept(id, Concept::KvCacheVType)),
        n_gpu_layers: None,
      },
    );
    (bytes > 0).then_some(bytes)
  })
  .await
  .ok()
  .flatten()
}

#[derive(Debug, PartialEq, Eq)]
struct SlotState {
  id: u32,
  tokens: u64,
  processing: bool,
}

/// `GET /slots`. `None` when the endpoint is off (`--no-slots`) or the server
/// does not answer.
async fn list_slots(port: u16) -> Option<Vec<SlotState>> {
  let (status, body) = http(port, "GET", "/slots", None, LIST_TIMEOUT).await.ok()?;
  if status != 200 {
    log::debug!("slot save: GET /slots on port {port} answered {status}");
    return None;
  }
  parse_slots(&body)
}

/// A slot reports `n_prompt_tokens` only once it has run a task, so a slot
/// without the field holds nothing worth saving.
fn parse_slots(body: &[u8]) -> Option<Vec<SlotState>> {
  let slots: Vec<serde_json::Value> = serde_json::from_slice(body).ok()?;
  Some(
    slots
      .iter()
      .filter_map(|s| {
        Some(SlotState {
          id: u32::try_from(s.get("id")?.as_u64()?).ok()?,
          tokens: s
            .get("n_prompt_tokens")
            .and_then(serde_json::Value::as_u64)
            .unwrap_or(0),
          processing: s
            .get("is_processing")
            .and_then(serde_json::Value::as_bool)
            .unwrap_or(false),
        })
      })
      .collect(),
  )
}

#[derive(Debug, PartialEq, Eq)]
struct ActionReply {
  tokens: u64,
  bytes: u64,
}

/// `POST /slots/{id}?action=save|restore`. `file` is a bare name: the server
/// resolves it inside its `--slot-save-path` and refuses path separators.
async fn slot_action(
  port: u16,
  slot: u32,
  action: &str,
  file: &str,
) -> Result<ActionReply, String> {
  let body = serde_json::json!({ "filename": file }).to_string();
  let path = format!("/slots/{slot}?action={action}");
  let (status, reply) = http(port, "POST", &path, Some(&body), CALL_TIMEOUT)
    .await
    .map_err(|e| e.to_string())?;
  parse_action_reply(status, &reply)
}

fn parse_action_reply(status: u16, body: &[u8]) -> Result<ActionReply, String> {
  let json: serde_json::Value = serde_json::from_slice(body).unwrap_or_default();
  if status != 200 {
    let message = json
      .pointer("/error/message")
      .and_then(serde_json::Value::as_str)
      .unwrap_or("no error message");
    return Err(format!("HTTP {status}: {message}"));
  }
  let field = |save: &str, restore: &str| {
    json
      .get(save)
      .or_else(|| json.get(restore))
      .and_then(serde_json::Value::as_u64)
  };
  match (field("n_saved", "n_restored"), field("n_written", "n_read")) {
    // A save of an empty slot answers 200 with zero tokens.
    (Some(tokens), Some(bytes)) if tokens > 0 => Ok(ActionReply { tokens, bytes }),
    (Some(_), _) => Err("the slot holds no tokens".to_string()),
    _ => Err("unexpected reply".to_string()),
  }
}

/// One HTTP/1.1 exchange over raw TCP with `Connection: close`, the same
/// no-dependency transport as the `/props` fetch beside this module.
async fn http(
  port: u16,
  method: &str,
  path: &str,
  body: Option<&str>,
  timeout: Duration,
) -> std::io::Result<(u16, Vec<u8>)> {
  let payload = body.unwrap_or("");
  let request = format!(
    "{method} {path} HTTP/1.1\r\nHost: 127.0.0.1:{port}\r\nConnection: close\r\n\
     Accept: application/json\r\nContent-Type: application/json\r\n\
     Content-Length: {}\r\n\r\n{payload}",
    payload.len()
  );
  let exchange = async {
    let mut sock = TcpStream::connect(("127.0.0.1", port)).await?;
    sock.write_all(request.as_bytes()).await?;
    let mut buf = Vec::with_capacity(4096);
    let mut chunk = [0u8; 8192];
    loop {
      let n = sock.read(&mut chunk).await?;
      if n == 0 {
        break;
      }
      buf.extend_from_slice(&chunk[..n]);
      // `/slots` grows with `--parallel`; a few KiB per slot.
      if buf.len() > 4 * 1024 * 1024 {
        break;
      }
    }
    Ok::<_, std::io::Error>(buf)
  };
  let raw = tokio::time::timeout(timeout, exchange)
    .await
    .map_err(|_| std::io::Error::new(std::io::ErrorKind::TimedOut, "timed out"))??;
  split_response(&raw)
    .map(|(status, reply)| (status, reply.to_vec()))
    .ok_or_else(|| std::io::Error::new(std::io::ErrorKind::InvalidData, "malformed HTTP response"))
}

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
  use tokio::io::AsyncBufReadExt;
  use tokio::net::TcpListener;

  fn temp(label: &str) -> PathBuf {
    crate::util::test_temp::unique_temp_dir(&format!("slot-save-{label}"))
  }

  fn chat_params(model: &Path) -> LaunchParams {
    LaunchParams::new(model.to_path_buf(), LaunchMode::Chat)
  }

  /// Params seeded the way a real launch is, against `cache`.
  fn seeded(cache: &Path, model: &Path) -> LaunchParams {
    let mut p = chat_params(model);
    seed(&SlotSaveConfig::default(), Some(cache), &mut p);
    p
  }

  fn slot_dir(p: &LaunchParams) -> PathBuf {
    PathBuf::from(p.launch_config.get(DIR_KEY).expect("seeded"))
  }

  fn secs(t: SystemTime) -> u64 {
    t.duration_since(UNIX_EPOCH).unwrap().as_secs()
  }

  /// Write a complete save: one slot file of `bytes` plus its manifest.
  fn write_set(dir: &Path, fp: &str, saved_at: u64, bytes: usize) {
    std::fs::create_dir_all(dir).unwrap();
    let file = slot_file_name(fp, 0);
    std::fs::write(dir.join(&file), vec![0u8; bytes]).unwrap();
    let manifest = Manifest {
      version: MANIFEST_VERSION,
      model_path: PathBuf::from("/m/x.gguf"),
      saved_at,
      slots: vec![SavedSlot {
        id: 0,
        file,
        n_tokens: 10,
        bytes: bytes as u64,
      }],
    };
    write_manifest(dir, fp, &manifest).unwrap();
  }

  fn names(dir: &Path) -> Vec<String> {
    let mut out: Vec<String> = std::fs::read_dir(dir)
      .map(|d| {
        d.flatten()
          .map(|e| e.file_name().to_string_lossy().into_owned())
          .collect()
      })
      .unwrap_or_default();
    out.sort();
    out
  }

  const FP_A: &str = "aaaaaaaaaaaaaaaa";
  const FP_B: &str = "bbbbbbbbbbbbbbbb";
  const FP_C: &str = "cccccccccccccccc";

  // ---- config ----

  #[test]
  fn config_defaults_are_on_with_a_cap_and_an_age_limit() {
    let cfg = SlotSaveConfig::default();
    assert!(cfg.enabled);
    assert_eq!(cfg.max_bytes(), Some(32 * GIB));
    assert_eq!(cfg.max_age(), Some(Duration::from_secs(24 * 3600)));
    assert_eq!(
      cfg.root(Some(Path::new("/c"))),
      Some(PathBuf::from("/c/slots"))
    );
    assert_eq!(
      cfg.root(None),
      None,
      "no cache dir and no `dir`: nowhere to save"
    );
  }

  #[test]
  fn zero_means_no_cap_and_no_age_limit() {
    let cfg = SlotSaveConfig {
      max_gib: 0,
      max_age_hours: 0,
      ..Default::default()
    };
    assert_eq!(cfg.max_bytes(), None);
    assert_eq!(cfg.max_age(), None);
  }

  #[test]
  fn a_configured_dir_wins_over_the_cache_dir() {
    let cfg = SlotSaveConfig {
      dir: Some(PathBuf::from("/data/slots")),
      ..Default::default()
    };
    assert_eq!(
      cfg.root(Some(Path::new("/c"))),
      Some(PathBuf::from("/data/slots"))
    );
  }

  #[test]
  fn config_parses_from_yaml_and_refuses_unknown_keys() {
    let cfg: SlotSaveConfig =
      yaml_serde::from_str("enabled: false\nmax_gib: 8\nmax_age_hours: 2\ndir: /x").unwrap();
    assert_eq!(
      cfg,
      SlotSaveConfig {
        enabled: false,
        dir: Some(PathBuf::from("/x")),
        max_gib: 8,
        max_age_hours: 2,
      }
    );
    assert!(yaml_serde::from_str::<SlotSaveConfig>("max_gb: 8").is_err());
  }

  // ---- seed ----

  #[test]
  fn seed_sets_the_dir_and_creates_it_owner_only() {
    let cache = temp("seed");
    let p = seeded(&cache, Path::new("/models/Qwen 3.gguf"));
    let dir = slot_dir(&p);
    assert!(dir.starts_with(cache.join("slots")));
    assert!(
      dir.is_dir(),
      "llama-server refuses a missing --slot-save-path"
    );
    let name = dir.file_name().unwrap().to_str().unwrap();
    assert!(name.starts_with("Qwen_3-"), "{name}");
    #[cfg(unix)]
    {
      use std::os::unix::fs::PermissionsExt;
      let mode = std::fs::metadata(&dir).unwrap().permissions().mode() & 0o777;
      assert_eq!(mode, 0o700, "a slot file holds the conversation");
    }
    assert_eq!(
      p.launch_config.get(MAX_AGE_KEY).map(String::as_str),
      Some("86400")
    );
    std::fs::remove_dir_all(&cache).ok();
  }

  #[test]
  fn seed_clears_an_inherited_dir_when_the_launch_should_not_save() {
    let cache = temp("seed-off");
    let model = Path::new("/models/a.gguf");
    let stale = |p: &mut LaunchParams| {
      p.launch_config
        .insert(DIR_KEY.to_string(), "/stale".to_string());
    };

    let mut off = chat_params(model);
    stale(&mut off);
    let disabled = SlotSaveConfig {
      enabled: false,
      ..Default::default()
    };
    seed(&disabled, Some(&cache), &mut off);
    assert!(!off.launch_config.contains_key(DIR_KEY), "disabled");

    for mode in [LaunchMode::Embedding, LaunchMode::Rerank] {
      let mut p = LaunchParams::new(model.to_path_buf(), mode);
      stale(&mut p);
      seed(&SlotSaveConfig::default(), Some(&cache), &mut p);
      assert!(!p.launch_config.contains_key(DIR_KEY), "{mode:?}");
    }

    let mut manual = chat_params(model);
    manual.extras = vec!["--slot-save-path".into(), "/mine".into()];
    seed(&SlotSaveConfig::default(), Some(&cache), &mut manual);
    assert!(
      !manual.launch_config.contains_key(DIR_KEY),
      "a hand-passed flag is the user's to manage"
    );

    for flags in [
      &["--spec-type", "draft-simple"][..],
      &["-md", "/draft.gguf"][..],
      &["--model-draft=/draft.gguf"][..],
    ] {
      let mut spec = chat_params(model);
      spec.extras = flags.iter().map(std::ffi::OsString::from).collect();
      seed(&SlotSaveConfig::default(), Some(&cache), &mut spec);
      assert!(!spec.launch_config.contains_key(DIR_KEY), "{flags:?}");
    }

    let mut nowhere = chat_params(model);
    seed(&SlotSaveConfig::default(), None, &mut nowhere);
    assert!(!nowhere.launch_config.contains_key(DIR_KEY), "no cache dir");
    assert!(
      !cache.join("slots").exists(),
      "nothing created for any of them"
    );
    std::fs::remove_dir_all(&cache).ok();
  }

  #[test]
  fn model_dir_names_differ_for_two_files_with_one_name() {
    let a = model_dir_name(Path::new("/a/model.gguf"));
    let b = model_dir_name(Path::new("/b/model.gguf"));
    assert_ne!(a, b);
    assert!(a.starts_with("model-") && b.starts_with("model-"));
    assert_eq!(a, model_dir_name(Path::new("/a/model.gguf")), "stable");
  }

  /// The MTP directive is resolved after the seed, so a seeded launch can
  /// still turn out to speculate. It then gets no flag, no save and no restore.
  #[tokio::test]
  async fn an_mtp_launch_is_left_out_even_when_seeded() {
    let cache = temp("mtp");
    let model = cache.join("m.gguf");
    std::fs::write(&model, b"x").unwrap();
    let mut p = seeded(&cache, &model);
    let dir = slot_dir(&p);
    assert_eq!(dir_for(&p), Some(dir.clone()));
    p.mtp_directive = Some(crate::launch::params::MtpDirective { draft_model: None });
    assert_eq!(dir_for(&p), None);
    let argv = super::super::compose::compose(&p, 41100);
    assert!(!argv.iter().any(|a| a == FLAG));

    let stub = one_busy_slot_stub();
    let port = stub_server(dir.clone(), stub.clone()).await;
    assert_eq!(save(&SlotSaveConfig::default(), &p, port).await, None);
    assert_eq!(restore(&p, port).await, None);
    assert!(stub.lock().unwrap().calls.is_empty());
    std::fs::remove_dir_all(&cache).ok();
  }

  // ---- compose ----

  #[test]
  fn the_flag_is_emitted_only_for_a_seeded_launch() {
    let cache = temp("argv");
    let model = Path::new("/models/a.gguf");
    let argv = |p: &LaunchParams| -> Vec<String> {
      super::super::compose::compose(p, 41100)
        .iter()
        .map(|a| a.to_string_lossy().into_owned())
        .collect()
    };
    assert!(!argv(&chat_params(model)).contains(&FLAG.to_string()));
    let p = seeded(&cache, model);
    let args = argv(&p);
    let at = args.iter().position(|a| a == FLAG).expect("flag emitted");
    assert_eq!(args[at + 1], slot_dir(&p).to_str().unwrap());
    std::fs::remove_dir_all(&cache).ok();
  }

  // ---- fingerprint ----

  #[test]
  fn fingerprint_ignores_the_slot_dir_and_follows_the_argv_and_the_file() {
    let cache = temp("fp");
    let model = cache.join("m.gguf");
    std::fs::write(&model, b"one").unwrap();
    let base = chat_params(&model);
    let fp = fingerprint(&base);
    assert_eq!(fp.len(), FINGERPRINT_LEN);
    assert_eq!(owned_fingerprint(&manifest_name(&fp)), Some(fp.as_str()));
    assert_eq!(
      fp,
      fingerprint(&seeded(&cache, &model)),
      "where the files live is not part of what the cache is"
    );

    let mut other_ctx = base.clone();
    other_ctx.ctx = Some(8192);
    assert_ne!(fp, fingerprint(&other_ctx), "a different argv");

    let mut other_extras = base.clone();
    other_extras.extras = vec!["--lora".into(), "/l.gguf".into()];
    assert_ne!(fp, fingerprint(&other_extras), "extras are argv too");

    std::fs::write(&model, b"another quant").unwrap();
    assert_ne!(fp, fingerprint(&base), "the file at that path changed");
    std::fs::remove_dir_all(&cache).ok();
  }

  // ---- file names ----

  #[test]
  fn only_names_this_module_writes_are_owned() {
    assert_eq!(owned_fingerprint("aaaaaaaaaaaaaaaa.json"), Some(FP_A));
    assert_eq!(owned_fingerprint("aaaaaaaaaaaaaaaa.json.tmp"), Some(FP_A));
    assert_eq!(owned_fingerprint("aaaaaaaaaaaaaaaa.slot12.bin"), Some(FP_A));
    for foreign in [
      "notes.json",
      "aaaaaaaaaaaaaaaa.txt",
      "aaaaaaaaaaaaaaaa.slot.bin",
      "aaaaaaaaaaaaaaaa.slotx.bin",
      "aaaaaaaaaaaaaaa.json",
      "zzzzzzzzzzzzzzzz.json",
      "aaaaaaaaaaaaaaaa",
    ] {
      assert_eq!(owned_fingerprint(foreign), None, "{foreign}");
    }
  }

  #[test]
  fn remove_set_takes_one_save_and_leaves_the_rest() {
    let dir = temp("remove");
    write_set(&dir, FP_A, 1, 100);
    write_set(&dir, FP_B, 1, 50);
    std::fs::write(dir.join("keep.txt"), b"x").unwrap();
    assert!(remove_set(&dir, FP_A) >= 100);
    assert_eq!(
      names(&dir),
      vec![
        manifest_name(FP_B),
        slot_file_name(FP_B, 0),
        "keep.txt".to_string()
      ]
    );
    std::fs::remove_dir_all(&dir).ok();
  }

  #[test]
  fn a_manifest_of_another_version_is_not_read() {
    let dir = temp("version");
    write_set(&dir, FP_A, 1, 1);
    let path = dir.join(manifest_name(FP_A));
    assert!(read_manifest(&path).is_some());
    let bumped = std::fs::read_to_string(&path)
      .unwrap()
      .replace("\"version\": 1", "\"version\": 2");
    std::fs::write(&path, bumped).unwrap();
    assert!(read_manifest(&path).is_none());
    std::fs::remove_dir_all(&dir).ok();
  }

  // ---- prune ----

  #[test]
  fn prune_drops_saves_past_the_age_limit() {
    let root = temp("age");
    let now = SystemTime::now();
    write_set(&root.join("m1"), FP_A, secs(now) - 7200, 10);
    write_set(&root.join("m2"), FP_B, secs(now) - 60, 10);
    let pruned = prune(&root, None, Some(Duration::from_secs(3600)), now);
    assert_eq!(pruned.sets, 1);
    assert!(names(&root.join("m1")).is_empty());
    assert_eq!(names(&root.join("m2")).len(), 2);
    assert_eq!(prune(&root, None, None, now), Pruned::default(), "no limit");
    std::fs::remove_dir_all(&root).ok();
  }

  #[test]
  fn prune_drops_the_least_recently_saved_until_under_the_cap() {
    let root = temp("cap");
    let now = SystemTime::now();
    write_set(&root.join("m1"), FP_A, secs(now) - 300, 400);
    write_set(&root.join("m1"), FP_B, secs(now) - 200, 400);
    write_set(&root.join("m2"), FP_C, secs(now) - 100, 400);
    let pruned = prune(&root, Some(900), None, now);
    assert_eq!(pruned.sets, 1, "one save over: only the oldest goes");
    assert_eq!(
      names(&root.join("m1")),
      vec![manifest_name(FP_B), slot_file_name(FP_B, 0)]
    );
    assert_eq!(names(&root.join("m2")).len(), 2);
    std::fs::remove_dir_all(&root).ok();
  }

  #[test]
  fn a_save_larger_than_the_cap_goes_before_the_others() {
    let root = temp("oversize");
    let now = SystemTime::now();
    write_set(&root.join("m1"), FP_A, secs(now) - 300, 100);
    write_set(&root.join("m2"), FP_B, secs(now), 5000);
    let pruned = prune(&root, Some(1000), None, now);
    assert_eq!(pruned.sets, 1);
    assert_eq!(
      names(&root.join("m1")).len(),
      2,
      "the older, smaller save stays"
    );
    assert!(names(&root.join("m2")).is_empty());
    std::fs::remove_dir_all(&root).ok();
  }

  #[test]
  fn a_slot_file_without_a_manifest_is_removed_only_once_it_is_old() {
    let root = temp("orphan");
    let dir = root.join("m1");
    std::fs::create_dir_all(&dir).unwrap();
    std::fs::write(dir.join(slot_file_name(FP_A, 0)), b"partial").unwrap();
    std::fs::write(dir.join("unrelated.bin"), b"x").unwrap();
    let now = SystemTime::now();
    assert_eq!(
      prune(&root, None, None, now).sets,
      0,
      "it may be a save still being written"
    );
    let later = now + ORPHAN_GRACE + Duration::from_secs(60);
    assert_eq!(prune(&root, None, None, later).sets, 1);
    assert_eq!(names(&dir), vec!["unrelated.bin".to_string()]);
    std::fs::remove_dir_all(&root).ok();
  }

  #[test]
  fn clear_removes_every_save_and_keeps_foreign_files() {
    let root = temp("clear");
    write_set(&root.join("m1"), FP_A, secs(SystemTime::now()), 10);
    write_set(&root.join("m2"), FP_B, secs(SystemTime::now()), 10);
    std::fs::write(root.join("m2").join("mine.txt"), b"x").unwrap();
    assert_eq!(clear(&root).sets, 2);
    assert!(!root.join("m1").exists(), "an emptied model dir is removed");
    assert_eq!(names(&root.join("m2")), vec!["mine.txt".to_string()]);
    std::fs::remove_dir_all(&root).ok();

    let empty = temp("clear-empty");
    write_set(&empty.join("m1"), FP_A, 1, 10);
    clear(&empty);
    assert!(!empty.exists(), "nothing left: the root goes too");
  }

  #[test]
  fn boot_cleanup_prunes_when_on_and_clears_when_off() {
    let root = temp("boot");
    let now = secs(SystemTime::now());
    write_set(&root.join("m1"), FP_A, now - 3 * 86400, 10);
    write_set(&root.join("m2"), FP_B, now - 60, 10);
    assert_eq!(boot_cleanup(&SlotSaveConfig::default(), &root).sets, 1);
    assert!(names(&root.join("m1")).is_empty(), "past the 24 h default");
    assert_eq!(names(&root.join("m2")).len(), 2);

    let off = SlotSaveConfig {
      enabled: false,
      ..Default::default()
    };
    assert_eq!(boot_cleanup(&off, &root).sets, 1);
    assert!(!root.exists(), "switched off: the disk comes back");
    assert_eq!(
      boot_cleanup(&off, &root),
      Pruned::default(),
      "nothing there"
    );
  }

  // ---- wire parsing ----

  #[test]
  fn split_response_reads_the_status_and_the_body() {
    let raw = b"HTTP/1.1 400 Bad Request\r\nContent-Length: 2\r\n\r\n{}";
    assert_eq!(split_response(raw), Some((400, &b"{}"[..])));
    assert_eq!(split_response(b"garbage"), None);
  }

  #[test]
  fn parse_slots_counts_only_slots_that_ran_a_task() {
    // Shapes from llama-server build 11525: an untouched slot has no
    // `n_prompt_tokens`.
    let body = br#"[
      {"id":0,"n_ctx":16384,"speculative":false,"is_processing":false},
      {"id":3,"n_ctx":16384,"speculative":false,"is_processing":false,"id_task":0,"n_prompt_tokens":3571},
      {"id":2,"n_ctx":16384,"is_processing":true,"n_prompt_tokens":12}
    ]"#;
    assert_eq!(
      parse_slots(body),
      Some(vec![
        SlotState {
          id: 0,
          tokens: 0,
          processing: false
        },
        SlotState {
          id: 3,
          tokens: 3571,
          processing: false
        },
        SlotState {
          id: 2,
          tokens: 12,
          processing: true
        },
      ])
    );
    assert_eq!(parse_slots(b"{\"error\":{}}"), None);
  }

  #[test]
  fn parse_action_reply_reads_save_and_restore_and_refuses_the_rest() {
    let save = br#"{"id_slot":3,"filename":"f","n_saved":3571,"n_written":117072100,"timings":{"save_ms":32.0}}"#;
    assert_eq!(
      parse_action_reply(200, save),
      Ok(ActionReply {
        tokens: 3571,
        bytes: 117072100
      })
    );
    let restore = br#"{"id_slot":3,"filename":"f","n_restored":3571,"n_read":117072100,"timings":{"restore_ms":21.8}}"#;
    assert_eq!(
      parse_action_reply(200, restore),
      Ok(ActionReply {
        tokens: 3571,
        bytes: 117072100
      })
    );
    let empty = br#"{"id_slot":0,"filename":"f","n_saved":0,"n_written":36}"#;
    assert!(parse_action_reply(200, empty).is_err(), "an empty slot");
    let refused = br#"{"error":{"code":400,"message":"Unable to restore slot: No available space in KV cache or invalid slot save file","type":"invalid_request_error"}}"#;
    let err = parse_action_reply(400, refused).unwrap_err();
    assert!(
      err.contains("HTTP 400") && err.contains("invalid slot save file"),
      "{err}"
    );
    assert!(parse_action_reply(200, b"{}").is_err());
  }

  // ---- save / restore against a stub server ----

  /// What the stub answers `GET /slots` with, and what it saw.
  #[derive(Default)]
  struct Stub {
    slots_status: u16,
    slots_body: String,
    /// Bytes a save writes per token.
    bytes_per_token: usize,
    /// `Some(status)` makes every restore answer that status.
    restore_status: Option<u16>,
    calls: Vec<String>,
  }

  /// A minimal stand-in for llama-server's slot endpoints: `save` writes a file
  /// into `dir` the way the real server does, `restore` reads it back.
  async fn stub_server(dir: PathBuf, stub: Arc<Mutex<Stub>>) -> u16 {
    let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let port = listener.local_addr().unwrap().port();
    tokio::spawn(async move {
      loop {
        let Ok((mut sock, _)) = listener.accept().await else {
          return;
        };
        let (rd, mut wr) = sock.split();
        let mut br = tokio::io::BufReader::new(rd);
        let mut line = String::new();
        br.read_line(&mut line).await.unwrap();
        let mut parts = line.split_whitespace();
        let method = parts.next().unwrap_or("").to_string();
        let path = parts.next().unwrap_or("").to_string();
        let mut length = 0usize;
        loop {
          let mut header = String::new();
          br.read_line(&mut header).await.unwrap();
          if header == "\r\n" || header.is_empty() {
            break;
          }
          if let Some(v) = header.to_ascii_lowercase().strip_prefix("content-length:") {
            length = v.trim().parse().unwrap();
          }
        }
        let mut body = vec![0u8; length];
        br.read_exact(&mut body).await.unwrap();
        let (status, reply) = {
          let mut s = stub.lock().unwrap();
          s.calls.push(format!("{method} {path}"));
          let file = serde_json::from_slice::<serde_json::Value>(&body)
            .ok()
            .and_then(|v| v.get("filename")?.as_str().map(str::to_string));
          if path == "/slots" {
            (s.slots_status, s.slots_body.clone())
          } else if path.ends_with("action=save") {
            let tokens = 1000usize;
            let bytes = tokens * s.bytes_per_token;
            std::fs::write(dir.join(file.unwrap()), vec![7u8; bytes]).unwrap();
            (
              200,
              format!("{{\"n_saved\":{tokens},\"n_written\":{bytes}}}"),
            )
          } else if let Some(status) = s.restore_status {
            (status, "{\"error\":{\"message\":\"refused\"}}".to_string())
          } else {
            let bytes = std::fs::metadata(dir.join(file.unwrap())).unwrap().len();
            (200, format!("{{\"n_restored\":1000,\"n_read\":{bytes}}}"))
          }
        };
        let response = format!(
          "HTTP/1.1 {status} X\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{reply}",
          reply.len()
        );
        wr.write_all(response.as_bytes()).await.unwrap();
        let _ = wr.shutdown().await;
      }
    });
    port
  }

  fn one_busy_slot_stub() -> Arc<Mutex<Stub>> {
    Arc::new(Mutex::new(Stub {
      slots_status: 200,
      slots_body: r#"[{"id":0,"is_processing":false},
        {"id":1,"is_processing":false,"n_prompt_tokens":1000},
        {"id":2,"is_processing":true,"n_prompt_tokens":50}]"#
        .to_string(),
      bytes_per_token: 4,
      ..Default::default()
    }))
  }

  #[tokio::test]
  async fn save_writes_the_idle_slots_that_hold_tokens_then_restore_consumes_them() {
    let cache = temp("roundtrip");
    let model = cache.join("m.gguf");
    std::fs::write(&model, b"not a gguf").unwrap();
    let p = seeded(&cache, &model);
    let dir = slot_dir(&p);
    let stub = one_busy_slot_stub();
    let port = stub_server(dir.clone(), stub.clone()).await;

    let saved = save(&SlotSaveConfig::default(), &p, port).await;
    assert_eq!(
      saved,
      Some(Saved {
        slots: 1,
        tokens: 1000,
        bytes: 4000
      })
    );
    let fp = fingerprint(&p);
    assert_eq!(
      names(&dir),
      vec![manifest_name(&fp), slot_file_name(&fp, 1)],
      "the empty slot and the busy slot are not saved"
    );
    #[cfg(unix)]
    {
      use std::os::unix::fs::PermissionsExt;
      let mode = std::fs::metadata(dir.join(slot_file_name(&fp, 1)))
        .unwrap()
        .permissions()
        .mode();
      assert_eq!(mode & 0o777, 0o600);
    }

    let restored = restore(&p, port).await;
    assert_eq!(
      restored,
      Some(Restored {
        slots: 1,
        tokens: 1000
      })
    );
    assert!(names(&dir).is_empty(), "a restored save is deleted");
    assert_eq!(
      stub.lock().unwrap().calls,
      vec![
        "GET /slots".to_string(),
        "POST /slots/1?action=save".to_string(),
        "POST /slots/1?action=restore".to_string(),
      ],
      "the slot goes back to the id it was saved from"
    );
    assert_eq!(restore(&p, port).await, None, "nothing left to restore");
    std::fs::remove_dir_all(&cache).ok();
  }

  #[tokio::test]
  async fn a_launch_with_another_argv_does_not_read_the_save() {
    let cache = temp("mismatch");
    let model = cache.join("m.gguf");
    std::fs::write(&model, b"x").unwrap();
    let p = seeded(&cache, &model);
    let dir = slot_dir(&p);
    let stub = one_busy_slot_stub();
    let port = stub_server(dir.clone(), stub.clone()).await;
    save(&SlotSaveConfig::default(), &p, port)
      .await
      .expect("saved");

    let mut other = p.clone();
    other.ctx = Some(4096);
    assert_eq!(restore(&other, port).await, None);
    assert_eq!(
      names(&dir).len(),
      2,
      "the save stays for the launch it fits"
    );
    assert!(!stub
      .lock()
      .unwrap()
      .calls
      .iter()
      .any(|c| c.contains("restore")));
    std::fs::remove_dir_all(&cache).ok();
  }

  #[tokio::test]
  async fn a_save_past_the_age_limit_is_deleted_not_restored() {
    let cache = temp("stale");
    let model = cache.join("m.gguf");
    std::fs::write(&model, b"x").unwrap();
    let p = seeded(&cache, &model);
    let dir = slot_dir(&p);
    let stub = one_busy_slot_stub();
    let port = stub_server(dir.clone(), stub.clone()).await;
    let old = secs(SystemTime::now()) - 2 * 86400;
    write_set(&dir, &fingerprint(&p), old, 10);

    assert_eq!(restore(&p, port).await, None);
    assert!(names(&dir).is_empty());
    assert!(
      stub.lock().unwrap().calls.is_empty(),
      "the server is not asked"
    );
    std::fs::remove_dir_all(&cache).ok();
  }

  #[tokio::test]
  async fn a_save_the_server_refuses_is_deleted() {
    let cache = temp("refused");
    let model = cache.join("m.gguf");
    std::fs::write(&model, b"x").unwrap();
    let p = seeded(&cache, &model);
    let dir = slot_dir(&p);
    let stub = one_busy_slot_stub();
    stub.lock().unwrap().restore_status = Some(400);
    let port = stub_server(dir.clone(), stub.clone()).await;
    write_set(&dir, &fingerprint(&p), secs(SystemTime::now()), 10);

    assert_eq!(restore(&p, port).await, None);
    assert!(names(&dir).is_empty(), "it would not load next time either");
    std::fs::remove_dir_all(&cache).ok();
  }

  #[tokio::test]
  async fn nothing_is_saved_without_a_seeded_dir_or_a_slots_endpoint() {
    let cache = temp("nosave");
    let model = cache.join("m.gguf");
    std::fs::write(&model, b"x").unwrap();
    let p = seeded(&cache, &model);
    let dir = slot_dir(&p);
    let stub = one_busy_slot_stub();
    let port = stub_server(dir.clone(), stub.clone()).await;
    let cfg = SlotSaveConfig::default();

    assert_eq!(save(&cfg, &chat_params(&model), port).await, None);
    assert!(
      stub.lock().unwrap().calls.is_empty(),
      "unseeded: no call at all"
    );

    // `--no-slots` answers 501.
    stub.lock().unwrap().slots_status = 501;
    assert_eq!(save(&cfg, &p, port).await, None);

    // Every slot empty.
    {
      let mut s = stub.lock().unwrap();
      s.slots_status = 200;
      s.slots_body = r#"[{"id":0,"is_processing":false}]"#.to_string();
    }
    assert_eq!(save(&cfg, &p, port).await, None);
    assert!(names(&dir).is_empty());
    assert!(!stub
      .lock()
      .unwrap()
      .calls
      .iter()
      .any(|c| c.contains("action=save")));
    std::fs::remove_dir_all(&cache).ok();
  }

  #[tokio::test]
  async fn a_new_save_replaces_the_previous_one_of_the_same_launch() {
    let cache = temp("replace");
    let model = cache.join("m.gguf");
    std::fs::write(&model, b"x").unwrap();
    let p = seeded(&cache, &model);
    let dir = slot_dir(&p);
    let fp = fingerprint(&p);
    let stub = one_busy_slot_stub();
    let port = stub_server(dir.clone(), stub.clone()).await;

    // The previous save held a slot the server no longer has tokens in.
    write_set(&dir, &fp, secs(SystemTime::now()) - 60, 10);
    std::fs::write(dir.join(slot_file_name(&fp, 9)), b"old").unwrap();
    // Another model's save is left alone.
    let other = dir.parent().unwrap().join("other-model");
    write_set(&other, FP_A, secs(SystemTime::now()) - 600, 10);

    save(&SlotSaveConfig::default(), &p, port)
      .await
      .expect("saved");
    assert_eq!(
      names(&dir),
      vec![manifest_name(&fp), slot_file_name(&fp, 1)]
    );
    assert_eq!(names(&other).len(), 2);
    std::fs::remove_dir_all(&cache).ok();
  }

  /// A GGUF with Llama-3.2-1B's attention geometry: 16 layers, 8 KV heads of
  /// 64, so 32,768 bytes of f16 KV per token. The spike measured 32.8 KB.
  fn write_llama_1b_header(path: &Path) {
    let bytes = crate::gguf::test_fixtures::FixtureBuilder::new()
      .with_arch("llama")
      .with_block_count(16)
      .with_head_count(32)
      .with_head_count_kv(8)
      .with_embedding_length(2048)
      .build();
    std::fs::write(path, bytes).unwrap();
  }

  #[tokio::test]
  async fn the_estimate_is_the_kv_cache_for_the_saved_tokens() {
    let cache = temp("estimate");
    let model = cache.join("m.gguf");
    write_llama_1b_header(&model);
    let mut p = chat_params(&model);
    assert_eq!(estimate_bytes(&p, 100_000).await, Some(3_276_800_000));
    assert!(p.knobs.set_by_name("cache-type-k", "q8_0"));
    assert!(p.knobs.set_by_name("cache-type-v", "q8_0"));
    let quantized = estimate_bytes(&p, 100_000).await.expect("estimate");
    assert!(quantized < 3_276_800_000 * 6 / 10, "{quantized}");

    std::fs::write(&model, b"not a gguf").unwrap();
    assert_eq!(
      estimate_bytes(&p, 100_000).await,
      None,
      "no header: no guess"
    );
    std::fs::remove_dir_all(&cache).ok();
  }

  #[tokio::test]
  async fn a_save_estimated_over_the_cap_is_refused_before_writing() {
    let cache = temp("overcap");
    let model = cache.join("m.gguf");
    write_llama_1b_header(&model);
    let p = seeded(&cache, &model);
    let dir = slot_dir(&p);
    let stub = one_busy_slot_stub();
    // 40,000 tokens x 32,768 bytes is 1.22 GiB.
    stub.lock().unwrap().slots_body =
      r#"[{"id":0,"is_processing":false,"n_prompt_tokens":40000}]"#.to_string();
    let port = stub_server(dir.clone(), stub.clone()).await;
    let cfg = SlotSaveConfig {
      max_gib: 1,
      ..Default::default()
    };
    assert_eq!(save(&cfg, &p, port).await, None);
    assert_eq!(stub.lock().unwrap().calls, vec!["GET /slots".to_string()]);
    assert!(names(&dir).is_empty());
    std::fs::remove_dir_all(&cache).ok();
  }
}
