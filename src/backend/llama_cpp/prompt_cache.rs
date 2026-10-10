//! Keep a conversation's prompt cache across an unload — llama.cpp only.
//!
//! When the idle sweep or make-room stops a llama.cpp launch mid-conversation,
//! the next request to that model reprocesses the whole prompt. `llama-server`
//! can write a slot's KV cache to disk and read it back (`--slot-save-path` plus
//! `POST /slots/{id_slot}?action=save|restore`), which turns that reprocessing
//! into a sub-second file read: measured 2 881-3 196 s to reprocess 100k tokens
//! against 0.7 s to save and 0.5-1.3 s to restore
//! (`docs/spikes/2026-10-10-prompt-cache-across-unload.md`).
//!
//! Three pieces:
//!
//! - **The launch** always carries `--slot-save-path <dir>` (see
//!   [`prepare_launch_dir`]), so the save/restore endpoints are live and every
//!   launch of a model writes into one directory.
//! - **Before an eviction stop** ([`preserve]`) every idle slot holding at least
//!   [`PromptCacheConfig::min_tokens`] prompt tokens is saved into that directory
//!   and recorded in its `manifest.json`. Only evictions save: a stop the user
//!   asked for is the end of that launch's conversation.
//! - **Before a launch is declared Ready** ([`restore`]) every manifest entry
//!   whose `signature` matches the launch is pushed back into the same slot id.
//!   The returning conversation then finds its cache without help: llama.cpp
//!   routes a request to the slot whose prompt shares the longest common prefix
//!   with it (`--slot-prompt-similarity`, default 0.1), and reuses only the tokens
//!   the two prompts actually have in common. A cache left over from a conversation
//!   that never returns is therefore never wrong, just unused.
//!
//! Files live under `<cache_dir>/prompt-cache/llamacpp/<key>/`, one directory per
//! `(model, mode)`. The manifest is the index; [`prune`] keeps it inside
//! [`PromptCacheConfig::max_bytes`] (oldest entries go first) and drops anything
//! older than [`PromptCacheConfig::max_age_secs`].
//!
//! The `signature` is what decides whether a saved cache may be reused: it covers the
//! model, the mode, the composed server argv except the parts that say *where* and
//! *how big* rather than *what the KV cache holds* (port, cache dir, device
//! placement, context window), and the server's build id. A launch that changes the
//! chat template or the KV type gets a different signature and simply starts cold.
//! The engine validates the rest — a file that cannot fit the new context or whose
//! KV layout differs is rejected by `llama-server` with an error, never loaded half.

use std::path::{Path, PathBuf};
use std::time::{Duration, SystemTime, UNIX_EPOCH};

use serde::{Deserialize, Serialize};

use crate::daemon::context::MethodContext;
use crate::daemon::registry::LaunchId;
use crate::launch::mode::LaunchMode;
use crate::launch::params::LaunchParams;

/// `launch_config` key carrying this launch's slot-save directory. Seeded by
/// [`prepare_launch_dir`] from config; absent means the feature is off for this
/// launch, which is the whole check every other path here makes.
pub const LLAMACPP_KNOB_PROMPT_CACHE_DIR: &str = "prompt_cache_dir";

/// Manifest format version. A file that carries anything else is replaced rather
/// than guessed at.
const MANIFEST_VERSION: u32 = 1;

/// Manifest file name inside a model's cache directory.
const MANIFEST_FILE: &str = "manifest.json";

/// Bound on one `/slots` save or restore round trip. Measured worst case is 2.7 s
/// to save and 1.3 s to restore 100k tokens; this only has to be far enough above
/// that to catch a wedged server, and short enough that a wedged server cannot eat
/// the readiness probe's budget.
const SLOT_OP_TIMEOUT: Duration = Duration::from_secs(60);

/// Bound on the two small reads (`GET /slots`, `GET /props`) that decide what to
/// save and what a file is compatible with.
const QUERY_TIMEOUT: Duration = Duration::from_secs(5);

/// Serialises manifest read-modify-write cycles inside this process. Two launches
/// of one model share a directory, and a manifest is written only after the saves
/// it describes have landed, so a torn file would mean a directory full of caches
/// nobody reads.
static MANIFEST_LOCK: tokio::sync::Mutex<()> = tokio::sync::Mutex::const_new(());

/// `backend.llamacpp.prompt_cache` — when a prompt cache is written, where it
/// lives, and how much disk and how much time it is allowed to occupy.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields, rename_all = "snake_case")]
pub struct PromptCacheConfig {
  /// Save on eviction and restore on launch (factory `true`). Applies to chat
  /// launches only; an embedding or rerank launch has no conversation to keep.
  pub enabled: bool,
  /// Don't bother saving a slot holding fewer prompt tokens than this. Below it
  /// the file is a few MB and the reprocessing the cache would have saved is under
  /// a second on a GPU.
  pub min_tokens: u32,
  /// Disk budget for one model's cache directory. Over the budget, the oldest
  /// entries are deleted until the total fits.
  pub max_bytes: u64,
  /// Discard a saved cache older than this. A conversation that has not come back
  /// within this window is not coming back, and the file is only dead weight.
  pub max_age_secs: u64,
}

impl Default for PromptCacheConfig {
  fn default() -> Self {
    Self {
      enabled: true,
      min_tokens: 512,
      max_bytes: 4 * 1024 * 1024 * 1024,
      max_age_secs: 6 * 60 * 60,
    }
  }
}

/// One saved slot.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct Entry {
  /// File name inside the directory, as handed to `llama-server`.
  pub filename: String,
  /// The slot it was saved from, and the slot [`restore`] puts it back in.
  pub slot: u32,
  /// The `signature` of the launch that saved it.
  pub signature: String,
  /// Prompt tokens the slot held.
  pub n_tokens: u64,
  /// Bytes the file takes.
  pub bytes: u64,
  /// Seconds since the Unix epoch when the save finished.
  pub saved_at: u64,
}

/// A directory's manifest, in memory.
#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct Manifest {
  #[serde(default = "manifest_version")]
  pub version: u32,
  #[serde(default)]
  pub entries: Vec<Entry>,
}

fn manifest_version() -> u32 {
  MANIFEST_VERSION
}

impl Manifest {
  /// Read a directory's manifest. Anything unreadable or of another version reads
  /// as empty — a bad manifest costs a cold start, it never blocks a launch, and
  /// the next save rewrites it.
  fn load(dir: &Path) -> Manifest {
    let text = match std::fs::read_to_string(dir.join(MANIFEST_FILE)) {
      Ok(text) => text,
      Err(_) => return Manifest::default(),
    };
    match serde_json::from_str::<Manifest>(&text) {
      Ok(m) if m.version == MANIFEST_VERSION => m,
      _ => {
        log::warn!(
          "prompt cache: {}/{} is unusable; starting a fresh one",
          dir.display(),
          MANIFEST_FILE
        );
        Manifest::default()
      }
    }
  }

  /// Write the manifest, atomically. A crash mid-write must not leave a truncated
  /// file behind, so the new contents land in a temp file that is then renamed.
  fn save(&self, dir: &Path) -> std::io::Result<()> {
    let tmp = dir.join(format!("{MANIFEST_FILE}.tmp"));
    let body = serde_json::to_string_pretty(self)
      .map_err(|e| std::io::Error::new(std::io::ErrorKind::InvalidData, e))?;
    std::fs::write(&tmp, body)?;
    std::fs::rename(&tmp, dir.join(MANIFEST_FILE))
  }

  fn total_bytes(&self) -> u64 {
    self.entries.iter().map(|e| e.bytes).sum()
  }
}

/// What prune decided, split so the caller can delete files and log the reason.
#[derive(Debug, Default, PartialEq, Eq)]
pub struct Pruned {
  /// Entries the manifest keeps.
  pub keep: Vec<Entry>,
  /// Entries removed, with the reason for each.
  pub dropped: Vec<(Entry, &'static str)>,
}

/// Drop entries the config no longer allows: files that are gone, saves older than
/// `max_age_secs`, and then the oldest until the directory is inside `max_bytes`.
///
/// Pure: it never touches the filesystem, so the byte and age arithmetic is
/// testable without a disk.
pub fn prune(entries: Vec<Entry>, cfg: &PromptCacheConfig, now: u64) -> Pruned {
  let mut keep = Vec::new();
  let mut dropped = Vec::new();
  for entry in entries {
    if entry.saved_at > now {
      // A clock that went backwards (or a hand-edited manifest) must not make an
      // entry immortal.
      dropped.push((entry, "saved in the future"));
      continue;
    }
    if now - entry.saved_at > cfg.max_age_secs {
      dropped.push((entry, "expired"));
      continue;
    }
    keep.push(entry);
  }
  // Newest first, so the byte cap evicts the stalest conversation rather than the
  // one that is most likely to come back.
  keep.sort_by(|a, b| b.saved_at.cmp(&a.saved_at).then(a.slot.cmp(&b.slot)));
  let mut total = keep.iter().map(|e| e.bytes).sum::<u64>();
  let mut over = Vec::new();
  while total > cfg.max_bytes {
    match keep.pop() {
      Some(entry) => {
        total = total.saturating_sub(entry.bytes);
        over.push(entry);
      }
      None => break,
    }
  }
  dropped.extend(over.into_iter().map(|e| (e, "over the disk budget")));
  Pruned { keep, dropped }
}

/// Files in `dir` that look like our slot saves but no live entry names — what a
/// save that died half way leaves behind.
fn orphan_files(dir: &Path, entries: &[Entry]) -> Vec<PathBuf> {
  let Ok(read) = std::fs::read_dir(dir) else {
    return Vec::new();
  };
  let live: Vec<&str> = entries.iter().map(|e| e.filename.as_str()).collect();
  read
    .flatten()
    .map(|e| e.path())
    .filter(|p| {
      p.file_name()
        .and_then(|n| n.to_str())
        .is_some_and(|n| is_slot_file(n) && !live.contains(&n))
    })
    .collect()
}

/// Whether a file name is one of this module's slot saves: `s<slot>-<sig>.bin`.
fn is_slot_file(name: &str) -> bool {
  let Some(rest) = name.strip_prefix('s').and_then(|r| r.strip_suffix(".bin")) else {
    return false;
  };
  match rest.split_once('-') {
    Some((slot, sig)) => {
      !slot.is_empty()
        && slot.bytes().all(|b| b.is_ascii_digit())
        && sig.len() == 16
        && sig.bytes().all(|b| b.is_ascii_hexdigit())
    }
    None => false,
  }
}

/// Drop entries whose file is no longer in `dir`, returning what went.
///
/// A file can go missing because it was deleted by hand or because a restore the
/// engine rejected deleted it. Left in place, such an entry is retried and reported
/// on every launch of the model forever.
fn drop_missing(dir: &Path, entries: &mut Vec<Entry>) -> Vec<Entry> {
  let gone: Vec<Entry> = entries
    .iter()
    .filter(|e| !dir.join(&e.filename).is_file())
    .cloned()
    .collect();
  entries.retain(|e| dir.join(&e.filename).is_file());
  gone
}

/// Delete what [`prune`] dropped plus anything the manifest lost track of, and
/// report how much that freed.
fn apply_prune(dir: &Path, pruned: &Pruned) -> u64 {
  let mut freed = 0u64;
  for (entry, reason) in &pruned.dropped {
    let path = dir.join(&entry.filename);
    match std::fs::remove_file(&path) {
      Ok(()) => freed = freed.saturating_add(entry.bytes),
      // Already gone: the manifest entry was stale, nothing to free.
      Err(e) if e.kind() == std::io::ErrorKind::NotFound => {}
      Err(e) => log::warn!("prompt cache: cannot remove {}: {e}", path.display()),
    }
    log::debug!(
      "prompt cache: dropped slot {} save ({} tokens, {} bytes) — {reason}",
      entry.slot,
      entry.n_tokens,
      entry.bytes
    );
  }
  for path in orphan_files(dir, &pruned.keep) {
    let bytes = std::fs::metadata(&path).map(|m| m.len()).unwrap_or(0);
    if std::fs::remove_file(&path).is_ok() {
      freed = freed.saturating_add(bytes);
      log::debug!("prompt cache: removed untracked {}", path.display());
    }
  }
  freed
}

/// Root of the tree: `<cache_dir>/prompt-cache/llamacpp`. `None` when the platform
/// gives no cache dir, which turns the feature off rather than writing somewhere
/// surprising.
fn root() -> Option<PathBuf> {
  crate::util::paths::cache_dir().map(|d| d.join("prompt-cache").join("llamacpp"))
}

/// One directory per `(model, mode)`: the weights decide what the cached tokens
/// mean, and a chat launch and an embedding launch of one file never share a
/// conversation.
fn cache_dir_for(model_path: &Path, mode: LaunchMode) -> Option<PathBuf> {
  let canonical = crate::util::paths::resolve_symlinks(model_path);
  let mut hasher = blake3::Hasher::new();
  hasher.update(canonical.to_string_lossy().as_bytes());
  hasher.update(&[0]);
  hasher.update(mode.label().as_bytes());
  Some(root()?.join(&hasher.finalize().to_hex().as_str()[..16]))
}

/// Argv positions that say where the model runs or how large a window it gets, not
/// what its KV cache holds. Excluded from [`signature`] so a cache survives a
/// launch that moved the model between devices or resized its window: the
/// contents of a cached prefix do not depend on either, and `llama-server` refuses
/// a file that cannot fit the window it is given.
///
/// Everything else in the composed argv — KV cache types, flash attention, slot
/// count, speculative decoding, the RoPE scaling, any `extras` — changes what the
/// saved bytes mean, so it is part of the signature.
const SIGNATURE_VOLATILE: &[&str] = &[
  "--port",
  "--slot-save-path",
  "-ngl",
  "--n-gpu-layers",
  "--n-cpu-moe",
  "--device",
  "--tensor-split",
  "--main-gpu",
  "--split-mode",
  "-c",
  "--ctx-size",
  "--fit-ctx",
];

/// The composed argv this launch's KV cache depends on, with the volatile
/// positions and their values filtered out.
///
/// Composed at port `0`: the port is filtered anyway, and composing needs *a*
/// port. The slot-save path comes from `launch_config`, so both sides of a
/// save/restore pair derive it the same way from the same params.
fn signature_argv(params: &LaunchParams) -> Vec<String> {
  let argv = super::compose::compose(params, 0);
  let mut out: Vec<String> = Vec::with_capacity(argv.len());
  let mut drop_value = false;
  for arg in argv {
    let text = arg.to_string_lossy().into_owned();
    if drop_value {
      drop_value = false;
      continue;
    }
    if SIGNATURE_VOLATILE.contains(&text.as_str()) {
      drop_value = true;
      continue;
    }
    out.push(text);
  }
  out
}

/// What a saved cache must match to be reusable: the model, the mode, the argv that
/// decides the shape of the KV cache, and the server build that wrote it.
///
/// The build id is the cheap guard against a state-format change between two
/// `llama-server` builds; `llama-server` checks a version of its own inside the
/// file and errors out rather than mis-reading it, so this only saves a doomed
/// restore attempt. `None` (an unreachable server) leaves it out of the hash,
/// which can lose a restore to a mismatch but never cause a bad one.
fn signature(params: &LaunchParams, build_info: Option<&str>) -> String {
  let mut hasher = blake3::Hasher::new();
  for part in signature_argv(params) {
    hasher.update(part.as_bytes());
    hasher.update(&[0]);
  }
  hasher.update(b"build:");
  hasher.update(build_info.unwrap_or("unknown").as_bytes());
  hasher.finalize().to_hex().to_string()[..16].to_string()
}

/// File name for one slot's save. `llama-server` validates the name (no path
/// separators), so it stays flat and short.
fn slot_filename(slot: u32, signature: &str) -> String {
  format!("s{slot}-{signature}.bin")
}

/// A slot as `GET /slots` reports it, narrowed to what the save decision needs.
#[derive(Debug, Clone, PartialEq, Eq)]
struct Slot {
  id: u32,
  /// Prompt tokens resident in the slot. `None` for a slot that has never served a
  /// request — `llama-server` reports the field only once a slot has had a task.
  n_prompt_tokens: Option<u64>,
  is_processing: bool,
}

/// Narrow `GET /slots` to the save decision: a slot that is idle and holds at
/// least `min_tokens` of prompt.
///
/// A processing slot is skipped rather than waited out: eviction only stops
/// launches with nothing in flight, so a busy slot here means a request arrived
/// after that check, and blocking the stop on it would be wrong.
fn save_targets(slots: &[serde_json::Value], min_tokens: u32) -> Vec<u32> {
  parse_slots(slots)
    .into_iter()
    .filter(|s| !s.is_processing)
    .filter(|s| {
      s.n_prompt_tokens
        .is_some_and(|n| n >= u64::from(min_tokens))
    })
    .map(|s| s.id)
    .collect()
}

/// Parse `GET /slots`. Anything the shape does not carry is `None`, not a guess.
fn parse_slots(slots: &[serde_json::Value]) -> Vec<Slot> {
  slots
    .iter()
    .filter_map(|v| {
      Some(Slot {
        id: v.get("id")?.as_u64()? as u32,
        n_prompt_tokens: v.get("n_prompt_tokens").and_then(|n| n.as_u64()),
        is_processing: v.get("is_processing").and_then(|b| b.as_bool())?,
      })
    })
    .collect()
}

/// The directory this launch was given, when the feature is on for it.
fn dir_of(params: &LaunchParams) -> Option<PathBuf> {
  params
    .launch_config
    .get(LLAMACPP_KNOB_PROMPT_CACHE_DIR)
    .map(PathBuf::from)
}

/// Stamp the launch's slot-save directory into `launch_config`, creating it.
///
/// Called from the backend's config seeding, before `compose` reads the map. Chat
/// mode only: an embedding or rerank launch answers one-shot requests, so its slot
/// cache is bytes nobody reads. `None` cache dir (no platform cache dir) leaves the
/// key unset, which is how every other path here learns the feature is off.
pub fn prepare_launch_dir(cfg: &PromptCacheConfig, params: &mut LaunchParams) {
  if !cfg.enabled || params.mode != LaunchMode::Chat {
    return;
  }
  let Some(dir) = cache_dir_for(&params.model_path, params.mode) else {
    return;
  };
  // llama-server refuses a `--slot-save-path` that is not a directory, so create
  // it here rather than lose the launch to a missing path.
  if let Err(e) = create_private_dir(&dir) {
    log::warn!("prompt cache: cannot create {}: {e}", dir.display());
    return;
  }
  params.launch_config.insert(
    LLAMACPP_KNOB_PROMPT_CACHE_DIR.to_string(),
    dir.display().to_string(),
  );
}

/// `create_dir_all` plus mode 0700 on the leaf.
///
/// A slot file holds what was in the conversation. The engine writes it with its own
/// umask, so the directory is where the contents get protected: without execute for
/// group and other, nothing else can reach the file inside it.
fn create_private_dir(dir: &Path) -> std::io::Result<()> {
  std::fs::create_dir_all(dir)?;
  #[cfg(unix)]
  {
    use std::os::unix::fs::PermissionsExt;
    std::fs::set_permissions(dir, std::fs::Permissions::from_mode(0o700))?;
  }
  #[cfg(not(unix))]
  let _ = dir;
  Ok(())
}

/// Whether `--slot-save-path` belongs on this launch's argv.
pub fn slot_save_path_arg(params: &LaunchParams) -> Option<String> {
  dir_of(params).map(|d| d.display().to_string())
}

/// Seconds since the Unix epoch, or `0` when the clock is before it.
fn now_secs() -> u64 {
  SystemTime::now()
    .duration_since(UNIX_EPOCH)
    .map(|d| d.as_secs())
    .unwrap_or_default()
}

/// Save every worth-saving slot of one launch and record the files.
///
/// Called with the launch still running, immediately before an eviction stop
/// ([`crate::proxy::eviction`]'s only backend call before `stop`). Best-effort in
/// every way: anything that fails — unreachable server, no directory, a save the
/// engine rejects — logs and returns, because a failed save costs a slow first
/// request while a failed stop costs memory.
pub async fn preserve(ctx: &MethodContext, launch_id: &LaunchId) {
  let cfg = &ctx.backend.llamacpp.prompt_cache;
  let Some(row) = ctx
    .state
    .snapshot()
    .await
    .running
    .iter()
    .find(|r| r.launch_id.as_ref() == Some(launch_id))
    .cloned()
  else {
    return;
  };
  let Some(dir) = dir_of(&row.params) else {
    return;
  };
  let port = row.port;
  let Some(slots) = query_json(port, "/slots", QUERY_TIMEOUT).await else {
    log::debug!(
      "prompt cache: {launch:?} has no readable /slots; nothing saved",
      launch = launch_id.as_str()
    );
    return;
  };
  let slots = slots.as_array().cloned().unwrap_or_default();
  let targets = save_targets(&slots, cfg.min_tokens);
  if targets.is_empty() {
    log::debug!(
      "prompt cache: nothing worth saving on {} ({} slot(s) below {} tokens)",
      launch_id.as_str(),
      slots.len(),
      cfg.min_tokens
    );
    return;
  }
  let build = build_info(port).await;
  let signature = signature(&row.params, build.as_deref());
  let started = std::time::Instant::now();
  let mut saved: Vec<Entry> = Vec::new();
  for slot in targets {
    let filename = slot_filename(slot, &signature);
    match request_save(port, slot, &filename).await {
      Ok(stats) => saved.push(Entry {
        filename,
        slot,
        signature: signature.clone(),
        n_tokens: stats.tokens,
        bytes: stats.bytes,
        saved_at: now_secs(),
      }),
      Err(e) => log::warn!(
        "prompt cache: save of slot {slot} on {} failed: {e}",
        launch_id.as_str()
      ),
    }
  }
  if saved.is_empty() {
    return;
  }
  let total: u64 = saved.iter().map(|e| e.bytes).sum();
  let _guard = MANIFEST_LOCK.lock().await;
  let mut manifest = Manifest::load(&dir);
  for entry in drop_missing(&dir, &mut manifest.entries) {
    log::debug!(
      "prompt cache: forgetting the entry for {}, whose file is gone",
      entry.filename
    );
  }
  for entry in &saved {
    manifest
      .entries
      .retain(|e| !(e.slot == entry.slot && e.signature == entry.signature));
    manifest.entries.push(entry.clone());
  }
  manifest.version = MANIFEST_VERSION;
  let pruned = prune(std::mem::take(&mut manifest.entries), cfg, now_secs());
  let freed = apply_prune(&dir, &pruned);
  manifest.entries = pruned.keep;
  if let Err(e) = manifest.save(&dir) {
    log::warn!(
      "prompt cache: cannot write {}/manifest.json: {e}",
      dir.display()
    );
  }
  log::info!(
    "prompt cache: saved {} slot(s) of {} ({} tokens, {:.1} MiB{}) in {:?}; {} now holds {:.1} MiB",
    saved.len(),
    launch_id.as_str(),
    saved.iter().map(|e| e.n_tokens).sum::<u64>(),
    total as f64 / (1024.0 * 1024.0),
    if freed > 0 {
      format!(", freed {:.1} MiB", freed as f64 / (1024.0 * 1024.0))
    } else {
      String::new()
    },
    started.elapsed(),
    dir.display(),
    manifest.total_bytes() as f64 / (1024.0 * 1024.0),
  );
}

/// Restore this launch's saved prompt caches into the child that just came up,
/// before it is declared Ready.
///
/// One attempt per matching manifest entry, lowest slot first. Nothing is consumed:
/// a saved cache stays until it ages out or the disk budget evicts it, so a launch
/// that comes up and is never asked anything does not cost the next launch its
/// cache. Only a file the engine rejects goes, because a signature that matches the
/// launch that wrote it will keep rejecting it. Everything here is best-effort: a
/// launch that restores nothing behaves the way it did before this feature existed.
pub async fn restore(port: u16, params: &LaunchParams) {
  let Some(dir) = dir_of(params) else {
    return;
  };
  let manifest = Manifest::load(&dir);
  if manifest.entries.is_empty() {
    return;
  }
  let build = build_info(port).await;
  let signature = signature(params, build.as_deref());
  let mut candidates: Vec<Entry> = manifest
    .entries
    .into_iter()
    .filter(|e| e.signature == signature)
    .collect();
  if candidates.is_empty() {
    return;
  }
  candidates.sort_by_key(|e| e.slot);
  let started = std::time::Instant::now();
  let mut restored = 0u64;
  let mut tokens = 0u64;
  let mut failures = 0usize;
  for entry in &candidates {
    let path = dir.join(&entry.filename);
    if !path.is_file() {
      log::debug!(
        "prompt cache: {} is gone; forgetting its entry",
        path.display()
      );
      failures += 1;
      continue;
    }
    match request_restore(port, entry.slot, &entry.filename).await {
      Ok(n) => {
        restored += 1;
        tokens += n;
      }
      Err(e) => {
        failures += 1;
        log::warn!(
          "prompt cache: cannot restore slot {} from {}: {e}",
          entry.slot,
          path.display()
        );
        // The file cannot be used by this launch or, since the signature matches
        // the launch that wrote it, by any future one of the same shape.
        if let Err(err) = std::fs::remove_file(&path) {
          if err.kind() != std::io::ErrorKind::NotFound {
            log::warn!("prompt cache: cannot remove {}: {err}", path.display());
          }
        }
      }
    }
  }
  if restored > 0 {
    log::info!(
      "prompt cache: restored {restored} slot(s) ({tokens} prompt tokens) in {:?}{}",
      started.elapsed(),
      if failures > 0 {
        format!(", {failures} unusable")
      } else {
        String::new()
      }
    );
  } else if failures > 0 {
    log::info!(
      "prompt cache: {} saved slot(s) unusable in {:?}",
      failures,
      started.elapsed()
    );
  }
}

/// What a successful save reported.
struct SaveStats {
  tokens: u64,
  bytes: u64,
}

async fn request_save(port: u16, slot: u32, filename: &str) -> Result<SaveStats, String> {
  let body = format!("{{\"filename\":\"{filename}\"}}");
  let v = post_json(
    port,
    &format!("/slots/{slot}?action=save"),
    &body,
    SLOT_OP_TIMEOUT,
  )
  .await?;
  Ok(SaveStats {
    tokens: v.get("n_saved").and_then(|n| n.as_u64()).unwrap_or(0),
    bytes: v.get("n_written").and_then(|n| n.as_u64()).unwrap_or(0),
  })
}

async fn request_restore(port: u16, slot: u32, filename: &str) -> Result<u64, String> {
  let body = format!("{{\"filename\":\"{filename}\"}}");
  let v = post_json(
    port,
    &format!("/slots/{slot}?action=restore"),
    &body,
    SLOT_OP_TIMEOUT,
  )
  .await?;
  Ok(v.get("n_restored").and_then(|n| n.as_u64()).unwrap_or(0))
}

/// The server's build id from `/props`, used as a format guard in [`signature`].
async fn build_info(port: u16) -> Option<String> {
  query_json(port, "/props", QUERY_TIMEOUT)
    .await?
    .get("build_info")
    .and_then(|v| v.as_str())
    .map(|s| s.to_string())
}

/// `GET` a JSON body from the child. `None` on any failure; the callers all treat
/// an unreachable server as "nothing to do".
async fn query_json(port: u16, path: &str, timeout: Duration) -> Option<serde_json::Value> {
  let body = exchange(port, format!("GET {path} HTTP/1.1\r\nHost: 127.0.0.1:{port}\r\nAccept: application/json\r\nConnection: close\r\n\r\n"), timeout).await.ok()?;
  serde_json::from_slice(&body).ok()
}

/// `POST` a JSON body to the child and decode its reply, or describe the failure.
async fn post_json(
  port: u16,
  path: &str,
  body: &str,
  timeout: Duration,
) -> Result<serde_json::Value, String> {
  let request = format!(
    "POST {path} HTTP/1.1\r\nHost: 127.0.0.1:{port}\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{body}",
    body.len()
  );
  let reply = exchange(port, request, timeout).await?;
  serde_json::from_slice(&reply).map_err(|e| format!("unreadable reply: {e}"))
}

/// Send one hand-rolled HTTP/1.1 request to the child and return the body on a
/// 2xx, or the engine's own error text. Same no-dep stance as
/// [`super::actuals`]: `Connection: close` lets a read to EOF end the response
/// without a keep-alive dance.
async fn exchange(port: u16, request: String, timeout: Duration) -> Result<Vec<u8>, String> {
  use tokio::io::{AsyncReadExt, AsyncWriteExt};
  let fut = async {
    let mut sock = tokio::net::TcpStream::connect(("127.0.0.1", port))
      .await
      .map_err(|e| format!("connect to {port}: {e}"))?;
    sock
      .write_all(request.as_bytes())
      .await
      .map_err(|e| format!("write: {e}"))?;
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
      if buf.len() > 512 * 1024 {
        return Err("reply larger than expected".to_string());
      }
    }
    Ok(buf)
  };
  let raw = tokio::time::timeout(timeout, fut)
    .await
    .map_err(|_| "timed out".to_string())??;
  let split = raw
    .windows(4)
    .position(|w| w == b"\r\n\r\n")
    .ok_or_else(|| "reply without headers".to_string())?;
  let head = String::from_utf8_lossy(&raw[..split]);
  let code = head
    .split_whitespace()
    .nth(1)
    .and_then(|s| s.parse::<u16>().ok())
    .unwrap_or(0);
  let body = &raw[split + 4..];
  if !(200..300).contains(&code) {
    let text = String::from_utf8_lossy(body).to_string();
    return Err(match serde_json::from_str::<serde_json::Value>(&text) {
      Ok(v) => v
        .get("error")
        .and_then(|e| e.get("message"))
        .and_then(|m| m.as_str())
        .unwrap_or(text.as_str())
        .to_string(),
      Err(_) => text,
    });
  }
  Ok(body.to_vec())
}

#[cfg(test)]
mod tests {
  use super::*;
  use crate::launch::knobs::value::{KnobSet, Scalar};
  use crate::launch::params::LaunchParams;

  fn entry(slot: u32, sig: &str, tokens: u64, bytes: u64, saved_at: u64) -> Entry {
    Entry {
      filename: slot_filename(slot, sig),
      slot,
      signature: sig.to_string(),
      n_tokens: tokens,
      bytes,
      saved_at,
    }
  }

  fn cfg() -> PromptCacheConfig {
    PromptCacheConfig {
      enabled: true,
      min_tokens: 512,
      max_bytes: 1000,
      max_age_secs: 100,
    }
  }

  #[cfg(unix)]
  #[test]
  fn the_slot_directory_keeps_everyone_else_out() {
    use std::os::unix::fs::PermissionsExt;
    let root = tempfile::tempdir().unwrap();
    let dir = root.path().join("prompt-cache/llamacpp/a85838d46bac9c74");
    create_private_dir(&dir).unwrap();
    assert_eq!(
      std::fs::metadata(&dir).unwrap().permissions().mode() & 0o777,
      0o700,
      "a slot file holds the conversation, so nothing else may traverse in"
    );
  }

  #[test]
  fn slot_file_names_are_flat_and_recognised() {
    let name = slot_filename(12, "0123456789abcdef");
    assert_eq!(name, "s12-0123456789abcdef.bin");
    assert!(is_slot_file(&name));
    assert!(is_slot_file("s0-0123456789abcdef.bin"));
    // A signature of the wrong length, a path, a foreign file: none are ours.
    assert!(!is_slot_file("s0-0123456789abcde.bin"));
    assert!(!is_slot_file("../s0-0123456789abcdef.bin"));
    assert!(!is_slot_file("manifest.json"));
    assert!(!is_slot_file("slot-0.bin"));
    assert!(!is_slot_file("sX-0123456789abcdef.bin"));
  }

  #[test]
  fn prune_keeps_everything_inside_both_limits() {
    let entries = vec![entry(0, "a", 10, 400, 90), entry(1, "a", 20, 400, 95)];
    let p = prune(entries, &cfg(), 100);
    assert_eq!(p.keep.len(), 2);
    assert!(p.dropped.is_empty());
    assert_eq!(p.keep[0].slot, 1, "newest first");
  }

  #[test]
  fn prune_drops_expired_then_oldest_over_the_cap() {
    // cfg(): max_age_secs 100, max_bytes 1000. now = 200.
    let expired = entry(0, "a", 10, 100, 10);
    let older = entry(1, "a", 10, 600, 150);
    let fresher = entry(2, "a", 10, 600, 190);
    let p = prune(vec![expired, older.clone(), fresher.clone()], &cfg(), 200);
    assert_eq!(p.keep.len(), 1);
    assert_eq!(p.dropped.len(), 2);
    assert!(p
      .dropped
      .iter()
      .any(|(e, r)| e.slot == 0 && *r == "expired"));
    assert!(
      p.dropped
        .iter()
        .any(|(e, r)| e.slot == 1 && *r == "over the disk budget"),
      "the older of the two must go"
    );
    assert!(p.keep.iter().any(|e| e.slot == 2));
  }

  #[test]
  fn prune_sorts_a_clock_that_went_backwards_into_the_past() {
    let future = entry(0, "a", 10, 10, 500);
    let p = prune(vec![future], &cfg(), 100);
    assert!(p.keep.is_empty());
    assert_eq!(p.dropped[0].1, "saved in the future");
  }

  #[test]
  fn prune_of_one_oversized_entry_leaves_nothing() {
    // A single save bigger than the whole budget cannot coexist with anything,
    // including itself: it goes rather than being kept and re-dropped every pass.
    let big = entry(0, "a", 10, 5000, 99);
    let p = prune(vec![big], &cfg(), 100);
    assert!(p.keep.is_empty());
    assert_eq!(p.dropped.len(), 1);
  }

  #[test]
  fn signature_survives_a_resize_of_the_window() {
    // `--fit` can land a different window on two launches of one config, and a
    // smaller window still takes a cache that fits it; `llama-server` rejects the
    // ones that do not rather than loading them half.
    let mut wide = params(&[]);
    wide.ctx = Some(131072);
    let mut narrow = params(&[]);
    narrow.ctx = Some(4096);
    assert_eq!(signature(&wide, Some("b1")), signature(&narrow, Some("b1")));
  }

  #[test]
  fn save_targets_skips_busy_short_and_never_used_slots() {
    let slots = serde_json::from_str::<Vec<serde_json::Value>>(
      r#"[{"id":0,"is_processing":false},
          {"id":1,"is_processing":false,"n_prompt_tokens":511},
          {"id":2,"is_processing":false,"n_prompt_tokens":512},
          {"id":3,"is_processing":true,"n_prompt_tokens":9000}]"#,
    )
    .unwrap();
    assert_eq!(save_targets(&slots, 512), vec![2]);
    // A malformed row is dropped, not treated as a zero-token slot.
    let junk = serde_json::from_str::<Vec<serde_json::Value>>(r#"[{"id":9}]"#).unwrap();
    assert!(save_targets(&junk, 1).is_empty());
    assert!(save_targets(&[], 512).is_empty());
  }

  fn params(extra: &[&str]) -> LaunchParams {
    let mut p = LaunchParams::new("/models/m.gguf".into(), LaunchMode::Chat);
    p.extras = extra.iter().map(|s| s.into()).collect();
    p
  }

  #[test]
  fn signature_ignores_port_and_device_placement() {
    let base = params(&[]);
    let sig = signature(&base, Some("b11525-609290be6"));
    // Same launch, different port and different placement: same cache.
    let mut moved = LaunchParams::new("/models/m.gguf".into(), LaunchMode::Chat);
    moved.port = Some(9999);
    let mut knobs = KnobSet::new();
    knobs.set_scalar(crate::launch::knobs::kid("n-gpu-layers"), Scalar::U32(20));
    moved.knobs = knobs;
    assert_eq!(sig, signature(&moved, Some("b11525-609290be6")));
  }

  #[test]
  fn signature_follows_what_the_kv_cache_holds() {
    let base = signature(&params(&[]), Some("b1"));
    let mut kv = params(&[]);
    let mut knobs = KnobSet::new();
    knobs.set_scalar(
      crate::launch::knobs::kid("cache-type-k"),
      Scalar::Str("q8_0".into()),
    );
    kv.knobs = knobs;
    assert_ne!(
      base,
      signature(&kv, Some("b1")),
      "KV type changes the bytes"
    );
    assert_ne!(
      base,
      signature(&params(&["--flash-attn"]), Some("b1")),
      "flash attention changes the cell layout"
    );
    assert_ne!(
      base,
      signature(&params(&[]), Some("b2")),
      "a different server build is not the same file format"
    );
    let other_model = LaunchParams::new("/models/other.gguf".into(), LaunchMode::Chat);
    assert_ne!(base, signature(&other_model, Some("b1")));
    let embed = LaunchParams::new("/models/m.gguf".into(), LaunchMode::Embedding);
    assert_ne!(base, signature(&embed, Some("b1")));
  }

  #[test]
  fn signature_argv_drops_the_volatile_pairs_only() {
    let mut p = params(&[]);
    p.ctx = Some(8192);
    p.mmproj_path = Some("/models/mmproj.gguf".into());
    let argv = signature_argv(&p);
    assert!(
      !argv
        .iter()
        .any(|a| a == "--port" || a == "0" || a == "-ngl"),
      "{argv:?}"
    );
    // The window a launch is given says nothing about what the cached bytes hold, so
    // a resize does not throw the cache away.
    assert!(!argv.iter().any(|a| a == "-c" || a == "8192"), "{argv:?}");
    assert!(argv.contains(&"-m".to_string()));
    assert!(argv.contains(&"--mmproj".to_string()));
  }

  #[test]
  fn manifest_round_trips_and_bad_input_reads_empty() {
    let dir = tempfile::tempdir().unwrap();
    let m = Manifest {
      version: MANIFEST_VERSION,
      entries: vec![entry(3, "a", 611, 20_000, 123)],
    };
    m.save(dir.path()).unwrap();
    assert_eq!(Manifest::load(dir.path()), m);
    assert_eq!(m.total_bytes(), 20_000);

    std::fs::write(dir.path().join(MANIFEST_FILE), b"{ not json").unwrap();
    assert_eq!(Manifest::load(dir.path()), Manifest::default());
    std::fs::write(
      dir.path().join(MANIFEST_FILE),
      r#"{"version":99,"entries":[]}"#,
    )
    .unwrap();
    assert_eq!(Manifest::load(dir.path()), Manifest::default());
    // No file at all is the normal case for a first launch.
    std::fs::remove_file(dir.path().join(MANIFEST_FILE)).unwrap();
    assert_eq!(Manifest::load(dir.path()), Manifest::default());
  }

  #[test]
  fn manifest_save_leaves_no_temp_file() {
    let dir = tempfile::tempdir().unwrap();
    Manifest::default().save(dir.path()).unwrap();
    assert!(!dir.path().join(format!("{MANIFEST_FILE}.tmp")).exists());
  }

  #[test]
  fn drop_missing_forgets_entries_whose_file_is_gone() {
    let dir = tempfile::tempdir().unwrap();
    let here = entry(0, "0123456789abcdef", 10, 10, 1);
    let gone = entry(1, "fedcba9876543210", 10, 10, 1);
    std::fs::write(dir.path().join(&here.filename), b"x").unwrap();
    let mut entries = vec![here.clone(), gone.clone()];
    assert_eq!(drop_missing(dir.path(), &mut entries), vec![gone]);
    assert_eq!(entries, vec![here]);
  }

  #[test]
  fn orphan_files_are_ours_only() {
    let dir = tempfile::tempdir().unwrap();
    let mine = slot_filename(0, "0123456789abcdef");
    let stale = slot_filename(1, "fedcba9876543210");
    std::fs::write(dir.path().join(&mine), b"x").unwrap();
    std::fs::write(dir.path().join(&stale), b"x").unwrap();
    std::fs::write(dir.path().join("someone-else.bin"), b"x").unwrap();
    let kept = vec![entry(0, "0123456789abcdef", 1, 1, 1)];
    let orphans = orphan_files(dir.path(), &kept);
    assert_eq!(orphans.len(), 1);
    assert!(orphans[0].ends_with(&stale));
  }

  #[tokio::test]
  async fn exchange_reports_the_engines_own_error_text() {
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let port = listener.local_addr().unwrap().port();
    tokio::spawn(async move {
      let (mut sock, _) = listener.accept().await.unwrap();
      use tokio::io::{AsyncReadExt, AsyncWriteExt};
      let mut buf = [0u8; 1024];
      let _ = sock.read(&mut buf).await;
      let body = r#"{"error":{"code":400,"message":"Unable to restore slot: no space"}}"#;
      let reply = format!(
        "HTTP/1.1 400 Bad Request\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{body}",
        body.len()
      );
      let _ = sock.write_all(reply.as_bytes()).await;
    });
    let err = exchange(
      port,
      "POST /x HTTP/1.1\r\n\r\n".into(),
      Duration::from_secs(5),
    )
    .await
    .unwrap_err();
    assert_eq!(err, "Unable to restore slot: no space");
  }

  #[tokio::test]
  async fn exchange_decodes_a_2xx_body() {
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let port = listener.local_addr().unwrap().port();
    tokio::spawn(async move {
      let (mut sock, _) = listener.accept().await.unwrap();
      use tokio::io::{AsyncReadExt, AsyncWriteExt};
      let mut buf = [0u8; 1024];
      let _ = sock.read(&mut buf).await;
      let _ = sock
        .write_all(b"HTTP/1.1 200 OK\r\nContent-Length: 2\r\nConnection: close\r\n\r\n{}")
        .await;
    });
    assert!(query_json(port, "/props", Duration::from_secs(5))
      .await
      .is_some());
  }

  #[tokio::test]
  async fn an_unreachable_server_is_a_silent_no_op() {
    // Port 1 cannot be connected to as an unprivileged user, which is the case
    // `preserve` and `restore` must survive without logging an error per launch.
    let mut p = params(&[]);
    p.launch_config.insert(
      LLAMACPP_KNOB_PROMPT_CACHE_DIR.into(),
      "/nonexistent-dir-for-test".into(),
    );
    restore(1, &p).await;
  }

  #[test]
  fn prepare_launch_dir_is_off_for_non_chat_and_disabled_config() {
    let cfg = PromptCacheConfig::default();
    let mut embed = LaunchParams::new("/models/m.gguf".into(), LaunchMode::Embedding);
    prepare_launch_dir(&cfg, &mut embed);
    assert!(embed.launch_config.is_empty());

    let mut chat = LaunchParams::new("/models/m.gguf".into(), LaunchMode::Chat);
    prepare_launch_dir(
      &PromptCacheConfig {
        enabled: false,
        ..Default::default()
      },
      &mut chat,
    );
    assert!(chat.launch_config.is_empty());
  }

  #[test]
  fn cache_dir_is_per_model_and_mode_under_the_cache_dir() {
    let base = cache_dir_for(Path::new("/models/a.gguf"), LaunchMode::Chat).unwrap();
    let same = cache_dir_for(Path::new("/models/a.gguf"), LaunchMode::Chat).unwrap();
    let other = cache_dir_for(Path::new("/models/b.gguf"), LaunchMode::Chat).unwrap();
    let embed = cache_dir_for(Path::new("/models/a.gguf"), LaunchMode::Embedding).unwrap();
    assert_eq!(base, same);
    assert_ne!(base, other);
    assert_ne!(base, embed);
    assert!(base.starts_with(crate::util::paths::cache_dir().unwrap()));
    // Fits inside a slot file name's expectations: one flat directory component.
    assert_eq!(base.file_name().unwrap().to_string_lossy().len(), 16);
  }
}
