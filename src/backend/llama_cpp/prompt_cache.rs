//! Prompt cache across an unload.
//!
//! A stopped `llama-server` loses every slot's KV cache, so the next launch of
//! the same model reprocesses a conversation's whole prompt. llama.cpp can write
//! a slot's KV to disk (`--slot-save-path PATH` plus
//! `POST /slots/{id_slot}?action=save|restore`), so:
//!
//! - the proxy's eviction paths call [`save`] just before they stop a launch, and
//! - the supervisor calls [`restore`] once a fresh launch is ready, before it is
//!   declared Ready so no request can be routed past an empty cache.
//!
//! Both are best-effort: every failure is a log line, never a failed stop or a
//! withheld Ready. A restore that does not fit (ctx changed, model file
//! replaced, incompatible build) makes llama.cpp answer an error; the file is
//! then dropped rather than retried on every launch.
//!
//! **Slot matching.** llama.cpp matches a returning request to a slot by token
//! prefix, so no conversation id is needed: each idle slot's KV goes to a file
//! named after the launch's *identity* and its slot number, and the new launch
//! puts each file back in the slot it came from. Files whose identity does not
//! match the live launch are invisible to it, and a prompt that no longer
//! matches the returning request costs nothing but the space.
//!
//! Transport is `reqwest` against the loopback port of the launch, the same
//! stance as the lemonade client.

use std::path::{Path, PathBuf};
use std::time::{Duration, SystemTime, UNIX_EPOCH};

use serde::Deserialize;

use super::LlamaCppConfig;
use crate::daemon::supervisor::ManagedModel;

/// `launch_config` key carrying the directory this launch parks its slot KV
/// caches in. Written by `seed_launch_knobs` when the feature is on; its absence
/// is what makes the argv flag and both hooks no-ops, so a config flip can never
/// be inherited from `last_params`.
pub const KNOB_PROMPT_CACHE_DIR: &str = "prompt_cache_dir";

/// Subdirectory of `paths::cache_dir()` holding the files. Cache, not state:
/// every byte here is re-derivable by reprocessing a prompt.
const DIR_NAME: &str = "prompt-cache";
/// Naming rule below is `<prefix><slug>-<identity>-slot<N><suffix>`; `prune`
/// only touches files matching it, so anything else in the directory is left
/// alone.
const FILE_PREFIX: &str = "llamastash-pc-";
const FILE_SUFFIX: &str = ".bin";
/// Longest model-file stem kept in a file name, so a pathological file name
/// cannot push the save file past the 255-byte limit llama.cpp validates.
const SLUG_MAX: usize = 48;
/// Hex characters of the identity digest kept in a file name.
const IDENTITY_HEX: usize = 16;
/// Budget for one `POST /slots/{id}?action=save|restore`. A 100k-token save is
/// a few GiB and lands in a couple of seconds on NVMe; this bounds the tail, and
/// it also bounds how long an eviction can be held up by a slot that began
/// processing after the slot list was read.
const SLOT_RPC_TIMEOUT: Duration = Duration::from_secs(30);

/// One row of `GET /slots?debug=1`. `n_prompt_tokens` only appears with
/// `debug=1`; without it a slot's emptiness is unknown, which [`Slot::may_have_prompt`]
/// resolves by trying the save and dropping the file when it wrote nothing.
#[derive(Debug, Clone, Deserialize)]
struct Slot {
  id: i64,
  #[serde(default)]
  is_processing: bool,
  #[serde(default)]
  n_prompt_tokens: Option<i64>,
}

impl Slot {
  /// Worth a save? An unknown prompt size counts as yes.
  fn may_have_prompt(&self) -> bool {
    self.n_prompt_tokens.map(|n| n > 0).unwrap_or(true)
  }

  /// Free to receive a restore — a slot that already holds a prompt of its own
  /// would rather keep it.
  fn open_for_restore(&self) -> bool {
    self.n_prompt_tokens.map(|n| n == 0).unwrap_or(true)
  }
}

/// What `POST /slots/{id}?action=save|restore` reports. The two actions use
/// different field names; every field is optional so either response parses.
#[derive(Debug, Default, Deserialize)]
struct SlotAction {
  #[serde(default)]
  n_saved: Option<i64>,
  #[serde(default)]
  n_written: Option<u64>,
  #[serde(default)]
  n_restored: Option<i64>,
}

/// What a prune pass removed, for the log line.
#[derive(Debug, Default, PartialEq, Eq)]
pub(crate) struct Pruned {
  pub removed: usize,
  pub freed_bytes: u64,
}

/// The directory a *new* launch should park in: created on demand, expired files
/// dropped while we are here. `None` when the feature is off or the directory
/// cannot be made — the launch then runs with no `--slot-save-path` at all, which
/// is also what llama.cpp would do if it were given a missing path (it refuses to
/// start), so the flag is only ever emitted for a directory that exists.
pub(crate) fn prepare_dir(cfg: &LlamaCppConfig) -> Option<PathBuf> {
  if !cfg.prompt_cache {
    return None;
  }
  let dir = crate::util::paths::cache_dir()?.join(DIR_NAME);
  if std::fs::create_dir_all(&dir).is_err() {
    log::debug!(
      "prompt cache: cannot create {}; launches run without a saved prompt cache",
      dir.display()
    );
    return None;
  }
  let pruned = prune(&dir, cfg);
  report_pruned(&pruned);
  Some(dir)
}

/// One line per prune pass, so "the cap is too small" is visible without turning
/// on debug logging.
fn report_pruned(pruned: &Pruned) {
  if pruned.removed == 0 {
    return;
  }
  log::info!(
    "prompt cache: dropped {} saved slot(s) ({} MiB) past the age or size limit",
    pruned.removed,
    pruned.freed_bytes / (1024 * 1024)
  );
}

/// The directory this launch was given at launch time.
fn launch_dir(model: &ManagedModel) -> Option<PathBuf> {
  model
    .params()
    .launch_config
    .get(KNOB_PROMPT_CACHE_DIR)
    .map(PathBuf::from)
}

/// Write every idle slot's KV to the launch's directory. Called just before the
/// proxy's eviction paths stop a launch; a manual `stop` never saves.
pub(crate) async fn save(model: &ManagedModel, cfg: &LlamaCppConfig) {
  let Some(dir) = launch_dir(model) else {
    return;
  };
  let port = model.port();
  let slots = match list_slots(port).await {
    Ok(slots) => slots,
    Err(e) => {
      log::debug!("prompt cache: slot list from port {port} failed: {e}");
      return;
    }
  };
  let Some(identity) = identity(&model.params().model_path) else {
    return;
  };
  let mut saved = 0usize;
  let mut bytes = 0u64;
  let mut written: Vec<(PathBuf, u64)> = Vec::new();
  for slot in slots
    .iter()
    .filter(|s| !s.is_processing && s.may_have_prompt())
  {
    let name = slot_file_name(&model.params().model_path, &identity, slot.id);
    match slot_action(port, "save", slot.id, &name).await {
      Ok(res) => {
        let tokens = res.n_saved.unwrap_or(0);
        if tokens <= 0 {
          // Nothing cached in that slot; llama.cpp still wrote a header.
          let _ = std::fs::remove_file(dir.join(&name));
          continue;
        }
        saved += 1;
        bytes += res.n_written.unwrap_or(0);
        written.push((dir.join(&name), res.n_written.unwrap_or(0)));
        log::debug!(
          "prompt cache: slot {} saved to {name} ({tokens} tokens)",
          slot.id
        );
      }
      Err(e) => log::debug!(
        "prompt cache: slot {} save failed: {}",
        slot.id,
        e.message()
      ),
    }
  }
  if saved > 0 {
    log::info!(
      "prompt cache: saved {saved} idle slot(s) ({bytes} bytes) of {} — the next launch of it starts warm",
      model.params().model_path.display()
    );
  }
  let cap = cfg.prompt_cache_max_bytes();
  let pruned = prune(&dir, cfg);
  report_pruned(&pruned);
  // A single conversation bigger than the whole budget would otherwise look
  // like a save that silently did nothing.
  for (path, n) in &written {
    if *n > cap && !path.exists() {
      log::info!(
        "prompt cache: this slot's cache is {n} bytes, over the {cap}-byte prompt_cache_max_mb limit, so it was dropped; raise the cap to keep prompts this large"
      );
    }
  }
}

/// Put the parked KV back into the slots it came from. Called once, just before a
/// fresh launch is declared Ready, so the first request that arrives finds the
/// cache in place.
pub(crate) async fn restore(model: &ManagedModel) {
  let Some(dir) = launch_dir(model) else {
    return;
  };
  let port = model.port();
  let slots = match list_slots(port).await {
    Ok(slots) => slots,
    Err(e) => {
      log::debug!("prompt cache: slot list from port {port} failed: {e}");
      return;
    }
  };
  let Some(identity) = identity(&model.params().model_path) else {
    return;
  };
  let mut restored = 0usize;
  let mut tokens = 0i64;
  for slot in slots
    .iter()
    .filter(|s| !s.is_processing && s.open_for_restore())
  {
    let name = slot_file_name(&model.params().model_path, &identity, slot.id);
    let path = dir.join(&name);
    if !path.is_file() {
      continue;
    }
    match slot_action(port, "restore", slot.id, &name).await {
      Ok(res) => {
        restored += 1;
        tokens += res.n_restored.unwrap_or(0);
        log::debug!("prompt cache: slot {} restored from {name}", slot.id);
      }
      Err(ActionError::Rejected(e)) => {
        // The server answered and refused the file: its KV layout or prompt
        // length cannot fit this launch, and it will not start fitting later,
        // so the file has no future. A timeout or a dropped connection is a
        // different story and leaves the file alone.
        log::debug!("prompt cache: dropping {}: {e}", path.display());
        let _ = std::fs::remove_file(&path);
      }
      Err(ActionError::Unreachable(e)) => {
        log::debug!("prompt cache: slot {} restore failed: {e}", slot.id)
      }
    }
  }
  if restored > 0 {
    log::info!(
      "prompt cache: restored {restored} slot(s) ({tokens} tokens) for {} — the next request skips prefill",
      model.params().model_path.display()
    );
  }
}

/// `GET /slots?debug=1` on the launch's own port.
async fn list_slots(port: u16) -> Result<Vec<Slot>, String> {
  let url = format!("http://127.0.0.1:{port}/slots?debug=1");
  let body = client(SLOT_RPC_TIMEOUT)?
    .get(&url)
    .send()
    .await
    .map_err(|e| e.to_string())?
    .error_for_status()
    .map_err(|e| e.to_string())?
    .text()
    .await
    .map_err(|e| e.to_string())?;
  serde_json::from_str(&body).map_err(|e| format!("unparsable /slots response: {e}"))
}

/// One `save` / `restore` round trip for one slot.
async fn slot_action(
  port: u16,
  action: &str,
  slot: i64,
  filename: &str,
) -> Result<SlotAction, ActionError> {
  let url = format!("http://127.0.0.1:{port}/slots/{slot}?action={action}");
  let resp = client(SLOT_RPC_TIMEOUT)
    .map_err(ActionError::Unreachable)?
    .post(&url)
    .json(&serde_json::json!({ "filename": filename }))
    .send()
    .await
    .map_err(|e| ActionError::Unreachable(e.to_string()))?;
  let status = resp.status();
  let body = resp
    .text()
    .await
    .map_err(|e| ActionError::Unreachable(e.to_string()))?;
  if !status.is_success() {
    // llama.cpp answers `{"error":{"message": ...}}`; keep the message, drop
    // the envelope.
    let message = serde_json::from_str::<serde_json::Value>(&body)
      .ok()
      .and_then(|v| {
        v.pointer("/error/message")
          .and_then(|m| m.as_str())
          .map(str::to_string)
      })
      .unwrap_or_else(|| format!("HTTP {status}"));
    return Err(ActionError::Rejected(message));
  }
  serde_json::from_str(&body)
    .map_err(|e| ActionError::Unreachable(format!("unparsable {action} response: {e}")))
}

/// Why a slot action did not succeed. The difference matters for the file: a
/// refusal is a property of the file, a timeout is a property of the day.
#[derive(Debug)]
enum ActionError {
  Rejected(String),
  Unreachable(String),
}

impl ActionError {
  fn message(&self) -> &str {
    match self {
      ActionError::Rejected(m) | ActionError::Unreachable(m) => m,
    }
  }
}

fn client(timeout: Duration) -> Result<reqwest::Client, String> {
  reqwest::Client::builder()
    .timeout(timeout)
    // Loopback only; a proxy in the environment must not intercept it.
    .no_proxy()
    .build()
    .map_err(|e| e.to_string())
}

/// What has to agree for a saved file to be restorable: the model file itself
/// (path, size and mtime, so a re-download or a re-pull invalidates). The slot
/// layout is deliberately *not* part of it. Measured on llama-server b11525
/// (`scripts/spikes/kv_slot_compat.py`, see
/// `docs/spikes/2026-10-10-prompt-cache-across-unload.md`): a file saved at one
/// context size loads into a launch with a bigger one, a file saved from one slot
/// restores into another, and a KV layout the server cannot use (a different
/// `--cache-type-k`) is refused with a 400 rather than half-read — the answer
/// after such a refusal is byte-identical to a cold run. So the engine is the
/// judge of what fits, and keying the name on anything more would throw caches
/// away for nothing every time `--fit` resolves a different context.
fn identity(model_path: &Path) -> Option<String> {
  let meta = std::fs::metadata(model_path).ok()?;
  let mut hasher = blake3::Hasher::new();
  hasher.update(model_path.to_string_lossy().as_bytes());
  hasher.update(&meta.len().to_le_bytes());
  let mtime = meta
    .modified()
    .ok()
    .and_then(|t| t.duration_since(UNIX_EPOCH).ok())
    .map(|d| d.as_secs())
    .unwrap_or(0);
  hasher.update(&mtime.to_le_bytes());
  Some(hasher.finalize().to_hex()[..IDENTITY_HEX].to_string())
}

/// `<prefix><slug>-<identity>-slot<N>.bin`. The slug is only for a human reading
/// the directory; the identity digest is what makes the name unique, so the slug
/// is reduced to filename-safe ASCII and truncated.
fn slot_file_name(model_path: &Path, identity: &str, slot: i64) -> String {
  let stem = model_path
    .file_stem()
    .map(|s| s.to_string_lossy())
    .unwrap_or_else(|| "model".into());
  let mut slug = String::with_capacity(stem.len());
  for ch in stem.chars() {
    if slug.len() >= SLUG_MAX {
      break;
    }
    slug.push(if ch.is_ascii_alphanumeric() || ch == '.' || ch == '-' {
      ch
    } else {
      '_'
    });
  }
  if slug.is_empty() {
    slug.push_str("model");
  }
  format!("{FILE_PREFIX}{slug}-{identity}-slot{slot}{FILE_SUFFIX}")
}

/// A saved file, as [`plan_prune`] sees it.
#[derive(Debug, Clone)]
struct SavedFile {
  path: PathBuf,
  bytes: u64,
  mtime: SystemTime,
}

/// Delete expired files, then the oldest ones until what is left fits the cap.
/// The pass runs at launch prep and after every save, so the directory cannot
/// grow past the cap for long and stale files do not wait for a save to go.
fn prune(dir: &Path, cfg: &LlamaCppConfig) -> Pruned {
  let mut files = Vec::new();
  let Ok(entries) = std::fs::read_dir(dir) else {
    return Pruned::default();
  };
  for entry in entries.flatten() {
    let path = entry.path();
    let Some(name) = path.file_name().and_then(|n| n.to_str()) else {
      continue;
    };
    if !is_ours(name) {
      continue;
    }
    let Ok(meta) = entry.metadata() else {
      continue;
    };
    if !meta.is_file() {
      continue;
    }
    files.push(SavedFile {
      bytes: meta.len(),
      mtime: meta.modified().unwrap_or(UNIX_EPOCH),
      path,
    });
  }
  let doomed = plan_prune(
    &files,
    cfg.prompt_cache_max_bytes(),
    cfg.prompt_cache_ttl(),
    SystemTime::now(),
  );
  let mut out = Pruned::default();
  for path in doomed {
    let bytes = files
      .iter()
      .find(|f| f.path == path)
      .map(|f| f.bytes)
      .unwrap_or(0);
    match std::fs::remove_file(&path) {
      Ok(()) => {
        out.removed += 1;
        out.freed_bytes += bytes;
      }
      Err(e) => log::debug!("prompt cache: could not drop {}: {e}", path.display()),
    }
  }
  out
}

/// Which files go: everything older than `ttl` (when one is set), then the
/// oldest-first remainder until what is left fits `max_bytes`. A single file
/// larger than the whole cap is therefore deleted too — keeping it would leave
/// nothing else able to use the budget.
///
/// Pure so the ordering rules are testable without a filesystem.
fn plan_prune(
  files: &[SavedFile],
  max_bytes: u64,
  ttl: Option<Duration>,
  now: SystemTime,
) -> Vec<PathBuf> {
  let expired = |f: &SavedFile| match ttl {
    Some(ttl) => now
      .duration_since(f.mtime)
      .map(|age| age > ttl)
      .unwrap_or(false),
    None => false,
  };
  let mut doomed: Vec<PathBuf> = files
    .iter()
    .filter(|f| expired(f))
    .map(|f| f.path.clone())
    .collect();
  let mut keep: Vec<&SavedFile> = files.iter().filter(|f| !expired(f)).collect();
  // Newest first, so `pop` gives the oldest: the newest conversation is the one
  // a returning user needs.
  keep.sort_by_key(|f| std::cmp::Reverse(f.mtime));
  let mut total: u64 = keep.iter().map(|f| f.bytes).sum();
  while total > max_bytes {
    match keep.pop() {
      Some(f) => {
        total = total.saturating_sub(f.bytes);
        doomed.push(f.path.clone());
      }
      None => break,
    }
  }
  doomed
}

/// Does `name` follow our naming rule? Foreign files are never pruned.
fn is_ours(name: &str) -> bool {
  name.starts_with(FILE_PREFIX) && name.ends_with(FILE_SUFFIX)
}

impl LlamaCppConfig {
  /// The byte cap derived from the configured MiB, saturating so a huge value
  /// cannot overflow into a small one.
  pub(crate) fn prompt_cache_max_bytes(&self) -> u64 {
    self.prompt_cache_max_mb.saturating_mul(1024 * 1024)
  }

  /// How long a saved file stays useful. `0` days means "never expire".
  pub(crate) fn prompt_cache_ttl(&self) -> Option<Duration> {
    if self.prompt_cache_ttl_days == 0 {
      return None;
    }
    Some(Duration::from_secs(
      self.prompt_cache_ttl_days.saturating_mul(86_400),
    ))
  }
}

#[cfg(test)]
mod tests {
  use super::*;

  fn file(name: &str, bytes: u64, age_secs: u64) -> SavedFile {
    SavedFile {
      path: PathBuf::from(name),
      bytes,
      mtime: SystemTime::now() - Duration::from_secs(age_secs),
    }
  }

  /// A per-process scratch directory, so a run left behind by an aborted test
  /// cannot be mistaken for this one's.
  fn scratch(label: &str) -> PathBuf {
    std::env::temp_dir().join(format!(
      "llamastash-prompt-cache-{}-{label}",
      std::process::id()
    ))
  }

  #[test]
  fn identity_follows_the_model_file() {
    let path = Path::new(env!("CARGO_MANIFEST_DIR")).join("Cargo.toml");
    let id = identity(&path).expect("a readable model file");
    assert_eq!(id, identity(&path).unwrap(), "stable for one model file");
    assert_eq!(id.len(), IDENTITY_HEX);
    // A different model file, or one that is not there.
    assert_ne!(id, identity(&path.with_file_name("Cargo.lock")).unwrap());
    assert!(identity(&path.join("missing.gguf")).is_none());
  }

  #[test]
  fn size_and_mtime_of_the_model_enter_the_identity() {
    // A replaced model file must invalidate the caches saved from it: same name,
    // different size, and then same size again with a newer mtime.
    let dir = scratch("identity");
    std::fs::create_dir_all(&dir).unwrap();
    let a = dir.join("a.gguf");
    let b = dir.join("b.gguf");
    std::fs::write(&a, b"one").unwrap();
    std::fs::write(&b, b"two-and-a-bit-longer").unwrap();
    assert_ne!(identity(&a).unwrap(), identity(&b).unwrap());
    let c = dir.join("c.gguf");
    std::fs::write(&c, b"one").unwrap();
    std::fs::write(&a, b"one").unwrap();
    assert_ne!(
      identity(&a).unwrap(),
      identity(&c).unwrap(),
      "a replaced model file invalidates its saved caches"
    );
    let _ = std::fs::remove_dir_all(&dir);
  }

  #[test]
  fn file_names_are_single_component_and_filename_safe() {
    let name = slot_file_name(
      Path::new("/models/unsloth/Qwen3.8-Flash Next (UD)/deepseek-v4:pro_q8.bin"),
      "deadbeefdeadbeef",
      3,
    );
    assert!(name.starts_with(FILE_PREFIX) && name.ends_with(FILE_SUFFIX));
    assert!(name.contains("-slot3.bin"));
    assert_eq!(
      name.chars().filter(|c| *c == '/' || *c == '\\').count(),
      0,
      "llama.cpp validates the save file as one path component: {name}"
    );
    assert!(
      name
        .chars()
        .all(|c| c.is_ascii_alphanumeric() || matches!(c, '.' | '-' | '_' | ':')),
      "unexpected characters in {name}"
    );
    assert!(name.len() < 255);
  }

  #[test]
  fn long_and_empty_stems_still_name_a_file() {
    let long = slot_file_name(
      Path::new("/x/that_is_extremely_long_stem_beyond_the_cap.gguf"),
      "id",
      0,
    );
    assert!(long.len() < FILE_PREFIX.len() + SLUG_MAX + 64);
    assert!(long.ends_with("-id-slot0.bin"));
    assert!(slot_file_name(Path::new(""), "id", 0).starts_with(FILE_PREFIX));
  }

  #[test]
  fn only_our_own_names_are_prunable() {
    assert!(is_ours("llamastash-pc-m.gg-slot0.bin"));
    assert!(!is_ours("someone-elses.bin"));
    assert!(!is_ours("llamastash-pc-m.gg-slot0.txt"));
  }

  #[test]
  fn prune_keeps_the_newest_until_the_budget_fits() {
    let files = vec![
      file("old", 4_000, 300),
      file("middle", 4_000, 200),
      file("new", 4_000, 100),
    ];
    let now = SystemTime::now();
    let doomed = plan_prune(&files, 8_000, None, now);
    assert_eq!(doomed, vec![PathBuf::from("old")]);
    // Under the cap already: nothing goes.
    assert!(plan_prune(&files, 100_000, None, now).is_empty());
    // One file over the whole cap: it goes too, there is no point keeping it.
    assert_eq!(
      plan_prune(&[file("big", 10, 0)], 4, None, now),
      vec![PathBuf::from("big")]
    );
  }

  #[test]
  fn age_limit_runs_before_the_size_cap() {
    let files = vec![
      file("stale-but-big", 1_000, 900),
      file("fresh-small", 10, 10),
    ];
    let now = SystemTime::now();
    let doomed = plan_prune(&files, 10_000, Some(Duration::from_secs(600)), now);
    assert_eq!(doomed, vec![PathBuf::from("stale-but-big")]);
    // Its bytes no longer count against the cap once it is past the age limit.
    assert_eq!(
      plan_prune(&files, 50, Some(Duration::from_secs(600)), now).len(),
      1
    );
  }

  #[test]
  fn prune_touches_only_matching_files_in_the_directory() {
    let dir = scratch("prune");
    std::fs::create_dir_all(&dir).unwrap();
    let ours = dir.join(format!("{FILE_PREFIX}m-deadbeef-slot0{FILE_SUFFIX}"));
    let foreign = dir.join("keep-me.bin");
    std::fs::write(&ours, b"x").unwrap();
    std::fs::write(&foreign, b"x").unwrap();
    let cfg = LlamaCppConfig {
      prompt_cache_max_mb: 0,
      ..Default::default()
    };
    let pruned = prune(&dir, &cfg);
    assert_eq!(pruned.removed, 1);
    assert!(!ours.exists());
    assert!(foreign.exists());
    let _ = std::fs::remove_dir_all(&dir);
  }

  #[test]
  fn slot_rows_parse_with_and_without_debug_fields() {
    // llama.cpp's `GET /slots` without `debug=1`.
    let plain: Vec<Slot> = serde_json::from_str(
      r#"[{"id":0,"n_ctx":102400,"speculative":false,"is_processing":false}]"#,
    )
    .unwrap();
    assert_eq!(plain.len(), 1);
    assert!(plain[0].may_have_prompt(), "an unknown size is worth a try");
    assert!(plain[0].open_for_restore());
    // With `debug=1`, an idle slot with a prompt saves and takes no restore.
    let debug: Vec<Slot> = serde_json::from_str(
      r#"[{"id":0,"n_ctx":8192,"is_processing":false,"n_prompt_tokens":7,"n_prompt_tokens_cache":7},
          {"id":1,"n_ctx":8192,"is_processing":true,"n_prompt_tokens":3},
          {"id":2,"n_ctx":8192,"is_processing":false,"n_prompt_tokens":0}]"#,
    )
    .unwrap();
    assert!(debug[0].may_have_prompt() && !debug[0].open_for_restore());
    assert!(debug[1].may_have_prompt());
    assert!(debug[1].is_processing, "and is skipped for that alone");
    assert!(!debug[2].may_have_prompt() && debug[2].open_for_restore());
    // The two filters the callers use.
    let savable: Vec<i64> = debug
      .iter()
      .filter(|s| !s.is_processing && s.may_have_prompt())
      .map(|s| s.id)
      .collect();
    let restorable: Vec<i64> = debug
      .iter()
      .filter(|s| !s.is_processing && s.open_for_restore())
      .map(|s| s.id)
      .collect();
    assert_eq!(savable, vec![0]);
    assert_eq!(restorable, vec![2]);
  }

  #[test]
  fn action_responses_parse() {
    let save: SlotAction =
      serde_json::from_str(r#"{"id_slot":0,"filename":"x.bin","n_saved":1745,"n_written":14309796,"timings":{"save_ms":49.8}}"#)
        .unwrap();
    assert_eq!(save.n_saved, Some(1745));
    assert_eq!(save.n_written, Some(14309796));
    let restore: SlotAction = serde_json::from_str(
      r#"{"n_restored":1745,"n_read":14309796,"timings":{"restore_ms":42.9}}"#,
    )
    .unwrap();
    assert_eq!(restore.n_restored, Some(1745));
    assert_eq!(restore.n_written, None);
  }

  /// One canned HTTP answer per connection, enough to pin the wire contract.
  async fn spawn_fake(status: u16, body: &'static str) -> u16 {
    use tokio::io::{AsyncReadExt, AsyncWriteExt};
    use tokio::net::TcpListener;
    let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let port = listener.local_addr().unwrap().port();
    tokio::spawn(async move {
      while let Ok((mut sock, _)) = listener.accept().await {
        let mut buf = [0u8; 4096];
        let _ = sock.read(&mut buf).await;
        let resp = format!(
          "HTTP/1.1 {status} X\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{body}",
          body.len()
        );
        let _ = sock.write_all(resp.as_bytes()).await;
      }
    });
    port
  }

  #[tokio::test]
  async fn a_successful_action_names_the_slot_and_the_file() {
    let port = spawn_fake(200, r#"{"n_saved":7,"n_written":8192}"#).await;
    let res = slot_action(port, "save", 3, "llamastash-pc-m-deadbeef-slot3.bin")
      .await
      .expect("the round trip succeeds");
    assert_eq!(res.n_saved, Some(7));
    assert_eq!(res.n_written, Some(8192));
  }

  #[tokio::test]
  async fn a_refusal_is_about_the_file_not_about_the_moment() {
    // Only a refusal may justify deleting the saved file, so the two failure
    // kinds have to stay distinguishable.
    let port = spawn_fake(
      400,
      r#"{"error":{"code":400,"message":"Unable to restore slot: invalid slot save file","type":"invalid_request_error"}}"#,
    )
    .await;
    let err = slot_action(port, "restore", 0, "llamastash-pc-m-deadbeef-slot0.bin")
      .await
      .unwrap_err();
    assert!(matches!(err, ActionError::Rejected(_)));
    assert!(err.message().contains("invalid slot save file"));
  }

  #[tokio::test]
  async fn an_unanswered_action_is_not_a_refusal() {
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let port = listener.local_addr().unwrap().port();
    drop(listener);
    let err = slot_action(port, "restore", 0, "llamastash-pc-m-deadbeef-slot0.bin")
      .await
      .unwrap_err();
    assert!(matches!(err, ActionError::Unreachable(_)));
  }

  #[test]
  fn config_derives_a_byte_cap_and_an_age_limit() {
    let cfg = LlamaCppConfig::default();
    assert_eq!(
      cfg.prompt_cache_max_bytes(),
      cfg.prompt_cache_max_mb * 1024 * 1024
    );
    assert_eq!(
      cfg.prompt_cache_ttl(),
      Some(Duration::from_secs(7 * 86_400))
    );
    let off = LlamaCppConfig {
      prompt_cache_ttl_days: 0,
      prompt_cache_max_mb: u64::MAX,
      ..Default::default()
    };
    assert_eq!(off.prompt_cache_ttl(), None);
    assert!(off.prompt_cache_max_bytes() == u64::MAX, "no overflow");
  }

  #[test]
  fn a_disabled_feature_finds_no_directory() {
    let cfg = LlamaCppConfig {
      prompt_cache: false,
      ..Default::default()
    };
    assert!(prepare_dir(&cfg).is_none());
  }
}
