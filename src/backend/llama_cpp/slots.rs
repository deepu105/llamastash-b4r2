//! Disk-persisted prompt (KV) cache for llama.cpp slots.
//!
//! When the proxy's eviction machinery stops a llama.cpp launch mid-conversation,
//! the next request would otherwise reprocess the whole prompt. `llama-server`
//! can write a slot's KV cache to disk (`--slot-save-path` plus
//! `POST /slots/{id_slot}?action=save|restore`) and read it back into a fresh
//! launch of the same model at the same layout: measured 0.4-2.7 s to save and
//! 0.4-1.3 s to restore against 50-3200 s of plain reprocessing at a 100k-token
//! prompt ([`docs/spikes/2026-10-10-slot-kv-cache-persistence.md`](../../../docs/spikes/2026-10-10-slot-kv-cache-persistence.md)).
//!
//! The save side runs before an eviction stop ([`evict_save`], behind
//! [`Backend::before_evict_stop`](crate::backend::Backend::before_evict_stop));
//! the restore side runs on a fresh launch after the engine answers but before
//! Ready is published ([`restore_on_ready`], behind
//! [`Backend::on_ready`](crate::backend::Backend::on_ready)), so a request that
//! waited on auto-start still finds the cache in the slot. Files live under
//! `cache_dir()/slot-cache/`, named by a fingerprint of the launch inputs that
//! decide the KV layout; `backend.llamacpp.slot_cache_max_bytes` and
//! `slot_cache_max_age_days` cap and expire the pile.

use std::path::{Path, PathBuf};
use std::time::Duration;

use crate::daemon::actuals::Actuals;
use crate::daemon::context::MethodContext;
use crate::daemon::registry::LaunchId;
use crate::launch::params::LaunchParams;

/// Prompts below this many cached tokens are not worth a save: the file is
/// tens of MB and reprocessing is seconds.
const MIN_PROMPT_TOKENS: u64 = 256;

const LIST_TIMEOUT: Duration = Duration::from_secs(3);
const SAVE_TIMEOUT: Duration = Duration::from_secs(30);
const RESTORE_TIMEOUT: Duration = Duration::from_secs(30);

/// Create (if needed) and return the shared slot-cache directory, or `None`
/// when no cache dir resolves or the directory cannot be created. Mode `0700`:
/// a KV dump replays a conversation's bytes, so it is user-private like
/// `runtime.json`.
pub(super) fn ensure_cache_dir() -> Option<PathBuf> {
  let dir = crate::util::paths::cache_dir()?.join("slot-cache");
  match std::fs::create_dir_all(&dir) {
    Ok(()) => {
      #[cfg(unix)]
      {
        use std::os::unix::fs::PermissionsExt;
        let _ = std::fs::set_permissions(&dir, std::fs::Permissions::from_mode(0o700));
      }
      Some(dir)
    }
    Err(e) => {
      log::debug!("slot-cache: cannot create {}: {e}", dir.display());
      None
    }
  }
}

/// The fingerprint half of a save filename: a hash of the launch inputs that
/// decide the KV cache layout (model, resolved context window, K/V cache
/// types, flash-attn). A relaunched save and restore compute the same value
/// only when a dump written by the old layout could load into the new one —
/// a changed ctx or cache type is a miss, not a corrupt restore. llama.cpp
/// validates the file itself on load, so a wrong guess fails closed.
pub(crate) fn fingerprint(params: &LaunchParams, actuals: &Actuals) -> String {
  let ctx = actuals
    .resolved_ctx
    .or(params.ctx)
    .map(|c| c.to_string())
    .unwrap_or_else(|| "auto".to_string());
  let knob = |name: &str| {
    params
      .knobs
      .text_by_name(name)
      .unwrap_or_else(|| "default".to_string())
  };
  let input = format!(
    "{}\u{0}{}\u{0}{}\u{0}{}\u{0}{}",
    params.model_path.display(),
    ctx,
    knob("cache-type-k"),
    knob("cache-type-v"),
    knob("flash-attn"),
  );
  blake3::hash(input.as_bytes()).to_hex().as_str()[..16].to_string()
}

/// The engine-side filename for one slot's dump: `<fingerprint>_<slot_id>.bin`.
/// Flat and shell-safe — the engine rejects names with path separators.
fn save_filename(fp: &str, slot_id: u32) -> String {
  format!("{fp}_{slot_id}.bin")
}

/// Inverse of [`save_filename`] for one fingerprint: the slot id this file
/// restores into, or `None` when the name is not a dump of `fp`.
fn parse_restore_name(name: &str, fp: &str) -> Option<u32> {
  name
    .strip_prefix(fp)?
    .strip_prefix('_')?
    .strip_suffix(".bin")?
    .parse()
    .ok()
}

/// One engine slot as reported by `GET /slots`.
#[derive(Debug, Clone, PartialEq)]
pub(crate) struct EngineSlot {
  pub id: u32,
  /// A missing `is_processing` is read as busy so a save is never issued
  /// against a slot that may be mid-generation.
  pub processing: bool,
  /// Prompt tokens the slot still holds cached from its last task; `0` for a
  /// slot that never served one (the engine omits the field).
  pub prompt_tokens: u64,
}

fn parse_slots(body: &serde_json::Value) -> Vec<EngineSlot> {
  let Some(arr) = body.as_array() else {
    return Vec::new();
  };
  arr
    .iter()
    .filter_map(|s| {
      Some(EngineSlot {
        id: s.get("id")?.as_u64()? as u32,
        processing: s
          .get("is_processing")
          .and_then(|v| v.as_bool())
          .unwrap_or(true),
        prompt_tokens: s
          .get("n_prompt_tokens")
          .and_then(|v| v.as_u64())
          .unwrap_or(0),
      })
    })
    .collect()
}

/// Slots worth saving: idle and holding a prompt big enough to matter.
fn save_candidates(slots: &[EngineSlot]) -> Vec<u32> {
  slots
    .iter()
    .filter(|s| !s.processing && s.prompt_tokens >= MIN_PROMPT_TOKENS)
    .map(|s| s.id)
    .collect()
}

/// One slot-cache file as found on disk, for [`entries_to_delete`].
pub(crate) struct CacheEntry {
  pub name: String,
  pub bytes: u64,
  pub age: Duration,
}

/// Which slot-cache files to drop: first everything at least `max_age` old
/// (`None` = no age limit), then the oldest survivors until the total fits
/// `max_bytes`. An oversized newest file evicts everything older by design —
/// the point of the cap is to bound disk, and a dump that does not fit is
/// already useless for the next eviction of anything else.
fn entries_to_delete(
  entries: &[CacheEntry],
  max_bytes: u64,
  max_age: Option<Duration>,
) -> Vec<String> {
  let expired = |e: &&CacheEntry| max_age.is_some_and(|a| e.age >= a);
  let mut doomed: Vec<(&CacheEntry, bool)> = entries.iter().map(|e| (e, expired(&e))).collect();
  let mut total: u64 = doomed
    .iter()
    .filter(|(_, keep)| !*keep)
    .map(|(e, _)| e.bytes)
    .sum();
  if total > max_bytes {
    let mut survivors: Vec<&CacheEntry> = doomed
      .iter()
      .filter(|(_, keep)| !*keep)
      .map(|(e, _)| *e)
      .collect();
    survivors.sort_by_key(|e| std::cmp::Reverse(e.age));
    for e in survivors {
      if total <= max_bytes {
        break;
      }
      total = total.saturating_sub(e.bytes);
      if let Some(slot) = doomed.iter_mut().find(|(d, _)| d.name == e.name) {
        slot.1 = true;
      }
    }
  }
  doomed
    .into_iter()
    .filter(|(_, gone)| *gone)
    .map(|(e, _)| e.name.clone())
    .collect()
}

/// Apply [`entries_to_delete`] to a real directory. Best-effort: every I/O
/// failure is logged at debug and skipped (files are a cache, not state).
fn prune_dir(dir: &Path, max_bytes: u64, max_age: Option<Duration>) {
  let rd = match std::fs::read_dir(dir) {
    Ok(rd) => rd,
    Err(e) => {
      log::debug!("slot-cache: cannot list {}: {e}", dir.display());
      return;
    }
  };
  let mut entries = Vec::new();
  for entry in rd.flatten() {
    let name = entry.file_name().to_string_lossy().into_owned();
    if !name.ends_with(".bin") {
      continue;
    }
    let Ok(meta) = entry.metadata() else { continue };
    if !meta.is_file() {
      continue;
    }
    let age = meta
      .modified()
      .ok()
      .map(|m| m.elapsed().unwrap_or(Duration::ZERO))
      .unwrap_or(Duration::ZERO);
    entries.push(CacheEntry {
      name,
      bytes: meta.len(),
      age,
    });
  }
  for name in entries_to_delete(&entries, max_bytes, max_age) {
    if let Err(e) = std::fs::remove_file(dir.join(&name)) {
      log::debug!("slot-cache: prune failed for {name}: {e}");
    } else {
      log::debug!("slot-cache: pruned {name}");
    }
  }
}

fn age_limit(max_age_days: u32) -> Option<Duration> {
  if max_age_days == 0 {
    None
  } else {
    Some(Duration::from_secs(u64::from(max_age_days) * 86_400))
  }
}

/// `GET /slots` on the engine port. Capped body like the orphan probe.
async fn get_slots(port: u16, timeout: Duration) -> std::io::Result<Vec<EngineSlot>> {
  let client = reqwest::Client::builder()
    .timeout(timeout)
    .build()
    .map_err(|e| std::io::Error::other(e.to_string()))?;
  let resp = client
    .get(format!("http://127.0.0.1:{port}/slots"))
    .send()
    .await
    .map_err(|e| std::io::Error::other(e.to_string()))?;
  if !resp.status().is_success() {
    return Err(std::io::Error::other(format!(
      "engine returned {}",
      resp.status().as_u16()
    )));
  }
  let body = resp
    .bytes()
    .await
    .map_err(|e| std::io::Error::other(e.to_string()))?
    .to_vec();
  const MAX_BODY: usize = 1024 * 1024;
  if body.len() > MAX_BODY {
    return Err(std::io::Error::other("slot list too large"));
  }
  let json: serde_json::Value =
    serde_json::from_slice(&body).map_err(|e| std::io::Error::other(e.to_string()))?;
  Ok(parse_slots(&json))
}

/// `POST /slots/{id}?action=<action>` with `{"filename"}`. Returns
/// `(status, parsed body)`; the caller decides what a non-success means.
async fn post_slot_action(
  port: u16,
  slot_id: u32,
  action: &str,
  filename: &str,
  timeout: Duration,
) -> std::io::Result<(u16, serde_json::Value)> {
  let client = reqwest::Client::builder()
    .timeout(timeout)
    .build()
    .map_err(|e| std::io::Error::other(e.to_string()))?;
  let resp = client
    .post(format!(
      "http://127.0.0.1:{port}/slots/{slot_id}?action={action}"
    ))
    .json(&serde_json::json!({ "filename": filename }))
    .send()
    .await
    .map_err(|e| std::io::Error::other(e.to_string()))?;
  let status = resp.status().as_u16();
  let body = resp
    .bytes()
    .await
    .map_err(|e| std::io::Error::other(e.to_string()))?
    .to_vec();
  let json = serde_json::from_slice(&body).unwrap_or_else(|_| serde_json::json!({}));
  Ok((status, json))
}

/// Eviction stop is about to SIGTERM this launch: give every idle slot holding
/// a real conversation a moment to dump its KV cache to disk. Best-effort and
/// bounded by the request timeouts — the stop proceeds whatever happens.
pub(super) async fn evict_save(ctx: &MethodContext, launch_id: &LaunchId) {
  let cfg = &ctx.backend.llamacpp;
  if cfg.slot_cache_max_bytes == 0 {
    return;
  }
  let snap = ctx.state.snapshot().await;
  let Some(row) = snap
    .running
    .iter()
    .find(|r| r.launch_id.as_ref() == Some(launch_id))
  else {
    return;
  };
  // Only a launch that was actually given `--slot-save-path` can save.
  let Some(dir) = row
    .params
    .launch_config
    .get(crate::backend::llama_cpp::LLAMACPP_KNOB_SLOT_SAVE_DIR)
  else {
    log::debug!(
      "slot-cache: {} launched without a slot save dir; nothing to save",
      launch_id.as_str()
    );
    return;
  };
  let dir = PathBuf::from(dir);
  let slots = match get_slots(row.port, LIST_TIMEOUT).await {
    Ok(slots) => slots,
    Err(e) => {
      log::warn!(
        "slot-cache: {} ({}) slot list failed: {e}",
        launch_id.as_str(),
        row.port
      );
      return;
    }
  };
  let candidates = save_candidates(&slots);
  if candidates.is_empty() {
    log::debug!(
      "slot-cache: {} has no slot worth saving",
      launch_id.as_str()
    );
    return;
  }
  prune_dir(
    &dir,
    cfg.slot_cache_max_bytes,
    age_limit(cfg.slot_cache_max_age_days),
  );
  let fp = fingerprint(&row.params, &row.actuals);
  for id in candidates {
    let name = save_filename(&fp, id);
    match post_slot_action(row.port, id, "save", &name, SAVE_TIMEOUT).await {
      Ok((status, body)) if (200..300).contains(&status) => log::info!(
        "slot-cache: saved {launch} slot {id} to {name} ({} prompt tokens, {} bytes)",
        body.get("n_saved").and_then(|v| v.as_u64()).unwrap_or(0),
        body.get("n_written").and_then(|v| v.as_u64()).unwrap_or(0),
        launch = launch_id.as_str(),
      ),
      Ok((status, body)) => {
        log::warn!("slot-cache: save of slot {id} returned {status}: {body}")
      }
      Err(e) => log::warn!("slot-cache: save of slot {id} failed: {e}"),
    }
  }
  prune_dir(
    &dir,
    cfg.slot_cache_max_bytes,
    age_limit(cfg.slot_cache_max_age_days),
  );
}

/// Fresh launch of a model: restore every dump this fingerprint owns into the
/// matching slot, so the first request that arrives finds its prompt prefix
/// already cached. Runs before Ready is published; a miss (no file) costs one
/// directory read. Files are removed once restored — the next eviction writes
/// a newer one.
pub(super) async fn restore_on_ready(port: u16, params: &LaunchParams, actuals: &Actuals) {
  let Some(dir) = params
    .launch_config
    .get(crate::backend::llama_cpp::LLAMACPP_KNOB_SLOT_SAVE_DIR)
  else {
    return;
  };
  let dir = PathBuf::from(dir);
  let fp = fingerprint(params, actuals);
  let Ok(rd) = std::fs::read_dir(&dir) else {
    return;
  };
  let mut wanted: Vec<(u32, String)> = rd
    .flatten()
    .filter_map(|e| {
      let name = e.file_name().to_string_lossy().into_owned();
      parse_restore_name(&name, &fp).map(|id| (id, name))
    })
    .collect();
  if wanted.is_empty() {
    return;
  }
  wanted.sort();
  let live: Vec<u32> = match get_slots(port, LIST_TIMEOUT).await {
    Ok(slots) => slots.iter().map(|s| s.id).collect(),
    Err(e) => {
      log::debug!("slot-cache: slot list on port {port} failed: {e}");
      return;
    }
  };
  for (id, name) in wanted {
    // A dump for a slot id this launch does not have (lower `--parallel`
    // now) has nowhere to go; leave it for a launch that does.
    if !live.contains(&id) {
      continue;
    }
    match post_slot_action(port, id, "restore", &name, RESTORE_TIMEOUT).await {
      Ok((status, body)) if (200..300).contains(&status) => {
        log::info!(
          "slot-cache: restored slot {id} from {name} ({} tokens, {} bytes) into port {port}",
          body.get("n_restored").and_then(|v| v.as_u64()).unwrap_or(0),
          body.get("n_read").and_then(|v| v.as_u64()).unwrap_or(0),
        );
        if let Err(e) = std::fs::remove_file(dir.join(&name)) {
          log::debug!("slot-cache: cannot remove restored {name}: {e}");
        }
      }
      Ok((status, body)) => {
        log::warn!("slot-cache: restore of slot {id} from {name} returned {status}: {body}")
      }
      Err(e) => log::warn!("slot-cache: restore of slot {id} from {name} failed: {e}"),
    }
  }
}

/// Boot cleanup: drop dumps left by an earlier session that have already aged
/// out, before the first eviction pays for the same scan.
pub(super) fn prune_at_boot(max_bytes: u64, max_age_days: u32) {
  if max_bytes == 0 {
    return;
  }
  let Some(root) = crate::util::paths::cache_dir() else {
    return;
  };
  prune_dir(&root.join("slot-cache"), max_bytes, age_limit(max_age_days));
}

#[cfg(test)]
mod tests {
  use super::*;
  use crate::launch::mode::LaunchMode;
  use std::path::PathBuf;

  fn secs(n: u64) -> Duration {
    Duration::from_secs(n)
  }

  fn params(path: &str) -> LaunchParams {
    let mut p = LaunchParams::new(PathBuf::from(path), LaunchMode::Chat);
    p.ctx = Some(8192);
    p
  }

  fn actuals() -> Actuals {
    Actuals {
      resolved_ctx: Some(8192),
      ..Default::default()
    }
  }

  #[test]
  fn fingerprint_is_stable_and_covers_layout_inputs() {
    let base = fingerprint(&params("/m/a.gguf"), &actuals());
    assert_eq!(base.len(), 16);
    assert_eq!(base, fingerprint(&params("/m/a.gguf"), &actuals()));
    // Model path, resolved ctx, and both cache types all decide the KV
    // layout, so each must change the fingerprint.
    assert_ne!(base, fingerprint(&params("/m/b.gguf"), &actuals()));
    // Without engine actuals the requested ctx is the layout input; with
    // them the resolved one wins (covered by the next test).
    let mut other_ctx = params("/m/a.gguf");
    other_ctx.ctx = Some(4096);
    let no_actuals = Actuals::default();
    assert_ne!(
      fingerprint(&params("/m/a.gguf"), &no_actuals),
      fingerprint(&other_ctx, &no_actuals)
    );
    let mut kv = params("/m/a.gguf");
    kv.knobs
      .set_by_name("cache-type-k", "q8_0")
      .then_some(())
      .expect("known knob");
    assert_ne!(base, fingerprint(&kv, &actuals()));
  }

  #[test]
  fn fingerprint_prefers_engine_actuals_over_pinned_params() {
    // `--fit` can land a different window than `params.ctx` requested; the
    // engine's resolved value is what the on-disk layout follows.
    let mut pinned = params("/m/a.gguf");
    pinned.ctx = Some(16384);
    assert_eq!(
      fingerprint(&pinned, &actuals()),
      fingerprint(&params("/m/a.gguf"), &actuals())
    );
    let no_actuals = Actuals::default();
    assert_ne!(
      fingerprint(&pinned, &no_actuals),
      fingerprint(&params("/m/a.gguf"), &no_actuals)
    );
  }

  #[test]
  fn save_filenames_round_trip_per_fingerprint() {
    let name = save_filename("abcd1234abcd1234", 3);
    assert_eq!(name, "abcd1234abcd1234_3.bin");
    assert_eq!(parse_restore_name(&name, "abcd1234abcd1234"), Some(3));
    assert_eq!(parse_restore_name(&name, "ffff0000ffff0000"), None);
    assert_eq!(
      parse_restore_name("abcd1234abcd1234_x.bin", "abcd1234abcd1234"),
      None
    );
    assert_eq!(parse_restore_name("stray.txt", "abcd1234abcd1234"), None);
  }

  #[test]
  fn parse_slots_handles_the_engine_shape() {
    // Shape taken from upstream tools/server: a slot that served a prompt
    // carries n_prompt_tokens, one that never did omits it.
    let body = serde_json::json!([
      { "id": 0, "n_ctx": 8192, "speculative": false, "is_processing": false,
        "n_prompt_tokens": 5000, "n_prompt_tokens_processed": 5000 },
      { "id": 1, "n_ctx": 8192, "speculative": false, "is_processing": true },
      { "id": 2, "n_ctx": 8192, "speculative": false, "is_processing": false,
        "n_prompt_tokens": 10 },
    ]);
    assert_eq!(
      parse_slots(&body),
      vec![
        EngineSlot {
          id: 0,
          processing: false,
          prompt_tokens: 5000
        },
        EngineSlot {
          id: 1,
          processing: true,
          prompt_tokens: 0
        },
        EngineSlot {
          id: 2,
          processing: false,
          prompt_tokens: 10
        },
      ]
    );
    // A non-array body (proxy error page, empty reply) yields no slots.
    assert_eq!(parse_slots(&serde_json::json!({})), Vec::new());
  }

  #[test]
  fn save_candidates_skip_busy_and_tiny_slots() {
    let slots = vec![
      EngineSlot {
        id: 0,
        processing: false,
        prompt_tokens: MIN_PROMPT_TOKENS,
      },
      EngineSlot {
        id: 1,
        processing: true,
        prompt_tokens: 9000,
      },
      EngineSlot {
        id: 2,
        processing: false,
        prompt_tokens: MIN_PROMPT_TOKENS - 1,
      },
    ];
    assert_eq!(save_candidates(&slots), vec![0]);
  }

  fn entry(name: &str, bytes: u64, age: u64) -> CacheEntry {
    CacheEntry {
      name: name.into(),
      bytes,
      age: secs(age),
    }
  }

  #[test]
  fn prune_under_the_cap_keeps_everything() {
    let entries = vec![entry("a.bin", 100, 10), entry("b.bin", 100, 1)];
    assert!(entries_to_delete(&entries, 1000, Some(secs(60))).is_empty());
  }

  #[test]
  fn prune_drops_expired_files() {
    let entries = vec![entry("old.bin", 10, 3600), entry("new.bin", 10, 10)];
    assert_eq!(
      entries_to_delete(&entries, 1000, Some(secs(60))),
      vec!["old.bin"]
    );
    // A `None` age limit keeps the old file (bytes still decide).
    assert!(entries_to_delete(&entries, 1000, None).is_empty());
  }

  #[test]
  fn prune_evicts_oldest_first_until_under_cap() {
    let entries = vec![
      entry("oldest.bin", 60, 300),
      entry("middle.bin", 60, 200),
      entry("newest.bin", 60, 100),
    ];
    // 180 bytes over a 120 cap: the oldest goes first, then the middle one.
    assert_eq!(entries_to_delete(&entries, 120, None), vec!["oldest.bin"]);
    assert_eq!(
      entries_to_delete(&entries, 100, None),
      vec!["oldest.bin", "middle.bin"]
    );
    // Even an impossible cap empties the pile — bounded disk beats stale dumps.
    assert_eq!(entries_to_delete(&entries, 0, None).len(), 3,);
  }

  #[test]
  fn prune_expired_files_still_count_toward_the_cap_while_kept() {
    // Expired and over-cap interact: expired files go either way; the cap
    // counts only survivors.
    let entries = vec![entry("expired.bin", 90, 999), entry("keep.bin", 90, 1)];
    assert_eq!(
      entries_to_delete(&entries, 100, Some(secs(60))),
      vec!["expired.bin"]
    );
  }

  #[test]
  fn age_limit_maps_zero_days_to_no_limit() {
    assert_eq!(age_limit(0), None);
    assert_eq!(age_limit(3), Some(secs(3 * 86_400)));
  }

  /// The serde defaults and the `Default` impl must not drift — one reads the
  /// other's absence.
  #[test]
  fn slot_cache_config_defaults_agree() {
    let empty: crate::backend::llama_cpp::LlamaCppConfig = serde_json::from_str("{}").unwrap();
    assert_eq!(empty, crate::backend::llama_cpp::LlamaCppConfig::default());
    assert_eq!(empty.slot_cache_max_bytes, 16 * 1024 * 1024 * 1024);
    assert_eq!(empty.slot_cache_max_age_days, 3);
  }
}
