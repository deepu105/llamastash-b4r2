//! Cross-restart prompt cache for llama.cpp launches.
//!
//! When the proxy evicts a launch mid-conversation, the next request would
//! otherwise reprocess the whole prompt. llama-server can write a slot's KV
//! cache to disk (`POST /slots/{id}?action=save`) and read it back into a
//! freshly launched server (`action=restore`), given `--slot-save-path` at
//! startup (checked against llama.cpp b11525: `tools/server/server-context.cpp`).
//! LlamaStash drives both ends: a save pass runs through `Backend::before_stop`
//! just before an eviction stops the launch, and a restore pass runs through
//! `Backend::after_ready` on the relaunched server before it is handed
//! traffic. A restored slot keeps its old prompt tokens, so the returning
//! conversation is matched by slot id and by llama.cpp's own prefix reuse —
//! a prompt that does not extend the restored one simply reprocesses.
//!
//! Files live under `cache_dir/prompt-cache/<model-stem>-<hash>/`, one per
//! (model, slot, save). Restores consume their file (delete after the attempt);
//! `prune_to_cap` enforces a global size cap LRU-first so disk use stays
//! bounded even for files nothing restores.
//!
//! Everything here is best-effort: a failed save or restore costs the cache,
//! never the launch. Transport is the same hand-rolled loopback HTTP as
//! `super::actuals` — no dep for two endpoints.

use std::path::{Path, PathBuf};
use std::time::{Duration, SystemTime, UNIX_EPOCH};

use tokio::io::{AsyncReadExt, AsyncWriteExt};
use tokio::net::TcpStream;

/// Key in the boot-probed `Server::caps` map: the build advertises
/// `--slot-save-path` in its own `--help`.
pub const CAP_SLOT_SAVE_PATH: &str = "slot_save_path";

/// Whole-budget for the save pass that runs inline before an eviction stop.
/// A save is a KV dump to local disk; past this the eviction proceeds
/// uncached rather than stalling the sweep or the blocked request.
const SAVE_BUDGET: Duration = Duration::from_secs(60);

/// Whole-budget for the restore pass that runs between the readiness probe
/// and the `Ready` transition. Restore reads the KV back at RAM speed; the
/// budget only guards a hung server, which must not hold `Ready` hostage.
const RESTORE_BUDGET: Duration = Duration::from_secs(120);

/// Per-request ceiling for one slots call. The budgets above are the real
/// bound; this one only stops a dead socket from stretching them.
const HTTP_TIMEOUT: Duration = Duration::from_secs(30);

/// One row of `GET /slots`, narrowed to what save/restore needs (shape
/// verified live on llama.cpp b11525).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SlotInfo {
  pub id: u32,
  pub is_processing: bool,
  pub n_prompt_tokens: u64,
}

/// Parse a `GET /slots` body into [`SlotInfo`] rows. Pure so tests drive it
/// with captured payloads.
pub fn parse_slots(body: &serde_json::Value) -> Vec<SlotInfo> {
  body
    .as_array()
    .map(|rows| {
      rows
        .iter()
        .filter_map(|r| {
          Some(SlotInfo {
            id: r.get("id")?.as_u64()? as u32,
            is_processing: r.get("is_processing")?.as_bool()?,
            n_prompt_tokens: r.get("n_prompt_tokens")?.as_u64()?,
          })
        })
        .collect()
    })
    .unwrap_or_default()
}

/// Slot ids present on a running server. Rows for never-used slots can
/// omit everything but `id`, so a liveness check must not require the
/// other fields — dropping an id here silently discards a save file.
pub fn live_slot_ids(body: &serde_json::Value) -> Vec<u32> {
  body
    .as_array()
    .map(|rows| {
      rows
        .iter()
        .filter_map(|r| r.get("id").and_then(|id| id.as_u64()))
        .map(|id| id as u32)
        .collect()
    })
    .unwrap_or_default()
}

/// Per-model cache directory: `cache_dir/prompt-cache/<stem>-<hash>/`. Keyed
/// on the resolved model path so relaunches of the same file find the same
/// directory; the stem keeps listings readable, the hash keeps two models
/// that share a name (or a stem too long to keep whole) apart.
pub fn model_dir(model_path: &Path) -> Option<PathBuf> {
  let root = crate::util::paths::cache_dir()?;
  let hash = blake3::hash(model_path.to_string_lossy().as_bytes());
  let stem: String = model_path
    .file_stem()
    .map(|s| s.to_string_lossy().into_owned())
    .unwrap_or_else(|| "model".to_string())
    .chars()
    .map(|c| {
      if c.is_ascii_alphanumeric() || matches!(c, '.' | '_' | '-') {
        c
      } else {
        '-'
      }
    })
    .take(40)
    .collect();
  Some(
    root
      .join("prompt-cache")
      .join(format!("{stem}-{}", &hash.to_hex()[..12])),
  )
}

/// Create the per-model directory (llama-server refuses a
/// `--slot-save-path` that is not an existing directory) with private mode:
/// slot files carry the conversation's token ids in plain view.
pub fn prepare_dir(model_path: &Path) -> Option<PathBuf> {
  let dir = model_dir(model_path)?;
  if let Err(e) = std::fs::create_dir_all(&dir) {
    log::warn!("prompt cache: cannot create {}: {e}", dir.display());
    return None;
  }
  #[cfg(unix)]
  {
    use std::os::unix::fs::PermissionsExt;
    if let Err(e) = std::fs::set_permissions(&dir, std::fs::Permissions::from_mode(0o700)) {
      log::warn!("prompt cache: cannot chmod {}: {e}", dir.display());
    }
  }
  Some(dir)
}

/// `slot{N}-{epoch-seconds}.bin` — flat filename (llama.cpp's
/// `fs_validate_filename` rejects separators), sortable timestamp so "latest
/// per slot" is a string comparison and concurrent saves never collide.
fn save_filename(slot: u32, now: SystemTime) -> String {
  let secs = now
    .duration_since(UNIX_EPOCH)
    .map(|d| d.as_secs())
    .unwrap_or_default();
  format!("slot{slot}-{secs}.bin")
}

/// `(slot, stamp)` of a cache filename; anything else is foreign and ignored.
pub fn parse_saved_file(name: &str) -> Option<(u32, u64)> {
  let rest = name.strip_prefix("slot")?;
  let (slot, stamp) = rest.split_once('-')?;
  let stamp = stamp.strip_suffix(".bin")?;
  Some((slot.parse().ok()?, stamp.parse().ok()?))
}

/// Newest file per slot id — the one worth restoring. Older saves of the
/// same slot are superseded; ones from a concurrent launch of the same
/// model carry their own stamp and never overwrite.
pub fn latest_per_slot(dir: &Path) -> std::collections::BTreeMap<u32, (u64, PathBuf)> {
  let mut out: std::collections::BTreeMap<u32, (u64, PathBuf)> = Default::default();
  let Ok(entries) = std::fs::read_dir(dir) else {
    return out;
  };
  for entry in entries.flatten() {
    let name = entry.file_name().to_string_lossy().into_owned();
    let Some((slot, stamp)) = parse_saved_file(&name) else {
      continue;
    };
    match out.get(&slot) {
      Some((best, _)) if *best >= stamp => {}
      _ => {
        out.insert(slot, (stamp, entry.path()));
      }
    }
  }
  out
}

/// Enforce the global cap over every model directory under `root`
/// (`cache_dir/prompt-cache/` in production): newest files are kept until
/// they fit the cap, older ones are deleted oldest-first. A single newest
/// file above the cap is kept (it is worth more than nothing). Returns the
/// deleted count.
pub fn prune_to_cap(root: &Path, max_bytes: u64) -> usize {
  let mut files: Vec<(std::time::SystemTime, u64, PathBuf)> = Vec::new();
  let Ok(models) = std::fs::read_dir(root) else {
    return 0;
  };
  for model in models.flatten() {
    let Ok(entries) = std::fs::read_dir(model.path()) else {
      continue;
    };
    for entry in entries.flatten() {
      let name = entry.file_name().to_string_lossy().into_owned();
      if parse_saved_file(&name).is_none() {
        continue;
      }
      let Ok(meta) = entry.metadata() else { continue };
      let mtime = meta.modified().unwrap_or(UNIX_EPOCH);
      files.push((mtime, meta.len(), entry.path()));
    }
  }
  let mut total: u64 = files.iter().map(|(_, len, _)| *len).sum();
  if total <= max_bytes {
    return 0;
  }
  // Newest first; the newest save is never deleted (it is the one an ongoing
  // conversation may restore), the rest go oldest-first while over the cap.
  files.sort_by_key(|f| std::cmp::Reverse(f.0));
  let mut rest = files.into_iter();
  let _newest = rest.next();
  let mut deleted = 0usize;
  for (_, len, path) in rest.rev() {
    if total <= max_bytes {
      break;
    }
    match std::fs::remove_file(&path) {
      Ok(()) => {
        total = total.saturating_sub(len);
        deleted += 1;
      }
      Err(e) => log::warn!(
        "prompt cache: prune could not remove {}: {e}",
        path.display()
      ),
    }
  }
  if deleted > 0 {
    log::info!(
      "prompt cache: over the {} MiB cap, pruned {deleted} old file(s) ({} MiB remain)",
      max_bytes / (1024 * 1024),
      total / (1024 * 1024)
    );
  }
  deleted
}

/// The save pass, run just before an eviction stops `port`. Every slot that
/// sits idle with a prompt gets its own file; processing slots are skipped
/// (the sweep only evicts `inflight == 0` launches, so this is the residue
/// of a race window at worst).
pub async fn save_idle_slots(port: u16, dir: &Path, root: &Path, total_cap_bytes: u64) {
  let result = tokio::time::timeout(SAVE_BUDGET, save_all(port, dir)).await;
  match result {
    Ok(Ok(n)) if n > 0 => {
      prune_to_cap(root, total_cap_bytes);
    }
    Ok(Ok(_)) => log::debug!("prompt cache: nothing idle to save on port {port}"),
    Ok(Err(e)) => log::warn!("prompt cache: save on port {port} failed: {e}"),
    Err(_) => log::warn!(
      "prompt cache: save on port {port} exceeded {}s; evicting without it",
      SAVE_BUDGET.as_secs()
    ),
  }
}

async fn save_all(port: u16, dir: &Path) -> Result<usize, std::io::Error> {
  let slots = parse_slots(&raw_json("GET", port, "/slots", None).await?);
  let mut saved = 0usize;
  for slot in slots
    .into_iter()
    .filter(|s| !s.is_processing && s.n_prompt_tokens > 0)
  {
    let filename = save_filename(slot.id, SystemTime::now());
    match slot_action(port, slot.id, "save", &filename).await {
      Ok(v) => {
        let n_saved = v.get("n_saved").and_then(|x| x.as_u64()).unwrap_or(0);
        let n_bytes = v.get("n_written").and_then(|x| x.as_u64()).unwrap_or(0);
        log::info!(
          "prompt cache: saved slot {} ({} tokens, {} MiB) to {}",
          slot.id,
          n_saved,
          n_bytes / (1024 * 1024),
          dir.join(&filename).display()
        );
        saved += 1;
      }
      Err(e) => log::warn!("prompt cache: save slot {} failed: {e}", slot.id),
    }
  }
  Ok(saved)
}

/// The restore pass, run between the readiness probe and the `Ready`
/// transition on a freshly launched server. Consumes each file it attempts
/// (success or failure): a file that will not restore now will not restore
/// later either, and keeping it would re-pay the attempt on every start.
pub async fn restore_slots(port: u16, dir: &Path) {
  match tokio::time::timeout(RESTORE_BUDGET, restore_all(port, dir)).await {
    Err(_) => log::warn!(
      "prompt cache: restore on port {port} exceeded {}s; going Ready without it",
      RESTORE_BUDGET.as_secs()
    ),
    Ok(Err(e)) => {
      log::warn!("prompt cache: restore pass on port {port} failed: {e}; going Ready without it")
    }
    Ok(Ok(())) => {}
  }
}

async fn restore_all(port: u16, dir: &Path) -> Result<(), std::io::Error> {
  let latest = latest_per_slot(dir);
  if latest.is_empty() {
    return Ok(());
  }
  log::info!(
    "prompt cache: {} save file(s) to restore on port {port} from {}",
    latest.len(),
    dir.display()
  );
  let live = live_slot_ids(&raw_json("GET", port, "/slots", None).await?);
  for (slot, (_, path)) in latest {
    if !live.contains(&slot) {
      // The relaunch has fewer slots than the save (preset changed); the
      // conversation would have to reprocess regardless.
      log::warn!(
        "prompt cache: dropping {} — the relaunched server has no slot {slot}",
        path.display()
      );
      let _ = std::fs::remove_file(&path);
      continue;
    }
    let filename = path
      .file_name()
      .map(|s| s.to_string_lossy().into_owned())
      .unwrap_or_default();
    match slot_action(port, slot, "restore", &filename).await {
      Ok(v) => {
        let n = v.get("n_restored").and_then(|x| x.as_u64()).unwrap_or(0);
        log::info!(
          "prompt cache: restored slot {} ({n} tokens) from {}",
          slot,
          path.display()
        );
      }
      Err(e) => log::warn!(
        "prompt cache: restore slot {slot} from {} failed: {e}; it reprocesses",
        path.display()
      ),
    }
    let _ = std::fs::remove_file(&path);
  }
  Ok(())
}

/// `POST /slots/{id}?action={action}` with llama.cpp's `{"filename": ...}`
/// body. Returns the parsed result object, or an error carrying the
/// server's message.
async fn slot_action(
  port: u16,
  slot: u32,
  action: &str,
  filename: &str,
) -> Result<serde_json::Value, std::io::Error> {
  let body = serde_json::json!({ "filename": filename }).to_string();
  let v = raw_json(
    "POST",
    port,
    &format!("/slots/{slot}?action={action}"),
    Some(&body),
  )
  .await?;
  if v.get("error").is_some() {
    let msg = v
      .pointer("/error/message")
      .and_then(|m| m.as_str())
      .unwrap_or("rejected");
    return Err(std::io::Error::other(msg));
  }
  Ok(v)
}

/// One loopback request, JSON body back. `Connection: close` so reading to
/// EOF terminates, same stance as `super::actuals::fetch_props_body`.
async fn raw_json(
  method: &str,
  port: u16,
  path: &str,
  body: Option<&str>,
) -> Result<serde_json::Value, std::io::Error> {
  let fut = async {
    let mut sock = TcpStream::connect(("127.0.0.1", port)).await?;
    let head = match body {
      Some(b) => format!("{method} {path} HTTP/1.1\r\nHost: 127.0.0.1:{port}\r\nConnection: close\r\nContent-Type: application/json\r\nContent-Length: {}\r\n\r\n{b}", b.len()),
      None => format!("{method} {path} HTTP/1.1\r\nHost: 127.0.0.1:{port}\r\nConnection: close\r\nAccept: application/json\r\n\r\n"),
    };
    sock.write_all(head.as_bytes()).await?;
    let mut buf = Vec::with_capacity(8192);
    let mut chunk = [0u8; 8192];
    loop {
      let n = sock.read(&mut chunk).await?;
      if n == 0 {
        break;
      }
      buf.extend_from_slice(&chunk[..n]);
      // Slot lists can be big (each carries its decoded prompt); cap below
      // what a healthy response costs so a misbehaving peer cannot grow the
      // daemon's buffer. 64 MiB fits a full slot prompt on a large context.
      if buf.len() > 64 * 1024 * 1024 {
        return Err(std::io::Error::other("slots response too large"));
      }
    }
    let split = buf
      .windows(4)
      .position(|w| w == b"\r\n\r\n")
      .ok_or_else(|| std::io::Error::other("slots response had no header boundary"))?;
    let status_line =
      String::from_utf8_lossy(&buf[..buf.iter().position(|b| *b == b'\r').unwrap_or(0)]);
    let status: u16 = status_line
      .split_whitespace()
      .nth(1)
      .and_then(|s| s.parse().ok())
      .unwrap_or(0);
    let value: serde_json::Value = serde_json::from_slice(&buf[split + 4..])
      .map_err(|e| std::io::Error::other(format!("unparsable slots response ({status}): {e}")))?;
    if status >= 400 && value.get("error").is_none() {
      return Err(std::io::Error::other(format!("HTTP {status}")));
    }
    Ok(value)
  };
  tokio::time::timeout(HTTP_TIMEOUT, fut)
    .await
    .map_err(|_| std::io::Error::new(std::io::ErrorKind::TimedOut, "slots request timed out"))?
}

#[cfg(test)]
mod tests {
  use super::*;

  #[test]
  fn slots_rows_parse_and_ignore_unshapely_entries() {
    let body: serde_json::Value = serde_json::from_str(
      r#"[{"id":0,"is_processing":false,"n_prompt_tokens":42,"params":{}},
          {"id":1,"is_processing":true,"n_prompt_tokens":7},
          {"id":2},{"junk":true}]"#,
    )
    .unwrap();
    assert_eq!(
      parse_slots(&body),
      vec![
        SlotInfo {
          id: 0,
          is_processing: false,
          n_prompt_tokens: 42
        },
        SlotInfo {
          id: 1,
          is_processing: true,
          n_prompt_tokens: 7
        },
      ]
    );
    assert_eq!(parse_slots(&serde_json::json!({})), Vec::new());
    // A never-used slot row can omit all but `id`; the liveness check
    // still has to see that slot (dropping it discards a save file).
    assert_eq!(live_slot_ids(&body), vec![0u32, 1, 2]);
    assert_eq!(live_slot_ids(&serde_json::json!({})), Vec::<u32>::new());
  }

  #[test]
  fn cache_names_parse_only_our_shape() {
    assert_eq!(
      parse_saved_file("slot0-1760000000.bin"),
      Some((0, 1760000000))
    );
    assert_eq!(parse_saved_file("slot12-1.bin"), Some((12, 1)));
    assert_eq!(parse_saved_file("slot-5.bin"), None);
    assert_eq!(parse_saved_file("slotX-1.bin"), None);
    assert_eq!(parse_saved_file("slot1-1.tmp"), None);
    assert_eq!(parse_saved_file("notes.txt"), None);
  }

  #[test]
  fn latest_per_slot_keeps_one_file_per_slot_newest_wins() {
    let dir = crate::util::test_temp::unique_temp_dir("latest-per-slot");
    for name in [
      "slot0-100.bin",
      "slot0-300.bin",
      "slot0-200.bin",
      "slot1-150.bin",
      "foreign.bin",
    ] {
      std::fs::write(dir.join(name), b"x").unwrap();
    }
    let latest = latest_per_slot(&dir);
    assert_eq!(latest.len(), 2);
    assert_eq!(latest[&0].0, 300);
    assert_eq!(latest[&1].0, 150);
  }

  #[test]
  fn prune_keeps_newest_and_fits_the_cap() {
    let root = crate::util::test_temp::unique_temp_dir("prune-cap");
    let m1 = root.join("m1");
    let m2 = root.join("m2");
    std::fs::create_dir_all(&m1).unwrap();
    std::fs::create_dir_all(&m2).unwrap();
    let body = vec![b'x'; 1000];
    // Oldest → newest: m1 old, m1 mid, m2 newest. A foreign file sits in m1
    // too — it is neither counted nor touched.
    std::fs::write(m1.join("slot0-100.bin"), &body).unwrap();
    std::thread::sleep(Duration::from_millis(5));
    std::fs::write(m1.join("slot0-200.bin"), &body).unwrap();
    std::fs::write(m1.join("foreign.bin"), &body).unwrap();
    std::thread::sleep(Duration::from_millis(5));
    std::fs::write(m2.join("slot1-300.bin"), &body).unwrap();

    // Two files fit in 2500; the oldest (slot0-100) goes, foreign stays.
    assert_eq!(prune_to_cap(&root, 2500), 1);
    assert!(!m1.join("slot0-100.bin").exists());
    assert!(m1.join("slot0-200.bin").exists());
    assert!(m2.join("slot1-300.bin").exists());
    assert!(m1.join("foreign.bin").exists());

    // A cap smaller than the newest file keeps exactly that one file.
    assert_eq!(prune_to_cap(&root, 1), 1);
    assert!(!m1.join("slot0-200.bin").exists());
    assert!(m2.join("slot1-300.bin").exists());

    // Room to spare: nothing moves.
    assert_eq!(prune_to_cap(&root, 1_000_000), 0);
    assert!(m2.join("slot1-300.bin").exists());
  }

  #[test]
  fn model_dir_is_stable_path_scoped_and_bounded() {
    let a = model_dir(Path::new("/models/Llama 3.2-1B.gguf")).unwrap();
    let b = model_dir(Path::new("/models/Llama 3.2-1B.gguf")).unwrap();
    let c = model_dir(Path::new("/models/other/Llama 3.2-1B.gguf")).unwrap();
    assert_eq!(a, b, "same path → same dir");
    assert_ne!(a, c, "different paths never share a dir");
    let stem = a.file_name().unwrap().to_string_lossy().into_owned();
    assert!(stem.starts_with("Llama-3.2-1B-"), "got {stem}");
    assert!(a.starts_with(crate::util::paths::cache_dir().unwrap()));
  }

  #[test]
  fn save_filename_is_slot_prefixed_flat_and_sortable() {
    let early = save_filename(3, UNIX_EPOCH + Duration::from_secs(100));
    let late = save_filename(3, UNIX_EPOCH + Duration::from_secs(200));
    assert_eq!(early, "slot3-100.bin");
    assert!(early < late);
    assert!(!early.contains(std::path::MAIN_SEPARATOR));
  }
}
