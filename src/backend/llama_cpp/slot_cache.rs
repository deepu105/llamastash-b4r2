//! Prompt-cache persistence across unloads: `--slot-save-path` + the
//! `/slots/{id}?action=save|restore` endpoints.
//!
//! When the idle sweep or make-room evicts a llama.cpp launch mid-conversation,
//! the next request would otherwise reprocess the whole prompt (a 100k-token
//! prompt on CPU costs tens of minutes; see
//! `docs/spikes/2026-10-10-slot-prompt-cache.md`). Before an evict-style stop,
//! every idle slot holding a prompt is saved to disk; when the same model
//! launches again, the files are restored into their slots before the launch
//! is advertised Ready. llama.cpp's own slot selection (LCP similarity against
//! each slot's cached prompt) then matches a returning conversation to the
//! restored slot — this module does no conversation matching.
//!
//! Files live under `<state_dir>/slot-cache/<model-path-hash>/slot-<id>.bin`.
//! A save file holds the whole KV window of one slot (~32 KB/token for a
//! Llama-3.2-1B-class model), so the per-model directory is capped by total
//! size (oldest files go first) and files older than the configured TTL are
//! dropped on the next save or restore pass.
//!
//! Transport is hand-rolled HTTP/1.1 over raw TCP with `Connection: close`,
//! the same no-dep stance as [`super::actuals`] and [`crate::daemon::probe`].

use std::path::{Path, PathBuf};
use std::time::{Duration, SystemTime, UNIX_EPOCH};

use serde_json::Value;
use tokio::io::{AsyncReadExt, AsyncWriteExt};
use tokio::net::TcpStream;

/// Per-request budget for a save. A 100k-token slot save measured ~2.5 s on
/// NVMe (spike, 2026-10-10); 60 s covers slow disks with a wide margin while
/// still bounding an eviction pause.
const SAVE_TIMEOUT: Duration = Duration::from_secs(60);
/// Per-file budget for a restore. Runs before the launch is advertised Ready,
/// so this is the worst-case extra wait on a cold start; reads are
/// page-cache-fast (3.3 GB restored in ~1 s in the spike).
const RESTORE_TIMEOUT: Duration = Duration::from_secs(120);
/// Budget for `GET /slots`. A local, immediate reply.
const SLOT_LIST_TIMEOUT: Duration = Duration::from_secs(5);

/// The directory slot files for one model live in: `<state_dir>/slot-cache/`
/// plus a hash of the canonical model path. Hashed because model paths carry
/// separators and length limits a directory name should not; two paths that
/// differ at all get different directories.
pub fn slot_dir_for(state_dir: &Path, model_path: &Path) -> PathBuf {
  use sha2::Digest;
  let canonical = model_path
    .canonicalize()
    .unwrap_or_else(|_| model_path.to_path_buf());
  let digest = sha2::Sha256::digest(canonical.display().to_string().as_bytes());
  let mut tag = String::with_capacity(16);
  for byte in &digest[..8] {
    tag.push_str(&format!("{byte:02x}"));
  }
  state_dir.join("slot-cache").join(tag)
}

/// The save file name for one slot id — what `action=save|restore` accepts
/// (`fs_validate_filename` upstream rejects path separators; a flat
/// `slot-<id>.bin` passes and round-trips).
pub fn slot_filename(id: u32) -> String {
  format!("slot-{id}.bin")
}

/// The slot id a save file name names, or `None` for a foreign file.
pub fn slot_id_from_filename(name: &str) -> Option<u32> {
  let digits = name.strip_prefix("slot-")?.strip_suffix(".bin")?;
  if digits.is_empty() || !digits.bytes().all(|b| b.is_ascii_digit()) {
    return None;
  }
  digits.parse().ok()
}

/// One file's pruning inputs.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct FileStamp {
  pub name: String,
  pub bytes: u64,
  pub mtime: Duration,
}

/// Which slot files to delete: every file past its TTL, then whole files
/// oldest-mtime-first until the remaining set fits `max_bytes`. Returns the
/// names to delete.
pub fn prune_plan(
  files: &[FileStamp],
  now: Duration,
  max_bytes: u64,
  ttl: Duration,
) -> Vec<String> {
  let mut doomed: Vec<(Duration, String)> = Vec::new();
  let mut keep: Vec<&FileStamp> = Vec::new();
  for f in files {
    if now.checked_sub(f.mtime).is_some_and(|age| age > ttl) {
      doomed.push((f.mtime, f.name.clone()));
    } else {
      keep.push(f);
    }
  }
  let mut total: u64 = keep.iter().map(|f| f.bytes).sum();
  if total > max_bytes {
    // Stable on mtime: equal mtimes keep in name order so the plan is
    // deterministic.
    keep.sort_by_key(|f| (f.mtime, f.name.clone()));
    for f in keep {
      if total <= max_bytes {
        break;
      }
      total -= f.bytes;
      doomed.push((f.mtime, f.name.clone()));
    }
  }
  doomed.sort();
  doomed.into_iter().map(|(_, name)| name).collect()
}

/// The `GET /slots` verdict for one slot: which idle slots hold a prompt.
struct SlotView {
  id: u32,
  is_processing: bool,
  n_prompt_tokens: u64,
}

fn parse_slots(body: &Value) -> Vec<SlotView> {
  body
    .as_array()
    .map(|arr| {
      arr
        .iter()
        .filter_map(|s| {
          Some(SlotView {
            id: s.get("id")?.as_u64()? as u32,
            is_processing: s
              .get("is_processing")
              .and_then(Value::as_bool)
              .unwrap_or(true),
            // Idle slots that never served a task carry no prompt fields.
            n_prompt_tokens: s
              .get("n_prompt_tokens")
              .and_then(Value::as_u64)
              .unwrap_or(0),
          })
        })
        .collect()
    })
    .unwrap_or_default()
}

/// Eviction-side save: every idle slot with a cached prompt is saved as
/// `slot-<id>.bin` into `dir`, then the directory is pruned to the cap.
/// Best-effort: a failing slot is logged and skipped; the caller stops the
/// launch either way. Returns the number of slots saved.
pub async fn save_prompt_caches(port: u16, dir: &Path, max_bytes: u64, ttl: Duration) -> usize {
  let dir_string = dir.display().to_string();
  if let Err(e) = std::fs::create_dir_all(dir) {
    log::warn!("slot-cache: cannot create {dir_string}: {e}; skipping save");
    return 0;
  }
  let slots = match get_slots(port).await {
    Ok(slots) => slots,
    Err(e) => {
      log::warn!("slot-cache: GET /slots on port {port} failed ({e}); skipping save");
      return 0;
    }
  };
  let mut saved = 0usize;
  for slot in slots
    .iter()
    .filter(|s| !s.is_processing && s.n_prompt_tokens > 0)
  {
    let filename = slot_filename(slot.id);
    let body = format!("{{\"filename\":\"{filename}\"}}");
    match post_slot_action(port, slot.id, "save", &body, SAVE_TIMEOUT).await {
      Ok(res) => {
        log::info!(
          "slot-cache: saved slot {} ({} prompt tokens, {} bytes) to {dir_string}/{filename}",
          slot.id,
          slot.n_prompt_tokens,
          res.get("n_written").and_then(Value::as_u64).unwrap_or(0),
        );
        saved += 1;
      }
      Err(e) => log::warn!(
        "slot-cache: save of slot {} failed ({e}); continuing",
        slot.id
      ),
    }
  }
  if saved > 0 {
    prune_dir(dir, max_bytes, ttl);
  }
  saved
}

/// Readiness-side restore: after pruning TTL-expired files, each
/// `slot-<id>.bin` whose slot exists on this server is restored into it.
/// Best-effort: a file that fails to restore (context too small, corrupted,
/// model changed) is logged and skipped — a cache miss, not a failed launch.
/// Returns the number of slots restored.
pub async fn restore_prompt_caches(port: u16, dir: &Path, ttl: Duration) -> usize {
  if !dir.is_dir() {
    return 0;
  }
  prune_dir(dir, u64::MAX, ttl);
  let mut entries: Vec<(u32, String)> = Vec::new();
  let Ok(read) = std::fs::read_dir(dir) else {
    return 0;
  };
  for entry in read.flatten() {
    let name = entry.file_name().to_string_lossy().to_string();
    if let Some(id) = slot_id_from_filename(&name) {
      entries.push((id, name));
    }
  }
  if entries.is_empty() {
    return 0;
  }
  // `action=restore` wraps out-of-range ids onto a live slot upstream, so
  // only restore ids this server actually has; a launch with fewer parallel
  // slots than the last one must not spill a file onto an unrelated slot.
  let live: Vec<u32> = match get_slots(port).await {
    Ok(slots) => slots.into_iter().map(|s| s.id).collect(),
    Err(e) => {
      log::warn!("slot-cache: GET /slots on port {port} failed ({e}); skipping restore");
      return 0;
    }
  };
  entries.sort();
  let mut restored = 0usize;
  for (id, name) in entries {
    if !live.contains(&id) {
      log::debug!("slot-cache: no slot {id} on this launch; leaving {name}");
      continue;
    }
    let body = format!("{{\"filename\":\"{name}\"}}");
    match post_slot_action(port, id, "restore", &body, RESTORE_TIMEOUT).await {
      Ok(res) => {
        log::info!(
          "slot-cache: restored slot {id} from {name} ({} tokens, {} bytes)",
          res.get("n_restored").and_then(Value::as_u64).unwrap_or(0),
          res.get("n_read").and_then(Value::as_u64).unwrap_or(0),
        );
        restored += 1;
      }
      Err(e) => log::warn!("slot-cache: restore of slot {id} from {name} failed ({e})"),
    }
  }
  restored
}

/// Drop TTL-expired files in `dir`, then enforce `max_bytes` oldest-first.
fn prune_dir(dir: &Path, max_bytes: u64, ttl: Duration) {
  let Ok(read) = std::fs::read_dir(dir) else {
    return;
  };
  let now = system_now();
  let mut files = Vec::new();
  for entry in read.flatten() {
    let name = entry.file_name().to_string_lossy().to_string();
    if slot_id_from_filename(&name).is_none() {
      continue;
    }
    let Ok(meta) = entry.metadata() else { continue };
    let mtime = meta
      .modified()
      .ok()
      .and_then(|t| t.duration_since(UNIX_EPOCH).ok())
      .unwrap_or(Duration::ZERO);
    files.push(FileStamp {
      name,
      bytes: meta.len(),
      mtime,
    });
  }
  for name in prune_plan(&files, now, max_bytes, ttl) {
    let path = dir.join(&name);
    if let Err(e) = std::fs::remove_file(&path) {
      log::warn!("slot-cache: pruning {} failed: {e}", path.display());
    } else {
      log::info!("slot-cache: pruned {}", path.display());
    }
  }
}

fn system_now() -> Duration {
  SystemTime::now()
    .duration_since(UNIX_EPOCH)
    .unwrap_or(Duration::ZERO)
}

async fn get_slots(port: u16) -> Result<Vec<SlotView>, String> {
  let body = http(port, "GET", "/slots", None, SLOT_LIST_TIMEOUT)
    .await
    .map_err(|e| e.to_string())?;
  let value: Value = serde_json::from_slice(&body).map_err(|e| e.to_string())?;
  Ok(parse_slots(&value))
}

async fn post_slot_action(
  port: u16,
  id: u32,
  action: &str,
  body: &str,
  timeout: Duration,
) -> Result<Value, String> {
  let path = format!("/slots/{id}?action={action}");
  let raw = http(port, "POST", &path, Some(body), timeout)
    .await
    .map_err(|e| e.to_string())?;
  serde_json::from_slice(&raw).map_err(|e| format!("unparseable reply: {e}"))
}

/// One request/reply over loopback TCP. The server closes after a
/// `Connection: close` reply, so a read-to-EOF terminates the body.
async fn http(
  port: u16,
  method: &str,
  path: &str,
  body: Option<&str>,
  timeout: Duration,
) -> Result<Vec<u8>, String> {
  let request = match body {
    Some(b) => format!(
      "{method} {path} HTTP/1.1\r\nHost: 127.0.0.1:{port}\r\nConnection: close\r\nContent-Type: application/json\r\nContent-Length: {}\r\n\r\n{b}",
      b.len()
    ),
    None => format!(
      "{method} {path} HTTP/1.1\r\nHost: 127.0.0.1:{port}\r\nConnection: close\r\nAccept: application/json\r\n\r\n"
    ),
  };
  let fut = async {
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
    }
    Ok::<_, std::io::Error>(buf)
  };
  let raw = tokio::time::timeout(timeout, fut)
    .await
    .map_err(|_| "timeout".to_string())?
    .map_err(|e| e.to_string())?;
  split_http_body(&raw)
}

/// Status check + body of a raw HTTP response. Errors carry the status line
/// and a bounded body excerpt — an `action=save` failure is reported in the
/// JSON body with a non-2xx status.
fn split_http_body(raw: &[u8]) -> Result<Vec<u8>, String> {
  let head = String::from_utf8_lossy(raw.split(|b| *b == b'\r').next().unwrap_or(b""))
    .trim_end()
    .to_string();
  let Some(idx) = raw.windows(4).position(|w| w == b"\r\n\r\n") else {
    return Err(format!("no HTTP headers (head: {head})"));
  };
  let status: u16 = head
    .split_whitespace()
    .nth(1)
    .and_then(|s| s.parse().ok())
    .unwrap_or(0);
  if !(200..300).contains(&status) {
    let body = String::from_utf8_lossy(&raw[idx + 4..]);
    let excerpt: String = body.chars().take(400).collect();
    return Err(format!("{head}: {excerpt}"));
  }
  Ok(raw[idx + 4..].to_vec())
}

#[cfg(test)]
mod tests {
  use super::*;

  fn stamp(name: &str, bytes: u64, mtime_secs: u64) -> FileStamp {
    FileStamp {
      name: name.into(),
      bytes,
      mtime: Duration::from_secs(mtime_secs),
    }
  }

  #[test]
  fn slot_filenames_round_trip() {
    assert_eq!(slot_filename(0), "slot-0.bin");
    assert_eq!(slot_id_from_filename("slot-0.bin"), Some(0));
    assert_eq!(slot_id_from_filename("slot-42.bin"), Some(42));
  }

  #[test]
  fn foreign_file_names_are_rejected() {
    for name in [
      "slot-.bin",
      "slot-a.bin",
      "slot-1.txt",
      "other.bin",
      "slot-1.bin.exe",
      "",
    ] {
      assert_eq!(slot_id_from_filename(name), None, "{name}");
    }
  }

  #[test]
  fn a_slot_dir_is_hashed_per_model_path() {
    let dir = slot_dir_for(Path::new("/state"), Path::new("/models/a.gguf"));
    assert!(dir.starts_with("/state/slot-cache/"));
    assert_eq!(
      dir,
      slot_dir_for(Path::new("/state"), Path::new("/models/a.gguf"))
    );
    assert_ne!(
      dir,
      slot_dir_for(Path::new("/state"), Path::new("/models/b.gguf"))
    );
  }

  #[test]
  fn expired_files_are_pruned_before_any_size_cut() {
    let files = vec![
      stamp("slot-0.bin", 100, 10),
      stamp("slot-1.bin", 100, 160),
      stamp("slot-2.bin", 100, 190),
    ];
    // now=200, ttl=50 → slot-0 (age 190) expires; the rest fit the cap.
    let doomed = prune_plan(
      &files,
      Duration::from_secs(200),
      1000,
      Duration::from_secs(50),
    );
    assert_eq!(doomed, vec!["slot-0.bin".to_string()]);
  }

  #[test]
  fn over_cap_prunes_oldest_first_until_it_fits() {
    let files = vec![
      stamp("slot-0.bin", 500, 10),
      stamp("slot-1.bin", 300, 20),
      stamp("slot-2.bin", 400, 30),
    ];
    // Total 1200, cap 800: oldest (slot-0) goes (→700 ≤ 800).
    let doomed = prune_plan(
      &files,
      Duration::from_secs(100),
      800,
      Duration::from_secs(3600),
    );
    assert_eq!(doomed, vec!["slot-0.bin".to_string()]);
  }

  #[test]
  fn a_zero_cap_keeps_nothing_and_clock_skew_is_not_expired() {
    let files = vec![stamp("slot-0.bin", 1, 500)];
    assert_eq!(
      prune_plan(&files, Duration::from_secs(600), 0, Duration::from_secs(10)),
      vec!["slot-0.bin".to_string()]
    );
    // A file with a future mtime (clock skew) must not count as expired.
    assert!(prune_plan(&files, Duration::from_secs(100), 10, Duration::from_secs(1)).is_empty());
  }

  #[test]
  fn slots_parse_skips_entries_without_id_and_defaults_busy() {
    let body: serde_json::Value = serde_json::from_str(
      r#"[{"id":0,"is_processing":false,"n_prompt_tokens":120},
          {"id":1,"is_processing":true,"n_prompt_tokens":50},
          {"id":2},          {"no_id":true}]"#,
    )
    .unwrap();
    let slots = parse_slots(&body);
    assert_eq!(slots.len(), 3);
    assert_eq!(
      (
        slots[0].id,
        slots[0].is_processing,
        slots[0].n_prompt_tokens
      ),
      (0, false, 120)
    );
    assert_eq!(
      (
        slots[1].id,
        slots[1].is_processing,
        slots[1].n_prompt_tokens
      ),
      (1, true, 50)
    );
    // A slot that never served a task has no prompt fields — reads as 0, and
    // a slot with no `is_processing` reads as busy (never save into a race).
    assert_eq!(
      (
        slots[2].id,
        slots[2].is_processing,
        slots[2].n_prompt_tokens
      ),
      (2, true, 0)
    );
  }

  #[test]
  fn http_body_split_rejects_non_2xx_with_an_excerpt() {
    let ok = b"HTTP/1.1 200 OK\r\nContent-Type: application/json\r\n\r\n{\"n_saved\":1}";
    assert_eq!(split_http_body(ok).unwrap(), b"{\"n_saved\":1}");
    let err =
      b"HTTP/1.1 400 Bad Request\r\n\r\n{\"error\":{\"msg\":\"Unable to restore slot: invalid\"}}";
    let e = split_http_body(err).unwrap_err();
    assert!(e.contains("400"), "{e}");
    assert!(e.contains("Unable to restore slot"), "{e}");
    assert!(split_http_body(b"garbage").is_err());
  }

  #[test]
  fn prune_removes_real_files() {
    let dir = crate::util::test_temp::unique_temp_dir("slot-prune");
    std::fs::write(dir.join("slot-0.bin"), b"x").unwrap();
    std::fs::write(dir.join("slot-1.bin"), b"x").unwrap();
    std::fs::write(dir.join("notes.txt"), b"x").unwrap();
    // A ttl of 0 expires both slot files; the foreign file is left alone.
    prune_dir(&dir, u64::MAX, Duration::ZERO);
    assert!(!dir.join("slot-0.bin").exists());
    assert!(!dir.join("slot-1.bin").exists());
    assert!(dir.join("notes.txt").exists());
    std::fs::remove_dir_all(&dir).unwrap();
  }

  /// A stand-in llama-server: answers `GET /slots` from a fixed body and
  /// records the slot-action requests it receives, so the save and restore
  /// drivers can be exercised end-to-end over a real socket without the
  /// engine.
  mod stub {
    use tokio::io::{AsyncReadExt, AsyncWriteExt};

    pub struct Stub {
      pub port: u16,
      pub requests: Arc<tokio::sync::Mutex<Vec<String>>>,
    }

    use std::sync::Arc;

    /// `slots_body` is what `GET /slots` returns; every action request
    /// replies 200 with `{"ok":true}` unless the path names a slot listed
    /// in `fail_ids` (400 + an error body, like upstream).
    pub async fn spawn(slots_body: &'static str, fail_ids: Vec<u32>) -> Stub {
      let listener = tokio::net::TcpListener::bind(("127.0.0.1", 0))
        .await
        .unwrap();
      let port = listener.local_addr().unwrap().port();
      let requests: Arc<tokio::sync::Mutex<Vec<String>>> = Arc::new(Default::default());
      let sink = requests.clone();
      let fails = Arc::new(fail_ids);
      tokio::spawn(async move {
        loop {
          let Ok((mut sock, _)) = listener.accept().await else {
            return;
          };
          let sink = sink.clone();
          let fail = fails.clone();
          tokio::spawn(async move {
            let mut buf = vec![0u8; 4096];
            let n = sock.read(&mut buf).await.unwrap_or(0);
            let head = String::from_utf8_lossy(&buf[..n]).to_string();
            let path = head
              .split_whitespace()
              .nth(1)
              .unwrap_or_default()
              .to_string();
            sink.lock().await.push(head);
            let status;
            let body;
            if path == "/slots" {
              status = "200 OK";
              body = slots_body.to_string();
            } else {
              let bad = fail
                .iter()
                .any(|id| path.starts_with(&format!("/slots/{id}?")));
              if bad {
                status = "400 Bad Request";
                body = r#"{"error":{"msg":"Invalid slot ID"}}"#.to_string();
              } else {
                status = "200 OK";
                body = r#"{"ok":true}"#.to_string();
              }
            }
            let reply = format!(
              "HTTP/1.1 {status}\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{body}",
              body.len()
            );
            let _ = sock.write_all(reply.as_bytes()).await;
          });
        }
      });
      Stub { port, requests }
    }
  }

  #[tokio::test]
  async fn save_saves_only_idle_slots_with_a_prompt_and_prunes() {
    let dir = crate::util::test_temp::unique_temp_dir("slot-save");
    let stub = stub::spawn(
      r#"[{"id":0,"is_processing":false,"n_prompt_tokens":100},
          {"id":1,"is_processing":true,"n_prompt_tokens":50},
          {"id":2,"is_processing":false}]"#,
      vec![],
    )
    .await;
    // A stale file from a departed slot: back-dated so the post-save prune
    // sees it past the TTL.
    std::fs::write(dir.join("slot-9.bin"), b"stale").unwrap();
    let f = std::fs::File::options()
      .write(true)
      .open(dir.join("slot-9.bin"))
      .unwrap();
    f.set_times(
      std::fs::FileTimes::new()
        .set_modified(std::time::SystemTime::UNIX_EPOCH + Duration::from_secs(1)),
    )
    .unwrap();
    drop(f);
    let saved = save_prompt_caches(stub.port, &dir, u64::MAX, Duration::from_secs(3600)).await;
    assert_eq!(saved, 1, "only the idle prompt-bearing slot is saved");
    let reqs = stub.requests.lock().await;
    assert!(reqs.iter().any(|r| r.starts_with("GET /slots")));
    let save = reqs
      .iter()
      .find(|r| r.contains("/slots/0?action=save"))
      .expect("save for slot 0");
    assert!(
      save.contains("filename") && save.contains("slot-0.bin"),
      "{save}"
    );
    assert!(!reqs
      .iter()
      .any(|r| r.contains("slot-1") || r.contains("/slots/1?")));
    assert!(!reqs.iter().any(|r| r.contains("/slots/2?")));
    drop(reqs);
    assert!(
      !dir.join("slot-9.bin").exists(),
      "expired file pruned after save"
    );
    std::fs::remove_dir_all(&dir).unwrap();
  }

  #[tokio::test]
  async fn a_failing_slot_does_not_stop_the_others() {
    let dir = crate::util::test_temp::unique_temp_dir("slot-save-partial");
    let stub = stub::spawn(
      r#"[{"id":0,"is_processing":false,"n_prompt_tokens":10},
          {"id":1,"is_processing":false,"n_prompt_tokens":10}]"#,
      vec![0],
    )
    .await;
    let saved = save_prompt_caches(stub.port, &dir, u64::MAX, Duration::from_secs(3600)).await;
    assert_eq!(saved, 1, "slot 0 fails (stub 400), slot 1 still saved");
    std::fs::remove_dir_all(&dir).unwrap();
  }

  #[tokio::test]
  async fn restore_skips_files_whose_slot_this_server_lacks() {
    let dir = crate::util::test_temp::unique_temp_dir("slot-restore");
    std::fs::write(dir.join("slot-0.bin"), b"x").unwrap();
    std::fs::write(dir.join("slot-7.bin"), b"x").unwrap();
    // This launch has slots 0 and 1 only — `action=restore` wraps out-of-range
    // ids upstream, so slot 7's file must never be pushed onto a live slot.
    let stub = stub::spawn(
      r#"[{"id":0,"is_processing":false},{"id":1,"is_processing":false}]"#,
      vec![],
    )
    .await;
    let restored = restore_prompt_caches(stub.port, &dir, Duration::from_secs(3600)).await;
    assert_eq!(restored, 1);
    let reqs = stub.requests.lock().await;
    assert!(reqs.iter().any(|r| r.contains("/slots/0?action=restore")));
    assert!(!reqs.iter().any(|r| r.contains("/slots/7?")));
    drop(reqs);
    std::fs::remove_dir_all(&dir).unwrap();
  }

  #[tokio::test]
  async fn a_missing_dir_is_a_quiet_no_op_on_both_sides() {
    let dir = crate::util::test_temp::unique_temp_dir("slot-missing");
    std::fs::remove_dir_all(&dir).unwrap();
    let stub = stub::spawn("[]", vec![]).await;
    assert_eq!(
      restore_prompt_caches(stub.port, &dir, Duration::from_secs(3600)).await,
      0
    );
    // Save creates the directory (it runs at stop, where a first-ever save
    // is legitimate), and with no prompt-bearing slots it saves nothing.
    assert_eq!(
      save_prompt_caches(stub.port, &dir, u64::MAX, Duration::from_secs(3600)).await,
      0
    );
    std::fs::remove_dir_all(&dir).unwrap();
  }
}
