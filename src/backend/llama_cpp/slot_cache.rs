//! Prompt cache across unloads: llama-server slot save/restore transport
//! and the on-disk layout of the saved caches.
//!
//! `llama-server` can write a slot's prompt + KV cache to disk
//! (`--slot-save-path DIR`, then `POST /slots/{id}?action=save|restore` with a
//! `{"filename": …}` body). llamastash writes the files just before the
//! eviction sweeper or make-room stops a llama.cpp launch, and reads them back
//! when a launch of the same model next reaches Ready, so a conversation that
//! outlives an unload reprocesses only its new tail. See
//! `docs/spikes/2026-10-10-prompt-cache-across-unload.md` for the measured
//! economics and `docs/architecture.md § Prompt cache across unloads` for when
//! each side runs.
//!
//! Files live under `<cache dir>/slot-cache/<model header hash>/slot-<id>.bin`
//! (one file per slot, stable name so a returning launch restores its own
//! slots), and the whole tree is capped: after every save the oldest files
//! across all models are deleted until the tree fits
//! `backend.llamacpp.prompt_cache_max_bytes`.
//!
//! Transport is hand-rolled raw TCP like [`super::actuals`] — the same
//! no-HTTP-dependency stance. llama.cpp validates `filename`
//! ([`slot_filename`]’s output is inside its allowlist by construction).

use std::path::{Path, PathBuf};
use std::time::{Duration, SystemTime};

use tokio::io::{AsyncReadExt, AsyncWriteExt};
use tokio::net::TcpStream;

/// Directory name under the daemon's cache dir holding the per-model slot
/// cache subdirs.
const SLOT_CACHE_SUBDIR: &str = "slot-cache";

/// The per-model file name for a slot's saved cache. llama.cpp takes any
/// `fs_validate_filename`-safe name and echoes it in the response; a stable
/// name per slot id is what lets a fresh launch restore the file its slot was
/// saved under. The file itself pins the slot id too — restoring into a
/// different slot id is rejected by the engine.
pub fn slot_filename(id: i64) -> String {
  format!("slot-{id}.bin")
}

/// The slot id a file name written by [`slot_filename`] carries, `None` for
/// anything else in the directory (stray files are left alone).
pub fn parse_slot_filename(name: &str) -> Option<i64> {
  name
    .strip_prefix("slot-")?
    .strip_suffix(".bin")?
    .parse::<i64>()
    .ok()
}

/// The slot-cache root: `<cache dir>/slot-cache`. `None` when no cache dir
/// resolves (same posture as the log dir).
pub fn cache_root() -> Option<PathBuf> {
  crate::util::paths::cache_dir().map(|d| d.join(SLOT_CACHE_SUBDIR))
}

/// The per-model subdirectory: `<root>/<model header hash>`. Keyed on the
/// BLAKE3 header hash alone (not the path) so a renamed file keeps its cache
/// and a replaced file (same path, new content → new hash) starts fresh.
pub fn model_cache_dir(root: &Path, header_hex: &str) -> PathBuf {
  root.join(header_hex)
}

/// One slot as `GET /slots` reports it (fields verified against llama.cpp
/// `609290be6`, build 11525: `id`, `is_processing`, and `n_prompt_tokens`
/// which is absent until the slot has served a prompt).
#[derive(Debug, Clone, PartialEq)]
pub struct SlotBrief {
  pub id: i64,
  pub is_processing: bool,
  pub n_prompt_tokens: Option<u64>,
}

/// Parse a `GET /slots` JSON body.
fn parse_slots(body: &serde_json::Value) -> Vec<SlotBrief> {
  body
    .as_array()
    .map(|arr| {
      arr
        .iter()
        .filter_map(|s| {
          Some(SlotBrief {
            id: s.get("id")?.as_i64()?,
            is_processing: s.get("is_processing").and_then(|v| v.as_bool())?,
            n_prompt_tokens: s.get("n_prompt_tokens").and_then(|v| v.as_u64()),
          })
        })
        .collect()
    })
    .unwrap_or_default()
}

async fn http_request(
  port: u16,
  method: &str,
  path: &str,
  body: Option<&str>,
  timeout: Duration,
) -> std::io::Result<serde_json::Value> {
  let mut head = format!(
    "{method} {path} HTTP/1.1\r\nHost: 127.0.0.1:{port}\r\nConnection: close\r\nAccept: application/json\r\n"
  );
  if let Some(b) = body {
    head.push_str(&format!(
      "Content-Type: application/json\r\nContent-Length: {}\r\n",
      b.len()
    ));
  }
  head.push_str("\r\n");
  let payload = format!("{head}{}", body.unwrap_or(""));
  let fut = async {
    let mut sock = TcpStream::connect(("127.0.0.1", port)).await?;
    sock.write_all(payload.as_bytes()).await?;
    // `Connection: close` makes the server hang up after the body; a hard cap
    // bounds a misbehaving peer. `/slots` in metrics mode (the default) is
    // small — prompt *text* is only included when it isn't asked for.
    let mut buf = Vec::with_capacity(4096);
    let mut chunk = [0u8; 4096];
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
  let resp = tokio::time::timeout(timeout, fut).await.map_err(|_| {
    std::io::Error::new(std::io::ErrorKind::TimedOut, "{method} {path} timed out")
  })??;
  let Some(split) = resp.windows(4).position(|w| w == b"\r\n\r\n") else {
    return Err(std::io::Error::new(
      std::io::ErrorKind::InvalidData,
      "{method} {path}: no HTTP header",
    ));
  };
  let status_line = String::from_utf8_lossy(&resp[..split.min(64)]);
  let status: u16 = status_line
    .split_whitespace()
    .nth(1)
    .and_then(|s| s.parse().ok())
    .unwrap_or(0);
  let body_bytes = &resp[split + 4..];
  let parsed: serde_json::Value = serde_json::from_slice(body_bytes).unwrap_or_default();
  if !(200..300).contains(&status) {
    // llama.cpp errors as `{"error": {"code": …, "message": …}}`.
    let msg = parsed
      .pointer("/error/message")
      .and_then(|v| v.as_str())
      .unwrap_or("unknown error");
    return Err(std::io::Error::other(format!(
      "{method} {path}: HTTP {status}: {msg}"
    )));
  }
  Ok(parsed)
}

/// `GET /slots`.
pub async fn list_slots(port: u16, timeout: Duration) -> std::io::Result<Vec<SlotBrief>> {
  let body = http_request(port, "GET", "/slots", None, timeout).await?;
  Ok(parse_slots(&body))
}

/// The numeric reply of one slot action (`n_saved` / `n_restored`).
fn action_count(v: &serde_json::Value, key: &str) -> u64 {
  v.get(key).and_then(|v| v.as_u64()).unwrap_or(0)
}

/// `POST /slots/{id}?action=save` — write this slot's prompt + KV cache to
/// `filename` under the launch's `--slot-save-path` dir. llama.cpp *defers*
/// the task while the slot is processing, so on a slot that raced back into
/// traffic this call blocks (bounded by `timeout`).
pub async fn save_slot(
  port: u16,
  id: i64,
  filename: &str,
  timeout: Duration,
) -> std::io::Result<u64> {
  let body = format!("{{\"filename\": {filename:?}}}");
  let v = http_request(
    port,
    "POST",
    &format!("/slots/{id}?action=save"),
    Some(&body),
    timeout,
  )
  .await?;
  Ok(action_count(&v, "n_saved"))
}

/// `POST /slots/{id}?action=restore` — load `filename` back into the slot.
/// Fails (engine-side) when the file's slot id differs or the restored prompt
/// does not fit the slot's context; the caller logs and moves on.
pub async fn restore_slot(
  port: u16,
  id: i64,
  filename: &str,
  timeout: Duration,
) -> std::io::Result<u64> {
  let body = format!("{{\"filename\": {filename:?}}}");
  let v = http_request(
    port,
    "POST",
    &format!("/slots/{id}?action=restore"),
    Some(&body),
    timeout,
  )
  .await?;
  Ok(action_count(&v, "n_restored"))
}

/// What one [`save_all`] pass wrote.
#[derive(Debug, Default, PartialEq)]
pub struct SaveSummary {
  pub slots_saved: usize,
  pub tokens: u64,
}

/// Save every slot holding a prompt. Slots that never served a prompt have
/// nothing to write and are skipped (an empty save is an engine error).
pub async fn save_all(port: u16, timeout: Duration) -> std::io::Result<(SaveSummary, usize)> {
  let slots = list_slots(port, timeout).await?;
  let mut sum = SaveSummary::default();
  let mut failed = 0usize;
  for slot in slots {
    if slot.n_prompt_tokens.unwrap_or(0) == 0 {
      continue;
    }
    match save_slot(port, slot.id, &slot_filename(slot.id), timeout).await {
      Ok(n) => {
        sum.slots_saved += 1;
        sum.tokens += n;
      }
      Err(e) => {
        failed += 1;
        log::warn!("prompt cache: save of slot {} failed: {e}", slot.id);
      }
    }
  }
  Ok((sum, failed))
}

/// What one [`restore_all`] pass loaded.
#[derive(Debug, Default, PartialEq)]
pub struct RestoreSummary {
  pub slots_restored: usize,
  pub tokens: u64,
}

/// Restore every saved file in `dir` into its slot. A per-file failure (the
/// usual case: the new launch fits a smaller context than the save was taken
/// at) is logged and skipped — the launch serves cold and the file stays for a
/// future launch that fits.
pub async fn restore_all(
  port: u16,
  dir: &Path,
  timeout: Duration,
) -> std::io::Result<(RestoreSummary, usize)> {
  let mut sum = RestoreSummary::default();
  let mut failed = 0usize;
  let mut files: Vec<(i64, String)> = Vec::new();
  if let Ok(entries) = std::fs::read_dir(dir) {
    for entry in entries.flatten() {
      if !entry.path().is_file() {
        continue;
      }
      if let (Ok(name), Some(id)) = (
        entry.file_name().into_string(),
        entry.file_name().to_str().and_then(parse_slot_filename),
      ) {
        files.push((id, name));
      }
    }
  }
  files.sort_by_key(|(id, _)| *id);
  for (id, name) in files {
    match restore_slot(port, id, &name, timeout).await {
      Ok(n) => {
        sum.slots_restored += 1;
        sum.tokens += n;
      }
      Err(e) => {
        failed += 1;
        log::warn!("prompt cache: restore of `{name}` failed: {e}");
      }
    }
  }
  Ok((sum, failed))
}

/// One cache file as the size census sees it.
#[derive(Debug, Clone)]
pub struct CacheFile {
  pub path: PathBuf,
  pub size: u64,
  pub mtime: SystemTime,
}

/// Walk the slot-cache tree (one model-dir level deep) into a census.
/// Unreadable entries are skipped rather than failing the pass — the sweep is
/// best-effort housekeeping on the eviction path.
fn census(root: &Path) -> Vec<CacheFile> {
  let mut out = Vec::new();
  let Ok(models) = std::fs::read_dir(root) else {
    return out;
  };
  for model in models.flatten() {
    let Ok(files) = std::fs::read_dir(model.path()) else {
      continue;
    };
    for file in files.flatten() {
      let Ok(meta) = file.metadata() else {
        continue;
      };
      if !meta.is_file() {
        continue;
      }
      let mtime = meta.modified().unwrap_or(SystemTime::UNIX_EPOCH);
      out.push(CacheFile {
        path: file.path(),
        size: meta.len(),
        mtime,
      });
    }
  }
  out
}

/// The delete plan for [`prune`]: oldest-mtime first until the tree fits
/// `max_bytes`, never touching files under `keep` (the launch currently
/// being saved — its fresh files must survive its own eviction). Pure so the
/// ordering and the keep rule are unit-testable without disk.
fn plan_prune(files: &[CacheFile], max_bytes: u64, keep: &Path) -> Vec<PathBuf> {
  let total: u64 = files.iter().map(|f| f.size).sum();
  if total <= max_bytes {
    return Vec::new();
  }
  let mut victims: Vec<&CacheFile> = files.iter().filter(|f| !f.path.starts_with(keep)).collect();
  victims.sort_by_key(|f| f.mtime);
  let mut freed = 0u64;
  let mut out = Vec::new();
  for f in victims {
    out.push(f.path.clone());
    freed += f.size;
    if total.saturating_sub(freed) <= max_bytes {
      break;
    }
  }
  out
}

/// Enforce the size cap after a save: delete the oldest files across all
/// models until the tree fits, then drop empty model dirs. Returns the number
/// of files deleted.
pub fn prune(root: &Path, max_bytes: u64, keep: &Path) -> usize {
  let files = census(root);
  let victims = plan_prune(&files, max_bytes, keep);
  let mut deleted = 0usize;
  for path in &victims {
    match std::fs::remove_file(path) {
      Ok(()) => deleted += 1,
      Err(e) if e.kind() == std::io::ErrorKind::NotFound => {}
      Err(e) => log::warn!(
        "prompt cache: prune could not delete {}: {e}",
        path.display()
      ),
    }
  }
  if deleted > 0 {
    if let Ok(models) = std::fs::read_dir(root) {
      for model in models.flatten() {
        let _ = std::fs::remove_dir(model.path());
      }
    }
  }
  deleted
}

#[cfg(test)]
mod tests {
  use super::*;

  fn cf(path: &str, size: u64, age_secs: u64) -> CacheFile {
    CacheFile {
      path: PathBuf::from(path),
      size,
      mtime: SystemTime::UNIX_EPOCH + Duration::from_secs(10_000 - age_secs),
    }
  }

  #[test]
  fn slot_filenames_are_llama_cpp_safe_and_round_trip() {
    assert_eq!(slot_filename(0), "slot-0.bin");
    assert_eq!(slot_filename(41), "slot-41.bin");
    assert_eq!(parse_slot_filename("slot-0.bin"), Some(0));
    assert_eq!(parse_slot_filename("slot-41.bin"), Some(41));
    for junk in [
      "slot.bin",
      "slot-.bin",
      "slot-x.bin",
      "slot-1.txt",
      "other.bin",
      "",
    ] {
      assert_eq!(parse_slot_filename(junk), None, "{junk}");
    }
  }

  #[test]
  fn parses_slot_briefs_tolerantly() {
    // A never-used slot carries no `n_prompt_tokens`; a non-array body and
    // entries missing `id` contribute nothing rather than panicking.
    let v: serde_json::Value = serde_json::from_str(
      r#"[{"id":0,"n_ctx":8192,"is_processing":false},
          {"id":1,"n_ctx":8192,"is_processing":true,"n_prompt_tokens":4242},
          {"is_processing":false},
          {"id":2,"is_processing":false,"n_prompt_tokens":0}]"#,
    )
    .unwrap();
    assert_eq!(
      parse_slots(&v),
      vec![
        SlotBrief {
          id: 0,
          is_processing: false,
          n_prompt_tokens: None
        },
        SlotBrief {
          id: 1,
          is_processing: true,
          n_prompt_tokens: Some(4242)
        },
        SlotBrief {
          id: 2,
          is_processing: false,
          n_prompt_tokens: Some(0)
        },
      ]
    );
    assert!(parse_slots(&serde_json::json!({"error": "x"})).is_empty());
  }

  #[test]
  fn prune_noops_under_the_cap() {
    let files = vec![
      cf("/r/m1/slot-0.bin", 100, 5),
      cf("/r/m2/slot-0.bin", 100, 1),
    ];
    assert!(plan_prune(&files, 200, Path::new("/r/m3")).is_empty());
    assert!(plan_prune(&files, 201, Path::new("/r/m3")).is_empty());
  }

  #[test]
  fn prune_deletes_oldest_first_until_the_tree_fits() {
    let files = vec![
      cf("/r/m1/slot-0.bin", 60, 30),
      cf("/r/m2/slot-0.bin", 60, 20),
      cf("/r/m3/slot-0.bin", 60, 10),
    ];
    // total 180 → cap 130: free ≥ 50 → the oldest alone suffices.
    let victims = plan_prune(&files, 130, Path::new("/r/none"));
    assert_eq!(victims, vec![PathBuf::from("/r/m1/slot-0.bin")]);
    // cap 70: free ≥ 110 → oldest two.
    let tighter = plan_prune(&files, 70, Path::new("/r/none"));
    assert_eq!(
      tighter,
      vec![
        PathBuf::from("/r/m1/slot-0.bin"),
        PathBuf::from("/r/m2/slot-0.bin")
      ]
    );
  }

  #[test]
  fn prune_never_touches_the_launch_being_saved() {
    // The just-written file is the *only* thing over the cap: everything else
    // is deleted and the current launch's cache still survives its own save.
    let files = vec![
      cf("/r/m1/slot-0.bin", 50, 30),
      cf("/r/fresh/slot-0.bin", 200, 0),
    ];
    let victims = plan_prune(&files, 100, Path::new("/r/fresh"));
    assert_eq!(victims, vec![PathBuf::from("/r/m1/slot-0.bin")]);
    // Nothing else left to give: the keep set is not consumed even then.
    let solo = vec![cf("/r/fresh/slot-0.bin", 200, 0)];
    assert!(plan_prune(&solo, 100, Path::new("/r/fresh")).is_empty());
  }

  #[test]
  fn prune_removes_real_files_and_empty_model_dirs() {
    let root =
      std::env::temp_dir().join(format!("llamastash-slotcache-prune-{}", std::process::id()));
    let old = root.join("model-old");
    let fresh = root.join("model-fresh");
    std::fs::create_dir_all(&old).unwrap();
    std::fs::create_dir_all(&fresh).unwrap();
    std::fs::write(old.join("slot-0.bin"), vec![b'x'; 100]).unwrap();
    std::fs::write(fresh.join("slot-0.bin"), vec![b'y'; 100]).unwrap();
    // The fresh file is what was just written (keep); the old one is dead at
    // a 150-byte cap. Rewriting `fresh` makes its mtime strictly newer than
    // `old` without needing a clock tick or a filetime dep.
    std::fs::write(fresh.join("slot-0.bin"), vec![b'y'; 100]).unwrap();
    let deleted = prune(&root, 150, &fresh);
    assert_eq!(deleted, 1);
    assert!(!old.exists(), "empty model dir removed with its file");
    assert!(fresh.join("slot-0.bin").exists(), "keep set survives");
    std::fs::remove_dir_all(&root).ok();
  }

  // ---- HTTP client against a stub server ----

  /// Canned-response TCP stub: serves `routes` (in order of first match on
  /// `"METHOD /path"`) forever, one connection per call.
  async fn stub_server(routes: Vec<(String, u16, String)>) -> (u16, tokio::task::JoinHandle<()>) {
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let port = listener.local_addr().unwrap().port();
    let task = tokio::spawn(async move {
      loop {
        let (mut sock, _) = match listener.accept().await {
          Ok(s) => s,
          Err(_) => break,
        };
        let mut buf = [0u8; 4096];
        let n = sock.read(&mut buf).await.unwrap_or(0);
        let head = String::from_utf8_lossy(&buf[..n]).into_owned();
        let req_line = head.lines().next().unwrap_or("");
        let (method, path) = (
          req_line.split_whitespace().next().unwrap_or(""),
          req_line
            .split_whitespace()
            .nth(1)
            .map(|p| p.split('?').next().unwrap_or(p).to_string())
            .unwrap_or_default(),
        );
        let key = format!("{method} {path}");
        let (status, body) = routes
          .iter()
          .find(|(k, _, _)| *k == key)
          .map(|(_, s, b)| (*s, b.clone()))
          .unwrap_or((404, "{}".to_string()));
        let resp = format!(
          "HTTP/1.1 {status} OK\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{body}",
          body.len()
        );
        sock.write_all(resp.as_bytes()).await.ok();
      }
    });
    (port, task)
  }

  #[tokio::test]
  async fn list_slots_parses_the_live_shape() {
    let (port, task) = stub_server(vec![(
      "GET /slots".into(),
      200,
      r#"[{"id":0,"n_ctx":8192,"is_processing":false,"n_prompt_tokens":1234}]"#.into(),
    )])
    .await;
    let slots = list_slots(port, Duration::from_secs(5)).await.unwrap();
    assert_eq!(slots[0].n_prompt_tokens, Some(1234));
    task.abort();
  }

  #[tokio::test]
  async fn slot_actions_post_the_filename_and_read_the_count() {
    let (port, task) = stub_server(vec![
      (
        "POST /slots/0".into(),
        200,
        r#"{"id_slot":0,"filename":"slot-0.bin","n_saved":99,"n_written":7}"#.into(),
      ),
      (
        "POST /slots/1".into(),
        400,
        r#"{"error":{"code":400,"message":"Unable to restore slot: does not fit"}}"#.into(),
      ),
    ])
    .await;
    assert_eq!(
      save_slot(port, 0, "slot-0.bin", Duration::from_secs(5))
        .await
        .unwrap(),
      99
    );
    let err = restore_slot(port, 1, "slot-1.bin", Duration::from_secs(5))
      .await
      .unwrap_err();
    assert!(
      err.to_string().contains("does not fit"),
      "the engine's own message rides through: {err}"
    );
    task.abort();
  }

  #[tokio::test]
  async fn save_all_skips_empty_slots_and_survives_a_failing_one() {
    let (port, task) = stub_server(vec![
      (
        "GET /slots".into(),
        200,
        r#"[{"id":0,"is_processing":false},
            {"id":1,"is_processing":false,"n_prompt_tokens":10},
            {"id":2,"is_processing":true,"n_prompt_tokens":20}]"#
          .into(),
      ),
      (
        "POST /slots/1".into(),
        200,
        r#"{"n_saved":10,"n_written":330}"#.into(),
      ),
      (
        "POST /slots/2".into(),
        500,
        r#"{"error":{"message":"Unable to save slot"}}"#.into(),
      ),
    ])
    .await;
    let (sum, failed) = save_all(port, Duration::from_secs(5)).await.unwrap();
    assert_eq!(
      sum,
      SaveSummary {
        slots_saved: 1,
        tokens: 10
      }
    );
    assert_eq!(failed, 1, "slot 2's failure is counted, not fatal");
    task.abort();
  }

  #[tokio::test]
  async fn restore_all_restores_each_saved_slot_and_keeps_going() {
    let dir = std::env::temp_dir().join(format!(
      "llamastash-slotcache-restore-{}",
      std::process::id()
    ));
    std::fs::create_dir_all(&dir).unwrap();
    std::fs::write(dir.join("slot-0.bin"), b"kv").unwrap();
    std::fs::write(dir.join("slot-1.bin"), b"kv").unwrap();
    std::fs::write(dir.join("notes.txt"), b"stray").unwrap();
    let (port, task) = stub_server(vec![
      (
        "POST /slots/0".into(),
        200,
        r#"{"id_slot":0,"n_restored":42}"#.into(),
      ),
      (
        "POST /slots/1".into(),
        400,
        r#"{"error":{"message":"Restored prompt does not fit in the slot context"}}"#.into(),
      ),
    ])
    .await;
    let (sum, failed) = restore_all(port, &dir, Duration::from_secs(5))
      .await
      .unwrap();
    assert_eq!(
      sum,
      RestoreSummary {
        slots_restored: 1,
        tokens: 42
      }
    );
    assert_eq!(failed, 1);
    task.abort();
    std::fs::remove_dir_all(&dir).ok();
  }

  #[tokio::test]
  async fn http_errors_surface_status_and_message() {
    let (port, task) = stub_server(vec![(
      "GET /slots".into(),
      501,
      r#"{"error":{"code":501,"message":"This server does not support slots action. Start it with `--slot-save-path`"}}"#.into(),
    )])
    .await;
    let err = list_slots(port, Duration::from_secs(5)).await.unwrap_err();
    assert!(err.to_string().contains("HTTP 501"));
    assert!(err.to_string().contains("does not support slots action"));
    task.abort();
  }
}
