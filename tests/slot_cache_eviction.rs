//! Prompt-cache persistence across an eviction: the llama.cpp save/restore
//! hooks against the fake `llama-server`'s slot endpoints.
//!
//! The fixture writes a marker file per `action=save|restore` into its
//! `--slot-save-path`, so a test can prove the daemon saved the slot *through
//! the engine's own endpoint* before the evict stop, and restored it during
//! readiness before the launch reached `Ready`.

#![cfg(feature = "test-fixtures")]

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::Duration;

use llamastash::backend::llama_cpp::{
  LlamaCppBackend, LLAMACPP_KNOB_SLOT_DIR, LLAMACPP_KNOB_SLOT_MAX_BYTES, LLAMACPP_KNOB_SLOT_SAVE,
  LLAMACPP_KNOB_SLOT_TTL_SECS,
};
use llamastash::config::loader::PortRange;
use llamastash::daemon::context::{LaunchEnv, MethodContext, PersistedState};
use llamastash::daemon::probe::ProbeOptions;
use llamastash::daemon::registry::SupervisorRegistry;
use llamastash::daemon::shutdown::ShutdownToken;
use llamastash::daemon::state_store::DaemonState;
use llamastash::daemon::supervisor::{
  spawn as supervisor_spawn, LaunchOrigin, ManagedModel, ManagedSpawn, ManagedState,
};
use llamastash::discovery::ModelCatalog;
use llamastash::gguf::identity::ModelId;
use llamastash::launch::mode::LaunchMode;
use llamastash::launch::params::LaunchParams;
use llamastash::proxy::eviction;
use llamastash::proxy::state::ProxyState;
use llamastash::proxy::DEFAULT_BODY_LIMIT_BYTES;
use tokio::time::sleep;

fn unique_temp(label: &str) -> PathBuf {
  llamastash::test_support::unique_temp_dir("ls-sc", label)
}

fn fake_binary() -> PathBuf {
  PathBuf::from(env!("CARGO_BIN_EXE_fake_llama_server"))
}

fn fast_probe() -> ProbeOptions {
  ProbeOptions {
    interval: Duration::from_millis(30),
    timeout: Duration::from_secs(15),
  }
}

fn allocate_port() -> u16 {
  let l = std::net::TcpListener::bind("127.0.0.1:0").expect("bind ephemeral");
  l.local_addr().unwrap().port()
}

fn allocate_port_range() -> PortRange {
  llamastash::test_support::allocate_port_range(8)
}

async fn wait_for_ready(model: &ManagedModel) {
  let deadline = std::time::Instant::now() + Duration::from_secs(5);
  loop {
    if matches!(model.state().await, ManagedState::Ready) {
      return;
    }
    if std::time::Instant::now() > deadline {
      panic!("supervisor never reached Ready");
    }
    sleep(Duration::from_millis(20)).await;
  }
}

/// Seed what `seed_launch_knobs` + `seed_binary_caps` would project for a
/// slot-cache-enabled launch: the per-model directory plus the cap/TTL and
/// the per-binary veto.
fn slot_cache_params(slot_dir: &Path) -> LaunchParams {
  let mut params = LaunchParams::new(PathBuf::from("/tmp/ls-sc.gguf"), LaunchMode::Chat);
  params
    .launch_config
    .insert(LLAMACPP_KNOB_SLOT_SAVE.to_string(), "true".to_string());
  params.launch_config.insert(
    LLAMACPP_KNOB_SLOT_DIR.to_string(),
    slot_dir.display().to_string(),
  );
  params
    .launch_config
    .insert(LLAMACPP_KNOB_SLOT_MAX_BYTES.to_string(), "4096".to_string());
  params.launch_config.insert(
    LLAMACPP_KNOB_SLOT_TTL_SECS.to_string(),
    (14u64 * 86_400).to_string(),
  );
  params
}

async fn pre_launch(
  log_dir: &Path,
  registry: &SupervisorRegistry,
  params: LaunchParams,
) -> ManagedModel {
  let port = allocate_port();
  let id = ModelId {
    path: PathBuf::from(format!("/tmp/ls-sc-{port}.gguf")),
    header_blake3: [0u8; 32],
  };
  // `process_spec` composes the argv, so `--slot-save-path` reaches the
  // fixture exactly the way a real launch would.
  let plan = LlamaCppBackend::new().process_spec(&params, port, fake_binary(), fast_probe());
  let model = supervisor_spawn(ManagedSpawn {
    id,
    params,
    port,
    mode: LaunchMode::Chat,
    log_path: log_dir.join("slot.log"),
    plan,
    origin: LaunchOrigin::AutoStart,
    fit_gate: None,
    resolved_backend: "llamacpp".to_string(),
  })
  .await
  .expect("spawn");
  wait_for_ready(&model).await;
  let launch_id = registry.next_id();
  registry.insert(launch_id, model.clone()).await;
  model
}

async fn build_state(registry: SupervisorRegistry) -> Arc<ProxyState> {
  let catalog = ModelCatalog::new();
  let token = ShutdownToken::new();
  let env = LaunchEnv {
    binary: Some(fake_binary()),
    port_range: allocate_port_range(),
    log_dir: PathBuf::from("/tmp/ls-sc-logs"),
    probe: fast_probe(),
    arch_defaults: BTreeMap::new(),
    servers: Default::default(),
    default_launch_mode: Default::default(),
  };
  let ctx = MethodContext::with_catalog(token, catalog)
    .with_supervisors(registry)
    .with_state(PersistedState::new(DaemonState::default(), None))
    .with_launch_env(env);
  ProxyState::from_context(&ctx, false, true, DEFAULT_BODY_LIMIT_BYTES)
}

/// An evicted llama.cpp launch saves its prompt-bearing slot to the slot
/// directory before the stop lands.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn eviction_saves_the_slot_before_stopping() {
  let dir = unique_temp("evict-save");
  let log_dir = dir.join("logs");
  std::fs::create_dir_all(&log_dir).unwrap();
  let slot_dir = dir.join("slot-cache");
  let registry = SupervisorRegistry::new();
  let model = pre_launch(&log_dir, &registry, slot_cache_params(&slot_dir)).await;
  let state = build_state(registry).await;

  state.touch_mru(model.id()).await;
  sleep(Duration::from_millis(5)).await;
  eviction::sweep_once(&state, Duration::from_nanos(1)).await;

  let deadline = std::time::Instant::now() + Duration::from_secs(5);
  loop {
    if matches!(model.state().await, ManagedState::Stopped) {
      break;
    }
    assert!(
      std::time::Instant::now() < deadline,
      "supervisor never stopped: {:?}",
      model.state().await
    );
    sleep(Duration::from_millis(20)).await;
  }
  // The fixture records one marker per engine call: the daemon's save rode
  // the real `POST /slots/{id}?action=save`, and the save landed while the
  // server was still up (a stopped server could not have answered it).
  assert!(
    slot_dir.join("save-slot-0.bin").exists(),
    "evict-time save marker missing: {:?}",
    std::fs::read_dir(&slot_dir).map(|r| r.flatten().map(|e| e.file_name()).collect::<Vec<_>>())
  );
  std::fs::remove_dir_all(&dir).ok();
}

/// Files in the slot directory are restored through the engine during
/// readiness: by the time the launch reports `Ready`, the slot is already
/// warm, so the first request hits the cache rather than racing the restore.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn restore_happens_before_the_launch_is_ready() {
  let dir = unique_temp("restore-ready");
  let log_dir = dir.join("logs");
  std::fs::create_dir_all(&log_dir).unwrap();
  let slot_dir = dir.join("slot-cache");
  std::fs::create_dir_all(&slot_dir).unwrap();
  // The file a previous eviction saved for slot 0.
  std::fs::write(slot_dir.join("slot-0.bin"), b"kv").unwrap();
  let registry = SupervisorRegistry::new();
  let model = pre_launch(&log_dir, &registry, slot_cache_params(&slot_dir)).await;
  assert!(matches!(model.state().await, ManagedState::Ready));
  assert!(
    slot_dir.join("restore-slot-0.bin").exists(),
    "slot was not restored during readiness: {:?}",
    model.state().await,
  );
  std::fs::remove_dir_all(&dir).ok();
}

/// The over-cap sweep at save time drops the oldest file.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn a_second_eviction_prunes_over_cap_files() {
  let dir = unique_temp("evict-cap");
  let log_dir = dir.join("logs");
  std::fs::create_dir_all(&log_dir).unwrap();
  let slot_dir = dir.join("slot-cache");
  std::fs::create_dir_all(&slot_dir).unwrap();
  // A resident old file at the whole cap, written first so it is the
  // oldest-mtime file. The new save (small) must push the directory over the
  // cap and prune the old file.
  let old = slot_dir.join("slot-3.bin");
  std::fs::write(&old, vec![b'x'; 4096]).unwrap();
  let registry = SupervisorRegistry::new();
  let model = pre_launch(&log_dir, &registry, slot_cache_params(&slot_dir)).await;
  let state = build_state(registry).await;

  state.touch_mru(model.id()).await;
  sleep(Duration::from_millis(5)).await;
  eviction::sweep_once(&state, Duration::from_nanos(1)).await;

  let deadline = std::time::Instant::now() + Duration::from_secs(5);
  loop {
    if matches!(model.state().await, ManagedState::Stopped) {
      break;
    }
    assert!(
      std::time::Instant::now() < deadline,
      "supervisor never stopped: {:?}",
      model.state().await
    );
    sleep(Duration::from_millis(20)).await;
  }
  assert!(slot_dir.join("save-slot-0.bin").exists());
  assert!(
    !old.exists(),
    "the over-cap oldest file should have been pruned"
  );
  std::fs::remove_dir_all(&dir).ok();
}
