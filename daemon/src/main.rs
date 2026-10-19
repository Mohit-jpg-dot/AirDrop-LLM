// =============================================================================
// AirDrop-LLM — Daemon Entry Point
// =============================================================================
//
// The main orchestration hub that initializes all subsystems based on the
// runtime mode (master or worker) and coordinates the full daemon lifecycle:
//
//   Master mode (`airdrop start`):
//     1. Bind UDP multicast listener for peer discovery.
//     2. Start the embedded HTTP UI server on the configured port.
//     3. Accept worker registrations into the cluster registry.
//     4. Run network benchmarks against each new peer.
//     5. Compute the asymmetric scheduling plan.
//     6. Coordinate distributed inference sessions.
//
//   Worker mode (`airdrop join`):
//     1. Broadcast UDP discovery ping to find the master node.
//     2. Register hardware profile with the master.
//     3. Establish TCP data channel for tensor streaming.
//     4. Execute assigned inference layers on local hardware.
//
// =============================================================================

mod cluster;
mod discovery;
mod benchmark;
mod scheduler;
mod tensor;
mod transport;
mod fallback;
mod web;
mod delta;
mod supervisor;
mod orchestrator;
mod gateway;

use clap::Parser;
use std::sync::Arc;
use tokio::sync::RwLock;
use tracing::{info, warn, error};

use cluster::{ClusterState, NodeRole};

// ---------------------------------------------------------------------------
// CLI Argument Definition
// ---------------------------------------------------------------------------

/// AirDrop-LLM: Offline distributed inference daemon.
#[derive(Parser, Debug, Clone)]
#[command(name = "airdrop-daemon", version, about)]
struct Args {
    /// Execution mode: 'master' (cluster head) or 'worker' (peer node).
    #[arg(long, default_value = "master")]
    mode: String,

    /// Port for the embedded management UI HTTP server.
    #[arg(long, default_value_t = 3000)]
    ui_port: u16,

    /// Path to the GGUF model file for local inference.
    #[arg(long)]
    model: Option<String>,

    /// Enable verbose debug logging output.
    #[arg(long, default_value_t = false)]
    verbose: bool,
}

// ---------------------------------------------------------------------------
// Daemon Entrypoint
// ---------------------------------------------------------------------------

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    let args = Args::parse();

    // Initialize structured tracing subscriber with level filter.
    // Verbose mode enables DEBUG-level output; default is INFO.
    let log_filter = if args.verbose { "debug" } else { "info" };
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| tracing_subscriber::EnvFilter::new(log_filter)),
        )
        .with_target(true)
        .with_thread_ids(true)
        .init();

    info!("═══════════════════════════════════════════════════");
    info!("  AirDrop-LLM Daemon v{}", env!("CARGO_PKG_VERSION"));
    info!("  Mode: {}", args.mode.to_uppercase());
    info!("═══════════════════════════════════════════════════");

    // Build the shared cluster state object.
    // This is the central truth store accessed by all subsystems through
    // an Arc<RwLock<>> for safe concurrent reads with exclusive writes.
    let cluster_state = Arc::new(RwLock::new(ClusterState::new(args.model.clone())));

    // Determine execution role and launch the appropriate subsystem tree.
    let role = match args.mode.to_lowercase().as_str() {
        "master" => NodeRole::Master,
        "worker" => NodeRole::Worker,
        other => {
            error!("Invalid mode '{}'. Expected 'master' or 'worker'.", other);
            std::process::exit(1);
        }
    };

    match role {
        NodeRole::Master => run_master(args, cluster_state).await?,
        NodeRole::Worker => run_worker(args, cluster_state).await?,
    }

    Ok(())
}

// ---------------------------------------------------------------------------
// Master Node Execution Pipeline
// ---------------------------------------------------------------------------

/// Launches the full master node subsystem stack:
///   - UDP discovery listener (background task)
///   - TCP transport acceptor (background task)
///   - Embedded HTTP UI server (background task)
///   - Inference orchestration loop (foreground)
async fn run_master(args: Args, state: Arc<RwLock<ClusterState>>) -> anyhow::Result<()> {
    info!("Initializing master node subsystems...");

    // Collect and log local hardware profile
    let hw_profile = cluster::detect_hardware_profile();
    {
        let mut s = state.write().await;
        s.set_local_node(0, hw_profile.clone(), NodeRole::Master);
    }
    info!("Local hardware: {}", hw_profile);

    // --- Subsystem 1: UDP Multicast Discovery Listener ---
    // Listens on port 9998 for worker PING broadcasts and registers them.
    let discovery_state = Arc::clone(&state);
    let discovery_handle = tokio::spawn(async move {
        if let Err(e) = discovery::run_master_listener(discovery_state).await {
            error!("Discovery listener failed: {}", e);
        }
    });
    info!("UDP discovery listener active on port {}", discovery::DISCOVERY_PORT);

    // --- Subsystem 2: TCP Transport Acceptor ---
    // Accepts point-to-point data channels from registered workers.
    let transport_state = Arc::clone(&state);
    let transport_handle = tokio::spawn(async move {
        if let Err(e) = transport::run_tcp_acceptor(transport_state).await {
            error!("TCP transport acceptor failed: {}", e);
        }
    });
    info!("TCP transport acceptor active on port {}", transport::DATA_PORT);

    // --- Subsystem 3: Embedded HTTP UI Server ---
    let ui_state = Arc::clone(&state);
    let ui_port = args.ui_port;
    let web_handle = tokio::spawn(async move {
        if let Err(e) = web::run_ui_server(ui_port, ui_state).await {
            error!("UI server failed: {}", e);
        }
    });
    info!("Management UI server starting on http://localhost:{}", args.ui_port);

    // --- Subsystem 4: Performance Supervisor & FIFO Task Orchestrator ---
    let supervisor = supervisor::NetworkSupervisor::new(Arc::clone(&state));
    let orchestrator = orchestrator::InferenceOrchestrator::new(
        Arc::clone(&state),
        Arc::clone(&supervisor),
    );

    // --- Subsystem 5: OpenAI-Compatible REST API Gateway (:9995) ---
    let gw_state = Arc::clone(&state);
    let gw_orch = Arc::clone(&orchestrator);
    let gateway_handle = tokio::spawn(async move {
        if let Err(e) = gateway::run_openai_gateway(
            gateway::OPENAI_GATEWAY_PORT,
            gw_state,
            gw_orch,
        )
        .await
        {
            error!("OpenAI API Gateway failed: {}", e);
        }
    });
    info!("OpenAI REST API Gateway active on http://localhost:{}", gateway::OPENAI_GATEWAY_PORT);

    // --- Subsystem 6: Inference Orchestration Loop ---
    // This runs in the foreground, blocking until shutdown is signaled.
    let orch_state = Arc::clone(&state);
    let orch_handle = tokio::spawn(async move {
        orchestrate_inference(orch_state).await;
    });

    // Wait for all subsystems (runs until SIGTERM or fatal error)
    tokio::select! {
        _ = discovery_handle => warn!("Discovery subsystem exited."),
        _ = transport_handle => warn!("Transport subsystem exited."),
        _ = web_handle => warn!("UI server subsystem exited."),
        _ = gateway_handle => warn!("OpenAI Gateway subsystem exited."),
        _ = orch_handle => warn!("Orchestration subsystem exited."),
        _ = tokio::signal::ctrl_c() => {
            info!("Received shutdown signal. Terminating daemon...");
        }
    }

    info!("Master daemon shutdown complete.");
    Ok(())
}

// ---------------------------------------------------------------------------
// Worker Node Execution Pipeline
// ---------------------------------------------------------------------------

/// Launches the worker node subsystem stack:
///   - UDP discovery broadcaster (registers with master)
///   - TCP data channel establishment
///   - Local inference executor (waits for layer assignments)
async fn run_worker(args: Args, state: Arc<RwLock<ClusterState>>) -> anyhow::Result<()> {
    info!("Initializing worker node subsystems...");

    // Collect local hardware profile for registration payload
    let hw_profile = cluster::detect_hardware_profile();
    {
        let mut s = state.write().await;
        s.set_local_node(u32::MAX, hw_profile.clone(), NodeRole::Worker);
    }
    info!("Local hardware: {}", hw_profile);

    // --- Step 1: Discover Master Node ---
    // Broadcast AIRDROP_LLM_PING on UDP port 9998 until the master responds.
    info!("Broadcasting discovery ping to find cluster master...");
    let master_addr = discovery::run_worker_broadcast().await?;
    info!("Master node discovered at: {}", master_addr);

    // --- Step 2: Register with Master ---
    // Send hardware profile payload over TCP to the master's registration port.
    transport::register_with_master(&master_addr, &hw_profile).await?;
    info!("Successfully registered with master node.");

    // --- Step 3: Establish Persistent TCP Data Channel ---
    // This channel is used for bidirectional tensor streaming during inference.
    let data_channel = transport::connect_data_channel(&master_addr).await?;
    info!("Data channel established with master.");

    // --- Step 4: Worker Execution Loop ---
    // Waits for layer execution commands from master, runs local inference,
    // and streams intermediate tensors back over the data channel.
    let worker_state = Arc::clone(&state);
    let exec_handle = tokio::spawn(async move {
        tensor::worker_execution_loop(data_channel, worker_state, args.model).await;
    });

    tokio::select! {
        _ = exec_handle => warn!("Worker execution loop exited."),
        _ = tokio::signal::ctrl_c() => {
            info!("Received shutdown signal. Terminating worker...");
        }
    }

    info!("Worker daemon shutdown complete.");
    Ok(())
}

// ---------------------------------------------------------------------------
// Inference Orchestration (Master)
// ---------------------------------------------------------------------------

/// Master-side inference orchestration loop.
///
/// This loop waits for inference requests (from the UI or API), checks
/// the cluster state for available workers, computes the scheduling plan,
/// dispatches layer assignments, and coordinates tensor streaming.
async fn orchestrate_inference(state: Arc<RwLock<ClusterState>>) {
    info!("Inference orchestration loop started. Waiting for requests...");

    loop {
        // Check for pending inference requests from the UI/API layer
        let request = {
            let s = state.read().await;
            s.dequeue_inference_request()
        };

        if let Some(request) = request {
            info!("Processing inference request: {}", request.id);

            // Read the current cluster topology and benchmark results
            let cluster_snapshot = {
                let s = state.read().await;
                s.snapshot()
            };

            // Compute the asymmetric layer scheduling plan
            let schedule = scheduler::compute_schedule(&cluster_snapshot);
            info!("Computed schedule: {:?}", schedule);

            // Execute the distributed inference pipeline
            let result = tensor::run_pipeline(
                &schedule,
                &cluster_snapshot,
                &request,
                Arc::clone(&state),
            )
            .await;

            match result {
                Ok(output) => {
                    info!("Inference complete for request {}", request.id);
                    let mut s = state.write().await;
                    s.complete_inference_request(&request.id, output);
                }
                Err(e) => {
                    warn!("Distributed inference failed: {}. Activating fallback.", e);
                    // Trigger the local fallback protocol
                    let fallback_output = fallback::run_local_fallback(
                        &request,
                        Arc::clone(&state),
                    )
                    .await;

                    let mut s = state.write().await;
                    s.complete_inference_request(&request.id, fallback_output);
                }
            }
        } else {
            // No pending requests — yield the executor for 100ms
            tokio::time::sleep(tokio::time::Duration::from_millis(100)).await;
        }
    }
}
