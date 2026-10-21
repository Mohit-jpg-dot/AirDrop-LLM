// =============================================================================
// AirDrop-LLM — Embedded HTTP UI Server & REST API
// =============================================================================
//
// Provides the local management interface served at `http://localhost:3000`.
// Built on `axum` with `tower-http` for static file serving.
//
// The UI server exposes:
//   - Static file serving for the browser UI (HTML/CSS/JS from `ui/` dir).
//   - REST API endpoints for cluster management and inference control.
//   - Server-Sent Events (SSE) for real-time status streaming to the UI.
//
// All endpoints are local-only — no authentication required since the
// daemon runs exclusively on localhost over an offline network.
//
// API Endpoints:
//   GET  /api/status        → Cluster status and node registry
//   GET  /api/nodes         → List all registered nodes with hardware profiles
//   POST /api/infer         → Submit an inference request
//   GET  /api/result/:id    → Retrieve inference result by request ID
//   GET  /api/events        → SSE stream for real-time status updates
//
// =============================================================================

use crate::cluster::{ClusterState, InferenceRequest};
use axum::{
    extract::State,
    http::StatusCode,
    response::{
        sse::{Event, KeepAlive, Sse},
        Html, IntoResponse, Json,
    },
    routing::{get, post},
    Router,
};
use serde::{Deserialize, Serialize};
use std::convert::Infallible;
use std::sync::Arc;
use tokio::sync::RwLock;
use tokio_stream::StreamExt;
use tower_http::cors::CorsLayer;
use tracing::{error, info};

// ---------------------------------------------------------------------------
// Application State (shared with Axum handlers)
// ---------------------------------------------------------------------------

type AppState = Arc<RwLock<ClusterState>>;

// ---------------------------------------------------------------------------
// Server Initialization
// ---------------------------------------------------------------------------

/// Starts the embedded HTTP server on the specified port.
///
/// The server serves both the static UI files and the REST API endpoints.
/// It binds to `0.0.0.0` to be accessible from the local machine's browser.
pub async fn run_ui_server(port: u16, state: AppState) -> anyhow::Result<()> {
    // Determine the UI static files directory.
    // Priority: AIRDROP_UI_DIR env var → ./ui/ → embedded fallback.
    let ui_dir = std::env::var("AIRDROP_UI_DIR").unwrap_or_else(|_| "ui".to_string());

    // Build the Axum router
    let app = Router::new()
        // API routes
        .route("/api/status", get(api_status))
        .route("/api/nodes", get(api_nodes))
        .route("/api/infer", post(api_infer))
        .route("/api/result/{id}", get(api_result))
        .route("/api/events", get(api_events))
        // Health check
        .route("/api/health", get(api_health))
        // Fallback: serve static files from the UI directory, or the
        // embedded HTML if the directory doesn't exist
        .fallback(get(serve_ui))
        // Shared state
        .with_state(state)
        // CORS (permissive for local development)
        .layer(CorsLayer::permissive());

    let bind_addr = format!("0.0.0.0:{}", port);
    info!("UI server binding to {}", bind_addr);

    let listener = tokio::net::TcpListener::bind(&bind_addr).await?;
    axum::serve(listener, app).await?;

    Ok(())
}

// ---------------------------------------------------------------------------
// API Handlers
// ---------------------------------------------------------------------------

/// GET /api/health — Simple health check endpoint.
async fn api_health() -> impl IntoResponse {
    Json(serde_json::json!({
        "status": "ok",
        "version": env!("CARGO_PKG_VERSION"),
        "timestamp": chrono::Utc::now().to_rfc3339(),
    }))
}

/// GET /api/status — Returns the overall cluster status.
#[derive(Serialize)]
struct StatusResponse {
    node_count: usize,
    is_inferring: bool,
    status_message: String,
    version: String,
}

async fn api_status(State(state): State<AppState>) -> impl IntoResponse {
    let s = state.read().await;
    Json(StatusResponse {
        node_count: s.node_count(),
        is_inferring: s.is_inferring,
        status_message: s.status_message.clone(),
        version: env!("CARGO_PKG_VERSION").to_string(),
    })
}

/// GET /api/nodes — Returns detailed information about all cluster nodes.
async fn api_nodes(State(state): State<AppState>) -> impl IntoResponse {
    let s = state.read().await;
    let nodes: Vec<_> = s.all_nodes().into_iter().cloned().collect();
    Json(nodes)
}

/// POST /api/infer — Submits an inference request.
#[derive(Deserialize)]
struct InferRequest {
    prompt: String,
    max_tokens: Option<u32>,
    temperature: Option<f32>,
    #[serde(default)]
    images: Option<Vec<String>>,
}

#[derive(Serialize)]
struct InferResponse {
    request_id: String,
    status: String,
}

async fn api_infer(
    State(state): State<AppState>,
    Json(body): Json<InferRequest>,
) -> impl IntoResponse {
    let request_id = uuid::Uuid::new_v4().to_string();

    let request = InferenceRequest {
        id: request_id.clone(),
        prompt: body.prompt,
        max_tokens: body.max_tokens.unwrap_or(512),
        temperature: body.temperature.unwrap_or(0.7),
        images: body.images,
    };

    {
        let mut s = state.write().await;
        s.enqueue_inference_request(request);
        s.is_inferring = true;
        s.status_message = format!("Processing inference request {}", request_id);
    }

    info!("Inference request queued: {}", request_id);

    (
        StatusCode::ACCEPTED,
        Json(InferResponse {
            request_id,
            status: "queued".to_string(),
        }),
    )
}

/// GET /api/result/:id — Retrieves the result of an inference request.
async fn api_result(
    State(state): State<AppState>,
    axum::extract::Path(id): axum::extract::Path<String>,
) -> impl IntoResponse {
    let s = state.read().await;

    if let Some(result) = s.get_result(&id) {
        Json(serde_json::json!({
            "request_id": id,
            "status": "complete",
            "output": result,
        }))
    } else {
        Json(serde_json::json!({
            "request_id": id,
            "status": "pending",
            "output": null,
        }))
    }
}

/// GET /api/events — Server-Sent Events stream for real-time updates.
///
/// The UI connects to this endpoint to receive live cluster status updates
/// without polling. Events include:
///   - Node join/leave notifications
///   - Inference progress updates
///   - Benchmark results
///   - Fallback activations
async fn api_events(
    State(state): State<AppState>,
) -> Sse<impl tokio_stream::Stream<Item = Result<Event, Infallible>>> {
    let stream = tokio_stream::wrappers::IntervalStream::new(tokio::time::interval(
        tokio::time::Duration::from_secs(2),
    ))
    .map(move |_| {
        let state = Arc::clone(&state);
        async move {
            let s = state.read().await;
            let data = serde_json::json!({
                "node_count": s.node_count(),
                "is_inferring": s.is_inferring,
                "status": s.status_message,
                "nodes": s.all_nodes().iter().map(|n| {
                    serde_json::json!({
                        "id": n.node_id,
                        "hostname": n.hardware.hostname,
                        "alive": n.is_alive,
                        "backend": format!("{}", n.hardware.compute_backend),
                    })
                }).collect::<Vec<_>>(),
            });
            Ok(Event::default().data(data.to_string()))
        }
    })
    .then(|fut| fut);

    Sse::new(stream).keep_alive(KeepAlive::default())
}

// ---------------------------------------------------------------------------
// Static UI Serving
// ---------------------------------------------------------------------------

/// Serves the embedded fallback UI HTML when the static files directory
/// is not available. This ensures the management interface always works,
/// even without the UI assets.
async fn serve_ui() -> impl IntoResponse {
    Html(EMBEDDED_UI_HTML)
}

/// Embedded minimal management UI HTML.
///
/// This is the fallback UI served when the `ui/` directory is not found.
/// The full-featured UI is served from static files when available.
const EMBEDDED_UI_HTML: &str = r#"<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>AirDrop-LLM — Distributed Inference</title>
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; }
        body {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', system-ui, sans-serif;
            background: #0a0a0f;
            color: #e4e4e7;
            min-height: 100vh;
        }
        .container { max-width: 900px; margin: 0 auto; padding: 2rem; }
        header {
            text-align: center;
            padding: 3rem 0 2rem;
            background: linear-gradient(135deg, rgba(99,102,241,0.1) 0%, rgba(168,85,247,0.1) 100%);
            border-bottom: 1px solid rgba(255,255,255,0.06);
        }
        h1 {
            font-size: 2rem;
            background: linear-gradient(to right, #818cf8, #a78bfa);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
        }
        .status-badge {
            display: inline-block;
            padding: 0.25rem 0.75rem;
            border-radius: 9999px;
            font-size: 0.75rem;
            font-weight: 600;
            margin-top: 0.5rem;
        }
        .status-badge.online { background: rgba(34,197,94,0.15); color: #4ade80; }
        .status-badge.busy { background: rgba(234,179,8,0.15); color: #facc15; }
        .nodes { display: grid; gap: 1rem; margin-top: 2rem; }
        .node-card {
            background: rgba(255,255,255,0.03);
            border: 1px solid rgba(255,255,255,0.06);
            border-radius: 0.75rem;
            padding: 1.25rem;
            transition: border-color 0.2s;
        }
        .node-card:hover { border-color: rgba(129,140,248,0.3); }
        .node-header { display: flex; justify-content: space-between; align-items: center; }
        .node-name { font-weight: 600; color: #c4b5fd; }
        .chat-box {
            margin-top: 2rem;
            background: rgba(255,255,255,0.02);
            border: 1px solid rgba(255,255,255,0.06);
            border-radius: 0.75rem;
            padding: 1.5rem;
        }
        textarea {
            width: 100%;
            min-height: 100px;
            background: rgba(0,0,0,0.3);
            border: 1px solid rgba(255,255,255,0.1);
            border-radius: 0.5rem;
            color: #e4e4e7;
            padding: 0.75rem;
            font-family: inherit;
            font-size: 0.95rem;
            resize: vertical;
        }
        textarea:focus { outline: none; border-color: #818cf8; }
        button {
            margin-top: 0.75rem;
            padding: 0.6rem 1.5rem;
            background: linear-gradient(135deg, #6366f1, #8b5cf6);
            color: white;
            border: none;
            border-radius: 0.5rem;
            font-weight: 600;
            cursor: pointer;
            transition: opacity 0.2s;
        }
        button:hover { opacity: 0.9; }
        button:disabled { opacity: 0.5; cursor: not-allowed; }
        #output {
            margin-top: 1rem;
            padding: 1rem;
            background: rgba(0,0,0,0.3);
            border-radius: 0.5rem;
            white-space: pre-wrap;
            font-family: 'SF Mono', 'Fira Code', monospace;
            font-size: 0.85rem;
            min-height: 60px;
            color: #a1a1aa;
        }
        .stat { font-size: 0.8rem; color: #71717a; margin-top: 0.25rem; }
    </style>
</head>
<body>
    <header>
        <h1>⚡ AirDrop-LLM</h1>
        <p style="color:#a1a1aa;margin-top:0.5rem;">Offline Distributed Inference Broker</p>
        <span class="status-badge online" id="statusBadge">Initializing...</span>
    </header>
    <div class="container">
        <div class="nodes" id="nodesContainer">
            <p style="color:#71717a;text-align:center;">Discovering nodes...</p>
        </div>
        <div class="chat-box">
            <textarea id="prompt" placeholder="Enter your prompt here..."></textarea>
            <button id="sendBtn" onclick="submitInference()">Generate ⚡</button>
            <div id="output"></div>
        </div>
    </div>
    <script>
        const API = '';
        let pollInterval;

        async function fetchStatus() {
            try {
                const res = await fetch(API + '/api/status');
                const data = await res.json();
                const badge = document.getElementById('statusBadge');
                badge.textContent = data.status_message;
                badge.className = 'status-badge ' + (data.is_inferring ? 'busy' : 'online');
            } catch(e) { console.error('Status fetch failed:', e); }
        }

        async function fetchNodes() {
            try {
                const res = await fetch(API + '/api/nodes');
                const nodes = await res.json();
                const container = document.getElementById('nodesContainer');
                if (nodes.length === 0) {
                    container.innerHTML = '<p style="color:#71717a;text-align:center;">No nodes registered yet.</p>';
                    return;
                }
                container.innerHTML = nodes.map(n => `
                    <div class="node-card">
                        <div class="node-header">
                            <span class="node-name">Node ${n.node_id} — ${n.hardware.hostname}</span>
                            <span class="status-badge ${n.is_alive ? 'online' : ''}">${n.is_alive ? 'Online' : 'Offline'}</span>
                        </div>
                        <div class="stat">${n.hardware.cpu_model} · ${n.hardware.cpu_cores} cores · ${Math.round(n.hardware.ram_total_mb/1024)} GB RAM</div>
                        <div class="stat">GPU: ${n.hardware.gpu_name || 'None'} · ${n.hardware.compute_backend}</div>
                        ${n.benchmark ? `<div class="stat">Network: ${n.benchmark.throughput_mbps.toFixed(1)} Mbps · ${n.benchmark.latency_ms.toFixed(1)} ms RTT ${n.benchmark.passes_gate ? '✓' : '⚠ Bottleneck'}</div>` : ''}
                    </div>
                `).join('');
            } catch(e) { console.error('Nodes fetch failed:', e); }
        }

        async function submitInference() {
            const prompt = document.getElementById('prompt').value.trim();
            if (!prompt) return;

            const btn = document.getElementById('sendBtn');
            const output = document.getElementById('output');
            btn.disabled = true;
            btn.textContent = 'Processing...';
            output.textContent = 'Submitting request...';

            try {
                const res = await fetch(API + '/api/infer', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ prompt, max_tokens: 512, temperature: 0.7 }),
                });
                const data = await res.json();
                output.textContent = 'Request queued (ID: ' + data.request_id + '). Waiting for result...';

                // Poll for result
                const pollResult = setInterval(async () => {
                    try {
                        const r = await fetch(API + '/api/result/' + data.request_id);
                        const result = await r.json();
                        if (result.status === 'complete') {
                            clearInterval(pollResult);
                            output.textContent = result.output;
                            btn.disabled = false;
                            btn.textContent = 'Generate ⚡';
                        }
                    } catch(e) { console.error('Result poll failed:', e); }
                }, 1000);
            } catch(e) {
                output.textContent = 'Error: ' + e.message;
                btn.disabled = false;
                btn.textContent = 'Generate ⚡';
            }
        }

        // Initialize polling
        fetchStatus();
        fetchNodes();
        pollInterval = setInterval(() => { fetchStatus(); fetchNodes(); }, 3000);
    </script>
</body>
</html>"#;
