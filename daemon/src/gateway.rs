// =============================================================================
// AirDrop-LLM — OpenAI-Compatible Off-Heap REST API Gateway (:9995)
// =============================================================================
//
// High-performance, asynchronous HTTP reverse proxy server engine listening
// on local port 9995, built using Axum and Tokio.
//
// Complies with the standard OpenAI API specification for seamless, zero-config
// integration with IDE coding agents (OpenHands, Aider, Continue, Cursor, etc.).
//
// Endpoints:
//   - GET  /v1/models            → Returns standard active model registry catalog.
//   - POST /v1/chat/completions  → Intercepts message streams, user parameters,
//                                  and routes inference into the cluster FIFO orchestrator.
//                                  Supports continuous Server-Sent Events (SSE) streaming.
//   - GET  /health               → Liveness and capacity readiness probe.
//
// =============================================================================

use crate::cluster::ClusterState;
use crate::orchestrator::{InferenceOrchestrator, OrchestrationError, PriorityLevel};
use axum::{
    extract::State,
    http::{header, HeaderMap, HeaderValue, StatusCode},
    response::{
        sse::{Event, KeepAlive, Sse},
        IntoResponse, Response,
    },
    routing::{get, post},
    Json, Router,
};
use serde::{Deserialize, Serialize};
use std::convert::Infallible;
use std::sync::Arc;
use tokio::sync::{mpsc, RwLock};
use tokio_stream::wrappers::ReceiverStream;
use tokio_stream::StreamExt;
use tower_http::cors::CorsLayer;
use tracing::{debug, error, info, warn};
use uuid::Uuid;

// ---------------------------------------------------------------------------
// Constants & Configuration
// ---------------------------------------------------------------------------

/// Default listening port for the OpenAI-compatible gateway.
pub const OPENAI_GATEWAY_PORT: u16 = 9995;

/// Model identifier advertised to client IDEs.
pub const DEFAULT_MODEL_ID: &str = "pooled-large-core";

// ---------------------------------------------------------------------------
// OpenAI Data Transfer Objects (DTOs)
// ---------------------------------------------------------------------------

/// Standard model list container matching OpenAI specification.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelListResponse {
    pub object: String,
    pub data: Vec<ModelDescriptor>,
}

/// Individual model descriptor record.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelDescriptor {
    pub id: String,
    pub object: String,
    pub created: u64,
    pub owned_by: String,
    pub permission: Vec<String>,
    pub root: String,
    pub parent: Option<String>,
}

/// Incoming chat completion payload.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChatCompletionRequest {
    pub model: String,
    pub messages: Vec<ChatMessage>,
    #[serde(default)]
    pub temperature: Option<f32>,
    #[serde(default)]
    pub max_tokens: Option<u32>,
    #[serde(default)]
    pub stream: Option<bool>,
}

/// Message element within the conversation thread.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChatMessage {
    pub role: String,
    pub content: String,
}

/// Non-streaming chat completion response.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChatCompletionResponse {
    pub id: String,
    pub object: String,
    pub created: u64,
    pub model: String,
    pub choices: Vec<ChatChoice>,
    pub usage: UsageInfo,
}

/// Single choice element in non-streaming response.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChatChoice {
    pub index: usize,
    pub message: ChatMessage,
    pub finish_reason: Option<String>,
}

/// Token usage accounting statistics.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct UsageInfo {
    pub prompt_tokens: usize,
    pub completion_tokens: usize,
    pub total_tokens: usize,
}

/// Server-Sent Events (SSE) streaming chunk format.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChatCompletionChunk {
    pub id: String,
    pub object: String,
    pub created: u64,
    pub model: String,
    pub choices: Vec<ChunkChoice>,
}

/// Delta payload representing sequential token output.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChunkChoice {
    pub index: usize,
    pub delta: ChunkDelta,
    pub finish_reason: Option<String>,
}

/// Incremental text fragment within an SSE chunk.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChunkDelta {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub role: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub content: Option<String>,
}

/// Standardized API error payload.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ApiErrorEnvelope {
    pub error: ApiErrorDetail,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ApiErrorDetail {
    pub message: String,
    #[serde(rename = "type")]
    pub error_type: String,
    pub code: Option<String>,
}

// ---------------------------------------------------------------------------
// Shared Gateway State
// ---------------------------------------------------------------------------

#[derive(Clone)]
pub struct GatewayState {
    pub cluster_state: Arc<RwLock<ClusterState>>,
    pub orchestrator: Arc<InferenceOrchestrator>,
}

// ---------------------------------------------------------------------------
// Server Entry Point
// ---------------------------------------------------------------------------

/// Initializes and launches the Axum HTTP API gateway server on port 9995.
pub async fn run_openai_gateway(
    port: u16,
    cluster_state: Arc<RwLock<ClusterState>>,
    orchestrator: Arc<InferenceOrchestrator>,
) -> anyhow::Result<()> {
    let state = GatewayState {
        cluster_state,
        orchestrator,
    };

    let router = Router::new()
        .route("/v1/models", get(handle_get_models))
        .route("/v1/chat/completions", post(handle_chat_completions))
        .route("/health", get(handle_health_probe))
        .with_state(state)
        .layer(CorsLayer::permissive());

    let bind_addr = format!("0.0.0.0:{}", port);
    info!(
        "[GATEWAY] OpenAI-compatible reverse proxy listening on http://{}",
        bind_addr
    );

    let listener = tokio::net::TcpListener::bind(&bind_addr).await?;
    axum::serve(listener, router).await?;

    Ok(())
}

// ---------------------------------------------------------------------------
// Route Handlers
// ---------------------------------------------------------------------------

/// GET /v1/models — Advertises the active model cluster to IDE tools.
async fn handle_get_models() -> impl IntoResponse {
    let now = chrono::Utc::now().timestamp() as u64;
    let catalog = ModelListResponse {
        object: "list".to_string(),
        data: vec![ModelDescriptor {
            id: DEFAULT_MODEL_ID.to_string(),
            object: "model".to_string(),
            created: now,
            owned_by: "airdrop-llm".to_string(),
            permission: vec![],
            root: DEFAULT_MODEL_ID.to_string(),
            parent: None,
        }],
    };

    Json(catalog)
}

/// GET /health — Gateway liveness and supervisor health indicator.
async fn handle_health_probe(State(state): State<GatewayState>) -> impl IntoResponse {
    let queue_depth = state.orchestrator.pending_tasks_count();
    let is_degraded = state.orchestrator.is_degraded_mode().await;

    Json(serde_json::json!({
        "status": if is_degraded { "degraded" } else { "healthy" },
        "engine": "AirDrop-LLM Multi-Node Orchestrator",
        "port": OPENAI_GATEWAY_PORT,
        "queue_depth": queue_depth,
        "active_slot_busy": state.orchestrator.is_slot_busy(),
        "degraded_fallback_active": is_degraded,
    }))
}

/// POST /v1/chat/completions — Main dispatch pipeline.
///
/// Parses incoming chat history, transforms messages into an execution prompt,
/// submits the task to the FIFO orchestrator, and either streams SSE chunks or
/// awaits the unified response string.
async fn handle_chat_completions(
    State(state): State<GatewayState>,
    Json(payload): Json<ChatCompletionRequest>,
) -> Response {
    let req_id = format!("chatcmpl-{}", Uuid::new_v4());
    let prompt_text = format_chatml_prompt(&payload.messages);
    let is_streaming = payload.stream.unwrap_or(false);
    let max_tokens = payload.max_tokens.unwrap_or(1024);
    let temperature = payload.temperature.unwrap_or(0.7);

    debug!(
        "[GATEWAY] Received chat completion: id={} stream={} max_tokens={}",
        req_id, is_streaming, max_tokens
    );

    if is_streaming {
        // --- STREAMING MODE (SSE) ---
        let (token_tx, token_rx) = mpsc::channel::<String>(256);

        // Submit to the FIFO Task Orchestrator
        let submission_res = state
            .orchestrator
            .submit_streaming_task(
                req_id.clone(),
                prompt_text.clone(),
                max_tokens,
                temperature,
                PriorityLevel::Normal,
                token_tx,
            )
            .await;

        match submission_res {
            Ok(_) => {
                // Return SSE stream adapter
                let stream_id = req_id.clone();
                let rx_stream = ReceiverStream::new(token_rx);
                let created_ts = chrono::Utc::now().timestamp() as u64;

                let sse_stream = rx_stream.map(move |token| {
                    let chunk = ChatCompletionChunk {
                        id: stream_id.clone(),
                        object: "chat.completion.chunk".to_string(),
                        created: created_ts,
                        model: DEFAULT_MODEL_ID.to_string(),
                        choices: vec![ChunkChoice {
                            index: 0,
                            delta: ChunkDelta {
                                role: None,
                                content: Some(token),
                            },
                            finish_reason: None,
                        }],
                    };
                    let json_payload = serde_json::to_string(&chunk).unwrap_or_default();
                    Ok::<Event, Infallible>(Event::default().data(json_payload))
                });

                // Attach terminal [DONE] event stream wrapper
                let done_stream = tokio_stream::once(Ok::<Event, Infallible>(
                    Event::default().data("[DONE]"),
                ));
                let combined_stream = sse_stream.chain(done_stream);

                Sse::new(combined_stream)
                    .keep_alive(KeepAlive::default())
                    .into_response()
            }
            Err(OrchestrationError::LoadSheddingActive(reason)) => {
                warn!("[GATEWAY] Request {} rejected by Load Shedding: {}", req_id, reason);
                make_error_response(
                    StatusCode::TOO_MANY_REQUESTS,
                    "Too Many Requests: Network cluster under gridlock. Request rejected to protect active stream.",
                    "load_shedding_rejection",
                    Some("429"),
                )
            }
            Err(OrchestrationError::QueueSaturated) => {
                warn!("[GATEWAY] Request {} rejected: FIFO queue capacity saturated", req_id);
                make_error_response(
                    StatusCode::TOO_MANY_REQUESTS,
                    "Capacity Limit Reached: Cluster task queue is full. Try again shortly.",
                    "queue_saturated",
                    Some("429"),
                )
            }
            Err(e) => {
                error!("[GATEWAY] Orchestrator submission failed: {}", e);
                make_error_response(
                    StatusCode::INTERNAL_SERVER_ERROR,
                    &format!("Inference orchestration error: {}", e),
                    "internal_orchestration_error",
                    None,
                )
            }
        }
    } else {
        // --- NON-STREAMING MODE ---
        let submission_res = state
            .orchestrator
            .submit_blocking_task(
                req_id.clone(),
                prompt_text.clone(),
                max_tokens,
                temperature,
                PriorityLevel::Normal,
            )
            .await;

        match submission_res {
            Ok(full_output) => {
                let now = chrono::Utc::now().timestamp() as u64;
                let prompt_tokens = prompt_text.split_whitespace().count();
                let completion_tokens = full_output.split_whitespace().count();

                let resp = ChatCompletionResponse {
                    id: req_id,
                    object: "chat.completion".to_string(),
                    created: now,
                    model: DEFAULT_MODEL_ID.to_string(),
                    choices: vec![ChatChoice {
                        index: 0,
                        message: ChatMessage {
                            role: "assistant".to_string(),
                            content: full_output,
                        },
                        finish_reason: Some("stop".to_string()),
                    }],
                    usage: UsageInfo {
                        prompt_tokens,
                        completion_tokens,
                        total_tokens: prompt_tokens + completion_tokens,
                    },
                };
                Json(resp).into_response()
            }
            Err(OrchestrationError::LoadSheddingActive(reason)) => {
                make_error_response(
                    StatusCode::TOO_MANY_REQUESTS,
                    &format!("Load Shedding Active: {}", reason),
                    "load_shedding_rejection",
                    Some("429"),
                )
            }
            Err(OrchestrationError::QueueSaturated) => {
                make_error_response(
                    StatusCode::TOO_MANY_REQUESTS,
                    "Task queue full. Please retry momentarily.",
                    "queue_saturated",
                    Some("429"),
                )
            }
            Err(e) => {
                make_error_response(
                    StatusCode::INTERNAL_SERVER_ERROR,
                    &format!("Inference failure: {}", e),
                    "inference_failure",
                    None,
                )
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Helpers & Utilities
// ---------------------------------------------------------------------------

/// Converts OpenAI chat message objects into structured ChatML prompt string.
pub fn format_chatml_prompt(messages: &[ChatMessage]) -> String {
    let mut out = String::new();
    for msg in messages {
        out.push_str(&format!("<|im_start|>{}\n{}<|im_end|>\n", msg.role, msg.content.trim()));
    }
    out.push_str("<|im_start|>assistant\n");
    out
}

/// Generates standardized JSON error response with headers.
fn make_error_response(
    status: StatusCode,
    msg: &str,
    err_type: &str,
    code: Option<&str>,
) -> Response {
    let mut headers = HeaderMap::new();
    if status == StatusCode::TOO_MANY_REQUESTS {
        headers.insert(header::RETRY_AFTER, HeaderValue::from_static("5"));
    }

    let payload = ApiErrorEnvelope {
        error: ApiErrorDetail {
            message: msg.to_string(),
            error_type: err_type.to_string(),
            code: code.map(String::from),
        },
    };

    (status, headers, Json(payload)).into_response()
}
