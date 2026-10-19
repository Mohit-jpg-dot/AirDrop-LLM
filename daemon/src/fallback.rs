// =============================================================================
// AirDrop-LLM — Automated Fault-Tolerant Local Fallback Protocol
// =============================================================================
//
// Provides structural stability against sudden node disconnections during
// active inference sessions. When a worker drops off the network (battery
// death, hotspot disconnect, hardware failure), the master daemon must:
//
//   1. Instantly intercept the broken pipeline execution chain.
//   2. NOT crash the application or leave the UI in an error state.
//   3. Hot-swap to a local standalone inference model that fits entirely
//      within the host machine's available memory.
//   4. Continue serving the user's request with degraded but functional
//      inference using a compact model (e.g., Qwen 2.5 Coder 1.5B or
//      LFM 2.5 Small).
//   5. Downscale context window boundaries to fit memory constraints.
//   6. Notify the UI of the fallback activation for transparency.
//
// All network socket operations in the tensor streaming pipeline are
// wrapped in strict fault-tolerant catch blocks (Result + timeout) to
// ensure that failures are caught and handled, never propagated as panics.
//
// =============================================================================

use crate::cluster::{ClusterState, InferenceRequest};
use std::sync::Arc;
use tokio::process::Command;
use tokio::sync::RwLock;
use tracing::{error, info, warn};

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

/// Compact fallback models suitable for single-node execution on resource-
/// constrained devices (16GB RAM).
///
/// These are ordered by preference — the daemon tries each in sequence
/// until it finds one that exists on the local filesystem.
const FALLBACK_MODEL_CANDIDATES: &[FallbackModel] = &[
    FallbackModel {
        name: "Qwen 2.5 Coder 1.5B (Q4_K_M)",
        filename: "qwen2.5-coder-1.5b-q4_k_m.gguf",
        context_window: 4096,
        memory_requirement_mb: 1200,
    },
    FallbackModel {
        name: "LFM 2.5 Small (Q4_K_M)",
        filename: "lfm-2.5-small-q4_k_m.gguf",
        context_window: 4096,
        memory_requirement_mb: 1500,
    },
    FallbackModel {
        name: "Phi-3.5 Mini (Q4_K_M)",
        filename: "phi-3.5-mini-q4_k_m.gguf",
        context_window: 4096,
        memory_requirement_mb: 2400,
    },
    FallbackModel {
        name: "TinyLlama 1.1B (Q8_0)",
        filename: "tinyllama-1.1b-q8_0.gguf",
        context_window: 2048,
        memory_requirement_mb: 1200,
    },
];

/// Default search paths for fallback model files.
const MODEL_SEARCH_PATHS: &[&str] = &[
    "models/",
    "~/.airdrop-llm/models/",
    "~/.cache/airdrop-llm/models/",
    "/usr/local/share/airdrop-llm/models/",
];

/// Maximum context window reduction factor for emergency memory conservation.
const EMERGENCY_CONTEXT_REDUCTION: f32 = 0.5;

/// Inference engine binary for fallback execution.
const FALLBACK_ENGINE: &str = "llama-cli";

// ---------------------------------------------------------------------------
// Fallback Model Descriptor
// ---------------------------------------------------------------------------

/// Describes a compact model suitable for local-only fallback inference.
#[derive(Debug, Clone)]
struct FallbackModel {
    /// Human-readable model name for logging and UI display.
    name: &'static str,

    /// Expected GGUF filename on disk.
    filename: &'static str,

    /// Maximum context window in tokens.
    context_window: u32,

    /// Approximate memory requirement in MB.
    memory_requirement_mb: u64,
}

// ---------------------------------------------------------------------------
// Fallback Entrypoint
// ---------------------------------------------------------------------------

/// Activates the local fallback inference protocol.
///
/// This function is called when the distributed pipeline fails due to
/// a worker disconnection. It:
///   1. Searches for a compact fallback model on the local filesystem.
///   2. Downscales the context window if memory is tight.
///   3. Runs inference locally using the fallback model.
///   4. Returns the generated text (may be lower quality than the
///      distributed model, but keeps the application alive).
///
/// # Arguments
/// * `request` - The original inference request that was being processed.
/// * `state`   - Shared cluster state for memory checks and UI notifications.
///
/// # Returns
/// The generated text string, or a graceful error message if even
/// local fallback fails.
pub async fn run_local_fallback(
    request: &InferenceRequest,
    state: Arc<RwLock<ClusterState>>,
) -> String {
    warn!("════════════════════════════════════════════════════════");
    warn!("  FALLBACK PROTOCOL ACTIVATED");
    warn!("  Distributed pipeline failed — switching to local model");
    warn!("════════════════════════════════════════════════════════");

    // Update cluster state to reflect fallback mode
    {
        let mut s = state.write().await;
        s.status_message = "⚠ Fallback: running local inference (worker disconnected)".to_string();
    }

    // Step 1: Find a suitable fallback model
    let (model_path, model_info) = match find_fallback_model().await {
        Some(result) => result,
        None => {
            error!(
                "No fallback model found! Searched paths: {:?}",
                MODEL_SEARCH_PATHS
            );
            let mut s = state.write().await;
            s.status_message = "✗ Fallback failed: no local model found".to_string();
            return generate_graceful_error_response(request);
        }
    };

    info!("Fallback model selected: {} at {}", model_info.name, model_path);

    // Step 2: Check available memory and adjust context window
    let available_ram_mb = get_available_memory_mb();
    let context_window = compute_safe_context_window(
        model_info.context_window,
        model_info.memory_requirement_mb,
        available_ram_mb,
    );

    info!(
        "Memory check: {} MB available, model needs {} MB, context window: {} tokens",
        available_ram_mb, model_info.memory_requirement_mb, context_window
    );

    // Step 3: Truncate the prompt to fit within the reduced context window
    let truncated_prompt = truncate_prompt(&request.prompt, context_window);

    // Step 4: Execute local inference with the fallback model
    let result = execute_fallback_inference(
        &model_path,
        &truncated_prompt,
        request.max_tokens.min(context_window / 2), // Cap output tokens
        request.temperature,
    )
    .await;

    match result {
        Ok(output) => {
            info!("Fallback inference complete ({} chars generated)", output.len());
            let mut s = state.write().await;
            s.status_message = format!(
                "✓ Fallback complete (used {} — quality may be reduced)",
                model_info.name
            );
            output
        }
        Err(e) => {
            error!("Fallback inference failed: {}", e);
            let mut s = state.write().await;
            s.status_message = format!("✗ Fallback inference failed: {}", e);
            generate_graceful_error_response(request)
        }
    }
}

// ---------------------------------------------------------------------------
// Model Discovery
// ---------------------------------------------------------------------------

/// Searches the filesystem for a suitable fallback model.
///
/// Iterates through candidate models and search paths, returning the
/// first match found. Prioritizes smaller models that are more likely
/// to fit in constrained memory.
async fn find_fallback_model() -> Option<(String, &'static FallbackModel)> {
    for model in FALLBACK_MODEL_CANDIDATES {
        for search_dir in MODEL_SEARCH_PATHS {
            // Expand home directory tilde
            let expanded = if search_dir.starts_with("~") {
                if let Some(home) = dirs_home() {
                    search_dir.replacen("~", &home, 1)
                } else {
                    continue;
                }
            } else {
                search_dir.to_string()
            };

            let full_path = format!("{}{}", expanded, model.filename);

            if tokio::fs::metadata(&full_path).await.is_ok() {
                info!("Found fallback model: {} at {}", model.name, full_path);
                return Some((full_path, model));
            }
        }
    }

    None
}

/// Returns the user's home directory path as a string.
fn dirs_home() -> Option<String> {
    std::env::var("HOME")
        .or_else(|_| std::env::var("USERPROFILE"))
        .ok()
}

// ---------------------------------------------------------------------------
// Memory Management
// ---------------------------------------------------------------------------

/// Returns the currently available system memory in megabytes.
fn get_available_memory_mb() -> u64 {
    let mut sys = sysinfo::System::new_all();
    sys.refresh_all();
    sys.available_memory() / (1024 * 1024)
}

/// Computes a safe context window size based on available memory.
///
/// If available memory is tight relative to the model's requirements,
/// the context window is reduced to prevent OOM conditions.
fn compute_safe_context_window(
    base_context: u32,
    model_memory_mb: u64,
    available_memory_mb: u64,
) -> u32 {
    // If we have at least 2x the model's memory requirement, use full context
    if available_memory_mb >= model_memory_mb * 2 {
        return base_context;
    }

    // If we have 1.5x, reduce context by 25%
    if available_memory_mb >= (model_memory_mb as f64 * 1.5) as u64 {
        return (base_context as f32 * 0.75) as u32;
    }

    // Tight memory: apply emergency reduction
    let reduced = (base_context as f32 * EMERGENCY_CONTEXT_REDUCTION) as u32;
    warn!(
        "Memory pressure detected! Context window reduced: {} → {} tokens",
        base_context, reduced
    );

    reduced.max(256) // Absolute minimum context
}

// ---------------------------------------------------------------------------
// Prompt Truncation
// ---------------------------------------------------------------------------

/// Truncates the input prompt to fit within the specified context window.
///
/// Uses a simple character-based heuristic (4 chars ≈ 1 token) to
/// approximate token count without requiring a tokenizer dependency.
fn truncate_prompt(prompt: &str, max_context_tokens: u32) -> String {
    // Reserve half the context for generation, use the other half for prompt
    let max_prompt_tokens = max_context_tokens / 2;
    let max_chars = (max_prompt_tokens * 4) as usize; // ~4 chars per token

    if prompt.len() <= max_chars {
        return prompt.to_string();
    }

    warn!(
        "Prompt truncated from {} to {} chars for fallback context window",
        prompt.len(),
        max_chars
    );

    // Keep the end of the prompt (most recent context is usually most relevant)
    let truncated = &prompt[prompt.len() - max_chars..];

    // Find a clean word boundary
    if let Some(space_idx) = truncated.find(' ') {
        format!("...{}", &truncated[space_idx + 1..])
    } else {
        format!("...{}", truncated)
    }
}

// ---------------------------------------------------------------------------
// Fallback Inference Execution
// ---------------------------------------------------------------------------

/// Executes inference using the compact fallback model via subprocess.
///
/// The subprocess command is wrapped in strict error handling and timeouts
/// to ensure it never blocks the daemon indefinitely.
async fn execute_fallback_inference(
    model_path: &str,
    prompt: &str,
    max_tokens: u32,
    temperature: f32,
) -> anyhow::Result<String> {
    info!(
        "Launching fallback inference: model={}, max_tokens={}, temp={}",
        model_path, max_tokens, temperature
    );

    let mut cmd = Command::new(FALLBACK_ENGINE);
    cmd.arg("-m").arg(model_path)
        .arg("-p").arg(prompt)
        .arg("-n").arg(max_tokens.to_string())
        .arg("--temp").arg(temperature.to_string())
        .arg("--ctx-size").arg("2048") // Conservative context size
        .arg("--threads").arg(num_cpus().to_string()) // Use all available cores
        .stdout(std::process::Stdio::piped())
        .stderr(std::process::Stdio::piped());

    // Execute with a generous timeout (fallback should still be reasonably fast)
    let output = tokio::time::timeout(
        tokio::time::Duration::from_secs(120),
        cmd.output(),
    )
    .await
    .map_err(|_| anyhow::anyhow!("Fallback inference timed out after 120s"))?
    .map_err(|e| anyhow::anyhow!("Failed to spawn fallback engine: {}", e))?;

    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        return Err(anyhow::anyhow!(
            "Fallback engine exited with status {}: {}",
            output.status,
            stderr.chars().take(500).collect::<String>()
        ));
    }

    let stdout = String::from_utf8_lossy(&output.stdout).trim().to_string();
    Ok(stdout)
}

/// Returns the number of available CPU cores for local inference threads.
fn num_cpus() -> usize {
    std::thread::available_parallelism()
        .map(|p| p.get())
        .unwrap_or(4)
}

// ---------------------------------------------------------------------------
// Graceful Error Response
// ---------------------------------------------------------------------------

/// Generates a user-friendly error message when all fallback mechanisms fail.
///
/// This ensures the UI always receives a response, even in catastrophic
/// failure scenarios. The message explains the situation and suggests
/// recovery steps.
fn generate_graceful_error_response(request: &InferenceRequest) -> String {
    format!(
        "[AirDrop-LLM Fallback Notice]\n\n\
         Your inference request could not be completed because:\n\
         • A worker node disconnected during processing.\n\
         • No local fallback model was found on this machine.\n\n\
         To enable automatic fallback, place a compact GGUF model in:\n\
         ~/.airdrop-llm/models/\n\n\
         Recommended fallback models:\n\
         • qwen2.5-coder-1.5b-q4_k_m.gguf (~1.2 GB)\n\
         • tinyllama-1.1b-q8_0.gguf (~1.1 GB)\n\n\
         Your original prompt has been preserved:\n\
         \"{}...\"\n",
        request.prompt.chars().take(100).collect::<String>()
    )
}
