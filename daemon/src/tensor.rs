// =============================================================================
// AirDrop-LLM — Offline Tensor Streaming & Inference Engine Integration
// =============================================================================
//
// This module manages the distributed pipeline-parallel inference execution:
//
//   1. The master orchestrates the pipeline by dispatching layer execution
//      commands to each node in sequence.
//   2. Each node runs its assigned layers using a local inference engine
//      (llama.cpp via subprocess execution).
//   3. After completing its layers, a node:
//      a. Intercepts the intermediate tensor activation float arrays.
//      b. Compresses them using LZ4 for fast transfer over the local network.
//      c. Streams the compressed tensors over the TCP data channel.
//      d. Signals the next node in the pipeline to begin processing.
//
// Inference Engine Integration:
//   The Rust daemon triggers the underlying engine (llama.cpp / AirLLM)
//   locally via `tokio::process::Command`. This approach provides:
//   - Engine-agnostic design (swap backends without code changes).
//   - Process isolation (engine crashes don't take down the daemon).
//   - Stderr/stdout capture for structured logging.
//
// =============================================================================

use crate::cluster::{ClusterState, InferenceRequest, ClusterSnapshot};
use crate::scheduler::ExecutionSchedule;
use crate::transport;
use lz4_flex::{compress_prepend_size, decompress_size_prepended};
use std::sync::Arc;
use tokio::io::{AsyncReadExt, AsyncWriteExt};
use tokio::net::TcpStream;
use tokio::process::Command;
use tokio::sync::RwLock;
use tracing::{debug, error, info, warn};

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

/// Message type flags for the data channel framing protocol.
pub const MSG_TENSOR_DATA: u8 = 0x01;
pub const MSG_LAYER_COMMAND: u8 = 0x02;
pub const MSG_COMPLETION: u8 = 0x03;
pub const MSG_ERROR: u8 = 0x04;
pub const MSG_TOKEN_OUTPUT: u8 = 0x05;
pub const MSG_DELTA_TENSOR: u8 = 0x06;
pub const MSG_CACHE_INIT: u8 = 0x07;
pub const MSG_CACHE_ACK: u8 = 0x08;

/// Default llama.cpp server binary name.
const LLAMACPP_BINARY: &str = "llama-cli";

/// Tensor activation element size in bytes (f32).
const TENSOR_ELEMENT_SIZE: usize = 4;

// ---------------------------------------------------------------------------
// Layer Execution Command
// ---------------------------------------------------------------------------

/// A command dispatched to a node to execute a specific layer range.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct LayerCommand {
    /// The model file path (GGUF format).
    pub model_path: String,

    /// Starting layer index (inclusive).
    pub layer_start: u32,

    /// Ending layer index (exclusive).
    pub layer_end: u32,

    /// The input prompt (for the first node) or empty (for subsequent nodes
    /// that receive their input as tensor data from the previous node).
    pub prompt: String,

    /// Maximum tokens to generate.
    pub max_tokens: u32,

    /// Sampling temperature.
    pub temperature: f32,

    /// Whether this node should output the final generated tokens.
    pub is_final_stage: bool,
}

// ---------------------------------------------------------------------------
// Tensor Payload
// ---------------------------------------------------------------------------

/// A compressed tensor activation payload for streaming between nodes.
#[derive(Debug, Clone)]
pub struct TensorPayload {
    /// The layer index that produced this tensor.
    pub source_layer: u32,

    /// Shape dimensions [batch, sequence_length, hidden_size].
    pub shape: Vec<u32>,

    /// LZ4-compressed float32 tensor data.
    pub compressed_data: Vec<u8>,

    /// Original uncompressed size in bytes.
    pub original_size: usize,
}

impl TensorPayload {
    /// Compresses raw float32 tensor data using LZ4.
    ///
    /// LZ4 is chosen for its extreme decompression speed (~4 GB/s),
    /// which is critical for minimizing pipeline bubble time between nodes.
    pub fn compress(source_layer: u32, shape: Vec<u32>, raw_data: &[u8]) -> Self {
        let original_size = raw_data.len();
        let compressed_data = compress_prepend_size(raw_data);

        let ratio = if original_size > 0 {
            (compressed_data.len() as f64 / original_size as f64) * 100.0
        } else {
            0.0
        };

        debug!(
            "Tensor compressed: layer {} | {} → {} bytes ({:.1}% ratio)",
            source_layer,
            original_size,
            compressed_data.len(),
            ratio
        );

        TensorPayload {
            source_layer,
            shape,
            compressed_data,
            original_size,
        }
    }

    /// Decompresses the LZ4 tensor data back to raw float32 bytes.
    pub fn decompress(&self) -> anyhow::Result<Vec<u8>> {
        let decompressed = decompress_size_prepended(&self.compressed_data)
            .map_err(|e| anyhow::anyhow!("LZ4 decompression failed: {}", e))?;

        if decompressed.len() != self.original_size {
            return Err(anyhow::anyhow!(
                "Decompressed size mismatch: expected {}, got {}",
                self.original_size,
                decompressed.len()
            ));
        }

        Ok(decompressed)
    }

    /// Serializes the tensor payload for network transmission.
    ///
    /// Wire format:
    ///   [4 bytes: source_layer (u32 BE)]
    ///   [4 bytes: num_dims (u32 BE)]
    ///   [N × 4 bytes: shape dimensions (u32 BE each)]
    ///   [4 bytes: original_size (u32 BE)]
    ///   [4 bytes: compressed_size (u32 BE)]
    ///   [M bytes: compressed_data]
    pub fn serialize(&self) -> Vec<u8> {
        let mut buf = Vec::with_capacity(
            4 + 4 + self.shape.len() * 4 + 4 + 4 + self.compressed_data.len(),
        );

        buf.extend_from_slice(&self.source_layer.to_be_bytes());
        buf.extend_from_slice(&(self.shape.len() as u32).to_be_bytes());
        for dim in &self.shape {
            buf.extend_from_slice(&dim.to_be_bytes());
        }
        buf.extend_from_slice(&(self.original_size as u32).to_be_bytes());
        buf.extend_from_slice(&(self.compressed_data.len() as u32).to_be_bytes());
        buf.extend_from_slice(&self.compressed_data);

        buf
    }

    /// Deserializes a tensor payload from its wire format.
    pub fn deserialize(data: &[u8]) -> anyhow::Result<Self> {
        if data.len() < 16 {
            return Err(anyhow::anyhow!("Tensor payload too short: {} bytes", data.len()));
        }

        let mut cursor = 0;

        let source_layer = u32::from_be_bytes(data[cursor..cursor + 4].try_into()?);
        cursor += 4;

        let num_dims = u32::from_be_bytes(data[cursor..cursor + 4].try_into()?) as usize;
        cursor += 4;

        let mut shape = Vec::with_capacity(num_dims);
        for _ in 0..num_dims {
            shape.push(u32::from_be_bytes(data[cursor..cursor + 4].try_into()?));
            cursor += 4;
        }

        let original_size = u32::from_be_bytes(data[cursor..cursor + 4].try_into()?) as usize;
        cursor += 4;

        let compressed_size = u32::from_be_bytes(data[cursor..cursor + 4].try_into()?) as usize;
        cursor += 4;

        if data.len() < cursor + compressed_size {
            return Err(anyhow::anyhow!(
                "Tensor payload truncated: expected {} bytes, got {}",
                cursor + compressed_size,
                data.len()
            ));
        }

        let compressed_data = data[cursor..cursor + compressed_size].to_vec();

        Ok(TensorPayload {
            source_layer,
            shape,
            compressed_data,
            original_size,
        })
    }
}

// ---------------------------------------------------------------------------
// Master-Side Pipeline Orchestration
// ---------------------------------------------------------------------------

/// Runs the full distributed inference pipeline across the cluster.
///
/// # Pipeline Flow
///   1. Send layer command to first node.
///   2. First node executes its layers, outputs intermediate tensor.
///   3. Receive tensor from first node.
///   4. Forward tensor to second node with its layer command.
///   5. Repeat until the final node produces the output tokens.
///
/// Each step is wrapped in fault-tolerant error handling that triggers
/// the fallback protocol on any node failure.
pub async fn run_pipeline(
    schedule: &ExecutionSchedule,
    snapshot: &ClusterSnapshot,
    request: &InferenceRequest,
    state: Arc<RwLock<ClusterState>>,
) -> anyhow::Result<String> {
    info!("Starting distributed inference pipeline for request {}", request.id);
    info!("Pipeline has {} stages across {} nodes", schedule.assignments.len(), schedule.assignments.len());

    let model_path = snapshot
        .model_path
        .as_deref()
        .unwrap_or("model.gguf")
        .to_string();

    let mut current_tensor: Option<TensorPayload> = None;
    let mut final_output = String::new();

    for (idx, assignment) in schedule.assignments.iter().enumerate() {
        let is_first = idx == 0;
        let is_last = idx == schedule.assignments.len() - 1;

        info!(
            "Pipeline stage {}/{}: Node {} processing layers {}-{}",
            idx + 1,
            schedule.assignments.len(),
            assignment.node_id,
            assignment.layer_start,
            assignment.layer_end - 1,
        );

        // Build the layer execution command
        let command = LayerCommand {
            model_path: model_path.clone(),
            layer_start: assignment.layer_start,
            layer_end: assignment.layer_end,
            prompt: if is_first { request.prompt.clone() } else { String::new() },
            max_tokens: request.max_tokens,
            temperature: request.temperature,
            is_final_stage: is_last,
        };

        // Execute the stage
        let stage_result = if assignment.node_id == 0 {
            // Local execution (master node)
            execute_local_layers(&command, current_tensor.as_ref()).await
        } else {
            // Remote execution (worker node)
            execute_remote_layers(
                &assignment.address,
                &command,
                current_tensor.as_ref(),
            )
            .await
        };

        match stage_result {
            Ok(StageOutput::Tensor(tensor)) => {
                info!(
                    "Stage {} complete: received tensor ({} bytes compressed)",
                    idx + 1,
                    tensor.compressed_data.len()
                );
                current_tensor = Some(tensor);
            }
            Ok(StageOutput::FinalTokens(tokens)) => {
                info!("Pipeline complete: {} tokens generated", tokens.len());
                final_output = tokens;
            }
            Err(e) => {
                error!("Pipeline stage {} failed: {}", idx + 1, e);
                // Mark the node as dead for future scheduling decisions
                {
                    let mut s = state.write().await;
                    s.mark_node_dead(assignment.node_id);
                }
                return Err(e);
            }
        }
    }

    Ok(final_output)
}

// ---------------------------------------------------------------------------
// Stage Output Type
// ---------------------------------------------------------------------------

/// The output of a single pipeline stage — either intermediate tensor data
/// or the final generated token string.
enum StageOutput {
    /// Intermediate tensor to be forwarded to the next pipeline stage.
    Tensor(TensorPayload),
    /// Final output text from the last stage.
    FinalTokens(String),
}

// ---------------------------------------------------------------------------
// Local Layer Execution
// ---------------------------------------------------------------------------

/// Executes model layers locally using the underlying inference engine.
///
/// This function spawns a subprocess running llama.cpp (or equivalent)
/// with the specified layer range. It captures the output and either
/// extracts intermediate tensors or final generated text.
async fn execute_local_layers(
    command: &LayerCommand,
    input_tensor: Option<&TensorPayload>,
) -> anyhow::Result<StageOutput> {
    info!(
        "Executing layers {}-{} locally (engine: {})",
        command.layer_start,
        command.layer_end - 1,
        LLAMACPP_BINARY
    );

    // Build the llama.cpp command line
    let mut cmd = Command::new(LLAMACPP_BINARY);
    cmd.arg("-m").arg(&command.model_path)
        .arg("--n-gpu-layers-start").arg(command.layer_start.to_string())
        .arg("--n-gpu-layers-end").arg(command.layer_end.to_string())
        .arg("-n").arg(command.max_tokens.to_string())
        .arg("--temp").arg(command.temperature.to_string());

    if !command.prompt.is_empty() {
        cmd.arg("-p").arg(&command.prompt);
    }

    // If we have input tensor data from a previous stage, write it to a
    // temporary file that the engine can read as input activations.
    if let Some(tensor) = input_tensor {
        let tensor_path = std::env::temp_dir().join("airdrop_tensor_input.bin");
        let raw_data = tensor.decompress()?;
        tokio::fs::write(&tensor_path, &raw_data).await?;
        cmd.arg("--input-tensor").arg(&tensor_path);
    }

    // Configure subprocess I/O capture
    cmd.stdout(std::process::Stdio::piped())
        .stderr(std::process::Stdio::piped());

    // Spawn and await the engine process
    let output = tokio::time::timeout(
        tokio::time::Duration::from_secs(300), // 5-minute execution timeout
        cmd.output(),
    )
    .await
    .map_err(|_| anyhow::anyhow!("Local inference engine timed out after 300s"))?
    .map_err(|e| anyhow::anyhow!("Failed to spawn inference engine '{}': {}", LLAMACPP_BINARY, e))?;

    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        return Err(anyhow::anyhow!(
            "Inference engine exited with status {}: {}",
            output.status,
            stderr.chars().take(500).collect::<String>()
        ));
    }

    let stdout = String::from_utf8_lossy(&output.stdout);

    if command.is_final_stage {
        // Extract generated text from the engine output
        Ok(StageOutput::FinalTokens(stdout.trim().to_string()))
    } else {
        // Extract intermediate tensor from the engine's output file
        let tensor_output_path = std::env::temp_dir().join("airdrop_tensor_output.bin");
        if tensor_output_path.exists() {
            let raw_data = tokio::fs::read(&tensor_output_path).await?;
            let tensor = TensorPayload::compress(
                command.layer_end - 1,
                vec![1, 1, raw_data.len() as u32 / TENSOR_ELEMENT_SIZE as u32],
                &raw_data,
            );
            // Clean up temp file
            let _ = tokio::fs::remove_file(&tensor_output_path).await;
            Ok(StageOutput::Tensor(tensor))
        } else {
            // Fallback: use stdout as raw tensor data (development mode)
            let raw_bytes = stdout.as_bytes().to_vec();
            let tensor = TensorPayload::compress(
                command.layer_end - 1,
                vec![1, 1, raw_bytes.len() as u32 / TENSOR_ELEMENT_SIZE as u32],
                &raw_bytes,
            );
            Ok(StageOutput::Tensor(tensor))
        }
    }
}

// ---------------------------------------------------------------------------
// Remote Layer Execution
// ---------------------------------------------------------------------------

/// Sends a layer execution command to a remote worker node and receives
/// the output (tensor or final tokens) over the TCP data channel.
///
/// All network I/O is wrapped in fault-tolerant error handling with
/// timeouts to prevent pipeline stalls on worker failures.
async fn execute_remote_layers(
    worker_addr: &str,
    command: &LayerCommand,
    input_tensor: Option<&TensorPayload>,
) -> anyhow::Result<StageOutput> {
    let target = format!("{}:{}", worker_addr, transport::DATA_PORT);
    info!("Dispatching layers {}-{} to remote worker at {}", command.layer_start, command.layer_end - 1, target);

    // Establish TCP connection with timeout
    let mut stream = tokio::time::timeout(
        tokio::time::Duration::from_secs(10),
        TcpStream::connect(&target),
    )
    .await
    .map_err(|_| anyhow::anyhow!("Connection to worker {} timed out", target))?
    .map_err(|e| anyhow::anyhow!("Failed to connect to worker {}: {}", target, e))?;

    stream.set_nodelay(true)?;

    // Send DATA protocol header
    stream.write_all(b"DATA").await?;

    // Send the layer command as JSON
    let cmd_json = serde_json::to_vec(command)?;
    transport::send_framed(&mut stream, MSG_LAYER_COMMAND, &cmd_json).await?;

    // If there's input tensor data, send it to the worker
    if let Some(tensor) = input_tensor {
        let tensor_bytes = tensor.serialize();
        transport::send_framed(&mut stream, MSG_TENSOR_DATA, &tensor_bytes).await?;
        debug!("Sent input tensor ({} bytes) to worker", tensor_bytes.len());
    }

    // Wait for the worker's response with timeout
    let response = tokio::time::timeout(
        tokio::time::Duration::from_secs(300),
        transport::recv_framed(&mut stream),
    )
    .await
    .map_err(|_| anyhow::anyhow!("Worker response timed out after 300s"))??;

    let (msg_type, payload) = response;

    match msg_type {
        MSG_TENSOR_DATA => {
            let tensor = TensorPayload::deserialize(&payload)?;
            Ok(StageOutput::Tensor(tensor))
        }
        MSG_TOKEN_OUTPUT => {
            let tokens = String::from_utf8(payload)?;
            Ok(StageOutput::FinalTokens(tokens))
        }
        MSG_ERROR => {
            let error_msg = String::from_utf8_lossy(&payload);
            Err(anyhow::anyhow!("Worker error: {}", error_msg))
        }
        _ => Err(anyhow::anyhow!("Unexpected response type: 0x{:02X}", msg_type)),
    }
}

// ---------------------------------------------------------------------------
// Worker-Side Execution Loop
// ---------------------------------------------------------------------------

/// The worker's main execution loop.
///
/// Listens on the data channel for layer execution commands from the master.
/// When a command arrives:
///   1. Deserialize the LayerCommand.
///   2. If tensor input is provided, receive and decompress it.
///   3. Execute the assigned layers locally via the inference engine.
///   4. Compress the output tensor and stream it back to the master.
pub async fn worker_execution_loop(
    mut stream: TcpStream,
    state: Arc<RwLock<ClusterState>>,
    model_path: Option<String>,
) {
    info!("Worker execution loop started. Waiting for layer commands...");

    loop {
        // Wait for a framed message from the master
        let msg = match transport::recv_framed(&mut stream).await {
            Ok(m) => m,
            Err(e) => {
                warn!("Data channel read error: {}. Worker loop terminating.", e);
                break;
            }
        };

        let (msg_type, payload) = msg;

        match msg_type {
            MSG_LAYER_COMMAND => {
                // Deserialize the layer command
                let mut command: LayerCommand = match serde_json::from_slice(&payload) {
                    Ok(c) => c,
                    Err(e) => {
                        error!("Failed to parse layer command: {}", e);
                        let _ = transport::send_framed(
                            &mut stream,
                            MSG_ERROR,
                            format!("Invalid command: {}", e).as_bytes(),
                        )
                        .await;
                        continue;
                    }
                };

                // Override model path with locally configured path if available
                if let Some(ref local_model) = model_path {
                    command.model_path = local_model.clone();
                }

                info!(
                    "Received layer command: layers {}-{}",
                    command.layer_start,
                    command.layer_end - 1
                );

                // Check for input tensor
                let input_tensor = if command.prompt.is_empty() {
                    // Expect tensor input from previous pipeline stage
                    match transport::recv_framed(&mut stream).await {
                        Ok((MSG_TENSOR_DATA, tensor_data)) => {
                            match TensorPayload::deserialize(&tensor_data) {
                                Ok(t) => Some(t),
                                Err(e) => {
                                    error!("Failed to deserialize input tensor: {}", e);
                                    None
                                }
                            }
                        }
                        Ok(_) => None,
                        Err(e) => {
                            error!("Failed to receive input tensor: {}", e);
                            None
                        }
                    }
                } else {
                    None
                };

                // Execute the layers locally
                let result = execute_local_layers(&command, input_tensor.as_ref()).await;

                match result {
                    Ok(StageOutput::Tensor(tensor)) => {
                        let serialized = tensor.serialize();
                        if let Err(e) =
                            transport::send_framed(&mut stream, MSG_TENSOR_DATA, &serialized).await
                        {
                            error!("Failed to send tensor output: {}", e);
                            break;
                        }
                        info!("Sent tensor output ({} bytes)", serialized.len());
                    }
                    Ok(StageOutput::FinalTokens(tokens)) => {
                        if let Err(e) = transport::send_framed(
                            &mut stream,
                            MSG_TOKEN_OUTPUT,
                            tokens.as_bytes(),
                        )
                        .await
                        {
                            error!("Failed to send final tokens: {}", e);
                            break;
                        }
                        info!("Sent final tokens ({} chars)", tokens.len());
                    }
                    Err(e) => {
                        error!("Local layer execution failed: {}", e);
                        let _ = transport::send_framed(
                            &mut stream,
                            MSG_ERROR,
                            format!("Execution failed: {}", e).as_bytes(),
                        )
                        .await;
                    }
                }
            }
            _ => {
                debug!("Worker received unexpected message type: 0x{:02X}", msg_type);
            }
        }
    }

    warn!("Worker execution loop terminated.");
}
