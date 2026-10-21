// =============================================================================
// AirDrop-LLM — Cluster State & Hardware Profile Management
// =============================================================================
//
// Central state management for the distributed inference cluster. This module
// provides:
//
//   - `HardwareProfile`: Detected hardware capabilities of a single node
//     (CPU, RAM, GPU vendor/VRAM, compute backend).
//   - `ClusterNode`: A registered node in the cluster with its network
//     address, hardware profile, benchmark results, and health status.
//   - `ClusterState`: The shared truth store for the entire cluster topology,
//     accessed through Arc<RwLock<ClusterState>> across all subsystems.
//   - `detect_hardware_profile()`: Introspects the local machine's hardware
//     using the `sysinfo` crate.
//
// =============================================================================

use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::fmt;
use sysinfo::System;
use tracing::info;

// ---------------------------------------------------------------------------
// Node Role
// ---------------------------------------------------------------------------

/// The operational role of a node within the cluster.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum NodeRole {
    /// Cluster head: coordinates inference, hosts UI, manages scheduling.
    Master,
    /// Peer worker: executes assigned model layers and streams tensors.
    Worker,
}

// ---------------------------------------------------------------------------
// Compute Backend
// ---------------------------------------------------------------------------

/// The primary compute acceleration backend available on a node.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum ComputeBackend {
    /// Apple Metal Performance Shaders (Apple Silicon unified memory).
    AppleMetal,
    /// NVIDIA CUDA (discrete GPU with dedicated VRAM).
    NvidiaCuda,
    /// AMD ROCm (discrete GPU).
    AmdRocm,
    /// CPU-only fallback (no GPU acceleration).
    CpuOnly,
}

impl fmt::Display for ComputeBackend {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            ComputeBackend::AppleMetal => write!(f, "Apple Metal"),
            ComputeBackend::NvidiaCuda => write!(f, "NVIDIA CUDA"),
            ComputeBackend::AmdRocm => write!(f, "AMD ROCm"),
            ComputeBackend::CpuOnly => write!(f, "CPU Only"),
        }
    }
}

// ---------------------------------------------------------------------------
// Hardware Profile
// ---------------------------------------------------------------------------

/// Complete hardware capability descriptor for a single node.
/// Populated via local introspection and transmitted during peer registration.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HardwareProfile {
    /// Human-readable CPU model string (e.g., "Apple M5" or "AMD Ryzen 9 7950X").
    pub cpu_model: String,

    /// Number of logical CPU cores available.
    pub cpu_cores: usize,

    /// Total system RAM in megabytes.
    pub ram_total_mb: u64,

    /// Available (free) system RAM in megabytes at detection time.
    pub ram_available_mb: u64,

    /// GPU device name if detected (e.g., "NVIDIA GeForce RTX 4070").
    pub gpu_name: Option<String>,

    /// Dedicated GPU VRAM in megabytes (0 for unified memory architectures).
    pub gpu_vram_mb: u64,

    /// The primary compute acceleration backend.
    pub compute_backend: ComputeBackend,

    /// Operating system identifier.
    pub os_name: String,

    /// Machine hostname for display purposes.
    pub hostname: String,
}

impl fmt::Display for HardwareProfile {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "{} | {} cores | {} MB RAM | GPU: {} ({} MB VRAM) | Backend: {} | OS: {}",
            self.cpu_model,
            self.cpu_cores,
            self.ram_total_mb,
            self.gpu_name.as_deref().unwrap_or("None"),
            self.gpu_vram_mb,
            self.compute_backend,
            self.os_name,
        )
    }
}

// ---------------------------------------------------------------------------
// Network Benchmark Results
// ---------------------------------------------------------------------------

/// Results from the point-to-point network performance benchmark.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BenchmarkResult {
    /// Measured throughput in megabits per second (Mbps).
    pub throughput_mbps: f64,

    /// Average round-trip time in milliseconds across ping rounds.
    pub latency_ms: f64,

    /// Whether the connection meets minimum quality thresholds.
    /// Threshold: throughput >= 150 Mbps AND latency <= 20ms.
    pub passes_gate: bool,

    /// Timestamp of the benchmark execution.
    pub timestamp: chrono::DateTime<chrono::Utc>,
}

// ---------------------------------------------------------------------------
// Cluster Node
// ---------------------------------------------------------------------------

/// Represents a single registered node in the cluster.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ClusterNode {
    /// Unique node identifier within the cluster (0 = master).
    pub node_id: u32,

    /// Network socket address (IP:port) of the node.
    pub address: String,

    /// Detected hardware capabilities.
    pub hardware: HardwareProfile,

    /// Network benchmark results (populated after benchmark pass).
    pub benchmark: Option<BenchmarkResult>,

    /// Operational role in the cluster.
    pub role: NodeRole,

    /// Whether the node is currently reachable and healthy.
    pub is_alive: bool,

    /// Last successful heartbeat timestamp.
    pub last_heartbeat: chrono::DateTime<chrono::Utc>,
}

// ---------------------------------------------------------------------------
// Inference Request
// ---------------------------------------------------------------------------

/// A pending inference request submitted through the UI or API.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct InferenceRequest {
    /// Unique request identifier.
    pub id: String,

    /// The input prompt text.
    pub prompt: String,

    /// Maximum number of tokens to generate.
    pub max_tokens: u32,

    /// Sampling temperature (0.0 = deterministic, 1.0+ = creative).
    pub temperature: f32,

    /// Optional base64-encoded image payloads for multimodal inference.
    #[serde(default)]
    pub images: Option<Vec<String>>,
}

// ---------------------------------------------------------------------------
// Cluster Snapshot (read-only view for scheduling)
// ---------------------------------------------------------------------------

/// An immutable snapshot of the cluster state for use in scheduling decisions.
/// Avoids holding the RwLock during potentially long computations.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ClusterSnapshot {
    pub nodes: Vec<ClusterNode>,
    pub model_path: Option<String>,
}

// ---------------------------------------------------------------------------
// Cluster State
// ---------------------------------------------------------------------------

/// The central mutable state store for the distributed inference cluster.
///
/// All subsystems access this through `Arc<RwLock<ClusterState>>` to ensure
/// safe concurrent reads with exclusive writes. The master node owns the
/// authoritative copy; workers maintain a local subset.
pub struct ClusterState {
    /// Registry of all nodes in the cluster, keyed by node_id.
    nodes: HashMap<u32, ClusterNode>,

    /// Monotonically increasing counter for assigning node IDs to new workers.
    next_node_id: u32,

    /// Queue of pending inference requests from the UI/API.
    inference_queue: Vec<InferenceRequest>,

    /// Completed inference results, keyed by request ID.
    completed_results: HashMap<String, String>,

    /// Path to the primary model file (GGUF format).
    model_path: Option<String>,

    /// Whether the cluster is currently executing a distributed inference pass.
    pub is_inferring: bool,

    /// Current system status message for UI display.
    pub status_message: String,
}

impl ClusterState {
    /// Creates a new empty cluster state.
    pub fn new(model_path: Option<String>) -> Self {
        Self {
            nodes: HashMap::new(),
            next_node_id: 1, // 0 is reserved for the master
            inference_queue: Vec::new(),
            completed_results: HashMap::new(),
            model_path,
            is_inferring: false,
            status_message: "Initializing...".to_string(),
        }
    }

    /// Registers the local node (self) into the cluster map.
    pub fn set_local_node(&mut self, node_id: u32, hardware: HardwareProfile, role: NodeRole) {
        let node = ClusterNode {
            node_id,
            address: "127.0.0.1".to_string(),
            hardware,
            benchmark: None,
            role,
            is_alive: true,
            last_heartbeat: chrono::Utc::now(),
        };
        self.nodes.insert(node_id, node);
        self.status_message = format!("Node {} registered (local)", node_id);
    }

    /// Registers a remote worker node discovered via UDP broadcast.
    /// Returns the assigned node ID.
    pub fn register_worker(&mut self, address: String, hardware: HardwareProfile) -> u32 {
        let node_id = self.next_node_id;
        self.next_node_id += 1;

        let node = ClusterNode {
            node_id,
            address: address.clone(),
            hardware,
            benchmark: None,
            role: NodeRole::Worker,
            is_alive: true,
            last_heartbeat: chrono::Utc::now(),
        };

        info!("Registered worker node {} at {}", node_id, address);
        self.nodes.insert(node_id, node);
        self.status_message = format!("Worker {} joined from {}", node_id, address);

        node_id
    }

    /// Updates benchmark results for a specific node.
    pub fn set_benchmark(&mut self, node_id: u32, result: BenchmarkResult) {
        if let Some(node) = self.nodes.get_mut(&node_id) {
            node.benchmark = Some(result);
        }
    }

    /// Marks a node as disconnected (dead).
    pub fn mark_node_dead(&mut self, node_id: u32) {
        if let Some(node) = self.nodes.get_mut(&node_id) {
            node.is_alive = false;
            self.status_message = format!("Node {} disconnected", node_id);
        }
    }

    /// Updates the heartbeat timestamp for a node.
    pub fn heartbeat(&mut self, node_id: u32) {
        if let Some(node) = self.nodes.get_mut(&node_id) {
            node.last_heartbeat = chrono::Utc::now();
            node.is_alive = true;
        }
    }

    /// Returns a list of all currently alive worker nodes.
    pub fn alive_workers(&self) -> Vec<&ClusterNode> {
        self.nodes
            .values()
            .filter(|n| n.role == NodeRole::Worker && n.is_alive)
            .collect()
    }

    /// Returns the total number of registered nodes (including master).
    pub fn node_count(&self) -> usize {
        self.nodes.len()
    }

    /// Enqueues a new inference request.
    pub fn enqueue_inference_request(&mut self, request: InferenceRequest) {
        self.inference_queue.push(request);
    }

    /// Dequeues the next pending inference request, if any.
    pub fn dequeue_inference_request(&self) -> Option<InferenceRequest> {
        // Note: actual dequeue requires &mut self; this is a peek for the
        // read-locked orchestration loop. The real dequeue happens via
        // a separate method called under write lock.
        self.inference_queue.first().cloned()
    }

    /// Removes and returns the next pending inference request.
    pub fn pop_inference_request(&mut self) -> Option<InferenceRequest> {
        if self.inference_queue.is_empty() {
            None
        } else {
            Some(self.inference_queue.remove(0))
        }
    }

    /// Records a completed inference result.
    pub fn complete_inference_request(&mut self, request_id: &str, output: String) {
        self.completed_results.insert(request_id.to_string(), output);
        self.is_inferring = false;
        self.status_message = format!("Inference complete: {}", request_id);
    }

    /// Retrieves a completed inference result by request ID.
    pub fn get_result(&self, request_id: &str) -> Option<&String> {
        self.completed_results.get(request_id)
    }

    /// Creates an immutable snapshot of the cluster state for scheduling.
    pub fn snapshot(&self) -> ClusterSnapshot {
        ClusterSnapshot {
            nodes: self.nodes.values().cloned().collect(),
            model_path: self.model_path.clone(),
        }
    }

    /// Returns all nodes for API/UI serialization.
    pub fn all_nodes(&self) -> Vec<&ClusterNode> {
        self.nodes.values().collect()
    }
}

// ---------------------------------------------------------------------------
// Hardware Detection
// ---------------------------------------------------------------------------

/// Introspects the local machine's hardware capabilities using the `sysinfo`
/// crate and platform-specific heuristics.
///
/// GPU detection strategy:
///   - macOS: Assumes Apple Silicon with Metal backend if `aarch64`.
///   - Linux/Windows: Checks for NVIDIA GPU via `nvidia-smi` availability.
///   - Falls back to CPU-only if no GPU is detected.
pub fn detect_hardware_profile() -> HardwareProfile {
    let mut sys = System::new_all();
    sys.refresh_all();

    // CPU information
    let cpu_model = sys
        .cpus()
        .first()
        .map(|c| c.brand().to_string())
        .unwrap_or_else(|| "Unknown CPU".to_string());
    let cpu_cores = sys.cpus().len();

    // Memory information (sysinfo reports in bytes, convert to MB)
    let ram_total_mb = sys.total_memory() / (1024 * 1024);
    let ram_available_mb = sys.available_memory() / (1024 * 1024);

    // OS information
    let os_name = format!(
        "{} {}",
        System::name().unwrap_or_else(|| "Unknown".to_string()),
        System::os_version().unwrap_or_else(|| "".to_string()),
    );

    let hostname = System::host_name().unwrap_or_else(|| "unknown".to_string());

    // GPU detection via platform heuristics
    let (gpu_name, gpu_vram_mb, compute_backend) = detect_gpu();

    HardwareProfile {
        cpu_model,
        cpu_cores,
        ram_total_mb,
        ram_available_mb,
        gpu_name,
        gpu_vram_mb,
        compute_backend,
        os_name,
        hostname,
    }
}

/// Detects GPU hardware using platform-specific strategies.
///
/// Returns: (gpu_name, vram_mb, compute_backend)
fn detect_gpu() -> (Option<String>, u64, ComputeBackend) {
    // Strategy 1: Check for Apple Silicon (macOS aarch64)
    if cfg!(target_os = "macos") && cfg!(target_arch = "aarch64") {
        // Apple Silicon uses unified memory — the "VRAM" is the system RAM.
        // We report the total system memory as the available unified pool.
        let mut sys = System::new_all();
        sys.refresh_all();
        let unified_mb = sys.total_memory() / (1024 * 1024);

        return (
            Some("Apple Silicon (Unified Memory)".to_string()),
            unified_mb,
            ComputeBackend::AppleMetal,
        );
    }

    // Strategy 2: Check for NVIDIA GPU via nvidia-smi
    if let Ok(output) = std::process::Command::new("nvidia-smi")
        .args(["--query-gpu=name,memory.total", "--format=csv,noheader,nounits"])
        .output()
    {
        if output.status.success() {
            let stdout = String::from_utf8_lossy(&output.stdout);
            let line = stdout.trim();
            if let Some((name, vram_str)) = line.split_once(", ") {
                let vram_mb = vram_str.trim().parse::<u64>().unwrap_or(0);
                return (
                    Some(name.trim().to_string()),
                    vram_mb,
                    ComputeBackend::NvidiaCuda,
                );
            }
        }
    }

    // Strategy 3: Check for AMD GPU via rocm-smi
    if let Ok(output) = std::process::Command::new("rocm-smi")
        .args(["--showproductname", "--showmeminfo", "vram"])
        .output()
    {
        if output.status.success() {
            return (
                Some("AMD ROCm GPU".to_string()),
                0, // ROCm VRAM parsing requires more complex output parsing
                ComputeBackend::AmdRocm,
            );
        }
    }

    // Fallback: CPU-only
    (None, 0, ComputeBackend::CpuOnly)
}
