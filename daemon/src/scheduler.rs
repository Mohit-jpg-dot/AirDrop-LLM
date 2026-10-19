// =============================================================================
// AirDrop-LLM — Dynamic Asymmetric Scheduling & Matrix Allocation
// =============================================================================
//
// The scheduler is responsible for dividing model processing layers across
// heterogeneous cluster nodes based on their actual hardware capabilities
// and network benchmark results — NOT a naive equal split.
//
// Scheduling Strategy:
//   1. Read hardware profiles from all alive nodes in the cluster.
//   2. Compute a normalized "compute score" for each node based on:
//      - Available memory (RAM or VRAM) for layer weight storage.
//      - GPU compute throughput (Metal vs CUDA vs CPU-only).
//      - Network benchmark results (throughput & latency penalties).
//   3. Allocate model layers proportionally to each node's score.
//   4. Assign lighter loads to constrained nodes (e.g., MacBook Air M5
//      with 16GB unified memory) and heavier loads to powerful nodes
//      (e.g., Lenovo Legion with RTX 4070 + 12GB VRAM).
//
// This produces a pipeline-parallel execution plan where each node
// processes its assigned layer range sequentially, streaming intermediate
// tensor activations to the next node in the pipeline.
//
// =============================================================================

use crate::cluster::{ClusterNode, ClusterSnapshot, ComputeBackend, NodeRole};
use serde::{Deserialize, Serialize};
use tracing::{debug, info, warn};

// ---------------------------------------------------------------------------
// Layer Assignment
// ---------------------------------------------------------------------------

/// Represents the layer range assigned to a single node in the pipeline.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LayerAssignment {
    /// The node ID receiving this assignment.
    pub node_id: u32,

    /// The network address of the node.
    pub address: String,

    /// Starting layer index (inclusive).
    pub layer_start: u32,

    /// Ending layer index (exclusive).
    pub layer_end: u32,

    /// Number of layers assigned.
    pub layer_count: u32,

    /// The node's computed capability score (for debugging/UI display).
    pub compute_score: f64,

    /// Position in the pipeline ring (0 = first, N-1 = last).
    pub pipeline_position: u32,
}

// ---------------------------------------------------------------------------
// Execution Schedule
// ---------------------------------------------------------------------------

/// The complete pipeline-parallel execution plan for a distributed inference pass.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExecutionSchedule {
    /// Ordered list of layer assignments forming the pipeline.
    /// Index 0 processes first, streams to index 1, etc.
    pub assignments: Vec<LayerAssignment>,

    /// Total number of model layers being distributed.
    pub total_layers: u32,

    /// Whether any node failed the benchmark gate and received a penalty.
    pub has_bottleneck_nodes: bool,

    /// Human-readable summary of the scheduling decision.
    pub summary: String,
}

// ---------------------------------------------------------------------------
// Compute Score Calculation
// ---------------------------------------------------------------------------

/// Weight multipliers for different compute backends.
/// These reflect the relative inference throughput of each accelerator type.
const BACKEND_WEIGHT_METAL: f64 = 1.0;     // Apple Silicon baseline
const BACKEND_WEIGHT_CUDA: f64 = 1.8;      // NVIDIA CUDA (typically faster for LLMs)
const BACKEND_WEIGHT_ROCM: f64 = 1.3;      // AMD ROCm (competitive but less optimized)
const BACKEND_WEIGHT_CPU: f64 = 0.3;       // CPU-only (significantly slower)

/// Penalty factor applied to nodes that failed the network benchmark gate.
const BOTTLENECK_PENALTY: f64 = 0.5;

/// Minimum layer allocation per node (prevents trivial assignments).
const MIN_LAYERS_PER_NODE: u32 = 2;

/// Computes a normalized capability score for a single cluster node.
///
/// The score is a composite metric based on:
///   - Available compute memory (GB) weighted by backend type.
///   - Backend-specific throughput multiplier.
///   - Network quality penalty if benchmark failed.
///
/// # Returns
/// A floating-point score where higher = more capable.
fn compute_node_score(node: &ClusterNode) -> f64 {
    // Base memory score: use VRAM for discrete GPUs, RAM for unified memory
    let memory_gb = match node.hardware.compute_backend {
        ComputeBackend::AppleMetal => {
            // Apple Silicon unified memory: use total RAM as the compute pool.
            // Discount ~4GB for OS/system overhead.
            let usable_mb = node.hardware.ram_total_mb.saturating_sub(4096);
            usable_mb as f64 / 1024.0
        }
        ComputeBackend::NvidiaCuda | ComputeBackend::AmdRocm => {
            // Discrete GPU: use dedicated VRAM.
            node.hardware.gpu_vram_mb as f64 / 1024.0
        }
        ComputeBackend::CpuOnly => {
            // CPU-only: use available RAM with heavy discount.
            let usable_mb = node.hardware.ram_available_mb.saturating_sub(2048);
            usable_mb as f64 / 1024.0
        }
    };

    // Backend throughput multiplier
    let backend_multiplier = match node.hardware.compute_backend {
        ComputeBackend::AppleMetal => BACKEND_WEIGHT_METAL,
        ComputeBackend::NvidiaCuda => BACKEND_WEIGHT_CUDA,
        ComputeBackend::AmdRocm => BACKEND_WEIGHT_ROCM,
        ComputeBackend::CpuOnly => BACKEND_WEIGHT_CPU,
    };

    // Network quality penalty
    let network_factor = match &node.benchmark {
        Some(bench) if !bench.passes_gate => {
            debug!(
                "Node {} has bottleneck penalty (throughput: {:.1} Mbps, latency: {:.1} ms)",
                node.node_id, bench.throughput_mbps, bench.latency_ms
            );
            BOTTLENECK_PENALTY
        }
        _ => 1.0, // No penalty (benchmark passed or not yet run)
    };

    let score = memory_gb * backend_multiplier * network_factor;
    debug!(
        "Node {} score: {:.2} (mem: {:.1} GB × backend: {:.1} × network: {:.1})",
        node.node_id, score, memory_gb, backend_multiplier, network_factor
    );

    score.max(0.1) // Floor to prevent zero-score nodes
}

// ---------------------------------------------------------------------------
// Schedule Computation
// ---------------------------------------------------------------------------

/// Computes the pipeline-parallel execution schedule for the cluster.
///
/// # Algorithm
///   1. Filter to alive nodes only.
///   2. Compute capability scores for each node.
///   3. Normalize scores into proportional layer allocations.
///   4. Assign layer ranges in pipeline order (master first, workers follow).
///   5. Enforce minimum layer allocation per node.
///
/// # Arguments
/// * `snapshot` - Immutable cluster state snapshot.
///
/// # Returns
/// A complete `ExecutionSchedule` with ordered layer assignments.
pub fn compute_schedule(snapshot: &ClusterSnapshot) -> ExecutionSchedule {
    // Default total layers for common model architectures
    // This would be dynamically read from the model file in production.
    let total_layers: u32 = 32; // Llama-3 8B default

    // Filter to alive nodes and compute scores
    let mut scored_nodes: Vec<(&ClusterNode, f64)> = snapshot
        .nodes
        .iter()
        .filter(|n| n.is_alive)
        .map(|n| (n, compute_node_score(n)))
        .collect();

    // Sort: master first, then workers by score (descending)
    scored_nodes.sort_by(|a, b| {
        // Master always goes first in the pipeline
        match (a.0.role, b.0.role) {
            (NodeRole::Master, NodeRole::Worker) => std::cmp::Ordering::Less,
            (NodeRole::Worker, NodeRole::Master) => std::cmp::Ordering::Greater,
            _ => b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal),
        }
    });

    if scored_nodes.is_empty() {
        warn!("No alive nodes in cluster! Cannot compute schedule.");
        return ExecutionSchedule {
            assignments: vec![],
            total_layers,
            has_bottleneck_nodes: false,
            summary: "No nodes available.".to_string(),
        };
    }

    // Single-node fast path: assign all layers to the only node
    if scored_nodes.len() == 1 {
        let (node, score) = &scored_nodes[0];
        return ExecutionSchedule {
            assignments: vec![LayerAssignment {
                node_id: node.node_id,
                address: node.address.clone(),
                layer_start: 0,
                layer_end: total_layers,
                layer_count: total_layers,
                compute_score: *score,
                pipeline_position: 0,
            }],
            total_layers,
            has_bottleneck_nodes: false,
            summary: format!("Single-node execution: all {} layers on node {}", total_layers, node.node_id),
        };
    }

    // Compute total score for proportional allocation
    let total_score: f64 = scored_nodes.iter().map(|(_, s)| s).sum();
    let has_bottleneck = scored_nodes.iter().any(|(n, _)| {
        n.benchmark.as_ref().map_or(false, |b| !b.passes_gate)
    });

    // Allocate layers proportionally to scores
    let mut assignments: Vec<LayerAssignment> = Vec::with_capacity(scored_nodes.len());
    let mut layer_cursor: u32 = 0;

    for (position, (node, score)) in scored_nodes.iter().enumerate() {
        // Proportional allocation
        let proportion = score / total_score;
        let mut allocated = (proportion * total_layers as f64).round() as u32;

        // Enforce minimum
        allocated = allocated.max(MIN_LAYERS_PER_NODE);

        // Clamp to remaining layers
        let remaining = total_layers.saturating_sub(layer_cursor);
        if position == scored_nodes.len() - 1 {
            // Last node gets all remaining layers
            allocated = remaining;
        } else {
            allocated = allocated.min(remaining);
        }

        if allocated == 0 {
            continue;
        }

        let layer_start = layer_cursor;
        let layer_end = layer_cursor + allocated;

        assignments.push(LayerAssignment {
            node_id: node.node_id,
            address: node.address.clone(),
            layer_start,
            layer_end,
            layer_count: allocated,
            compute_score: *score,
            pipeline_position: position as u32,
        });

        layer_cursor = layer_end;
    }

    // Build human-readable summary
    let summary = build_schedule_summary(&assignments, total_layers, has_bottleneck);
    info!("{}", summary);

    ExecutionSchedule {
        assignments,
        total_layers,
        has_bottleneck_nodes: has_bottleneck,
        summary,
    }
}

// ---------------------------------------------------------------------------
// Schedule Summary
// ---------------------------------------------------------------------------

/// Builds a human-readable summary of the scheduling decision.
fn build_schedule_summary(
    assignments: &[LayerAssignment],
    total_layers: u32,
    has_bottleneck: bool,
) -> String {
    let mut lines = vec![
        format!("═══ Pipeline Schedule ({} layers across {} nodes) ═══", total_layers, assignments.len()),
    ];

    if has_bottleneck {
        lines.push("⚠  Bottleneck nodes detected — routing overrides applied.".to_string());
    }

    for a in assignments {
        lines.push(format!(
            "  Node {} [pos {}]: layers {}-{} ({} layers, score: {:.2}) @ {}",
            a.node_id,
            a.pipeline_position,
            a.layer_start,
            a.layer_end - 1,
            a.layer_count,
            a.compute_score,
            a.address,
        ));
    }

    lines.join("\n")
}

// ---------------------------------------------------------------------------
// Schedule Recalculation (Hot Path)
// ---------------------------------------------------------------------------

/// Recomputes the schedule after a node failure.
///
/// This is called by the fallback module when a worker disconnects
/// mid-inference. It excludes the dead node and redistributes its
/// layers to the remaining alive nodes.
pub fn recompute_schedule_without_node(
    snapshot: &ClusterSnapshot,
    dead_node_id: u32,
) -> ExecutionSchedule {
    info!(
        "Recomputing schedule without node {} (node failure)",
        dead_node_id
    );

    // Create a filtered snapshot excluding the dead node
    let filtered = ClusterSnapshot {
        nodes: snapshot
            .nodes
            .iter()
            .filter(|n| n.node_id != dead_node_id)
            .cloned()
            .collect(),
        model_path: snapshot.model_path.clone(),
    };

    compute_schedule(&filtered)
}
