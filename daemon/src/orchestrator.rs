// =============================================================================
// AirDrop-LLM — Multi-Tenant Capacity Interaction & FIFO Batch Orchestrator
// =============================================================================
//
// Thread-safe First-In, First-Out (FIFO) Task Orchestrator utilizing bounded
// memory channels (`tokio::sync::mpsc`).
//
// Solves GPU memory thrashing and network pipeline deadlock by locking
// distributed tensor operations to a single active user session at a time.
// Secondary requests are buffered in a strict FIFO bounded queue or rejected
// with HTTP 429 when capacity is saturated or the network supervisor engages
// Load Shedding.
//
// =============================================================================

use crate::cluster::{ClusterState, InferenceRequest};
use crate::fallback;
use crate::scheduler;
use crate::supervisor::NetworkSupervisor;
use crate::tensor;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::Arc;
use std::time::{Duration, Instant};
use tokio::sync::{mpsc, oneshot, Mutex, RwLock};
use tracing::{debug, error, info, warn};

// ---------------------------------------------------------------------------
// Constants & Limits
// ---------------------------------------------------------------------------

/// Maximum pending requests permitted in the bounded memory queue.
pub const MAX_FIFO_QUEUE_CAPACITY: usize = 32;

/// Default timeout waiting in the queue before expiration.
pub const QUEUE_WAIT_TIMEOUT: Duration = Duration::from_secs(45);

// ---------------------------------------------------------------------------
// Priority & Task Models
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PriorityLevel {
    High,
    Normal,
    Low,
}

/// Output transmission channel for the task.
pub enum TaskResponseChannel {
    /// Non-streaming: Return complete accumulated string output.
    Blocking(oneshot::Sender<Result<String, String>>),
    /// Streaming: Sequentially pipe token strings as they emerge from the pipeline.
    Streaming {
        token_tx: mpsc::Sender<String>,
        done_tx: oneshot::Sender<Result<(), String>>,
    },
}

/// Central task envelope wrapping an incoming user request.
pub struct InferenceTask {
    pub id: String,
    pub prompt: String,
    pub max_tokens: u32,
    pub temperature: f32,
    pub priority: PriorityLevel,
    pub enqueued_at: Instant,
    pub response_channel: TaskResponseChannel,
}

// ---------------------------------------------------------------------------
// Error Classification
// ---------------------------------------------------------------------------

#[derive(Debug)]
pub enum OrchestrationError {
    QueueSaturated,
    LoadSheddingActive(String),
    QueueTimeout,
    ExecutionFailed(String),
    ChannelClosed,
}

impl std::fmt::Display for OrchestrationError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::QueueSaturated => write!(f, "FIFO queue capacity saturated"),
            Self::LoadSheddingActive(r) => write!(f, "Load shedding active: {}", r),
            Self::QueueTimeout => write!(f, "Request timed out waiting in the FIFO queue"),
            Self::ExecutionFailed(e) => write!(f, "Inference execution failed: {}", e),
            Self::ChannelClosed => write!(f, "Client channel disconnected"),
        }
    }
}

impl std::error::Error for OrchestrationError {}

// ---------------------------------------------------------------------------
// FIFO Task Orchestrator Engine
// ---------------------------------------------------------------------------

pub struct InferenceOrchestrator {
    /// Inbound task channel transmitter.
    task_tx: mpsc::Sender<InferenceTask>,

    /// Count of currently queued tasks awaiting execution.
    queue_depth: Arc<AtomicUsize>,

    /// Atomic flag indicating whether the single GPU inference slot is active.
    slot_busy: Arc<AtomicBool>,

    /// Shared cluster state reference.
    cluster_state: Arc<RwLock<ClusterState>>,

    /// Network performance supervisor observing telemetry.
    supervisor: Arc<NetworkSupervisor>,

    /// Local standalone fallback lock flag.
    degraded_mode: Arc<AtomicBool>,
}

impl InferenceOrchestrator {
    /// Constructs and starts the FIFO orchestrator worker loop.
    pub fn new(
        cluster_state: Arc<RwLock<ClusterState>>,
        supervisor: Arc<NetworkSupervisor>,
    ) -> Arc<Self> {
        let (task_tx, task_rx) = mpsc::channel::<InferenceTask>(MAX_FIFO_QUEUE_CAPACITY);
        let queue_depth = Arc::new(AtomicUsize::new(0));
        let slot_busy = Arc::new(AtomicBool::new(false));
        let degraded_mode = Arc::new(AtomicBool::new(false));

        let orchestrator = Arc::new(Self {
            task_tx,
            queue_depth: Arc::clone(&queue_depth),
            slot_busy: Arc::clone(&slot_busy),
            cluster_state: Arc::clone(&cluster_state),
            supervisor: Arc::clone(&supervisor),
            degraded_mode: Arc::clone(&degraded_mode),
        });

        // Launch background worker processing tasks sequentially
        let worker_orch = Arc::clone(&orchestrator);
        tokio::spawn(async move {
            worker_orch.run_task_consumer_loop(task_rx).await;
        });

        orchestrator
    }

    /// Enqueues a streaming inference task.
    pub async fn submit_streaming_task(
        &self,
        id: String,
        prompt: String,
        max_tokens: u32,
        temperature: f32,
        priority: PriorityLevel,
        token_tx: mpsc::Sender<String>,
    ) -> Result<(), OrchestrationError> {
        self.pre_flight_check().await?;

        let (done_tx, done_rx) = oneshot::channel();
        let task = InferenceTask {
            id,
            prompt,
            max_tokens,
            temperature,
            priority,
            enqueued_at: Instant::now(),
            response_channel: TaskResponseChannel::Streaming { token_tx, done_tx },
        };

        self.enqueue_task(task).await?;

        // Wait for the task to finish or encounter execution error
        match done_rx.await {
            Ok(Ok(())) => Ok(()),
            Ok(Err(err)) => Err(OrchestrationError::ExecutionFailed(err)),
            Err(_) => Err(OrchestrationError::ChannelClosed),
        }
    }

    /// Enqueues a non-streaming (blocking) inference task.
    pub async fn submit_blocking_task(
        &self,
        id: String,
        prompt: String,
        max_tokens: u32,
        temperature: f32,
        priority: PriorityLevel,
    ) -> Result<String, OrchestrationError> {
        self.pre_flight_check().await?;

        let (resp_tx, resp_rx) = oneshot::channel();
        let task = InferenceTask {
            id,
            prompt,
            max_tokens,
            temperature,
            priority,
            enqueued_at: Instant::now(),
            response_channel: TaskResponseChannel::Blocking(resp_tx),
        };

        self.enqueue_task(task).await?;

        match resp_rx.await {
            Ok(Ok(output)) => Ok(output),
            Ok(Err(err)) => Err(OrchestrationError::ExecutionFailed(err)),
            Err(_) => Err(OrchestrationError::ChannelClosed),
        }
    }

    /// Validates queue capacity and supervisor metrics before accepting task.
    async fn pre_flight_check(&self) -> Result<(), OrchestrationError> {
        // 1. Check network load shedding state
        if let Some(reason) = self.supervisor.should_shed_load().await {
            return Err(OrchestrationError::LoadSheddingActive(reason));
        }

        // 2. Check FIFO queue depth limit
        let current_depth = self.queue_depth.load(Ordering::Acquire);
        if current_depth >= MAX_FIFO_QUEUE_CAPACITY {
            return Err(OrchestrationError::QueueSaturated);
        }

        Ok(())
    }

    /// Dispatches task into bounded MPSC queue.
    async fn enqueue_task(&self, task: InferenceTask) -> Result<(), OrchestrationError> {
        self.queue_depth.fetch_add(1, Ordering::Release);

        match self.task_tx.send(task).await {
            Ok(_) => Ok(()),
            Err(_) => {
                self.queue_depth.fetch_sub(1, Ordering::Release);
                Err(OrchestrationError::QueueSaturated)
            }
        }
    }

    /// Sequential FIFO Consumer Loop: Locks execution to a single user session at a time.
    async fn run_task_consumer_loop(&self, mut task_rx: mpsc::Receiver<InferenceTask>) {
        info!("[ORCHESTRATOR] Single-session FIFO execution scheduler active.");

        while let Some(task) = task_rx.recv().await {
            self.queue_depth.fetch_sub(1, Ordering::Release);

            // Check if task timed out while waiting in queue
            if task.enqueued_at.elapsed() > QUEUE_WAIT_TIMEOUT {
                warn!("[ORCHESTRATOR] Task {} expired in queue, dropping", task.id);
                match task.response_channel {
                    TaskResponseChannel::Blocking(tx) => {
                        let _ = tx.send(Err("Request timed out in FIFO queue".to_string()));
                    }
                    TaskResponseChannel::Streaming { done_tx, .. } => {
                        let _ = done_tx.send(Err("Request timed out in FIFO queue".to_string()));
                    }
                }
                continue;
            }

            // Acquire exclusive GPU execution slot
            self.slot_busy.store(true, Ordering::Release);
            debug!("[ORCHESTRATOR] Locked execution slot for task {}", task.id);

            self.execute_task_pipeline(task).await;

            // Release slot
            self.slot_busy.store(false, Ordering::Release);
            debug!("[ORCHESTRATOR] Released execution slot.");
        }
    }

    /// Dispatches task to distributed cluster pipeline or degrades to local fallback.
    async fn execute_task_pipeline(&self, task: InferenceTask) {
        let req = InferenceRequest {
            id: task.id.clone(),
            prompt: task.prompt.clone(),
            max_tokens: task.max_tokens,
            temperature: task.temperature,
        };

        // Check if supervisor mandates local standalone execution to bypass network jitter
        let is_degraded = self.supervisor.is_degraded_fallback_active().await;
        self.degraded_mode.store(is_degraded, Ordering::Release);

        if is_degraded {
            info!(
                "[ORCHESTRATOR] Hot-swapping task {} to local standalone engine (Network degraded)",
                task.id
            );
            let fallback_result = fallback::run_local_fallback(&req, Arc::clone(&self.cluster_state)).await;
            self.deliver_result(task.response_channel, Ok(fallback_result)).await;
            return;
        }

        // Standard cluster execution flow
        let cluster_snapshot = {
            let s = self.cluster_state.read().await;
            s.snapshot()
        };

        let schedule = scheduler::compute_schedule(&cluster_snapshot);

        // Run distributed pipeline
        let pipeline_res = tensor::run_pipeline(
            &schedule,
            &cluster_snapshot,
            &req,
            Arc::clone(&self.cluster_state),
        )
        .await;

        match pipeline_res {
            Ok(output) => {
                self.deliver_result(task.response_channel, Ok(output)).await;
            }
            Err(e) => {
                warn!(
                    "[ORCHESTRATOR] Distributed pipeline failed ({}). Executing hot-swap fallback...",
                    e
                );
                // Trigger supervisor degraded state
                self.supervisor.trigger_degraded_state(format!("Pipeline error: {}", e)).await;

                // Execute immediate local standalone fallback
                let fallback_output = fallback::run_local_fallback(&req, Arc::clone(&self.cluster_state)).await;
                self.deliver_result(task.response_channel, Ok(fallback_output)).await;
            }
        }
    }

    /// Emits final output to the caller (blocking string or streaming tokens).
    async fn deliver_result(&self, channel: TaskResponseChannel, result: Result<String, String>) {
        match channel {
            TaskResponseChannel::Blocking(tx) => {
                let _ = tx.send(result);
            }
            TaskResponseChannel::Streaming { token_tx, done_tx } => {
                match result {
                    Ok(full_text) => {
                        // In live execution, tokens stream out sequentially.
                        // Here we feed tokens with minimal latency to emulate continuous stream
                        for word in full_text.split_inclusive(' ') {
                            if token_tx.send(word.to_string()).await.is_err() {
                                break;
                            }
                            tokio::time::sleep(Duration::from_millis(15)).await;
                        }
                        let _ = done_tx.send(Ok(()));
                    }
                    Err(err_msg) => {
                        let _ = done_tx.send(Err(err_msg));
                    }
                }
            }
        }
    }

    pub fn pending_tasks_count(&self) -> usize {
        self.queue_depth.load(Ordering::Acquire)
    }

    pub fn is_slot_busy(&self) -> bool {
        self.slot_busy.load(Ordering::Acquire)
    }

    pub async fn is_degraded_mode(&self) -> bool {
        self.degraded_mode.load(Ordering::Acquire)
    }
}
