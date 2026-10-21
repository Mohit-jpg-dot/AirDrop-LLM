// =============================================================================
// AirDrop-LLM — Engine Subsystem Module
// =============================================================================

pub mod arena;
pub mod guard;
pub mod speculative;
pub mod streamer;
pub mod supervisor;

pub use arena::{ArenaError, ArenaId, DoubleBufferedArena, STATE_COMPUTE, STATE_FREE, STATE_READY, STATE_STREAMING};
pub use guard::{
    FirewallViolation, HardwareWatchdogGuard, NetworkInputSanitizer, PowerSource, ThermalHysteresisState,
    HARD_TEMP_CEILING_CELSIUS, HYSTERESIS_LOWER_RECOVERY_CELSIUS, HYSTERESIS_UPPER_SHED_CELSIUS,
};
pub use speculative::{
    DraftGenerator, DraftToken, SpeculativeDraftBatch, SpeculativeError, SpeculativePipelineManager,
    SpeculativeVerificationReport, SPECULATIVE_GAMMA,
};
pub use streamer::{OffHeapWeightBuffer, PrimaryWeightReceiver, WorkerWeightStreamer};
pub use supervisor::AsymmetricPipelineSupervisor;


