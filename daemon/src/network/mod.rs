// =============================================================================
// AirDrop-LLM — Network Subsystem Module
// =============================================================================

pub mod heartbeat;

pub use heartbeat::{KernelTcpWatchdog, NetworkConnectivityPanic};
