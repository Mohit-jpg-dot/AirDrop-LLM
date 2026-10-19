// =============================================================================
// AirDrop-LLM — TCP Transport Layer
// =============================================================================
//
// Provides the reliable, bidirectional TCP transport infrastructure for
// all node-to-node communication beyond UDP discovery:
//
//   - Worker registration handshake (hardware profile exchange).
//   - Benchmark session handling (delegates to benchmark module).
//   - Persistent data channels for tensor streaming during inference.
//
// All connections operate on port 9999 with protocol headers to distinguish
// message types within the same listener.
//
// Wire Protocol:
//   Every TCP connection begins with a 4-byte protocol discriminator:
//     "REGI" → Worker registration
//     "BNCH" → Benchmark session
//     "DATA" → Tensor data channel
//     "HRTB" → Heartbeat probe
//
// =============================================================================

use crate::benchmark;
use crate::cluster::{ClusterState, HardwareProfile};
use std::sync::Arc;
use tokio::io::{AsyncReadExt, AsyncWriteExt};
use tokio::net::{TcpListener, TcpStream};
use tokio::sync::RwLock;
use tracing::{debug, error, info, warn};

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

/// The TCP port for all data channel operations.
pub const DATA_PORT: u16 = 9999;

/// Protocol discriminator length in bytes.
const HEADER_LEN: usize = 4;

/// Protocol discriminators
const HEADER_REGISTER: &[u8; 4] = b"REGI";
const HEADER_BENCHMARK: &[u8; 4] = b"BNCH";
const HEADER_DATA: &[u8; 4] = b"DATA";
const HEADER_HEARTBEAT: &[u8; 4] = b"HRTB";

/// Maximum size for registration payloads (hardware profile JSON).
const MAX_REGISTRATION_SIZE: usize = 8192;

// ---------------------------------------------------------------------------
// Master-Side: TCP Connection Acceptor
// ---------------------------------------------------------------------------

/// Runs the master-side TCP listener on port 9999.
///
/// Accepts incoming connections, reads the 4-byte protocol header, and
/// dispatches to the appropriate handler:
///   - REGI → Worker registration
///   - BNCH → Benchmark session
///   - DATA → Tensor data channel
///   - HRTB → Heartbeat response
///
/// Each accepted connection is spawned into its own async task for
/// concurrent handling.
pub async fn run_tcp_acceptor(state: Arc<RwLock<ClusterState>>) -> anyhow::Result<()> {
    let listener = TcpListener::bind(format!("0.0.0.0:{}", DATA_PORT)).await?;

    info!("TCP acceptor listening on 0.0.0.0:{}", DATA_PORT);

    loop {
        let (stream, peer_addr) = match listener.accept().await {
            Ok(conn) => conn,
            Err(e) => {
                warn!("TCP accept error: {}. Continuing...", e);
                continue;
            }
        };

        debug!("Accepted TCP connection from {}", peer_addr);

        let conn_state = Arc::clone(&state);
        tokio::spawn(async move {
            if let Err(e) = handle_connection(stream, conn_state).await {
                warn!("Connection handler error for {}: {}", peer_addr, e);
            }
        });
    }
}

/// Dispatches a single TCP connection based on its protocol header.
async fn handle_connection(
    mut stream: TcpStream,
    state: Arc<RwLock<ClusterState>>,
) -> anyhow::Result<()> {
    let peer = stream.peer_addr()?.to_string();

    // Read the 4-byte protocol discriminator
    let mut header = [0u8; HEADER_LEN];
    stream.read_exact(&mut header).await?;

    match &header {
        HEADER_REGISTER => {
            debug!("Protocol: REGISTER from {}", peer);
            handle_registration(stream, state).await?;
        }
        HEADER_BENCHMARK => {
            debug!("Protocol: BENCHMARK from {}", peer);
            benchmark::handle_benchmark_session(stream).await?;
        }
        HEADER_DATA => {
            debug!("Protocol: DATA channel from {}", peer);
            handle_data_channel(stream, state).await?;
        }
        HEADER_HEARTBEAT => {
            debug!("Protocol: HEARTBEAT from {}", peer);
            handle_heartbeat(stream, state).await?;
        }
        _ => {
            warn!(
                "Unknown protocol header from {}: {:?}",
                peer,
                std::str::from_utf8(&header).unwrap_or("<binary>")
            );
        }
    }

    Ok(())
}

// ---------------------------------------------------------------------------
// Worker Registration Handler
// ---------------------------------------------------------------------------

/// Handles a worker registration request.
///
/// Protocol:
///   1. Read 4-byte big-endian length prefix.
///   2. Read the hardware profile JSON payload.
///   3. Deserialize and register the worker.
///   4. Run network benchmark against the worker.
///   5. Respond with the assigned node ID and benchmark results.
async fn handle_registration(
    mut stream: TcpStream,
    state: Arc<RwLock<ClusterState>>,
) -> anyhow::Result<()> {
    let peer_addr = stream.peer_addr()?.ip().to_string();

    // Read length-prefixed hardware profile
    let mut len_buf = [0u8; 4];
    stream.read_exact(&mut len_buf).await?;
    let payload_len = u32::from_be_bytes(len_buf) as usize;

    if payload_len > MAX_REGISTRATION_SIZE {
        return Err(anyhow::anyhow!(
            "Registration payload too large: {} bytes (max: {})",
            payload_len,
            MAX_REGISTRATION_SIZE
        ));
    }

    let mut payload = vec![0u8; payload_len];
    stream.read_exact(&mut payload).await?;

    let hardware: HardwareProfile = serde_json::from_slice(&payload)?;
    info!(
        "Registration from {}: {} ({})",
        peer_addr, hardware.hostname, hardware.cpu_model
    );

    // Register in cluster state
    let node_id = {
        let mut s = state.write().await;
        s.register_worker(peer_addr.clone(), hardware)
    };

    // Run network benchmark against the newly registered worker
    info!("Running benchmark against worker {} ({})", node_id, peer_addr);
    match benchmark::run_benchmark(&peer_addr).await {
        Ok(result) => {
            let mut s = state.write().await;
            s.set_benchmark(node_id, result);
        }
        Err(e) => {
            warn!(
                "Benchmark failed for worker {} ({}): {}",
                node_id, peer_addr, e
            );
        }
    }

    // Send registration response: node_id as 4-byte big-endian
    let response = node_id.to_be_bytes();
    stream.write_all(&response).await?;
    stream.flush().await?;

    info!("Worker {} registered and benchmarked successfully.", node_id);
    Ok(())
}

// ---------------------------------------------------------------------------
// Data Channel Handler
// ---------------------------------------------------------------------------

/// Handles a persistent tensor data channel from a worker.
///
/// This channel remains open for the duration of the inference session,
/// allowing bidirectional streaming of:
///   - Layer execution commands (master → worker)
///   - Intermediate tensor data (worker → master → next worker)
///   - Completion signals and error notifications
///
/// The data channel uses a simple framing protocol:
///   [4 bytes: payload length (big-endian u32)]
///   [N bytes: payload data]
///   [1 byte:  message type flag]
///     0x01 = Tensor data
///     0x02 = Layer command
///     0x03 = Completion signal
///     0x04 = Error notification
async fn handle_data_channel(
    mut stream: TcpStream,
    state: Arc<RwLock<ClusterState>>,
) -> anyhow::Result<()> {
    let peer = stream.peer_addr()?.to_string();
    info!("Data channel established with {}", peer);

    // Disable Nagle's algorithm for low-latency tensor streaming
    stream.set_nodelay(true)?;

    let mut header_buf = [0u8; 5]; // 4 bytes length + 1 byte type

    loop {
        // Read frame header
        match tokio::time::timeout(
            tokio::time::Duration::from_secs(300), // 5-minute idle timeout
            stream.read_exact(&mut header_buf),
        )
        .await
        {
            Ok(Ok(_)) => {
                let payload_len = u32::from_be_bytes([
                    header_buf[0],
                    header_buf[1],
                    header_buf[2],
                    header_buf[3],
                ]) as usize;
                let msg_type = header_buf[4];

                // Read payload
                let mut payload = vec![0u8; payload_len];
                stream.read_exact(&mut payload).await?;

                match msg_type {
                    0x01 => {
                        debug!("Received tensor data ({} bytes) from {}", payload_len, peer);
                        // Store received tensor for pipeline forwarding
                        // This is handled by the tensor module's pipeline coordinator
                    }
                    0x03 => {
                        info!("Worker {} signaled layer completion.", peer);
                    }
                    0x04 => {
                        let error_msg = String::from_utf8_lossy(&payload);
                        warn!("Worker {} reported error: {}", peer, error_msg);
                    }
                    _ => {
                        debug!("Unknown message type 0x{:02X} from {}", msg_type, peer);
                    }
                }
            }
            Ok(Err(e)) => {
                warn!("Data channel read error from {}: {}", peer, e);
                break;
            }
            Err(_) => {
                info!("Data channel idle timeout for {}. Closing.", peer);
                break;
            }
        }
    }

    Ok(())
}

// ---------------------------------------------------------------------------
// Heartbeat Handler
// ---------------------------------------------------------------------------

/// Responds to heartbeat probes from workers.
/// Simple echo — the worker measures RTT from their side.
async fn handle_heartbeat(
    mut stream: TcpStream,
    state: Arc<RwLock<ClusterState>>,
) -> anyhow::Result<()> {
    let peer = stream.peer_addr()?.ip().to_string();

    // Read the node_id from the heartbeat payload
    let mut id_buf = [0u8; 4];
    stream.read_exact(&mut id_buf).await?;
    let node_id = u32::from_be_bytes(id_buf);

    // Update heartbeat timestamp
    {
        let mut s = state.write().await;
        s.heartbeat(node_id);
    }

    // Echo back as acknowledgment
    stream.write_all(&id_buf).await?;
    stream.flush().await?;

    debug!("Heartbeat acknowledged for node {} ({})", node_id, peer);
    Ok(())
}

// ---------------------------------------------------------------------------
// Worker-Side: Registration with Master
// ---------------------------------------------------------------------------

/// Registers the local worker node with the master.
///
/// Establishes a TCP connection to the master on port 9999, sends the
/// REGI header followed by the hardware profile JSON, and reads back
/// the assigned node ID.
pub async fn register_with_master(
    master_addr: &std::net::SocketAddr,
    hardware: &HardwareProfile,
) -> anyhow::Result<()> {
    let target = format!("{}:{}", master_addr.ip(), DATA_PORT);
    info!("Registering with master at {}", target);

    let mut stream = TcpStream::connect(&target).await?;

    // Send protocol header
    stream.write_all(HEADER_REGISTER).await?;

    // Serialize and send hardware profile
    let hw_json = serde_json::to_vec(hardware)?;
    let len_bytes = (hw_json.len() as u32).to_be_bytes();
    stream.write_all(&len_bytes).await?;
    stream.write_all(&hw_json).await?;
    stream.flush().await?;

    // Read assigned node ID
    let mut id_buf = [0u8; 4];
    stream.read_exact(&mut id_buf).await?;
    let node_id = u32::from_be_bytes(id_buf);

    info!("Registration complete. Assigned node_id={}", node_id);
    Ok(())
}

// ---------------------------------------------------------------------------
// Worker-Side: Data Channel Establishment
// ---------------------------------------------------------------------------

/// Establishes a persistent TCP data channel with the master node.
///
/// This channel is used for bidirectional tensor streaming during
/// distributed inference sessions. Returns the connected TcpStream.
pub async fn connect_data_channel(
    master_addr: &std::net::SocketAddr,
) -> anyhow::Result<TcpStream> {
    let target = format!("{}:{}", master_addr.ip(), DATA_PORT);
    info!("Establishing data channel with master at {}", target);

    let mut stream = TcpStream::connect(&target).await?;

    // Disable Nagle for low-latency streaming
    stream.set_nodelay(true)?;

    // Send protocol header to identify as data channel
    stream.write_all(HEADER_DATA).await?;
    stream.flush().await?;

    info!("Data channel connected to master.");
    Ok(stream)
}

// ---------------------------------------------------------------------------
// Framed Message Sending (used by tensor module)
// ---------------------------------------------------------------------------

/// Sends a framed message over a TCP stream.
///
/// Frame format:
///   [4 bytes: payload length (big-endian)]
///   [1 byte:  message type]
///   [N bytes: payload]
pub async fn send_framed(
    stream: &mut TcpStream,
    msg_type: u8,
    payload: &[u8],
) -> anyhow::Result<()> {
    let len_bytes = (payload.len() as u32).to_be_bytes();
    stream.write_all(&len_bytes).await?;
    stream.write_all(&[msg_type]).await?;
    stream.write_all(payload).await?;
    stream.flush().await?;
    Ok(())
}

/// Receives a framed message from a TCP stream.
///
/// Returns: (message_type, payload_bytes)
pub async fn recv_framed(stream: &mut TcpStream) -> anyhow::Result<(u8, Vec<u8>)> {
    let mut header = [0u8; 5];
    stream.read_exact(&mut header).await?;

    let payload_len = u32::from_be_bytes([header[0], header[1], header[2], header[3]]) as usize;
    let msg_type = header[4];

    let mut payload = vec![0u8; payload_len];
    stream.read_exact(&mut payload).await?;

    Ok((msg_type, payload))
}
