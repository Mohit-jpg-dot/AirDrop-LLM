// =============================================================================
// AirDrop-LLM — UDP Multicast Peer Discovery
// =============================================================================
//
// Implements the zero-configuration peer discovery protocol using UDP
// multicast broadcasting over the local network segment.
//
// Protocol:
//   1. Worker nodes broadcast "AIRDROP_LLM_PING:{hardware_profile_json}"
//      to the multicast group on port 9998 every 2 seconds.
//   2. The master node listens on port 9998, captures incoming pings,
//      deserializes the hardware payload, and registers the worker.
//   3. The master responds with "AIRDROP_LLM_PONG:{assigned_node_id}"
//      back to the worker's source address.
//   4. The worker stops broadcasting upon receiving the PONG.
//
// This approach works over both smartphone hotspot LANs and Wi-Fi Direct
// networks without requiring mDNS, SSDP, or any external service.
//
// =============================================================================

use crate::cluster::{ClusterState, HardwareProfile};
use std::net::SocketAddr;
use std::sync::Arc;
use tokio::net::UdpSocket;
use tokio::sync::RwLock;
use tracing::{debug, error, info, warn};

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

/// The UDP port used for peer discovery broadcasts.
pub const DISCOVERY_PORT: u16 = 9998;

/// Protocol ping prefix — workers broadcast this to announce presence.
const PING_PREFIX: &str = "AIRDROP_LLM_PING:";

/// Protocol pong prefix — master responds with this to acknowledge registration.
const PONG_PREFIX: &str = "AIRDROP_LLM_PONG:";

/// Maximum UDP datagram size for discovery messages.
/// Hardware profiles are typically < 1KB; 4KB provides generous headroom.
const MAX_DATAGRAM_SIZE: usize = 4096;

/// Interval between worker broadcast attempts (milliseconds).
const BROADCAST_INTERVAL_MS: u64 = 2000;

/// Maximum number of broadcast attempts before giving up.
const MAX_BROADCAST_ATTEMPTS: u32 = 150; // 5 minutes at 2s intervals

// ---------------------------------------------------------------------------
// Master: Discovery Listener
// ---------------------------------------------------------------------------

/// Runs the master-side UDP discovery listener.
///
/// Binds to `0.0.0.0:9998` and listens for incoming `AIRDROP_LLM_PING`
/// datagrams from worker nodes. On receipt, it:
///   1. Deserializes the embedded hardware profile JSON.
///   2. Registers the worker in the cluster state.
///   3. Sends an `AIRDROP_LLM_PONG:{node_id}` response.
///
/// This function runs indefinitely until the task is cancelled.
pub async fn run_master_listener(state: Arc<RwLock<ClusterState>>) -> anyhow::Result<()> {
    // Bind to all interfaces on the discovery port.
    // `SO_REUSEADDR` is set implicitly by tokio's UdpSocket on most platforms.
    let socket = UdpSocket::bind(format!("0.0.0.0:{}", DISCOVERY_PORT)).await?;

    // Enable broadcast reception — required for receiving directed broadcasts
    // on some network configurations (hotspot LANs).
    socket.set_broadcast(true)?;

    info!(
        "Discovery listener bound to 0.0.0.0:{}. Waiting for worker pings...",
        DISCOVERY_PORT
    );

    let mut buf = vec![0u8; MAX_DATAGRAM_SIZE];

    loop {
        // Block until a datagram arrives
        let (bytes_read, src_addr) = match socket.recv_from(&mut buf).await {
            Ok(result) => result,
            Err(e) => {
                warn!("UDP recv error: {}. Retrying...", e);
                continue;
            }
        };

        // Parse the datagram payload as UTF-8
        let payload = match std::str::from_utf8(&buf[..bytes_read]) {
            Ok(s) => s,
            Err(_) => {
                debug!("Received non-UTF8 datagram from {}. Ignoring.", src_addr);
                continue;
            }
        };

        // Check if this is a valid discovery ping
        if !payload.starts_with(PING_PREFIX) {
            debug!("Received unknown datagram from {}: {:?}", src_addr, payload);
            continue;
        }

        // Extract the hardware profile JSON from the ping payload
        let hw_json = &payload[PING_PREFIX.len()..];
        let hardware: HardwareProfile = match serde_json::from_str(hw_json) {
            Ok(hw) => hw,
            Err(e) => {
                warn!(
                    "Failed to parse hardware profile from {}: {}",
                    src_addr, e
                );
                continue;
            }
        };

        info!(
            "Discovery ping from {} — {} ({})",
            src_addr, hardware.hostname, hardware.cpu_model
        );

        // Register the worker in the cluster state
        let node_id = {
            let mut s = state.write().await;

            // Deduplicate: check if this address is already registered
            let existing = s.all_nodes().iter().any(|n| {
                n.address == src_addr.ip().to_string() && n.is_alive
            });

            if existing {
                debug!("Worker at {} is already registered. Sending re-ack.", src_addr);
                // Still send a PONG to handle the case where the worker
                // didn't receive the previous acknowledgment
                0 // Will be overridden below
            } else {
                s.register_worker(src_addr.ip().to_string(), hardware)
            }
        };

        // Send acknowledgment back to the worker
        let pong_payload = format!("{}{}", PONG_PREFIX, node_id);
        match socket.send_to(pong_payload.as_bytes(), src_addr).await {
            Ok(_) => {
                info!("Sent registration acknowledgment to {} (node_id={})", src_addr, node_id);
            }
            Err(e) => {
                error!("Failed to send PONG to {}: {}", src_addr, e);
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Worker: Discovery Broadcaster
// ---------------------------------------------------------------------------

/// Runs the worker-side UDP discovery broadcast loop.
///
/// Broadcasts `AIRDROP_LLM_PING:{hardware_profile_json}` on the local
/// network every 2 seconds until the master responds with a PONG.
///
/// Returns the master node's socket address upon successful discovery.
pub async fn run_worker_broadcast() -> anyhow::Result<SocketAddr> {
    // Bind to an ephemeral port for sending broadcasts
    let socket = UdpSocket::bind("0.0.0.0:0").await?;
    socket.set_broadcast(true)?;

    // Collect the local hardware profile for the ping payload
    let hardware = crate::cluster::detect_hardware_profile();
    let hw_json = serde_json::to_string(&hardware)?;
    let ping_payload = format!("{}{}", PING_PREFIX, hw_json);

    // The broadcast destination: 255.255.255.255 on the discovery port.
    // This works on both standard LANs and smartphone hotspot networks.
    let broadcast_addr: SocketAddr = format!("255.255.255.255:{}", DISCOVERY_PORT)
        .parse()
        .expect("Invalid broadcast address");

    info!(
        "Broadcasting discovery ping to {} (attempt interval: {}ms)",
        broadcast_addr, BROADCAST_INTERVAL_MS
    );

    let mut buf = vec![0u8; MAX_DATAGRAM_SIZE];

    for attempt in 1..=MAX_BROADCAST_ATTEMPTS {
        // Send the discovery ping
        match socket.send_to(ping_payload.as_bytes(), broadcast_addr).await {
            Ok(bytes_sent) => {
                debug!("Broadcast attempt {}: sent {} bytes", attempt, bytes_sent);
            }
            Err(e) => {
                warn!("Broadcast send error (attempt {}): {}", attempt, e);
            }
        }

        // Wait for a PONG response with a timeout
        let recv_future = socket.recv_from(&mut buf);
        let timeout = tokio::time::timeout(
            tokio::time::Duration::from_millis(BROADCAST_INTERVAL_MS),
            recv_future,
        );

        match timeout.await {
            Ok(Ok((bytes_read, src_addr))) => {
                let payload = match std::str::from_utf8(&buf[..bytes_read]) {
                    Ok(s) => s,
                    Err(_) => continue,
                };

                if payload.starts_with(PONG_PREFIX) {
                    let node_id_str = &payload[PONG_PREFIX.len()..];
                    info!(
                        "Master discovered at {}. Assigned node_id={}",
                        src_addr, node_id_str
                    );
                    return Ok(src_addr);
                }
            }
            Ok(Err(e)) => {
                debug!("UDP recv error during discovery: {}", e);
            }
            Err(_) => {
                // Timeout — no response received within the interval.
                // Continue broadcasting.
                if attempt % 10 == 0 {
                    info!(
                        "Still searching for cluster master... (attempt {}/{})",
                        attempt, MAX_BROADCAST_ATTEMPTS
                    );
                }
            }
        }
    }

    Err(anyhow::anyhow!(
        "Failed to discover cluster master after {} attempts ({} seconds). \
         Ensure the master node is running ('airdrop start') on the same network.",
        MAX_BROADCAST_ATTEMPTS,
        (MAX_BROADCAST_ATTEMPTS as u64 * BROADCAST_INTERVAL_MS) / 1000
    ))
}
