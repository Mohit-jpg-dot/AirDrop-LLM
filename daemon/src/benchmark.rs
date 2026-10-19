// =============================================================================
// AirDrop-LLM — Network Performance Benchmarking Engine
// =============================================================================
//
// Before initiating a distributed inference session, the master node must
// evaluate the true network performance characteristics of each TCP link
// to every registered worker. This prevents the scheduler from assigning
// layers to nodes that would bottleneck the pipeline.
//
// Benchmark Procedure (per worker):
//   1. TCP throughput test: Blast a 10 MB synthetic byte array (simulating
//      a compressed tensor chunk) and measure transfer time → Mbps.
//   2. Latency test: 10 rapid ping/pong round-trips measuring RTT → ms.
//
// Evaluation Gate:
//   - Throughput must be >= 150 Mbps
//   - Latency must be <= 20 ms
//   If either condition fails, the node is flagged as a bottleneck and
//   the scheduler applies routing overrides (reduced layer allocation or
//   exclusion from the pipeline).
//
// =============================================================================

use crate::cluster::BenchmarkResult;
use std::net::SocketAddr;
use std::time::Instant;
use tokio::io::{AsyncReadExt, AsyncWriteExt};
use tokio::net::TcpStream;
use tracing::{debug, info, warn};

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

/// Size of the throughput test payload in bytes (10 MB).
const THROUGHPUT_PAYLOAD_SIZE: usize = 10 * 1024 * 1024;

/// Number of ping/pong rounds for latency measurement.
const PING_ROUNDS: u32 = 10;

/// Size of the ping payload in bytes (small, to isolate RTT from bandwidth).
const PING_PAYLOAD_SIZE: usize = 64;

/// Minimum acceptable throughput in Mbps.
const MIN_THROUGHPUT_MBPS: f64 = 150.0;

/// Maximum acceptable round-trip latency in milliseconds.
const MAX_LATENCY_MS: f64 = 20.0;

/// TCP port for benchmark operations (reuses the data port).
pub const BENCHMARK_PORT: u16 = 9999;

/// Protocol header for benchmark initiation.
const BENCH_HEADER: &[u8] = b"AIRDROP_BENCH";

/// Protocol header for ping operations.
const PING_HEADER: &[u8] = b"AIRDROP_PING\0";

/// Protocol header for pong responses.
const PONG_HEADER: &[u8] = b"AIRDROP_PONG\0";

// ---------------------------------------------------------------------------
// Master-Side Benchmark Executor
// ---------------------------------------------------------------------------

/// Runs the full benchmark suite against a single worker node.
///
/// This function is called by the master node after a worker registers.
/// It establishes a dedicated TCP connection to the worker's benchmark
/// port, runs the throughput and latency tests, and returns the results.
///
/// # Arguments
/// * `worker_addr` - The worker's IP address.
///
/// # Returns
/// A `BenchmarkResult` containing throughput (Mbps), latency (ms), and
/// whether the connection passes the quality gate.
pub async fn run_benchmark(worker_addr: &str) -> anyhow::Result<BenchmarkResult> {
    let addr: SocketAddr = format!("{}:{}", worker_addr, BENCHMARK_PORT).parse()?;

    info!("Starting network benchmark against {}", addr);

    // Establish benchmark TCP connection with timeout
    let mut stream = tokio::time::timeout(
        tokio::time::Duration::from_secs(10),
        TcpStream::connect(addr),
    )
    .await
    .map_err(|_| anyhow::anyhow!("Benchmark connection timed out"))??;

    // Disable Nagle's algorithm for accurate latency measurement
    stream.set_nodelay(true)?;

    // Send benchmark initiation header
    stream.write_all(BENCH_HEADER).await?;
    stream.flush().await?;

    // Wait for acknowledgment
    let mut ack_buf = [0u8; 2];
    stream.read_exact(&mut ack_buf).await?;
    if &ack_buf != b"OK" {
        return Err(anyhow::anyhow!("Benchmark handshake failed"));
    }

    // --- Phase 1: Throughput Test ---
    let throughput_mbps = measure_throughput(&mut stream).await?;
    info!("  Throughput: {:.2} Mbps", throughput_mbps);

    // --- Phase 2: Latency Test ---
    let latency_ms = measure_latency(&mut stream).await?;
    info!("  Latency:    {:.2} ms (avg over {} rounds)", latency_ms, PING_ROUNDS);

    // --- Evaluation Gate ---
    let passes_gate = throughput_mbps >= MIN_THROUGHPUT_MBPS && latency_ms <= MAX_LATENCY_MS;

    if passes_gate {
        info!(
            "  ✓ Benchmark PASSED (≥{} Mbps, ≤{} ms)",
            MIN_THROUGHPUT_MBPS, MAX_LATENCY_MS
        );
    } else {
        warn!(
            "  ✗ Benchmark FAILED — Infrastructure bottleneck detected!\n    \
             Throughput: {:.2} Mbps (min: {} Mbps)\n    \
             Latency:    {:.2} ms (max: {} ms)\n    \
             → Scheduler will apply routing override for this node.",
            throughput_mbps, MIN_THROUGHPUT_MBPS, latency_ms, MAX_LATENCY_MS
        );
    }

    Ok(BenchmarkResult {
        throughput_mbps,
        latency_ms,
        passes_gate,
        timestamp: chrono::Utc::now(),
    })
}

// ---------------------------------------------------------------------------
// Throughput Measurement
// ---------------------------------------------------------------------------

/// Measures TCP throughput by sending a 10 MB payload and timing the transfer.
///
/// Protocol:
///   1. Master sends 4-byte length prefix (big-endian u32).
///   2. Master sends the full payload.
///   3. Worker reads the full payload and responds with "DONE".
///   4. Master calculates throughput from elapsed time.
async fn measure_throughput(stream: &mut TcpStream) -> anyhow::Result<f64> {
    // Generate the test payload (pseudo-random pattern to prevent compression)
    let payload = generate_test_payload(THROUGHPUT_PAYLOAD_SIZE);

    // Send length prefix
    let len_bytes = (THROUGHPUT_PAYLOAD_SIZE as u32).to_be_bytes();
    stream.write_all(&len_bytes).await?;

    // Start timing
    let start = Instant::now();

    // Send the full payload in chunks to prevent buffer overflow
    const CHUNK_SIZE: usize = 64 * 1024; // 64 KB write chunks
    for chunk in payload.chunks(CHUNK_SIZE) {
        stream.write_all(chunk).await?;
    }
    stream.flush().await?;

    // Wait for completion acknowledgment from the worker
    let mut done_buf = [0u8; 4];
    stream.read_exact(&mut done_buf).await?;

    let elapsed = start.elapsed();

    // Calculate throughput in Mbps
    // (bytes * 8 bits/byte) / (elapsed_seconds * 1_000_000 bits/Mbit)
    let bits = (THROUGHPUT_PAYLOAD_SIZE as f64) * 8.0;
    let mbps = bits / (elapsed.as_secs_f64() * 1_000_000.0);

    debug!(
        "Throughput test: {} bytes in {:.3}s = {:.2} Mbps",
        THROUGHPUT_PAYLOAD_SIZE,
        elapsed.as_secs_f64(),
        mbps
    );

    Ok(mbps)
}

// ---------------------------------------------------------------------------
// Latency Measurement
// ---------------------------------------------------------------------------

/// Measures network RTT latency via 10 rapid ping/pong exchanges.
///
/// Protocol (per round):
///   1. Master sends PING_HEADER + 64 bytes.
///   2. Worker immediately echoes PONG_HEADER + 64 bytes.
///   3. Master records the round-trip time.
///
/// Returns the average RTT across all rounds in milliseconds.
async fn measure_latency(stream: &mut TcpStream) -> anyhow::Result<f64> {
    let ping_data = vec![0xAA_u8; PING_PAYLOAD_SIZE];
    let mut pong_buf = vec![0u8; PONG_HEADER.len() + PING_PAYLOAD_SIZE];

    let mut total_rtt_us: u64 = 0;
    let mut successful_rounds: u32 = 0;

    for round in 0..PING_ROUNDS {
        let start = Instant::now();

        // Send ping
        stream.write_all(PING_HEADER).await?;
        stream.write_all(&ping_data).await?;
        stream.flush().await?;

        // Receive pong with timeout
        let recv_result = tokio::time::timeout(
            tokio::time::Duration::from_millis(1000),
            stream.read_exact(&mut pong_buf),
        )
        .await;

        match recv_result {
            Ok(Ok(_)) => {
                let rtt = start.elapsed();
                total_rtt_us += rtt.as_micros() as u64;
                successful_rounds += 1;
                debug!("  Ping round {}: {:.3} ms", round + 1, rtt.as_secs_f64() * 1000.0);
            }
            Ok(Err(e)) => {
                warn!("  Ping round {} read error: {}", round + 1, e);
            }
            Err(_) => {
                warn!("  Ping round {} timed out", round + 1);
            }
        }
    }

    if successful_rounds == 0 {
        return Err(anyhow::anyhow!("All ping rounds failed"));
    }

    // Average RTT in milliseconds
    let avg_rtt_ms = (total_rtt_us as f64) / (successful_rounds as f64) / 1000.0;

    Ok(avg_rtt_ms)
}

// ---------------------------------------------------------------------------
// Worker-Side Benchmark Responder
// ---------------------------------------------------------------------------

/// Handles incoming benchmark requests on the worker side.
///
/// This function is called when the worker receives a TCP connection
/// with the BENCH_HEADER. It responds to throughput and latency tests
/// initiated by the master.
pub async fn handle_benchmark_session(mut stream: TcpStream) -> anyhow::Result<()> {
    info!("Benchmark session started from {}", stream.peer_addr()?);

    // Send acknowledgment
    stream.write_all(b"OK").await?;
    stream.flush().await?;

    // --- Phase 1: Throughput Test (Receive) ---
    // Read length prefix
    let mut len_buf = [0u8; 4];
    stream.read_exact(&mut len_buf).await?;
    let payload_len = u32::from_be_bytes(len_buf) as usize;

    debug!("Benchmark: receiving {} byte throughput payload", payload_len);

    // Read the full payload (discard — we only need to measure transfer time)
    let mut received = 0usize;
    let mut read_buf = vec![0u8; 64 * 1024];
    while received < payload_len {
        let to_read = std::cmp::min(read_buf.len(), payload_len - received);
        let n = stream.read(&mut read_buf[..to_read]).await?;
        if n == 0 {
            return Err(anyhow::anyhow!("Connection closed during throughput test"));
        }
        received += n;
    }

    // Signal completion
    stream.write_all(b"DONE").await?;
    stream.flush().await?;

    debug!("Benchmark: throughput test complete ({} bytes received)", received);

    // --- Phase 2: Latency Test (Echo) ---
    let header_len = PING_HEADER.len();
    let mut ping_buf = vec![0u8; header_len + PING_PAYLOAD_SIZE];

    for round in 0..PING_ROUNDS {
        // Receive ping
        match tokio::time::timeout(
            tokio::time::Duration::from_secs(5),
            stream.read_exact(&mut ping_buf),
        )
        .await
        {
            Ok(Ok(_)) => {
                // Immediately echo as pong
                stream.write_all(PONG_HEADER).await?;
                stream.write_all(&ping_buf[header_len..]).await?;
                stream.flush().await?;
                debug!("Benchmark: ping round {} echoed", round + 1);
            }
            Ok(Err(e)) => {
                warn!("Benchmark: ping round {} read error: {}", round + 1, e);
                break;
            }
            Err(_) => {
                warn!("Benchmark: ping round {} timed out", round + 1);
                break;
            }
        }
    }

    info!("Benchmark session complete.");
    Ok(())
}

// ---------------------------------------------------------------------------
// Utility: Test Payload Generator
// ---------------------------------------------------------------------------

/// Generates a pseudo-random byte array of the specified size.
///
/// Uses a simple LCG (Linear Congruential Generator) to produce
/// non-compressible data that simulates real tensor payload characteristics.
fn generate_test_payload(size: usize) -> Vec<u8> {
    let mut data = vec![0u8; size];
    let mut state: u64 = 0xDEADBEEF_CAFEBABE;

    for byte in data.iter_mut() {
        // LCG: state = (a * state + c) mod 2^64
        state = state.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        *byte = (state >> 33) as u8;
    }

    data
}
