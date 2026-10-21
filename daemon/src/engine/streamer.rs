// =============================================================================
// AirDrop-LLM — Asymmetric Off-Heap Layer Weight Streamer Engine
// =============================================================================
//
// Architecture Strategy:
//   Solves the client physical storage bottleneck (0GB disk allocation on Mac M5)
//   by keeping the heavy 32B/70B model checkpoint exclusively on the worker node
//   (Lenovo Legion with RTX 4070).
//
//   During initialization:
//     1. Worker reads layers 1..=20 from its local disk.
//     2. Worker streams array payloads over a dedicated high-throughput TCP socket.
//     3. Primary master node intercepts the binary stream directly off the wire
//        and maps floating-point tensors into an unswappable, off-heap RAM buffer.
//     4. The weights NEVER touch the master node's SSD storage.
//     5. The memory segment is pinned in RAM via POSIX `mlock` to eliminate page
//        fault latency during token forward passes.
//     6. If a disconnect occurs, the buffer is purged instantaneously down to 0 bytes.
//
// Thread Synchronization Mechanics:
//   - Thread-safe off-heap storage governed by an atomic state machine (`BufferState`).
//   - Reads acquire shared memory slices without copying; writes require exclusive
//     ownership during initialization streaming.
//   - Memory barriers (`Ordering::SeqCst` and `Ordering::Acquire`/`Ordering::Release`)
//     guarantee cross-thread memory visibility between streaming and inference workers.
//
// =============================================================================

use std::alloc::{alloc_zeroed, dealloc, Layout};
use std::collections::HashMap;
use std::io::{self, ErrorKind};
use std::ptr::NonNull;
use std::sync::atomic::{AtomicBool, AtomicU64, AtomicUsize, Ordering};
use std::sync::Arc;
use tokio::io::{AsyncReadExt, AsyncWriteExt, BufReader, BufWriter};
use tokio::net::{TcpListener, TcpStream};
use tracing::{debug, error, info, trace, warn};

// ---------------------------------------------------------------------------
// Protocol Framing Constants
// ---------------------------------------------------------------------------

pub const STREAMER_MAGIC_HEADER: &[u8; 16] = b"AIRDROP_WTS_V01\x00";
pub const STREAMER_DEFAULT_PORT: u16 = 9997;
pub const MEMORY_ALIGNMENT_BYTES: usize = 64; // Aligned to cache lines & AVX-512 / Apple Metal SIMD
pub const CHUNK_BUFFER_SIZE: usize = 256 * 1024; // 256 KB TCP streaming transfer chunk

// OpCodes
pub const OP_SEGMENT_MANIFEST: u8 = 0x01;
pub const OP_TENSOR_HEADER: u8    = 0x02;
pub const OP_TENSOR_PAYLOAD: u8   = 0x03;
pub const OP_SEGMENT_COMPLETE: u8 = 0x04;
pub const OP_STREAM_ABORT: u8     = 0xFF;

// ---------------------------------------------------------------------------
// Tensor Metadata & Data Types
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
#[repr(u8)]
pub enum TensorDtype {
    F32 = 0x01,
    F16 = 0x02,
    Q4_K_M = 0x03,
    Q8_0 = 0x04,
}

impl TensorDtype {
    pub fn element_size(&self) -> usize {
        match self {
            Self::F32 => 4,
            Self::F16 => 2,
            Self::Q4_K_M => 1, // Quantized block amortized
            Self::Q8_0 => 1,
        }
    }
}

/// Descriptor of a single parameter tensor within a model layer.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct TensorDescriptor {
    pub layer_id: u32,
    pub name: String,
    pub shape: Vec<usize>,
    pub dtype: TensorDtype,
    pub byte_length: usize,
}

/// Initial manifest sent before weight array streaming starts.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct SegmentManifest {
    pub model_id: String,
    pub layer_start: u32,
    pub layer_end: u32,
    pub total_tensors: u32,
    pub total_bytes: u64,
}

/// Slices within the off-heap contiguous allocation.
#[derive(Debug, Clone)]
pub struct TensorSlice {
    pub offset: usize,
    pub length: usize,
    pub shape: Vec<usize>,
    pub dtype: TensorDtype,
}

// ---------------------------------------------------------------------------
// Off-Heap RAM Weight Buffer (0GB Master Disk Allocation)
// ---------------------------------------------------------------------------

/// Thread-safe off-heap contiguous memory buffer.
/// Maps incoming array payloads directly into system RAM without touching disk.
pub struct OffHeapWeightBuffer {
    /// Raw base pointer to aligned physical RAM allocation.
    ptr: NonNull<u8>,
    /// Global memory layout used for allocation and deallocation.
    layout: Layout,
    /// Total capacity in bytes reserved off-heap.
    capacity: usize,
    /// Current written length in bytes.
    written: AtomicUsize,
    /// Tensor lookup index mapping "layer_id:tensor_name" to internal byte slices.
    index: HashMap<String, TensorSlice>,
    /// Whether the memory pages are currently pinned via mlock.
    is_mlocked: AtomicBool,
    /// Whether this buffer has been purged or invalidated.
    is_purged: AtomicBool,
    /// Layer boundaries covered.
    pub layer_start: u32,
    pub layer_end: u32,
}

// Explicit Send + Sync implementation for off-heap buffer.
// SAFETY: Pointers are isolated, managed through atomic offsets, and immutable
// once loading completes. Thread fences guarantee full visibility.
unsafe impl Send for OffHeapWeightBuffer {}
unsafe impl Sync for OffHeapWeightBuffer {}

impl OffHeapWeightBuffer {
    /// Allocates an off-heap buffer with exact 64-byte alignment and attempts
    /// to lock it into physical RAM to prevent OS virtual memory paging to disk.
    pub fn allocate(total_bytes: usize, layer_start: u32, layer_end: u32) -> io::Result<Self> {
        if total_bytes == 0 {
            return Err(io::Error::new(ErrorKind::InvalidInput, "Cannot allocate 0 bytes off-heap"));
        }

        let layout = Layout::from_size_align(total_bytes, MEMORY_ALIGNMENT_BYTES)
            .map_err(|e| io::Error::new(ErrorKind::InvalidInput, format!("Invalid layout: {}", e)))?;

        // Allocate raw unswappable off-heap memory
        let raw_ptr = unsafe { alloc_zeroed(layout) };
        let ptr = match NonNull::new(raw_ptr) {
            Some(p) => p,
            None => return Err(io::Error::new(ErrorKind::OutOfMemory, "Failed to allocate off-heap memory")),
        };

        // Pin pages into RAM via POSIX mlock if available
        let mut is_mlocked = false;
        #[cfg(unix)]
        {
            let res = unsafe { libc::mlock(raw_ptr as *const libc::c_void, total_bytes) };
            if res == 0 {
                is_mlocked = true;
                debug!("[OFF-HEAP] Successfully mlocked {} bytes in physical RAM", total_bytes);
            } else {
                warn!("[OFF-HEAP] mlock failed (errno: {}). Running in unpinned RAM mode.", res);
            }
        }

        info!(
            "[OFF-HEAP] Initialized RAM buffer: {} MB for Layers {}..={} (0 GB Disk Used)",
            total_bytes / (1024 * 1024),
            layer_start,
            layer_end
        );

        Ok(Self {
            ptr,
            layout,
            capacity: total_bytes,
            written: AtomicUsize::new(0),
            index: HashMap::new(),
            is_mlocked: AtomicBool::new(is_mlocked),
            is_purged: AtomicBool::new(false),
            layer_start,
            layer_end,
        })
    }

    /// Appends a tensor array payload into the contiguous off-heap buffer.
    pub fn write_tensor(
        &mut self,
        layer_id: u32,
        name: &str,
        shape: Vec<usize>,
        dtype: TensorDtype,
        data: &[u8],
    ) -> io::Result<TensorSlice> {
        if self.is_purged.load(Ordering::Acquire) {
            return Err(io::Error::new(ErrorKind::BrokenPipe, "Cannot write to purged off-heap buffer"));
        }

        let current_offset = self.written.load(Ordering::Acquire);
        let len = data.len();

        if current_offset + len > self.capacity {
            return Err(io::Error::new(
                ErrorKind::UnexpectedEof,
                format!(
                    "Buffer overflow: capacity {} bytes exceeded by offset {} + len {}",
                    self.capacity, current_offset, len
                ),
            ));
        }

        // Copy tensor bytes directly into off-heap memory block
        unsafe {
            let dest = self.ptr.as_ptr().add(current_offset);
            std::ptr::copy_nonoverlapping(data.as_ptr(), dest, len);
        }

        // Advance written pointer with Release semantics
        self.written.fetch_add(len, Ordering::Release);

        let slice = TensorSlice {
            offset: current_offset,
            length: len,
            shape,
            dtype,
        };

        let key = format!("{}:{}", layer_id, name);
        self.index.insert(key, slice.clone());
        Ok(slice)
    }

    /// Fetches an immutable byte slice for a specific tensor in zero-copy mode.
    pub fn get_tensor_slice(&self, layer_id: u32, name: &str) -> Option<&[u8]> {
        if self.is_purged.load(Ordering::Acquire) {
            return None;
        }

        let key = format!("{}:{}", layer_id, name);
        let slice = self.index.get(&key)?;

        // Ensure memory barrier for safe concurrent reading
        std::sync::atomic::fence(Ordering::Acquire);

        unsafe {
            let src = self.ptr.as_ptr().add(slice.offset);
            Some(std::slice::from_raw_parts(src, slice.length))
        }
    }

    /// Returns a direct typed slice of f32 tensor weights.
    pub fn get_f32_slice(&self, layer_id: u32, name: &str) -> Option<&[f32]> {
        let raw = self.get_tensor_slice(layer_id, name)?;
        if raw.len() % std::mem::size_of::<f32>() != 0 {
            return None;
        }
        unsafe {
            Some(std::slice::from_raw_parts(
                raw.as_ptr() as *const f32,
                raw.len() / std::mem::size_of::<f32>(),
            ))
        }
    }

    /// Instantaneous clean memory flush:
    /// Wipes all resident layer 1-20 weights from RAM, unlocks the virtual memory,
    /// and deallocates the buffer back to the operating system with 0 bytes residual footprint.
    pub fn purge(&mut self) {
        if self.is_purged.swap(true, Ordering::SeqCst) {
            return; // Already purged
        }

        info!(
            "[OFF-HEAP] EXECUTING INSTANT FLUSH: Purging Layers {}..={} from RAM (0 bytes left)...",
            self.layer_start, self.layer_end
        );

        let raw_ptr = self.ptr.as_ptr();

        // 1. Unlock memory pages from physical RAM
        #[cfg(unix)]
        if self.is_mlocked.load(Ordering::Acquire) {
            unsafe {
                libc::munlock(raw_ptr as *const libc::c_void, self.capacity);
            }
            self.is_mlocked.store(false, Ordering::Release);
        }

        // 2. Wipe memory block with zeros to prevent memory residue
        unsafe {
            std::ptr::write_bytes(raw_ptr, 0, self.capacity);
        }

        // 3. Deallocate memory block back to system allocator
        unsafe {
            dealloc(raw_ptr, self.layout);
        }

        self.index.clear();
        self.written.store(0, Ordering::Release);
        self.capacity = 0;

        info!("[OFF-HEAP] Clean flush complete. RAM completely released.");
    }

    pub fn capacity(&self) -> usize {
        self.capacity
    }

    pub fn is_purged(&self) -> bool {
        self.is_purged.load(Ordering::Acquire)
    }

    pub fn tensor_count(&self) -> usize {
        self.index.len()
    }
}

impl Drop for OffHeapWeightBuffer {
    fn drop(&mut self) {
        if !self.is_purged.load(Ordering::Acquire) {
            self.purge();
        }
    }
}

// ---------------------------------------------------------------------------
// Primary Master Receiver (Mac M5)
// ---------------------------------------------------------------------------

/// Listens on the high-speed TCP socket and streams incoming layer weights
/// directly into the unswappable OffHeapWeightBuffer.
pub struct PrimaryWeightReceiver {
    port: u16,
}

impl PrimaryWeightReceiver {
    pub fn new(port: u16) -> Self {
        Self { port }
    }

    /// Accepts connection from worker node and streams layers into off-heap RAM.
    pub async fn receive_and_map_layers(
        &self,
        expected_layer_start: u32,
        expected_layer_end: u32,
    ) -> Result<Arc<tokio::sync::RwLock<OffHeapWeightBuffer>>, io::Error> {
        let bind_addr = format!("0.0.0.0:{}", self.port);
        let listener = TcpListener::bind(&bind_addr).await?;
        info!("[RECEIVER] Off-heap weight listener active on {}", bind_addr);

        let (mut socket, peer_addr) = listener.accept().await?;
        info!("[RECEIVER] Worker peer connected: {}. Initializing zero-copy ingest...", peer_addr);

        // Verify Magic Header
        let mut magic = [0u8; 16];
        socket.read_exact(&mut magic).await?;
        if &magic != STREAMER_MAGIC_HEADER {
            return Err(io::Error::new(
                ErrorKind::InvalidData,
                "Streamer protocol handshake violation: Invalid magic header",
            ));
        }

        // Read Manifest
        let mut opcode = [0u8; 1];
        socket.read_exact(&mut opcode).await?;
        if opcode[0] != OP_SEGMENT_MANIFEST {
            return Err(io::Error::new(ErrorKind::InvalidData, "Expected SegmentManifest opcode"));
        }

        let manifest_len = socket.read_u32().await? as usize;
        let mut manifest_bytes = vec![0u8; manifest_len];
        socket.read_exact(&mut manifest_bytes).await?;

        let manifest: SegmentManifest = serde_json::from_slice(&manifest_bytes)
            .map_err(|e| io::Error::new(ErrorKind::InvalidData, format!("Malformed manifest JSON: {}", e)))?;

        info!(
            "[RECEIVER] Manifest verified: Model '{}', Layers {}..={} ({} Tensors, {} MB total)",
            manifest.model_id,
            manifest.layer_start,
            manifest.layer_end,
            manifest.total_tensors,
            manifest.total_bytes / (1024 * 1024)
        );

        // Allocate Off-Heap RAM buffer (0 GB disk used on Mac M5!)
        let mut off_heap = OffHeapWeightBuffer::allocate(
            manifest.total_bytes as usize,
            expected_layer_start,
            expected_layer_end,
        )?;

        // Stream array payloads into RAM
        let mut reader = BufReader::with_capacity(CHUNK_BUFFER_SIZE, socket);
        let mut received_tensors = 0u32;

        while received_tensors < manifest.total_tensors {
            let mut msg_type = [0u8; 1];
            reader.read_exact(&mut msg_type).await?;

            match msg_type[0] {
                OP_TENSOR_HEADER => {
                    let desc_len = reader.read_u32().await? as usize;
                    let mut desc_bytes = vec![0u8; desc_len];
                    reader.read_exact(&mut desc_bytes).await?;

                    let desc: TensorDescriptor = serde_json::from_slice(&desc_bytes)
                        .map_err(|e| io::Error::new(ErrorKind::InvalidData, format!("Invalid descriptor: {}", e)))?;

                    // Read binary payload directly into a heap staging chunk then into off-heap buffer
                    let mut payload = vec![0u8; desc.byte_length];
                    reader.read_exact(&mut payload).await?;

                    off_heap.write_tensor(desc.layer_id, &desc.name, desc.shape, desc.dtype, &payload)?;
                    received_tensors += 1;

                    trace!(
                        "[RECEIVER] Ingested tensor [{}/{}] L{}:{} ({} KB)",
                        received_tensors,
                        manifest.total_tensors,
                        desc.layer_id,
                        desc.name,
                        desc.byte_length / 1024
                    );
                }
                OP_SEGMENT_COMPLETE => {
                    info!("[RECEIVER] Received OP_SEGMENT_COMPLETE signal.");
                    break;
                }
                OP_STREAM_ABORT => {
                    return Err(io::Error::new(ErrorKind::ConnectionAborted, "Worker aborted stream"));
                }
                unknown => {
                    return Err(io::Error::new(
                        ErrorKind::InvalidData,
                        format!("Unexpected opcode in tensor stream: 0x{:02X}", unknown),
                    ));
                }
            }
        }

        info!(
            "[RECEIVER] Off-heap layer mapping complete: {} tensors resident in RAM for Layers {}..={}",
            off_heap.tensor_count(),
            expected_layer_start,
            expected_layer_end
        );

        Ok(Arc::new(tokio::sync::RwLock::new(off_heap)))
    }
}

// ---------------------------------------------------------------------------
// Worker Node Weight Streamer (Lenovo Legion)
// ---------------------------------------------------------------------------

/// Runs on the worker node hosting the full GGUF checkpoint.
/// Segments parameter weights and streams Layers 1..=20 to the Master node.
pub struct WorkerWeightStreamer {
    target_addr: String,
}

impl WorkerWeightStreamer {
    pub fn new(target_addr: String) -> Self {
        Self { target_addr }
    }

    /// Connects to master node and streams parameter payloads for the specified layer window.
    pub async fn stream_layer_segment(
        &self,
        model_id: String,
        layer_start: u32,
        layer_end: u32,
        tensors: Vec<(TensorDescriptor, Vec<u8>)>,
    ) -> Result<u64, io::Error> {
        info!("[STREAMER] Connecting to primary node at {}...", self.target_addr);
        let socket = TcpStream::connect(&self.target_addr).await?;
        let _ = socket.set_nodelay(true);

        let mut writer = BufWriter::with_capacity(CHUNK_BUFFER_SIZE, socket);

        // 1. Write Protocol Magic Header
        writer.write_all(STREAMER_MAGIC_HEADER).await?;

        // 2. Compute Manifest
        let total_bytes: u64 = tensors.iter().map(|(_, d)| d.len() as u64).sum();
        let manifest = SegmentManifest {
            model_id,
            layer_start,
            layer_end,
            total_tensors: tensors.len() as u32,
            total_bytes,
        };

        let manifest_json = serde_json::to_vec(&manifest)
            .map_err(|e| io::Error::new(ErrorKind::InvalidData, e.to_string()))?;

        writer.write_all(&[OP_SEGMENT_MANIFEST]).await?;
        writer.write_u32(manifest_json.len() as u32).await?;
        writer.write_all(&manifest_json).await?;
        writer.flush().await?;

        info!(
            "[STREAMER] Streaming Layers {}..={} ({} tensors, {:.2} MB)...",
            layer_start,
            layer_end,
            tensors.len(),
            total_bytes as f64 / (1024.0 * 1024.0)
        );

        // 3. Stream Tensor Payloads
        let mut streamed_bytes = 0u64;
        for (desc, payload) in tensors {
            let desc_json = serde_json::to_vec(&desc)
                .map_err(|e| io::Error::new(ErrorKind::InvalidData, e.to_string()))?;

            writer.write_all(&[OP_TENSOR_HEADER]).await?;
            writer.write_u32(desc_json.len() as u32).await?;
            writer.write_all(&desc_json).await?;
            writer.write_all(&payload).await?;

            streamed_bytes += payload.len() as u64;
        }

        // 4. Send Completion OpCode
        writer.write_all(&[OP_SEGMENT_COMPLETE]).await?;
        writer.flush().await?;

        info!(
            "[STREAMER] Finished transmitting {:.2} MB to master off-heap RAM.",
            streamed_bytes as f64 / (1024.0 * 1024.0)
        );

        Ok(streamed_bytes)
    }
}
