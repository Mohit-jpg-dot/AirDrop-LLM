#!/usr/bin/env node

// =============================================================================
// AirDrop-LLM — Cross-Platform Native Binary Installer
// =============================================================================
//
// Lifecycle: Executes automatically via npm "postinstall" hook.
//
// Responsibilities:
//   1. Detect the host platform (darwin | linux | win32) and CPU architecture
//      (arm64 | x64) using the native Node.js `os` module.
//   2. Resolve the matching pre-compiled Rust daemon binary target triple.
//   3. Fetch the compressed release artifact from GitHub Releases.
//   4. Extract and inject the binary into the local `bin/` execution directory.
//   5. Set executable permissions on POSIX systems.
//
// Supported target triples:
//   - aarch64-apple-darwin        (macOS Apple Silicon)
//   - x86_64-apple-darwin         (macOS Intel)
//   - aarch64-unknown-linux-gnu   (Linux ARM64)
//   - x86_64-unknown-linux-gnu    (Linux x64)
//   - aarch64-pc-windows-msvc     (Windows ARM64)
//   - x86_64-pc-windows-msvc      (Windows x64)
// =============================================================================

"use strict";

const os = require("os");
const fs = require("fs");
const path = require("path");
const https = require("https");
const { execSync } = require("child_process");
const { createWriteStream, mkdirSync, chmodSync, unlinkSync } = fs;
const { pipeline } = require("stream");
const { promisify } = require("util");

const streamPipeline = promisify(pipeline);

// ---------------------------------------------------------------------------
// Configuration Constants
// ---------------------------------------------------------------------------

/** GitHub repository coordinates for release artifact resolution. */
const GITHUB_OWNER = "airdrop-llm";
const GITHUB_REPO = "airdrop-llm";

/** Package version — sourced from package.json to lock binary/CLI parity. */
const PACKAGE_VERSION = require("../package.json").version;

/** Binary output directory within the npm package tree. */
const BIN_DIR = path.resolve(__dirname, "..", "bin");

/** Final daemon binary filename (platform-specific extension applied below). */
const BINARY_NAME = os.platform() === "win32" ? "airdrop-daemon.exe" : "airdrop-daemon";

/** Destination path for the installed binary. */
const BINARY_PATH = path.join(BIN_DIR, BINARY_NAME);

// ---------------------------------------------------------------------------
// Platform → Target Triple Resolution
// ---------------------------------------------------------------------------

/**
 * Maps the host environment signature (os.platform + os.arch) to the
 * corresponding Rust cross-compilation target triple used in CI release
 * builds. Throws on unsupported platform/arch combinations.
 *
 * @returns {{ triple: string, archiveExt: string }} Target metadata.
 */
function resolveTarget() {
  const platform = os.platform(); // 'darwin' | 'linux' | 'win32'
  const arch = os.arch();         // 'arm64' | 'x64'

  /** Platform-architecture matrix → Rust target triple. */
  const TARGET_MAP = {
    "darwin-arm64":  "aarch64-apple-darwin",
    "darwin-x64":    "x86_64-apple-darwin",
    "linux-arm64":   "aarch64-unknown-linux-gnu",
    "linux-x64":     "x86_64-unknown-linux-gnu",
    "win32-arm64":   "aarch64-pc-windows-msvc",
    "win32-x64":     "x86_64-pc-windows-msvc",
  };

  const key = `${platform}-${arch}`;
  const triple = TARGET_MAP[key];

  if (!triple) {
    console.error(
      `[airdrop-llm] ✗ Unsupported platform/architecture: ${key}\n` +
      `  Supported combinations: ${Object.keys(TARGET_MAP).join(", ")}`
    );
    process.exit(1);
  }

  // Windows releases are distributed as .zip; all POSIX targets use .tar.gz
  const archiveExt = platform === "win32" ? "zip" : "tar.gz";

  console.log(`[airdrop-llm] Detected environment: ${key} → ${triple}`);
  return { triple, archiveExt };
}

// ---------------------------------------------------------------------------
// GitHub Release Artifact Download
// ---------------------------------------------------------------------------

/**
 * Constructs the GitHub Releases download URL for the target binary archive.
 *
 * Release naming convention:
 *   https://github.com/{owner}/{repo}/releases/download/v{version}/
 *   airdrop-daemon-v{version}-{triple}.{ext}
 *
 * @param {string} triple    - Rust target triple.
 * @param {string} ext       - Archive file extension ('tar.gz' | 'zip').
 * @returns {string}           Fully qualified download URL.
 */
function buildDownloadUrl(triple, ext) {
  const tag = `v${PACKAGE_VERSION}`;
  const filename = `airdrop-daemon-${tag}-${triple}.${ext}`;
  return `https://github.com/${GITHUB_OWNER}/${GITHUB_REPO}/releases/download/${tag}/${filename}`;
}

/**
 * Downloads a file from the given URL, following up to 5 HTTP 3xx redirects
 * (GitHub Releases serves artifacts via a redirect chain).
 *
 * @param {string} url           - Source URL.
 * @param {string} destination   - Local filesystem write path.
 * @param {number} [maxRedirects=5] - Maximum redirect hops.
 * @returns {Promise<void>}
 */
function downloadFile(url, destination, maxRedirects = 5) {
  return new Promise((resolve, reject) => {
    if (maxRedirects <= 0) {
      return reject(new Error("Too many redirects while downloading binary."));
    }

    https.get(url, { headers: { "User-Agent": "airdrop-llm-installer" } }, (response) => {
      // Handle HTTP redirects (301, 302, 303, 307, 308)
      if (response.statusCode >= 300 && response.statusCode < 400 && response.headers.location) {
        console.log(`[airdrop-llm]   ↳ Following redirect → ${response.headers.location.substring(0, 80)}...`);
        return downloadFile(response.headers.location, destination, maxRedirects - 1)
          .then(resolve)
          .catch(reject);
      }

      if (response.statusCode !== 200) {
        return reject(
          new Error(
            `Failed to download binary: HTTP ${response.statusCode}\n` +
            `  URL: ${url}\n` +
            `  Ensure that release artifacts exist for version ${PACKAGE_VERSION}.`
          )
        );
      }

      const totalBytes = parseInt(response.headers["content-length"], 10) || 0;
      let downloadedBytes = 0;

      response.on("data", (chunk) => {
        downloadedBytes += chunk.length;
        if (totalBytes > 0) {
          const pct = ((downloadedBytes / totalBytes) * 100).toFixed(1);
          process.stdout.write(`\r[airdrop-llm]   Downloading... ${pct}%`);
        }
      });

      const fileStream = createWriteStream(destination);
      streamPipeline(response, fileStream)
        .then(() => {
          if (totalBytes > 0) process.stdout.write("\n");
          resolve();
        })
        .catch(reject);
    }).on("error", reject);
  });
}

// ---------------------------------------------------------------------------
// Archive Extraction
// ---------------------------------------------------------------------------

/**
 * Extracts the daemon binary from the downloaded archive.
 *
 * - For .tar.gz archives: uses the system `tar` command.
 * - For .zip archives: uses PowerShell's Expand-Archive on Windows.
 *
 * @param {string} archivePath - Path to the downloaded archive.
 * @param {string} ext         - Archive extension ('tar.gz' | 'zip').
 */
function extractArchive(archivePath, ext) {
  console.log(`[airdrop-llm]   Extracting binary from archive...`);

  if (ext === "tar.gz") {
    // Extract only the binary file from the tarball into BIN_DIR
    execSync(`tar -xzf "${archivePath}" -C "${BIN_DIR}" ${BINARY_NAME}`, {
      stdio: "pipe",
    });
  } else if (ext === "zip") {
    // Windows: extract via PowerShell, then move binary to BIN_DIR
    const tempDir = path.join(BIN_DIR, "__extract_tmp");
    mkdirSync(tempDir, { recursive: true });

    execSync(
      `powershell -NoProfile -Command "Expand-Archive -Force -Path '${archivePath}' -DestinationPath '${tempDir}'"`,
      { stdio: "pipe" }
    );

    const extractedBinary = path.join(tempDir, BINARY_NAME);
    if (fs.existsSync(extractedBinary)) {
      fs.renameSync(extractedBinary, BINARY_PATH);
    }

    // Clean up temporary extraction directory
    fs.rmSync(tempDir, { recursive: true, force: true });
  }

  // Remove the archive file after successful extraction
  try {
    unlinkSync(archivePath);
  } catch (_) {
    // Non-fatal: archive cleanup failure is acceptable
  }
}

// ---------------------------------------------------------------------------
// Permission Configuration
// ---------------------------------------------------------------------------

/**
 * Sets the executable permission bit on the extracted binary (POSIX only).
 * On Windows, executables do not require explicit permission flags.
 */
function setExecutablePermissions() {
  if (os.platform() !== "win32") {
    chmodSync(BINARY_PATH, 0o755);
    console.log(`[airdrop-llm]   Set executable permissions on ${BINARY_NAME}`);
  }
}

// ---------------------------------------------------------------------------
// Installer Entrypoint
// ---------------------------------------------------------------------------

async function main() {
  console.log(`[airdrop-llm] ──────────────────────────────────────────────`);
  console.log(`[airdrop-llm] AirDrop-LLM Binary Installer v${PACKAGE_VERSION}`);
  console.log(`[airdrop-llm] ──────────────────────────────────────────────`);

  // Step 1: Resolve target triple from host environment
  const { triple, archiveExt } = resolveTarget();

  // Step 2: Ensure the binary output directory exists
  mkdirSync(BIN_DIR, { recursive: true });

  // Step 3: Check if the binary is already installed (idempotent reinstalls)
  if (fs.existsSync(BINARY_PATH)) {
    console.log(`[airdrop-llm] ✓ Binary already exists at ${BINARY_PATH}`);
    console.log(`[airdrop-llm]   Skipping download. Delete the binary to force reinstall.`);
    return;
  }

  // Step 4: Build download URL and fetch the archive
  const downloadUrl = buildDownloadUrl(triple, archiveExt);
  const archivePath = path.join(BIN_DIR, `airdrop-daemon.${archiveExt}`);

  console.log(`[airdrop-llm]   Source: ${downloadUrl}`);

  try {
    await downloadFile(downloadUrl, archivePath);
  } catch (err) {
    console.error(`\n[airdrop-llm] ✗ Binary download failed:\n  ${err.message}`);
    console.error(`[airdrop-llm]   The CLI will attempt to compile from source on first run.`);
    console.error(`[airdrop-llm]   Requires: Rust toolchain (rustup.rs)`);
    process.exit(0); // Non-fatal: allow npm install to complete
  }

  // Step 5: Extract the binary from the archive
  try {
    extractArchive(archivePath, archiveExt);
  } catch (err) {
    console.error(`[airdrop-llm] ✗ Archive extraction failed:\n  ${err.message}`);
    process.exit(1);
  }

  // Step 6: Set executable permissions (POSIX)
  setExecutablePermissions();

  // Step 7: Verify installation
  if (fs.existsSync(BINARY_PATH)) {
    const stats = fs.statSync(BINARY_PATH);
    const sizeMB = (stats.size / (1024 * 1024)).toFixed(2);
    console.log(`[airdrop-llm] ✓ Binary installed successfully (${sizeMB} MB)`);
    console.log(`[airdrop-llm]   Location: ${BINARY_PATH}`);
  } else {
    console.error(`[airdrop-llm] ✗ Binary not found after extraction. Installation failed.`);
    process.exit(1);
  }

  console.log(`[airdrop-llm] ──────────────────────────────────────────────`);
  console.log(`[airdrop-llm] Run 'airdrop start' to launch a cluster head.`);
  console.log(`[airdrop-llm] Run 'airdrop join'  to connect as a worker.`);
  console.log(`[airdrop-llm] ──────────────────────────────────────────────`);
}

main().catch((err) => {
  console.error(`[airdrop-llm] ✗ Installer encountered a fatal error:\n`, err);
  process.exit(1);
});
