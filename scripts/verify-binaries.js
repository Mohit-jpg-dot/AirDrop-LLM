#!/usr/bin/env node

// =============================================================================
// AirDrop-LLM — Pre-publish Binary Verification Hook
// =============================================================================
// Lifecycle: Invoked automatically via npm "prepublishOnly" script hook.
// Purpose:
//   1. Validates that the Rust daemon has compiled into target/release/ or bin/
//   2. Ensures the distribution artifact exists and possesses executable permissions
//   3. Prepares the package tree for npm publish / GitHub Packages distribution
// =============================================================================

"use strict";

const fs = require("fs");
const path = require("path");
const os = require("os");

const ROOT_DIR = path.resolve(__dirname, "..");
const BIN_DIR = path.join(ROOT_DIR, "bin");
const BINARY_NAME = os.platform() === "win32" ? "airdrop-daemon.exe" : "airdrop-daemon";
const TARGET_BINARY = path.join(BIN_DIR, BINARY_NAME);

// Candidate build output locations from cargo build
const CARGO_RELEASE_PATHS = [
  path.join(ROOT_DIR, "daemon", "target", "release", BINARY_NAME),
  path.join(ROOT_DIR, "target", "release", BINARY_NAME),
  TARGET_BINARY
];

console.log("[airdrop-llm:verify] Running pre-publish binary distribution verification...");

let sourceBinary = null;
for (const p of CARGO_RELEASE_PATHS) {
  if (fs.existsSync(p)) {
    sourceBinary = p;
    break;
  }
}

if (!sourceBinary) {
  console.warn(
    `[airdrop-llm:verify] Notice: Compiled daemon binary not found at typical cargo locations:\n` +
    `  ${CARGO_RELEASE_PATHS.join("\n  ")}\n` +
    `Checking repository bin/ distribution directory...`
  );
}

if (sourceBinary && sourceBinary !== TARGET_BINARY) {
  if (!fs.existsSync(BIN_DIR)) {
    fs.mkdirSync(BIN_DIR, { recursive: true });
  }
  fs.copyFileSync(sourceBinary, TARGET_BINARY);
  console.log(`[airdrop-llm:verify] Staged release binary from ${sourceBinary} -> ${TARGET_BINARY}`);
}

if (fs.existsSync(TARGET_BINARY)) {
  const stats = fs.statSync(TARGET_BINARY);
  if (os.platform() !== "win32") {
    // Ensure executable permissions (chmod 0755)
    fs.chmodSync(TARGET_BINARY, 0o755);
  }
  console.log(
    `[airdrop-llm:verify] ✓ Binary verified successfully:\n` +
    `  Path: ${TARGET_BINARY}\n` +
    `  Size: ${(stats.size / 1024).toFixed(2)} KB\n` +
    `  Permissions: 0755 (executable)`
  );
  process.exit(0);
} else {
  console.error(
    `[airdrop-llm:verify] ✗ Error: Verification failed. Binary ${TARGET_BINARY} is missing.\n` +
    `Please ensure "cargo build --release --manifest-path ./daemon/Cargo.toml" completes successfully.`
  );
  process.exit(1);
}
