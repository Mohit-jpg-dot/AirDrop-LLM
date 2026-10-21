// =============================================================================
// AirDrop-LLM — High-Speed Asynchronous Frontend SSE Stream Router
// =============================================================================
//
// Architecture Strategy:
//   Bypasses WebSocket packet encapsulation overhead and eliminates user-interface
//   text-rendering latency by streaming raw Server-Sent Events (SSE) directly from
//   the local Axum OpenAI-compatible reverse proxy gateway (:9995).
//
// Memory & Rendering Mechanics:
//   1. Zero-Allocation Byte Streaming:
//      Utilizes the native Streams API (`response.body.getReader()`) paired with a
//      streaming `TextDecoder("utf-8", { stream: true })`. Network chunks are decoded
//      incrementally into memory without creating massive heap copies.
//
//   2. Microsecond String Pointer Boundary Slicing:
//      Incoming network buffers are segmented along newline boundaries (`\n`).
//      Incomplete token fragments trailing a chunk boundary are retained in a compact
//      remainder variable (`bufferRemainder`) to guarantee UTF-8 multiline coherence.
//
//   3. Reflow-Free DOM Insertion (Direct TextNode Append):
//      Avoids destructive `innerHTML` re-parsing on every token arrival. Tokens are
//      injected directly into an active DOM `Text` node using `appendData()`, keeping
//      the layout engine quiescent and eliminating 60fps frame drops.
//
//   4. Fault-Tolerant Circuit Breaker:
//      Catches connection dropouts, host hotspot disconnects, or socket closures
//      with structured `[UI-PANIC]` telemetry logging, reverting UI state cleanly.
//
// =============================================================================

"use strict";

(function (global) {
    // -------------------------------------------------------------------------
    // Configuration Defaults & Constants
    // -------------------------------------------------------------------------
    const DEFAULT_GATEWAY_URL = "http://localhost:9995/v1/chat/completions";
    const DEFAULT_MODEL_ID = "pooled-large-core";
    const SSE_DATA_PREFIX = "data: ";
    const SSE_DONE_MARKER = "[DONE]";
    const PREFIX_LEN = 6; // Length of "data: "

    // Module-level active stream controller for cancellation
    let currentAbortController = null;
    let isStreamActive = false;

    /**
     * Abort any ongoing inference stream immediately.
     */
    function abortActiveInferenceStream() {
        if (currentAbortController) {
            currentAbortController.abort();
            currentAbortController = null;
            isStreamActive = false;
        }
    }

    /**
     * Checks if a stream is currently active.
     * @returns {boolean}
     */
    function isStreamingActive() {
        return isStreamActive;
    }

    /**
     * High-speed zero-copy stream consumer targeting the local AirDrop-LLM Axum gateway (:9995).
     *
     * @param {Object} config - Execution configuration options.
     * @param {string|Array} config.prompt - User prompt string or OpenAI-style messages array.
     * @param {HTMLElement|string} [config.targetElement] - DOM container element or CSS selector to render text into.
     * @param {string} [config.gatewayUrl] - Endpoint URL (default: http://localhost:9995/v1/chat/completions).
     * @param {string} [config.model] - Model tag identifier (default: pooled-large-core).
     * @param {number} [config.maxTokens] - Max generation tokens (default: 512).
     * @param {number} [config.temperature] - Generation temperature (default: 0.7).
     * @param {Function} [config.onStart] - Callback fired upon receiving HTTP 200 headers (arg: ttftMs).
     * @param {Function} [config.onToken] - Callback fired on every incoming token chunk (args: tokenText, fullAccumulatedText).
     * @param {Function} [config.onComplete] - Callback fired when stream finishes (args: fullAccumulatedText, stats).
     * @param {Function} [config.onError] - Callback fired on network drop or exception (arg: Error).
     * @returns {Promise<string>} - Resolves with the complete accumulated text.
     */
    async function streamInferenceOutputToDashboard(config = {}) {
        const {
            prompt,
            targetElement,
            gatewayUrl = DEFAULT_GATEWAY_URL,
            model = DEFAULT_MODEL_ID,
            maxTokens = 512,
            temperature = 0.7,
            onStart = null,
            onToken = null,
            onComplete = null,
            onError = null,
        } = config;

        // Abort previous stream if still pumping
        abortActiveInferenceStream();

        const abortController = new AbortController();
        currentAbortController = abortController;
        isStreamActive = true;

        // Resolve Target DOM Element & Prepare Fast TextNode
        let domContainer = null;
        let textNodeTarget = null;

        if (targetElement) {
            domContainer = typeof targetElement === "string" ? document.querySelector(targetElement) : targetElement;
            if (domContainer) {
                // Clear container and create an isolated TextNode for microsecond updates
                domContainer.textContent = "";
                textNodeTarget = document.createTextNode("");
                domContainer.appendChild(textNodeTarget);
            }
        }

        // Format OpenAI-Compatible Chat Payload
        let messagesPayload;
        if (Array.isArray(prompt)) {
            messagesPayload = prompt;
        } else {
            messagesPayload = [
                {
                    role: "user",
                    content: typeof prompt === "string" ? prompt : String(prompt || ""),
                },
            ];
        }

        const requestPayload = {
            model: model,
            messages: messagesPayload,
            stream: true, // Force Axum gateway SSE data-path
            max_tokens: Number(maxTokens) || 512,
            temperature: Number(temperature) || 0.7,
        };

        const startTime = performance.now();
        let firstTokenReceived = false;
        let accumulatedText = "";
        let bufferRemainder = "";
        let tokenCount = 0;

        try {
            // Initiate Bare-Metal Fetch Stream targeting Port 9995
            const response = await fetch(gatewayUrl, {
                method: "POST",
                headers: {
                    "Content-Type": "application/json",
                    Accept: "text/event-stream",
                },
                body: JSON.stringify(requestPayload),
                signal: abortController.signal,
            });

            if (!response.ok) {
                const errorBody = await response.text().catch(() => "");
                throw new Error(
                    `[UI-PANIC] Gateway HTTP ${response.status} (${response.statusText}): ${errorBody || "Inference pipeline rejected request"}`
                );
            }

            if (!response.body) {
                throw new Error("[UI-PANIC] ReadableStream unsupported or missing response body from gateway.");
            }

            // Acquire Bare-Metal Stream Reader
            const reader = response.body.getReader();
            const textDecoder = new TextDecoder("utf-8", { stream: true });

            // -----------------------------------------------------------------
            // Non-Blocking Asynchronous Chunk Ingestion Loop
            // -----------------------------------------------------------------
            while (true) {
                const { done, value } = await reader.read();

                if (done) {
                    break;
                }

                if (!value || value.length === 0) {
                    continue;
                }

                // Decode incoming byte chunk without full string reallocation
                const decodedChunk = textDecoder.decode(value, { stream: true });
                const combinedBuffer = bufferRemainder + decodedChunk;

                // Segment network buffer along newline delimiters
                const lines = combinedBuffer.split("\n");

                // Retain the trailing incomplete segment
                bufferRemainder = lines.pop() || "";

                // Process complete SSE lines
                for (let i = 0; i < lines.length; i++) {
                    let line = lines[i];

                    // Strip carriage return if present
                    if (line.charCodeAt(line.length - 1) === 13) {
                        line = line.slice(0, -1);
                    }

                    if (!line || !line.startsWith(SSE_DATA_PREFIX)) {
                        continue;
                    }

                    // Memory-efficient 6-char slice: removes "data: "
                    const payloadString = line.slice(PREFIX_LEN).trim();

                    // Suppress structural heartbeat delimiter
                    if (payloadString === SSE_DONE_MARKER) {
                        continue;
                    }

                    // Fast Minified JSON Deserialization
                    try {
                        const parsed = JSON.parse(payloadString);
                        const deltaContent = parsed?.choices?.[0]?.delta?.content;

                        if (deltaContent && typeof deltaContent === "string") {
                            // Capture Time to First Token (TTFT)
                            if (!firstTokenReceived) {
                                firstTokenReceived = true;
                                const ttftMs = Math.round(performance.now() - startTime);
                                if (typeof onStart === "function") {
                                    try {
                                        onStart(ttftMs);
                                    } catch (_) {}
                                }
                            }

                            tokenCount++;
                            accumulatedText += deltaContent;

                            // Direct DOM TextNode Update (Zero HTML Re-parsing)
                            if (textNodeTarget) {
                                textNodeTarget.appendData(deltaContent);
                            }

                            // Optional Token Callback
                            if (typeof onToken === "function") {
                                onToken(deltaContent, accumulatedText);
                            }
                        }
                    } catch (_) {
                        // Ignore malformed transient line fragments
                    }
                }
            }

            // Flush any remaining line buffer after stream EOF
            if (bufferRemainder && bufferRemainder.startsWith(SSE_DATA_PREFIX)) {
                const trailingPayload = bufferRemainder.slice(PREFIX_LEN).trim();
                if (trailingPayload && trailingPayload !== SSE_DONE_MARKER) {
                    try {
                        const parsed = JSON.parse(trailingPayload);
                        const delta = parsed?.choices?.[0]?.delta?.content;
                        if (delta) {
                            accumulatedText += delta;
                            if (textNodeTarget) textNodeTarget.appendData(delta);
                        }
                    } catch (_) {}
                }
            }

            // -----------------------------------------------------------------
            // Stream Finalization & Telemetry Statistics
            // -----------------------------------------------------------------
            const totalDurationMs = Math.round(performance.now() - startTime);
            const tokensPerSecond = totalDurationMs > 0 ? ((tokenCount / totalDurationMs) * 1000).toFixed(1) : "0.0";

            const streamStats = {
                tokensGenerated: tokenCount,
                durationMs: totalDurationMs,
                tokensPerSecond: parseFloat(tokensPerSecond),
                modelUsed: model,
            };

            isStreamActive = false;
            currentAbortController = null;

            if (typeof onComplete === "function") {
                try {
                    onComplete(accumulatedText, streamStats);
                } catch (_) {}
            }

            return accumulatedText;
        } catch (error) {
            isStreamActive = false;
            currentAbortController = null;

            // Handle deliberate user cancellation
            if (error.name === "AbortError") {
                return accumulatedText;
            }

            // Structured Diagnostic Warning
            console.error(
                `%c[UI-PANIC]%c High-speed SSE stream connection dropped on ${gatewayUrl}: ${error.message}`,
                "color: #ef4444; font-weight: bold; background: rgba(239,68,68,0.1); padding: 2px 4px; border-radius: 4px;",
                "color: #f87171;"
            );

            if (typeof onError === "function") {
                try {
                    onError(error);
                } catch (_) {}
            }

            throw error;
        }
    }

    // -------------------------------------------------------------------------
    // Universal Export (Window Global & ES / CommonJS module bindings)
    // -------------------------------------------------------------------------
    const API = {
        streamInferenceOutputToDashboard,
        abortActiveInferenceStream,
        isStreamingActive,
        DEFAULT_GATEWAY_URL,
    };

    if (typeof module !== "undefined" && module.exports) {
        module.exports = API;
    }

    if (typeof define === "function" && define.amd) {
        define([], () => API);
    }

    global.streamInferenceOutputToDashboard = streamInferenceOutputToDashboard;
    global.abortActiveInferenceStream = abortActiveInferenceStream;
    global.isStreamingActive = isStreamingActive;
    global.AirDropGatewayListener = API;
})(typeof window !== "undefined" ? window : globalThis);
