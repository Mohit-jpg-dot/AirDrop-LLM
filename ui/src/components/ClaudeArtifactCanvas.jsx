import React, { useState, useEffect, useMemo, useCallback } from "react";

/**
 * Generates an isolated, safe HTML document string for the sandboxed iframe.
 * Handles pure HTML documents, HTML snippets, SVG markup, JavaScript, CSS, and Markdown.
 */
function buildSandboxSrcDoc(codePayload, language) {
  const code = typeof codePayload === "string" ? codePayload : String(codePayload || "");
  const lang = (language || "").toLowerCase().trim();

  // If already a full HTML document, inject directly
  if (code.includes("<!DOCTYPE") || code.includes("<html") || code.includes("<HTML")) {
    return code;
  }

  // Standalone SVG markup
  if (lang === "svg" || code.trim().startsWith("<svg")) {
    return `<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <style>
    * { margin: 0; padding: 0; box-sizing: border-box; }
    body {
      width: 100vw;
      height: 100vh;
      display: flex;
      align-items: center;
      justify-content: center;
      background-color: #ffffff;
      padding: 24px;
      overflow: auto;
    }
    svg {
      max-width: 100%;
      max-height: 100%;
      height: auto;
      filter: drop-shadow(0 2px 8px rgba(0,0,0,0.06));
    }
  </style>
</head>
<body>
  ${code}
</body>
</html>`;
  }

  // HTML fragments or UI layouts
  if (lang === "html" || lang === "htm" || lang === "xml" || code.trim().startsWith("<")) {
    return `<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>AirDrop-LLM Artifact Preview</title>
  <style>
    *, *::before, *::after { box-sizing: border-box; }
    body {
      margin: 0;
      padding: 20px;
      font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
      color: #111827;
      background-color: #ffffff;
      line-height: 1.5;
    }
  </style>
</head>
<body>
  ${code}
</body>
</html>`;
  }

  // JavaScript / TypeScript execution demo
  if (lang === "javascript" || lang === "js" || lang === "ts" || lang === "typescript") {
    return `<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>JavaScript Artifact Sandbox</title>
  <style>
    * { box-sizing: border-box; margin: 0; padding: 0; }
    body {
      padding: 20px;
      font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, monospace;
      font-size: 13px;
      color: #1f2937;
      background-color: #ffffff;
    }
    #console-out {
      margin-top: 16px;
      padding: 12px;
      background: #f8fafc;
      border: 1px solid #e2e8f0;
      border-radius: 6px;
      white-space: pre-wrap;
      max-height: 300px;
      overflow-y: auto;
      color: #0f172a;
    }
    .log-entry { margin-bottom: 4px; }
    .log-info { color: #2563eb; }
    .log-error { color: #dc2626; }
    .log-success { color: #16a34a; }
  </style>
</head>
<body>
  <div id="app"></div>
  <div id="root"></div>
  <div id="console-header" style="font-weight: 600; font-size: 11px; text-transform: uppercase; color: #64748b; letter-spacing: 0.05em;">Console Execution Output:</div>
  <div id="console-out"></div>
  <script>
    (function() {
      const out = document.getElementById("console-out");
      function log(msg, type) {
        const line = document.createElement("div");
        line.className = "log-entry " + (type ? "log-" + type : "");
        line.textContent = (typeof msg === "object" ? JSON.stringify(msg, null, 2) : String(msg));
        out.appendChild(line);
      }
      const _origLog = console.log;
      const _origError = console.error;
      const _origWarn = console.warn;
      console.log = function(...args) { log(args.join(" "), "info"); _origLog.apply(console, args); };
      console.error = function(...args) { log(args.join(" "), "error"); _origError.apply(console, args); };
      console.warn = function(...args) { log(args.join(" "), "warn"); _origWarn.apply(console, args); };
      try {
        ${code}
      } catch (err) {
        console.error("Uncaught Sandbox Exception: " + err.message);
      }
    })();
  </script>
</body>
</html>`;
  }

  // CSS Styling Demo
  if (lang === "css") {
    return `<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <style>
    ${code}
  </style>
</head>
<body style="background-color: #ffffff; padding: 24px; font-family: sans-serif;">
  <div class="demo-container">
    <h1>CSS Artifact Preview</h1>
    <p>Styles are applied directly to the document context.</p>
    <button class="btn btn-primary">Button Element</button>
    <div class="card" style="margin-top: 16px; padding: 16px; border: 1px solid #ddd; border-radius: 8px;">
      <h3>Sample Card</h3>
      <p>Interactive preview container.</p>
    </div>
  </div>
</body>
</html>`;
  }

  // Generic Markdown / Plaintext / Script Viewport
  return `<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <style>
    body {
      margin: 0;
      padding: 24px;
      font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
      background-color: #ffffff;
      color: #111827;
      line-height: 1.6;
    }
    pre {
      background: #f8fafc;
      padding: 16px;
      border-radius: 6px;
      border: 1px solid #e2e8f0;
      overflow-x: auto;
      font-family: ui-monospace, Menlo, monospace;
      font-size: 13px;
      white-space: pre-wrap;
    }
  </style>
</head>
<body>
  <div style="font-size: 12px; font-weight: 600; text-transform: uppercase; color: #6b7280; margin-bottom: 8px;">Source Payload:</div>
  <pre>${code.replace(/</g, "&lt;").replace(/>/g, "&gt;")}</pre>
</body>
</html>`;
}

/**
 * ClaudeArtifactCanvas
 *
 * Sliding split-pane modal overlay pinned rightward (w-[50vw] h-screen)
 * to render live code execution previews side-by-side with localized terminal chats.
 *
 * @param {Object} props
 * @param {boolean} props.isOpen - System synchronization state controlling overlay visibility.
 * @param {Function} props.onClose - Dismissal callback triggered via close button, backdrop, or Escape.
 * @param {string} props.codePayload - Active code text string injected into the sandbox.
 * @param {string} props.language - Programming syntax tag (e.g. 'html', 'js', 'react', 'rust').
 */
export const ClaudeArtifactCanvas = React.memo(
  ({ isOpen = false, onClose, codePayload = "", language = "plaintext" }) => {
    // Isolated Tabbed Navigation Header state: 'preview' or 'code'
    const [activeTab, setActiveTab] = useState("preview");
    const [copied, setCopied] = useState(false);
    const [isFullscreen, setIsFullscreen] = useState(false);

    // Default to 'preview' for visual formats (html/svg/js), 'code' for systems code (rust/bash/python)
    useEffect(() => {
      if (isOpen) {
        const lang = (language || "").toLowerCase();
        if (["html", "svg", "htm", "jsx", "tsx", "js", "css"].includes(lang)) {
          setActiveTab("preview");
        } else {
          setActiveTab("code");
        }
      }
    }, [isOpen, language]);

    // Handle Escape key closure
    useEffect(() => {
      const handleKeyDown = (e) => {
        if (e.key === "Escape" && isOpen && typeof onClose === "function") {
          onClose();
        }
      };
      window.addEventListener("keydown", handleKeyDown);
      return () => window.removeEventListener("keydown", handleKeyDown);
    }, [isOpen, onClose]);

    // Zero-allocation clipboard copy handler
    const handleCopy = useCallback(async () => {
      try {
        if (navigator && navigator.clipboard && navigator.clipboard.writeText) {
          await navigator.clipboard.writeText(codePayload);
        } else {
          const ta = document.createElement("textarea");
          ta.value = codePayload;
          ta.style.position = "fixed";
          ta.style.left = "-999999px";
          document.body.appendChild(ta);
          ta.select();
          document.execCommand("copy");
          document.body.removeChild(ta);
        }
        setCopied(true);
        setTimeout(() => setCopied(false), 2000);
      } catch (err) {
        console.warn("[ClaudeArtifactCanvas] Clipboard copy failed:", err);
      }
    }, [codePayload]);

    // Compute sandbox srcDoc memoized against payload changes
    const srcDocContent = useMemo(() => {
      return buildSandboxSrcDoc(codePayload, language);
    }, [codePayload, language]);

    const displayLang = (language || "plaintext").trim().toUpperCase();

    return (
      <>
        {/* Backdrop overlay for focus containment */}
        <div
          className={`claude-artifact-backdrop ${isOpen ? "open" : ""}`}
          onClick={onClose}
          aria-hidden={!isOpen}
        />

        {/* Sliding Artifact Canvas Viewport */}
        <aside
          className={`claude-artifact-container ${isOpen ? "open" : ""} ${
            isFullscreen ? "fullscreen" : ""
          }`}
          role="dialog"
          aria-modal="true"
          aria-label="Split-Pane Artifact Canvas"
        >
          {/* Isolated Tabbed Navigation Header (#16161a) */}
          <header className="claude-artifact-header">
            <div className="claude-artifact-header-left">
              <div className="artifact-pill-badge">
                <svg
                  className="artifact-icon"
                  viewBox="0 0 24 24"
                  width="14"
                  height="14"
                  fill="none"
                  stroke="currentColor"
                  strokeWidth="2.2"
                  strokeLinecap="round"
                  strokeLinejoin="round"
                >
                  <rect x="3" y="3" width="18" height="18" rx="2" ry="2" />
                  <line x1="9" y1="3" x2="9" y2="21" />
                </svg>
                <span className="artifact-title">Artifact Canvas</span>
              </div>
              <span className="artifact-lang-tag">{displayLang}</span>

              {/* Lock-Free Tab Switcher Anchors */}
              <div className="claude-artifact-tabs">
                <button
                  type="button"
                  className={`artifact-tab-btn ${activeTab === "code" ? "active" : ""}`}
                  onClick={() => setActiveTab("code")}
                >
                  <svg
                    viewBox="0 0 24 24"
                    width="13"
                    height="13"
                    fill="none"
                    stroke="currentColor"
                    strokeWidth="2"
                    strokeLinecap="round"
                    strokeLinejoin="round"
                  >
                    <polyline points="16 18 22 12 16 6" />
                    <polyline points="8 6 2 12 8 18" />
                  </svg>
                  <span>Code</span>
                </button>
                <button
                  type="button"
                  className={`artifact-tab-btn ${activeTab === "preview" ? "active" : ""}`}
                  onClick={() => setActiveTab("preview")}
                >
                  <svg
                    viewBox="0 0 24 24"
                    width="13"
                    height="13"
                    fill="none"
                    stroke="currentColor"
                    strokeWidth="2"
                    strokeLinecap="round"
                    strokeLinejoin="round"
                  >
                    <polygon points="5 3 19 12 5 21 5 3" />
                  </svg>
                  <span>Preview</span>
                </button>
              </div>
            </div>

            {/* Header Right Utility Controls */}
            <div className="claude-artifact-header-right">
              {/* Copy Code Button */}
              <button
                type="button"
                onClick={handleCopy}
                className={`artifact-ctrl-btn ${copied ? "copied" : ""}`}
                title="Copy code to clipboard"
                aria-label="Copy code"
              >
                {copied ? (
                  <>
                    <svg
                      viewBox="0 0 24 24"
                      width="12"
                      height="12"
                      fill="none"
                      stroke="#10b981"
                      strokeWidth="2.4"
                      strokeLinecap="round"
                      strokeLinejoin="round"
                    >
                      <polyline points="20 6 9 17 4 12" />
                    </svg>
                    <span style={{ color: "#10b981", fontWeight: 600 }}>Copied!</span>
                  </>
                ) : (
                  <>
                    <svg
                      viewBox="0 0 24 24"
                      width="12"
                      height="12"
                      fill="none"
                      stroke="currentColor"
                      strokeWidth="2"
                      strokeLinecap="round"
                      strokeLinejoin="round"
                    >
                      <rect x="9" y="9" width="13" height="13" rx="2" ry="2" />
                      <path d="M5 15H4a2 2 0 0 1-2-2V4a2 2 0 0 1 2-2h9a2 2 0 0 1 2 2v1" />
                    </svg>
                    <span>Copy</span>
                  </>
                )}
              </button>

              {/* Expand / Fullscreen Toggle */}
              <button
                type="button"
                onClick={() => setIsFullscreen((prev) => !prev)}
                className="artifact-ctrl-btn icon-only"
                title={isFullscreen ? "Restore standard split width" : "Expand to fullscreen"}
                aria-label="Toggle Fullscreen"
              >
                <svg
                  viewBox="0 0 24 24"
                  width="13"
                  height="13"
                  fill="none"
                  stroke="currentColor"
                  strokeWidth="2"
                  strokeLinecap="round"
                  strokeLinejoin="round"
                >
                  {isFullscreen ? (
                    <>
                      <polyline points="4 14 10 14 10 20" />
                      <polyline points="20 10 14 10 14 4" />
                      <line x1="14" y1="10" x2="21" y2="3" />
                      <line x1="3" y1="21" x2="10" y2="14" />
                    </>
                  ) : (
                    <>
                      <polyline points="15 3 21 3 21 9" />
                      <polyline points="9 21 3 21 3 15" />
                      <line x1="21" y1="3" x2="14" y2="10" />
                      <line x1="3" y1="21" x2="10" y2="14" />
                    </>
                  )}
                </svg>
              </button>

              {/* Close Button */}
              <button
                type="button"
                onClick={onClose}
                className="artifact-close-btn"
                title="Close Artifact Canvas (Esc)"
                aria-label="Close Artifact Canvas"
              >
                <svg
                  viewBox="0 0 24 24"
                  width="15"
                  height="15"
                  fill="none"
                  stroke="currentColor"
                  strokeWidth="2.2"
                  strokeLinecap="round"
                  strokeLinejoin="round"
                >
                  <line x1="18" y1="6" x2="6" y2="18" />
                  <line x1="6" y1="6" x2="18" y2="18" />
                </svg>
              </button>
            </div>
          </header>

          {/* Primary Viewport Body */}
          <div className="claude-artifact-body">
            {/* "Code" Tab Viewport */}
            {activeTab === "code" && (
              <div className="claude-artifact-code-panel">
                <pre className="claude-artifact-pre">
                  <code className="claude-artifact-code whitespace-pre-wrap">
                    {codePayload || "// No code payload available"}
                  </code>
                </pre>
              </div>
            )}

            {/* "Preview" Tab Viewport (Strict White Background Context Box) */}
            {activeTab === "preview" && (
              <div className="claude-artifact-preview-panel bg-white">
                <iframe
                  title="AirDrop-LLM Artifact Sandbox"
                  sandbox="allow-scripts"
                  srcDoc={srcDocContent}
                  className="claude-artifact-iframe"
                  loading="lazy"
                />
              </div>
            )}
          </div>
        </aside>
      </>
    );
  }
);

ClaudeArtifactCanvas.displayName = "ClaudeArtifactCanvas";

export default ClaudeArtifactCanvas;
