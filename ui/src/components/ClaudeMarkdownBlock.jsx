import React, { useState, useEffect, useRef, useMemo, useCallback } from "react";
import { ClaudeArtifactCanvas } from "./ClaudeArtifactCanvas";

/**
 * High-speed HTML entity escaper ensuring zero-allocation safety before span insertion.
 */
function escapeCodeHtml(str) {
  return str
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;")
    .replace(/'/g, "&#039;");
}

/**
 * Pre-compiled language regular expression cache for single-pass O(N) tokenization.
 * Eliminates re-compilation overhead during real-time 60fps streaming token arrivals.
 */
const LEXER_CACHE = new Map();

function getLanguageLexer(lang) {
  const normalized = (lang || "plaintext").toLowerCase().trim();
  if (LEXER_CACHE.has(normalized)) {
    return LEXER_CACHE.get(normalized);
  }

  // HTML / XML Syntax Rules
  if (normalized === "html" || normalized === "xml" || normalized === "htm" || normalized === "svg") {
    const lexer = /(?<comm><!--.*?-->)|(?<str>"[^"\\]*(?:\\.[^"\\]*)*"|'[^'\\]*(?:\\.[^'\\]*)*')|(?<kw><\/?[\w-]+|\/?>)|(?<type>\b[\w:-]+(?==))/g;
    LEXER_CACHE.set(normalized, lexer);
    return lexer;
  }

  // CSS / SCSS Syntax Rules
  if (normalized === "css" || normalized === "scss") {
    const lexer = /(?<comm>\/\*.*?\*\/)|(?<str>"[^"\\]*(?:\\.[^"\\]*)*"|'[^'\\]*(?:\\.[^'\\]*)*')|(?<type>\b[\w-]+(?=\s*:))|(?<num>#[0-9a-fA-F]{3,8}|\b\d+(?:\.\d+)?(?:px|rem|em|%|vh|vw|s|ms|deg|fr)?\b)|(?<kw>[.#][\w-]+)/g;
    LEXER_CACHE.set(normalized, lexer);
    return lexer;
  }

  // Core Systems & Scripting Languages (Python, Rust, JS/TS, Bash, C/C++)
  let comm = "\\/\\/.*$|\\/\\*.*?\\*\\/";
  let str = "\"(\\\\.|[^\"])*\"|'(\\\\.|[^'])*'|`(\\\\.|[^`])*`";
  let kw = "\\b(?:function|return|const|let|var|if|else|for|while|do|switch|case|break|continue|default|class|extends|import|export|from|as|new|this|typeof|instanceof|try|catch|finally|throw|async|await|yield)\\b";
  let types = "\\b(?:true|false|null|undefined|NaN|Infinity|boolean|string|number|any|void|never|unknown|Promise|Array|Object|Map|Set|Symbol|RegExp|Error)\\b";
  let num = "\\b\\d+(?:\\.\\d+)?\\b";
  let fn = "\\b[a-zA-Z_]\\w*(?=\\()";

  if (normalized === "python" || normalized === "py") {
    comm = "#.*$";
    str = '"""[\\s\\S]*?"""|\'\'\'[\\s\\S]*?\'\'\'|"(\\\\.|[^"])*"|\'(\\\\.|[^\'])*\'';
    kw = "\\b(?:def|class|return|if|elif|else|for|while|break|continue|pass|import|from|as|try|except|finally|raise|with|async|await|lambda|yield|global|nonlocal|assert|in|is|not|and|or)\\b";
    types = "\\b(?:True|False|None|self|cls|int|float|str|bool|list|dict|set|tuple|bytes|range|enumerate|zip|map|filter|print|len|open|super|type|object)\\b";
  } else if (normalized === "rust" || normalized === "rs") {
    kw = "\\b(?:fn|let|mut|if|else|for|while|loop|match|return|break|continue|struct|enum|impl|trait|type|pub|mod|use|as|crate|super|self|Self|unsafe|async|await|where|move|ref|dyn)\\b";
    types = "\\b(?:bool|char|str|String|usize|u8|u16|u32|u64|u128|i8|i16|i32|i64|i128|f32|f64|Option|Result|Some|None|Ok|Err|Vec|Box|Arc|Rc|Mutex|Cell|RefCell)\\b";
    num = "\\b\\d+(?:\\.\\d+)?(?:_?\\w*)?\\b";
  } else if (normalized === "bash" || normalized === "sh" || normalized === "shell" || normalized === "zsh") {
    comm = "#.*$";
    str = '"(\\\\.|[^"])*"|\'(\\\\.|[^\'])*\'';
    kw = "\\b(?:if|then|else|elif|fi|case|esac|for|while|until|do|done|in|function|select|echo|exit|export|source|alias|cd|ls|cat|grep|mkdir|rm|chmod|chown)\\b";
    types = "(?:\\$[A-Za-z_0-9]+|\\$\\{[^}]+\\}|--?[a-zA-Z0-9_-]+)";
  } else if (normalized === "c" || normalized === "cpp" || normalized === "h" || normalized === "hpp") {
    kw = "\\b(?:auto|break|case|class|const|continue|default|do|else|enum|extern|for|goto|if|inline|namespace|new|operator|private|protected|public|register|return|sizeof|static|struct|switch|template|this|throw|try|typedef|typeid|typename|union|using|virtual|volatile|while)\\b";
    types = "\\b(?:int|long|short|char|void|float|double|bool|size_t|uint8_t|uint16_t|uint32_t|uint64_t|int8_t|int16_t|int32_t|int64_t|std::string|std::vector|std::unique_ptr|std::shared_ptr)\\b";
  } else if (normalized === "json") {
    comm = "\\/\\/.*$";
    str = '"(\\\\.|[^"])*"';
    kw = '\\b(?:true|false|null)\\b';
    types = '"[^"\\\\]*"(?=\\s*:)';
    num = "\\b\\d+(?:\\.\\d+)?\\b";
  }

  const lexer = new RegExp(
    `(?<comm>${comm})|(?<str>${str})|(?<kw>${kw})|(?<type>${types})|(?<num>${num})|(?<fn>${fn})`,
    "g"
  );
  LEXER_CACHE.set(normalized, lexer);
  return lexer;
}

/**
 * Tokenizes a single code line in a single native regex pass, mapping tokens directly
 * into high-contrast Claude syntax styles without nested regex collisions.
 */
function colorizeCodeLine(line, lexer) {
  if (!line || !line.trim()) {
    return escapeCodeHtml(line);
  }

  lexer.lastIndex = 0;
  let lastIndex = 0;
  let out = "";
  let m;

  while ((m = lexer.exec(line)) !== null) {
    if (m.index > lastIndex) {
      out += escapeCodeHtml(line.slice(lastIndex, m.index));
    }

    const { comm, str, kw, type, num, fn } = m.groups;

    if (comm) {
      // Code Comments -> slate gray italic
      out += `<span class="claude-code-comment text-gray-500 italic">${escapeCodeHtml(comm)}</span>`;
    } else if (str) {
      // String Literals -> muted forest green / emerald
      out += `<span class="claude-code-str text-[#50fa7b] text-emerald-400">${escapeCodeHtml(str)}</span>`;
    } else if (kw) {
      // Control Keywords -> vibrant violet / amber
      out += `<span class="claude-code-kw text-[#ff79c6] text-amber-400 font-semibold">${escapeCodeHtml(kw)}</span>`;
    } else if (type) {
      // Built-in Types & Identifiers -> clean cyan
      out += `<span class="claude-code-type text-cyan-400">${escapeCodeHtml(type)}</span>`;
    } else if (num) {
      // Integers & Numbers -> bright orange
      out += `<span class="claude-code-num text-orange-400">${escapeCodeHtml(num)}</span>`;
    } else if (fn) {
      // Function Calls -> sky blue
      out += `<span class="claude-code-fn text-sky-300">${escapeCodeHtml(fn)}</span>`;
    } else {
      out += escapeCodeHtml(m[0]);
    }

    lastIndex = lexer.lastIndex;
  }

  if (lastIndex < line.length) {
    out += escapeCodeHtml(line.slice(lastIndex));
  }

  return out;
}

/**
 * Lightweight Native Regex Syntax Highlighter Pipeline
 * Processes raw code blocks line-by-line using high-speed string and regex pointer arrays.
 * Delivers vibrant Claude-style syntax palettes with zero heavy external runtime dependencies.
 *
 * @param {string} rawCode - Source code string payload.
 * @param {string} language - Target language tag (e.g., 'python', 'rust', 'javascript', 'html', 'css').
 * @returns {string} Fully highlighted HTML string ready for zero-latency DOM painting.
 */
export function colorizeCodeSyntaxNatively(rawCode, language = "plaintext") {
  if (!rawCode) return "";
  const code = typeof rawCode === "string" ? rawCode : String(rawCode || "");
  const lang = (language || "plaintext").toLowerCase().trim();

  if (lang === "plaintext" || lang === "text" || !lang) {
    return escapeCodeHtml(code);
  }

  const lexer = getLanguageLexer(lang);
  const lines = code.split("\n");
  const highlightedLines = lines.map((line) => colorizeCodeLine(line, lexer));
  return highlightedLines.join("\n");
}

/**
 * High-performance, Claude-style syntax pane control button.
 * Manages asynchronous clipboard copy operations with a strict 2000ms self-resetting state.
 */
const CopyCodeButton = React.memo(({ code }) => {
  const [copied, setCopied] = useState(false);
  const resetTimerRef = useRef(null);

  // Clean up timer on unmount to prevent memory leaks
  useEffect(() => {
    return () => {
      if (resetTimerRef.current) {
        clearTimeout(resetTimerRef.current);
      }
    };
  }, []);

  const handleCopy = useCallback(
    async (e) => {
      e.stopPropagation();
      try {
        if (navigator && navigator.clipboard && navigator.clipboard.writeText) {
          await navigator.clipboard.writeText(code);
        } else {
          // Zero-allocation fallback for non-secure / legacy browser contexts
          const textArea = document.createElement("textarea");
          textArea.value = code;
          textArea.style.position = "fixed";
          textArea.style.left = "-999999px";
          textArea.style.top = "-999999px";
          textArea.setAttribute("readonly", "");
          document.body.appendChild(textArea);
          textArea.focus();
          textArea.select();
          document.execCommand("copy");
          document.body.removeChild(textArea);
        }

        setCopied(true);

        if (resetTimerRef.current) {
          clearTimeout(resetTimerRef.current);
        }

        // Strict 2000ms delay before resetting clipboard state
        resetTimerRef.current = setTimeout(() => {
          setCopied(false);
        }, 2000);
      } catch (err) {
        console.warn("[ClaudeMarkdownBlock] Clipboard write failed:", err);
      }
    },
    [code]
  );

  return (
    <button
      type="button"
      onClick={handleCopy}
      className={`claude-code-copy-btn ${copied ? "copied" : ""}`}
      aria-label={copied ? "Code copied to clipboard" : "Copy code to clipboard"}
      title={copied ? "Copied to clipboard" : "Copy code"}
    >
      {copied ? (
        <>
          <svg
            className="copy-icon check-icon"
            viewBox="0 0 24 24"
            width="13"
            height="13"
            fill="none"
            stroke="#10b981"
            strokeWidth="2.4"
            strokeLinecap="round"
            strokeLinejoin="round"
          >
            <polyline points="20 6 9 17 4 12" />
          </svg>
          <span className="copy-label success-text">Copied!</span>
        </>
      ) : (
        <>
          <svg
            className="copy-icon"
            viewBox="0 0 24 24"
            width="13"
            height="13"
            fill="none"
            stroke="currentColor"
            strokeWidth="2"
            strokeLinecap="round"
            strokeLinejoin="round"
          >
            <rect x="9" y="9" width="13" height="13" rx="2" ry="2" />
            <path d="M5 15H4a2 2 0 0 1-2-2V4a2 2 0 0 1 2-2h9a2 2 0 0 1 2 2v1" />
          </svg>
          <span className="copy-label">Copy code</span>
        </>
      )}
    </button>
  );
});

CopyCodeButton.displayName = "CopyCodeButton";

/**
 * Isolated Claude-style programming syntax visual frame.
 * Matte-dark backdrop (#0f0f11), low-opacity dark borders (#333), rounded-lg geometry,
 * high-density control top bar (#1e1e1e), and native regex syntax highlighted container.
 */
const ClaudeCodeBlock = React.memo(
  ({ language, code, isStreaming = false, onOpenArtifact }) => {
    const displayLang = (language || "plaintext").trim().toUpperCase();

    // High-speed, lock-free syntax colorization passes without heavy runtime AST overhead
    const highlightedHtml = useMemo(() => {
      return colorizeCodeSyntaxNatively(code, language);
    }, [code, language]);

    return (
      <div className="claude-code-container" data-language={language}>
        {/* Structural Header Strip (#1e1e1e) */}
        <div className="claude-code-header">
          <span className="claude-code-lang" title={`Language: ${displayLang}`}>
            {displayLang}
          </span>
          <div className="claude-code-actions">
            {/* Connector Shortcut Button: Open Artifact ↗ */}
            <button
              type="button"
              onClick={() => onOpenArtifact && onOpenArtifact(code, language)}
              className="claude-artifact-open-btn"
              title="Open code in Split-Pane Artifact Canvas"
              aria-label="Open Artifact"
            >
              <svg
                className="artifact-link-icon"
                viewBox="0 0 24 24"
                width="12"
                height="12"
                fill="none"
                stroke="currentColor"
                strokeWidth="2.2"
                strokeLinecap="round"
                strokeLinejoin="round"
              >
                <path d="M18 13v6a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2V8a2 2 0 0 1 2-2h6" />
                <polyline points="15 3 21 3 21 9" />
                <line x1="10" y1="14" x2="21" y2="3" />
              </svg>
              <span>Open Artifact ↗</span>
            </button>

            {/* Copy Code Clipboard Button */}
            <CopyCodeButton code={code} />
          </div>
        </div>

        {/* Pre-formatted Zero-Allocation Code Canvas with Native Syntax Colors */}
        <div className="claude-code-body">
          <pre className="claude-code-pre">
            <code
              className="claude-code-content whitespace-pre"
              dangerouslySetInnerHTML={{ __html: highlightedHtml }}
            />
            {isStreaming && <span className="claude-streaming-cursor" aria-hidden="true" />}
          </pre>
        </div>
      </div>
    );
  }
);

ClaudeCodeBlock.displayName = "ClaudeCodeBlock";

/**
 * Precision scanner targeting inline text attributes.
 * Strips raw double asterisks ('**') completely from the view, wrapping the internal
 * values inside a strong tag styled with a high-contrast cyan highlight ('font-bold text-cyan-300').
 * Also cleanly parses inline code spans (`code`).
 *
 * @param {string} text - Raw string slice to tokenize.
 * @param {string} lineKey - Unique key prefix for React reconciliation.
 * @returns {React.ReactNode}
 */
function renderInlineTypography(text, lineKey) {
  if (!text) return null;

  // Precise scanner for inline bold (/\*\*([\s\(\S\)]*?)\*\*/g) and inline code (`...`)
  const inlineRegex = /(`[^`]+`|\*\*[\s\(\S\)]*?\*\*)/g;
  const parts = [];
  let lastIdx = 0;
  let match;

  while ((match = inlineRegex.exec(text)) !== null) {
    if (match.index > lastIdx) {
      parts.push({
        type: "text",
        value: text.slice(lastIdx, match.index),
      });
    }

    const token = match[0];
    if (token.startsWith("`") && token.endsWith("`")) {
      parts.push({
        type: "code",
        value: token.slice(1, -1),
      });
    } else if (token.startsWith("**") && token.endsWith("**")) {
      // Strip raw double asterisks completely from the layout
      parts.push({
        type: "bold",
        value: token.slice(2, -2),
      });
    }

    lastIdx = inlineRegex.lastIndex;
  }

  if (lastIdx < text.length) {
    parts.push({
      type: "text",
      value: text.slice(lastIdx),
    });
  }

  return parts.map((part, pIdx) => {
    if (part.type === "bold") {
      return (
        <strong
          key={`${lineKey}-b-${pIdx}`}
          className="font-bold text-cyan-300 claude-bold-highlight"
        >
          {part.value}
        </strong>
      );
    }
    if (part.type === "code") {
      return (
        <code key={`${lineKey}-c-${pIdx}`} className="claude-inline-code">
          {part.value}
        </code>
      );
    }
    return <React.Fragment key={`${lineKey}-t-${pIdx}`}>{part.value}</React.Fragment>;
  });
}

/**
 * Zero-Copy Line-by-Line Markdown Interceptor
 * Parses narrative sections line-by-line using microsecond pointer splits (.split("\n")).
 * Eradicates raw markdown symbols (###, ##, #, -, *, **) from the final DOM tree,
 * converting them into clean, high-contrast, typography-focused developer layouts.
 *
 * @param {string} content - Raw narrative text block.
 * @param {string} blockKey - Unique key prefix.
 * @returns {React.ReactNode}
 */
export function renderCleanClaudeTypography(content, blockKey = "narrative") {
  if (!content) return null;

  // Microsecond pointer split along newline delimiters
  const lines = content.split("\n");

  return (
    <div className="claude-narrative-block font-sans text-xs text-gray-300 leading-6 space-y-1">
      {lines.map((rawLine, lineIdx) => {
        const lineKey = `${blockKey}-L${lineIdx}`;
        const trimmed = rawLine.trim();

        // Empty line handling
        if (!trimmed) {
          if (lineIdx < lines.length - 1) {
            return (
              <div
                key={lineKey}
                className="h-2 claude-narrative-empty-line"
                aria-hidden="true"
              />
            );
          }
          return null;
        }

        // ---------------------------------------------------------------------
        // 1. Structural Headings: Intercept and strip raw hash tokens completely
        // ---------------------------------------------------------------------

        // '### ' -> h4 ('text-sm font-bold text-cyan-400 mt-4 mb-2')
        if (rawLine.startsWith("### ")) {
          const strippedHeading = rawLine.slice(4).trim();
          return (
            <h4
              key={lineKey}
              className="text-sm font-bold text-cyan-400 mt-4 mb-2 claude-heading-h4 tracking-tight"
            >
              {renderInlineTypography(strippedHeading, lineKey)}
            </h4>
          );
        }

        // '## ' -> h3 ('text-base font-extrabold text-white mt-5 mb-2')
        if (rawLine.startsWith("## ")) {
          const strippedHeading = rawLine.slice(3).trim();
          return (
            <h3
              key={lineKey}
              className="text-base font-extrabold text-white mt-5 mb-2 claude-heading-h3 tracking-tight"
            >
              {renderInlineTypography(strippedHeading, lineKey)}
            </h3>
          );
        }

        // '# ' -> uppercase h2 header segment
        if (rawLine.startsWith("# ")) {
          const strippedHeading = rawLine.slice(2).trim();
          return (
            <h2
              key={lineKey}
              className="text-lg font-extrabold uppercase text-white mt-6 mb-2 tracking-wide claude-heading-h2"
            >
              {renderInlineTypography(strippedHeading, lineKey)}
            </h2>
          );
        }

        // ---------------------------------------------------------------------
        // 2. Structural List Bullets: Scan line starts for dash ('- ') or asterisk ('* ')
        //    Strip raw characters and map to isolated grid with neon cyan bullet ('•')
        // ---------------------------------------------------------------------
        if (rawLine.startsWith("- ") || rawLine.startsWith("* ")) {
          const strippedBullet = rawLine.slice(2).trim();
          return (
            <div
              key={lineKey}
              className="claude-list-item grid grid-cols-[16px_1fr] items-start gap-1 my-0.5 text-xs text-gray-300 leading-6"
            >
              <span
                className="claude-bullet text-cyan-400 font-bold select-none text-center"
                aria-hidden="true"
              >
                •
              </span>
              <div className="claude-list-content">
                {renderInlineTypography(strippedBullet, lineKey)}
              </div>
            </div>
          );
        }

        // Optional numeric lists ('1. ', '2. ') with high-contrast cyan index
        const numMatch = rawLine.match(/^(\d+)\.\s+(.*)$/);
        if (numMatch) {
          const indexNum = numMatch[1];
          const strippedNum = numMatch[2].trim();
          return (
            <div
              key={lineKey}
              className="claude-list-item grid grid-cols-[18px_1fr] items-start gap-1 my-0.5 text-xs text-gray-300 leading-6"
            >
              <span
                className="claude-bullet-num text-cyan-400 font-mono text-[11px] font-semibold select-none"
                aria-hidden="true"
              >
                {indexNum}.
              </span>
              <div className="claude-list-content">
                {renderInlineTypography(strippedNum, lineKey)}
              </div>
            </div>
          );
        }

        // ---------------------------------------------------------------------
        // 3. Dense Telemetry Paragraph: Compact text-xs text-gray-300 leading-6
        // ---------------------------------------------------------------------
        return (
          <p
            key={lineKey}
            className="claude-narrative-paragraph text-xs text-gray-300 leading-6 m-0"
          >
            {renderInlineTypography(rawLine, lineKey)}
          </p>
        );
      })}
    </div>
  );
}

/**
 * Functional component wrapper for memoized narrative typography rendering.
 */
const NarrativeTextBlock = React.memo(({ content, segmentKey }) => {
  return renderCleanClaudeTypography(content, segmentKey);
});

NarrativeTextBlock.displayName = "NarrativeTextBlock";

/**
 * ClaudeMarkdownBlock
 *
 * Core streaming markdown parser component. Implements a non-blocking regular expression
 * boundary scanner inside the render pipeline to segment standard narrative text fields
 * from structured markdown code blocks.
 *
 * Automatically feeds code blocks into colorizeCodeSyntaxNatively for fluid 60fps
 * streaming syntax highlighting without heavy external dependencies.
 *
 * @param {Object} props
 * @param {string} props.rawResponseText - Active incoming token stream text string.
 */
export const ClaudeMarkdownBlock = React.memo(({ rawResponseText = "" }) => {
  // Strict synchronization hooks for Split-Pane Artifact Canvas state
  const [isArtifactActive, setIsArtifactActive] = useState(false);
  const [targetCode, setTargetCode] = useState("");
  const [targetLanguage, setTargetLanguage] = useState("");
  const [activeSegmentKey, setActiveSegmentKey] = useState(null);

  // Non-blocking boundary scanner execution memoized across token passes
  const parsedSegments = useMemo(() => {
    const text = typeof rawResponseText === "string" ? rawResponseText : String(rawResponseText || "");
    if (!text) return [];

    const segments = [];
    // Regular expression matching closed code blocks: /```(\w*)\n([\s\S]*?)```/g
    const codeBlockRegex = /```(\w*)\n([\s\S]*?)```/g;
    let lastIndex = 0;
    let match;

    while ((match = codeBlockRegex.exec(text)) !== null) {
      // Narrative segment before the matched code block
      if (match.index > lastIndex) {
        const narrative = text.slice(lastIndex, match.index);
        if (narrative) {
          segments.push({
            type: "narrative",
            content: narrative,
            key: `narrative-${lastIndex}`,
          });
        }
      }

      const language = match[1] || "";
      const codeContent = match[2];

      segments.push({
        type: "code",
        language: language,
        code: codeContent,
        isStreaming: false,
        key: `code-${match.index}`,
      });

      lastIndex = codeBlockRegex.lastIndex;
    }

    // Trailing buffer handling: check for an active unclosed code block during active streaming
    if (lastIndex < text.length) {
      const remaining = text.slice(lastIndex);
      // Scanner for unclosed trailing block: ```lang\ncode...
      const unclosedMatch = remaining.match(/```(\w*)\n([\s\S]*)$/);

      if (unclosedMatch) {
        const narrativeBefore = remaining.slice(0, unclosedMatch.index);
        if (narrativeBefore) {
          segments.push({
            type: "narrative",
            content: narrativeBefore,
            key: `narrative-${lastIndex}`,
          });
        }

        segments.push({
          type: "code",
          language: unclosedMatch[1] || "",
          code: unclosedMatch[2],
          isStreaming: true,
          key: `code-streaming-${lastIndex + (unclosedMatch.index || 0)}`,
        });
      } else {
        // Plain trailing narrative
        segments.push({
          type: "narrative",
          content: remaining,
          key: `narrative-tail-${lastIndex}`,
        });
      }
    }

    return segments;
  }, [rawResponseText]);

  // Keep open artifact canvas in live sync if the user opened an actively streaming code block
  useEffect(() => {
    if (isArtifactActive && activeSegmentKey) {
      const match = parsedSegments.find((seg) => seg.key === activeSegmentKey);
      if (match && match.type === "code" && match.code !== targetCode) {
        setTargetCode(match.code);
      }
    }
  }, [parsedSegments, isArtifactActive, activeSegmentKey, targetCode]);

  // Open Artifact shortcut handler: binds state instantly without interrupting active stream
  const handleOpenArtifact = useCallback((code, language, segmentKey) => {
    setTargetCode(code);
    setTargetLanguage(language);
    setActiveSegmentKey(segmentKey);
    setIsArtifactActive(true);
  }, []);

  const handleCloseArtifact = useCallback(() => {
    setIsArtifactActive(false);
    setActiveSegmentKey(null);
  }, []);

  if (!parsedSegments || parsedSegments.length === 0) {
    return null;
  }

  return (
    <div className="claude-markdown-root font-sans">
      {parsedSegments.map((segment) => {
        if (segment.type === "code") {
          return (
            <ClaudeCodeBlock
              key={segment.key}
              language={segment.language}
              code={segment.code}
              isStreaming={segment.isStreaming}
              onOpenArtifact={(code, lang) => handleOpenArtifact(code, lang, segment.key)}
            />
          );
        }
        return (
          <NarrativeTextBlock
            key={segment.key}
            segmentKey={segment.key}
            content={segment.content}
          />
        );
      })}

      {/* Split-Pane Sliding Artifact Canvas Modal */}
      <ClaudeArtifactCanvas
        isOpen={isArtifactActive}
        onClose={handleCloseArtifact}
        codePayload={targetCode}
        language={targetLanguage}
      />
    </div>
  );
});

ClaudeMarkdownBlock.displayName = "ClaudeMarkdownBlock";

export default ClaudeMarkdownBlock;
