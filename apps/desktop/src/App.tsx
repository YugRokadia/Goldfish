import { useEffect, useRef, useState } from "react";
import { getCurrentWindow } from "@tauri-apps/api/window";
import { invoke } from "@tauri-apps/api/core";
import { openUrl } from "@tauri-apps/plugin-opener";

type Theme = "dark" | "light";
type SettingsPage = "main" | "appearance" | "accounts" | "privacy";

type BackendContext = {
  memory_id: number;
  title: string;
  path: string;
  source_type: string;
  score: number;
  reasons: string[];
};

type BackendResult = {
  id: number;
  source_type: string;
  content_type: string;
  title: string;
  path: string;
  content: string;
  modified_at: number | null;
  accessed_at: number | null;
  is_deleted: number;
  deleted_at: number | null;
  metadata_json:
    | string
    | {
        source?: string;
        provider?: string;
        message_id?: string;
        thread_id?: string;
        sender?: string;
        recipient?: string;
        date?: string;
        url?: string;
      }
    | null;
  rank: number;
  lexical_rank: number | null;
  semantic_rank: number | null;
  context: BackendContext[] | null;
};

const RESULTS_VIEW_HEIGHT = 270;
const CLOSE_ANIMATION_DURATION = 120;

function getFileType(result: BackendResult): string {
  const contentType = result.content_type?.trim();

  if (contentType) {
    return contentType.toUpperCase();
  }

  const extension = result.title.split(".").pop();

  if (!extension || extension === result.title) {
    return "FILE";
  }

  return extension.toUpperCase();
}

function getParentPath(path: string): string {
  const normalized = path.replace(/\\/g, "/");
  const parts = normalized.split("/");

  if (parts.length <= 1) {
    return path;
  }

  parts.pop();

  return parts.join("/");
}

function getFolderName(path: string): string {
  const normalized = path.replace(/\\/g, "/");
  const parts = normalized.split("/").filter(Boolean);

  if (parts.length < 2) {
    return "";
  }

  return parts[parts.length - 2];
}

function formatRelativeTime(timestamp: number | null): string {
  if (!timestamp) {
    return "";
  }

  const timestampMs =
    timestamp < 10_000_000_000
      ? timestamp * 1000
      : timestamp;

  const diff = Date.now() - timestampMs;

  if (diff < 0) {
    return "JUST NOW";
  }

  const minute = 60 * 1000;
  const hour = 60 * minute;
  const day = 24 * hour;

  if (diff < minute) {
    return "JUST NOW";
  }

  if (diff < hour) {
    return `${Math.floor(diff / minute)}M AGO`;
  }

  if (diff < day) {
    return `${Math.floor(diff / hour)}H AGO`;
  }

  if (diff < 7 * day) {
    return `${Math.floor(diff / day)}D AGO`;
  }

  return new Date(timestampMs)
    .toLocaleDateString(undefined, {
      day: "numeric",
      month: "short",
      year: "numeric",
    })
    .toUpperCase();
}

function isLikelyGmailQuery(value: string): boolean {
  const normalized = value.trim().toLowerCase();

  if (!normalized) {
    return false;
  }

  // Explicit Gmail search operators.
  if (/\b(?:from|to|cc|bcc|subject|after|before):/i.test(normalized)) {
    return true;
  }

  // Clear natural-language email intent.
  return /\b(?:email|emails|e-mail|e-mails|mail|mails|gmail)\b/i.test(
    normalized,
  );
}

function App() {
  const [query, setQuery] = useState("");
  const [results, setResults] = useState<BackendResult[]>([]);
  const [selectedIndex, setSelectedIndex] = useState(0);
  const [loading, setLoading] = useState(false);

  /*
   * Theme
   *
   * Dark is the default.
   * The selected theme is persisted locally so RecallX
   * remembers the choice after restarting.
   */
  const [theme, setTheme] = useState<Theme>(() => {
    const savedTheme = localStorage.getItem("recallx-theme");

    if (savedTheme === "light") {
      return "light";
    }

    return "dark";
  });

  const [settingsOpen, setSettingsOpen] = useState(false);
  const [settingsPage, setSettingsPage] =
    useState<SettingsPage>("main");

  const [gmailConnected, setGmailConnected] =
    useState(false);

  const [gmailLoading, setGmailLoading] =
    useState(false);

  const [gmailError, setGmailError] =
    useState("");

  // Controls the short closing animation before the
  // native Tauri window is hidden.
  const [isClosing, setIsClosing] = useState(false);

  const inputRef = useRef<HTMLInputElement>(null);
  const resultsScrollRef = useRef<HTMLDivElement>(null);
  const settingsRef = useRef<HTMLDivElement>(null);

  // Each query gets a unique sequence number.
  // Only the newest request can update the UI.
  const searchSequence = useRef(0);

  const hasQuery = query.trim().length > 0;

  // -------------------------------------------------------------------------
  // Gmail account integration
  // -------------------------------------------------------------------------

  const loadGmailStatus = async () => {
    try {
      const response = await fetch(
        "http://127.0.0.1:8000/gmail/status",
        { cache: "no-store" },
      );

      if (!response.ok) {
        throw new Error(
          `Gmail status failed: ${response.status}`,
        );
      }

      const data = await response.json();
      setGmailConnected(Boolean(data.connected));
    } catch (error) {
      console.error(
        "Failed to get Gmail status:",
        error,
      );
    }
  };

  const connectGmail = async () => {
    if (gmailLoading) return;

    setGmailLoading(true);
    setGmailError("");

    try {
      const response = await fetch(
        "http://127.0.0.1:8000/gmail/connect",
        {
          method: "POST",
          cache: "no-store",
        },
      );

      if (!response.ok) {
        const detail = await response.text();

        throw new Error(
          detail ||
            `Gmail connection failed: ${response.status}`,
        );
      }

      const data = await response.json();

      setGmailConnected(
        Boolean(data.connected ?? true),
      );
    } catch (error) {
      console.error(
        "Gmail connection failed:",
        error,
      );

      setGmailError(
        error instanceof Error
          ? error.message
          : "Could not connect Gmail.",
      );
    } finally {
      setGmailLoading(false);
    }
  };

  const disconnectGmail = async () => {
    if (gmailLoading) return;

    setGmailLoading(true);
    setGmailError("");

    try {
      const response = await fetch(
        "http://127.0.0.1:8000/gmail/disconnect",
        {
          method: "POST",
          cache: "no-store",
        },
      );

      if (!response.ok) {
        const detail = await response.text();

        throw new Error(
          detail ||
            `Gmail disconnect failed: ${response.status}`,
        );
      }

      setGmailConnected(false);
    } catch (error) {
      console.error(
        "Gmail disconnect failed:",
        error,
      );

      setGmailError(
        error instanceof Error
          ? error.message
          : "Could not disconnect Gmail.",
      );
    } finally {
      setGmailLoading(false);
    }
  };

  useEffect(() => {
    if (
      settingsOpen &&
      settingsPage === "accounts"
    ) {
      void loadGmailStatus();
    }
  }, [settingsOpen, settingsPage]);

  // -------------------------------------------------------------------------
  // Persist theme
  // -------------------------------------------------------------------------

  useEffect(() => {
    localStorage.setItem(
      "recallx-theme",
      theme,
    );
  }, [theme]);

  // -------------------------------------------------------------------------
  // Close settings when clicking outside
  // -------------------------------------------------------------------------

  useEffect(() => {
    if (!settingsOpen) {
      return;
    }

    const handlePointerDown = (
      event: MouseEvent,
    ) => {
      const target = event.target as Node;

      if (
        settingsRef.current &&
        !settingsRef.current.contains(target)
      ) {
        setSettingsOpen(false);
      }
    };

    document.addEventListener(
      "mousedown",
      handlePointerDown,
    );

    return () => {
      document.removeEventListener(
        "mousedown",
        handlePointerDown,
      );
    };
  }, [settingsOpen]);

  // -------------------------------------------------------------------------
  // Search backend as the user types
  // -------------------------------------------------------------------------

  useEffect(() => {
    const controller = new AbortController();
    const currentSequence =
      ++searchSequence.current;

    const trimmedQuery = query.trim();

    if (!trimmedQuery) {
      setResults([]);
      setSelectedIndex(0);
      setLoading(false);

      return () => {
        controller.abort();
      };
    }

    // Clear previous results immediately so stale results don't remain visible.
    setResults([]);
    setSelectedIndex(0);
    setLoading(true);

    const searchBackend = async () => {
      try {
        const response = await fetch(
          `http://127.0.0.1:8000/search?q=${encodeURIComponent(
            trimmedQuery,
          )}&limit=20`,
          {
            signal: controller.signal,
            cache: "no-store",
          },
        );

        if (!response.ok) {
          throw new Error(
            `Search request failed with status ${response.status}`,
          );
        }

        const data = await response.json();

        // Ignore results from an old query.
        if (
          controller.signal.aborted ||
          currentSequence !==
            searchSequence.current
        ) {
          return;
        }

        setResults(
          Array.isArray(data.results)
            ? data.results
            : [],
        );

        setSelectedIndex(0);
      } catch (error) {
        if (
          error instanceof DOMException &&
          error.name === "AbortError"
        ) {
          return;
        }

        if (
          controller.signal.aborted ||
          currentSequence !==
            searchSequence.current
        ) {
          return;
        }

        console.error(
          "RecallX search failed:",
          error,
        );

        setResults([]);
        setSelectedIndex(0);
      } finally {
        if (
          !controller.signal.aborted &&
          currentSequence ===
            searchSequence.current
        ) {
          setLoading(false);
        }
      }
    };

    // Local search stays responsive, while Gmail gets a slightly longer
    // debounce so typing an email query does not trigger a network request
    // for every intermediate query.
    const debounceMs = isLikelyGmailQuery(trimmedQuery)
      ? 350
      : 220;

    const timeout = window.setTimeout(
      searchBackend,
      debounceMs,
    );

    return () => {
      window.clearTimeout(timeout);
      controller.abort();
    };
  }, [query]);

  // -------------------------------------------------------------------------
  // Dynamic window height
  // -------------------------------------------------------------------------

  useEffect(() => {
    const updateWindowHeight =
      async () => {
        try {
          let height = 70;

          if (hasQuery) {
            if (loading) {
              height = 120;
            } else if (
              results.length === 0
            ) {
              height = 120;
            } else {
              // Fixed results viewport.
              // Additional results are accessed through scrolling.
              height =
                70 +
                RESULTS_VIEW_HEIGHT +
                18;

              /*
               * If settings is open while searching, make sure the
               * settings menu also has enough room to render.
               */
              if (settingsOpen) {
                height += 120;
              }
            }
          } else if (settingsOpen) {
            /* Nested settings pages need more room than the main menu. */
            height =
              settingsPage === "accounts"
                ? 300
                : 190;
          }

          await invoke(
            "set_window_height",
            {
              height,
            },
          );
        } catch (error) {
          console.error(
            "Failed to resize RecallX:",
            error,
          );
        }
      };

    updateWindowHeight();
  }, [
    hasQuery,
    loading,
    results.length,
    settingsOpen,
    settingsPage,
  ]);

  // -------------------------------------------------------------------------
  // Keep selected result valid
  // -------------------------------------------------------------------------

  useEffect(() => {
    if (results.length === 0) {
      setSelectedIndex(0);
      return;
    }

    setSelectedIndex((current) =>
      Math.min(
        current,
        results.length - 1,
      ),
    );
  }, [results.length]);

  // -------------------------------------------------------------------------
  // Automatically scroll selected result into view
  // -------------------------------------------------------------------------

  useEffect(() => {
    const container =
      resultsScrollRef.current;

    if (!container) {
      return;
    }

    const selectedElement =
      container.querySelector<HTMLElement>(
        `[data-result-index="${selectedIndex}"]`,
      );

    if (!selectedElement) {
      return;
    }

    selectedElement.scrollIntoView({
      behavior: "smooth",
      block: "nearest",
    });
  }, [selectedIndex, results]);

  // -------------------------------------------------------------------------
  // Focus input
  // -------------------------------------------------------------------------

  useEffect(() => {
    inputRef.current?.focus();

    const timeout = window.setTimeout(
      () => {
        inputRef.current?.focus();
      },
      50,
    );

    return () => {
      window.clearTimeout(timeout);
    };
  }, []);

  const getGmailUrl = (
    result: BackendResult,
  ): string | null => {
    if (result.source_type !== "gmail") {
      return null;
    }

    if (
      result.metadata_json &&
      typeof result.metadata_json === "object" &&
      typeof result.metadata_json.url === "string"
    ) {
      return result.metadata_json.url;
    }

    if (typeof result.metadata_json === "string") {
      try {
        const metadata = JSON.parse(
          result.metadata_json,
        );

        if (typeof metadata.url === "string") {
          return metadata.url;
        }
      } catch {
        // Ignore malformed metadata.
      }
    }

    return null;
  };

  const getGmailMetadata = (result: BackendResult) => {
    if (result.source_type !== "gmail") {
      return null;
    }

    if (
      result.metadata_json &&
      typeof result.metadata_json === "object"
    ) {
      return result.metadata_json;
    }

    if (typeof result.metadata_json === "string") {
      try {
        const metadata = JSON.parse(result.metadata_json);
        if (metadata && typeof metadata === "object") {
          return metadata as {
            sender?: string;
            recipient?: string;
            date?: string;
            url?: string;
          };
        }
      } catch {
        // Ignore malformed metadata.
      }
    }

    return null;
  };

  // -------------------------------------------------------------------------
  // Open selected result
  // -------------------------------------------------------------------------

  const openSelectedResult = async (index = selectedIndex) => {
    const selectedResult =
      results[index];

    if (!selectedResult) {
      return;
    }

    // Deleted memories remain searchable but cannot be opened.
    if (selectedResult.is_deleted) {
      return;
    }

    try {
      // Gmail result → open the original Gmail message.
      if (selectedResult.source_type === "gmail") {
        const gmailUrl =
          getGmailUrl(selectedResult);

        if (!gmailUrl) {
          console.error(
            "Gmail result has no URL:",
            selectedResult,
          );
          return;
        }

        // Hide RecallX natively before opening Gmail.
        // Keep Tauri's openUrl() so the original Gmail URL is preserved.
        try {
          await invoke("hide_window");
        } catch (error) {
          console.error(
            "Failed to hide RecallX before opening Gmail:",
            error,
          );
        }

        await openUrl(gmailUrl);

        return;
      }

      // Local filesystem result → existing behavior.
      await invoke("open_file", {
        path: selectedResult.path,
      });
    } catch (error) {
      console.error(
        "Failed to open result:",
        error,
      );
    }
  };

  // -------------------------------------------------------------------------
  // Close RecallX with animation
  // -------------------------------------------------------------------------

  const closeRecallX = async () => {
    if (isClosing) {
      return;
    }

    setIsClosing(true);

    await new Promise<void>(
      (resolve) => {
        window.setTimeout(
          resolve,
          CLOSE_ANIMATION_DURATION,
        );
      },
    );

    try {
      await getCurrentWindow().hide();
    } catch (error) {
      console.error(
        "Failed to hide RecallX:",
        error,
      );
    } finally {
      setIsClosing(false);
    }
  };

  // -------------------------------------------------------------------------
  // Theme selection
  // -------------------------------------------------------------------------

  const selectTheme = (
    nextTheme: Theme,
  ) => {
    setTheme(nextTheme);
    setSettingsPage("main");

    // Return focus to search after selecting an option.
    window.setTimeout(() => {
      inputRef.current?.focus();
    }, 0);
  };

  const openSettingsPage = (
    page: SettingsPage,
  ) => {
    setSettingsPage(page);
    setGmailError("");
  };

  const closeSettings = () => {
    setSettingsOpen(false);
    setSettingsPage("main");
    setGmailError("");
  };

  const goBackToSettings = () => {
    setSettingsPage("main");
    setGmailError("");
  };

  // -------------------------------------------------------------------------
  // Keyboard controls
  // -------------------------------------------------------------------------

  const handleKeyDown = async (
    event: React.KeyboardEvent<HTMLInputElement>,
  ) => {
    if (event.key === "Escape") {
      event.preventDefault();

      if (settingsOpen) {
        if (settingsPage !== "main") {
          goBackToSettings();
        } else {
          closeSettings();
        }

        return;
      }

      await closeRecallX();

      return;
    }

    if (event.key === "ArrowDown") {
      event.preventDefault();

      if (results.length === 0) {
        return;
      }

      setSelectedIndex((current) =>
        Math.min(
          current + 1,
          results.length - 1,
        ),
      );

      return;
    }

    if (event.key === "ArrowUp") {
      event.preventDefault();

      if (results.length === 0) {
        return;
      }

      setSelectedIndex((current) =>
        Math.max(
          current - 1,
          0,
        ),
      );

      return;
    }

    if (event.key === "Enter") {
      event.preventDefault();

      await openSelectedResult();
    }
  };

  return (
    <main
      className={[
        "recallx",
        `theme-${theme}`,
        hasQuery ? "has-results" : "",
        isClosing ? "closing" : "",
      ]
        .filter(Boolean)
        .join(" ")}
    >
      <div className="search-container">
        <div className="search-icon">
          ⌕
        </div>

        <input
          ref={inputRef}
          autoFocus
          type="text"
          value={query}
          onChange={(event) =>
            setQuery(event.target.value)
          }
          onKeyDown={handleKeyDown}
          placeholder="Search your memory..."
          aria-label="Search your memory"
        />

        {/* ---------------------------------------------------------------- */}
        {/* Settings                                                         */}
        {/* ---------------------------------------------------------------- */}

        <div
          ref={settingsRef}
          className="settings-wrapper"
        >
          <button
            type="button"
            className={[
              "settings-button",
              settingsOpen
                ? "settings-button-active"
                : "",
            ]
              .filter(Boolean)
              .join(" ")}
            onClick={() => {
              setSettingsOpen(
                (open) => !open,
              );
              setSettingsPage("main");
              setGmailError("");
            }}
            aria-label="Settings"
            aria-expanded={settingsOpen}
          >
            <span className="settings-dots">
              <i />
              <i />
              <i />
            </span>
          </button>

          {settingsOpen && (
            <div className="settings-menu">
              {settingsPage === "main" && (
                <>
                  <div className="settings-title">
                    SETTINGS
                  </div>

                  <button
                    type="button"
                    className="settings-option settings-option-chevron"
                    onClick={() =>
                      openSettingsPage(
                        "appearance",
                      )
                    }
                  >
                    <span className="settings-option-icon">
                      ◐
                    </span>

                    <span className="settings-option-label">
                      Appearance
                    </span>

                    <span className="settings-chevron">
                      ›
                    </span>
                  </button>

                  <button
                    type="button"
                    className="settings-option settings-option-chevron"
                    onClick={() =>
                      openSettingsPage(
                        "accounts",
                      )
                    }
                  >
                    <span className="settings-option-icon">
                      ◎
                    </span>

                    <span className="settings-option-label">
                      Accounts
                    </span>

                    <span className="settings-chevron">
                      ›
                    </span>
                  </button>

                  <button
                    type="button"
                    className="settings-option settings-option-chevron"
                    onClick={() =>
                      openSettingsPage(
                        "privacy",
                      )
                    }
                  >
                    <span className="settings-option-icon">
                      ◈
                    </span>

                    <span className="settings-option-label">
                      Privacy &amp; Indexing
                    </span>

                    <span className="settings-chevron">
                      ›
                    </span>
                  </button>
                </>
              )}

              {settingsPage ===
                "appearance" && (
                <>
                  <button
                    type="button"
                    className="settings-back"
                    onClick={
                      goBackToSettings
                    }
                  >
                    <span>‹</span>
                    <span>
                      Appearance
                    </span>
                  </button>

                  <button
                    type="button"
                    className={`settings-option ${
                      theme === "dark"
                        ? "active"
                        : ""
                    }`}
                    onClick={() =>
                      selectTheme("dark")
                    }
                  >
                    <span className="settings-option-icon">
                      ◐
                    </span>

                    <span className="settings-option-label">
                      Dark
                    </span>

                    {theme === "dark" && (
                      <span className="settings-check">
                        ✓
                      </span>
                    )}
                  </button>

                  <button
                    type="button"
                    className={`settings-option ${
                      theme === "light"
                        ? "active"
                        : ""
                    }`}
                    onClick={() =>
                      selectTheme("light")
                    }
                  >
                    <span className="settings-option-icon">
                      ○
                    </span>

                    <span className="settings-option-label">
                      Light
                    </span>

                    {theme === "light" && (
                      <span className="settings-check">
                        ✓
                      </span>
                    )}
                  </button>
                </>
              )}

              {settingsPage ===
                "accounts" && (
                <>
                  <button
                    type="button"
                    className="settings-back"
                    onClick={
                      goBackToSettings
                    }
                  >
                    <span>‹</span>
                    <span>
                      Accounts
                    </span>
                  </button>

                  <div className="settings-account-section">
                    <div className="settings-account-provider">
                      Google
                    </div>

                    <div className="settings-account-row">
                      <div className="settings-account-info">
                        <span className="settings-account-icon">
                          G
                        </span>

                        <span className="settings-account-name">
                          Gmail
                        </span>
                      </div>

                      <button
                        type="button"
                        className={`settings-account-action ${
                          gmailConnected
                            ? "connected"
                            : ""
                        }`}
                        disabled={
                          gmailLoading
                        }
                        onClick={() =>
                          gmailConnected
                            ? void disconnectGmail()
                            : void connectGmail()
                        }
                      >
                        {gmailLoading
                          ? "..."
                          : gmailConnected
                            ? "Connected ✓"
                            : "Connect"}
                      </button>
                    </div>

                    {gmailConnected &&
                      !gmailLoading && (
                        <div className="settings-account-status">
                          Gmail is connected to RecallX.
                        </div>
                      )}

                    {gmailError && (
                      <div className="settings-account-error">
                        {gmailError}
                      </div>
                    )}
                  </div>

                  <div className="settings-account-section">
                    <div className="settings-account-provider">
                      Microsoft
                    </div>

                    <div className="settings-account-row">
                      <div className="settings-account-info">
                        <span className="settings-account-icon">
                          M
                        </span>

                        <span className="settings-account-name">
                          Outlook
                        </span>
                      </div>

                      <button
                        type="button"
                        className="settings-account-action"
                        disabled
                        title="Outlook integration coming next"
                      >
                        Soon
                      </button>
                    </div>
                  </div>
                </>
              )}

              {settingsPage ===
                "privacy" && (
                <>
                  <button
                    type="button"
                    className="settings-back"
                    onClick={
                      goBackToSettings
                    }
                  >
                    <span>‹</span>
                    <span>
                      Privacy &amp; Indexing
                    </span>
                  </button>

                  <div className="settings-placeholder">
                    <div className="settings-placeholder-title">
                      Privacy &amp; Indexing
                    </div>

                    <div className="settings-placeholder-text">
                      Indexing controls will be added here.
                    </div>
                  </div>
                </>
              )}
            </div>
          )}
        </div>
      </div>

      {hasQuery && (
        <div className="results-container">
          {loading ? (
            <div className="status-container">
              <span className="status-dot" />

              <span>
                Searching your memory...
              </span>
            </div>
          ) : results.length === 0 ? (
            <div className="status-container empty">
              <span>
                No memories found
              </span>
            </div>
          ) : (
            <>
              <div
                ref={resultsScrollRef}
                className="results-scroll"
              >
                {results.map(
                  (result, index) => {
                    const isSelected =
                      index === selectedIndex;

                    const isDeleted =
                      result.is_deleted === 1;

                    const fileType =
                      getFileType(result);
                    const isGmail =
                      result.source_type === "gmail";
                    const gmailMetadata =
                      isGmail
                        ? getGmailMetadata(result)
                        : null;

                    const folderName =
                      getFolderName(
                        result.path,
                      );

                    const gmailDate =
                      gmailMetadata?.date
                        ? new Date(
                            gmailMetadata.date,
                          ).toLocaleDateString(
                            undefined,
                            {
                              day: "numeric",
                              month: "short",
                              year: "numeric",
                            },
                          )
                        : "";

                    return (
                      <button
                        type="button"
                        data-result-index={
                          index
                        }
                        className={[
                          "result-card",
                          isSelected
                            ? "selected"
                            : "",
                          isDeleted
                            ? "deleted-result"
                            : "",
                        ]
                          .filter(Boolean)
                          .join(" ")}
                        key={result.id}
                        onMouseEnter={() =>
                          setSelectedIndex(
                            index,
                          )
                        }
                        onClick={() => {
                          if (!isDeleted) {
                            setSelectedIndex(index);
                            void openSelectedResult(index);
                          }
                        }}
                      >
                        <div className="result-main">
                          {/* ------------------------------------------------ */}
                          {/* File type                                         */}
                          {/* ------------------------------------------------ */}

                          <div
                            className={[
                              "file-type-badge",
                              `file-type-${fileType.toLowerCase()}`,
                              result.source_type === "gmail"
                                ? "file-type-gmail"
                                : "",
                            ]
                              .filter(Boolean)
                              .join(" ")}
                          >
                            {isDeleted ? (
                              "×"
                            ) : result.source_type === "gmail" ? (
                              <span
                                className="mail-icon"
                                aria-label="Gmail"
                                title="Gmail"
                              >
                                ✉
                              </span>
                            ) : (
                              fileType
                            )}
                          </div>

                          {/* ------------------------------------------------ */}
                          {/* File information                                  */}
                          {/* ------------------------------------------------ */}

                          <div className="result-content">
                            <div className="result-name">
                              {result.title ||
                                result.path}
                            </div>

                            {isGmail ? (
                              <>
                                <div className="result-path">
                                  {gmailMetadata?.sender ||
                                    "Gmail"}
                                  {gmailDate
                                    ? ` · ${gmailDate}`
                                    : ""}
                                </div>

                                {result.content ? (
                                  <div className="result-path">
                                    {result.content}
                                  </div>
                                ) : null}
                              </>
                            ) : (
                              <div className="result-path">
                                {getParentPath(
                                  result.path,
                                )}
                              </div>
                            )}

                            <div className="result-footer">
                              {isDeleted ? (
                                <span className="deleted-label">
                                  Deleted memory
                                </span>
                              ) : (
                                <span className="source-label">
                                  {isGmail
                                    ? "GMAIL"
                                    : result.source_type}
                                </span>
                              )}

                              {isSelected &&
                                !isDeleted && (
                                  <span className="open-hint">
                                    Enter to open
                                  </span>
                                )}
                            </div>
                          </div>

                          {/* ------------------------------------------------ */}
                          {/* Cheap local metadata                             */}
                          {/* ------------------------------------------------ */}

                          <div className="result-meta">
                            {!isGmail &&
                              result.modified_at && (
                                <span className="meta-tag">
                                  {formatRelativeTime(
                                    result.modified_at,
                                  )}
                                </span>
                              )}

                            {!isGmail &&
                              folderName && (
                                <span className="meta-tag">
                                  {folderName}
                                </span>
                              )}

                            {isGmail &&
                              gmailMetadata?.recipient && (
                                <span className="meta-tag">
                                  TO: {gmailMetadata.recipient}
                                </span>
                              )}

                            {result.context &&
                              result.context
                                .length >
                                0 && (
                                <span className="meta-tag">
                                  {
                                    result
                                      .context
                                      .length
                                  }{" "}
                                  related
                                </span>
                              )}
                          </div>
                        </div>
                      </button>
                    );
                  },
                )}
              </div>

              {results.length > 1 && (
                <div className="more-results">
                  {results.length} results

                  <span className="scroll-hint">
                    {" "}
                    · Scroll to browse
                  </span>
                </div>
              )}
            </>
          )}
        </div>
      )}
    </main>
  );
}

export default App;