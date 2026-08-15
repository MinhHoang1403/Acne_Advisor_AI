import { useState, useEffect, useCallback, useRef } from 'react';
import Sidebar from './components/Sidebar.jsx';
import ChatWindow from './components/ChatWindow.jsx';
import {
  CONNECTION_STATES,
  HEALTH_TIMING,
  classifyHealthResult,
  isBackendReachable,
  nextHealthDelayMs,
  shouldTreatChatErrorAsDisconnected,
} from './api/connectivity.js';
import {
  sendChatMessage,
  checkBackendHealth,
  fetchSessions,
  fetchMessages,
  renameSession as apiRenameSession,
  hideSession as apiHideSession,
  deleteAllChatSessions,
} from './api/chatApi.js';
import {
  loadSessions,
  saveSessions,
  loadActiveSessionId,
  saveActiveSessionId,
  loadHistoryHiddenAt,
  saveHistoryHiddenAt,
  generateId,
  clearLocalChatCache,
} from './utils/storage.js';
import {
  deriveChatTitleFromFirstUserMessage,
  shouldGenerateSessionTitle,
} from './utils/chatTitle.js';

export default function App() {
  // ── State ──────────────────────────────────────────────
  const [sessions, setSessions] = useState(() => loadSessions());
  const [activeSessionId, setActiveSessionId] = useState(() => loadActiveSessionId());
  const [message, setMessage] = useState('');
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState(null);
  const [sidebarOpen, setSidebarOpen] = useState(true);
  const [connectionStatus, setConnectionStatus] = useState({
    state: CONNECTION_STATES.CHECKING,
    message: 'Đang chờ kết nối',
    health: null,
    reason: null,
  });
  const [loadingMessages, setLoadingMessages] = useState(false);
  const [historyHiddenAt, setHistoryHiddenAt] = useState(() => loadHistoryHiddenAt());
  const requestInFlight = useRef(false);

  const backendSessionsLoaded = useRef(false);
  const healthTimerRef = useRef(null);
  const healthAbortRef = useRef(null);
  const healthSequenceRef = useRef(0);
  const healthAttemptRef = useRef(0);

  // ── Derived ────────────────────────────────────────────
  const activeSession = sessions.find((s) => s.id === activeSessionId) || null;
  const chatHistory = activeSession ? activeSession.messages : [];
  const backendOnline = isBackendReachable(connectionStatus);

  // ── Persistence to localStorage ────────────────────────
  useEffect(() => {
    saveSessions(sessions);
  }, [sessions]);

  useEffect(() => {
    saveActiveSessionId(activeSessionId);
  }, [activeSessionId]);

  useEffect(() => {
    saveHistoryHiddenAt(historyHiddenAt);
  }, [historyHiddenAt]);

  // ── Backend health loop: startup order, restart, degraded state ────────
  useEffect(() => {
    let cancelled = false;

    const clearHealthTimer = () => {
      if (healthTimerRef.current) {
        clearTimeout(healthTimerRef.current);
        healthTimerRef.current = null;
      }
    };

    const scheduleNextCheck = (state) => {
      if (cancelled) return;
      clearHealthTimer();
      const delay = nextHealthDelayMs(state, healthAttemptRef.current, document.visibilityState);
      healthTimerRef.current = setTimeout(() => runHealthCheck('scheduled'), delay);
    };

    const runHealthCheck = async (reason) => {
      const sequence = ++healthSequenceRef.current;
      healthAbortRef.current?.abort();
      const controller = new AbortController();
      healthAbortRef.current = controller;

      setConnectionStatus((prev) => {
        if (prev.state === CONNECTION_STATES.CONNECTED || prev.state === CONNECTION_STATES.DEGRADED) {
          return prev;
        }
        return {
          ...prev,
          state: reason === 'startup' ? CONNECTION_STATES.CHECKING : CONNECTION_STATES.RECOVERING,
          message: 'Đang chờ kết nối',
        };
      });

      const result = await checkBackendHealth({
        timeoutMs: HEALTH_TIMING.timeoutMs,
        signal: controller.signal,
      });

      if (cancelled || sequence !== healthSequenceRef.current) return;

      const nextStatus = classifyHealthResult(result);
      setConnectionStatus(nextStatus);

      if (isBackendReachable(nextStatus)) {
        healthAttemptRef.current = 0;
      } else {
        healthAttemptRef.current += 1;
      }

      scheduleNextCheck(nextStatus.state);
    };

    const handleVisibilityChange = () => {
      if (document.visibilityState === 'visible') {
        runHealthCheck('visible');
      }
    };

    document.addEventListener('visibilitychange', handleVisibilityChange);
    runHealthCheck('startup');

    return () => {
      cancelled = true;
      clearHealthTimer();
      healthAbortRef.current?.abort();
      document.removeEventListener('visibilitychange', handleVisibilityChange);
    };
  }, []);

  // ── Load persisted sessions once backend is reachable ────────
  useEffect(() => {
    if (!backendOnline || backendSessionsLoaded.current) return;
    backendSessionsLoaded.current = true;

    (async () => {
      try {
        const backendSessions = await fetchSessions();
        const normalizedBackendSessions = backendSessions.map((s) => ({
          id: s.id,
          title: s.title,
          createdAt: new Date(s.created_at).getTime(),
          updatedAt: new Date(s.updated_at).getTime(),
          hidden: s.hidden,
          messages: [],
          _fromBackend: true,
        }));

        setSessions((prev) => {
          const localOnly = prev.filter((session) => !session._fromBackend);
          const backendIds = new Set(normalizedBackendSessions.map((session) => session.id));
          return [
            ...localOnly.filter((session) => !backendIds.has(session.id)),
            ...normalizedBackendSessions,
          ];
        });

        const backendIds = new Set(normalizedBackendSessions.map((s) => s.id));
        if (!activeSessionId || !backendIds.has(activeSessionId)) {
          setActiveSessionId((current) => current || normalizedBackendSessions[0]?.id || null);
        }
      } catch (err) {
        backendSessionsLoaded.current = false;
        console.warn('Failed to load sessions from backend:', err);
      }
    })();
  }, [activeSessionId, backendOnline]);

  // ── Load messages from backend when selecting a session ─
  const _loadMessagesFromBackend = useCallback(async (sessionId) => {
    if (!backendOnline) return;

    setLoadingMessages(true);
    try {
      const msgs = await fetchMessages(sessionId);
      if (msgs && msgs.length > 0) {
        const formattedMsgs = msgs.map((m) => ({
          role: m.role,
          content: m.content,
          id: m.id,
          data: m.role === 'assistant' ? {
            answer: m.content,
            sources: m.sources || [],
            metadata: m.metadata || {},
            source_metadata: m.metadata?.source_metadata || [],
          } : undefined,
        }));

        setSessions((prev) =>
          prev.map((s) => {
            if (s.id === sessionId) {
              return { ...s, messages: formattedMsgs, _fromBackend: true };
            }
            return s;
          })
        );
      }
    } catch (err) {
      console.warn('Failed to load messages from backend:', err);
    } finally {
      setLoadingMessages(false);
    }
  }, [backendOnline]);

  // ── Handlers ───────────────────────────────────────────
  const handleNewChat = useCallback(() => {
    setActiveSessionId(null);
    setError(null);
    setMessage('');
  }, []);

  const handleSelectSession = useCallback(
    (id) => {
      setActiveSessionId(id);
      setError(null);
      if (window.innerWidth < 768) {
        setSidebarOpen(false);
      }

      // Load messages from backend if this session came from backend
      // and doesn't have messages loaded yet
      const session = sessions.find((s) => s.id === id);
      if (session && session._fromBackend && (!session.messages || session.messages.length === 0)) {
        _loadMessagesFromBackend(id);
      } else if (backendOnline && session) {
        // Even if we have local messages, refresh from backend to get latest
        _loadMessagesFromBackend(id);
      }
    },
    [sessions, backendOnline, _loadMessagesFromBackend]
  );

  const handleRenameSession = useCallback(async (id, newTitle) => {
    // Always update local state first
    setSessions((prev) =>
      prev.map((s) => (s.id === id ? { ...s, title: newTitle, updatedAt: Date.now() } : s))
    );

    // Then sync to backend if online
    if (backendOnline) {
      try {
        await apiRenameSession(id, newTitle);
      } catch (err) {
        console.warn('Failed to rename session on backend:', err);
      }
    }
  }, [backendOnline]);

  const handleHideSession = useCallback(
    async (id) => {
      // Update local state — only set hidden, never delete
      setSessions((prev) =>
        prev.map((s) => (s.id === id ? { ...s, hidden: true, updatedAt: Date.now() } : s))
      );

      // If the hidden session was the active one, switch to the most recent visible session
      if (activeSessionId === id) {
        const remaining = sessions
          .filter((s) => s.id !== id && !s.hidden)
          .sort((a, b) => b.updatedAt - a.updatedAt);
        setActiveSessionId(remaining.length > 0 ? remaining[0].id : null);
      }

      // Sync to backend if online
      if (backendOnline) {
        try {
          await apiHideSession(id);
        } catch (err) {
          console.warn('Failed to hide session on backend:', err);
        }
      }
    },
    [activeSessionId, sessions, backendOnline]
  );

  const handleClearLocalHistory = useCallback(() => {
    clearLocalChatCache();
    setSessions([]);
    setActiveSessionId(null);
    setHistoryHiddenAt(null);
  }, []);

  const handleHideAllSessions = useCallback(() => {
    setHistoryHiddenAt(Date.now());
    setActiveSessionId(null);
    setError(null);
  }, []);

  const handleShowHistory = useCallback(() => {
    setHistoryHiddenAt(null);
  }, []);

  const handleDeleteAllSessions = useCallback(async () => {
    const confirmed = window.confirm(
      'Bạn có chắc muốn xoá toàn bộ lịch sử chat? Hành động này sẽ xoá dữ liệu trong PostgreSQL và cache Redis liên quan, không thể hoàn tác.'
    );
    if (!confirmed) return;

    try {
      const result = await deleteAllChatSessions();
      console.info('Deleted chat history:', result);
      clearLocalChatCache();
      setSessions([]);
      setActiveSessionId(null);
      setHistoryHiddenAt(null);
      setError(null);

      if (backendOnline) {
        const backendSessions = await fetchSessions();
        const normalizedBackendSessions = backendSessions.map((s) => ({
          id: s.id,
          title: s.title,
          createdAt: new Date(s.created_at).getTime(),
          updatedAt: new Date(s.updated_at).getTime(),
          hidden: s.hidden,
          messages: [],
          _fromBackend: true,
        }));
        setSessions(normalizedBackendSessions);
      }
    } catch (err) {
      console.error('Failed to delete all chat history:', err);
      setError('Không xoá được lịch sử chat. Hãy kiểm tra backend rồi thử lại.');
    }
  }, [backendOnline]);

  const sendQuestion = useCallback(
    async (userMessageText, modelConfig = null) => {
      const trimmedMessage = userMessageText.trim();
      if (!trimmedMessage || isLoading || requestInFlight.current) return false;

      setError(null);
      setIsLoading(true);
      requestInFlight.current = true;

      let currentSessionId = activeSessionId;
      let generatedSessionTitle = null;

      // Create a new session if none is active
      if (!currentSessionId) {
        currentSessionId = generateId();
        generatedSessionTitle = deriveChatTitleFromFirstUserMessage(trimmedMessage);
        const newSession = {
          id: currentSessionId,
          title: generatedSessionTitle,
          createdAt: Date.now(),
          updatedAt: Date.now(),
          hidden: false,
          messages: [],
        };
        setSessions((prev) => [newSession, ...prev]);
        setActiveSessionId(currentSessionId);
      } else {
        const currentSession = sessions.find((s) => s.id === currentSessionId);
        if (currentSession && shouldGenerateSessionTitle(currentSession.title, currentSession.messages)) {
          generatedSessionTitle = deriveChatTitleFromFirstUserMessage(trimmedMessage);
        }
      }

      // Add user message to session
      const userMsgObj = { role: 'user', content: trimmedMessage };
      setSessions((prev) =>
        prev.map((s) => {
          if (s.id === currentSessionId) {
            const title =
              generatedSessionTitle && shouldGenerateSessionTitle(s.title, s.messages)
                ? generatedSessionTitle
                : s.title;
            return { ...s, title, messages: [...s.messages, userMsgObj], updatedAt: Date.now() };
          }
          return s;
        })
      );

      // Compile conversation history (max 6 recent messages, role+content only)
      const currentSessionObj = sessions.find((s) => s.id === currentSessionId) || {
        messages: [],
      };
      const history = currentSessionObj.messages
        .slice(-6)
        .map((m) => ({ role: m.role, content: m.content }));

      try {
        const data = await sendChatMessage({
          message: trimmedMessage,
          sessionId: currentSessionId,
          conversationHistory: history,
          llmProvider: modelConfig?.llmProvider,
          llmModel: modelConfig?.llmModel,
          allowModelFallback: modelConfig?.allowModelFallback,
        });

        // Update session_id from backend response if different
        const responseSessionId = data.session_id || currentSessionId;
        if (responseSessionId !== currentSessionId) {
          // Backend assigned a different session ID — update
          setSessions((prev) =>
            prev.map((s) => {
              if (s.id === currentSessionId) {
                return { ...s, id: responseSessionId };
              }
              return s;
            })
          );
          setActiveSessionId(responseSessionId);
        }

        if (backendOnline && generatedSessionTitle) {
          try {
            await apiRenameSession(responseSessionId, generatedSessionTitle);
          } catch (err) {
            console.warn('Failed to sync generated session title:', err);
          }
        }

        // Add assistant message
        const assistantMsgObj = { role: 'assistant', content: data.answer, data };
        setSessions((prev) =>
          prev.map((s) => {
            if (s.id === currentSessionId || s.id === responseSessionId) {
              return { ...s, messages: [...s.messages, assistantMsgObj] };
            }
            return s;
          })
        );
        return true;
      } catch (err) {
        setError(err?.message || 'Backend không thể xử lý yêu cầu. Vui lòng thử lại.');

        if (shouldTreatChatErrorAsDisconnected(err)) {
          const health = await checkBackendHealth({ timeoutMs: HEALTH_TIMING.timeoutMs });
          setConnectionStatus(classifyHealthResult(health));
        } else {
          setConnectionStatus((prev) => {
            if (prev.state === CONNECTION_STATES.DISCONNECTED || prev.state === CONNECTION_STATES.RECOVERING) {
              return {
                state: CONNECTION_STATES.CONNECTED,
                message: 'Backend đã kết nối.',
                health: prev.health,
                reason: null,
              };
            }
            return prev;
          });
        }
        return false;
      } finally {
        setIsLoading(false);
        requestInFlight.current = false;
      }
    },
    [activeSessionId, sessions, isLoading, backendOnline]
  );

  const handleSubmit = useCallback(async (text, modelConfig) => {
    const success = await sendQuestion(text, modelConfig);
    if (success) {
      setMessage('');
    }
  }, [sendQuestion]);

  // ── Render ─────────────────────────────────────────────
  return (
    <div className="app-layout">
      <Sidebar
        sessions={sessions}
        activeSessionId={activeSessionId}
        sidebarOpen={sidebarOpen}
        backendOnline={backendOnline}
        connectionStatus={connectionStatus}
        onNewChat={handleNewChat}
        onSelectSession={handleSelectSession}
        onRenameSession={handleRenameSession}
        onHideSession={handleHideSession}
        onClearLocalHistory={handleClearLocalHistory}
        historyHidden={historyHiddenAt !== null}
        historyHiddenAt={historyHiddenAt}
        onHideAllSessions={handleHideAllSessions}
        onShowHistory={handleShowHistory}
        onDeleteAllSessions={handleDeleteAllSessions}
        onToggleSidebar={() => setSidebarOpen(!sidebarOpen)}
      />
      <ChatWindow
        chatHistory={chatHistory}
        isLoading={isLoading || loadingMessages}
        error={error}
        message={message}
        setMessage={setMessage}
        onSubmit={handleSubmit}
        onSendQuestion={sendQuestion}
        sidebarOpen={sidebarOpen}
        onToggleSidebar={() => setSidebarOpen(!sidebarOpen)}
      />
    </div>
  );
}
