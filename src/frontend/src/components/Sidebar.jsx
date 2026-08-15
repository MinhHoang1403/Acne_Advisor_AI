import { useEffect, useRef, useState } from 'react';

const PENDING_CONNECTION_STATES = new Set(['checking', 'recovering']);

function compactConnectionLabel(connectionState) {
  if (connectionState === 'connected') return 'Đang hoạt động';
  if (connectionState === 'degraded') return 'Kết nối hạn chế';
  return 'Đang chờ kết nối';
}

export default function Sidebar({
  sessions,
  activeSessionId,
  sidebarOpen,
  backendOnline = true,
  connectionStatus = null,
  onNewChat,
  onSelectSession,
  onRenameSession,
  onHideSession,
  onClearLocalHistory,
  historyHidden = false,
  historyHiddenAt = null,
  onHideAllSessions,
  onShowHistory,
  onDeleteAllSessions,
  onToggleSidebar,
}) {
  const [menuOpenForId, setMenuOpenForId] = useState(null);
  const [historyMenuOpen, setHistoryMenuOpen] = useState(false);
  const historyMenuRef = useRef(null);

  const visibleSessions = sessions
    .filter((s) => !s.hidden && (!historyHiddenAt || s.createdAt > historyHiddenAt))
    .sort((a, b) => b.updatedAt - a.updatedAt);
  const hasVisibleStoredSessions = sessions.some((s) => !s.hidden);
  const connectionState = connectionStatus?.state || (backendOnline ? 'connected' : 'disconnected');
  const isConnectionPending = PENDING_CONNECTION_STATES.has(connectionState);
  const showConnectionBanner = connectionState !== 'connected' && !isConnectionPending;
  const connectionBannerClass = `sidebar-offline-banner sidebar-offline-banner-${connectionState}`;
  const compactStatusLabel = compactConnectionLabel(connectionState);
  const connectionMessage =
    (isConnectionPending ? compactStatusLabel : connectionStatus?.message) ||
    (backendOnline
      ? 'Backend đã kết nối.'
      : 'Đang dùng lịch sử cục bộ vì chưa kết nối backend.');

  useEffect(() => {
    if (!historyMenuOpen) return;

    const handlePointerDown = (event) => {
      if (historyMenuRef.current && !historyMenuRef.current.contains(event.target)) {
        setHistoryMenuOpen(false);
      }
    };

    const handleKeyDown = (event) => {
      if (event.key === 'Escape') {
        setHistoryMenuOpen(false);
      }
    };

    document.addEventListener('pointerdown', handlePointerDown);
    document.addEventListener('keydown', handleKeyDown);
    return () => {
      document.removeEventListener('pointerdown', handlePointerDown);
      document.removeEventListener('keydown', handleKeyDown);
    };
  }, [historyMenuOpen]);

  const handleRename = (e, id) => {
    e.stopPropagation();
    setMenuOpenForId(null);
    const session = sessions.find((s) => s.id === id);
    if (!session) return;
    const newTitle = prompt('Nhập tên mới cho đoạn chat:', session.title);
    if (newTitle && newTitle.trim()) {
      onRenameSession(id, newTitle.trim());
    }
  };

  const handleHide = (e, id) => {
    e.stopPropagation();
    setMenuOpenForId(null);
    onHideSession(id);
  };

  const handleHistoryAction = (action) => {
    setHistoryMenuOpen(false);
    action();
  };

  return (
    <aside className={`sidebar ${sidebarOpen ? 'sidebar-open' : 'sidebar-closed'}`} aria-label="Lịch sử chat">
      {/* App Title & Subtitle */}
      <div className="sidebar-header">
        <div className="sidebar-brand-row">
          <div className="sidebar-brand-copy">
            <h1 className="sidebar-title">Acne Advisor AI</h1>
          </div>
          <button
            type="button"
            className="sidebar-inline-toggle-btn"
            onClick={onToggleSidebar}
            aria-label="Ẩn thanh lịch sử chat"
            title="Ẩn lịch sử chat"
          >
            <svg width="18" height="18" fill="none" viewBox="0 0 24 24" stroke="currentColor">
              <path
                strokeLinecap="round"
                strokeLinejoin="round"
                strokeWidth="2"
                d="M4 6h16M4 12h16M4 18h16"
              />
            </svg>
          </button>
        </div>
      </div>

      {/* New chat button */}
      <div className="sidebar-new-chat">
        <button className="new-chat-btn" onClick={onNewChat} type="button">
          <svg width="16" height="16" fill="none" viewBox="0 0 24 24" stroke="currentColor">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M12 4v16m8-8H4" />
          </svg>
          Đoạn chat mới
        </button>
      </div>

      {/* Chat history list */}
      <div className="sidebar-history">
        <div className="sidebar-history-header">
          <div className="sidebar-history-label">Lịch sử chat</div>
          <div className="sidebar-history-menu" ref={historyMenuRef}>
            <button
              type="button"
              className={`sidebar-history-menu-btn ${historyMenuOpen ? 'sidebar-history-menu-btn-active' : ''}`}
              onClick={() => {
                setMenuOpenForId(null);
                setHistoryMenuOpen((open) => !open);
              }}
              aria-haspopup="menu"
              aria-expanded={historyMenuOpen}
              aria-label="Mở tùy chọn lịch sử chat"
              title="Tùy chọn lịch sử chat"
            >
              <svg width="16" height="16" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                <path
                  strokeLinecap="round"
                  strokeLinejoin="round"
                  strokeWidth="2"
                  d="M5 12h.01M12 12h.01M19 12h.01"
                />
              </svg>
            </button>
            {historyMenuOpen && (
              <div className="sidebar-history-dropdown" role="menu">
                {historyHidden ? (
                  <button
                    type="button"
                    className="sidebar-history-dropdown-item"
                    onClick={() => handleHistoryAction(onShowHistory)}
                    role="menuitem"
                  >
                    Hiện lại lịch sử
                  </button>
                ) : (
                  <button
                    type="button"
                    className="sidebar-history-dropdown-item"
                    onClick={() => handleHistoryAction(onHideAllSessions)}
                    disabled={!hasVisibleStoredSessions}
                    role="menuitem"
                  >
                    Ẩn tất cả
                  </button>
                )}
                <button
                  type="button"
                  className="sidebar-history-dropdown-item sidebar-history-dropdown-item-danger"
                  onClick={() => handleHistoryAction(onDeleteAllSessions)}
                  disabled={sessions.length === 0}
                  role="menuitem"
                >
                  Xoá tất cả
                </button>
              </div>
            )}
          </div>
        </div>
        <div className="sidebar-sessions-list">
          {visibleSessions.map((session) => (
            <div key={session.id} className="session-item-wrapper">
              <button
                className={`session-item ${activeSessionId === session.id ? 'session-item-active' : ''}`}
                onClick={() => onSelectSession(session.id)}
                title={session.title}
                aria-current={activeSessionId === session.id ? 'page' : undefined}
              >
                {session.title}
              </button>
              <button
                className={`session-menu-btn ${
                  menuOpenForId === session.id || activeSessionId === session.id
                    ? 'session-menu-btn-visible'
                    : ''
                }`}
                onClick={(e) => {
                  e.stopPropagation();
                  setMenuOpenForId(menuOpenForId === session.id ? null : session.id);
                }}
                title="Tùy chọn"
                aria-label={`Tùy chọn cho ${session.title}`}
                aria-haspopup="menu"
                aria-expanded={menuOpenForId === session.id}
              >
                <svg width="16" height="16" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                  <path
                    strokeLinecap="round"
                    strokeLinejoin="round"
                    strokeWidth="2"
                    d="M5 12h.01M12 12h.01M19 12h.01M6 12a1 1 0 11-2 0 1 1 0 012 0zm7 0a1 1 0 11-2 0 1 1 0 012 0zm7 0a1 1 0 11-2 0 1 1 0 012 0z"
                  />
                </svg>
              </button>

              {/* Dropdown menu */}
              {menuOpenForId === session.id && (
                <div className="session-dropdown" onMouseLeave={() => setMenuOpenForId(null)}>
                  <button className="dropdown-item" onClick={(e) => handleRename(e, session.id)}>
                    <svg width="16" height="16" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                      <path
                        strokeLinecap="round"
                        strokeLinejoin="round"
                        strokeWidth="2"
                        d="M15.232 5.232l3.536 3.536m-2.036-5.036a2.5 2.5 0 113.536 3.536L6.5 21.036H3v-3.572L16.732 3.732z"
                      />
                    </svg>
                    Đổi tên
                  </button>
                  <button className="dropdown-item dropdown-item-danger" onClick={(e) => handleHide(e, session.id)}>
                    <svg width="16" height="16" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                      <path
                        strokeLinecap="round"
                        strokeLinejoin="round"
                        strokeWidth="2"
                        d="M13.875 18.825A10.05 10.05 0 0112 19c-4.478 0-8.268-2.943-9.543-7a9.97 9.97 0 011.563-3.029m5.858.908a3 3 0 114.243 4.243M9.878 9.878l4.242 4.242M9.88 9.88l-3.29-3.29m7.532 7.532l3.29 3.29M3 3l3.59 3.59m0 0A9.953 9.953 0 0112 5c4.478 0 8.268 2.943 9.543 7a10.025 10.025 0 01-4.132 5.411m0 0L21 21"
                      />
                    </svg>
                    Ẩn khỏi lịch sử
                  </button>
                </div>
              )}
            </div>
          ))}
          {visibleSessions.length === 0 && (
            <div className="sidebar-empty">
              {historyHidden ? 'Đã ẩn lịch sử chat' : 'Chưa có lịch sử chat'}
            </div>
          )}
        </div>
      </div>

      {/* Offline banner */}
      {showConnectionBanner && (
        <div className={connectionBannerClass}>
          <svg width="14" height="14" fill="none" viewBox="0 0 24 24" stroke="currentColor">
            <path
              strokeLinecap="round"
              strokeLinejoin="round"
              strokeWidth="2"
              d="M12 9v2m0 4h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z"
            />
          </svg>
          <span>{connectionMessage}</span>
          {connectionState === 'disconnected' && onClearLocalHistory && (
            <button
              type="button"
              className="sidebar-offline-clear"
              onClick={onClearLocalHistory}
            >
              Xoá lịch sử cục bộ
            </button>
          )}
        </div>
      )}
      <div className="sidebar-footer">
        <span className={`sidebar-footer-dot sidebar-footer-dot-${connectionState}`} />
        <span>{compactStatusLabel}</span>
      </div>
    </aside>
  );
}
