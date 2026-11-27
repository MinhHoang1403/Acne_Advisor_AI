import { useRef, useEffect, useState, useCallback, useMemo } from 'react';
import ChatMessage from './ChatMessage.jsx';
import EmptyState from './EmptyState.jsx';
import ChatInput from './ChatInput.jsx';
import { formatText } from '../utils/markdown.js';

export default function ChatWindow({
  chatHistory,
  sidebarOpen = true,
  isLoading,
  isRequestInFlight = false,
  loadingMessages = false,
  streamingStage = '',
  streamingDraft = '',
  error,
  message,
  setMessage,
  onSubmit,
  onCancel,
  onSendQuestion,
  onToggleSidebar,
  modelConfig = null,
  onModelConfigChange = null,
  backendOnline = true,
}) {
  const scrollAreaRef = useRef(null);
  const messagesEndRef = useRef(null);
  const isNearBottomRef = useRef(true);
  const prevHistoryLenRef = useRef(chatHistory.length);
  const [showScrollBottom, setShowScrollBottom] = useState(false);

  const handleScroll = useCallback(() => {
    const el = scrollAreaRef.current;
    if (!el) return;
    const distanceFromBottom = el.scrollHeight - el.scrollTop - el.clientHeight;
    const nearBottom = distanceFromBottom < 140;
    isNearBottomRef.current = nearBottom;
    setShowScrollBottom(!nearBottom && chatHistory.length > 0);
  }, [chatHistory.length]);

  const scrollToBottom = useCallback((behavior = 'smooth') => {
    const el = scrollAreaRef.current;
    if (!el) return;
    isNearBottomRef.current = true;
    setShowScrollBottom(false);
    el.scrollTo({ top: el.scrollHeight, behavior });
  }, []);

  // Cuộn mượt khi có lượt tin nhắn mới; bám sát tức thì không giật khung hình khi stream token.
  useEffect(() => {
    const historyGrew = chatHistory.length > prevHistoryLenRef.current;
    prevHistoryLenRef.current = chatHistory.length;

    if (historyGrew) {
      scrollToBottom('smooth');
      return;
    }
    if (isNearBottomRef.current && scrollAreaRef.current) {
      scrollAreaRef.current.scrollTop = scrollAreaRef.current.scrollHeight;
    }
  }, [chatHistory, isLoading, streamingDraft, scrollToBottom]);

  const renderedStreamDraft = useMemo(
    () => (streamingDraft ? formatText(streamingDraft) : null),
    [streamingDraft],
  );

  return (
    <div className="chat-window">
      {sidebarOpen && (
        <div
          className="sidebar-mobile-backdrop"
          onClick={onToggleSidebar}
          aria-hidden="true"
        />
      )}

      {!sidebarOpen && (
        <button
          className="sidebar-reopen-btn"
          onClick={onToggleSidebar}
          title="Mở lịch sử chat"
          aria-label="Mở thanh lịch sử chat"
          type="button"
        >
          <svg width="20" height="20" fill="none" viewBox="0 0 24 24" stroke="currentColor">
            <path
              strokeLinecap="round"
              strokeLinejoin="round"
              strokeWidth="2"
              d="M4 6h16M4 12h16M4 18h16"
            />
          </svg>
        </button>
      )}

      {/* Vùng scroll chỉ render history; không tự tải hoặc mutate session. */}
      <div
        ref={scrollAreaRef}
        onScroll={handleScroll}
        className={`chat-scroll-area ${chatHistory.length === 0 ? 'chat-scroll-area-empty' : ''}`}
      >
        {chatHistory.length === 0 ? (
          loadingMessages ? (
            <div className="chat-messages-list">
              <div className="chat-message chat-message-assistant" aria-live="polite">
                <div className="chat-message-inner">
                  <div className="chat-message-content">
                    <div className="loading-dots-wrapper">
                      <div className="loading-dots">
                        <div className="loading-dot" />
                        <div className="loading-dot" style={{ animationDelay: '0.15s' }} />
                        <div className="loading-dot" style={{ animationDelay: '0.3s' }} />
                      </div>
                      <span className="loading-stage-label">Đang tải lịch sử hội thoại...</span>
                    </div>
                  </div>
                </div>
              </div>
            </div>
          ) : (
            <>
              <EmptyState onSendQuestion={onSendQuestion} />
              {error && (
                <div className="chat-messages-list">
                  <div className="chat-message chat-message-assistant">
                    <div className="chat-message-inner">
                      <div className="chat-error" role="alert">
                        <span>{error}</span>
                      </div>
                    </div>
                  </div>
                </div>
              )}
            </>
          )
        ) : (
          <div className="chat-messages-list">
            {chatHistory.map((msg, index) => (
              <ChatMessage key={msg.id || `${msg.role}-${index}`} msg={msg} />
            ))}

            {/* Trạng thái xử lý thời gian thực & luồng văn bản đang sinh trực tiếp */}
            {isLoading && (
              <div className="chat-message chat-message-assistant chat-message-streaming" aria-live="polite">
                <div className="chat-message-inner">
                  <div className="chat-message-content">
                    {renderedStreamDraft ? (
                      <div className="chat-assistant-text">
                        <div className="chat-formatted-text chat-streaming-text">
                          {renderedStreamDraft}
                        </div>
                        {streamingStage && (
                          <div className="chat-stream-stage-inline">
                            <span className="chat-stream-stage-dot" />
                            <span>{streamingStage}</span>
                          </div>
                        )}
                      </div>
                    ) : (
                      <div className="loading-dots-wrapper">
                        <div className="loading-dots">
                          <div className="loading-dot" />
                          <div className="loading-dot" style={{ animationDelay: '0.15s' }} />
                          <div className="loading-dot" style={{ animationDelay: '0.3s' }} />
                        </div>
                        {streamingStage && (
                          <span className="loading-stage-label">{streamingStage}</span>
                        )}
                      </div>
                    )}
                  </div>
                </div>
              </div>
            )}

            {/* Error thuộc request hiện tại, không phải assistant message persisted. */}
            {error && (
              <div className="chat-message chat-message-assistant">
                <div className="chat-message-inner">
                  <div className="chat-error" role="alert">
                    <svg
                      width="20"
                      height="20"
                      fill="none"
                      viewBox="0 0 24 24"
                      stroke="currentColor"
                      className="error-icon"
                    >
                      <path
                        strokeLinecap="round"
                        strokeLinejoin="round"
                        strokeWidth="2"
                        d="M12 8v4m0 4h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z"
                      />
                    </svg>
                    <span>{error}</span>
                  </div>
                </div>
              </div>
            )}

            <div ref={messagesEndRef} style={{ height: '16px' }} />
          </div>
        )}
      </div>

      {showScrollBottom && chatHistory.length > 0 && (
        <button
          type="button"
          className="chat-scroll-bottom-btn"
          onClick={() => scrollToBottom('smooth')}
          aria-label="Cuộn xuống tin nhắn mới nhất"
          title="Xuống tin nhắn mới nhất"
        >
          <svg width="16" height="16" fill="none" viewBox="0 0 24 24" stroke="currentColor">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M19 14l-7 7m0 0l-7-7m7 7V3" />
          </svg>
          <span>Xuống cuối</span>
        </button>
      )}

      {/* Composer nhận controlled value và submit handler từ App. */}
      <ChatInput
        message={message}
        setMessage={setMessage}
        isLoading={isLoading || isRequestInFlight}
        onSubmit={onSubmit}
        onCancel={onCancel}
        modelConfig={modelConfig}
        onModelConfigChange={onModelConfigChange}
        backendOnline={backendOnline}
      />
    </div>
  );
}
