import { memo, useCallback, useEffect, useId, useMemo, useRef, useState } from 'react';
import { formatText } from '../utils/markdown.js';
import {
  answerModelDisplayName,
  nonGeneratedResponseDetails,
  sourceDisplayLabels,
} from '../utils/presentationMetadata.js';

function ChatMessage({ msg }) {
  const [detailsOpen, setDetailsOpen] = useState(false);
  const [copied, setCopied] = useState(false);
  const copyTimerRef = useRef(null);
  const detailPanelId = useId();
  const isUser = msg.role === 'user';
  const data = msg.data || null;
  const sourceLabels = useMemo(() => sourceDisplayLabels(data), [data]);
  const answerModelName = useMemo(() => answerModelDisplayName(data), [data]);
  const nonGeneratedDetails = useMemo(() => nonGeneratedResponseDetails(data), [data]);
  const formattedAssistantBody = useMemo(
    () => (isUser ? null : formatText(msg.content)),
    [isUser, msg.content],
  );
  const hasAnswerDetails =
    sourceLabels.length > 0 || Boolean(answerModelName) || Boolean(nonGeneratedDetails);

  useEffect(() => () => {
    if (copyTimerRef.current) clearTimeout(copyTimerRef.current);
  }, []);

  const handleCopy = useCallback(async () => {
    if (!msg.content || !navigator?.clipboard?.writeText) return;
    try {
      await navigator.clipboard.writeText(msg.content);
      setCopied(true);
      if (copyTimerRef.current) clearTimeout(copyTimerRef.current);
      copyTimerRef.current = setTimeout(() => {
        copyTimerRef.current = null;
        setCopied(false);
      }, 1800);
    } catch {
      setCopied(false);
    }
  }, [msg.content]);

  return (
    <div className={`chat-message ${isUser ? 'chat-message-user' : 'chat-message-assistant'}`}>
      <div className="chat-message-inner">
        {/* Raw user text và formatted assistant text có presentation owner riêng. */}
        <div className="chat-message-content">
          {isUser ? (
            <div className="chat-user-text">{msg.content}</div>
          ) : (
            <div className="chat-assistant-text">
              <div className="chat-formatted-text">{formattedAssistantBody}</div>

              {/* Source label dùng display metadata; raw source IDs vẫn ở response data. */}
              <div className="chat-message-extras">
                <div className="answer-toolbar-row">
                  <button
                    type="button"
                    className={`answer-copy-btn ${copied ? 'answer-copy-btn-copied' : ''}`}
                    onClick={handleCopy}
                    aria-label={copied ? 'Đã sao chép câu trả lời' : 'Sao chép câu trả lời'}
                    title={copied ? 'Đã sao chép' : 'Sao chép câu trả lời'}
                  >
                    {copied ? (
                      <svg width="13" height="13" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2.2" d="M5 13l4 4L19 7" />
                      </svg>
                    ) : (
                      <svg width="13" height="13" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                        <path
                          strokeLinecap="round"
                          strokeLinejoin="round"
                          strokeWidth="1.9"
                          d="M8 16H6a2 2 0 01-2-2V6a2 2 0 012-2h8a2 2 0 012 2v2m-6 12h8a2 2 0 002-2v-8a2 2 0 00-2-2h-8a2 2 0 00-2 2v8a2 2 0 002 2z"
                        />
                      </svg>
                    )}
                    <span>{copied ? 'Đã chép' : 'Sao chép'}</span>
                  </button>

                  {data && hasAnswerDetails && (
                    <button
                      type="button"
                      className="answer-details-toggle"
                      aria-expanded={detailsOpen}
                      aria-controls={detailPanelId}
                      onClick={() => setDetailsOpen((open) => !open)}
                    >
                      <span>Chi tiết</span>
                      {sourceLabels.length > 0 && (
                        <span className="answer-details-count">{sourceLabels.length} nguồn</span>
                      )}
                      <svg width="14" height="14" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                        <path
                          strokeLinecap="round"
                          strokeLinejoin="round"
                          strokeWidth="2"
                          d={detailsOpen ? 'M5 15l7-7 7 7' : 'M19 9l-7 7-7-7'}
                        />
                      </svg>
                    </button>
                  )}
                </div>

                {data && hasAnswerDetails && (
                  <div className="answer-details">
                    {detailsOpen && (
                      <div id={detailPanelId} className="answer-details-panel">
                        {sourceLabels.length > 0 && (
                          <div className="chat-meta-sources">
                            <span className="chat-meta-label">Tài liệu tham khảo:</span>
                            <ul className="chat-meta-source-list">
                              {sourceLabels.map((label) => (
                                <li key={label}>{label}</li>
                              ))}
                            </ul>
                          </div>
                        )}
                        {answerModelName && (
                          <div className="chat-meta-info">
                            <span title="Mô hình trả lời">{answerModelName}</span>
                          </div>
                        )}
                        {nonGeneratedDetails && (
                          <div className="chat-meta-info">
                            <span>Trạng thái: {nonGeneratedDetails.status}</span>
                            {nonGeneratedDetails.reason && (
                              <span>Lý do: {nonGeneratedDetails.reason}</span>
                            )}
                            {nonGeneratedDetails.decisionModel && (
                              <span>Model quyết định: {nonGeneratedDetails.decisionModel}</span>
                            )}
                          </div>
                        )}
                      </div>
                    )}
                  </div>
                )}
              </div>
            </div>
          )}
        </div>
      </div>
    </div>
  );
}

export default memo(ChatMessage);
