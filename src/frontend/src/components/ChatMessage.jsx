import { useId, useState } from 'react';
import { formatText } from '../utils/markdown.js';
import { responseBadgeLabel, sourceDisplayLabels } from '../utils/presentationMetadata.js';

export default function ChatMessage({ msg }) {
  const [detailsOpen, setDetailsOpen] = useState(false);
  const detailPanelId = useId();
  const isUser = msg.role === 'user';
  const data = msg.data || null;
  const sourceLabels = sourceDisplayLabels(data);
  const hasMetadata = Boolean(data?.metadata && Object.keys(data.metadata).length > 0);
  const hasAnswerDetails = Boolean(data && (sourceLabels.length > 0 || hasMetadata));

  return (
    <div className={`chat-message ${isUser ? 'chat-message-user' : 'chat-message-assistant'}`}>
      <div className="chat-message-inner">
        {/* Raw user text và formatted assistant text có presentation owner riêng. */}
        <div className="chat-message-content">
          {isUser ? (
            <div className="chat-user-text">{msg.content}</div>
          ) : (
            <div className="chat-assistant-text">
              <div className="chat-formatted-text">{formatText(msg.content)}</div>

              {/* Source label dùng display metadata; raw source IDs vẫn ở response data. */}
              {data && (
                <div className="chat-message-extras">
                  {hasAnswerDetails && (
                    <div className="answer-details">
                      <button
                        type="button"
                        className="answer-details-toggle"
                        aria-expanded={detailsOpen}
                        aria-controls={detailPanelId}
                        onClick={() => setDetailsOpen((open) => !open)}
                      >
                        <span>Chi tiết</span>
                        <svg width="14" height="14" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                          <path
                            strokeLinecap="round"
                            strokeLinejoin="round"
                            strokeWidth="2"
                            d={detailsOpen ? 'M5 15l7-7 7 7' : 'M19 9l-7 7-7-7'}
                          />
                        </svg>
                      </button>

                      {detailsOpen && (
                        <div id={detailPanelId} className="answer-details-panel">
                          {sourceLabels.length > 0 && (
                            <div className="chat-meta-sources">
                              <span className="chat-meta-label">Nguồn đã truy hồi:</span>
                              <ul className="chat-meta-source-list">
                                {sourceLabels.map((label) => (
                                  <li key={label}>{label}</li>
                                ))}
                              </ul>
                            </div>
                          )}
                          {hasMetadata && (
                            <div className="chat-meta-info">
                              <span title="Mô hình ngôn ngữ">
                                {responseBadgeLabel(data.metadata)}
                              </span>
                              {data.metadata.retrieval && (
                                <span title="Phương pháp truy xuất">🔍 {data.metadata.retrieval}</span>
                              )}
                            </div>
                          )}
                        </div>
                      )}
                    </div>
                  )}
                </div>
              )}
            </div>
          )}
        </div>
      </div>
    </div>
  );
}
