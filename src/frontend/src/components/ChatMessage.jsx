import { useId, useState } from 'react';
import { formatText } from '../utils/markdown.js';
import { answerModelDisplayName, sourceDisplayLabels } from '../utils/presentationMetadata.js';

export default function ChatMessage({ msg }) {
  const [detailsOpen, setDetailsOpen] = useState(false);
  const detailPanelId = useId();
  const isUser = msg.role === 'user';
  const data = msg.data || null;
  const sourceLabels = sourceDisplayLabels(data);
  const answerModelName = answerModelDisplayName(data);
  const hasAnswerDetails = sourceLabels.length > 0 || Boolean(answerModelName);

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
                              <span className="chat-meta-label">Tài liệu tham khảo</span>
                              <ul className="chat-meta-source-list">
                                {sourceLabels.map((label) => (
                                  <li key={label}>{label}</li>
                                ))}
                              </ul>
                            </div>
                          )}
                          {answerModelName && (
                            <div className="chat-meta-info">
                              <span title="Mô hình trả lời">✨ {answerModelName}</span>
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
