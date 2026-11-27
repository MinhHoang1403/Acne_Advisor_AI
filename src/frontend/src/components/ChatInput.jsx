import { useRef, useEffect, useState } from 'react';
import ModelSelector from './ModelSelector.jsx';

const MAX_MESSAGE_CHARS = 500;

export default function ChatInput({
  message,
  setMessage,
  isLoading,
  onSubmit,
  onCancel,
  modelConfig: externalModelConfig = null,
  onModelConfigChange: externalOnModelConfigChange = null,
  backendOnline = true,
}) {
  const textareaRef = useRef(null);
  const [internalModelConfig, setInternalModelConfig] = useState(null);
  const modelConfig = externalModelConfig ?? internalModelConfig;
  const handleModelConfigChange = externalOnModelConfigChange || setInternalModelConfig;

  // Resize chỉ ảnh hưởng presentation và bị chặn ở 200px để composer ổn định.
  useEffect(() => {
    const el = textareaRef.current;
    if (!el) return;
    if (!message) {
      el.style.height = '';
      return;
    }
    el.style.height = 'auto';
    el.style.height = `${Math.min(el.scrollHeight, 160)}px`;
  }, [message]);

  const handleKeyDown = (e) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      if (!isLoading && message.trim() && message.length <= MAX_MESSAGE_CHARS) {
        onSubmit(message, modelConfig);
      }
    }
  };

  const handleFormSubmit = (e) => {
    e.preventDefault();
    if (!isLoading && message.trim() && message.length <= MAX_MESSAGE_CHARS) {
      onSubmit(message, modelConfig);
    }
  };

  const isOverLimit = message.length > MAX_MESSAGE_CHARS;
  const isNearLimit = !isOverLimit && message.length >= 440;
  const isInvalid = isOverLimit || !message.trim();

  return (
    <div className="chat-input-wrapper">
      <div className="chat-input-container">
        <form onSubmit={handleFormSubmit} className="chat-input-form">
          <div className="chat-input-actions chat-composer-row">
            <textarea
              ref={textareaRef}
              value={message}
              onChange={(e) => setMessage(e.target.value)}
              onKeyDown={handleKeyDown}
              placeholder={isLoading ? 'Đang xử lý câu hỏi hiện tại...' : 'Hỏi về tình trạng mụn của bạn...'}
              aria-label="Nhập câu hỏi về tình trạng mụn"
              disabled={isLoading}
              className={`chat-textarea ${isOverLimit ? 'chat-textarea-error' : ''}`}
              rows="1"
            />
            <ModelSelector
              onModelConfigChange={handleModelConfigChange}
              backendOnline={backendOnline}
            />
            <div className="chat-input-right-actions">
              <div
                className={`chat-char-counter ${
                  isOverLimit ? 'chat-char-counter-error' : isNearLimit ? 'chat-char-counter-warn' : ''
                }`}
              >
                {message.length} / {MAX_MESSAGE_CHARS}
              </div>
              {isLoading && onCancel ? (
                <button
                  type="button"
                  onClick={onCancel}
                  className="chat-send-btn chat-stop-btn"
                  aria-label="Dừng tạo phản hồi"
                  title="Dừng tạo phản hồi"
                >
                  <svg width="14" height="14" viewBox="0 0 24 24" fill="currentColor">
                    <rect x="6" y="6" width="12" height="12" rx="2.5" />
                  </svg>
                </button>
              ) : (
                <button
                  type="submit"
                  disabled={isLoading || isInvalid}
                  className={`chat-send-btn ${!isInvalid && !isLoading ? 'chat-send-btn-active' : ''}`}
                  aria-label="Gửi câu hỏi"
                  title="Gửi câu hỏi"
                >
                  <svg width="20" height="20" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                    <path
                      strokeLinecap="round"
                      strokeLinejoin="round"
                      strokeWidth="2"
                      d="M5 10l7-7m0 0l7 7m-7-7v18"
                    />
                  </svg>
                </button>
              )}
            </div>
          </div>
          {isOverLimit && (
            <div className="chat-input-error-msg">
              Câu hỏi hơi dài. Vui lòng rút gọn dưới 500 ký tự hoặc tách thành nhiều câu nhỏ.
            </div>
          )}
        </form>
      </div>
    </div>
  );
}
