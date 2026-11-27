import { API_BASE_URL, buildApiUrl } from '../config/api.js';

export { API_BASE_URL };

// Module này sở hữu HTTP timeout, JSON contract và error mapping. Nó không giữ
// React/session state; App quyết định cách hiển thị và phục hồi sau lỗi.
const STATUS_MESSAGES = {
  429: 'Hệ thống đang nhận quá nhiều yêu cầu. Vui lòng thử lại sau.',
  500: 'Backend không thể xử lý yêu cầu. Vui lòng thử lại.',
  503: 'Dịch vụ AI tạm thời chưa sẵn sàng. Vui lòng thử lại sau.',
  504: 'Yêu cầu xử lý quá thời gian. Vui lòng thử lại hoặc chọn mô hình khác.',
};

const API_TIMEOUT_MS = 10000;
const CHAT_TIMEOUT_MS = 225000;

async function fetchWithTimeout(url, options = {}, timeoutMs = API_TIMEOUT_MS, consume = (response) => response) {
  // Signal nội bộ áp deadline; parent signal vẫn có thể hủy khi component unmount.
  const controller = new AbortController();
  const parentSignal = options.signal;
  const timeoutId = setTimeout(() => controller.abort('timeout'), timeoutMs);
  const abortFromParent = () => controller.abort(parentSignal.reason);

  if (parentSignal) {
    if (parentSignal.aborted) controller.abort(parentSignal.reason);
    else parentSignal.addEventListener('abort', abortFromParent, { once: true });
  }

  try {
    const response = await fetch(url, { ...options, signal: controller.signal });
    return await consume(response);
  } catch (error) {
    if (controller.signal.aborted && !parentSignal?.aborted) {
      const timeoutError = new Error(`Request exceeded ${timeoutMs}ms.`);
      timeoutError.name = 'TimeoutError';
      throw timeoutError;
    }
    throw error;
  } finally {
    clearTimeout(timeoutId);
    parentSignal?.removeEventListener('abort', abortFromParent);
  }
}

async function fetchJson(url, options = {}, timeoutMs = API_TIMEOUT_MS) {
  return fetchWithTimeout(url, options, timeoutMs, async (response) => {
    if (!response.ok) throw await parseApiError(response);
    try {
      return await response.json();
    } catch (cause) {
      if (cause?.name === 'AbortError') throw cause;
      const error = new Error('Backend trả về phản hồi không hợp lệ. Vui lòng thử lại.');
      error.status = 502;
      error.code = 'invalid_response';
      error.cause = cause;
      throw error;
    }
  });
}

export async function parseApiError(response, fallbackPrefix = 'Lỗi server') {
  let detail;
  try {
    const data = await response.clone().json();
    detail = data?.detail || data;
  } catch {
    detail = null;
  }

  const backendMessage = typeof detail === 'object' && detail?.message ? detail.message : null;
  let message = STATUS_MESSAGES[response.status] || `${fallbackPrefix}: ${response.status}`;
  if ((response.status === 400 || response.status === 409 || response.status === 422) && backendMessage) {
    message = backendMessage;
  } else if (response.status === 409) {
    message = 'Yêu cầu trước đang được xử lý hoặc xung đột trạng thái. Vui lòng thử lại sau giây lát.';
  } else if (response.status === 400 || response.status === 422) {
    message = 'Yêu cầu không hợp lệ. Vui lòng kiểm tra nội dung và thử lại.';
  }
  const error = new Error(message);
  error.status = response.status;
  if (typeof detail === 'object' && detail) {
    error.code = detail.code;
    error.retryable = detail.retryable;
    error.errorType = detail.error_type;
  }
  return error;
}

function normalizeListResponse(data) {
  if (Array.isArray(data)) return data;
  if (data && Array.isArray(data.value)) return data.value;
  if (data && Array.isArray(data.Value)) return data.Value;
  return [];
}

/**
 * Gửi một chat request có timeout dài hơn health/session operations.
 * @param {Object} params
 * @param {string} params.message - The user's message text.
 * @param {string|null} params.sessionId - Current session ID.
 * @param {Array<{role: string, content: string}>} params.conversationHistory - Recent conversation history (max 6).
 * @returns {Promise<Object>} The API response data (includes session_id).
 */
export async function sendChatMessage({
  message,
  sessionId,
  conversationHistory = [],
  llmProvider,
  llmModel,
  allowModelFallback,
  bypassCache = false,
  requestId = globalThis.crypto.randomUUID(),
  signal,
  timeoutMs = CHAT_TIMEOUT_MS,
}) {
  try {
    return await fetchJson(buildApiUrl('/chat'), {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      signal,
      body: JSON.stringify({
        request_id: requestId,
        message,
        user_id: null,
        session_id: sessionId,
        conversation_history: conversationHistory,
        llm_provider: llmProvider,
        llm_model: llmModel,
        allow_model_fallback: allowModelFallback,
        bypass_cache: bypassCache,
      }),
    }, timeoutMs);
  } catch (err) {
    if (err?.status) throw err;
    if (signal?.aborted) {
      const error = new Error('Yêu cầu đã được hủy.');
      error.isCanceled = true;
      throw error;
    }
    const timedOut = err?.name === 'TimeoutError';
    const error = new Error(
      timedOut
        ? 'Yêu cầu xử lý quá thời gian. Vui lòng thử lại hoặc chọn mô hình khác.'
        : `Không thể kết nối tới backend. Hãy kiểm tra FastAPI tại ${API_BASE_URL}.`,
    );
    error.cause = err;
    error.status = timedOut ? 504 : null;
    error.isNetworkError = !timedOut;
    error.isTimeout = timedOut;
    throw error;
  }

}

function parseSseBlocks(buffer) {
  const normalized = buffer.replace(/\r\n/g, '\n');
  const blocks = normalized.split('\n\n');
  const remainder = blocks.pop() || '';
  const events = [];

  for (const block of blocks) {
    if (!block.trim()) continue;
    let eventName = 'message';
    const dataLines = [];
    for (const line of block.split('\n')) {
      if (line.startsWith('event:')) {
        eventName = line.slice(6).trim();
      } else if (line.startsWith('data:')) {
        dataLines.push(line.slice(5).trimStart());
      }
    }
    if (dataLines.length === 0) continue;
    try {
      events.push({ event: eventName, data: JSON.parse(dataLines.join('\n')) });
    } catch {
      // Bỏ qua frame rỗng hoặc comment keep-alive không phải JSON.
    }
  }
  return { events, remainder };
}

/**
 * Gửi chat request qua SSE (`/chat/stream`) để hiển thị tiến độ và từng cụm từ thời gian thực,
 * tự động fallback sang JSON nếu môi trường mock/proxy không trả `text/event-stream`.
 */
export async function streamChatMessage({
  message,
  sessionId,
  conversationHistory = [],
  llmProvider,
  llmModel,
  allowModelFallback,
  bypassCache = false,
  requestId = globalThis.crypto.randomUUID(),
  signal,
  timeoutMs = CHAT_TIMEOUT_MS,
  onStage,
  onDelta,
  onReset,
}) {
  const payloadBody = JSON.stringify({
    request_id: requestId,
    message,
    user_id: null,
    session_id: sessionId,
    conversation_history: conversationHistory,
    llm_provider: llmProvider,
    llm_model: llmModel,
    allow_model_fallback: allowModelFallback,
    bypass_cache: bypassCache,
  });

  try {
    return await fetchWithTimeout(
      buildApiUrl('/chat/stream'),
      {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          Accept: 'text/event-stream, application/json',
        },
        signal,
        body: payloadBody,
      },
      timeoutMs,
      async (response) => {
        if (!response.ok) throw await parseApiError(response);

        const contentType = response.headers.get('content-type') || '';
        if (!contentType.includes('text/event-stream') || !response.body) {
          return await response.json();
        }

        const reader = response.body.getReader();
        const decoder = new TextDecoder('utf-8');
        let buffer = '';
        let completedPayload = null;

        const handleFrame = (frame) => {
          if (frame.event === 'stage') {
            onStage?.(frame.data);
          } else if (frame.event === 'delta' && typeof frame.data?.delta === 'string') {
            onDelta?.(frame.data.delta);
          } else if (frame.event === 'reset') {
            onReset?.();
          } else if (frame.event === 'complete') {
            completedPayload = frame.data;
          } else if (frame.event === 'error') {
            const status = Number(frame.data?.status) || 500;
            const detailMsg = frame.data?.detail?.message;
            const err = new Error(
              (status === 400 || status === 409 || status === 422) && detailMsg
                ? detailMsg
                : STATUS_MESSAGES[status] || detailMsg || 'Backend không thể xử lý yêu cầu. Vui lòng thử lại.',
            );
            err.status = status;
            err.code = frame.data?.detail?.code;
            throw err;
          }
        };

        try {
          while (true) {
            const { value, done } = await reader.read();
            if (done) break;
            buffer += decoder.decode(value, { stream: true });
            const parsed = parseSseBlocks(buffer);
            buffer = parsed.remainder;

            for (const frame of parsed.events) {
              handleFrame(frame);
            }
          }
        } finally {
          try {
            reader.releaseLock();
          } catch {
            // Reader có thể đã đóng khi stream kết thúc hoặc bị hủy.
          }
        }

        buffer += decoder.decode();
        if (buffer.trim()) {
          const tail = parseSseBlocks(`${buffer}\n\n`);
          for (const frame of tail.events) {
            handleFrame(frame);
          }
        }

        if (!completedPayload) {
          const err = new Error('Luồng phản hồi bị ngắt trước khi hoàn tất. Vui lòng thử lại.');
          err.status = 502;
          err.code = 'incomplete_stream';
          throw err;
        }
        return completedPayload;
      },
    );
  } catch (err) {
    if (err?.status) throw err;
    if (signal?.aborted) {
      const error = new Error('Yêu cầu đã được hủy.');
      error.isCanceled = true;
      throw error;
    }
    const timedOut = err?.name === 'TimeoutError';
    const error = new Error(
      timedOut
        ? 'Yêu cầu xử lý quá thời gian. Vui lòng thử lại hoặc chọn mô hình khác.'
        : `Không thể kết nối tới backend. Hãy kiểm tra FastAPI tại ${API_BASE_URL}.`,
    );
    error.cause = err;
    error.status = timedOut ? 504 : null;
    error.isNetworkError = !timedOut;
    error.isTimeout = timedOut;
    throw error;
  }
}

/**
 * Kiểm tra backend reachable và trả raw health state cho connectivity classifier.
 * @returns {Promise<{state: string, reachable: boolean, health: Object|null, reason: string|null}>}
 */
export async function checkBackendHealth({ timeoutMs = 4000, signal } = {}) {
  try {
    return await fetchWithTimeout(buildApiUrl('/health'), {
      method: 'GET',
      signal,
    }, timeoutMs, async (response) => {
      const health = await response.json().catch((error) => {
        if (error?.name === 'AbortError') throw error;
        return null;
      });

      if (!response.ok) {
        return {
          state: 'degraded',
          reachable: true,
          health,
          reason: `health_http_${response.status}`,
        };
      }

      if (health?.status === 'ok') {
        return { state: 'connected', reachable: true, health, reason: null };
      }

      return {
        state: 'degraded',
        reachable: true,
        health,
        reason: health?.status || 'health_degraded',
      };
    });
  } catch (err) {
    const timedOut = err?.name === 'AbortError' || err?.name === 'TimeoutError';
    return {
      state: 'disconnected',
      reachable: false,
      health: null,
      reason: timedOut ? 'health_timeout' : 'network_error',
      error: err,
    };
  }
}

/**
 * Lấy danh sách model được backend công bố.
 * @returns {Promise<Object>}
 */
export async function fetchModels() {
  return fetchJson(buildApiUrl('/models'));
}

/**
 * Lấy chat session summaries từ backend.
 * @param {string|null} userId
 * @param {boolean} includeHidden
 * @returns {Promise<Array>}
 */
export async function fetchSessions(userId = null, includeHidden = false) {
  const params = new URLSearchParams();
  if (userId) params.set('user_id', userId);
  if (includeHidden) params.set('include_hidden', 'true');

  const data = await fetchJson(buildApiUrl(`/chat/sessions?${params.toString()}`));
  return normalizeListResponse(data);
}

/**
 * Lấy messages của một session cụ thể.
 * @param {string} sessionId
 * @returns {Promise<Array>}
 */
export async function fetchMessages(sessionId) {
  const messages = [];
  const pageSize = 200;
  while (true) {
    const data = await fetchJson(buildApiUrl(`/chat/sessions/${sessionId}/messages?limit=${pageSize}&offset=${messages.length}`));
    const page = normalizeListResponse(data);
    messages.push(...page);
    if (page.length < pageSize) return messages;
  }
}

/**
 * Đổi tên session ở backend.
 * @param {string} sessionId
 * @param {string} title
 * @returns {Promise<Object>}
 */
export async function renameSession(sessionId, title) {
  return fetchJson(buildApiUrl(`/chat/sessions/${sessionId}/rename`), {
    method: 'PATCH',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ title }),
  });
}

/**
 * Ẩn session bằng flag ở backend, không xóa dữ liệu.
 * @param {string} sessionId
 * @returns {Promise<Object>}
 */
export async function hideSession(sessionId) {
  return fetchJson(buildApiUrl(`/chat/sessions/${sessionId}/hide`), {
    method: 'PATCH',
  });
}

/**
 * Xóa chat history persisted và answer cache do ứng dụng sở hữu.
 * @returns {Promise<Object>} deletion counts
 */
export async function deleteAllChatSessions() {
  return fetchJson(buildApiUrl('/chat/sessions'), {
    method: 'DELETE',
  });
}
