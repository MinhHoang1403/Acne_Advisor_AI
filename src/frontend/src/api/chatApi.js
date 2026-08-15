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

async function fetchWithTimeout(url, options = {}, timeoutMs = API_TIMEOUT_MS) {
  // Signal nội bộ áp deadline; parent signal vẫn có thể hủy khi component unmount.
  const controller = new AbortController();
  const parentSignal = options.signal;
  const timeoutId = setTimeout(() => controller.abort('timeout'), timeoutMs);

  if (parentSignal) {
    if (parentSignal.aborted) controller.abort(parentSignal.reason);
    else parentSignal.addEventListener('abort', () => controller.abort(parentSignal.reason), { once: true });
  }

  try {
    return await fetch(url, { ...options, signal: controller.signal });
  } catch (error) {
    if (controller.signal.aborted && !parentSignal?.aborted) {
      const timeoutError = new Error(`Request exceeded ${timeoutMs}ms.`);
      timeoutError.name = 'TimeoutError';
      throw timeoutError;
    }
    throw error;
  } finally {
    clearTimeout(timeoutId);
  }
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
  if ((response.status === 400 || response.status === 422) && backendMessage) {
    message = backendMessage;
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
  timeoutMs = CHAT_TIMEOUT_MS,
}) {
  let response;
  try {
    response = await fetchWithTimeout(buildApiUrl('/chat'), {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
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

  if (!response.ok) {
    throw await parseApiError(response);
  }

  return response.json();
}

/**
 * Kiểm tra backend reachable và trả raw health state cho connectivity classifier.
 * @returns {Promise<{state: string, reachable: boolean, health: Object|null, reason: string|null}>}
 */
export async function checkBackendHealth({ timeoutMs = 4000, signal } = {}) {
  const controller = new AbortController();
  const timeoutId = setTimeout(() => controller.abort(), timeoutMs);

  if (signal) {
    if (signal.aborted) {
      controller.abort();
    } else {
      signal.addEventListener('abort', () => controller.abort(), { once: true });
    }
  }

  try {
    const response = await fetch(buildApiUrl('/health'), {
      method: 'GET',
      signal: controller.signal,
    });
    const health = await response.json().catch(() => null);

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
  } catch (err) {
    const timedOut = err?.name === 'AbortError';
    return {
      state: 'disconnected',
      reachable: false,
      health: null,
      reason: timedOut ? 'health_timeout' : 'network_error',
      error: err,
    };
  } finally {
    clearTimeout(timeoutId);
  }
}

/**
 * Lấy danh sách model được backend công bố.
 * @returns {Promise<Object>}
 */
export async function fetchModels() {
  const response = await fetchWithTimeout(buildApiUrl('/models'));
  if (!response.ok) throw await parseApiError(response);
  return response.json();
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

  const response = await fetchWithTimeout(buildApiUrl(`/chat/sessions?${params.toString()}`));
  if (!response.ok) throw await parseApiError(response);
  const data = await response.json();
  return normalizeListResponse(data);
}

/**
 * Lấy messages của một session cụ thể.
 * @param {string} sessionId
 * @returns {Promise<Array>}
 */
export async function fetchMessages(sessionId) {
  const response = await fetchWithTimeout(buildApiUrl(`/chat/sessions/${sessionId}/messages`));
  if (!response.ok) throw await parseApiError(response);
  const data = await response.json();
  return normalizeListResponse(data);
}

/**
 * Đổi tên session ở backend.
 * @param {string} sessionId
 * @param {string} title
 * @returns {Promise<Object>}
 */
export async function renameSession(sessionId, title) {
  const response = await fetchWithTimeout(buildApiUrl(`/chat/sessions/${sessionId}/rename`), {
    method: 'PATCH',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ title }),
  });
  if (!response.ok) throw await parseApiError(response);
  return response.json();
}

/**
 * Ẩn session bằng flag ở backend, không xóa dữ liệu.
 * @param {string} sessionId
 * @returns {Promise<Object>}
 */
export async function hideSession(sessionId) {
  const response = await fetchWithTimeout(buildApiUrl(`/chat/sessions/${sessionId}/hide`), {
    method: 'PATCH',
  });
  if (!response.ok) throw await parseApiError(response);
  return response.json();
}

/**
 * Xóa chat history persisted và answer cache do ứng dụng sở hữu.
 * @returns {Promise<Object>} deletion counts
 */
export async function deleteAllChatSessions() {
  const response = await fetchWithTimeout(buildApiUrl('/chat/sessions'), {
    method: 'DELETE',
  });
  if (!response.ok) throw await parseApiError(response);
  return response.json();
}
