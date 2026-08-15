/**
 * Tiện ích localStorage cho session state của frontend.
 * Các key là compatibility contract để lịch sử hiện có tiếp tục đọc được.
 *
 * Key: 'acneAdvisorSessions' — JSON array of session objects.
 * Key: 'acneAdvisorActiveSession' — string ID of the active session.
 */

const SESSIONS_KEY = 'acneAdvisorSessions';
const ACTIVE_SESSION_KEY = 'acneAdvisorActiveSession';
const HISTORY_HIDDEN_KEY = 'acneAdvisorHistoryHidden';

/**
 * Load sessions và điền các field tùy chọn còn thiếu.
 * @returns {Array} Array of session objects.
 */
export function loadSessions() {
  try {
    const saved = localStorage.getItem(SESSIONS_KEY);
    if (!saved) return [];
    const parsed = JSON.parse(saved);
    // Default field được bổ sung khi đọc, không rewrite storage ngay tại đây.
    return parsed.map((s) => ({
      ...s,
      hidden: s.hidden || false,
      updatedAt: s.updatedAt || s.createdAt || Date.now(),
    }));
  } catch {
    return [];
  }
}

/**
 * Save sessions array to localStorage.
 * @param {Array} sessions
 */
export function saveSessions(sessions) {
  localStorage.setItem(SESSIONS_KEY, JSON.stringify(sessions));
}

/**
 * Load the active session ID from localStorage.
 * @returns {string|null}
 */
export function loadActiveSessionId() {
  return localStorage.getItem(ACTIVE_SESSION_KEY) || null;
}

/**
 * Save the active session ID to localStorage.
 * @param {string|null} id
 */
export function saveActiveSessionId(id) {
  if (id) {
    localStorage.setItem(ACTIVE_SESSION_KEY, id);
  } else {
    localStorage.removeItem(ACTIVE_SESSION_KEY);
  }
}

export function loadHistoryHiddenAt() {
  const rawValue = localStorage.getItem(HISTORY_HIDDEN_KEY);
  if (!rawValue) return null;

  // Boolean flag vẫn được hiểu như thời điểm ẩn hiện tại để giữ compatibility.
  if (rawValue === 'true') return Date.now();

  const parsed = Number(rawValue);
  return Number.isFinite(parsed) ? parsed : null;
}

export function saveHistoryHiddenAt(hiddenAt) {
  if (hiddenAt) {
    localStorage.setItem(HISTORY_HIDDEN_KEY, String(hiddenAt));
  } else {
    localStorage.removeItem(HISTORY_HIDDEN_KEY);
  }
}

/**
 * Clear local chat sessions and active session selection.
 */
export function clearLocalChatCache() {
  localStorage.removeItem(SESSIONS_KEY);
  localStorage.removeItem(ACTIVE_SESSION_KEY);
  localStorage.removeItem(HISTORY_HIDDEN_KEY);
  sessionStorage.removeItem(SESSIONS_KEY);
  sessionStorage.removeItem(ACTIVE_SESSION_KEY);
  sessionStorage.removeItem(HISTORY_HIDDEN_KEY);
}

/**
 * Generate a random session ID.
 * @returns {string}
 */
export function generateId() {
  return Math.random().toString(36).substring(2, 11);
}
