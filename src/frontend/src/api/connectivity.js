export const CONNECTION_STATES = Object.freeze({
  CHECKING: 'checking',
  CONNECTED: 'connected',
  DEGRADED: 'degraded',
  DISCONNECTED: 'disconnected',
  RECOVERING: 'recovering',
});

export const HEALTH_TIMING = Object.freeze({
  timeoutMs: 4000,
  startupBackoffMs: [1000, 2000, 4000, 8000],
  disconnectedPollMs: 12000,
  reachablePollMs: 45000,
  hiddenTabPollMs: 60000,
});

// ``degraded`` vẫn reachable: UI có thể tải history và để backend tự trả lỗi
// dependency cụ thể; chỉ network failure mới là disconnected.
export function isBackendReachable(connectionState) {
  return ['connected', 'degraded'].includes(connectionState?.state);
}

export function classifyHealthResult(result) {
  if (!result?.reachable) {
    return {
      state: CONNECTION_STATES.DISCONNECTED,
      message: 'Đang dùng lịch sử cục bộ vì chưa kết nối backend.',
      health: null,
      reason: result?.reason || 'network_error',
    };
  }

  if (result.state === CONNECTION_STATES.CONNECTED || result.health?.status === 'ok') {
    return {
      state: CONNECTION_STATES.CONNECTED,
      message: 'Backend đã kết nối.',
      health: result.health || null,
      reason: null,
    };
  }

  return {
    state: CONNECTION_STATES.DEGRADED,
    message: 'Backend đã kết nối nhưng một số dịch vụ chưa sẵn sàng.',
    health: result.health || null,
    reason: result.reason || result.health?.status || 'degraded',
  };
}

export function nextHealthDelayMs(connectionState, attempt, documentVisibility = 'visible') {
  // Startup dùng backoff theo attempt; trạng thái ổn định và tab ẩn poll thưa hơn
  // để tránh tạo request nền không cần thiết.
  if (documentVisibility === 'hidden') {
    return HEALTH_TIMING.hiddenTabPollMs;
  }

  if (connectionState === CONNECTION_STATES.CONNECTED || connectionState === CONNECTION_STATES.DEGRADED) {
    return HEALTH_TIMING.reachablePollMs;
  }

  return HEALTH_TIMING.startupBackoffMs[attempt] || HEALTH_TIMING.disconnectedPollMs;
}

export function shouldTreatChatErrorAsDisconnected(error) {
  return !error?.status;
}
