import assert from 'node:assert/strict';
import fs from 'node:fs';
import test from 'node:test';

import { checkBackendHealth, fetchModels, parseApiError, sendChatMessage } from './chatApi.js';

function jsonResponse(status, body) {
  return new Response(JSON.stringify(body), {
    status,
    headers: { 'Content-Type': 'application/json' },
  });
}

test('parseApiError maps structured HTTP errors safely', async () => {
  const badRequest = await parseApiError(jsonResponse(400, { detail: { message: 'Nội dung không hợp lệ.' } }));
  const validation = await parseApiError(jsonResponse(422, { detail: { message: 'Sai schema.' } }));
  const rateLimited = await parseApiError(jsonResponse(429, { detail: { message: 'raw backend rate' } }));
  const server = await parseApiError(jsonResponse(500, { detail: { message: 'raw backend server' } }));
  const unavailable = await parseApiError(jsonResponse(503, { detail: { message: 'raw backend unavailable' } }));
  const timeout = await parseApiError(jsonResponse(504, { detail: { message: 'raw backend timeout' } }));

  assert.equal(badRequest.message, 'Nội dung không hợp lệ.');
  assert.equal(validation.message, 'Sai schema.');
  assert.equal(rateLimited.message, 'Hệ thống đang nhận quá nhiều yêu cầu. Vui lòng thử lại sau.');
  assert.equal(server.message, 'Backend không thể xử lý yêu cầu. Vui lòng thử lại.');
  assert.equal(unavailable.message, 'Dịch vụ AI tạm thời chưa sẵn sàng. Vui lòng thử lại sau.');
  assert.equal(timeout.message, 'Yêu cầu xử lý quá thời gian. Vui lòng thử lại hoặc chọn mô hình khác.');
});

test('sendChatMessage distinguishes network failure from HTTP errors', async (t) => {
  const originalFetch = globalThis.fetch;
  t.after(() => {
    globalThis.fetch = originalFetch;
  });

  globalThis.fetch = async () => {
    throw new TypeError('network down');
  };

  await assert.rejects(
    () => sendChatMessage({ message: 'mụn', sessionId: null }),
    (error) => {
      assert.equal(error.status, null);
      assert.match(error.message, /Không thể kết nối tới backend/);
      return true;
    },
  );
});

test('sendChatMessage aborts after its finite client timeout', async (t) => {
  const originalFetch = globalThis.fetch;
  t.after(() => {
    globalThis.fetch = originalFetch;
  });

  globalThis.fetch = async (_url, init) =>
    new Promise((_resolve, reject) => {
      init.signal.addEventListener('abort', () => {
        const error = new Error('aborted');
        error.name = 'AbortError';
        reject(error);
      });
    });

  await assert.rejects(
    () => sendChatMessage({ message: 'mụn', sessionId: null, timeoutMs: 1 }),
    (error) => {
      assert.equal(error.status, 504);
      assert.equal(error.isTimeout, true);
      assert.match(error.message, /quá thời gian/);
      return true;
    },
  );
});

test('checkBackendHealth returns degraded for reachable degraded backend', async (t) => {
  const originalFetch = globalThis.fetch;
  t.after(() => {
    globalThis.fetch = originalFetch;
  });

  globalThis.fetch = async (url) => {
    assert.match(String(url), /\/health$/);
    return jsonResponse(200, {
      status: 'degraded',
      checks: { qdrant: { status: 'unavailable' } },
    });
  };

  const health = await checkBackendHealth({ timeoutMs: 50 });

  assert.equal(health.reachable, true);
  assert.equal(health.state, 'degraded');
  assert.equal(health.health.status, 'degraded');
});

test('checkBackendHealth returns disconnected for timeout or network failure', async (t) => {
  const originalFetch = globalThis.fetch;
  t.after(() => {
    globalThis.fetch = originalFetch;
  });

  globalThis.fetch = async (_url, init) =>
    new Promise((_resolve, reject) => {
      init.signal.addEventListener('abort', () => {
        const error = new Error('aborted');
        error.name = 'AbortError';
        reject(error);
      });
    });

  const health = await checkBackendHealth({ timeoutMs: 1 });

  assert.equal(health.reachable, false);
  assert.equal(health.state, 'disconnected');
  assert.equal(health.reason, 'health_timeout');
});

test('sendChatMessage sends selected provider and model id', async (t) => {
  const originalFetch = globalThis.fetch;
  t.after(() => {
    globalThis.fetch = originalFetch;
  });

  let capturedBody = null;
  globalThis.fetch = async (_url, init) => {
    capturedBody = JSON.parse(init.body);
    return jsonResponse(200, { answer: 'ok' });
  };

  await sendChatMessage({
    message: 'Bốn cơ chế chính gây mụn là gì?',
    sessionId: 'session-1',
    llmProvider: 'gemini',
    llmModel: 'gemini-3.5-flash-lite',
    allowModelFallback: true,
  });

  assert.equal(capturedBody.llm_provider, 'gemini');
  assert.equal(capturedBody.llm_model, 'gemini-3.5-flash-lite');
  assert.equal(capturedBody.allow_model_fallback, true);
});

test('fetchModels accepts Gemini 3.1 Flash-Lite catalog entries', async (t) => {
  const originalFetch = globalThis.fetch;
  t.after(() => {
    globalThis.fetch = originalFetch;
  });

  globalThis.fetch = async () =>
    jsonResponse(200, {
      default_provider: 'gemini',
      default_model: 'gemini-3.5-flash-lite',
      models: [
        {
          provider: 'gemini',
          model_id: 'gemini-3.5-flash-lite',
          display_name: 'Gemini 3.5 Flash-Lite',
          available: true,
          is_default: true,
        },
        {
          provider: 'gemini',
          model_id: 'gemini-3.1-flash-lite',
          display_name: 'Gemini 3.1 Flash-Lite',
          available: true,
          is_default: false,
        },
      ],
    });

  const catalog = await fetchModels();
  const flashLite = catalog.models.find((model) => model.model_id === 'gemini-3.1-flash-lite');
  assert.equal(flashLite.display_name, 'Gemini 3.1 Flash-Lite');
  assert.equal(flashLite.provider, 'gemini');
});

test('health, models, and chat use the same canonical API base', async (t) => {
  const originalFetch = globalThis.fetch;
  t.after(() => {
    globalThis.fetch = originalFetch;
  });

  const urls = [];
  globalThis.fetch = async (url, init = {}) => {
    urls.push(String(url));
    if (String(url).endsWith('/chat')) {
      assert.equal(init.method, 'POST');
      return jsonResponse(200, { answer: 'ok' });
    }
    if (String(url).endsWith('/health')) {
      return jsonResponse(200, { status: 'ok' });
    }
    return jsonResponse(200, { models: [] });
  };

  await checkBackendHealth({ timeoutMs: 50 });
  await fetchModels();
  await sendChatMessage({ message: 'mụn', sessionId: null });

  assert.deepEqual(urls, [
    'http://127.0.0.1:8000/health',
    'http://127.0.0.1:8000/models',
    'http://127.0.0.1:8000/chat',
  ]);
});

test('frontend model selector keeps fallback routing details internal', () => {
  const selectorSource = fs.readFileSync(new URL('../components/ModelSelector.jsx', import.meta.url), 'utf8');
  const messageSource = fs.readFileSync(new URL('../components/ChatMessage.jsx', import.meta.url), 'utf8');

  assert.match(selectorSource, /acneAdvisorSelectedModel/);
  assert.match(selectorSource, /data\.default_provider/);
  assert.match(selectorSource, /m\.is_default/);
  assert.match(messageSource, /answerModelDisplayName/);
  assert.doesNotMatch(messageSource, /responseBadgeLabel|requested_provider|requested_model|dự phòng từ/);
});
