import assert from 'node:assert/strict';
import test from 'node:test';

import { DEFAULT_API_BASE_URL, buildApiUrl, resolveApiBaseUrl } from './api.js';

test('API base URL defaults to local 127.0.0.1 backend', () => {
  assert.equal(resolveApiBaseUrl(''), DEFAULT_API_BASE_URL);
  assert.equal(DEFAULT_API_BASE_URL, 'http://127.0.0.1:8000');
});

test('API base URL trims trailing slash and endpoints avoid double slash', () => {
  const base = resolveApiBaseUrl(' http://localhost:8000/// ');

  assert.equal(base, 'http://localhost:8000');
  assert.equal(buildApiUrl('/health', base), 'http://localhost:8000/health');
  assert.equal(buildApiUrl('chat', base), 'http://localhost:8000/chat');
  assert.equal(buildApiUrl('/models', 'http://127.0.0.1:8000/'), 'http://127.0.0.1:8000/models');
});

test('API base URL rejects unsafe schemes', () => {
  assert.throws(() => resolveApiBaseUrl('javascript:alert(1)'), /Invalid VITE_API_URL protocol/);
  assert.throws(() => resolveApiBaseUrl('file:///tmp/api'), /Invalid VITE_API_URL protocol/);
});

test('same-origin API prefix works without browser globals', () => {
  assert.equal(resolveApiBaseUrl('/api/'), '/api');
  assert.equal(buildApiUrl('/chat', '/api'), '/api/chat');
  assert.equal(buildApiUrl('/chat/sessions?include_hidden=true', '/api'), '/api/chat/sessions?include_hidden=true');
  assert.throws(() => resolveApiBaseUrl('//external.example/api'), /Invalid VITE_API_URL/);
});
