import assert from 'node:assert/strict';
import test from 'node:test';
import { prepareChatTurn } from './chatRequest.js';

test('retry reuses request identity and the history before the failed user message', () => {
  const before = [{ role: 'assistant', content: 'Previous answer' }];
  const first = prepareChatTurn({ sessionId: 'a', message: 'Question', messages: before });
  const messages = [...before, { role: 'user', content: 'Question', pendingRequest: first }];
  const retry = prepareChatTurn({ sessionId: 'a', message: 'Question', messages });
  assert.equal(retry.requestId, first.requestId);
  assert.equal(retry.isRetry, true);
  assert.deepEqual(retry.history, before);
  assert.match(first.requestId, /^[a-f0-9-]{36}$/);
  assert.equal(first.isRetry, false);
});

test('a completed turn or changed model/question gets a new identity', () => {
  const first = prepareChatTurn({ sessionId: 'a', message: 'Question', messages: [] });
  const pending = [{ role: 'user', content: 'Question', pendingRequest: first }];
  for (const options of [
    { message: 'Different question', messages: pending },
    { message: 'Question', modelConfig: { llmModel: 'other' }, messages: pending },
    { message: 'Question', messages: [...pending, { role: 'assistant', content: 'Answer' }] },
    { sessionId: 'b', message: 'Question', messages: pending },
  ]) {
    assert.notEqual(prepareChatTurn({ sessionId: 'a', ...options }).requestId, first.requestId);
  }
});
