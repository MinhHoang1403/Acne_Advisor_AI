import test from 'node:test';
import assert from 'node:assert/strict';

import { mergeSessionMessages, mergeSessionSummaries } from './sessionMerge.js';

test('same-ID server summary preserves unsynced local messages', () => {
  const local = [{
    id: 'session-a',
    title: 'Local',
    createdAt: 10,
    updatedAt: 30,
    messages: [{ role: 'user', content: 'unsynced' }],
  }];
  const server = [{
    id: 'session-a',
    title: 'Server',
    createdAt: 20,
    updatedAt: 25,
    messages: [],
    _fromBackend: true,
  }];

  const merged = mergeSessionSummaries(local, server);

  assert.equal(merged.length, 1);
  assert.equal(merged[0].title, 'Server');
  assert.deepEqual(merged[0].messages, local[0].messages);
  assert.equal(merged[0].updatedAt, 30);
});

test('server refresh deduplicates persisted local content and keeps local tail order', () => {
  const persisted = { id: 'server-1', role: 'user', content: 'first' };
  const localCopy = { role: 'user', content: 'first' };
  const unsynced = { role: 'assistant', content: 'second' };

  assert.deepEqual(
    mergeSessionMessages([persisted], [localCopy, unsynced]),
    [persisted, unsynced],
  );
});

test('different sessions remain isolated during summary merge', () => {
  const local = [{ id: 'local', createdAt: 1, updatedAt: 1, messages: [] }];
  const server = [{ id: 'server', createdAt: 2, updatedAt: 2, messages: [] }];

  assert.deepEqual(
    mergeSessionSummaries(local, server).map((session) => session.id),
    ['local', 'server'],
  );
});
