import assert from 'node:assert/strict';
import test from 'node:test';

import { answerModelDisplayName, sourceDisplayLabels } from './presentationMetadata.js';

test('answerModelDisplayName uses the actual answering model without routing details', () => {
  assert.equal(
    answerModelDisplayName({ metadata: { provider: 'gemini', model: 'gemini-3.5-flash' } }),
    'Gemini 3.5 Flash',
  );
  assert.equal(
    answerModelDisplayName({ metadata: { provider: 'ollama', model: 'qwen3:8b' } }),
    'Qwen3 8B Local',
  );
  assert.equal(
    answerModelDisplayName({
      metadata: {
        provider: 'gemini',
        model: 'gemini-3.5-flash',
        cache: { hit: true },
        cached_from_provider: 'ollama',
        cached_from_model: 'qwen3:8b',
      },
    }),
    'Qwen3 8B Local',
  );
});

test('answerModelDisplayName omits deterministic safety and missing model identities', () => {
  assert.equal(
    answerModelDisplayName({
      metadata: {
        provider: 'system',
        model: null,
        requested_provider: 'gemini',
        requested_model: 'gemini-3.5-flash',
        response_origin: 'deterministic_safety',
      },
    }),
    '',
  );
  assert.equal(answerModelDisplayName({ metadata: {} }), '');
  assert.equal(answerModelDisplayName(null), '');
});

test('answer details support sources with model, either one alone, or neither', () => {
  const modelData = { metadata: { provider: 'gemini', model: 'gemini-3.5-flash' } };
  const sourceData = {
    source_metadata: [{ source_id: 'source-1', display_name: 'Tài liệu tiếng Việt về mụn trứng cá' }],
  };
  const combinedData = { ...modelData, ...sourceData };

  assert.deepEqual(sourceDisplayLabels(combinedData), ['Tài liệu tiếng Việt về mụn trứng cá']);
  assert.equal(answerModelDisplayName(combinedData), 'Gemini 3.5 Flash');
  assert.deepEqual(sourceDisplayLabels(modelData), []);
  assert.equal(answerModelDisplayName(modelData), 'Gemini 3.5 Flash');
  assert.deepEqual(sourceDisplayLabels(sourceData), ['Tài liệu tiếng Việt về mụn trứng cá']);
  assert.equal(answerModelDisplayName(sourceData), '');
  assert.deepEqual(sourceDisplayLabels(null), []);
  assert.equal(answerModelDisplayName(null), '');
});

test('sourceDisplayLabels prefers friendly metadata and hides raw technical ids', () => {
  const labels = sourceDisplayLabels({
    sources: ['entity:active_ingredient', 'web_raw_dataset.json'],
    source_metadata: [
      {
        source_id: 'entity:active_ingredient',
        display_name: 'Cơ sở tri thức hoạt chất',
      },
      {
        source_id: 'web_raw_dataset.json',
        display_name: 'Bộ dữ liệu kiến thức mụn',
      },
    ],
  });

  assert.deepEqual(labels, ['Cơ sở tri thức hoạt chất', 'Bộ dữ liệu kiến thức mụn']);
  assert.equal(labels.some((label) => label.startsWith('entity:')), false);
  assert.equal(labels.includes('web_raw_dataset.json'), false);
});

test('sourceDisplayLabels does not expose raw source ids without display metadata', () => {
  assert.deepEqual(
    sourceDisplayLabels({ sources: ['qd_4416_cut.pdf', 'internal-source-id'] }),
    [],
  );
});

test('sourceDisplayLabels removes duplicate labels from distinct child source ids', () => {
  assert.deepEqual(
    sourceDisplayLabels({
      source_metadata: [
        { source_id: 'web-1', display_name: 'DermNet — Acne' },
        { source_id: 'web-2', display_name: 'DermNet — Acne' },
      ],
    }),
    ['DermNet — Acne'],
  );
});
