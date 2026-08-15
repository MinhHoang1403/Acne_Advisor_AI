import assert from 'node:assert/strict';
import test from 'node:test';

import {
  responseBadgeLabel,
  sourceDisplayLabels,
} from './presentationMetadata.js';

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

test('responseBadgeLabel labels deterministic system output without a retired guardrail concept', () => {
  const label = responseBadgeLabel({
    provider: 'system',
    model: null,
    response_origin: 'deterministic',
  });

  assert.equal(label, '🧭 Hướng dẫn an toàn');
  assert.equal(label.includes('Guardrail'), false);
});

test('responseBadgeLabel identifies a deterministic safety override', () => {
  const label = responseBadgeLabel({
    provider: 'system',
    model: null,
    response_origin: 'deterministic_safety',
  });

  assert.equal(label, 'An toàn');
});
