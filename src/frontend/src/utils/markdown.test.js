import assert from 'node:assert/strict';
import test from 'node:test';

import { formatText } from './markdown.js';

function collectByType(nodes, type, found = []) {
  const list = Array.isArray(nodes) ? nodes : [nodes];
  for (const node of list) {
    if (!node || typeof node !== 'object') continue;
    if (node.type === type) found.push(node);
    collectByType(node.props?.children, type, found);
  }
  return found;
}

function collectText(nodes) {
  if (nodes == null || nodes === false) return '';
  if (typeof nodes === 'string' || typeof nodes === 'number') return String(nodes);
  if (Array.isArray(nodes)) return nodes.map(collectText).join('');
  if (typeof nodes === 'object') return collectText(nodes.props?.children);
  return '';
}

test('formatText renders GFM table as table elements without raw pipe paragraph', () => {
  const rendered = formatText(
    [
      '| Loại mụn | Đặc điểm |',
      '|---|---|',
      '| Mụn đầu đen | Nhân mụn mở |',
      '| Mụn đầu trắng | Nhân mụn đóng |',
    ].join('\n'),
  );

  assert.equal(collectByType(rendered, 'table').length, 1);
  assert.equal(collectByType(rendered, 'th').length, 2);
  assert.equal(collectByType(rendered, 'td').length, 4);
  assert.equal(collectByType(rendered, 'p').length, 0);
  assert.match(collectText(rendered), /Mụn đầu đen/);
});

test('formatText keeps provider badges and sources outside answer rendering', () => {
  const rendered = formatText('Differin chứa adapalene và thuộc nhóm **retinoid bôi**.');

  assert.equal(collectByType(rendered, 'table').length, 0);
  assert.equal(collectByType(rendered, 'strong').length, 1);
  assert.doesNotMatch(collectText(rendered), /Gemini|Qwen|Nguồn:/);
});

test('formatText renders nested bold and italic without visible markdown markers', () => {
  const rendered = formatText('**Vai trò của vi khuẩn *Cutibacterium acnes* (*C. acnes*):**');

  const visibleText = collectText(rendered);
  assert.equal(collectByType(rendered, 'strong').length, 1);
  assert.equal(collectByType(rendered, 'em').length, 2);
  assert.match(visibleText, /Cutibacterium acnes/);
  assert.doesNotMatch(visibleText, /\*\*/);
});

test('formatText renders bold phrase with following content without raw markers', () => {
  const rendered = formatText('- **Vai trò của Cutibacterium acnes (C. acnes):** Nội dung');

  const visibleText = collectText(rendered);
  assert.equal(collectByType(rendered, 'strong').length, 1);
  assert.match(visibleText, /Nội dung/);
  assert.doesNotMatch(visibleText, /\*\*/);
});

test('formatText renders requested bold headings without visible markdown markers', () => {
  const rendered = formatText(['**Bã nhờn**', 'Nội dung.', '', '**Viêm**', 'Nội dung.'].join('\n'));

  const visibleText = collectText(rendered);
  assert.equal(collectByType(rendered, 'strong').length, 2);
  assert.match(visibleText, /Bã nhờn/);
  assert.match(visibleText, /Viêm/);
  assert.doesNotMatch(visibleText, /\*\*/);
});

test('formatText renders headings, lists, code blocks, and safe external links', () => {
  const rendered = formatText(
    [
      '## Lưu ý an toàn',
      '- Dùng `adapalene` theo hướng dẫn.',
      '[Nguồn ngoài](https://example.com)',
      '```',
      'no raw html',
      '```',
    ].join('\n'),
  );

  const links = collectByType(rendered, 'a');
  assert.equal(collectByType(rendered, 'h4').length, 1);
  assert.equal(collectByType(rendered, 'ul').length, 1);
  assert.equal(collectByType(rendered, 'code').length, 2);
  assert.equal(collectByType(rendered, 'pre').length, 1);
  assert.equal(links.length, 1);
  assert.equal(links[0].props.target, '_blank');
  assert.equal(links[0].props.rel, 'noopener noreferrer');
});

test('formatText preserves nested bullet hierarchy', () => {
  const rendered = formatText(
    [
      '- Retinoid bôi',
      '  - Adapalene',
      '  - Tretinoin',
      '- Hoạt chất kháng khuẩn',
      '  - Benzoyl peroxide',
    ].join('\n'),
  );

  assert.equal(collectByType(rendered, 'ul').length, 3);
  assert.equal(collectByType(rendered, 'li').length, 5);
  assert.match(collectText(rendered), /Adapalene/);
  assert.match(collectText(rendered), /Benzoyl peroxide/);
});

test('formatText renders unsafe markdown links as plain text', () => {
  const rendered = formatText('[Không mở](javascript:alert(1)) và [data](data:text/html,bad)');

  assert.equal(collectByType(rendered, 'a').length, 0);
  assert.match(collectText(rendered), /Không mở/);
  assert.match(collectText(rendered), /data/);
  assert.doesNotMatch(collectText(rendered), /javascript:|data:text/);
});
