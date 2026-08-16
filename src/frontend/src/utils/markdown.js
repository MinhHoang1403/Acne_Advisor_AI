import React from 'react';

function isTableRow(line) {
  const trimmed = line.trim();
  return trimmed.startsWith('|') && trimmed.endsWith('|') && trimmed.split('|').length >= 3;
}

function isSeparatorRow(line) {
  if (!isTableRow(line)) return false;
  return splitTableRow(line).every((cell) => /^:?-{3,}:?$/.test(cell.trim()));
}

function splitTableRow(line) {
  return line.trim().replace(/^\|/, '').replace(/\|$/, '').split('|').map((cell) => cell.trim());
}

function isSafeLinkHref(href) {
  return /^https?:\/\//i.test(href);
}

function renderItalic(text, keyPrefix) {
  const parts = String(text).split(/(\*[^*\n]+\*)/g).filter((part) => part !== '');
  return parts.map((part, index) => {
    if (part.startsWith('*') && part.endsWith('*') && part.length > 2) {
      return React.createElement('em', { key: `${keyPrefix}-em-${index}` }, part.slice(1, -1));
    }
    return part;
  });
}

function renderInline(text, keyPrefix) {
  const parts = String(text).split(/(\*\*.+?\*\*|`[^`]+`|\[[^\]]+\]\([^)]+\))/g).filter((part) => part !== '');
  return parts.map((part, index) => {
    if (part.startsWith('**') && part.endsWith('**')) {
      return React.createElement(
        'strong',
        { key: `${keyPrefix}-strong-${index}` },
        renderItalic(part.slice(2, -2), `${keyPrefix}-strong-${index}`),
      );
    }
    if (part.startsWith('`') && part.endsWith('`')) {
      return React.createElement('code', { key: `${keyPrefix}-code-${index}`, className: 'chat-inline-code' }, part.slice(1, -1));
    }
    const linkMatch = part.match(/^\[([^\]]+)\]\(([^)]+)\)$/);
    if (linkMatch) {
      const href = linkMatch[2].trim();
      if (!isSafeLinkHref(href)) {
        return linkMatch[1];
      }
      return React.createElement(
        'a',
        {
          key: `${keyPrefix}-link-${index}`,
          href,
          target: '_blank',
          rel: 'noopener noreferrer',
        },
        linkMatch[1],
      );
    }
    return renderItalic(part, `${keyPrefix}-text-${index}`);
  });
}

function renderTable(lines, key) {
  const header = splitTableRow(lines[0]);
  const bodyRows = lines.slice(2).map(splitTableRow);
  return React.createElement(
    'div',
    { key, className: 'chat-table-wrapper' },
    React.createElement(
      'table',
      { className: 'chat-markdown-table' },
      React.createElement(
        'thead',
        null,
        React.createElement(
          'tr',
          null,
          header.map((cell, index) =>
            React.createElement('th', { key: `h-${index}` }, renderInline(cell, `${key}-h-${index}`)),
          ),
        ),
      ),
      React.createElement(
        'tbody',
        null,
        bodyRows.map((row, rowIndex) =>
          React.createElement(
            'tr',
            { key: `r-${rowIndex}` },
            row.map((cell, cellIndex) =>
              React.createElement(
                'td',
                { key: `c-${cellIndex}` },
                renderInline(cell, `${key}-r-${rowIndex}-c-${cellIndex}`),
              ),
            ),
          ),
        ),
      ),
    ),
  );
}

function renderList(lines, key) {
  const roots = [];
  const stack = [{ indent: -1, items: roots }];
  for (const line of lines) {
    const match = line.match(/^(\s*)[-*]\s+(.+)$/);
    if (!match) continue;
    const indent = match[1].replace(/\t/g, '  ').length;
    while (stack.length > 1 && indent <= stack[stack.length - 1].indent) {
      stack.pop();
    }
    const node = { content: match[2], children: [] };
    stack[stack.length - 1].items.push(node);
    stack.push({ indent, items: node.children });
  }

  function renderItems(items, listKey) {
    return React.createElement(
      'ul',
      { key: listKey, className: 'chat-list' },
      items.map((item, index) => {
        const children = renderInline(item.content, `${listKey}-li-${index}`);
        if (item.children.length > 0) {
          children.push(renderItems(item.children, `${listKey}-nested-${index}`));
        }
        return React.createElement(
          'li',
          { key: index, className: 'chat-list-item' },
          ...children,
        );
      }),
    );
  }

  return renderItems(roots, key);
}

function renderParagraph(line, key) {
  const heading = line.match(/^(#{1,4})\s+(.+)$/);
  if (heading) {
    const level = Math.min(heading[1].length + 2, 4);
    return React.createElement(
      `h${level}`,
      { key, className: 'chat-heading' },
      renderInline(heading[2].trim(), key),
    );
  }
  if (line.startsWith('**') && line.endsWith('**') && line.indexOf('**', 2) === line.length - 2) {
    return React.createElement(
      'p',
      { key, className: 'chat-bold-paragraph' },
      React.createElement('strong', null, renderItalic(line.slice(2, -2), `${key}-bold-paragraph`)),
    );
  }
  return React.createElement('p', { key, className: 'chat-paragraph' }, renderInline(line, key));
}

export function formatText(text) {
  if (!text) return '';

  const lines = String(text).replace(/\r\n/g, '\n').replace(/\r/g, '\n').split('\n');
  const elements = [];
  let index = 0;

  while (index < lines.length) {
    const line = lines[index];
    const trimmed = line.trim();
    const key = elements.length;

    if (trimmed === '') {
      elements.push(React.createElement('div', { key, className: 'chat-spacer' }));
      index += 1;
      continue;
    }

    if (trimmed.startsWith('```')) {
      const codeLines = [];
      index += 1;
      while (index < lines.length && !lines[index].trim().startsWith('```')) {
        codeLines.push(lines[index]);
        index += 1;
      }
      if (index < lines.length) index += 1;
      elements.push(
        React.createElement(
          'pre',
          { key, className: 'chat-code-block' },
          React.createElement('code', null, codeLines.join('\n')),
        ),
      );
      continue;
    }

    if (
      isTableRow(line)
      && index + 1 < lines.length
      && isSeparatorRow(lines[index + 1])
    ) {
      const tableLines = [line, lines[index + 1]];
      index += 2;
      while (index < lines.length && isTableRow(lines[index])) {
        tableLines.push(lines[index]);
        index += 1;
      }
      elements.push(renderTable(tableLines, key));
      continue;
    }

    if (/^\s*[-*]\s+/.test(line)) {
      const listLines = [];
      while (index < lines.length && /^\s*[-*]\s+/.test(lines[index])) {
        listLines.push(lines[index]);
        index += 1;
      }
      elements.push(renderList(listLines, key));
      continue;
    }

    elements.push(renderParagraph(trimmed, key));
    index += 1;
  }

  return elements;
}
