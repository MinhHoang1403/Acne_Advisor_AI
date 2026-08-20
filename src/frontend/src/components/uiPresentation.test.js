import assert from 'node:assert/strict';
import fs from 'node:fs';
import test from 'node:test';

function readSource(relativePath) {
  return fs.readFileSync(new URL(relativePath, import.meta.url), 'utf8');
}

test('composer hides Auto fallback by default and exposes it in advanced options', () => {
  const chatInput = readSource('./ChatInput.jsx');
  const modelSelector = readSource('./ModelSelector.jsx');

  assert.doesNotMatch(chatInput, /Auto fallback/);
  assert.match(modelSelector, /optionsOpen/);
  assert.match(modelSelector, /setOptionsOpen/);
  assert.match(modelSelector, /aria-expanded=\{optionsOpen\}/);
  assert.match(modelSelector, /aria-controls="model-advanced-options"/);
  assert.match(modelSelector, /role="dialog"/);
  assert.match(modelSelector, /Tùy chọn/);

  const panelIndex = modelSelector.indexOf('id="model-advanced-options"');
  const toggleIndex = modelSelector.indexOf('Auto fallback');
  const conditionalIndex = modelSelector.indexOf('{optionsOpen &&');

  assert.ok(conditionalIndex >= 0);
  assert.ok(panelIndex > conditionalIndex);
  assert.ok(toggleIndex > panelIndex);
});

test('advanced fallback toggle keeps existing localStorage and submit contract', () => {
  const chatInput = readSource('./ChatInput.jsx');
  const modelSelector = readSource('./ModelSelector.jsx');

  assert.match(modelSelector, /acneAdvisorAllowModelFallback/);
  assert.match(modelSelector, /checked=\{allowFallback\}/);
  assert.match(modelSelector, /setAllowFallback\(e\.target\.checked\)/);
  assert.match(modelSelector, /allowModelFallback: allowFallback/);
  assert.match(modelSelector, /storedFallbackPreference === null \? true : storedFallbackPreference === 'true'/);
  assert.match(chatInput, /onSubmit\(message, modelConfig\)/);
});

test('advanced fallback panel is a compact popover and does not expand the composer', () => {
  const styles = readSource('../styles.css');
  const popoverStyles = styles.slice(styles.indexOf('Empty state, brand, fallback popover, and title'));

  assert.match(popoverStyles, /\.model-selector-container\s*{[^}]*position: relative;[^}]*flex-wrap: nowrap;/s);
  assert.match(popoverStyles, /\.model-options-panel\s*{[^}]*position: absolute;/s);
  assert.match(popoverStyles, /\.model-options-panel\s*{[^}]*bottom: calc\(100% \+ 12px\);/s);
  assert.match(popoverStyles, /\.model-options-panel\s*{[^}]*width: min\(260px, calc\(100vw - 36px\)\);/s);
  assert.match(popoverStyles, /\.chat-input-form,[\s\S]*?\.model-selector-container\s*{[^}]*overflow: visible;/s);
  assert.match(popoverStyles, /\.model-options-panel\s*{[^}]*z-index: 140;/s);
});

test('model selector still renders catalog models from the backend', () => {
  const modelSelector = readSource('./ModelSelector.jsx');

  assert.match(modelSelector, /fetchModels/);
  assert.match(modelSelector, /data\.default_provider/);
  assert.match(modelSelector, /model-selector-dropdown/);
  assert.match(modelSelector, /models\.map/);
  assert.match(modelSelector, /disabled=\{!m\.available\}/);
  assert.match(modelSelector, /compactModelLabel/);
  assert.doesNotMatch(modelSelector, />\s*Model\s*</);
});

test('assistant messages no longer render a leading lightning avatar', () => {
  const chatMessage = readSource('./ChatMessage.jsx');
  const chatWindow = readSource('./ChatWindow.jsx');

  assert.doesNotMatch(chatMessage, /chat-avatar-assistant/);
  assert.doesNotMatch(chatMessage, /M13 10V3L4 14h7v7l9-11h-7z/);
  assert.doesNotMatch(chatWindow, /chat-avatar-assistant/);
  assert.doesNotMatch(chatWindow, /M13 10V3L4 14h7v7l9-11h-7z/);
});

test('source details and markdown table presentation remain accessible', () => {
  const chatMessage = readSource('./ChatMessage.jsx');
  const markdown = readSource('../utils/markdown.js');
  const styles = readSource('../styles.css');

  assert.match(chatMessage, /sourceDisplayLabels/);
  assert.doesNotMatch(chatMessage, /graph_facts|DebugPanel|guardrail/);
  assert.match(chatMessage, /answer-details-toggle/);
  assert.match(chatMessage, /aria-expanded=\{detailsOpen\}/);
  assert.match(chatMessage, /Chi tiết/);
  assert.match(chatMessage, /detailsOpen &&/);
  assert.match(chatMessage, /Tài liệu tham khảo/);
  assert.match(chatMessage, /Mô hình trả lời/);
  assert.match(chatMessage, /Boolean\(nonGeneratedDetails\)/);
  assert.match(chatMessage, /Trạng thái: \{nonGeneratedDetails\.status\}/);
  assert.match(chatMessage, /Lý do: \{nonGeneratedDetails\.reason\}/);
  assert.match(chatMessage, /Model quyết định: \{nonGeneratedDetails\.decisionModel\}/);
  assert.doesNotMatch(chatMessage, /fallback_reason_code|evidence_gap/);
  assert.doesNotMatch(chatMessage, /Nguồn đã truy hồi|responseBadgeLabel|metadata\.retrieval|dense_bm25_rrf/);
  assert.match(chatMessage, /chat-meta-source-list/);
  assert.doesNotMatch(chatMessage, /sourceLabels\.join\(', '\)/);
  assert.match(markdown, /className: 'chat-markdown-table'/);
  assert.match(styles, /\.chat-table-wrapper/);
  assert.match(styles, /\.answer-details-panel/);
});

test('composer focus stays on the container without native textarea rectangle', () => {
  const chatInput = readSource('./ChatInput.jsx');
  const styles = readSource('../styles.css');
  const interactionStyles = styles.slice(styles.indexOf('Composer focus and message hierarchy'));

  assert.match(chatInput, /<textarea/);
  assert.match(chatInput, /className=\{`chat-textarea/);
  assert.match(chatInput, /aria-label="Nhập câu hỏi về tình trạng mụn"/);
  assert.match(chatInput, /aria-label="Gửi câu hỏi"/);
  assert.match(interactionStyles, /\.chat-textarea,[\s\S]*?\.chat-textarea:focus-visible\s*{[^}]*outline: none;[^}]*box-shadow: none;/s);
  assert.match(interactionStyles, /\.chat-input-form:focus-within\s*{[^}]*box-shadow:/s);
});

test('message hierarchy keeps user text light and assistant markdown prominent', () => {
  const chatMessage = readSource('./ChatMessage.jsx');
  const markdown = readSource('../utils/markdown.js');
  const styles = readSource('../styles.css');
  const messageStyles = styles.slice(styles.indexOf('Composer focus and message hierarchy'));

  assert.match(chatMessage, /<div className="chat-user-text">\{msg\.content\}<\/div>/);
  assert.doesNotMatch(chatMessage, /<strong>\{msg\.content\}<\/strong>/);
  assert.match(chatMessage, /formatText\(msg\.content\)/);
  assert.match(markdown, /React\.createElement\(\s*'strong'/);
  assert.match(messageStyles, /\.chat-user-text\s*{[^}]*background: #f7f7f8;/s);
  assert.match(messageStyles, /\.chat-user-text\s*{[^}]*font-weight: 400;/s);
  assert.match(messageStyles, /\.chat-assistant-text\s*{[^}]*color: #0f172a;[^}]*line-height: 1\.74;/s);
});

test('white neutral theme and Acne Advisor AI favicon are active', () => {
  const styles = readSource('../styles.css');
  const indexHtml = readSource('../../index.html');
  const favicon = readSource('../../public/favicon.svg');
  const themeStyles = styles.slice(styles.indexOf('White/neutral theme overrides'));

  assert.match(themeStyles, /--app-bg: #ffffff/);
  assert.match(themeStyles, /\.app-layout,[\s\S]*?background: #ffffff;/);
  assert.match(themeStyles, /\.chat-user-text\s*{[^}]*background: #f3f4f6;/s);
  assert.match(indexHtml, /href="\/favicon\.svg"/);
  assert.match(favicon, /Acne Advisor AI/);
  assert.match(favicon, /<rect[^>]+fill="#111827"/);
  assert.doesNotMatch(favicon, /863bff|7e14ff|lightning|bolt/);
});

test('empty state renders four suggestion cards', () => {
  const emptyState = readSource('./EmptyState.jsx');
  const chatWindow = readSource('./ChatWindow.jsx');
  const styles = readSource('../styles.css');
  const emptyStateStyles = styles.slice(styles.indexOf('Empty state, brand, fallback popover, and title'));

  assert.match(emptyState, /Tôi có thể tư vấn gì liên quan đến mụn cho bạn\?/);
  assert.match(emptyState, /Hỏi về triệu chứng, hoạt chất, routine hoặc dấu hiệu cần đi khám\./);
  assert.match(emptyState, /Tư vấn theo triệu chứng/);
  assert.match(emptyState, /Tra cứu hoạt chất trị mụn/);
  assert.match(emptyState, /Khi nào cần đi khám/);
  assert.match(emptyState, /So sánh thuốc\/routine/);
  assert.match(emptyState, /onSendQuestion\(q\.prompt\)/);
  assert.match(chatWindow, /chat-scroll-area-empty/);
  assert.match(emptyStateStyles, /\.chat-scroll-area-empty \.empty-state\s*{[^}]*justify-content: center;/s);
  assert.match(emptyStateStyles, /\.chat-scroll-area-empty \.empty-state\s*{[^}]*transform: translateY\(18px\);/s);
});

test('sidebar chat history and accessibility labels are preserved', () => {
  const sidebar = readSource('./Sidebar.jsx');
  const chatWindow = readSource('./ChatWindow.jsx');
  const chatInput = readSource('./ChatInput.jsx');

  assert.match(sidebar, /visibleSessions\.map/);
  assert.match(sidebar, /onSelectSession\(session\.id\)/);
  assert.match(sidebar, /aria-current=\{activeSessionId === session\.id \? 'page' : undefined\}/);
  assert.match(sidebar, /aria-label="Lịch sử chat"/);
  assert.match(sidebar, /aria-label="Ẩn thanh lịch sử chat"/);
  assert.match(chatWindow, /aria-label="Mở thanh lịch sử chat"/);
  assert.match(chatInput, /aria-label="Nhập câu hỏi về tình trạng mụn"/);
  assert.match(chatInput, /aria-label="Gửi câu hỏi"/);
});

test('chat history titles are generated from the first user message only', () => {
  const app = readSource('../App.jsx');

  assert.match(app, /deriveChatTitleFromFirstUserMessage/);
  assert.match(app, /shouldGenerateSessionTitle/);
  assert.doesNotMatch(app, /substring\(0, 40\)/);
  assert.match(app, /apiRenameSession\(responseSessionId, generatedSessionTitle\)/);
  assert.match(app, /messages: \[\.\.\.s\.messages, userMsgObj\]/);
  assert.doesNotMatch(app, /assistantMsgObj[\s\S]*title:/);
});

test('main top header is removed and stable status text lives in sidebar', () => {
  const chatWindow = readSource('./ChatWindow.jsx');
  const sidebar = readSource('./Sidebar.jsx');
  const app = readSource('../App.jsx');
  const styles = readSource('../styles.css');
  const layoutStyles = styles.slice(styles.indexOf('Empty state, brand, fallback popover, and title'));
  const messageStyles = styles.slice(styles.indexOf('Composer focus and message hierarchy'));

  assert.doesNotMatch(chatWindow, /<header className="chat-header">/);
  assert.doesNotMatch(sidebar, /Backend sẵn sàng/);
  assert.doesNotMatch(sidebar, /Hệ thống đang hoạt động ổn định/);
  assert.doesNotMatch(app, /Đang kiểm tra kết nối backend/);
  assert.doesNotMatch(sidebar, /Đang kiểm tra kết nối backend/);
  assert.match(sidebar, /Đang hoạt động/);
  assert.match(sidebar, /Đang chờ kết nối/);
  assert.match(sidebar, /compactConnectionLabel/);
  assert.match(sidebar, /connectionState === 'connected'\) return 'Đang hoạt động'/);
  assert.match(sidebar, /return 'Đang chờ kết nối'/);
  assert.match(sidebar, /!isConnectionPending/);
  assert.match(sidebar, /Acne Advisor AI/);
  assert.doesNotMatch(sidebar, /sidebar-brand-mark/);
  assert.doesNotMatch(sidebar, /Thông tin tham khảo/);
  assert.match(styles, /\.chat-header\s*{[^}]*display: none;/s);
  assert.match(layoutStyles, /\.sidebar-inline-toggle-btn/);
  assert.match(layoutStyles, /\.sidebar-title\s*{[^}]*font-size: 20px;[^}]*font-weight: 800;/s);
  assert.match(styles, /\.sidebar-reopen-btn/);
  assert.match(styles, /\.sidebar-brand-mark,[\s\S]*?display: none;/);
  assert.match(messageStyles, /\.sidebar-offline-banner-checking,[\s\S]*?display: none;/);
});
