const SUGGESTED_QUESTIONS = [
  {
    title: 'Tư vấn theo triệu chứng',
    text: 'Mô tả loại mụn, vị trí và mức độ.',
    prompt: 'Tôi muốn được tư vấn mụn theo triệu chứng. Tôi nên mô tả những thông tin nào?',
  },
  {
    title: 'Tra cứu hoạt chất trị mụn',
    text: 'Tìm hiểu công dụng và lưu ý an toàn.',
    prompt: 'Tôi muốn tìm hiểu các hoạt chất trị mụn phổ biến và lưu ý an toàn.',
  },
  {
    title: 'Khi nào cần đi khám',
    text: 'Nhận biết dấu hiệu cần gặp bác sĩ.',
    prompt: 'Khi nào tình trạng mụn hoặc phản ứng thuốc cần đi khám bác sĩ?',
  },
  {
    title: 'So sánh thuốc/routine',
    text: 'Đối chiếu vai trò, tác dụng phụ và lưu ý.',
    prompt: 'Adapalene và benzoyl peroxide khác nhau thế nào?',
  },
];

export default function EmptyState({ onSendQuestion }) {
  return (
    <div className="empty-state">
      <div className="empty-state-icon">
        <svg width="32" height="32" fill="none" viewBox="0 0 24 24" stroke="currentColor">
          <path
            strokeLinecap="round"
            strokeLinejoin="round"
            strokeWidth="2"
            d="M19.428 15.428a2 2 0 00-1.022-.547l-2.387-.477a6 6 0 00-3.86.517l-.318.158a6 6 0 01-3.86.517L6.05 15.21a2 2 0 00-1.806.547M8 4h8l-1 1v5.172a2 2 0 00.586 1.414l5 5c1.26 1.26.367 3.414-1.415 3.414H4.828c-1.782 0-2.674-2.154-1.414-3.414l5-5A2 2 0 009 10.172V5L8 4z"
          />
        </svg>
      </div>
      <h2 className="empty-state-title">Tôi có thể tư vấn gì liên quan đến mụn cho bạn?</h2>
      <p className="empty-state-description">
        Hỏi về triệu chứng, hoạt chất, routine hoặc dấu hiệu cần đi khám.
      </p>
      <div className="empty-state-suggestions">
        {SUGGESTED_QUESTIONS.map((q) => (
          <button type="button" key={q.title} className="suggestion-btn" onClick={() => onSendQuestion(q.prompt)}>
            <span className="suggestion-btn-title">{q.title}</span>
            <span className="suggestion-btn-text">{q.text}</span>
          </button>
        ))}
      </div>
    </div>
  );
}
