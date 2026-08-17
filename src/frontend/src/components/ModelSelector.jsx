import { useState, useEffect } from 'react';
import { fetchModels } from '../api/chatApi.js';

function modelId(model) {
  return model.model_id || model.model;
}

function modelLabel(model) {
  return model.display_name || model.label || modelId(model);
}

function compactModelLabel(model) {
  const fullLabel = modelLabel(model);
  const id = modelId(model);
  const normalized = `${fullLabel} ${id}`.toLowerCase();
  if (normalized.includes('gemini') && normalized.includes('3.1')) return 'Gemini 3.1';
  if (normalized.includes('gemini') && normalized.includes('3.5')) return 'Gemini 3.5';
  if (normalized.includes('gemini') && normalized.includes('2.5')) return 'Gemini 2.5';
  if (normalized.includes('qwen') && normalized.includes('3:8b')) return 'Qwen 3:8B';
  if (normalized.includes('qwen')) return 'Qwen';
  return fullLabel
    .replace(/\s+Flash(?:-Lite)?/i, '')
    .replace(/\s+Local/i, '')
    .replace(/\s+/g, ' ')
    .trim();
}

export default function ModelSelector({ onModelConfigChange }) {
  const [models, setModels] = useState([]);
  const [optionsOpen, setOptionsOpen] = useState(false);
  const [selectedProvider, setSelectedProvider] = useState(
    () => localStorage.getItem('acneAdvisorSelectedProvider') || 'gemini'
  );
  const [selectedModel, setSelectedModel] = useState(
    () => localStorage.getItem('acneAdvisorSelectedModel') || ''
  );
  const [allowFallback, setAllowFallback] = useState(
    () => {
      const storedFallbackPreference = localStorage.getItem('acneAdvisorAllowModelFallback');
      return storedFallbackPreference === null ? true : storedFallbackPreference === 'true';
    }
  );
  const [loading, setLoading] = useState(true);

  // Fetch models from API
  useEffect(() => {
    let isMounted = true;
    fetchModels()
      .then(data => {
        if (isMounted) {
          const catalog = Array.isArray(data.models) ? data.models : [];
          setModels(catalog);
          const storedProvider = localStorage.getItem('acneAdvisorSelectedProvider') || 'gemini';
          const storedModel = localStorage.getItem('acneAdvisorSelectedModel') || '';
          const selectedKey = `${storedProvider}|${storedModel}`;
          const current = catalog.find((m) => `${m.provider}|${modelId(m)}` === selectedKey && m.available);
          if (!current) {
            const defaultModel = catalog.find(
              (m) => m.provider === data.default_provider && modelId(m) === (data.default_model_id || data.default_model)
            ) || catalog.find((m) => m.is_default) || catalog.find((m) => m.available);
            if (defaultModel) {
              setSelectedProvider(defaultModel.provider);
              setSelectedModel(modelId(defaultModel));
            }
          }
          setLoading(false);
        }
      })
      .catch(err => {
        console.error('Failed to fetch models:', err);
        if (isMounted) setLoading(false);
      });
    return () => { isMounted = false; };
  }, []);

  // Sync state upward and to localStorage whenever they change
  useEffect(() => {
    localStorage.setItem('acneAdvisorSelectedProvider', selectedProvider);
    localStorage.setItem('acneAdvisorSelectedModel', selectedModel);
    localStorage.setItem('acneAdvisorAllowModelFallback', allowFallback);

    if (onModelConfigChange) {
      onModelConfigChange({
        llmProvider: selectedProvider,
        llmModel: selectedModel,
        allowModelFallback: allowFallback
      });
    }
  }, [selectedProvider, selectedModel, allowFallback, onModelConfigChange]);

  const handleModelChange = (e) => {
    const val = e.target.value;
    const selectedOption = models.find(m => `${m.provider}|${modelId(m)}` === val);
    if (selectedOption) {
      setSelectedProvider(selectedOption.provider);
      setSelectedModel(modelId(selectedOption));
    }
  };
  const selectedOption = models.find(m => `${m.provider}|${modelId(m)}` === `${selectedProvider}|${selectedModel}`);
  const selectedTitle = selectedOption ? modelLabel(selectedOption) : selectedModel || 'Model';

  return (
    <div className="model-selector-container">
      <div className="model-selector-main">
        <select
          id="model-selector"
          value={`${selectedProvider}|${selectedModel}`}
          onChange={handleModelChange}
          disabled={loading}
          className="model-selector-dropdown"
          aria-label="Chọn model trả lời"
          title={selectedTitle}
        >
          {loading && <option>Đang tải model...</option>}
          {!loading && models.length === 0 && <option>Không có model khả dụng</option>}
          {!loading && models.map(m => (
            <option
              key={`${m.provider}|${modelId(m)}`}
              value={`${m.provider}|${modelId(m)}`}
              disabled={!m.available}
              title={modelLabel(m)}
            >
              {compactModelLabel(m)} {!m.available ? '(Offline)' : ''}
            </option>
          ))}
        </select>
      </div>

      <button
        type="button"
        className={`model-options-btn ${optionsOpen ? 'model-options-btn-active' : ''}`}
        aria-expanded={optionsOpen}
        aria-controls="model-advanced-options"
        aria-label="Mở tùy chọn nâng cao"
        title="Tùy chọn nâng cao"
        onClick={() => setOptionsOpen((open) => !open)}
      >
        <svg width="18" height="18" fill="none" viewBox="0 0 24 24" stroke="currentColor">
          <path
            strokeLinecap="round"
            strokeLinejoin="round"
            strokeWidth="2"
            d="M12 6V4m0 16v-2m6-6h2M4 12h2m10.95-4.95l1.414-1.414M5.636 18.364l1.414-1.414m0-9.9L5.636 5.636m12.728 12.728l-1.414-1.414"
          />
        </svg>
      </button>

      {optionsOpen && (
        <div
          id="model-advanced-options"
          className="model-options-panel"
          role="dialog"
          aria-label="Tùy chọn model"
        >
          <div className="model-options-panel-title">Tùy chọn</div>
          <label className="model-fallback-toggle" htmlFor="allow-model-fallback">
            <input
              id="allow-model-fallback"
              type="checkbox"
              checked={allowFallback}
              onChange={(e) => setAllowFallback(e.target.checked)}
            />
            <span>
              <strong>Auto fallback</strong>
              <small>Tự chuyển sang model dự phòng khi provider chính lỗi.</small>
            </span>
          </label>
        </div>
      )}
    </div>
  );
}
