export function modelDisplayName(provider, model) {
  const modelId = model || '';
  if (provider === 'gemini' || modelId.includes('gemini')) {
    if (modelId.includes('3.5')) return 'Gemini 3.5 Flash';
    if (modelId.includes('3.1-flash-lite')) return 'Gemini 3.1 Flash-Lite';
    if (modelId.includes('2.5')) return 'Gemini 2.5 Flash';
    return modelId || 'Gemini';
  }
  if (provider === 'ollama' || modelId.includes('qwen')) {
    if (modelId.includes('qwen3:8b')) return 'Qwen3 8B Local';
    if (modelId.includes('qwen3')) return 'Qwen3 Local';
    if (modelId.includes('qwen2')) return 'Qwen2.5 Local';
    return modelId ? `${modelId} Local` : 'Ollama';
  }
  if (provider === 'rule_based') return 'Rule-based';
  return modelId || provider || 'Unknown model';
}

export function sourceDisplayLabels(data) {
  const sourceMetadata = Array.isArray(data?.source_metadata) ? data.source_metadata : [];
  if (sourceMetadata.length > 0) {
    const seen = new Set();
    return sourceMetadata
      .filter((source) => {
        const key = source.source_id || source.display_name;
        if (!key || seen.has(key)) return false;
        seen.add(key);
        return true;
      })
      .map((source) => source.display_name)
      .filter(Boolean);
  }

  return Array.isArray(data?.sources)
    ? [...new Set(data.sources.filter(Boolean))]
    : [];
}

export function responseBadgeLabel(meta) {
  if (!meta) return '';

  if (meta.response_origin === 'deterministic_safety') {
    return 'An toàn';
  }
  if (meta.cache && meta.cache.hit) {
    const originalName = modelDisplayName(meta.cached_from_provider, meta.cached_from_model);
    return `♻️ Cached · originally answered by ${originalName}`;
  }
  if (meta.response_origin === 'safe_fallback') {
    return '🧭 Hướng dẫn an toàn';
  }
  if (meta.fallback_used && meta.fallback_provider) {
    const actualName = modelDisplayName(meta.provider, meta.model);
    const requestedName = modelDisplayName(meta.requested_provider, meta.requested_model);
    return `⚠️ ${actualName} · dự phòng từ ${requestedName}`;
  }
  if (meta.provider === 'gemini') return `⚡ ${modelDisplayName(meta.provider, meta.model)}`;
  if (meta.provider === 'ollama') return `🖥️ ${modelDisplayName(meta.provider, meta.model)}`;
  if (meta.response_origin === 'deterministic' || meta.provider === 'system') {
    return '🧭 Hướng dẫn an toàn';
  }
  return `⚡ ${meta.model || meta.provider || 'Unknown model'}`;
}
