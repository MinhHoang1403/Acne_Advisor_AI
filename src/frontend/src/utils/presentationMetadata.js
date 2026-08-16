export function answerModelDisplayName(data) {
  const metadata = data?.metadata;
  if (!metadata || metadata.response_origin === 'deterministic_safety' || metadata.provider === 'system') {
    return '';
  }

  const cachedModel = metadata.cache?.hit ? metadata.cached_from_model : null;
  const model = cachedModel || metadata.model || '';
  if (model.includes('gemini-3.5')) return 'Gemini 3.5 Flash';
  if (model.includes('gemini-3.1-flash-lite')) return 'Gemini 3.1 Flash-Lite';
  if (model.includes('qwen3:8b')) return 'Qwen3 8B Local';
  if (model.includes('qwen3')) return 'Qwen3 Local';
  return model;
}


export function sourceDisplayLabels(data) {
  const sourceMetadata = Array.isArray(data?.source_metadata) ? data.source_metadata : [];
  if (sourceMetadata.length > 0) {
    const seenIds = new Set();
    const seenLabels = new Set();
    return sourceMetadata
      .filter((source) => {
        const key = source.source_id || source.display_name;
        const label = source.display_name;
        if (!key || !label || seenIds.has(key) || seenLabels.has(label)) return false;
        seenIds.add(key);
        seenLabels.add(label);
        return true;
      })
      .map((source) => source.display_name)
      .filter(Boolean);
  }

  return [];
}
