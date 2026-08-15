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

  return [];
}
