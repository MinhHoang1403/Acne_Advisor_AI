export const DEFAULT_API_BASE_URL = 'http://127.0.0.1:8000';

function readViteApiUrl() {
  try {
    return import.meta.env?.VITE_API_URL;
  } catch {
    return undefined;
  }
}

export function resolveApiBaseUrl(rawValue = readViteApiUrl()) {
  const candidate = String(rawValue || DEFAULT_API_BASE_URL).trim();
  const withoutTrailingSlash = candidate.replace(/\/+$/, '');

  if (!withoutTrailingSlash) {
    return DEFAULT_API_BASE_URL;
  }

  // A root-relative API prefix works both in the browser and in Node contracts.
  // Reject protocol-relative URLs, which could silently switch to another host.
  if (withoutTrailingSlash.startsWith('/') && !withoutTrailingSlash.startsWith('//')) {
    return withoutTrailingSlash;
  }

  let parsed;
  try {
    parsed = new URL(withoutTrailingSlash, globalThis.location?.origin);
  } catch {
    throw new Error(`Invalid VITE_API_URL: ${candidate}`);
  }

  if (!['http:', 'https:'].includes(parsed.protocol)) {
    throw new Error(`Invalid VITE_API_URL protocol: ${parsed.protocol}`);
  }

  return withoutTrailingSlash;
}

export const API_BASE_URL = resolveApiBaseUrl();

export function buildApiUrl(path, baseUrl = API_BASE_URL) {
  const normalizedPath = String(path || '').replace(/^\/+/, '');
  return `${resolveApiBaseUrl(baseUrl)}/${normalizedPath}`;
}
