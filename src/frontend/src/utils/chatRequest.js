// A logical turn keeps its identity and pre-submit history across uncertain retries.
export function prepareChatTurn({ sessionId, message, modelConfig = null, messages = [] }) {
  const signature = JSON.stringify([
    message, modelConfig?.llmProvider || null, modelConfig?.llmModel || null,
    Boolean(modelConfig?.allowModelFallback),
  ]);
  const pending = messages.at(-1)?.pendingRequest;
  if (pending?.sessionId === sessionId && pending.signature === signature) {
    return { ...pending, isRetry: true };
  }
  return {
    requestId: globalThis.crypto.randomUUID(), sessionId, signature, isRetry: false,
    history: messages.filter((m) => !m.pendingRequest).slice(-6)
      .map((m) => ({ role: m.role, content: m.content })),
  };
}
