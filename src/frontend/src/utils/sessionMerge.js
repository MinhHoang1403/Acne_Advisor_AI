export function mergeSessionMessages(serverMessages = [], localMessages = []) {
  const merged = [...serverMessages];
  const serverIds = new Set(
    serverMessages.filter((message) => message?.id).map((message) => message.id),
  );
  const serverContentCounts = new Map();
  for (const message of serverMessages) {
    const identity = `content:${message?.role || ''}:${message?.content || ''}`;
    serverContentCounts.set(identity, (serverContentCounts.get(identity) || 0) + 1);
  }
  const matchedLocalCounts = new Map();
  const appendedLocalIds = new Set();
  for (const message of localMessages) {
    if (message?.id && (serverIds.has(message.id) || appendedLocalIds.has(message.id))) continue;
    const contentIdentity = `content:${message?.role || ''}:${message?.content || ''}`;
    const matched = matchedLocalCounts.get(contentIdentity) || 0;
    if (matched < (serverContentCounts.get(contentIdentity) || 0)) {
      matchedLocalCounts.set(contentIdentity, matched + 1);
      continue;
    }
    if (message?.id) appendedLocalIds.add(message.id);
    merged.push(message);
  }
  return merged;
}

export function mergeSessionSummaries(localSessions = [], serverSessions = []) {
  const localById = new Map(localSessions.map((session) => [session.id, session]));
  const serverIds = new Set(serverSessions.map((session) => session.id));
  const mergedServerSessions = serverSessions.map((serverSession) => {
    const localSession = localById.get(serverSession.id);
    if (!localSession) return serverSession;
    return {
      ...localSession,
      ...serverSession,
      createdAt: Math.min(localSession.createdAt, serverSession.createdAt),
      updatedAt: Math.max(localSession.updatedAt, serverSession.updatedAt),
      messages: mergeSessionMessages(
        serverSession.messages || [],
        localSession.messages || [],
      ),
    };
  });
  return [
    ...localSessions.filter((session) => !serverIds.has(session.id)),
    ...mergedServerSessions,
  ];
}
