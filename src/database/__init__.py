"""
Database access layer for Acne Advisor AI.

The active runtime uses PostgreSQL chat-history models/repositories and a
Qdrant-backed source-evidence adapter. Neo4j remains a frozen Phase 1 structural
store and is checked only by readiness/integrity tooling.
"""
