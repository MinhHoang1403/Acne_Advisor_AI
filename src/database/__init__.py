"""
Database access layer for Acne Advisor AI.

The active runtime uses PostgreSQL chat-history models/repositories and a
Qdrant-backed source-evidence adapter. Neo4j remains a structural knowledge
store used by build and integrity tooling, not runtime medical grounding.
"""
