"""Database access layer của Acne Advisor AI.

Runtime dùng PostgreSQL chat-history models/repositories và source-evidence
adapter dựa trên Qdrant. Neo4j là structural knowledge store cho build/integrity
tooling, không phải nguồn medical grounding ở runtime.
"""
