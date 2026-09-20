"""Knowledge entity schemas and rule-based normalizers."""

from src.knowledge.normalizer import DrugEntityNormalizer, normalize_text_key
from src.knowledge.entity_cards import build_entity_cards_from_taxonomy, entity_card_to_text
from src.knowledge.graph_schema import (
    build_entity_graph_records,
    get_entity_graph_constraints,
    get_entity_graph_indexes,
)
from src.knowledge.schemas import (
    EntityCard,
    canonical_text_key,
)
from src.knowledge.versioning import (
    get_embedding_metadata,
    get_knowledge_versions,
    resolve_active_knowledge_build_id,
)

__all__ = [
    "DrugEntityNormalizer",
    "EntityCard",
    "build_entity_cards_from_taxonomy",
    "build_entity_graph_records",
    "canonical_text_key",
    "entity_card_to_text",
    "get_embedding_metadata",
    "get_entity_graph_constraints",
    "get_entity_graph_indexes",
    "get_knowledge_versions",
    "resolve_active_knowledge_build_id",
    "normalize_text_key",
]
