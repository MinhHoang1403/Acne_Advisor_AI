from __future__ import annotations

from src.knowledge.normalizer import DEFAULT_TAXONOMY_PATH, DrugEntityNormalizer
from src.knowledge.taxonomy_models import migrate_v1_taxonomy, validate_taxonomy_catalog


def test_v1_migration_preserves_existing_products() -> None:
    data = DrugEntityNormalizer._load_taxonomy(DEFAULT_TAXONOMY_PATH)
    catalog = migrate_v1_taxonomy(data, source_path=DEFAULT_TAXONOMY_PATH)
    products = {entity.canonical_name for entity in catalog.entities if entity.entity_type == "drug_product"}

    assert {"Differin", "Epiduo", "Dalacin T"}.issubset(products)


def test_v1_migration_does_not_create_an_artificial_azelaic_acid_class() -> None:
    data = DrugEntityNormalizer._load_taxonomy(DEFAULT_TAXONOMY_PATH)
    catalog = migrate_v1_taxonomy(data, source_path=DEFAULT_TAXONOMY_PATH)
    classes = {entity.canonical_name for entity in catalog.entities if entity.entity_type == "drug_class"}

    assert "azelaic_acid" not in classes
    assert validate_taxonomy_catalog(catalog).passed is True
