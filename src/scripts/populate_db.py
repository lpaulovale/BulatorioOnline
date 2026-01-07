"""
Populate Database with Sample Drug Data

Run this script to populate the vector database with sample drug bulletins
for demonstration purposes.

Usage:
    python -m src.scripts.populate_db
"""

import logging
from src.database.metadata_cache import get_metadata_cache
from src.database.vector_store import get_vector_store
from src.scrapers.sample_data import get_sample_drugs

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def populate_database():
    """Populate the vector store with sample drug data."""
    logger.info("Starting database population with sample data...")
    
    vector_store = get_vector_store()
    cache = get_metadata_cache()
    
    drugs = get_sample_drugs()
    total_chunks = 0
    
    for drug in drugs:
        logger.info(f"Adding: {drug.name}")
        
        # Add to vector store
        if drug.text_content:
            chunks = vector_store.add_document(
                drug_id=drug.id,
                drug_name=drug.name,
                text_content=drug.text_content,
                metadata={
                    "company": drug.company,
                    "active_ingredient": drug.active_ingredient,
                    "bulletin_type": drug.bulletin_type
                }
            )
            total_chunks += chunks
        
        # Add to metadata cache
        cache.save_drug(
            drug_id=drug.id,
            drug_name=drug.name,
            company=drug.company,
            active_ingredient=drug.active_ingredient,
            content_hash=drug.content_hash(),
            is_indexed=True
        )
    
    logger.info(f"✅ Done! Added {len(drugs)} drugs with {total_chunks} chunks to the database.")
    logger.info(f"Total documents in vector store: {vector_store.count()}")
    
    # Log stats
    stats = cache.get_stats()
    logger.info(f"Cache stats: {stats}")


if __name__ == "__main__":
    populate_database()
