"""
Drug Data Service - Hybrid Update System

This service implements a smart hybrid approach for drug data:
1. Pre-loaded data for common medications (sample_data.py)
2. On-demand fetching when user requests a drug not in database
3. Freshness check - updates stale data before responding

This minimizes API calls while ensuring users always get current information.
"""

import logging
from datetime import datetime, timedelta
from typing import Optional

from src.config import get_settings
from src.database.metadata_cache import get_metadata_cache
from src.database.vector_store import get_vector_store
from src.scrapers.anvisa_scraper import AnvisaScraper, DrugBulletin
from src.scrapers.sample_data import get_drug_by_name, get_sample_drugs

logger = logging.getLogger(__name__)

# How old data can be before we consider it stale (in days)
DATA_FRESHNESS_DAYS = 30


class DrugDataService:
    """
    Hybrid drug data service with on-demand updates.
    
    This service:
    - Returns cached data if fresh
    - Attempts to fetch new data if cache is stale or missing
    - Falls back to sample data if ANVISA is unavailable
    """
    
    def __init__(self):
        self.vector_store = get_vector_store()
        self.cache = get_metadata_cache()
        self._initialized = False
    
    async def initialize(self):
        """Initialize with sample data if database is empty."""
        if self._initialized:
            return
        
        if self.vector_store.count() == 0:
            logger.info("Database empty, loading sample data...")
            await self._load_sample_data()
        
        self._initialized = True
    
    async def _load_sample_data(self):
        """Load sample drug data into the database."""
        drugs = get_sample_drugs()
        
        for drug in drugs:
            if drug.text_content:
                self.vector_store.add_document(
                    drug_id=drug.id,
                    drug_name=drug.name,
                    text_content=drug.text_content,
                    metadata={
                        "company": drug.company,
                        "active_ingredient": drug.active_ingredient,
                        "bulletin_type": drug.bulletin_type
                    }
                )
                
                self.cache.save_drug(
                    drug_id=drug.id,
                    drug_name=drug.name,
                    company=drug.company,
                    active_ingredient=drug.active_ingredient,
                    content_hash=drug.content_hash(),
                    is_indexed=True
                )
        
        logger.info(f"Loaded {len(drugs)} sample drugs into database")
    
    def _is_data_stale(self, drug_id: str) -> bool:
        """Check if cached data is stale and needs refreshing."""
        drug = self.cache.get_drug(drug_id)
        
        if not drug:
            return True
        
        last_scraped = drug.get("last_scraped")
        if not last_scraped:
            return True
        
        # Parse the timestamp
        if isinstance(last_scraped, str):
            try:
                last_scraped = datetime.fromisoformat(last_scraped)
            except ValueError:
                return True
        
        age = datetime.now() - last_scraped
        return age > timedelta(days=DATA_FRESHNESS_DAYS)
    
    async def get_drug_context(
        self,
        query: str,
        n_results: int = 5
    ) -> tuple[list[dict], bool]:
        """
        Get drug context for a query, with on-demand updates.
        
        Args:
            query: User's question or drug name
            n_results: Number of context chunks to retrieve
            
        Returns:
            Tuple of (context_list, was_updated)
        """
        await self.initialize()
        
        # First, try to get from existing data
        context = self.vector_store.search(query, n_results=n_results)
        
        was_updated = False
        
        # If no results or data is stale, try to fetch new data
        if not context:
            logger.info(f"No cached data for query: {query}")
            was_updated = await self._try_fetch_drug(query)
            
            if was_updated:
                context = self.vector_store.search(query, n_results=n_results)
        else:
            # Check if the top result's data is stale
            top_drug_id = context[0].get("metadata", {}).get("drug_id")
            if top_drug_id and self._is_data_stale(top_drug_id):
                logger.info(f"Data for {top_drug_id} is stale, attempting refresh...")
                was_updated = await self._try_refresh_drug(top_drug_id)
                
                if was_updated:
                    context = self.vector_store.search(query, n_results=n_results)
        
        return context, was_updated
    
    async def _try_fetch_drug(self, drug_name: str) -> bool:
        """
        Try to fetch drug data from ANVISA or use sample data.
        
        Returns True if new data was added.
        """
        # First check if we have it in sample data
        sample_drug = get_drug_by_name(drug_name)
        if sample_drug and sample_drug.text_content:
            logger.info(f"Found {drug_name} in sample data")
            return self._index_drug(sample_drug)
        
        # Try ANVISA API (may fail due to 403)
        try:
            async with AnvisaScraper() as scraper:
                results = await scraper.search_drugs(drug_name, page_size=1)
                
                if results:
                    drug_data = results[0]
                    drug_id = str(drug_data.get("idProduto", ""))
                    
                    if drug_id:
                        bulletin = await scraper.fetch_and_process_bulletin(drug_id)
                        if bulletin and bulletin.text_content:
                            return self._index_drug(bulletin)
        except Exception as e:
            logger.warning(f"Could not fetch from ANVISA: {e}")
        
        return False
    
    async def _try_refresh_drug(self, drug_id: str) -> bool:
        """
        Try to refresh stale drug data.
        
        Returns True if data was updated.
        """
        try:
            async with AnvisaScraper() as scraper:
                bulletin = await scraper.fetch_and_process_bulletin(drug_id)
                
                if bulletin and bulletin.text_content:
                    # Check if content actually changed
                    new_hash = bulletin.content_hash()
                    if not self.cache.needs_update(drug_id, new_hash):
                        logger.info(f"Drug {drug_id} content unchanged")
                        # Just update the timestamp
                        self.cache.save_drug(
                            drug_id=drug_id,
                            drug_name=bulletin.name,
                            content_hash=new_hash,
                            is_indexed=True
                        )
                        return False
                    
                    # Content changed, re-index
                    return self._index_drug(bulletin)
        except Exception as e:
            logger.warning(f"Could not refresh drug {drug_id}: {e}")
        
        return False
    
    def _index_drug(self, bulletin: DrugBulletin) -> bool:
        """Index a drug bulletin in the vector store."""
        if not bulletin.text_content:
            return False
        
        # Delete old chunks
        self.vector_store.delete_drug(bulletin.id)
        
        # Add new chunks
        chunks = self.vector_store.add_document(
            drug_id=bulletin.id,
            drug_name=bulletin.name,
            text_content=bulletin.text_content,
            metadata={
                "company": bulletin.company,
                "active_ingredient": bulletin.active_ingredient,
                "bulletin_type": bulletin.bulletin_type
            }
        )
        
        # Update cache
        self.cache.save_drug(
            drug_id=bulletin.id,
            drug_name=bulletin.name,
            company=bulletin.company,
            active_ingredient=bulletin.active_ingredient,
            content_hash=bulletin.content_hash(),
            is_indexed=True
        )
        
        logger.info(f"Indexed drug {bulletin.name} with {chunks} chunks")
        return chunks > 0
    
    def get_stats(self) -> dict:
        """Get service statistics."""
        cache_stats = self.cache.get_stats()
        return {
            "total_drugs": cache_stats["total_drugs"],
            "indexed_drugs": cache_stats["indexed_drugs"],
            "vector_documents": self.vector_store.count(),
            "freshness_days": DATA_FRESHNESS_DAYS,
            "last_scrape": cache_stats.get("last_scrape")
        }


# Singleton instance
_drug_service: Optional[DrugDataService] = None


def get_drug_service() -> DrugDataService:
    """Get or create the global drug data service."""
    global _drug_service
    if _drug_service is None:
        _drug_service = DrugDataService()
    return _drug_service
