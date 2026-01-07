"""
APScheduler Background Jobs for PharmaBula

HYBRID UPDATE SYSTEM:
- On-demand updates: Individual drugs are updated when users request them
- Weekly bulk updates: Check all indexed drugs for staleness
- This minimizes API calls while keeping data fresh

The DrugDataService handles on-demand updates at query time.
This scheduler handles bulk/maintenance operations.
"""

import asyncio
import logging
from datetime import datetime, timedelta
from typing import Optional

from apscheduler.schedulers.asyncio import AsyncIOScheduler
from apscheduler.triggers.cron import CronTrigger
from apscheduler.triggers.interval import IntervalTrigger

from src.config import get_settings
from src.database.metadata_cache import get_metadata_cache
from src.database.vector_store import get_vector_store

logger = logging.getLogger(__name__)


# Global scheduler instance
_scheduler: Optional[AsyncIOScheduler] = None

# Freshness threshold in days
DATA_FRESHNESS_DAYS = 30


async def check_stale_data_job():
    """
    Job: Check for stale drug data and attempt updates.
    
    This runs weekly to ensure all indexed drugs have fresh data.
    Individual on-demand updates happen at query time via DrugDataService.
    
    Runs: Weekly (Sunday at 2 AM)
    """
    logger.info("Starting weekly stale data check...")
    
    cache = get_metadata_cache()
    vector_store = get_vector_store()
    
    # Get all drugs from cache
    from src.scrapers.anvisa_scraper import AnvisaScraper
    
    stats = cache.get_stats()
    stale_count = 0
    updated_count = 0
    
    try:
        # Check each drug's freshness
        with cache._get_connection() as conn:
            cursor = conn.execute("""
                SELECT drug_id, drug_name, last_scraped 
                FROM drug_cache 
                WHERE is_indexed = TRUE
            """)
            drugs = cursor.fetchall()
        
        cutoff = datetime.now() - timedelta(days=DATA_FRESHNESS_DAYS)
        
        for drug in drugs:
            last_scraped = drug["last_scraped"]
            if isinstance(last_scraped, str):
                try:
                    last_scraped = datetime.fromisoformat(last_scraped)
                except ValueError:
                    last_scraped = None
            
            if not last_scraped or last_scraped < cutoff:
                stale_count += 1
                logger.info(f"Drug {drug['drug_name']} is stale, attempting update...")
                
                # Try to update via ANVISA (may fail due to 403)
                try:
                    async with AnvisaScraper() as scraper:
                        bulletin = await scraper.fetch_and_process_bulletin(drug["drug_id"])
                        
                        if bulletin and bulletin.text_content:
                            # Update vector store
                            vector_store.delete_drug(bulletin.id)
                            vector_store.add_document(
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
                            cache.save_drug(
                                drug_id=bulletin.id,
                                drug_name=bulletin.name,
                                company=bulletin.company,
                                active_ingredient=bulletin.active_ingredient,
                                content_hash=bulletin.content_hash(),
                                is_indexed=True
                            )
                            updated_count += 1
                            logger.info(f"Updated: {bulletin.name}")
                    
                    # Rate limiting
                    await asyncio.sleep(2)
                    
                except Exception as e:
                    logger.warning(f"Could not update {drug['drug_name']}: {e}")
        
        # Log results
        cache.log_scrape(
            drugs_found=stale_count,
            drugs_updated=updated_count,
            status="success"
        )
        
        logger.info(
            f"Stale data check completed: {stale_count} stale, {updated_count} updated"
        )
        
    except Exception as e:
        logger.error(f"Stale data check failed: {e}")
        cache.log_scrape(
            drugs_found=0,
            drugs_updated=0,
            status="error",
            error_message=str(e)
        )


async def load_sample_data_job():
    """
    Job: Ensure sample data is loaded on startup.
    
    This ensures the database always has base data available.
    Runs once at startup.
    """
    logger.info("Checking sample data...")
    
    vector_store = get_vector_store()
    
    if vector_store.count() == 0:
        logger.info("Loading sample drug data...")
        
        from src.scrapers.sample_data import get_sample_drugs
        from src.database.metadata_cache import get_metadata_cache
        
        cache = get_metadata_cache()
        drugs = get_sample_drugs()
        
        for drug in drugs:
            if drug.text_content:
                vector_store.add_document(
                    drug_id=drug.id,
                    drug_name=drug.name,
                    text_content=drug.text_content,
                    metadata={
                        "company": drug.company,
                        "active_ingredient": drug.active_ingredient,
                        "bulletin_type": drug.bulletin_type
                    }
                )
                
                cache.save_drug(
                    drug_id=drug.id,
                    drug_name=drug.name,
                    company=drug.company,
                    active_ingredient=drug.active_ingredient,
                    content_hash=drug.content_hash(),
                    is_indexed=True
                )
        
        logger.info(f"Loaded {len(drugs)} sample drugs")
    else:
        logger.info(f"Database already has {vector_store.count()} documents")


async def cleanup_cache_job():
    """
    Job: Clean up old cached data.
    
    Removes:
    - Downloaded PDFs older than 30 days
    - Old scrape history entries (older than 90 days)
    
    Runs: Weekly (Sunday at 3 AM)
    """
    logger.info("Running cache cleanup...")
    
    from pathlib import Path
    
    cache_dir = Path("./data/pdfs")
    
    # Clean old PDFs
    if cache_dir.exists():
        cutoff = datetime.now() - timedelta(days=30)
        
        deleted = 0
        for pdf_file in cache_dir.glob("*.pdf"):
            if datetime.fromtimestamp(pdf_file.stat().st_mtime) < cutoff:
                pdf_file.unlink()
                deleted += 1
        
        logger.info(f"Cleaned up {deleted} old PDF files")
    
    # Clean old scrape history
    cache = get_metadata_cache()
    with cache._get_connection() as conn:
        cutoff_date = datetime.now() - timedelta(days=90)
        conn.execute(
            "DELETE FROM scrape_history WHERE scrape_date < ?",
            (cutoff_date,)
        )
        conn.commit()
    
    logger.info("Cache cleanup completed")


async def health_check_job():
    """
    Job: Periodic health check and statistics logging.
    
    Logs system statistics for monitoring.
    Runs: Every 6 hours
    """
    cache = get_metadata_cache()
    vector_store = get_vector_store()
    
    stats = cache.get_stats()
    vector_count = vector_store.count()
    
    # Check for stale data
    stale_count = 0
    cutoff = datetime.now() - timedelta(days=DATA_FRESHNESS_DAYS)
    
    with cache._get_connection() as conn:
        cursor = conn.execute("""
            SELECT COUNT(*) FROM drug_cache 
            WHERE is_indexed = TRUE 
            AND (last_scraped IS NULL OR last_scraped < ?)
        """, (cutoff,))
        stale_count = cursor.fetchone()[0]
    
    logger.info(
        f"Health check: "
        f"{stats['total_drugs']} drugs in cache, "
        f"{stats['indexed_drugs']} indexed, "
        f"{vector_count} vector documents, "
        f"{stale_count} stale"
    )


def get_scheduler() -> AsyncIOScheduler:
    """Get or create the global scheduler instance."""
    global _scheduler
    
    if _scheduler is None:
        _scheduler = AsyncIOScheduler()
    
    return _scheduler


def setup_scheduler() -> AsyncIOScheduler:
    """
    Configure and setup the scheduler with hybrid update jobs.
    
    HYBRID SYSTEM:
    - On-demand: DrugDataService updates individual drugs at query time
    - Weekly: Bulk stale data check (Sunday 2 AM)
    - Weekly: Cache cleanup (Sunday 3 AM)
    - Regular: Health checks (every 6 hours)
    
    Returns:
        Configured AsyncIOScheduler instance
    """
    settings = get_settings()
    scheduler = get_scheduler()
    
    if not settings.enable_scheduler:
        logger.info("Scheduler disabled by configuration")
        return scheduler
    
    # Job 1: Weekly stale data check (Sunday at 2 AM)
    # On-demand updates handle most freshness, this is for bulk maintenance
    scheduler.add_job(
        check_stale_data_job,
        trigger=CronTrigger(day_of_week="sun", hour=2),
        id="check_stale_data",
        name="Weekly Stale Data Check",
        replace_existing=True,
        max_instances=1
    )
    
    # Job 2: Weekly cache cleanup (Sunday at 3 AM)
    scheduler.add_job(
        cleanup_cache_job,
        trigger=CronTrigger(day_of_week="sun", hour=3),
        id="cleanup_cache",
        name="Weekly Cache Cleanup",
        replace_existing=True,
        max_instances=1
    )
    
    # Job 3: Health check (every 6 hours)
    scheduler.add_job(
        health_check_job,
        trigger=IntervalTrigger(hours=6),
        id="health_check",
        name="System Health Check",
        replace_existing=True,
        max_instances=1
    )
    
    logger.info(
        f"Scheduler configured with {len(scheduler.get_jobs())} jobs. "
        f"Using hybrid update system: on-demand + weekly bulk checks."
    )
    
    return scheduler


async def run_initial_setup():
    """
    Run initial setup when the app starts.
    
    Ensures sample data is loaded if database is empty.
    """
    await load_sample_data_job()

