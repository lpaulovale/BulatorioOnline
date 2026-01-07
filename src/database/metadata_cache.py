"""
SQLite Metadata Cache for Drug Information

Caches drug metadata to avoid redundant API calls and track update history.
"""

import hashlib
import logging
import sqlite3
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from typing import Generator, Optional

from src.config import get_settings

logger = logging.getLogger(__name__)


class MetadataCache:
    """
    SQLite-based cache for drug bulletin metadata.
    
    Tracks which drugs have been scraped and when,
    enabling incremental updates.
    """
    
    def __init__(self, db_path: Optional[str] = None):
        """
        Initialize the metadata cache.
        
        Args:
            db_path: Path to SQLite database file
        """
        settings = get_settings()
        self.db_path = db_path or settings.sqlite_database_path
        
        # Ensure directory exists
        Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)
        
        self._init_database()
    
    @contextmanager
    def _get_connection(self) -> Generator[sqlite3.Connection, None, None]:
        """Get a database connection with context management."""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        try:
            yield conn
        finally:
            conn.close()
    
    def _init_database(self) -> None:
        """Initialize database schema."""
        with self._get_connection() as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS drug_cache (
                    drug_id TEXT PRIMARY KEY,
                    drug_name TEXT NOT NULL,
                    company TEXT,
                    active_ingredient TEXT,
                    content_hash TEXT,
                    last_scraped TIMESTAMP,
                    last_updated TIMESTAMP,
                    pdf_path TEXT,
                    is_indexed BOOLEAN DEFAULT FALSE
                )
            """)
            
            conn.execute("""
                CREATE TABLE IF NOT EXISTS scrape_history (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    scrape_date TIMESTAMP,
                    drugs_found INTEGER,
                    drugs_updated INTEGER,
                    status TEXT,
                    error_message TEXT
                )
            """)
            
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_drug_name 
                ON drug_cache(drug_name)
            """)
            
            conn.commit()
            logger.info("Metadata cache database initialized")
    
    def get_drug(self, drug_id: str) -> Optional[dict]:
        """
        Get cached drug information.
        
        Args:
            drug_id: Drug ID to look up
            
        Returns:
            Drug data dictionary or None if not cached
        """
        with self._get_connection() as conn:
            cursor = conn.execute(
                "SELECT * FROM drug_cache WHERE drug_id = ?",
                (drug_id,)
            )
            row = cursor.fetchone()
            return dict(row) if row else None
    
    def save_drug(
        self,
        drug_id: str,
        drug_name: str,
        company: str = "",
        active_ingredient: str = "",
        content_hash: str = "",
        pdf_path: str = "",
        is_indexed: bool = False
    ) -> None:
        """
        Save or update drug in cache.
        
        Args:
            drug_id: Unique drug ID
            drug_name: Name of the drug
            company: Manufacturing company
            active_ingredient: Active ingredient
            content_hash: Hash of content for change detection
            pdf_path: Path to cached PDF
            is_indexed: Whether drug is indexed in vector store
        """
        now = datetime.now()
        
        with self._get_connection() as conn:
            conn.execute("""
                INSERT INTO drug_cache 
                    (drug_id, drug_name, company, active_ingredient, 
                     content_hash, last_scraped, last_updated, pdf_path, is_indexed)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(drug_id) DO UPDATE SET
                    drug_name = excluded.drug_name,
                    company = excluded.company,
                    active_ingredient = excluded.active_ingredient,
                    content_hash = excluded.content_hash,
                    last_scraped = excluded.last_scraped,
                    last_updated = CASE 
                        WHEN drug_cache.content_hash != excluded.content_hash 
                        THEN excluded.last_updated 
                        ELSE drug_cache.last_updated 
                    END,
                    pdf_path = excluded.pdf_path,
                    is_indexed = excluded.is_indexed
            """, (
                drug_id, drug_name, company, active_ingredient,
                content_hash, now, now, pdf_path, is_indexed
            ))
            conn.commit()
    
    def needs_update(self, drug_id: str, new_content_hash: str) -> bool:
        """
        Check if a drug's content has changed and needs re-indexing.
        
        Args:
            drug_id: Drug ID to check
            new_content_hash: Hash of new content
            
        Returns:
            True if content has changed or drug is not cached
        """
        drug = self.get_drug(drug_id)
        if not drug:
            return True
        return drug.get("content_hash") != new_content_hash
    
    def mark_indexed(self, drug_id: str) -> None:
        """Mark a drug as indexed in the vector store."""
        with self._get_connection() as conn:
            conn.execute(
                "UPDATE drug_cache SET is_indexed = TRUE WHERE drug_id = ?",
                (drug_id,)
            )
            conn.commit()
    
    def get_unindexed_drugs(self) -> list[dict]:
        """Get list of drugs that haven't been indexed yet."""
        with self._get_connection() as conn:
            cursor = conn.execute(
                "SELECT * FROM drug_cache WHERE is_indexed = FALSE"
            )
            return [dict(row) for row in cursor.fetchall()]
    
    def log_scrape(
        self,
        drugs_found: int,
        drugs_updated: int,
        status: str = "success",
        error_message: str = ""
    ) -> None:
        """
        Log a scrape operation to history.
        
        Args:
            drugs_found: Number of drugs found
            drugs_updated: Number of drugs with updates
            status: Status of the scrape ("success" or "error")
            error_message: Error message if status is "error"
        """
        with self._get_connection() as conn:
            conn.execute("""
                INSERT INTO scrape_history 
                    (scrape_date, drugs_found, drugs_updated, status, error_message)
                VALUES (?, ?, ?, ?, ?)
            """, (datetime.now(), drugs_found, drugs_updated, status, error_message))
            conn.commit()
    
    def get_last_scrape(self) -> Optional[dict]:
        """Get information about the last scrape operation."""
        with self._get_connection() as conn:
            cursor = conn.execute("""
                SELECT * FROM scrape_history 
                ORDER BY scrape_date DESC LIMIT 1
            """)
            row = cursor.fetchone()
            return dict(row) if row else None
    
    def get_stats(self) -> dict:
        """Get cache statistics."""
        with self._get_connection() as conn:
            total = conn.execute(
                "SELECT COUNT(*) FROM drug_cache"
            ).fetchone()[0]
            
            indexed = conn.execute(
                "SELECT COUNT(*) FROM drug_cache WHERE is_indexed = TRUE"
            ).fetchone()[0]
            
            last_scrape = self.get_last_scrape()
            
            return {
                "total_drugs": total,
                "indexed_drugs": indexed,
                "pending_indexing": total - indexed,
                "last_scrape": last_scrape
            }


# Singleton instance
_metadata_cache: Optional[MetadataCache] = None


def get_metadata_cache() -> MetadataCache:
    """Get or create the global metadata cache instance."""
    global _metadata_cache
    if _metadata_cache is None:
        _metadata_cache = MetadataCache()
    return _metadata_cache
