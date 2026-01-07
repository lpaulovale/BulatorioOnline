"""
ANVISA Bulário Eletrônico Web Scraper

Scrapes drug bulletin (bula) information from ANVISA's public portal.
Since ANVISA doesn't provide a public API, web scraping is the only option.
"""

import asyncio
import hashlib
import logging
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Optional

import httpx
from bs4 import BeautifulSoup

logger = logging.getLogger(__name__)


@dataclass
class DrugBulletin:
    """Represents a drug bulletin (bula) from ANVISA."""
    
    id: str
    name: str
    company: str
    active_ingredient: str
    bulletin_type: str  # "paciente" or "profissional"
    pdf_url: Optional[str]
    text_content: Optional[str]
    last_updated: datetime
    
    def content_hash(self) -> str:
        """Generate hash of content for change detection."""
        content = f"{self.name}{self.company}{self.text_content or ''}"
        return hashlib.md5(content.encode()).hexdigest()


class AnvisaScraper:
    """
    Scraper for ANVISA Bulário Eletrônico.
    
    The Bulário Eletrônico is available at:
    https://consultas.anvisa.gov.br/#/bulario/
    
    This scraper fetches drug information and PDF bulletins.
    """
    
    BASE_URL = "https://consultas.anvisa.gov.br/api/consulta/bulario"
    SEARCH_URL = f"{BASE_URL}/medicamentos"
    
    def __init__(
        self,
        cache_dir: Optional[Path] = None,
        timeout: float = 30.0,
        max_retries: int = 3
    ):
        """
        Initialize the ANVISA scraper.
        
        Args:
            cache_dir: Directory to cache downloaded PDFs
            timeout: Request timeout in seconds
            max_retries: Maximum retry attempts for failed requests
        """
        self.cache_dir = cache_dir or Path("./data/pdfs")
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.timeout = timeout
        self.max_retries = max_retries
        self._client: Optional[httpx.AsyncClient] = None
    
    async def __aenter__(self):
        """Async context manager entry."""
        self._client = httpx.AsyncClient(
            timeout=self.timeout,
            headers={
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
                "Accept": "application/json, text/plain, */*",
                "Accept-Language": "pt-BR,pt;q=0.9,en-US;q=0.8,en;q=0.7",
                "Accept-Encoding": "gzip, deflate, br",
                "Referer": "https://consultas.anvisa.gov.br/",
                "Origin": "https://consultas.anvisa.gov.br",
                "Sec-Fetch-Dest": "empty",
                "Sec-Fetch-Mode": "cors",
                "Sec-Fetch-Site": "same-origin",
            },
            follow_redirects=True
        )
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        if self._client:
            await self._client.aclose()
    
    async def search_drugs(
        self,
        query: str,
        page: int = 1,
        page_size: int = 10
    ) -> list[dict]:
        """
        Search for drugs by name or active ingredient.
        
        Args:
            query: Search term (drug name or active ingredient)
            page: Page number for pagination
            page_size: Results per page
            
        Returns:
            List of drug information dictionaries
        """
        if not self._client:
            raise RuntimeError("Scraper not initialized. Use async with context.")
        
        params = {
            "filter[nomeProduto]": query,
            "page": page,
            "pageSize": page_size
        }
        
        for attempt in range(self.max_retries):
            try:
                response = await self._client.get(self.SEARCH_URL, params=params)
                response.raise_for_status()
                data = response.json()
                return data.get("content", [])
                
            except httpx.HTTPStatusError as e:
                if e.response.status_code == 403:
                    wait_time = 2 ** attempt  # Exponential backoff: 1s, 2s, 4s
                    logger.warning(f"403 Forbidden. Waiting {wait_time}s before retry {attempt + 1}/{self.max_retries}")
                    await asyncio.sleep(wait_time)
                else:
                    logger.error(f"HTTP error searching drugs: {e}")
                    return []
            except httpx.HTTPError as e:
                logger.error(f"Error searching drugs: {e}")
                if attempt == self.max_retries - 1:
                    return []
                await asyncio.sleep(1)
        
        return []
    
    async def get_bulletin_details(self, drug_id: str) -> Optional[DrugBulletin]:
        """
        Get detailed bulletin information for a specific drug.
        
        Args:
            drug_id: ANVISA drug registration ID
            
        Returns:
            DrugBulletin object or None if not found
        """
        if not self._client:
            raise RuntimeError("Scraper not initialized. Use async with context.")
        
        try:
            url = f"{self.BASE_URL}/{drug_id}"
            response = await self._client.get(url)
            response.raise_for_status()
            data = response.json()
            
            return DrugBulletin(
                id=str(drug_id),
                name=data.get("nomeProduto", ""),
                company=data.get("razaoSocial", ""),
                active_ingredient=data.get("principioAtivo", ""),
                bulletin_type=data.get("tipoBula", "paciente"),
                pdf_url=data.get("urlBula"),
                text_content=None,  # Will be filled after PDF extraction
                last_updated=datetime.now()
            )
            
        except httpx.HTTPError as e:
            logger.error(f"Error getting bulletin details: {e}")
            return None
    
    async def download_pdf(self, pdf_url: str, drug_id: str) -> Optional[Path]:
        """
        Download a PDF bulletin and cache it locally.
        
        Args:
            pdf_url: URL to the PDF file
            drug_id: Drug ID for naming the cached file
            
        Returns:
            Path to the downloaded file or None if failed
        """
        if not self._client:
            raise RuntimeError("Scraper not initialized. Use async with context.")
        
        cache_path = self.cache_dir / f"{drug_id}.pdf"
        
        # Check cache first
        if cache_path.exists():
            logger.debug(f"Using cached PDF for {drug_id}")
            return cache_path
        
        try:
            response = await self._client.get(pdf_url, follow_redirects=True)
            response.raise_for_status()
            
            cache_path.write_bytes(response.content)
            logger.info(f"Downloaded PDF for {drug_id}")
            return cache_path
            
        except httpx.HTTPError as e:
            logger.error(f"Error downloading PDF: {e}")
            return None
    
    async def extract_text_from_pdf(self, pdf_path: Path) -> Optional[str]:
        """
        Extract text content from a PDF file.
        
        Uses pdfplumber for reliable text extraction.
        
        Args:
            pdf_path: Path to the PDF file
            
        Returns:
            Extracted text or None if extraction failed
        """
        try:
            import pdfplumber
            
            text_parts = []
            with pdfplumber.open(pdf_path) as pdf:
                for page in pdf.pages:
                    page_text = page.extract_text()
                    if page_text:
                        text_parts.append(page_text)
            
            return "\n\n".join(text_parts) if text_parts else None
            
        except Exception as e:
            logger.error(f"Error extracting text from PDF: {e}")
            return None
    
    async def fetch_and_process_bulletin(
        self,
        drug_id: str
    ) -> Optional[DrugBulletin]:
        """
        Fetch bulletin, download PDF, and extract text.
        
        This is the main method for getting complete drug information.
        
        Args:
            drug_id: ANVISA drug registration ID
            
        Returns:
            Complete DrugBulletin with text content
        """
        bulletin = await self.get_bulletin_details(drug_id)
        if not bulletin or not bulletin.pdf_url:
            return bulletin
        
        pdf_path = await self.download_pdf(bulletin.pdf_url, drug_id)
        if pdf_path:
            bulletin.text_content = await self.extract_text_from_pdf(pdf_path)
        
        return bulletin


async def scrape_popular_drugs(scraper: AnvisaScraper) -> list[DrugBulletin]:
    """
    Scrape information for commonly used drugs in Brazil.
    
    This provides initial data for the vector database.
    """
    popular_drugs = [
        "paracetamol",
        "dipirona",
        "ibuprofeno",
        "amoxicilina",
        "omeprazol",
        "losartana",
        "metformina",
        "atenolol",
        "sinvastatina",
        "captopril"
    ]
    
    bulletins = []
    for drug_name in popular_drugs:
        logger.info(f"Searching for: {drug_name}")
        results = await scraper.search_drugs(drug_name, page_size=1)
        
        if results:
            drug_data = results[0]
            drug_id = str(drug_data.get("idProduto", ""))
            
            if drug_id:
                bulletin = await scraper.fetch_and_process_bulletin(drug_id)
                if bulletin:
                    bulletins.append(bulletin)
        
        # Be respectful to ANVISA servers
        await asyncio.sleep(1)
    
    return bulletins
