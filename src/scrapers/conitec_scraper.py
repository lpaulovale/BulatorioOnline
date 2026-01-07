"""
CONITEC Clinical Protocols Scraper

Scrapes clinical protocols (PCDT) from the Brazilian government's CONITEC portal.
PCDT = Protocolos Clínicos e Diretrizes Terapêuticas

Source: https://www.gov.br/conitec/pt-br
"""

import asyncio
import logging
import re
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Optional

import httpx
from bs4 import BeautifulSoup

logger = logging.getLogger(__name__)


@dataclass
class ClinicalProtocol:
    """Represents a PCDT clinical protocol from CONITEC."""
    
    id: str
    name: str
    disease: str
    publication_date: Optional[datetime]
    portaria_number: str
    pdf_url: Optional[str]
    summary_pdf_url: Optional[str]
    text_content: Optional[str] = None
    
    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            "id": self.id,
            "name": self.name,
            "disease": self.disease,
            "publication_date": self.publication_date.isoformat() if self.publication_date else None,
            "portaria_number": self.portaria_number,
            "pdf_url": self.pdf_url,
            "summary_pdf_url": self.summary_pdf_url,
            "has_content": bool(self.text_content)
        }


class CONITECScraper:
    """
    Scraper for CONITEC clinical protocols (PCDT).
    
    The portal is available at:
    https://www.gov.br/conitec/pt-br/assuntos/avaliacao-de-tecnologias-em-saude/protocolos-clinicos-e-diretrizes-terapeuticas/pcdt
    """
    
    BASE_URL = "https://www.gov.br/conitec/pt-br"
    PCDT_URL = f"{BASE_URL}/assuntos/avaliacao-de-tecnologias-em-saude/protocolos-clinicos-e-diretrizes-terapeuticas/pcdt"
    
    def __init__(
        self,
        cache_dir: Optional[Path] = None,
        timeout: float = 30.0,
        max_retries: int = 3
    ):
        """
        Initialize the CONITEC scraper.
        
        Args:
            cache_dir: Directory to cache downloaded PDFs
            timeout: Request timeout in seconds
            max_retries: Maximum retry attempts for failed requests
        """
        self.cache_dir = cache_dir or Path("data/protocols_cache")
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.timeout = timeout
        self.max_retries = max_retries
        self._client: Optional[httpx.AsyncClient] = None
    
    async def __aenter__(self):
        """Async context manager entry."""
        self._client = httpx.AsyncClient(
            timeout=self.timeout,
            headers={
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
                "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
                "Accept-Language": "pt-BR,pt;q=0.9,en;q=0.8"
            },
            follow_redirects=True
        )
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        if self._client:
            await self._client.aclose()
    
    async def search_protocols(
        self,
        query: str,
        max_results: int = 10
    ) -> list[ClinicalProtocol]:
        """
        Search for clinical protocols by disease or keyword.
        
        Args:
            query: Search term (disease name or keyword)
            max_results: Maximum number of results
            
        Returns:
            List of matching ClinicalProtocol objects
        """
        protocols = await self._fetch_protocol_list()
        
        # Filter by query
        query_lower = query.lower()
        matching = [
            p for p in protocols
            if query_lower in p.name.lower() or query_lower in p.disease.lower()
        ]
        
        return matching[:max_results]
    
    async def _fetch_protocol_list(self) -> list[ClinicalProtocol]:
        """
        Fetch the complete list of protocols from CONITEC.
        
        Returns:
            List of all available protocols
        """
        if not self._client:
            raise RuntimeError("Scraper not initialized. Use async with statement.")
        
        protocols = []
        
        try:
            response = await self._client.get(self.PCDT_URL)
            response.raise_for_status()
            
            soup = BeautifulSoup(response.text, 'html.parser')
            
            # Find all protocol links (they link to PDF or protocol pages)
            links = soup.find_all('a', href=True)
            
            for link in links:
                href = link.get('href', '')
                text = link.get_text(strip=True)
                
                # Skip non-protocol links
                if not text or len(text) < 3:
                    continue
                
                # Look for protocol PDFs or pages
                if '/midias/protocolos/' in href or 'pcdt' in href.lower():
                    protocol = self._parse_protocol_link(link, soup)
                    if protocol:
                        protocols.append(protocol)
            
            # Deduplicate by name
            seen = set()
            unique_protocols = []
            for p in protocols:
                if p.name not in seen:
                    seen.add(p.name)
                    unique_protocols.append(p)
            
            logger.info(f"Found {len(unique_protocols)} protocols")
            return unique_protocols
            
        except httpx.RequestError as e:
            logger.error(f"Failed to fetch protocol list: {e}")
            return []
    
    def _parse_protocol_link(
        self,
        link,
        soup: BeautifulSoup
    ) -> Optional[ClinicalProtocol]:
        """Parse a protocol link into a ClinicalProtocol object."""
        try:
            href = link.get('href', '')
            text = link.get_text(strip=True)
            
            # Skip summary/resumido links
            if 'resumido' in text.lower():
                return None
            
            # Skip navigation links
            if text in ('<<<', 'Topo', 'PCDT', 'DDT'):
                return None
            
            # Generate ID from name
            protocol_id = re.sub(r'[^\w]', '_', text.lower())[:50]
            
            # Determine PDF URL
            pdf_url = None
            if href.endswith('.pdf'):
                if href.startswith('http'):
                    pdf_url = href
                else:
                    pdf_url = f"{self.BASE_URL}{href}"
            elif '/midias/protocolos/' in href:
                if href.startswith('http'):
                    pdf_url = href
                else:
                    pdf_url = f"https://www.gov.br{href}"
            
            # Try to find portaria info (next sibling or parent text)
            portaria = ""
            next_el = link.find_next_sibling(string=True)
            if next_el:
                portaria = str(next_el).strip()[:100]
            
            return ClinicalProtocol(
                id=protocol_id,
                name=text,
                disease=text,  # Disease name is usually the protocol name
                publication_date=None,
                portaria_number=portaria,
                pdf_url=pdf_url,
                summary_pdf_url=None
            )
            
        except Exception as e:
            logger.debug(f"Failed to parse protocol link: {e}")
            return None
    
    async def get_protocol_details(
        self,
        protocol_id: str
    ) -> Optional[ClinicalProtocol]:
        """
        Get detailed information about a specific protocol.
        
        Args:
            protocol_id: Protocol identifier
            
        Returns:
            ClinicalProtocol with full details or None
        """
        protocols = await self.search_protocols(protocol_id.replace('_', ' '))
        
        for p in protocols:
            if p.id == protocol_id or protocol_id in p.id:
                return p
        
        return None
    
    async def download_pdf(
        self,
        pdf_url: str,
        protocol_id: str
    ) -> Optional[Path]:
        """
        Download a protocol PDF and cache it locally.
        
        Args:
            pdf_url: URL to the PDF file
            protocol_id: Protocol ID for naming the cached file
            
        Returns:
            Path to the downloaded file or None if failed
        """
        if not self._client:
            raise RuntimeError("Scraper not initialized.")
        
        cache_path = self.cache_dir / f"{protocol_id}.pdf"
        
        # Return cached file if exists
        if cache_path.exists():
            logger.debug(f"Using cached PDF: {cache_path}")
            return cache_path
        
        try:
            response = await self._client.get(pdf_url)
            response.raise_for_status()
            
            cache_path.write_bytes(response.content)
            logger.info(f"Downloaded PDF: {cache_path}")
            return cache_path
            
        except httpx.RequestError as e:
            logger.error(f"Failed to download PDF: {e}")
            return None
    
    async def extract_text_from_pdf(self, pdf_path: Path) -> Optional[str]:
        """
        Extract text content from a PDF file.
        
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
            
        except ImportError:
            logger.warning("pdfplumber not installed. Cannot extract PDF text.")
            return None
        except Exception as e:
            logger.error(f"Failed to extract PDF text: {e}")
            return None


# Common protocols for quick access
COMMON_PROTOCOLS = [
    "Diabetes Mellitus Tipo 2",
    "Hipertensão Arterial Sistêmica",
    "Artrite Reumatoide",
    "Esclerose Múltipla",
    "Epilepsia",
    "Dor Crônica",
    "Asma",
    "Doença Pulmonar Obstrutiva Crônica",
    "Hepatite C",
    "HIV/AIDS",
]
