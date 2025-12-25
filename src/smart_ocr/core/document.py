"""Document representation for OCR processing."""

from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import BinaryIO

from PIL import Image


class DocumentType(str, Enum):
    """Types of documents for routing decisions."""

    ACADEMIC = "academic"  # Research papers, scientific articles
    POLICY = "policy"  # Policy reports, government documents
    FINANCIAL = "financial"  # Financial reports, statements
    GENERAL = "general"  # General documents


@dataclass
class PageImage:
    """A single page as an image."""

    page_num: int
    image: Image.Image
    width: int = 0
    height: int = 0

    def __post_init__(self) -> None:
        if self.image:
            self.width, self.height = self.image.size

    @property
    def aspect_ratio(self) -> float:
        """Get page aspect ratio."""
        return self.width / max(self.height, 1)

    def is_landscape(self) -> bool:
        """Check if page is in landscape orientation."""
        return self.width > self.height


@dataclass
class Document:
    """A document to be processed."""

    path: Path
    pages: list[PageImage] = field(default_factory=list)
    doc_type: DocumentType = DocumentType.GENERAL
    detected_features: list[str] = field(default_factory=list)
    _file_size_mb: float = 0.0

    def __post_init__(self) -> None:
        if isinstance(self.path, str):
            self.path = Path(self.path)
        if self.path.exists():
            self._file_size_mb = self.path.stat().st_size / (1024 * 1024)

    @property
    def filename(self) -> str:
        """Get document filename."""
        return self.path.name

    @property
    def num_pages(self) -> int:
        """Get number of pages."""
        return len(self.pages)

    @property
    def size_mb(self) -> float:
        """Get file size in MB."""
        return self._file_size_mb

    def get_page(self, page_num: int) -> PageImage | None:
        """Get a specific page by number (1-indexed)."""
        for page in self.pages:
            if page.page_num == page_num:
                return page
        return None

    @classmethod
    def from_pdf(cls, path: Path | str, render_dpi: int | str = "auto") -> "Document":
        """Create document from a PDF file.

        Args:
            path: Path to PDF file
            render_dpi: DPI for rendering pages. Can be:
                - "auto": Smart detection based on document characteristics
                - int (e.g., 150, 200, 300): Explicit DPI value
        """
        import fitz  # PyMuPDF

        path = Path(path)
        doc = cls(path=path)

        pdf = fitz.open(path)

        # Determine DPI
        if render_dpi == "auto":
            dpi = cls._auto_detect_dpi(pdf)
        else:
            dpi = int(render_dpi)

        doc.detected_features.append(f"render_dpi={dpi}")

        for page_num in range(len(pdf)):
            page = pdf[page_num]
            mat = fitz.Matrix(dpi / 72, dpi / 72)
            pix = page.get_pixmap(matrix=mat)

            # Convert to PIL Image
            img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
            doc.pages.append(PageImage(page_num=page_num + 1, image=img))

        pdf.close()
        return doc

    @staticmethod
    def _auto_detect_dpi(pdf) -> int:
        """Auto-detect optimal DPI based on document characteristics.

        Logic:
        - Dense reports (>50 pages) → 200 DPI for small text/tables
        - Presentations (landscape) → 150 DPI (charts as figures anyway)
        - Default → 150 DPI
        """
        page_count = len(pdf)

        # Check first page orientation
        if page_count > 0:
            first_page = pdf[0]
            is_landscape = first_page.rect.width > first_page.rect.height
        else:
            is_landscape = False

        # Decision logic
        if page_count > 50:
            # Dense report - use higher DPI for tables/footnotes
            return 200
        elif is_landscape:
            # Presentation - standard DPI (charts extracted as figures)
            return 150
        else:
            # Default
            return 150

    def classify(self) -> DocumentType:
        """Classify document type based on content analysis."""
        # Simple heuristics for now - can be enhanced with ML
        filename_lower = self.filename.lower()

        # Academic indicators
        if any(kw in filename_lower for kw in ["paper", "article", "journal", "arxiv", "nber", "working"]):
            self.doc_type = DocumentType.ACADEMIC
            self.detected_features.append("academic_filename")
            return self.doc_type

        # Policy indicators
        if any(kw in filename_lower for kw in ["policy", "report", "fed", "ecb", "imf", "oecd"]):
            self.doc_type = DocumentType.POLICY
            self.detected_features.append("policy_filename")
            return self.doc_type

        # Financial indicators
        if any(kw in filename_lower for kw in ["financial", "annual", "quarterly", "10k", "10q"]):
            self.doc_type = DocumentType.FINANCIAL
            self.detected_features.append("financial_filename")
            return self.doc_type

        return self.doc_type
