"""Base engine adapter for OCR processing."""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from PIL import Image

from smart_ocr.core.result import FigureResult, PageResult, PageStatus


@dataclass
class EngineCapabilities:
    """Capabilities of an OCR engine."""

    name: str
    supports_pdf: bool = False
    supports_images: bool = True
    supports_batch: bool = False
    supports_figures: bool = False
    is_local: bool = True
    cost_per_page: float = 0.0  # 0 means free
    best_for: list[str] | None = None  # e.g., ["academic", "scientific"]

    def __post_init__(self) -> None:
        if self.best_for is None:
            self.best_for = []


class BaseEngine(ABC):
    """Abstract base class for OCR engines."""

    def __init__(self) -> None:
        self._initialized = False

    @property
    @abstractmethod
    def name(self) -> str:
        """Engine identifier."""
        ...

    @property
    @abstractmethod
    def capabilities(self) -> EngineCapabilities:
        """Engine capabilities."""
        ...

    @abstractmethod
    def initialize(self) -> bool:
        """Initialize the engine. Returns True if successful."""
        ...

    @abstractmethod
    def process_image(self, image: Image.Image, page_num: int = 1) -> PageResult:
        """Process a single image and return OCR result."""
        ...

    def process_pdf_page(self, pdf_path: Path, page_num: int) -> PageResult:
        """Process a single page from a PDF. Default extracts and processes image."""
        import fitz

        pdf = fitz.open(pdf_path)
        page = pdf[page_num - 1]  # 0-indexed

        # Render at 150 DPI
        mat = fitz.Matrix(150 / 72, 150 / 72)
        pix = page.get_pixmap(matrix=mat)
        img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)

        pdf.close()
        return self.process_image(img, page_num)

    def process_pdf(self, pdf_path: Path) -> list[PageResult]:
        """Process an entire PDF. Override for batch processing."""
        import fitz

        results = []
        pdf = fitz.open(pdf_path)
        num_pages = len(pdf)
        pdf.close()

        for page_num in range(1, num_pages + 1):
            result = self.process_pdf_page(pdf_path, page_num)
            results.append(result)

        return results

    def describe_figure(
        self,
        image: Image.Image,
        figure_type: str = "unknown",
        context: str = "",
    ) -> FigureResult:
        """Describe a figure. Override for figure-capable engines."""
        return FigureResult(
            figure_num=0,
            page_num=0,
            figure_type=figure_type,
            description="Figure description not supported by this engine",
        )

    def is_available(self) -> bool:
        """Check if engine is available and ready to use."""
        try:
            return self.initialize()
        except Exception:
            return False

    def _create_error_result(
        self,
        page_num: int,
        error_message: str,
    ) -> PageResult:
        """Create an error result."""
        return PageResult(
            page_num=page_num,
            status=PageStatus.ERROR,
            engine=self.name,
            error_message=error_message,
        )

    def _create_success_result(
        self,
        page_num: int,
        text: str,
        confidence: float | None = None,
        processing_time: float = 0.0,
        cost: float = 0.0,
    ) -> PageResult:
        """Create a success result."""
        return PageResult(
            page_num=page_num,
            text=text,
            status=PageStatus.SUCCESS,
            engine=self.name,
            confidence=confidence,
            processing_time=processing_time,
            cost=cost,
        )
