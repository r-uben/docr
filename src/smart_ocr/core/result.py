"""OCR result data structures."""

from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class PageStatus(str, Enum):
    """Status of page OCR processing."""

    PENDING = "pending"
    SUCCESS = "success"
    WARNING = "warning"  # Processed but with quality concerns
    ERROR = "error"
    SKIPPED = "skipped"


@dataclass
class FigureResult:
    """Result for a detected figure."""

    figure_num: int
    page_num: int
    figure_type: str  # chart, table, diagram, image, etc.
    description: str
    bbox: tuple[int, int, int, int] | None = None  # x1, y1, x2, y2
    engine: str = ""
    image_path: str | None = None  # Path to saved figure image


@dataclass
class PageResult:
    """Result for a single page."""

    page_num: int
    text: str = ""
    status: PageStatus = PageStatus.PENDING
    engine: str = ""
    confidence: float | None = None
    processing_time: float = 0.0
    cost: float = 0.0
    error_message: str = ""
    figures: list[FigureResult] = field(default_factory=list)
    audit_passed: bool = True
    audit_notes: list[str] = field(default_factory=list)

    @property
    def word_count(self) -> int:
        """Get word count from extracted text."""
        return len(self.text.split()) if self.text else 0

    @property
    def char_count(self) -> int:
        """Get character count from extracted text."""
        return len(self.text) if self.text else 0

    def needs_reprocessing(self) -> bool:
        """Check if page needs reprocessing due to poor quality."""
        if self.status == PageStatus.ERROR:
            return True
        if not self.audit_passed:
            return True
        if self.confidence is not None and self.confidence < 0.6:
            return True
        return False


@dataclass
class ProcessingStats:
    """Statistics for the processing run."""

    total_pages: int = 0
    pages_success: int = 0
    pages_warning: int = 0
    pages_error: int = 0
    pages_skipped: int = 0
    figures_detected: int = 0
    total_time: float = 0.0
    total_cost: float = 0.0
    engines_used: dict[str, int] = field(default_factory=dict)

    def add_page(self, result: PageResult) -> None:
        """Update stats with a page result."""
        self.total_pages += 1

        if result.status == PageStatus.SUCCESS:
            self.pages_success += 1
        elif result.status == PageStatus.WARNING:
            self.pages_warning += 1
        elif result.status == PageStatus.ERROR:
            self.pages_error += 1
        elif result.status == PageStatus.SKIPPED:
            self.pages_skipped += 1

        self.figures_detected += len(result.figures)
        self.total_cost += result.cost

        if result.engine:
            self.engines_used[result.engine] = self.engines_used.get(result.engine, 0) + 1

    @property
    def success_rate(self) -> float:
        """Calculate success rate."""
        if self.total_pages == 0:
            return 0.0
        return (self.pages_success + self.pages_warning) / self.total_pages


@dataclass
class OCRResult:
    """Complete OCR result for a document."""

    document_path: str = ""
    pages: list[PageResult] = field(default_factory=list)
    stats: ProcessingStats = field(default_factory=ProcessingStats)
    metadata: dict[str, Any] = field(default_factory=dict)

    def get_page(self, page_num: int) -> PageResult | None:
        """Get result for a specific page."""
        for page in self.pages:
            if page.page_num == page_num:
                return page
        return None

    def get_full_text(self, separator: str = "\n\n") -> str:
        """Get concatenated text from all pages."""
        texts = []
        for page in sorted(self.pages, key=lambda p: p.page_num):
            if page.text:
                texts.append(f"--- Page {page.page_num} ---\n{page.text}")
        return separator.join(texts)

    def get_pages_needing_reprocessing(self) -> list[int]:
        """Get list of page numbers that need reprocessing."""
        return [p.page_num for p in self.pages if p.needs_reprocessing()]

    def add_page_result(self, result: PageResult) -> None:
        """Add a page result (replaces existing if same page number)."""
        # Replace existing if same page number
        self.pages = [p for p in self.pages if p.page_num != result.page_num]
        self.pages.append(result)

    def recalculate_stats(self) -> None:
        """Recalculate stats from all page results."""
        self.stats = ProcessingStats()
        for page in self.pages:
            self.stats.add_page(page)

    def to_markdown(self) -> str:
        """Export as markdown document."""
        from pathlib import Path
        filename = Path(self.document_path).name if self.document_path else "Unknown"
        lines = [f"# OCR Result: {filename}", ""]

        # Summary
        lines.append("## Summary")
        lines.append(f"- Pages: {self.stats.pages_success}/{self.stats.total_pages} successful")
        lines.append(f"- Figures: {self.stats.figures_detected}")
        lines.append(f"- Cost: ${self.stats.total_cost:.4f}")
        lines.append("")

        # Content with inline figures
        lines.append("## Content")
        lines.append("")

        for page in sorted(self.pages, key=lambda p: p.page_num):
            lines.append(f"--- Page {page.page_num} ---")
            if page.text:
                lines.append(page.text)

            # Include figure descriptions for this page
            if page.figures:
                lines.append("")
                for fig in page.figures:
                    lines.append(f"**[Figure {fig.figure_num}]** ({fig.figure_type})")
                    lines.append(f"> {fig.description}")
                    lines.append("")

            lines.append("")

        return "\n".join(lines)
