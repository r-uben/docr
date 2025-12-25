"""Core data models for smart-ocr."""

from smart_ocr.core.config import AgentConfig
from smart_ocr.core.document import Document, PageImage
from smart_ocr.core.result import OCRResult, PageResult, ProcessingStats

__all__ = [
    "AgentConfig",
    "Document",
    "PageImage",
    "OCRResult",
    "PageResult",
    "ProcessingStats",
]
