import io
from pathlib import Path

import pytest

fitz = pytest.importorskip("fitz")
PIL = pytest.importorskip("PIL")

from smart_ocr.core.config import AgentConfig, EngineType
from smart_ocr.core.document import Document
from smart_ocr.core.result import FigureResult, OCRResult, PageResult
from smart_ocr.pipeline.processor import OCRPipeline


class _VisionStub:
    """Stub vision engine that always returns the same figure description."""

    def __init__(self, name: str = "vision-stub") -> None:
        self.name = name
        self.capabilities = type(
            "Caps",
            (),
            {"supports_figures": True, "cost_per_page": 0.0, "is_local": True},
        )

    def is_available(self) -> bool:
        return True

    def describe_figure(self, image, figure_type: str = "unknown", context: str = "") -> FigureResult:
        return FigureResult(
            figure_num=0,
            page_num=0,
            figure_type="chart",
            description="stub description",
            engine=self.name,
        )


def _make_pdf_with_image(tmp_path: Path) -> Path:
    img_bytes = io.BytesIO()
    from PIL import Image

    img = Image.new("RGB", (200, 100), color="red")
    img.save(img_bytes, format="PNG")
    img_bytes = img_bytes.getvalue()

    doc = fitz.open()
    page = doc.new_page()
    page.insert_image(page.rect, stream=img_bytes)

    pdf_path = tmp_path / "with_image.pdf"
    doc.save(pdf_path)
    doc.close()
    return pdf_path


def test_figure_pass_extracts_and_describes(tmp_path: Path) -> None:
    pdf_path = _make_pdf_with_image(tmp_path)

    config = AgentConfig(include_figures=True, output_dir=tmp_path)
    pipeline = OCRPipeline(config)

    # Swap engines to use vision stub
    stub = _VisionStub()
    pipeline.engines[EngineType.GEMINI] = stub
    pipeline.engines[EngineType.DEEPSEEK] = stub
    pipeline.engines[EngineType.MISTRAL] = stub

    # Build document and result with a single page placeholder
    document = Document.from_pdf(pdf_path)
    result = OCRResult(document_path=str(pdf_path))
    result.add_page_result(PageResult(page_num=1, text="context"))

    pipeline._run_stage4(document, result)

    page = result.get_page(1)
    assert page is not None
    assert len(page.figures) == 1
    fig = page.figures[0]
    assert fig.description == "stub description"
    assert fig.figure_num == 1
    assert fig.page_num == 1
