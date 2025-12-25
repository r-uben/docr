import json
from pathlib import Path

import pytest

pytest.importorskip("rich")

from smart_ocr.core.config import AgentConfig, EngineType
from smart_ocr.core.result import OCRResult
from smart_ocr.core.document import DocumentType
from smart_ocr.pipeline.processor import OCRPipeline
from smart_ocr.pipeline.router import EngineRouter


class _StubEngine:
    """Minimal engine stub for availability checks."""

    def __init__(self, available: bool) -> None:
        self._available = available
        # Provide fields accessed by pipeline selection/cost code.
        self.name = "stub"
        self.capabilities = type("Caps", (), {"supports_figures": False, "cost_per_page": 0.0, "is_local": True})

    def is_available(self) -> bool:
        return self._available


def _make_router(config: AgentConfig) -> EngineRouter:
    # Replace real engines with stubs to control availability without external deps.
    engines = {
        EngineType.NOUGAT: _StubEngine(False),
        EngineType.DEEPSEEK: _StubEngine(False),
        EngineType.MISTRAL: _StubEngine(False),
        EngineType.GEMINI: _StubEngine(False),
    }
    return EngineRouter(config, engines)


def test_primary_override_respected() -> None:
    config = AgentConfig()
    config.use_primary_override = True
    config.primary_engine = EngineType.MISTRAL

    router = _make_router(config)
    router.engines[EngineType.MISTRAL] = _StubEngine(True)

    choice = router.select_primary(DocumentType.GENERAL)
    assert choice == EngineType.MISTRAL


def test_fallback_override_respected() -> None:
    config = AgentConfig()
    config.use_fallback_override = True
    config.fallback_engine = EngineType.DEEPSEEK

    router = _make_router(config)
    router.engines[EngineType.DEEPSEEK] = _StubEngine(True)

    choice = router.select_fallback(primary=EngineType.NOUGAT)
    assert choice == EngineType.DEEPSEEK


def test_fallback_override_skips_when_unavailable() -> None:
    config = AgentConfig()
    config.use_fallback_override = True
    config.fallback_engine = EngineType.GEMINI

    router = _make_router(config)
    # Only Mistral is available; should fall back to preference order after skipping override.
    router.engines[EngineType.MISTRAL] = _StubEngine(True)

    choice = router.select_fallback(primary=EngineType.NOUGAT)
    assert choice == EngineType.MISTRAL


def test_save_output_defaults(tmp_path: Path) -> None:
    config = AgentConfig(output_dir=tmp_path, output_format="json", include_figures=False)
    pipeline = OCRPipeline(config)

    result = OCRResult(document_path="docs/paper.pdf")

    out_file = pipeline.save_output(result)
    assert out_file.suffix == ".json"
    assert out_file.name == "paper.json"
    assert out_file.parent == tmp_path / "paper"

    metadata_path = out_file.parent / "metadata.json"
    assert metadata_path.exists()
    metadata = json.loads(metadata_path.read_text())
    assert metadata["output_file"] == str(out_file)
    assert metadata["format"] == "json"


def test_save_output_custom_dir(tmp_path: Path) -> None:
    config = AgentConfig(output_dir=tmp_path, output_format="txt", include_figures=False)
    pipeline = OCRPipeline(config)
    result = OCRResult(document_path="docs/paper.pdf")

    custom_dir = tmp_path / "custom"
    out_file = pipeline.save_output(result, custom_dir)

    assert out_file == custom_dir / "paper.txt"
    assert (custom_dir / "metadata.json").exists()
