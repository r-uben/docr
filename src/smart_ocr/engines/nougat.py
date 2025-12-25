"""Nougat OCR engine adapter.

Uses the nougat-ocr-cli tool for academic paper OCR.
CLI: https://github.com/r-uben/nougat-ocr-cli
"""

import subprocess
import tempfile
import time
from pathlib import Path

from PIL import Image

from smart_ocr.core.config import NougatConfig
from smart_ocr.core.result import PageResult, PageStatus
from smart_ocr.engines.base import BaseEngine, EngineCapabilities


class NougatEngine(BaseEngine):
    """Adapter for Nougat OCR via CLI tool.

    Uses the nougat-ocr-cli which wraps Nougat for academic papers.
    """

    def __init__(self, config: NougatConfig | None = None) -> None:
        super().__init__()
        self.config = config or NougatConfig()

    @property
    def name(self) -> str:
        return "nougat"

    @property
    def capabilities(self) -> EngineCapabilities:
        return EngineCapabilities(
            name="nougat",
            supports_pdf=True,
            supports_images=True,
            supports_batch=False,
            supports_figures=False,  # No built-in figure description
            is_local=True,
            cost_per_page=0.0,
            best_for=["academic", "scientific", "equations", "papers"],
        )

    def initialize(self) -> bool:
        """Check if nougat-ocr-cli is available."""
        if self._initialized:
            return True

        try:
            # Check if CLI is installed
            result = subprocess.run(
                ["nougat-ocr-cli", "--help"],
                capture_output=True,
                text=True,
                timeout=10,
            )
            self._initialized = result.returncode == 0
            return self._initialized
        except (subprocess.SubprocessError, FileNotFoundError):
            return False

    def process_image(self, image: Image.Image, page_num: int = 1) -> PageResult:
        """Process a single image with Nougat CLI."""
        if not self._initialized and not self.initialize():
            return self._create_error_result(page_num, "Nougat CLI not installed")

        start_time = time.time()

        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)

            # Save image
            image_file = tmp_path / "input.png"
            image.save(image_file, format="PNG")

            # Run CLI
            try:
                cmd = [
                    "nougat-ocr-cli",
                    str(image_file),
                    "--output", str(tmp_path),
                    "--model", self.config.model,
                    "--batch-size", str(self.config.batch_size),
                ]

                result = subprocess.run(
                    cmd,
                    capture_output=True,
                    text=True,
                    timeout=self.config.timeout,
                )

                if result.returncode != 0:
                    return self._create_error_result(
                        page_num,
                        f"Nougat CLI failed: {result.stderr}"
                    )

                # Read output markdown
                output_file = tmp_path / "input.md"
                if not output_file.exists():
                    return self._create_error_result(
                        page_num,
                        "Nougat CLI did not generate output file"
                    )

                text = output_file.read_text()

                processing_time = time.time() - start_time

                return self._create_success_result(
                    page_num=page_num,
                    text=text,
                    processing_time=processing_time,
                )

            except subprocess.TimeoutExpired:
                return self._create_error_result(
                    page_num,
                    f"Nougat CLI timeout after {self.config.timeout}s"
                )
            except Exception as e:
                return self._create_error_result(page_num, str(e))

    def process_pdf(self, pdf_path: Path) -> list[PageResult]:
        """Process entire PDF with Nougat CLI (batch mode)."""
        if not self._initialized and not self.initialize():
            return [self._create_error_result(1, "Nougat CLI not installed")]

        start_time = time.time()

        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)

            try:
                cmd = [
                    "nougat-ocr-cli",
                    str(pdf_path),
                    "--output", str(tmp_path),
                    "--model", self.config.model,
                    "--batch-size", str(self.config.batch_size),
                ]

                result = subprocess.run(
                    cmd,
                    capture_output=True,
                    text=True,
                    timeout=self.config.timeout * 10,  # More time for full PDF
                )

                if result.returncode != 0:
                    return [self._create_error_result(1, f"Nougat CLI failed: {result.stderr}")]

                # Read output markdown
                pdf_stem = pdf_path.stem
                output_file = tmp_path / f"{pdf_stem}.md"
                if not output_file.exists():
                    # Try alternate naming
                    output_file = tmp_path / "output.md"

                if not output_file.exists():
                    return [self._create_error_result(1, "Nougat CLI did not generate output file")]

                text = output_file.read_text()
                processing_time = time.time() - start_time

                # Nougat outputs single markdown for entire PDF
                # Return as single page result
                return [
                    self._create_success_result(
                        page_num=1,
                        text=text,
                        processing_time=processing_time,
                    )
                ]

            except subprocess.TimeoutExpired:
                return [self._create_error_result(1, f"Nougat CLI timeout after {self.config.timeout * 10}s")]
            except Exception as e:
                return [self._create_error_result(1, str(e))]
