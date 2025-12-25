"""Mistral OCR engine adapter.

Uses the mistral-ocr-cli tool for cloud OCR processing via Mistral AI.
CLI: https://github.com/r-uben/mistral-ocr-cli
"""

import subprocess
import tempfile
import time
from pathlib import Path

from PIL import Image

from smart_ocr.core.config import MistralConfig
from smart_ocr.core.result import FigureResult, PageResult
from smart_ocr.engines.base import BaseEngine, EngineCapabilities


class MistralEngine(BaseEngine):
    """Adapter for Mistral OCR API via CLI tool."""

    COST_PER_PAGE = 0.001  # $0.001 per page

    def __init__(self, config: MistralConfig | None = None) -> None:
        super().__init__()
        self.config = config or MistralConfig()

    @property
    def name(self) -> str:
        return "mistral"

    @property
    def capabilities(self) -> EngineCapabilities:
        return EngineCapabilities(
            name="mistral",
            supports_pdf=True,
            supports_images=True,
            supports_batch=False,
            supports_figures=True,
            is_local=False,
            cost_per_page=self.COST_PER_PAGE,
            best_for=["general", "high-accuracy", "multilingual"],
        )

    def initialize(self) -> bool:
        """Check if mistral-ocr CLI is available."""
        if self._initialized:
            return True

        if not self.config.api_key:
            return False

        try:
            # Check if CLI is installed
            result = subprocess.run(
                ["mistral-ocr", "--version"],
                capture_output=True,
                text=True,
                timeout=10,
            )
            self._initialized = result.returncode == 0
            return self._initialized
        except (subprocess.SubprocessError, FileNotFoundError):
            return False

    def process_image(self, image: Image.Image, page_num: int = 1) -> PageResult:
        """Process a single image with Mistral CLI."""
        if not self._initialized and not self.initialize():
            return self._create_error_result(page_num, "Mistral CLI not installed (check API key)")

        start_time = time.time()

        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)

            # Save image
            image_file = tmp_path / "input.png"
            image.save(image_file, format="PNG")

            # Run CLI
            try:
                cmd = [
                    "mistral-ocr",
                    str(image_file),
                    "-o", str(tmp_path),
                    "--api-key", self.config.api_key,
                    "--model", self.config.model,
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
                        f"Mistral CLI failed: {result.stderr}"
                    )

                # Read output markdown
                # mistral-ocr creates output/<filename>.md
                output_file = tmp_path / "mistral_ocr_output" / "input.md"
                if not output_file.exists():
                    # Try alternate location
                    output_file = tmp_path / "input.md"

                if not output_file.exists():
                    return self._create_error_result(
                        page_num,
                        "Mistral CLI did not generate output file"
                    )

                text = output_file.read_text()

                # Remove metadata frontmatter if present
                if text.startswith("---"):
                    parts = text.split("---", 2)
                    if len(parts) >= 3:
                        text = parts[2].strip()

                processing_time = time.time() - start_time

                return self._create_success_result(
                    page_num=page_num,
                    text=text,
                    processing_time=processing_time,
                    cost=self.COST_PER_PAGE,
                )

            except subprocess.TimeoutExpired:
                return self._create_error_result(
                    page_num,
                    f"Mistral CLI timeout after {self.config.timeout}s"
                )
            except Exception as e:
                return self._create_error_result(page_num, str(e))

    def describe_figure(
        self,
        image: Image.Image,
        figure_type: str = "unknown",
        context: str = "",
    ) -> FigureResult:
        """Describe a figure using Mistral vision via CLI."""
        if not self._initialized and not self.initialize():
            return FigureResult(
                figure_num=0,
                page_num=0,
                figure_type=figure_type,
                description="Mistral CLI not installed",
            )

        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)

            # Save image
            image_file = tmp_path / "figure.png"
            image.save(image_file, format="PNG")

            try:
                cmd = [
                    "mistral-ocr",
                    str(image_file),
                    "-o", str(tmp_path),
                    "--api-key", self.config.api_key,
                    "--model", self.config.model,
                ]

                result = subprocess.run(
                    cmd,
                    capture_output=True,
                    text=True,
                    timeout=60,
                )

                if result.returncode != 0:
                    return FigureResult(
                        figure_num=0,
                        page_num=0,
                        figure_type=figure_type,
                        description=f"Error: {result.stderr}",
                    )

                # Read output
                output_file = tmp_path / "mistral_ocr_output" / "figure.md"
                if not output_file.exists():
                    output_file = tmp_path / "figure.md"

                if output_file.exists():
                    description = output_file.read_text().strip()

                    # Remove metadata if present
                    if description.startswith("---"):
                        parts = description.split("---", 2)
                        if len(parts) >= 3:
                            description = parts[2].strip()
                else:
                    description = "No description generated"

                # Try to extract figure type from response
                detected_type = figure_type
                for t in ["chart", "graph", "table", "diagram", "photo", "image"]:
                    if t in description.lower():
                        detected_type = t
                        break

                return FigureResult(
                    figure_num=0,
                    page_num=0,
                    figure_type=detected_type,
                    description=description,
                    engine=self.name,
                )

            except Exception as e:
                return FigureResult(
                    figure_num=0,
                    page_num=0,
                    figure_type=figure_type,
                    description=f"Error: {e}",
                )
