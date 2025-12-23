#!/usr/bin/env python3
"""Comprehensive OCR benchmark: olmOCR vs DeepSeek vs Nougat.

Tests on actual document pages with text, equations, and tables.
"""
import time
from pathlib import Path
import subprocess
import tempfile
from PIL import Image
import fitz  # PyMuPDF

def extract_pdf_page(pdf_path: Path, page_num: int) -> Image.Image:
    """Extract a single page from PDF as image."""
    doc = fitz.open(pdf_path)
    page = doc[page_num]
    pix = page.get_pixmap(dpi=150)
    img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
    doc.close()
    return img


def test_ollama_ocr(model_name: str, image: Image.Image) -> tuple[str, float]:
    """Test Ollama model with OCR prompt."""
    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
        image.save(tmp.name, format="PNG")
        tmp_path = Path(tmp.name)

    start = time.time()

    cmd = [
        "ollama", "run", model_name,
        "Extract all text from this document page. Preserve formatting, equations, and structure. Output only the extracted text.",
        str(tmp_path)
    ]

    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=120
        )
        elapsed = time.time() - start
        tmp_path.unlink()

        if result.returncode != 0:
            return f"ERROR: {result.stderr}", elapsed

        return result.stdout.strip(), elapsed

    except subprocess.TimeoutExpired:
        tmp_path.unlink()
        return "TIMEOUT", 120.0
    except Exception as e:
        tmp_path.unlink()
        return f"ERROR: {e}", 0.0


def test_nougat_cli(pdf_path: Path, page_num: int) -> tuple[str, float]:
    """Test nougat-ocr-cli on a specific PDF page."""
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = Path(tmpdir)

        start = time.time()

        cmd = [
            "nougat-ocr-cli",
            str(pdf_path),
            "--output", str(tmp_path),
            "--model", "0.1.0-small",
            "--batch-size", "1",
            "--pages", str(page_num)
        ]

        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=120
            )
            elapsed = time.time() - start

            if result.returncode != 0:
                return f"ERROR: {result.stderr}", elapsed

            # Nougat outputs based on PDF filename
            output_file = tmp_path / f"{pdf_path.stem}.md"
            if not output_file.exists():
                # Try without extension
                alt_output = tmp_path / f"{pdf_path.stem.split('.')[0]}.md"
                if alt_output.exists():
                    output_file = alt_output
                else:
                    # List what files were created for debugging
                    files = list(tmp_path.glob("*.md"))
                    if files:
                        output_file = files[0]
                    else:
                        return f"ERROR: No output file found in {tmp_path}", elapsed

            text = output_file.read_text()
            return text, elapsed

        except subprocess.TimeoutExpired:
            return "TIMEOUT", 120.0
        except Exception as e:
            return f"ERROR: {e}", 0.0


def analyze_output(text: str) -> dict:
    """Analyze OCR output quality."""
    if text.startswith("ERROR") or text == "TIMEOUT":
        return {
            "status": "failed",
            "chars": 0,
            "words": 0,
            "lines": 0,
            "has_equations": False,
        }

    lines = text.strip().split('\n')
    words = text.split()

    return {
        "status": "success",
        "chars": len(text),
        "words": len(words),
        "lines": len(lines),
        "has_equations": any(c in text for c in ['\\', '$', '=', '∑', '∫']),
    }


def main():
    # Use the test PDF
    test_pdf = Path("test_paper.pdf")

    if not test_pdf.exists():
        print(f"ERROR: Test PDF not found: {test_pdf}")
        print("Please provide test_paper.pdf in the current directory")
        return
    print("=" * 70)
    print("OCR Benchmark: olmOCR vs DeepSeek vs Nougat")
    print("=" * 70)
    print(f"Test PDF: {test_pdf}")
    print()

    # Test on first 3 pages
    test_pages = [0, 1, 2]

    models = [
        ("richardyoung/olmocr2:7b-q8", "olmOCR-2-7B (Q8)", test_ollama_ocr),
        ("deepseek-ocr:latest", "DeepSeek-OCR", test_ollama_ocr),
        ("nougat-cli", "Nougat (academic)", test_nougat_cli),
    ]

    all_results = []

    for page_num in test_pages:
        print(f"\n{'='*70}")
        print(f"PAGE {page_num + 1}")
        print(f"{'='*70}")

        # Extract page
        try:
            image = extract_pdf_page(test_pdf, page_num)
            print(f"Image size: {image.size}")
        except Exception as e:
            print(f"ERROR extracting page: {e}")
            continue

        page_results = []

        for model_id, model_name, test_func in models:
            print(f"\n📊 Testing: {model_name}")
            print("-" * 50)

            try:
                if "nougat" in model_id:
                    # Nougat needs PDF file and page number
                    text, elapsed = test_func(test_pdf, page_num)
                else:
                    # Ollama models need image
                    text, elapsed = test_func(model_id, image)

                analysis = analyze_output(text)

                page_results.append({
                    "model": model_name,
                    "time": elapsed,
                    "text": text,
                    "analysis": analysis
                })

                if analysis["status"] == "failed":
                    print(f"❌ {text}")
                else:
                    print(f"⏱️  Time: {elapsed:.2f}s")
                    print(f"📝 Chars: {analysis['chars']:,} | Words: {analysis['words']:,} | Lines: {analysis['lines']}")
                    print(f"🔢 Equations: {'Yes' if analysis['has_equations'] else 'No'}")
                    print(f"First 150 chars: {text[:150]}...")

            except Exception as e:
                print(f"❌ Exception: {e}")
                page_results.append({
                    "model": model_name,
                    "time": 0.0,
                    "text": f"ERROR: {e}",
                    "analysis": {"status": "failed", "chars": 0, "words": 0, "lines": 0}
                })

        all_results.append(page_results)

    # Final Summary
    print("\n" + "=" * 70)
    print("FINAL SUMMARY")
    print("=" * 70)

    for model_id, model_name, _ in models:
        total_time = 0.0
        total_chars = 0
        success_count = 0

        for page_results in all_results:
            for result in page_results:
                if result["model"] == model_name:
                    total_time += result["time"]
                    if result["analysis"]["status"] == "success":
                        total_chars += result["analysis"]["chars"]
                        success_count += 1

        avg_time = total_time / len(all_results) if all_results else 0
        avg_chars = total_chars / success_count if success_count else 0

        print(f"\n{model_name}")
        print(f"  Success: {success_count}/{len(all_results)} pages")
        print(f"  Avg time: {avg_time:.2f}s per page")
        print(f"  Avg output: {avg_chars:.0f} chars per page")
        print(f"  Total time: {total_time:.2f}s for {len(all_results)} pages")


if __name__ == "__main__":
    main()
