"""CLI for OCR Agent - Multi-Engine Document Processing."""

from pathlib import Path

import click
from rich.console import Console

from ocr_agent import __version__
from ocr_agent.core.config import AgentConfig, EngineType
from ocr_agent.pipeline.processor import OCRPipeline
from ocr_agent.ui.theme import AGENT_THEME, ENGINE_LABELS


console = Console(theme=AGENT_THEME)


@click.group()
@click.version_option(version=__version__, prog_name="ocr-agent")
def cli() -> None:
    """OCR Agent - Multi-Engine Document Processing.

    A multi-agent OCR system that uses cascading fallback
    between local and cloud engines for optimal quality and cost.
    """
    pass


@cli.command()
@click.argument("pdf_path", type=click.Path(exists=True, path_type=Path))
@click.option(
    "-o", "--output",
    type=click.Path(path_type=Path),
    help="Output file path (default: output/<doc_stem>/<doc_stem>.<ext>)",
)
@click.option(
    "-f", "--format",
    type=click.Choice(["markdown", "json", "txt"]),
    default="markdown",
    help="Output format",
)
@click.option(
    "--primary",
    type=click.Choice(["nougat", "deepseek", "mistral", "gemini"]),
    help="Override primary engine selection",
)
@click.option(
    "--fallback",
    type=click.Choice(["nougat", "deepseek", "mistral", "gemini"]),
    help="Override fallback engine selection",
)
@click.option(
    "--no-audit",
    is_flag=True,
    help="Skip quality audit stage",
)
@click.option(
    "--no-figures",
    is_flag=True,
    help="Skip figure processing stage",
)
@click.option(
    "--save-figures",
    is_flag=True,
    help="Save extracted figure images to output/<doc>/figures/",
)
@click.option(
    "-v", "--verbose",
    is_flag=True,
    help="Enable verbose output",
)
def process(
    pdf_path: Path,
    output: Path | None,
    format: str,
    primary: str | None,
    fallback: str | None,
    no_audit: bool,
    no_figures: bool,
    save_figures: bool,
    verbose: bool,
) -> None:
    """Process a PDF document with multi-agent OCR.

    Uses cascading fallback: free local engines first (Nougat/DeepSeek),
    quality audit with local LLM, then cloud fallback (Mistral/Gemini)
    for failed pages.

    Example:
        ocr-agent process paper.pdf -o extracted.md
    """
    config = AgentConfig(
        output_format=format,
        include_figures=not no_figures,
        save_figures=save_figures,
        verbose=verbose,
    )

    if no_audit:
        config.audit.enabled = False

    if primary:
        config.primary_engine = EngineType(primary)
        config.use_primary_override = True
    if fallback:
        config.fallback_engine = EngineType(fallback)
        config.use_fallback_override = True

    pipeline = OCRPipeline(config)

    try:
        result = pipeline.process(pdf_path)

        # Save output
        output_path = pipeline.save_output(result, output)
        console.print(f"\n[success]✓[/success] Output saved to: [info]{output_path}[/info]")

    except KeyboardInterrupt:
        console.print("\n[warning]⚠ Processing cancelled[/warning]")
        raise click.Abort()
    except Exception as e:
        console.print(f"\n[error]✗ Error:[/error] {e}")
        raise click.ClickException(str(e))


@cli.command()
@click.argument("pdf_dir", type=click.Path(exists=True, path_type=Path))
@click.option(
    "-o", "--output-dir",
    type=click.Path(path_type=Path),
    help="Output directory (default: output/)",
)
@click.option(
    "-f", "--format",
    type=click.Choice(["markdown", "json", "txt"]),
    default="markdown",
    help="Output format",
)
@click.option(
    "--primary",
    type=click.Choice(["nougat", "deepseek", "mistral", "gemini"]),
    help="Override primary engine selection",
)
@click.option(
    "--no-audit",
    is_flag=True,
    help="Skip quality audit stage",
)
@click.option(
    "--no-figures",
    is_flag=True,
    help="Skip figure processing stage",
)
@click.option(
    "--save-figures",
    is_flag=True,
    help="Save extracted figure images",
)
@click.option(
    "--limit",
    type=int,
    help="Maximum number of PDFs to process",
)
def batch(
    pdf_dir: Path,
    output_dir: Path | None,
    format: str,
    primary: str | None,
    no_audit: bool,
    no_figures: bool,
    save_figures: bool,
    limit: int | None,
) -> None:
    """Process all PDFs in a directory.

    Example:
        ocr-agent batch ~/Papers/ -o extracted/
    """
    # Find all PDFs
    pdf_files = sorted(pdf_dir.glob("*.pdf"))
    if not pdf_files:
        console.print(f"[warning]No PDF files found in {pdf_dir}[/warning]")
        return

    if limit:
        pdf_files = pdf_files[:limit]

    console.print(f"\n[bold]Batch Processing: {len(pdf_files)} PDFs[/bold]\n")

    config = AgentConfig(
        output_format=format,
        include_figures=not no_figures,
        save_figures=save_figures,
    )
    if output_dir:
        config.output_dir = output_dir
    if no_audit:
        config.audit.enabled = False
    if primary:
        config.primary_engine = EngineType(primary)
        config.use_primary_override = True

    pipeline = OCRPipeline(config)
    results: list[tuple[Path, bool, str]] = []

    for i, pdf_path in enumerate(pdf_files, 1):
        console.print(f"\n[dim][{i}/{len(pdf_files)}][/dim] {pdf_path.name}")
        try:
            result = pipeline.process(pdf_path)
            output_path = pipeline.save_output(result)
            results.append((pdf_path, True, str(output_path)))
            console.print(f"  [success]\\[+][/success] {output_path}")
        except KeyboardInterrupt:
            console.print("\n[warning]\\[!] cancelled[/warning]")
            break
        except Exception as e:
            results.append((pdf_path, False, str(e)))
            console.print(f"  [error]\\[x][/error] {e}")

    # Summary
    success = sum(1 for _, ok, _ in results if ok)
    failed = len(results) - success
    console.print(f"\n[bold]Summary:[/bold] {success} succeeded, {failed} failed")


@cli.command()
def engines() -> None:
    """Show available OCR engines and their status."""
    console.print("\n[header]engines[/header]\n")

    from ocr_agent.engines import (
        DeepSeekEngine,
        GeminiEngine,
        MistralEngine,
        NougatEngine,
    )

    engines_info = [
        ("nougat", NougatEngine(), "local, academic papers"),
        ("deepseek", DeepSeekEngine(), "local via ollama, general"),
        ("mistral", MistralEngine(), "cloud, $0.001/page"),
        ("gemini", GeminiEngine(), "cloud, $0.0002/page"),
    ]

    for name, engine, desc in engines_info:
        label = ENGINE_LABELS.get(name, name)
        available = engine.is_available()

        status = "+" if available else "x"
        style = "success" if available else "error"

        console.print(f"  [{style}]\\[{status}][/{style}] [{name}]{label}[/{name}] [dim]{desc}[/dim]")


@cli.command()
@click.option(
    "--ollama-host",
    default="http://localhost:11434",
    help="Ollama server URL",
)
def audit_status(ollama_host: str) -> None:
    """Check quality audit system status."""
    console.print("\n[header]audit[/header]\n")

    from ocr_agent.audit.llm_audit import LLMAuditor

    auditor = LLMAuditor(ollama_host=ollama_host)
    ollama_ok = auditor.is_available()

    status = "+" if ollama_ok else "x"
    style = "success" if ollama_ok else "error"
    
    console.print(f"  [{style}]\\[{status}][/{style}] [ollama]ollama[/ollama] [dim]{ollama_host}[/dim]")

    if ollama_ok:
        console.print(f"      [dim]model: {auditor.model}[/dim]")
        console.print("\n  [success]ready[/success]")
    else:
        console.print("\n  [warning]heuristics only (start ollama for llm audit)[/warning]")


@cli.command()
@click.argument("pdf_path", type=click.Path(exists=True, path_type=Path))
@click.option(
    "--engine",
    type=click.Choice(["nougat", "deepseek", "mistral", "gemini"]),
    default="gemini",
    help="Engine to use for description",
)
def describe_figures(pdf_path: Path, engine: str) -> None:
    """Extract and describe figures from a PDF.

    Uses vision-capable models to generate descriptions
    for charts, tables, and diagrams.
    """
    console.print(f"\n{pdf_path.name}\n")
    console.print("[warning][!] experimental[/warning]")
    console.print("[dim]use `ocr-agent process` with figures enabled[/dim]")


# Shorthand aliases
@cli.command("p")
@click.argument("pdf_path", type=click.Path(exists=True, path_type=Path))
@click.pass_context
def process_shorthand(ctx: click.Context, pdf_path: Path) -> None:
    """Shorthand for 'process' command."""
    ctx.invoke(process, pdf_path=pdf_path)


def main() -> None:
    """Entry point."""
    cli()


if __name__ == "__main__":
    main()
