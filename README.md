# OCR Agent

Multi-engine OCR with cascading fallback, quality audit, and figure extraction.

Process academic papers and documents using free local models first, with automatic cloud fallback for failed pages. Extract and describe figures using vision models.

## Features

- **Multi-engine OCR** — Nougat, DeepSeek (Ollama), Mistral, Gemini
- **Smart routing** — Free local engines first, cloud only when needed
- **Quality audit** — Heuristics + LLM review catches garbage text
- **Figure extraction** — Renders figures from PDFs, describes with vision models
- **Batch processing** — Process entire directories of papers
- **CLI** — Live progress, colored panels, cost tracking

## Quick Start

```bash
# Install with Poetry
poetry install

# Install cloud engine SDKs (optional)
poetry install --extras cloud

# Process a paper
ocr-agent process paper.pdf

# Process with figure images saved
ocr-agent process paper.pdf --save-figures

# Batch process a folder
ocr-agent batch ~/Papers/ --limit 10
```

## Example Output

Processing a 22-page economics paper:

```
ocr-agent v0.1.0

kuttner_2001_monetary_policy.pdf
22 pages, 1.2 MB
type: academic

(1) primary ocr
    deepseek
    [+] page 1
    [+] page 2
    ...

(2) quality audit
    [!] page 10 (19.2% garbage)

(3) fallback ocr
    gemini
    [+] page 10

(4) figure processing
    [+] fig 1 (p.1): unknown
    [+] fig 2 (p.8): scatter_plot
    [+] fig 3 (p.12): scatter_plot

---

done 22/22 pages
     3 figures
     241.5s
     $0.0002
     deepseek (21) + gemini (1)

-> output/kuttner_2001_monetary_policy/kuttner_2001_monetary_policy.md
```

Output structure:
```
output/kuttner_2001_monetary_policy/
├── kuttner_2001_monetary_policy.md   # Full OCR text + figure descriptions
├── metadata.json                      # Stats, engines used, cost
└── figures/                           # With --save-figures
    ├── figure_1_page1.png
    ├── figure_2_page8.png
    └── figure_3_page12.png
```

## Pipeline

```
PDF -> Primary OCR -> Quality Audit -> Fallback OCR -> Figure Pass -> Output
       (DeepSeek)     (heuristics      (Gemini for     (render +
                      + Ollama LLM)    flagged pages)   describe)
```

| Stage | Engine | Cost |
|-------|--------|------|
| Primary OCR | DeepSeek via Ollama | Free |
| Quality Audit | Ollama (qwen2.5) | Free |
| Fallback | Gemini 2.0 Flash | ~$0.0002/page |
| Figures | Gemini/DeepSeek | Included |

## Requirements

**Required:**
- Python 3.10+
- [Ollama](https://ollama.ai) with `deepseek-r1:8b` model

**Optional:**
- `GEMINI_API_KEY` or `GOOGLE_API_KEY` for cloud fallback
- `MISTRAL_API_KEY` for Mistral engine

```bash
# Start Ollama and pull DeepSeek
ollama pull deepseek-r1:8b

# Check engine status
ocr-agent engines
```

## CLI Commands

```bash
# Process single PDF
ocr-agent process paper.pdf [OPTIONS]
  -o, --output PATH      Output file path
  -f, --format           markdown|json|txt
  --primary ENGINE       Force primary engine
  --fallback ENGINE      Force fallback engine
  --no-audit             Skip quality audit
  --no-figures           Skip figure processing
  --save-figures         Save figure images to disk

# Batch process directory
ocr-agent batch ~/Papers/ [OPTIONS]
  --limit N              Process first N files
  --save-figures         Save all figure images

# Check engines
ocr-agent engines

# Check audit system
ocr-agent audit-status
```

## Configuration

Create `ocr-agent.yaml` in your project or home directory:

```yaml
# Engine selection
primary_engine: deepseek
fallback_engine: gemini

# Quality audit
audit:
  enabled: true
  min_word_count: 50
  garbage_threshold: 0.15

# Figure processing
include_figures: true
save_figures: false
figures_max_total: 25
figures_max_per_page: 3

# Output
output_dir: output
output_format: markdown
```

## Architecture

See [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) for detailed module documentation.

```
src/ocr_agent/
├── cli.py              # Click CLI
├── core/
│   ├── config.py       # AgentConfig dataclass
│   ├── document.py     # PDF loading
│   └── result.py       # OCRResult, PageResult, FigureResult
├── engines/
│   ├── base.py         # BaseEngine ABC
│   ├── deepseek.py     # Ollama/DeepSeek
│   ├── gemini.py       # Google Gemini
│   ├── mistral.py      # Mistral AI
│   └── nougat.py       # Nougat (academic)
├── audit/
│   ├── heuristics.py   # Garbage detection
│   └── llm_audit.py    # Ollama-based review
├── pipeline/
│   ├── processor.py    # 4-stage pipeline
│   └── router.py       # Engine selection
└── ui/                 # Rich console output
```

## License

MIT
