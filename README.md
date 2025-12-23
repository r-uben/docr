# docr

Multi-engine OCR with cascading fallback, quality audit, and figure extraction.

Process academic papers and documents using free local models first, with automatic cloud fallback for failed pages. Extract and describe figures using vision models.

## Features

- **Multi-engine OCR** — Nougat, DeepSeek, Mistral, Gemini (via dedicated CLI tools)
- **Smart routing** — Free local engines first, cloud only when needed
- **Quality audit** — Heuristics + LLM review catches garbage text
- **Figure extraction** — Renders figures from PDFs, describes with vision models
- **Batch processing** — Process entire directories of papers
- **CLI** — Live progress, colored panels, cost tracking
- **Modular engines** — Each OCR backend is a standalone CLI tool

## Quick Start

```bash
# Install globally (recommended)
pipx install docr --python python3.12

# Or install in a project
pip install docr

# Pull required Ollama models
ollama pull deepseek-r1:32b    # quality audit
ollama pull deepseek-ocr       # local OCR

# Process a paper
docr paper.pdf

# Process with figure images saved
docr paper.pdf --save-figures

# Batch process a folder
docr batch ~/Papers/ --limit 10
```

## Example Output

Processing a 22-page economics paper:

```
docr v0.1.0

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
output/<doc_stem>/
├── <doc_stem>.md      # Full OCR text + figure descriptions
├── metadata.json      # Stats, engines used, cost
└── figures/           # With --save-figures
    ├── figure_1_page1.png
    └── ...
```

The `metadata.json` contains:
```json
{
  "document": "/path/to/paper.pdf",
  "stats": {
    "total_pages": 22,
    "pages_success": 22,
    "total_cost": 0.0002,
    "total_time": 241.5
  },
  "engines_used": { "deepseek": 21, "gemini": 1 },
  "figures": 3,
  "pages_needing_reprocessing": []
}
```

See `examples/` for complete processed papers:
- `kuttner_2001/` - 22 pages, 3 figures (monetary policy)
- `sutskever_2014/` - 9 pages (sequence to sequence learning)
- `bernanke_kuttner_2005/` - 37 pages, 6 vector figures (stock market reaction)

## Pipeline Architecture

```
                                    ┌─────────────┐
                                    │  PDF Input  │
                                    └──────┬──────┘
                                           │
                                           ▼
                              ┌────────────────────────┐
                              │  Document Classifier   │
                              │  (academic vs general) │
                              └───────────┬────────────┘
                                          │
          ┌───────────────────────────────┼───────────────────────────────┐
          │                               │                               │
          ▼                               ▼                               ▼
    ACADEMIC PATH                   GENERAL PATH                    ROUTER AGENT
 (Nougat → DeepSeek)             (DeepSeek → Nougat)            (picks available engine)
          │                               │                               │
          └───────────────────────────────┴───────────────────────────────┘
                                          │
                          ┌───────────────┴───────────────┐
                          │   STAGE 1: PRIMARY OCR        │
                          │                               │
                          │  ┌─────────────────────────┐  │
                          │  │ Local Engines (FREE):   │  │
                          │  │                         │  │
                          │  │  • Nougat               │  │
                          │  │    nougat_ocr lib       │  │
                          │  │    model: 0.1.0-small   │  │
                          │  │                         │  │
                          │  │  • DeepSeek             │  │
                          │  │    Ollama               │  │
                          │  │    deepseek-ocr:latest  │  │
                          │  └─────────────────────────┘  │
                          │             OR                │
                          │  ┌─────────────────────────┐  │
                          │  │ Cloud Engines (PAID):   │  │
                          │  │                         │  │
                          │  │  • Gemini               │  │
                          │  │    Google genai SDK     │  │
                          │  │    gemini-3-flash       │  │
                          │  │    ~$0.0002/page        │  │
                          │  │                         │  │
                          │  │  • Mistral              │  │
                          │  │    Mistral SDK          │  │
                          │  │    pixtral-large        │  │
                          │  │    ~$0.001/page         │  │
                          │  └─────────────────────────┘  │
                          └───────────────┬───────────────┘
                                          │
                                   [Page Results]
                                          │
                                          ▼
                          ┌───────────────────────────────┐
                          │   STAGE 2: QUALITY AUDIT      │
                          │                               │
                          │  ┌─────────────────────────┐  │
                          │  │ Heuristics Checker      │  │
                          │  │ (rule-based, instant):  │  │
                          │  │  • Word count ≥ 50      │  │
                          │  │  • Garbage ratio < 15%  │  │
                          │  │  • Avg word length ok   │  │
                          │  │  • No unicode issues    │  │
                          │  │  • No repeated patterns │  │
                          │  └──────────┬──────────────┘  │
                          │             │                 │
                          │     [pages flagged?]          │
                          │             │                 │
                          │             ▼                 │
                          │  ┌─────────────────────────┐  │
                          │  │ Cross-Check (optional): │  │
                          │  │  Try 2nd local engine   │  │
                          │  └──────────┬──────────────┘  │
                          │             │                 │
                          │     [still flagged?]          │
                          │             │                 │
                          │             ▼                 │
                          │  ┌─────────────────────────┐  │
                          │  │ LLM Auditor (FREE):     │  │
                          │  │  Ollama                 │  │
                          │  │  deepseek-r1:32b        │  │
                          │  │  (reasoning model)      │  │
                          │  │  • Can override heur.   │  │
                          │  │  • verdict: acceptable, │  │
                          │  │    needs_review, poor   │  │
                          │  └──────────┬──────────────┘  │
                          └─────────────┼─────────────────┘
                                        │
                         ┌──────────────┴──────────────┐
                         │                             │
                    [all pass]                   [some flagged]
                         │                             │
                         │                             ▼
                         │             ┌───────────────────────────────┐
                         │             │   STAGE 3: FALLBACK OCR       │
                         │             │                               │
                         │             │  Router selects different     │
                         │             │  engine (not primary):        │
                         │             │   Gemini → Mistral →          │
                         │             │   DeepSeek → Nougat           │
                         │             │                               │
                         │             │  Reprocess flagged pages      │
                         │             └───────────────┬───────────────┘
                         │                             │
                         └─────────────┬───────────────┘
                                       │
                                  [All Pages]
                                       │
                                       ▼
                          ┌────────────────────────────┐
                          │  STAGE 4: FIGURE AGENT     │
                          │  (if enabled)              │
                          │                            │
                          │  ┌──────────────────────┐  │
                          │  │ PyMuPDF Extractor:   │  │
                          │  │  • Vector figures    │  │
                          │  │    (charts, plots)   │  │
                          │  │  • IMAGE blocks      │  │
                          │  │  • Embedded images   │  │
                          │  │  • Filter by size    │  │
                          │  │  • Render at 150 DPI │  │
                          │  └──────────┬───────────┘  │
                          │             │              │
                          │             ▼              │
                          │  ┌──────────────────────┐  │
                          │  │ Vision Engine:       │  │
                          │  │  Gemini 3 Flash      │  │
                          │  │  DeepSeek-OCR        │  │
                          │  │  Pixtral-Large       │  │
                          │  └──────────┬───────────┘  │
                          │             │              │
                          │      [Figure Results]      │
                          └─────────────┬──────────────┘
                                        │
                                        ▼
                          ┌─────────────────────────────┐
                          │      OUTPUT ASSEMBLY        │
                          │                             │
                          │  Format: markdown/json/txt  │
                          │                             │
                          │  output/<doc_stem>/         │
                          │    ├─ document.md           │
                          │    ├─ metadata.json         │
                          │    └─ figures/              │
                          │         └─ figure_N.png     │
                          └─────────────────────────────┘
```

**Cost optimization:**
- Local engines first (Nougat, DeepSeek via Ollama) - FREE
- Quality audit with local Ollama (llama3.2/qwen2.5) - FREE
- Cloud fallback only for failed pages (Gemini ~$0.0002/page)
- Typical 22-page paper: $0.0002 total (only 1 page needed cloud)

## Requirements

**Base:**
- Python 3.10+
- [Ollama](https://ollama.ai) with `deepseek-r1:32b` (for quality audit) or `llama3.2`/`qwen2.5` as lighter alternatives

**OCR Engines** (install individually based on needs):

| Engine | Installation | Type | Cost |
|--------|-------------|------|------|
| **DeepSeek** | `pip install deepseek-ocr-cli` | Local (Ollama) | Free |
| **Nougat** | `pip install nougat-ocr-cli` | Local (Python) | Free |
| **Gemini** | `pip install gemini-ocr-cli` | Cloud API | ~$0.0002/page |
| **Mistral** | `pip install mistral-ocr-cli` | Cloud API | ~$0.001/page |

**Installation examples:**

```bash
# Install with local engines only (recommended to start)
pip install docr[local]

# Install with all engines
pip install docr[all]

# Install specific engines
pip install docr[deepseek,gemini]
```

**Setup:**

```bash
# 1. Install Ollama and pull audit model (reasoning model recommended)
ollama pull deepseek-r1:32b  # best quality, or llama3.2 for speed

# 2. If using DeepSeek OCR, pull its model
ollama pull deepseek-ocr:latest

# 3. If using cloud engines, set API keys
export GEMINI_API_KEY="your-key"
export MISTRAL_API_KEY="your-key"

# 4. Check engine status
docr engines
```

## OCR Engine CLIs

Each OCR backend is a standalone CLI tool that can be used independently:

- **[deepseek-ocr-cli](https://github.com/r-uben/deepseek-ocr-cli)** - Local OCR via Ollama/DeepSeek
- **[gemini-ocr-cli](https://github.com/r-uben/gemini-ocr-cli)** - Cloud OCR via Google Gemini
- **[mistral-ocr-cli](https://github.com/r-uben/mistral-ocr-cli)** - Cloud OCR via Mistral AI
- **[nougat-ocr](https://github.com/facebookresearch/nougat)** - Academic papers (used as Python library)

## CLI Commands

```bash
# Simple usage
docr paper.pdf                    # Process a PDF
docr paper.pdf --save-figures     # Save figure images

# Full options
docr process paper.pdf [OPTIONS]
  -o, --output PATH      Output file path
  -f, --format           markdown|json|txt
  --primary ENGINE       Force primary engine
  --fallback ENGINE      Force fallback engine
  --no-audit             Skip quality audit
  --no-figures           Skip figure processing
  --save-figures         Save figure images to disk

# Batch process directory
docr batch ~/Papers/ [OPTIONS]
  --limit N              Process first N files
  --save-figures         Save all figure images

# Check engines
docr engines

# Check audit system
docr audit-status
```

## Configuration

Create `docr.yaml` in your project or home directory:

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
src/docr/
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
