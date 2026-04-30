# Workplace email utilities

**Open → ingest → process:** reusable Python helpers for mail archives (Maildir, CSV, paths on disk), normalized tables, time/thread/graph steps, batch/stream scaffolding, and export-oriented reporting. Outputs are meant for **later** pipelines—**workplace-text** (language semantics), NetSmith, notebooks, Parquet in a lake, or your own jobs—not a finished product UI.

> **Scope**  
> Non-NLP companion to **workplace-text**. No HTTP server. Sentiment and heavy text modeling stay in workplace-text; see [`spec.md`](spec.md) for the split.

## What lives here

| Layer | Role |
|--------|------|
| **Open / ingest** | Read Maildir trees and CSV dumps into structured rows (`workplace_email_utils.ingest`). |
| **Process** | Temporal columns, threading, graph construction, anomalies, predictive heuristics, optional fusion / FAISS helpers—utilities you can call from batch code or compose in `pipeline.py`. |
| **Emit** | Reports and dashboard-shaped JSON (`reporting`, `visualization`), file exports—not a hosted app. |

Install the library, import the pieces you need, and wire them into whatever runs *next*.

## Installation

- Python 3.10+
- Dependencies: `requirements.txt`

```bash
bash scripts/setup_venv.sh   # venv + pip install -r requirements.txt + pip install -e .
source venv/bin/activate
```

Apple Silicon (FAISS): `pip install faiss-cpu --no-build-isolation` if the default install fails.

## Quick start

### Ingest only

```python
from workplace_email_utils.ingest.email_parser import load_emails

df = load_emails("path/to/maildir", data_format="maildir", sample_size=5000)
# df → downstream enrichment (e.g. workplace-text), Parquet, etc.
```

### Full compose (optional)

`workplace_email_utils.pipeline.build_knowledge_model` still orchestrates multiple modules for experimentation; trim or replace stages as your production flow solidifies.

```python
from workplace_email_utils.pipeline import build_knowledge_model

model = build_knowledge_model(
    data_path="maildir",
    data_format="maildir",
    sample_size=10000,
    enable_threading=True,
    enable_classification=True,
)
```

### Streaming (batch iterator style)

```python
from workplace_email_utils.streaming.ingestion import stream_emails, EmailStream
from workplace_email_utils.streaming.processing import process_email_stream, RealTimeProcessor

stream = EmailStream(source="directory", source_path="maildir", batch_size=10)
processor = RealTimeProcessor(classify=True, extract_features=True)

for batch in stream_emails(stream, max_emails=100):
    processed = process_email_stream(batch, processor)
```

### Reports / exports

```python
from workplace_email_utils.reporting.report_generator import generate_report, ReportConfig
from workplace_email_utils.reporting.exports import export_report

config = ReportConfig(
    report_type="summary",
    sections=["overview", "analytics", "anomalies", "metrics"],
)
report = generate_report(df, config)
export_report(report, "reports/email_report.json", format="json")
```

## Module map (high level)

- **`ingest`** — open paths, parse messages → DataFrame-oriented rows  
- **`temporal_features`**, **`threading`** — time and conversation utilities  
- **`graph_features`**, **`network_analysis`** — graph build + metrics (see spec for overlap with NetSmith)  
- **`anomaly_detection`**, **`predictive`** — behavioral / ops-style helpers  
- **`classification`**, **`entity_extraction`**, **`content_features`** — mostly **non–workplace-text** paths today; prefer workplace-text for NLP and keep these for glue or legacy flows  
- **`streaming`**, **`distributed`** — throughput-oriented helpers  
- **`reporting`**, **`visualization`** — artifacts and structured payloads for downstream tools  
- **`fusion`**, **`vector_index`**, **`clustering`** — combine or index matrices you already computed elsewhere if useful  

Examples: `examples/`. API details: module docstrings.

## Project structure

```
.
├── pyproject.toml
├── requirements.txt
├── spec.md
├── src/workplace_email_utils/   # installable package (all modules + pipeline.py)
├── examples/
└── scripts/setup_venv.sh
```

Imports look like: `from workplace_email_utils.ingest.email_parser import load_emails`.

## License

Prototype / internal use unless you attach your own license.
