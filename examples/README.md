# Examples

Small runnable scripts for **ingest → process** flows. Use them as recipes; production jobs usually import `workplace_email_utils` from your own runner.

## Available Examples

### `basic_usage.py`
Basic example showing how to process emails and build an analytics model. Includes thread analysis and classification results.

### `network_analysis.py`
Demonstrates network analysis capabilities including influence metrics, community detection, and bridge node identification.

### `reporting.py`
Example of generating comprehensive reports and dashboards with multi-format export.

### `knowledge_base_example.py`
Demonstrates how to create and use knowledge bases for improved entity extraction with canonical names and aliases.

## Running examples

From the **repository root** (after `pip install -e .` from `scripts/setup_venv.sh` or the main README):

```bash
source venv/bin/activate   # Windows: venv\Scripts\activate

python examples/basic_usage.py
python examples/network_analysis.py
python examples/reporting.py
python examples/knowledge_base_example.py
```

Each script adds `src/` to `sys.path` if needed, so they also work before editable install when run from the repo root.

## Prerequisites

- Emails in `maildir/` directory or `emails.csv` file
- All dependencies installed (see main README.md)
- Virtual environment activated

