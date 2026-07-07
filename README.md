# SmartInvoiceAI

SmartInvoiceAI is a database-backed invoice intelligence app that uses OCR plus a local LLM to extract structured fields from invoice PDFs and images, reviews uncertain results, applies explainable risk rules, tracks vendors, visualizes spend, and exports reports.

The app opens with a seeded synthetic BremenTech GmbH invoice dataset so the dashboard is useful immediately.

## Features

- Persistent database layer with SQLAlchemy.
  - Local default: SQLite at `data/smart_invoice_ai.sqlite3`.
  - Production-ready options: MySQL or PostgreSQL through `DATABASE_URL`.
- Multi-page PDF support with native text extraction first.
- OCR/native PDF text extraction for scanned PDFs and image invoices.
- Local Ollama model support for invoice-to-JSON extraction and invoice assistant Q&A.
- Optional Groq fallback order for cloud model resilience.
- Few-shot extraction prompt plus deterministic regex fallback when AI extraction is unavailable.
- Document queue with processing job status.
- Human review workflow: Submitted, Reviewed, Approved, Rejected, Paid, Flagged.
- Bulk status updates and audit trail.
- Vendor management summary with history and max risk score.
- Dashboard KPIs, monthly spend, top vendors, anomaly detection.
- CSV, formatted Excel, and PDF report exports.
- FastAPI endpoint for external ingestion and invoice reads.
- Dockerfile and Docker Compose with MySQL.
- Automatic synthetic BremenTech GmbH demo data seeding when the database is empty.

## Screenshots

### Overview dashboard

![Overview dashboard](docs/screenshots/overview-dashboard.png)

### Add invoices and document queue

![Add invoices and document queue](docs/screenshots/add-invoices-queue.png)

### Invoice detail with risk breakdown

![Invoice detail with risk breakdown](docs/screenshots/invoice-detail-risk.png)

### Risk rules

![Risk rules](docs/screenshots/risk-rules.png)

## Run Locally

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
copy .env.example .env
streamlit run app.py
```

For local AI extraction, install Ollama, pull a model, and keep Ollama running:

```bash
ollama pull llama3.2:3b
ollama serve
```

Optional local model settings:

```env
OLLAMA_MODEL=llama3.2:3b
OLLAMA_BASE_URL=http://localhost:11434
```

Set `GROQ_API_KEY` only if you also want cloud fallback in `.env`, as an environment variable, or in `.streamlit/secrets.toml`.

If Ollama is not running and no Groq key is configured, the app still stores uploads and uses the deterministic parser for text-based PDFs.

## Database

SQLite is the local default:

```env
DATABASE_URL=sqlite:///data/smart_invoice_ai.sqlite3
```

MySQL example:

```env
DATABASE_URL=mysql+pymysql://smartinvoice:smartinvoice@localhost:3307/smartinvoice
```

PostgreSQL example:

```env
DATABASE_URL=postgresql+psycopg://smartinvoice:smartinvoice@localhost:5432/smartinvoice
```

## Docker

```bash
docker compose up --build
```

Services:

- Streamlit app: `http://localhost:8501`
- API: `http://localhost:8000`
- MySQL: `localhost:3307`

## API

Run the API without Docker:

```bash
uvicorn api:app --host 0.0.0.0 --port 8000
```

Endpoints:

- `GET /health`
- `GET /invoices`
- `POST /ingest` with multipart file upload

## Notes

- OCR for scanned PDFs needs Poppler and Tesseract installed locally. The Docker image includes both.
- Local LLM extraction expects an Ollama-compatible API at `OLLAMA_BASE_URL`.
- Disable automatic demo seeding with `SMARTINVOICEAI_DEMO_DATA=0`.
- Generated databases, exports, caches, virtual environments, and secrets are ignored by git.
