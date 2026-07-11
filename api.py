import os
import secrets
from contextlib import asynccontextmanager
from typing import Optional

from fastapi import Depends, FastAPI, File, Header, HTTPException, Query, UploadFile

from database import init_db, list_invoices, save_invoice
from risk_rules import score_invoice_risk
from utils import (
    MAX_DOCUMENT_BYTES,
    content_hash,
    detect_invoice_type,
    extract_invoice_with_fallback,
    model_dump,
    process_document_bytes,
)


@asynccontextmanager
async def lifespan(_: FastAPI):
    init_db()
    yield


app = FastAPI(title="SmartInvoiceAI API", version="1.0", lifespan=lifespan)


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


def require_api_key(x_api_key: Optional[str] = Header(default=None)) -> None:
    configured_key = os.getenv("SMARTINVOICEAI_API_KEY", "").strip()
    if not configured_key:
        raise HTTPException(status_code=503, detail="API authentication is not configured")
    if not secrets.compare_digest(x_api_key or "", configured_key):
        raise HTTPException(status_code=401, detail="Invalid API key")


@app.get("/invoices")
def invoices(
    status: Optional[str] = None,
    limit: int = Query(default=50, ge=1, le=200),
    offset: int = Query(default=0, ge=0),
    include_text: bool = False,
    _: None = Depends(require_api_key),
) -> list:
    return list_invoices(status=status, limit=limit, offset=offset, include_text=include_text)


@app.post("/ingest")
async def ingest_invoice(
    file: UploadFile = File(...),
    language: str = "English",
    _: None = Depends(require_api_key),
) -> dict:
    data = await file.read(MAX_DOCUMENT_BYTES + 1)

    try:
        document = process_document_bytes(file.filename or "invoice", data, file.content_type)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    invoice, confidence, model_used, warnings = extract_invoice_with_fallback(
        document=document,
        language=language,
        groq_api_key=os.getenv("GROQ_API_KEY"),
        local_model=os.getenv("OLLAMA_MODEL", "llama3.2:3b"),
        local_base_url=os.getenv("OLLAMA_BASE_URL", "http://localhost:11434"),
    )
    invoice_data = model_dump(invoice)
    existing = list_invoices()
    duplicate_count = sum(
        1
        for row in existing
        if row.get("file_hash") == content_hash(data)
        or (
            invoice.invoice_number
            and (row.get("data") or {}).get("invoice_number") == invoice.invoice_number
            and (not invoice.vendor_name or (row.get("data") or {}).get("vendor_name") == invoice.vendor_name)
        )
    )
    invoice_id = save_invoice(
        filename=document.filename,
        file_hash=document.file_hash,
        page_count=document.page_count,
        extracted_text=document.text,
        data=invoice_data,
        confidence_scores=confidence,
        category=detect_invoice_type(invoice_data),
        model_used=model_used,
        fraud_flags=score_invoice_risk(invoice_data, duplicate_count=duplicate_count, existing_rows=existing),
        warnings=warnings,
    )
    return {"invoice_id": invoice_id, "model_used": model_used, "warnings": warnings}
