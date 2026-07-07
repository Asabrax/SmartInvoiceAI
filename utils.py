import base64
import hashlib
import io
import json
import os
import re
import tempfile
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import pandas as pd
import requests
from groq import Groq
from PIL import Image, ImageEnhance
from pydantic import BaseModel, Field
from pypdf import PdfReader
from risk_rules import normalize_currency, score_invoice_risk


class LineItem(BaseModel):
    description: Optional[str] = Field(None, description="Product or service description.")
    quantity: Optional[float] = Field(None, description="Number of units.")
    unit_price: Optional[float] = Field(None, description="Price per unit.")
    total_price: Optional[float] = Field(None, description="Line item total.")


class InvoiceData(BaseModel):
    invoice_number: Optional[str] = Field(None, description="Invoice reference number.")
    invoice_date: Optional[str] = Field(None, description="Issue date.")
    due_date: Optional[str] = Field(None, description="Payment due date.")
    billing_address: Optional[str] = Field(None, description="Bill-to address.")
    shipping_address: Optional[str] = Field(None, description="Ship-to address.")
    vendor_name: Optional[str] = Field(None, description="Invoice issuer.")
    vendor_vat_id: Optional[str] = Field(None, description="Vendor VAT, tax, or business registration ID.")
    customer_name: Optional[str] = Field(None, description="Bill-to customer.")
    line_items: Optional[List[LineItem]] = Field(None, description="Invoice line items.")
    subtotal: Optional[float] = Field(None, description="Subtotal before tax.")
    tax: Optional[float] = Field(None, description="Tax amount.")
    total_amount: Optional[float] = Field(None, description="Final amount due.")
    currency: Optional[str] = Field(None, description="ISO currency code such as EUR, USD, GBP, or INR.")
    payment_terms: Optional[str] = Field(None, description="Payment terms such as Net 30, due on receipt, or bank transfer terms.")


@dataclass
class ProcessedDocument:
    filename: str
    content_type: str
    file_hash: str
    page_count: int
    text: str
    image_payloads: List[Dict[str, Any]]
    warnings: List[str]


def model_dump(model: BaseModel) -> Dict[str, Any]:
    if hasattr(model, "model_dump"):
        return model.model_dump()
    return model.dict()


def model_schema(model: type[BaseModel]) -> Dict[str, Any]:
    if hasattr(model, "model_json_schema"):
        return model.model_json_schema()
    return model.schema()


def process_image_upload(uploaded_file):
    if not uploaded_file:
        return None, None
    image_bytes = uploaded_file.read()
    suffix = uploaded_file.name.split(".")[-1].lower()
    mime_type = "image/jpeg" if suffix in ("jpg", "jpeg") else "image/png"
    return image_bytes, mime_type


def process_image_url(image_url):
    if not image_url:
        return None
    try:
        response = requests.get(image_url, timeout=20)
        response.raise_for_status()
        return response.content
    except Exception as exc:
        raise ValueError(f"Error loading image from URL: {exc}") from exc


def preprocess_image(image_bytes: bytes) -> bytes:
    try:
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        image = ImageEnhance.Contrast(image).enhance(1.7)
        image.thumbnail((1400, 1400), Image.Resampling.LANCZOS)
        output = io.BytesIO()
        image.save(output, format="JPEG", quality=92)
        return output.getvalue()
    except Exception as exc:
        raise ValueError(f"Image preprocessing error: {exc}") from exc


def display_image_preview(image_bytes):
    import streamlit as st

    try:
        st.image(Image.open(io.BytesIO(image_bytes)))
    except Exception as exc:
        st.error(f"Error displaying image: {exc}")


def setup_page():
    import streamlit as st

    st.set_page_config(page_title="SmartInvoiceAI", layout="wide")
    st.title("SmartInvoiceAI")


def select_input_method():
    import streamlit as st

    return st.radio("Input method", ["Upload documents", "Image URL"], horizontal=True)


def show_extraction_button():
    import streamlit as st

    return st.button("Extract invoice data", type="primary")


def display_results(invoice_data):
    import streamlit as st

    st.success("Data extracted successfully.")
    st.json(model_dump(invoice_data))


def display_error(message):
    import streamlit as st

    st.error(message)


def _normalize_amount(value: str) -> Optional[float]:
    if not value:
        return None
    cleaned = re.sub(r"[^0-9,.\-]", "", value).strip()
    if not cleaned:
        return None
    if cleaned.count(",") == 1 and cleaned.count(".") == 0:
        cleaned = cleaned.replace(",", ".")
    else:
        cleaned = cleaned.replace(",", "")
    try:
        return float(cleaned)
    except ValueError:
        return None


def _first_match(patterns: Iterable[str], text: str, flags: int = re.IGNORECASE) -> Optional[str]:
    for pattern in patterns:
        match = re.search(pattern, text, flags)
        if match:
            return match.group(1).strip(" :#-\n\t")
    return None


def _extract_amount(label_patterns: Iterable[str], text: str) -> Optional[float]:
    for label in label_patterns:
        pattern = rf"(?<![A-Za-z]){label}\b\s*[:\-]?\s*([$A-Z ]*)?([0-9][0-9,]*(?:\.[0-9]{{2}})?)"
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            return _normalize_amount(match.group(2))
    return None


def parse_invoice_text(text: str) -> Tuple[InvoiceData, Dict[str, float]]:
    compact = re.sub(r"\s+", " ", text or " ").strip()
    lines = [line.strip() for line in (text or "").splitlines() if line.strip()]
    invoice_number = _first_match(
        [
            r"(?:invoice|inv)\s*(?:number|no|#)\s*[:#-]?\s*([A-Z0-9][A-Z0-9\-\/]+)",
            r"\b(?:bill|receipt)\s*(?:number|no|#)\s*[:#-]?\s*([A-Z0-9][A-Z0-9\-\/]+)",
        ],
        compact,
    )
    invoice_date = _first_match(
        [
            r"(?:invoice\s*)?date\s*[:\-]?\s*([0-9]{1,4}[\/.\-][0-9]{1,2}[\/.\-][0-9]{2,4})",
            r"(?:issued|issue date)\s*[:\-]?\s*([0-9]{1,4}[\/.\-][0-9]{1,2}[\/.\-][0-9]{2,4})",
        ],
        compact,
    )
    due_date = _first_match(
        [r"(?:due date|payment due|pay by)\s*[:\-]?\s*([0-9]{1,4}[\/.\-][0-9]{1,2}[\/.\-][0-9]{2,4})"],
        compact,
    )
    currency = _first_match([r"\b(USD|EUR|GBP|CAD|AUD|INR)\b"], compact)
    if not currency:
        symbol = _first_match([r"([$€£₹])\s*[0-9]"], compact)
        currency = normalize_currency(symbol)
    else:
        currency = normalize_currency(currency)

    vendor_name = _first_match(
        [r"(?:vendor|from|seller|supplier)\s*[:\-]\s*([A-Za-z0-9 &.,'\-]{2,80})"],
        text,
    )
    if not vendor_name and lines:
        vendor_name = lines[0][:80]
    vendor_vat_id = _first_match(
        [
            r"(?:vendor\s*)?(?:vat|tax)\s*(?:id|number|no\.?)\s*[:#-]?\s*([A-Z]{2}[A-Z0-9 .\-]{6,18}|[A-Z0-9][A-Z0-9 .\-]{5,20})",
            r"(?:ust[\s.-]?idnr|ust[\s.-]?id|steuer[\s-]?nr)\s*[:#-]?\s*([A-Z0-9 .\-]{6,20})",
            r"(?:business|company|commercial)\s*registration\s*(?:id|number|no\.?)\s*[:#-]?\s*([A-Z0-9 .\-]{6,20})",
        ],
        compact,
    )
    if vendor_vat_id:
        vendor_vat_id = re.split(
            r"\s+(?:payment|terms|subtotal|vat|tax|total|invoice|due|bill)\b",
            vendor_vat_id,
            maxsplit=1,
            flags=re.IGNORECASE,
        )[0].strip(" .-")

    customer_name = _first_match(
        [r"(?:bill to|customer|client)\s*[:\-]\s*([A-Za-z0-9 &.,'\-]{2,80})"],
        text,
    )
    payment_terms = _first_match(
        [
            r"(?:payment terms|terms)\s*[:\-]\s*([A-Za-z0-9 ,.\-]{2,80})",
            r"\b(net\s*[0-9]{1,3})\b",
            r"\b(due on receipt)\b",
        ],
        compact,
    )
    if payment_terms:
        payment_terms = re.split(
            r"\s+(?:subtotal|vat|tax|total|invoice|due date|bill to|vendor)\b",
            payment_terms,
            maxsplit=1,
            flags=re.IGNORECASE,
        )[0].strip(" .-")
    subtotal = _extract_amount([r"subtotal", r"net amount"], compact)
    tax = _extract_amount([r"tax", r"vat", r"gst"], compact)
    total_amount = _extract_amount([r"grand total", r"amount due", r"balance due", r"total"], compact)

    data = InvoiceData(
        invoice_number=invoice_number,
        invoice_date=invoice_date,
        due_date=due_date,
        vendor_name=vendor_name,
        vendor_vat_id=vendor_vat_id,
        customer_name=customer_name,
        subtotal=subtotal,
        tax=tax,
        total_amount=total_amount,
        currency=currency,
        payment_terms=payment_terms,
        line_items=[],
    )
    confidence = {
        "invoice_number": 0.72 if invoice_number else 0.0,
        "invoice_date": 0.68 if invoice_date else 0.0,
        "due_date": 0.62 if due_date else 0.0,
        "vendor_name": 0.55 if vendor_name else 0.0,
        "vendor_vat_id": 0.66 if vendor_vat_id else 0.0,
        "customer_name": 0.5 if customer_name else 0.0,
        "subtotal": 0.64 if subtotal is not None else 0.0,
        "tax": 0.62 if tax is not None else 0.0,
        "total_amount": 0.7 if total_amount is not None else 0.0,
        "currency": 0.6 if currency else 0.0,
        "payment_terms": 0.55 if payment_terms else 0.0,
    }
    return data, confidence


def content_hash(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def image_payload_from_bytes(image_bytes: bytes, mime_type: str = "image/jpeg") -> Dict[str, Any]:
    encoded = base64.b64encode(image_bytes).decode("utf-8")
    return {"type": "image_url", "image_url": {"url": f"data:{mime_type};base64,{encoded}"}}


def _ocr_image_bytes(image_bytes: bytes) -> Tuple[str, Optional[str]]:
    try:
        import pytesseract
    except Exception:
        return "", "pytesseract is not installed; OCR fallback was skipped."
    try:
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        return pytesseract.image_to_string(image), None
    except Exception as exc:
        return "", f"OCR failed: {exc}"


def _pdf_page_images(pdf_bytes: bytes, max_pages: int = 3) -> Tuple[List[Dict[str, Any]], List[str]]:
    warnings: List[str] = []
    try:
        from pdf2image import convert_from_bytes
    except Exception:
        return [], ["pdf2image is not installed; PDF page image fallback was skipped."]
    try:
        images = convert_from_bytes(pdf_bytes, dpi=180, first_page=1, last_page=max_pages)
    except Exception as exc:
        return [], [f"PDF page rendering skipped: {exc}"]

    payloads = []
    for image in images:
        output = io.BytesIO()
        image.convert("RGB").save(output, format="JPEG", quality=90)
        payloads.append(image_payload_from_bytes(output.getvalue()))
    return payloads, warnings


def process_document_bytes(filename: str, data: bytes, content_type: Optional[str] = None) -> ProcessedDocument:
    suffix = Path(filename).suffix.lower()
    warnings: List[str] = []
    image_payloads: List[Dict[str, Any]] = []
    text = ""
    page_count = 1
    content_type = content_type or "application/octet-stream"

    if suffix == ".pdf" or content_type == "application/pdf":
        try:
            reader = PdfReader(io.BytesIO(data))
            page_count = len(reader.pages)
            text = "\n\n".join(page.extract_text() or "" for page in reader.pages).strip()
        except Exception as exc:
            warnings.append(f"Native PDF text extraction failed: {exc}")
        if len(text) < 40:
            rendered_payloads, render_warnings = _pdf_page_images(data)
            image_payloads.extend(rendered_payloads)
            warnings.extend(render_warnings)
            if image_payloads:
                warnings.append("PDF appears scanned; first pages were rendered for vision extraction.")
    else:
        content_type = "image/png" if suffix == ".png" else "image/jpeg"
        try:
            normalized = preprocess_image(data)
            image_payloads.append(image_payload_from_bytes(normalized, content_type))
            ocr_text, ocr_warning = _ocr_image_bytes(normalized)
            text = ocr_text.strip()
            if ocr_warning:
                warnings.append(ocr_warning)
        except Exception as exc:
            warnings.append(str(exc))

    return ProcessedDocument(
        filename=filename,
        content_type=content_type,
        file_hash=content_hash(data),
        page_count=page_count,
        text=text,
        image_payloads=image_payloads,
        warnings=warnings,
    )


def build_extraction_prompt(language: str, document_text: str = "") -> str:
    few_shot = {
        "data": {
            "invoice_number": "INV-1042",
            "invoice_date": "2026-01-15",
            "due_date": "2026-02-14",
            "vendor_name": "Northwind Services",
            "vendor_vat_id": "DE123456789",
            "customer_name": "Contoso GmbH",
            "subtotal": 1200.0,
            "tax": 228.0,
            "total_amount": 1428.0,
            "currency": "EUR",
            "payment_terms": "Net 30",
            "line_items": [{"description": "Consulting", "quantity": 8, "unit_price": 150, "total_price": 1200}],
        },
        "confidence_scores": {"invoice_number": 0.95, "vendor_vat_id": 0.91, "total_amount": 0.94},
    }
    text_block = f"\nNative/OCR text:\n{document_text[:12000]}" if document_text else ""
    return f"""
You extract invoice data in {language}. Return strict JSON with "data" and "confidence_scores".
Use this schema: {json.dumps(model_schema(InvoiceData), indent=2)}.
Prefer exact values from the invoice. Use null for missing fields.
Flag uncertainty by lowering the confidence score from 0.0 to 1.0.
Few-shot example: {json.dumps(few_shot)}
{text_block}
"""


class GroqClient:
    def __init__(self, api_key: str, models: Optional[List[str]] = None):
        self.client = Groq(api_key=api_key)
        configured = os.getenv("GROQ_MODELS")
        if models:
            self.models = models
        elif configured:
            self.models = [m.strip() for m in configured.split(",") if m.strip()]
        else:
            self.models = [
                "meta-llama/llama-4-scout-17b-16e-instruct",
                "llama-3.3-70b-versatile",
                "llama-3.1-8b-instant",
            ]

    def extract_invoice_data(
        self,
        prompt: str,
        image_content: Optional[Dict[str, Any]] = None,
        model: Optional[str] = None,
    ) -> Dict[str, Any]:
        errors = []
        models = [model] if model else self.models
        for candidate in models:
            try:
                content: Any = prompt
                if image_content:
                    content = [{"type": "text", "text": prompt}, image_content]
                response = self.client.chat.completions.create(
                    model=candidate,
                    messages=[{"role": "user", "content": content}],
                    temperature=0.2,
                    max_completion_tokens=1600,
                    response_format={"type": "json_object"},
                )
                payload = json.loads(response.choices[0].message.content)
                payload["_model_used"] = candidate
                return payload
            except Exception as exc:
                errors.append(f"{candidate}: {exc}")
        raise RuntimeError("All Groq models failed. " + " | ".join(errors))

    def run_chatbot_query(self, prompt: str, model: Optional[str] = None) -> str:
        errors = []
        models = [model] if model else self.models
        for candidate in models:
            try:
                response = self.client.chat.completions.create(
                    model=candidate,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.4,
                    max_tokens=700,
                    stream=False,
                )
                return response.choices[0].message.content
            except Exception as exc:
                errors.append(f"{candidate}: {exc}")
        raise RuntimeError("All Groq models failed. " + " | ".join(errors))


class OllamaClient:
    def __init__(self, model: str, base_url: str = "http://localhost:11434"):
        self.model = model.strip()
        self.base_url = base_url.rstrip("/")

    def _post_chat(self, prompt: str, *, json_response: bool) -> str:
        payload: Dict[str, Any] = {
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
            "stream": False,
            "options": {"temperature": 0.1 if json_response else 0.3},
        }
        if json_response:
            payload["format"] = "json"
        response = requests.post(f"{self.base_url}/api/chat", json=payload, timeout=90)
        response.raise_for_status()
        body = response.json()
        return str((body.get("message") or {}).get("content") or "")

    def extract_invoice_data(self, prompt: str) -> Dict[str, Any]:
        content = self._post_chat(prompt, json_response=True)
        payload = _json_payload(content)
        payload["_model_used"] = f"ollama:{self.model}"
        return payload

    def run_chatbot_query(self, prompt: str) -> str:
        return self._post_chat(prompt, json_response=False)


def _json_payload(content: str) -> Dict[str, Any]:
    try:
        return json.loads(content)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", content, re.DOTALL)
        if not match:
            raise
        return json.loads(match.group(0))


def _normalize_invoice_payload(data: Dict[str, Any]) -> Dict[str, Any]:
    normalized = dict(data or {})
    for key in (
        "invoice_number",
        "invoice_date",
        "due_date",
        "billing_address",
        "shipping_address",
        "vendor_name",
        "vendor_vat_id",
        "customer_name",
        "currency",
        "payment_terms",
    ):
        value = normalized.get(key)
        if isinstance(value, (dict, list)):
            normalized[key] = json.dumps(value, ensure_ascii=False)
    normalized["currency"] = normalize_currency(normalized.get("currency"))
    line_items = normalized.get("line_items")
    if not isinstance(line_items, list):
        normalized["line_items"] = []
    return normalized


def extract_invoice_with_fallback(
    document: ProcessedDocument,
    language: str,
    groq_api_key: Optional[str],
    preferred_models: Optional[List[str]] = None,
    local_model: Optional[str] = None,
    local_base_url: str = "http://localhost:11434",
    prefer_local: bool = True,
) -> Tuple[InvoiceData, Dict[str, float], str, List[str]]:
    warnings = list(document.warnings)
    prompt = build_extraction_prompt(language, document.text)
    model_used = "regex-fallback"

    if prefer_local and local_model:
        try:
            extracted = OllamaClient(local_model, local_base_url).extract_invoice_data(prompt)
            invoice = InvoiceData(**_normalize_invoice_payload(extracted.get("data", {})))
            confidence = extracted.get("confidence_scores", {}) or {}
            model_used = extracted.get("_model_used", f"ollama:{local_model}")
            if not any(value is not None and value != [] for value in model_dump(invoice).values()):
                warnings.append("Local model returned no usable fields; deterministic parser was used.")
            else:
                return invoice, confidence, model_used, warnings
        except Exception as exc:
            warnings.append(f"Local Ollama extraction failed; fallback was used. {exc}")

    if groq_api_key:
        try:
            client = GroqClient(api_key=groq_api_key, models=preferred_models)
            image_content = document.image_payloads[0] if document.image_payloads else None
            extracted = client.extract_invoice_data(prompt, image_content=image_content)
            invoice = InvoiceData(**_normalize_invoice_payload(extracted.get("data", {})))
            confidence = extracted.get("confidence_scores", {}) or {}
            model_used = extracted.get("_model_used", "groq")
            if not any(value is not None and value != [] for value in model_dump(invoice).values()):
                warnings.append("Groq returned no usable fields; deterministic parser was used.")
                return (*parse_invoice_text(document.text), "regex-fallback", warnings)
            return invoice, confidence, model_used, warnings
        except Exception as exc:
            warnings.append(f"Groq extraction failed; deterministic parser was used. {exc}")

    invoice, confidence = parse_invoice_text(document.text)
    return invoice, confidence, model_used, warnings


def detect_invoice_type(invoice_data: Dict[str, Any]) -> str:
    keywords = {
        "software": ["subscription", "license", "saas", "cloud", "hosting", "data warehouse", "security license", "backup storage"],
        "retail": ["store", "shop", "mart", "sku", "product", "office supply", "paper", "labels", "chairs", "print", "toner"],
        "service": [
            "consulting",
            "service",
            "hours",
            "labor",
            "professional",
            "logistics",
            "support retainer",
            "maintenance",
            "implementation",
            "facilities",
            "security",
            "courier",
        ],
        "utility": ["electric", "electricity", "energy", "water", "gas", "bill", "utility", "meter", "grid fees"],
        "travel": ["travel", "hotel", "flight", "taxi", "rail", "airport", "rental", "lodging"],
    }
    def values_only(value: Any) -> List[str]:
        if isinstance(value, dict):
            chunks: List[str] = []
            for nested in value.values():
                chunks.extend(values_only(nested))
            return chunks
        if isinstance(value, list):
            chunks = []
            for nested in value:
                chunks.extend(values_only(nested))
            return chunks
        return [str(value)]

    blob = " ".join(values_only(invoice_data or {})).lower()
    for category, terms in keywords.items():
        if any(term in blob for term in terms):
            return category
    return "general"


def fraud_flags(invoice: InvoiceData, duplicate_count: int = 0) -> List[str]:
    return score_invoice_risk(model_dump(invoice), duplicate_count=duplicate_count)


def _risk_label_for_flag(flag: str) -> str:
    text = flag.lower()
    if "duplicate" in text:
        return "Duplicate"
    if "3x" in text or "high total" in text or "high amount" in text:
        return "Amount anomaly"
    if "vat" in text:
        return "Missing VAT ID"
    if "due date" in text:
        return "Date issue"
    if "new vendor" in text:
        return "New vendor"
    if "currency" in text:
        return "Currency mismatch"
    if "round amount" in text:
        return "Round amount"
    if "tax" in text:
        return "Tax check"
    return str(flag).split(":", 1)[-1].strip()[:28]


def _risk_labels(flags: List[str]) -> str:
    labels: List[str] = []
    for flag in flags:
        label = _risk_label_for_flag(str(flag))
        if label and label not in labels:
            labels.append(label)
    return ", ".join(labels) or "None"


def invoice_dataframe(rows: List[Dict[str, Any]]) -> pd.DataFrame:
    columns = [
        "id",
        "status",
        "vendor_name",
        "invoice_number",
        "invoice_date",
        "due_date",
        "category",
        "currency",
        "subtotal",
        "tax",
        "total_amount",
        "risk_score",
        "model_used",
        "source",
        "source_file",
        "ocr_confidence",
        "risk_model",
        "risk_reason",
        "risk_labels",
    ]
    records = []
    for row in rows:
        data = row.get("data") or {}
        records.append(
            {
                "id": row.get("id"),
                "status": row.get("status"),
                "vendor_name": data.get("vendor_name"),
                "invoice_number": data.get("invoice_number"),
                "invoice_date": data.get("invoice_date"),
                "due_date": data.get("due_date"),
                "category": row.get("category"),
                "currency": data.get("currency"),
                "subtotal": data.get("subtotal"),
                "tax": data.get("tax"),
                "total_amount": data.get("total_amount"),
                "risk_score": row.get("risk_score"),
                "model_used": row.get("model_used"),
                "source": data.get("source") or ("PDF upload" if row.get("filename", "").lower().endswith(".pdf") else "Upload"),
                "source_file": data.get("source_file") or row.get("filename"),
                "ocr_confidence": data.get("ocr_confidence"),
                "risk_model": data.get("risk_model") or "rules_v1",
                "risk_reason": "; ".join(str(flag).split(":", 1)[-1].strip() for flag in row.get("fraud_flags") or []) or "No active flags",
                "risk_labels": _risk_labels(row.get("fraud_flags") or []),
            }
        )
    return pd.DataFrame(records, columns=columns)


def export_to_csv(invoice_data_list: List[InvoiceData]) -> str:
    return pd.DataFrame([model_dump(invoice) for invoice in invoice_data_list]).to_csv(index=False)


def write_temp_upload(filename: str, data: bytes) -> str:
    suffix = Path(filename).suffix or ".bin"
    handle = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
    handle.write(data)
    handle.close()
    return handle.name


def parse_date(value: Optional[str]) -> Optional[datetime]:
    if not value:
        return None
    for fmt in ("%Y-%m-%d", "%d/%m/%Y", "%m/%d/%Y", "%d-%m-%Y", "%m-%d-%Y", "%Y/%m/%d"):
        try:
            return datetime.strptime(value.strip(), fmt)
        except ValueError:
            continue
    return None
