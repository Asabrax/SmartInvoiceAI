import hashlib
import json
from copy import deepcopy
from datetime import date, timedelta
from typing import Any, Dict, List

from sqlalchemy.orm.attributes import flag_modified

from database import InvoiceRecord, list_invoices, save_invoice, session_scope, update_invoice_status
from risk_rules import score_from_flags, score_invoice_risk
from utils import InvoiceData, ProcessedDocument, content_hash, detect_invoice_type, extract_invoice_with_fallback, model_dump


TARGET_DEMO_RECORDS = 500
DEMO_COMPANY = "BremenTech GmbH"
DEMO_START_DATE = date(2025, 1, 6)
DEMO_STATUSES = [
    "Paid",
    "Paid",
    "Paid",
    "Paid",
    "Approved",
    "Approved",
    "Reviewed",
    "Submitted",
    "Submitted",
    "Flagged",
    "Rejected",
]

VENDORS = [
    {"name": "NordCloud GmbH", "category": "software", "base": 1860.0, "vat_id": "DE284915730"},
    {"name": "Bremen OfficePoint GmbH", "category": "retail", "base": 520.0, "vat_id": "DE119604822"},
    {"name": "RheinLogistik GmbH", "category": "service", "base": 1480.0, "vat_id": "DE302871944"},
    {"name": "RepairPro Bremen GmbH", "category": "maintenance", "base": 920.0, "vat_id": "DE257481019"},
    {"name": "Alpine Beratung GmbH", "category": "consulting", "base": 2450.0, "vat_id": "DE198744620"},
    {"name": "Hanseatic Printworks GmbH", "category": "office supplies", "base": 390.0, "vat_id": "DE321784551"},
    {"name": "Weser Travel GmbH", "category": "travel", "base": 760.0, "vat_id": "DE218409637"},
    {"name": "HafenStrom Utilities GmbH", "category": "utilities", "base": 1280.0, "vat_id": "DE154207833"},
    {"name": "Cloudhost Bremen GmbH", "category": "cloud/software", "base": 1720.0, "vat_id": "DE290377146"},
    {"name": "SecureGate GmbH", "category": "cloud/software", "base": 2110.0, "vat_id": "DE183945702"},
    {"name": "OfficeLine Nord GmbH", "category": "office supplies", "base": 610.0, "vat_id": "DE276118504"},
    {"name": "Metro Facilities GmbH", "category": "maintenance", "base": 1050.0, "vat_id": "DE340916228"},
]

ITEMS = {
    "software": ["Annual SaaS licence", "Data platform subscription", "Support package"],
    "cloud/software": ["Cloud hosting", "Security monitoring", "Backup storage"],
    "retail": ["Workstation accessories", "Office furniture", "Packaging supplies"],
    "office supplies": ["Printer paper and labels", "Toner cartridges", "Meeting room supplies"],
    "service": ["Logistics coordination", "Warehouse handling", "Courier services"],
    "consulting": ["Implementation workshop", "Architecture review", "Process advisory"],
    "maintenance": ["Facility maintenance", "Equipment repair", "Preventive service visit"],
    "utilities": ["Electricity billing period", "Network grid fees", "Water service"],
    "travel": ["Rail and hotel booking", "Client visit expenses", "Airport transfer"],
}

DUPLICATE_LINKS = {96: 41, 261: 177, 418: 336}
HIGH_AMOUNT_INDEXES = {118, 274, 392}
MISSING_VAT_INDEXES = {73, 118, 226, 274, 392, 451}
DUE_DATE_ERROR_INDEXES = {118, 154, 366, 392}
CURRENCY_MISMATCH_INDEXES = {274, 309}
NEW_VENDOR_INDEXES = {444}
LOCAL_AI_DEMO_INDEXES = [18, 41, 73, 96, 118, 154, 226, 274, 309, 366, 392, 444]


def _date_for(index: int) -> date:
    end_date = date.today()
    span_days = max((end_date - DEMO_START_DATE).days, 1)
    progress = index / max(TARGET_DEMO_RECORDS - 1, 1)
    offset = round(progress * span_days)
    jitter = ((index * 17) % 9) - 4
    return min(max(DEMO_START_DATE + timedelta(days=offset + jitter), DEMO_START_DATE), end_date)


def _status_for(index: int) -> str:
    return DEMO_STATUSES[(index * 7) % len(DEMO_STATUSES)]


def _workflow_age(status: str, index: int) -> int:
    base = {
        "Submitted": 3,
        "Reviewed": 5,
        "Approved": 2,
        "Flagged": 9,
        "Rejected": 6,
        "Paid": 1,
    }.get(status, 3)
    return max(1, base + (index % 5) - 2)


def _amount_for(index: int, vendor: Dict[str, Any], invoice_date: date) -> float:
    seasonality = 1 + ((invoice_date.month % 6) - 2) * 0.055
    operational_growth = 1 + (index / TARGET_DEMO_RECORDS) * 0.22
    cadence = 0.82 + ((index * 11) % 23) / 45
    subtotal = float(vendor["base"]) * seasonality * operational_growth * cadence
    if index in HIGH_AMOUNT_INDEXES:
        subtotal *= 3.1 + (index % 3) * 0.35
    if index in NEW_VENDOR_INDEXES:
        subtotal = 11800.0
    return round(subtotal, 2)


def _line_items(category: str, subtotal: float, index: int) -> List[Dict[str, Any]]:
    options = ITEMS.get(category, ITEMS["service"])
    first = options[index % len(options)]
    if subtotal > 5000:
        first_amount = round(subtotal * 0.72, 2)
        second_amount = round(subtotal - first_amount, 2)
        return [
            {"description": first, "quantity": 1.0, "unit_price": first_amount, "total_price": first_amount},
            {"description": "Project surcharge", "quantity": 1.0, "unit_price": second_amount, "total_price": second_amount},
        ]
    quantity = 1 + (index % 3)
    unit_price = round(subtotal / quantity, 2)
    return [{"description": first, "quantity": float(quantity), "unit_price": unit_price, "total_price": subtotal}]


def _new_vendor() -> Dict[str, Any]:
    return {"name": "KuestenKurier GmbH", "category": "service", "base": 11800.0, "vat_id": "DE359017266"}


def _synthetic_record(index: int, duplicate_of: Dict[str, Any] | None = None) -> Dict[str, Any]:
    invoice_date = _date_for(index)
    if duplicate_of:
        record = deepcopy(duplicate_of)
        record["invoice_date"] = invoice_date.isoformat()
        record["due_date"] = (invoice_date + timedelta(days=21)).isoformat()
        record["source_file"] = f"bt_invoice_{index + 1:04d}.pdf"
        record["workflow_age_days"] = _workflow_age(_status_for(index), index)
        record["_risk_events"] = ["duplicate"]
        return record

    vendor = _new_vendor() if index in NEW_VENDOR_INDEXES else VENDORS[(index * 5 + index // 11) % len(VENDORS)]
    subtotal = _amount_for(index, vendor, invoice_date)
    tax = round(subtotal * 0.19, 2)
    due_days = 21 + (index % 10)
    due_date = invoice_date + timedelta(days=due_days)
    risk_events: List[str] = []

    if index in DUE_DATE_ERROR_INDEXES:
        due_date = invoice_date - timedelta(days=4)
        risk_events.append("due_date_error")
    if index in HIGH_AMOUNT_INDEXES:
        risk_events.append("high_amount")
    if index in MISSING_VAT_INDEXES:
        risk_events.append("missing_vat")
    if index in CURRENCY_MISMATCH_INDEXES:
        risk_events.append("currency_mismatch")
    if index in NEW_VENDOR_INDEXES:
        risk_events.append("new_vendor_high_amount")

    currency = "USD" if index in CURRENCY_MISMATCH_INDEXES else "EUR"
    status = _status_for(index)
    return {
        "invoice_number": f"BT-{invoice_date.year}-{index + 1001:04d}",
        "invoice_date": invoice_date.isoformat(),
        "due_date": due_date.isoformat(),
        "billing_address": "BremenTech GmbH, Schlachte 12, 28195 Bremen",
        "shipping_address": "BremenTech GmbH, Schlachte 12, 28195 Bremen",
        "vendor_name": vendor["name"],
        "vendor_vat_id": "" if index in MISSING_VAT_INDEXES else vendor["vat_id"],
        "customer_name": DEMO_COMPANY,
        "line_items": _line_items(str(vendor["category"]), subtotal, index),
        "subtotal": subtotal,
        "tax": tax,
        "total_amount": round(subtotal + tax, 2),
        "currency": currency,
        "source": "PDF upload",
        "source_file": f"bt_invoice_{index + 1:04d}.pdf",
        "ocr_confidence": round(0.91 + ((index * 13) % 8) / 100, 2),
        "risk_model": "rules_v1",
        "workflow_age_days": _workflow_age(status, index),
        "_risk_events": risk_events,
    }


def _expanded_demo_records() -> List[Dict[str, Any]]:
    records: List[Dict[str, Any]] = []
    for index in range(TARGET_DEMO_RECORDS):
        duplicate_source = records[DUPLICATE_LINKS[index]] if index in DUPLICATE_LINKS else None
        records.append(_synthetic_record(index, duplicate_of=duplicate_source))
    return records


def _stored_data(record: Dict[str, Any]) -> Dict[str, Any]:
    return {key: value for key, value in record.items() if not key.startswith("_")}


def _demo_hash(index: int, record: Dict[str, Any]) -> str:
    payload = json.dumps(_stored_data(record), sort_keys=True, ensure_ascii=True)
    return hashlib.sha256(f"bremen-tech-demo-{index}-{payload}".encode("utf-8")).hexdigest()


def _confidence_for(record: Dict[str, Any]) -> Dict[str, float]:
    base = float(record.get("ocr_confidence") or 0.94)
    return {
        "invoice_number": min(base + 0.02, 0.99),
        "invoice_date": base,
        "due_date": 0.86 if "due_date_error" in record.get("_risk_events", []) else base,
        "vendor_name": min(base + 0.01, 0.99),
        "vendor_vat_id": 0.35 if "missing_vat" in record.get("_risk_events", []) else base,
        "customer_name": 0.95,
        "subtotal": base,
        "tax": base,
        "total_amount": min(base + 0.01, 0.99),
        "currency": 0.72 if "currency_mismatch" in record.get("_risk_events", []) else base,
    }


def _clean_confidence(scores: Dict[str, Any], fallback: Dict[str, float]) -> Dict[str, float]:
    cleaned: Dict[str, float] = {}
    for key, fallback_value in fallback.items():
        value = scores.get(key, fallback_value) if isinstance(scores, dict) else fallback_value
        try:
            numeric = float(value)
        except (TypeError, ValueError):
            numeric = fallback_value
        cleaned[key] = max(0.0, min(1.0, numeric))
    return cleaned


def _risk_flags(record: Dict[str, Any], duplicate_count: int = 0) -> List[str]:
    events = set(record.get("_risk_events", []))
    data = _stored_data(record)
    flags = score_invoice_risk(data, duplicate_count=duplicate_count)
    planted = {
        "duplicate": "+30 pts: Duplicate invoice number for the same vendor.",
        "high_amount": "+25 pts: Amount is more than 3x the normal vendor pattern.",
        "missing_vat": "+20 pts: Missing vendor VAT ID.",
        "due_date_error": "+15 pts: Due date is before the invoice date.",
        "new_vendor_high_amount": "+10 pts: New vendor with a high first invoice.",
        "currency_mismatch": "+10 pts: Currency differs from the EUR workspace policy.",
    }
    for event, flag in planted.items():
        if (event == "duplicate" and duplicate_count) or event in events:
            if flag not in flags:
                flags.append(flag)
    return flags


def _score_from_flags(flags: List[str]) -> float:
    return score_from_flags(flags)


def _invoice_model(record: Dict[str, Any]) -> InvoiceData:
    return InvoiceData(**_stored_data(record))


def _local_ai_demo_text(record: Dict[str, Any]) -> str:
    data = _stored_data(record)
    line_items = data.get("line_items") or []
    item_lines = "\n".join(
        f"- {item.get('description')}: qty {item.get('quantity')}, unit {item.get('unit_price')}, total {item.get('total_price')}"
        for item in line_items
    )
    return f"""
INVOICE
Vendor: {data.get('vendor_name')}
Vendor VAT ID: {data.get('vendor_vat_id') or 'MISSING'}
Invoice No: {data.get('invoice_number')}
Invoice Date: {data.get('invoice_date')}
Due Date: {data.get('due_date')}
Bill To: {data.get('customer_name')}
Billing Address: {data.get('billing_address')}
Currency: {data.get('currency')}

Line Items:
{item_lines}

Subtotal: {data.get('currency')} {data.get('subtotal')}
VAT: {data.get('currency')} {data.get('tax')}
Total: {data.get('currency')} {data.get('total_amount')}
Payment terms: Net 21 days.
""".strip()


def seed_local_ai_demo_data(local_model: str, local_base_url: str = "http://localhost:11434", limit: int = 12) -> int:
    if not local_model:
        return 0

    existing_rows = list_invoices()
    existing_ai_rows = [
        row
        for row in existing_rows
        if str(row.get("model_used") or "").startswith("ollama:")
        and "Local AI demo extraction" in " ".join(str(item) for item in row.get("warnings") or [])
    ]
    if len(existing_ai_rows) >= limit:
        return 0

    inserted = 0
    seen_keys = {
        ((row.get("data") or {}).get("invoice_number"), (row.get("data") or {}).get("vendor_name"))
        for row in existing_rows
        if (row.get("data") or {}).get("invoice_number")
    }
    records = _expanded_demo_records()
    start = len(existing_ai_rows)
    for source_index in LOCAL_AI_DEMO_INDEXES[start:limit]:
        source = records[source_index]
        text = _local_ai_demo_text(source)
        filename = f"local_ai_invoice_{source_index + 1:04d}.txt"
        file_hash = content_hash(text.encode("utf-8"))
        document = ProcessedDocument(
            filename=filename,
            content_type="text/plain",
            file_hash=file_hash,
            page_count=1,
            text=text,
            image_payloads=[],
            warnings=[],
        )
        invoice, confidence, model_used, warnings = extract_invoice_with_fallback(
            document=document,
            language="English",
            groq_api_key=None,
            local_model=local_model,
            local_base_url=local_base_url,
            prefer_local=True,
        )
        data = model_dump(invoice)
        source_data = _stored_data(source)
        data.update(
            {
                "vendor_vat_id": source_data.get("vendor_vat_id"),
                "source": "Local AI extraction",
                "source_file": filename,
                "ocr_confidence": 0.97,
                "risk_model": "rules_v1",
                "workflow_age_days": source_data.get("workflow_age_days", 2),
            }
        )
        identity = (data.get("invoice_number"), data.get("vendor_name"))
        duplicate_count = 1 if identity in seen_keys and identity[0] else 0
        if identity[0]:
            seen_keys.add(identity)
        flags = list(_risk_flags(source, duplicate_count=duplicate_count))
        invoice_id = save_invoice(
            filename=filename,
            file_hash=file_hash,
            page_count=1,
            extracted_text=text,
            data=data,
            confidence_scores=_clean_confidence(confidence, _confidence_for(source)),
            category=detect_invoice_type(data),
            model_used=model_used,
            fraud_flags=flags,
            warnings=[*warnings, "Local AI demo extraction via Ollama."],
        )
        update_invoice_status(invoice_id, _status_for(source_index), "Created by local AI demo extraction.")
        inserted += 1

    return inserted


def ensure_demo_data(force: bool = False) -> int:
    existing_rows = list_invoices()
    existing_demo_rows = [row for row in existing_rows if row.get("model_used") == "demo-dataset"]
    if not force and existing_rows and (len(existing_demo_rows) >= TARGET_DEMO_RECORDS or not existing_demo_rows):
        return 0

    records = _expanded_demo_records()
    inserted = 0
    start_index = 0 if force else len(existing_demo_rows)
    seen_keys: set[tuple[str | None, str | None]] = set()
    for row in existing_rows:
        data = row.get("data") or {}
        if data.get("invoice_number"):
            seen_keys.add((data.get("invoice_number"), data.get("vendor_name")))

    for index, record in enumerate(records[start_index:], start=start_index):
        invoice = _invoice_model(record)
        data = _stored_data(record)
        identity = (invoice.invoice_number, invoice.vendor_name)
        duplicate_count = 1 if identity in seen_keys and invoice.invoice_number else 0
        flags = _risk_flags(record, duplicate_count=duplicate_count)
        if identity[0]:
            seen_keys.add(identity)

        invoice_id = save_invoice(
            filename=str(data["source_file"]),
            file_hash=_demo_hash(index, record),
            page_count=1,
            extracted_text=json.dumps(data, ensure_ascii=True),
            data=data,
            confidence_scores=_confidence_for(record),
            category=detect_invoice_type(data),
            model_used="demo-dataset",
            fraud_flags=flags,
            warnings=["Synthetic invoice from the BremenTech GmbH demo portfolio."],
        )
        update_invoice_status(invoice_id, _status_for(index), "Seeded approval workflow.")
        inserted += 1

    return inserted


def normalize_demo_dates() -> int:
    records = _expanded_demo_records()
    updated = 0
    seen_keys: set[tuple[str | None, str | None]] = set()
    with session_scope() as session:
        rows = (
            session.query(InvoiceRecord)
            .filter(InvoiceRecord.model_used == "demo-dataset")
            .order_by(InvoiceRecord.id.asc())
            .all()
        )
        for index, row in enumerate(rows):
            if index >= len(records):
                break
            record = records[index]
            data = _stored_data(record)
            identity = (data.get("invoice_number"), data.get("vendor_name"))
            duplicate_count = 1 if identity in seen_keys and identity[0] else 0
            flags = _risk_flags(record, duplicate_count=duplicate_count)
            seen_keys.add(identity)

            row.filename = str(data["source_file"])
            row.file_hash = _demo_hash(index, record)
            row.page_count = 1
            row.status = _status_for(index)
            row.category = detect_invoice_type(data)
            row.risk_score = _score_from_flags(flags)
            row.extracted_text = json.dumps(data, ensure_ascii=True)
            row.data = data
            row.confidence_scores = _confidence_for(record)
            row.fraud_flags = flags
            row.warnings = ["Synthetic invoice from the BremenTech GmbH demo portfolio."]
            row.notes = ""
            for field in ("data", "confidence_scores", "fraud_flags", "warnings"):
                flag_modified(row, field)
            updated += 1
    return updated
