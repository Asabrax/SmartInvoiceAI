import os
import tempfile

import pytest

os.environ["DATABASE_URL"] = f"sqlite:///{tempfile.NamedTemporaryFile(suffix='.sqlite3', delete=False).name}"

import database
from demo_data import ensure_demo_data
from exports import invoices_to_csv, invoices_to_excel, invoices_to_pdf
from utils import detect_invoice_type, fraud_flags, model_dump, parse_invoice_text


def test_parse_invoice_text_extracts_core_fields():
    invoice, confidence = parse_invoice_text(
        """
        Acme Cloud Services
        Invoice # INV-2026-0042
        Date: 2026-06-15
        Due Date: 2026-07-15
        Bill To: Example GmbH
        Vendor VAT ID: DE123456789
        Payment Terms: Net 30
        Subtotal: EUR 1,200.00
        VAT: EUR 228.00
        Total: EUR 1,428.00
        """
    )

    assert invoice.invoice_number == "INV-2026-0042"
    assert invoice.vendor_name == "Acme Cloud Services"
    assert invoice.customer_name == "Example GmbH"
    assert invoice.vendor_vat_id == "DE123456789"
    assert invoice.payment_terms == "Net 30"
    assert invoice.total_amount == 1428.0
    assert invoice.currency == "EUR"
    assert confidence["vendor_vat_id"] > 0
    assert confidence["invoice_number"] > 0


def test_category_and_fraud_flags():
    invoice, _ = parse_invoice_text("Vendor: Acme Cloud\nInvoice No: A-1\nTotal: 10000.00")
    data = model_dump(invoice)

    assert detect_invoice_type(data) == "software"
    assert any("Large round amount" in flag for flag in fraud_flags(invoice))


def test_database_save_and_exports():
    database.init_db()
    invoice, confidence = parse_invoice_text("Acme\nInvoice No: A-2\nTotal: 42.00 USD")
    invoice_id = database.save_invoice(
        filename="sample.pdf",
        file_hash="abc123",
        page_count=1,
        extracted_text="sample",
        data=model_dump(invoice),
        confidence_scores=confidence,
        category="general",
        model_used="regex-fallback",
        fraud_flags=[],
        warnings=[],
    )

    rows = database.list_invoices()
    assert rows[0]["id"] == invoice_id
    assert "invoice_number" in invoices_to_csv(rows)
    assert invoices_to_excel(rows).startswith(b"PK")
    assert invoices_to_pdf(rows).startswith(b"%PDF")


def test_demo_data_seeds_fresh_database():
    inserted = ensure_demo_data(force=True)
    rows = database.list_invoices()

    assert inserted >= 1
    assert len(rows) >= inserted
    assert any(row["model_used"] == "demo-dataset" for row in rows)


def test_status_updates_validate_status_and_missing_ids():
    with pytest.raises(ValueError):
        database.update_invoice_status(1, "MadeUp")
    assert database.update_invoice_status(999999, "Approved") is False
    assert database.bulk_update_status([999998, 999999], "Approved") == 0
