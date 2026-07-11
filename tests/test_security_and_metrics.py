import pytest
from fastapi import HTTPException

from analytics import dashboard_metrics, monthly_trend, vendor_spend
from api import require_api_key
from utils import InvoiceData, LineItem, validate_document_upload


def invoice_row(invoice_id, status, amount, vendor="Acme", invoice_date="2026-01-10"):
    return {
        "id": invoice_id,
        "status": status,
        "risk_score": 0,
        "data": {
            "invoice_number": f"INV-{invoice_id}",
            "invoice_date": invoice_date,
            "vendor_name": vendor,
            "total_amount": amount,
        },
        "fraud_flags": [],
        "created_at": "2026-01-10T00:00:00",
    }


def test_upload_validation_checks_type_signature_and_size():
    validate_document_upload("invoice.pdf", b"%PDF-1.4\n", "application/pdf")

    with pytest.raises(ValueError, match="does not match"):
        validate_document_upload("invoice.pdf", b"not a pdf", "application/pdf")
    with pytest.raises(ValueError, match="Only PDF"):
        validate_document_upload("invoice.txt", b"hello", "text/plain")


def test_invoice_values_reject_negative_amounts():
    with pytest.raises(ValueError):
        InvoiceData(total_amount=-1)
    with pytest.raises(ValueError):
        LineItem(quantity=-1)


def test_api_key_is_required(monkeypatch):
    monkeypatch.delenv("SMARTINVOICEAI_API_KEY", raising=False)
    with pytest.raises(HTTPException) as missing_configuration:
        require_api_key(None)
    assert missing_configuration.value.status_code == 503

    monkeypatch.setenv("SMARTINVOICEAI_API_KEY", "test-secret")
    with pytest.raises(HTTPException) as invalid_key:
        require_api_key("wrong")
    assert invalid_key.value.status_code == 401
    assert require_api_key("test-secret") is None


def test_spend_metrics_exclude_unapproved_invoices():
    rows = [
        invoice_row(1, "Approved", 100),
        invoice_row(2, "Paid", 50),
        invoice_row(3, "Rejected", 1000),
        invoice_row(4, "Submitted", 500),
    ]

    metrics = dashboard_metrics(rows)
    assert metrics["invoice_count"] == 4
    assert metrics["total_spend"] == 150
    assert metrics["average_invoice"] == 75
    assert monthly_trend(rows).iloc[0]["total_amount"] == 150
    assert vendor_spend(rows).iloc[0]["total_amount"] == 150
