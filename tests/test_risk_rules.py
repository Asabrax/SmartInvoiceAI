from risk_rules import score_invoice_risk


def test_duplicate_invoice_flag():
    flags = score_invoice_risk(
        {
            "vendor_name": "NordCloud GmbH",
            "total_amount": 950.0,
            "currency": "EUR",
            "vendor_vat_id": "DE123456789",
        },
        duplicate_count=1,
    )

    assert any("Duplicate invoice number" in flag for flag in flags)


def test_due_date_before_invoice_date():
    flags = score_invoice_risk(
        {
            "vendor_name": "RepairPro Bremen GmbH",
            "invoice_date": "2026-05-15",
            "due_date": "2026-05-01",
            "total_amount": 850.0,
            "currency": "EUR",
            "vendor_vat_id": "DE123456789",
        }
    )

    assert any("Due date is before the invoice date" in flag for flag in flags)


def test_currency_symbol_normalization():
    flags = score_invoice_risk(
        {
            "vendor_name": "Bremen OfficePoint GmbH",
            "total_amount": 1200.0,
            "currency": "€",
            "vendor_vat_id": "DE987654321",
        },
        workspace_currency="EUR",
    )

    assert not any("Currency differs" in flag for flag in flags)


def test_missing_vat_high_amount():
    flags = score_invoice_risk(
        {
            "vendor_name": "HafenStrom Utilities GmbH",
            "total_amount": 2400.0,
            "currency": "EUR",
            "vendor_vat_id": "",
        }
    )

    assert any("Missing vendor VAT ID" in flag for flag in flags)
