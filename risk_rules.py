from __future__ import annotations

from datetime import datetime
from statistics import mean
from typing import Any, Dict, Iterable, List, Optional


def normalize_currency(value: Any) -> str:
    if value is None:
        return ""
    text = str(value).strip().upper()
    if not text:
        return ""
    mapping = {
        "€": "EUR",
        "EUR": "EUR",
        "EURO": "EUR",
        "$": "USD",
        "US$": "USD",
        "USD": "USD",
        "£": "GBP",
        "GBP": "GBP",
        "POUND": "GBP",
        "POUNDS": "GBP",
        "RS": "INR",
        "RS.": "INR",
        "₹": "INR",
        "INR": "INR",
    }
    compact = text.replace(" ", "")
    return mapping.get(text) or mapping.get(compact) or text


def _amount(value: Any) -> float:
    try:
        return float(value or 0)
    except (TypeError, ValueError):
        return 0.0


def _date(value: Any) -> Optional[datetime]:
    if not value:
        return None
    for fmt in ("%Y-%m-%d", "%d/%m/%Y", "%m/%d/%Y", "%d-%m-%Y", "%m-%d-%Y", "%Y/%m/%d"):
        try:
            return datetime.strptime(str(value).strip(), fmt)
        except ValueError:
            continue
    return None


def _vendor_history(rows: Iterable[Dict[str, Any]], vendor_name: Optional[str]) -> List[float]:
    if not vendor_name:
        return []
    vendor_key = vendor_name.lower()
    amounts: List[float] = []
    for row in rows:
        data = row.get("data") or row
        if str(data.get("vendor_name") or "").lower() == vendor_key:
            amount = _amount(data.get("total_amount"))
            if amount > 0:
                amounts.append(amount)
    return amounts


def score_invoice_risk(
    invoice_data: Dict[str, Any],
    *,
    duplicate_count: int = 0,
    existing_rows: Optional[Iterable[Dict[str, Any]]] = None,
    workspace_currency: str = "EUR",
) -> List[str]:
    rows = list(existing_rows or [])
    vendor_name = invoice_data.get("vendor_name")
    total = _amount(invoice_data.get("total_amount"))
    flags: List[str] = []

    if duplicate_count > 0:
        flags.append("+30 pts: Duplicate invoice number for the same vendor.")

    vendor_amounts = _vendor_history(rows, vendor_name)
    if vendor_amounts:
        vendor_average = mean(vendor_amounts)
        if vendor_average > 0 and total > vendor_average * 3:
            flags.append("+25 pts: Amount is more than 3x the normal vendor pattern.")

    vat_id = invoice_data.get("vendor_vat_id") or invoice_data.get("vat_id") or invoice_data.get("tax_id")
    if not vat_id and total >= 1000:
        flags.append("+20 pts: Missing vendor VAT ID.")

    invoice_date = _date(invoice_data.get("invoice_date"))
    due_date = _date(invoice_data.get("due_date"))
    if invoice_date and due_date and due_date < invoice_date:
        flags.append("+15 pts: Due date is before the invoice date.")

    if not vendor_amounts and total >= 5000:
        flags.append("+10 pts: New vendor with a high first invoice.")

    currency = normalize_currency(invoice_data.get("currency"))
    expected_currency = normalize_currency(workspace_currency)
    if not currency:
        flags.append("+10 pts: Missing currency.")
    elif expected_currency and currency != expected_currency:
        flags.append(f"+10 pts: Currency differs from the {expected_currency} workspace policy.")

    tax = _amount(invoice_data.get("tax"))
    if total and tax and tax > total * 0.3:
        flags.append("+15 pts: Tax exceeds 30 percent of total.")

    if total >= 1000 and total % 1000 == 0:
        flags.append("+10 pts: Large round amount.")

    return flags


def score_from_flags(flags: Iterable[str]) -> float:
    score = 0
    for flag in flags:
        prefix = str(flag).split("pts:", 1)[0].replace("+", "").strip()
        if prefix.isdigit():
            score += int(prefix)
    return float(min(score, 100))
