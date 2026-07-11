from typing import Any, Dict, List

import pandas as pd

from utils import invoice_dataframe, parse_date


SPEND_STATUSES = {"Approved", "Paid"}


def spend_dataframe(rows: List[Dict[str, Any]]) -> pd.DataFrame:
    df = invoice_dataframe(rows)
    if df.empty:
        return df
    return df[df["status"].isin(SPEND_STATUSES)].copy()


def dashboard_metrics(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    df = invoice_dataframe(rows)
    if df.empty:
        return {"invoice_count": 0, "total_spend": 0.0, "average_invoice": 0.0, "high_risk_count": 0, "pending_count": 0}
    spend_df = spend_dataframe(rows)
    totals = pd.to_numeric(spend_df["total_amount"], errors="coerce").fillna(0)
    risk = pd.to_numeric(df["risk_score"], errors="coerce").fillna(0)
    return {
        "invoice_count": int(len(df)),
        "total_spend": float(totals.sum()),
        "average_invoice": float(totals.mean()) if len(totals) else 0.0,
        "high_risk_count": int((risk >= 50).sum()),
        "pending_count": int(df["status"].isin(["Submitted", "Reviewed"]).sum()),
    }


def monthly_trend(rows: List[Dict[str, Any]]) -> pd.DataFrame:
    df = spend_dataframe(rows)
    if df.empty:
        return pd.DataFrame(columns=["month", "total_amount", "invoice_count", "average_amount", "flagged_rate"])
    df["parsed_date"] = df["invoice_date"].apply(parse_date)
    df = df.dropna(subset=["parsed_date"])
    if df.empty:
        return pd.DataFrame(columns=["month", "total_amount", "invoice_count", "average_amount", "flagged_rate"])
    df["month"] = df["parsed_date"].dt.to_period("M").astype(str)
    df["is_flagged"] = pd.to_numeric(df["risk_score"], errors="coerce").fillna(0) >= 50
    return (
        df.groupby("month", as_index=False)
        .agg(
            total_amount=("total_amount", "sum"),
            invoice_count=("id", "count"),
            average_amount=("total_amount", "mean"),
            flagged_rate=("is_flagged", "mean"),
        )
        .sort_values("month")
    )


def vendor_spend(rows: List[Dict[str, Any]]) -> pd.DataFrame:
    df = spend_dataframe(rows)
    if df.empty:
        return pd.DataFrame(columns=["vendor_name", "total_amount", "invoice_count", "risk_score"])
    df["vendor_name"] = df["vendor_name"].fillna("Unknown")
    return (
        df.groupby("vendor_name", as_index=False)
        .agg(total_amount=("total_amount", "sum"), invoice_count=("id", "count"), risk_score=("risk_score", "max"))
        .sort_values("total_amount", ascending=False)
    )


def score_anomalies(rows: List[Dict[str, Any]]) -> pd.DataFrame:
    df = invoice_dataframe(rows)
    if df.empty or len(df) < 3:
        return pd.DataFrame(columns=["id", "invoice_number", "vendor_name", "total_amount", "risk_score", "risk_labels"])

    df["risk_score"] = pd.to_numeric(df["risk_score"], errors="coerce").fillna(0)
    flagged = df[df["risk_score"] >= 50].copy()
    return flagged.sort_values("risk_score", ascending=False)[
        ["id", "invoice_number", "vendor_name", "total_amount", "risk_score", "risk_labels"]
    ]
