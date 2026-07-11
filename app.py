import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, List, Optional

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from analytics import SPEND_STATUSES, dashboard_metrics, monthly_trend, score_anomalies, vendor_spend
from database import (
    bulk_update_status,
    create_job,
    init_db,
    list_audit_events,
    list_invoices,
    list_jobs,
    save_invoice,
    update_job,
    update_invoice_status,
    vendor_summary,
)
from demo_data import ensure_demo_data, normalize_demo_dates, seed_local_ai_demo_data
from exports import invoices_to_csv, invoices_to_excel, invoices_to_pdf
from utils import (
    InvoiceData,
    content_hash,
    detect_invoice_type,
    extract_invoice_with_fallback,
    invoice_dataframe,
    model_dump,
    OllamaClient,
    process_document_bytes,
)
from risk_rules import score_invoice_risk


STATUS_OPTIONS = ["Submitted", "Reviewed", "Approved", "Rejected", "Paid", "Flagged"]
STATUS_COLORS = {
    "Approved": "#14b8a6",
    "Paid": "#22c55e",
    "Flagged": "#f97316",
    "Rejected": "#ef4444",
    "Reviewed": "#8b5cf6",
    "Submitted": "#0ea5e9",
}
CATEGORY_COLORS = {
    "software": "#7c3aed",
    "retail": "#0891b2",
    "service": "#f59e0b",
    "utility": "#16a34a",
    "travel": "#db2777",
    "consulting": "#e11d48",
    "maintenance": "#ea580c",
    "office supplies": "#0d9488",
    "cloud/software": "#2563eb",
    "general": "#64748b",
}


def format_money(value: float, currency: str = "EUR") -> str:
    symbol = "€" if currency == "EUR" else currency
    return f"{symbol}{value:,.2f}" if symbol == "€" else f"{symbol} {value:,.2f}"


def risk_level(score: float) -> str:
    if score >= 50:
        return "High"
    if score >= 20:
        return "Medium"
    return "Low"


def invoice_view_dataframe(rows: List[Dict]) -> pd.DataFrame:
    df = invoice_dataframe(rows)
    if df.empty:
        return pd.DataFrame(
            columns=["Invoice", "Vendor", "Status", "Category", "Amount", "Currency", "Risk", "Risk labels", "Due date"]
        )
    df["risk_level"] = pd.to_numeric(df["risk_score"], errors="coerce").fillna(0).apply(risk_level)
    view = df.rename(
        columns={
            "invoice_number": "Invoice",
            "vendor_name": "Vendor",
            "status": "Status",
            "category": "Category",
            "total_amount": "Amount",
            "currency": "Currency",
            "risk_level": "Risk",
            "due_date": "Due date",
            "risk_labels": "Risk labels",
        }
    )
    view["Risk score"] = pd.to_numeric(df["risk_score"], errors="coerce").fillna(0)
    columns = ["Invoice", "Vendor", "Status", "Category", "Amount", "Currency", "Risk", "Risk labels", "Due date", "Risk score"]
    return view[columns].sort_values(["Risk score", "Amount"], ascending=[False, False])


def show_invoice_table(df: pd.DataFrame, *, hide_score: bool = False) -> None:
    visible = df.drop(columns=["Risk score"], errors="ignore") if hide_score else df
    st.dataframe(
        visible,
        use_container_width=True,
        hide_index=True,
        column_config={
            "Amount": st.column_config.NumberColumn("Amount", format="EUR %.2f"),
            "Risk": st.column_config.TextColumn("Risk"),
            "Risk labels": st.column_config.TextColumn("Risk labels"),
            "Risk score": st.column_config.ProgressColumn("Risk score", min_value=0, max_value=100, format="%d"),
        },
    )


def status_breakdown(rows: List[Dict]) -> pd.DataFrame:
    df = invoice_dataframe(rows)
    if df.empty:
        return pd.DataFrame(columns=["Status", "Invoices"])
    return df.groupby("status", as_index=False).size().rename(columns={"status": "Status", "size": "Invoices"})


def workflow_aging(rows: List[Dict]) -> pd.DataFrame:
    df = invoice_dataframe(rows)
    if df.empty:
        return pd.DataFrame(columns=["Status", "Count", "Avg age", "Share"])
    records = []
    total = max(len(df), 1)
    for status, group in df.groupby("status"):
        ages = []
        for row in group.itertuples():
            source = next((item for item in rows if item.get("id") == row.id), {})
            data = source.get("data") or {}
            ages.append(float(data.get("workflow_age_days") or 0))
        avg_age = sum(ages) / max(len(ages), 1)
        records.append(
            {
                "Status": "Flagged for review" if status == "Flagged" else status,
                "Count": int(len(group)),
                "Avg age": f"{avg_age:.1f} days",
                "Share": f"{len(group) / total:.0%}",
            }
        )
    order = {"Submitted": 0, "Reviewed": 1, "Flagged": 2, "Approved": 3, "Rejected": 4, "Paid": 5}
    return pd.DataFrame(records).sort_values("Status", key=lambda s: s.map(order).fillna(99))


def category_spend(rows: List[Dict]) -> pd.DataFrame:
    df = invoice_dataframe(rows)
    if df.empty:
        return pd.DataFrame(columns=["Category", "Amount"])
    df = df[df["status"].isin(SPEND_STATUSES)]
    return (
        df.groupby("category", as_index=False)["total_amount"]
        .sum()
        .rename(columns={"category": "Category", "total_amount": "Amount"})
        .sort_values("Amount", ascending=False)
    )


def polish_chart(fig, mode: str):
    dark = mode == "Dark"
    paper = "#111827" if dark else "#ffffff"
    plot = "#111827" if dark else "#ffffff"
    text = "#f8fafc" if dark else "#0f172a"
    grid = "#334155" if dark else "#d8e0ea"
    axis = "#94a3b8" if dark else "#475569"
    fig.update_layout(
        template="plotly_dark" if dark else "plotly_white",
        paper_bgcolor=paper,
        plot_bgcolor=plot,
        font=dict(color=text, size=12),
        title=dict(font=dict(size=15, color=text)),
        margin=dict(l=14, r=14, t=48, b=14),
        legend=dict(font=dict(size=11, color=text), bgcolor="rgba(0,0,0,0)"),
        coloraxis_colorbar=dict(tickfont=dict(color=text), title=dict(font=dict(color=text))),
        hoverlabel=dict(
            bgcolor="#ffffff" if dark else "#0f172a",
            bordercolor="#38bdf8" if dark else "#2563eb",
            font=dict(color="#0f172a" if dark else "#ffffff", size=13),
        ),
    )
    fig.update_xaxes(
        gridcolor=grid,
        zerolinecolor=grid,
        linecolor=grid,
        tickfont=dict(color=axis),
        title_font=dict(color=axis),
    )
    fig.update_yaxes(
        gridcolor=grid,
        zerolinecolor=grid,
        linecolor=grid,
        tickfont=dict(color=axis),
        title_font=dict(color=axis),
    )
    return fig


def confidence_dataframe(scores: Dict) -> pd.DataFrame:
    if not scores:
        return pd.DataFrame(columns=["Field", "Confidence"])
    return pd.DataFrame(
        [{"Field": key.replace("_", " ").title(), "Confidence": round(float(value) * 100, 0)} for key, value in scores.items()]
    ).sort_values("Confidence", ascending=True)


def line_items_dataframe(data: Dict) -> pd.DataFrame:
    items = data.get("line_items") or []
    return pd.DataFrame(
        [
            {
                "Description": item.get("description"),
                "Qty": item.get("quantity"),
                "Unit price": item.get("unit_price"),
                "Line total": item.get("total_price"),
            }
            for item in items
        ]
    )


def render_status_badge(status: str) -> str:
    color = STATUS_COLORS.get(status, "#64748b")
    return f'<span style="background:{color}; color:white; padding:4px 9px; border-radius:999px; font-size:0.85rem;">{status}</span>'


def risk_reason_dataframe(flags: List[str]) -> pd.DataFrame:
    rows = []
    for flag in flags:
        text = str(flag)
        points = ""
        reason = text
        if "pts:" in text:
            points, reason = text.split("pts:", 1)
            points = points.replace("+", "").strip()
            reason = reason.strip()
        rows.append({"Points": points or "-", "Reason": reason})
    return pd.DataFrame(rows or [{"Points": "-", "Reason": "No active risk flags"}])


def recommended_action(score: float, flags: List[str]) -> tuple[str, str]:
    if score >= 50:
        return "Manual review required", "Multiple validation failures"
    if flags:
        return "Review before payment", "One or more policy checks need attention"
    return "Standard approval path", "No active risk flags"


def invoice_option_label(row: Dict) -> str:
    data = row.get("data") or {}
    invoice = data.get("invoice_number") or "Missing invoice"
    vendor = data.get("vendor_name") or "Unknown vendor"
    invoice_date = data.get("invoice_date") or "No date"
    amount = format_money(float(data.get("total_amount") or 0), data.get("currency") or "EUR")
    return f"{invoice} - {vendor} - {invoice_date} - {amount}"


def local_ai_rows(rows: List[Dict]) -> List[Dict]:
    return [row for row in rows if str(row.get("model_used") or "").startswith("ollama:")]


def answer_from_demo_data(rows: List[Dict], question: str) -> str:
    df = invoice_dataframe(rows)
    if df.empty:
        return "There are no invoices in the workspace yet."
    question_lower = question.lower()
    total = pd.to_numeric(df["total_amount"], errors="coerce").fillna(0)
    if "total" in question_lower or "spend" in question_lower:
        return f"Total invoice spend is {format_money(float(total.sum()))} across {len(df)} invoices."
    if "highest" in question_lower or "largest" in question_lower:
        row = df.loc[total.idxmax()]
        return f"The largest invoice is {row['invoice_number']} from {row['vendor_name']} for {format_money(float(row['total_amount']))}."
    if "vendor" in question_lower:
        vendors = vendor_spend(rows).head(5)
        names = ", ".join(f"{row.vendor_name} ({format_money(float(row.total_amount))})" for row in vendors.itertuples())
        return f"Top vendors by spend are {names}."
    if "risk" in question_lower or "flag" in question_lower:
        risk = pd.to_numeric(df["risk_score"], errors="coerce").fillna(0)
        high_risk = df[risk >= 50]
        return f"{len(high_risk)} invoices are high risk. The review queue should start with the flagged and submitted invoices."
    if "pending" in question_lower or "approval" in question_lower:
        pending = df[df["status"].isin(["Submitted", "Reviewed"])]
        return f"{len(pending)} invoices are pending action."
    return (
        f"This workspace has {len(df)} invoices, {format_money(float(total.sum()))} in spend, "
        f"and {int((pd.to_numeric(df['risk_score'], errors='coerce').fillna(0) >= 50).sum())} high-risk invoices."
    )


def get_secret(name: str) -> Optional[str]:
    value = os.getenv(name)
    if value:
        return value
    try:
        return st.secrets.get(name)
    except Exception:
        return None


def apply_theme(mode: str) -> None:
    dark = mode == "Dark"
    background = "#0f172a" if dark else "#f8fafc"
    surface = "#111827" if dark else "#ffffff"
    surface_alt = "#1f2937" if dark else "#eef4ff"
    text = "#e5e7eb" if dark else "#0f172a"
    muted = "#94a3b8" if dark else "#475569"
    border = "#334155" if dark else "#cbd5e1"
    accent = "#38bdf8" if dark else "#2563eb"
    accent_text = "#e0f2fe" if dark else "#0f172a"
    input_bg = "#0b1220" if dark else "#ffffff"
    code_bg = "#111827" if dark else "#f1f5f9"
    st.markdown(
        f"""
        <style>
        .stApp,
        [data-testid="stAppViewContainer"],
        [data-testid="stHeader"],
        [data-testid="stToolbar"] {{
            background: {background};
            color: {text};
        }}
        [data-testid="stSidebar"],
        [data-testid="stSidebarContent"] {{
            background: {surface};
            color: {text};
            border-right: 1px solid {border};
        }}
        [data-testid="stSidebar"] * {{
            color: {text};
        }}
        h1, h2, h3, h4, h5, h6, p, span, label,
        [data-testid="stMarkdownContainer"],
        [data-testid="stCaptionContainer"],
        [data-testid="stText"],
        .stRadio label,
        .stSelectbox label,
        .stTextArea label,
        .stTextInput label,
        .stFileUploader label {{
            color: {text};
        }}
        [data-testid="stCaptionContainer"],
        .small-muted {{
            color: {muted};
        }}
        [data-baseweb="tab-list"] {{
            gap: 8px;
            border-bottom: 1px solid {border};
        }}
        [data-baseweb="tab"] {{
            color: {muted};
            background: transparent;
            border-radius: 0;
            padding: 10px 4px;
        }}
        [data-baseweb="tab"][aria-selected="true"] {{
            color: {accent};
            border-bottom: 2px solid {accent};
        }}
        div[data-testid="stMetric"] {{
            background: {surface};
            border: 1px solid {border};
            border-radius: 8px;
            padding: 14px;
        }}
        div[data-testid="stMetric"] label,
        div[data-testid="stMetric"] [data-testid="stMetricValue"] {{
            color: {text};
        }}
        div[data-testid="stAlert"] {{
            background: {surface_alt};
            color: {accent_text};
            border: 1px solid {border};
        }}
        .stButton > button,
        .stDownloadButton > button {{
            background: {surface};
            color: {text};
            border: 1px solid {border};
            border-radius: 8px;
        }}
        .stButton > button[kind="primary"],
        .stButton > button[data-testid="baseButton-primary"] {{
            background: {accent};
            color: #ffffff;
            border-color: {accent};
        }}
        input, textarea,
        [data-baseweb="select"] > div,
        [data-baseweb="base-input"],
        [data-testid="stFileUploaderDropzone"] {{
            background: {input_bg};
            color: {text};
            border-color: {border};
        }}
        [data-testid="stCodeBlock"],
        code, pre {{
            background: {code_bg};
            color: {text};
        }}
        .stDataFrame, [data-testid="stDataFrame"] {{
            border: 1px solid {border};
            border-radius: 8px;
            overflow: hidden;
        }}
        </style>
        """,
        unsafe_allow_html=True,
    )


def existing_duplicate_count(rows: List[Dict], file_hash: str, invoice: InvoiceData) -> int:
    invoice_number = (invoice.invoice_number or "").lower()
    vendor_name = (invoice.vendor_name or "").lower()
    count = 0
    for row in rows:
        data = row.get("data") or {}
        same_hash = row.get("file_hash") == file_hash
        same_identity = invoice_number and data.get("invoice_number", "").lower() == invoice_number
        if same_identity and vendor_name:
            same_identity = data.get("vendor_name", "").lower() == vendor_name
        if same_hash or same_identity:
            count += 1
    return count


def process_upload(
    uploaded_file,
    language: str,
    groq_api_key: Optional[str],
    models: List[str],
    local_model: Optional[str],
    local_base_url: str,
) -> int:
    raw = uploaded_file.read()
    file_hash = content_hash(raw)
    job_id = create_job(uploaded_file.name, file_hash, "Queued", "Waiting for extraction")
    update_job(job_id, "Processing", "Reading document")
    try:
        document = process_document_bytes(uploaded_file.name, raw, uploaded_file.type)
        invoice, confidence, model_used, warnings = extract_invoice_with_fallback(
            document=document,
            language=language,
            groq_api_key=groq_api_key,
            preferred_models=models or None,
            local_model=local_model,
            local_base_url=local_base_url,
        )
        data = model_dump(invoice)
        category = detect_invoice_type(data)
        existing_rows = list_invoices()
        duplicate_count = existing_duplicate_count(existing_rows, document.file_hash, invoice)
        flags = score_invoice_risk(data, duplicate_count=duplicate_count, existing_rows=existing_rows)
        invoice_id = save_invoice(
            filename=document.filename,
            file_hash=document.file_hash,
            page_count=document.page_count,
            extracted_text=document.text,
            data=data,
            confidence_scores=confidence,
            category=category,
            model_used=model_used,
            fraud_flags=flags,
            warnings=warnings,
        )
        update_job(job_id, "Complete", "Invoice saved", invoice_id=invoice_id)
        return invoice_id
    except Exception as exc:
        update_job(job_id, "Error", str(exc))
        raise


def render_dashboard(rows: List[Dict], theme: str) -> None:
    metrics = dashboard_metrics(rows)
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Invoices", metrics["invoice_count"])
    c2.metric("Total spend", format_money(metrics["total_spend"]))
    c3.metric("Avg invoice amount", format_money(metrics["average_invoice"]))
    c4.metric("High priority risk", metrics["high_risk_count"])
    c5.metric("Pending approval", metrics["pending_count"])
    dataset_label = "local Ollama-extracted invoices" if local_ai_rows(rows) else "synthetic BremenTech GmbH fallback invoices"
    st.caption(f"Dataset: {dataset_label}, 2025-2026. Spend includes approved and paid invoices. High priority risk = risk score >= 50.")

    if not rows:
        st.info("No invoices yet. Upload PDFs or images in Intake.")
        return

    trend = monthly_trend(rows)
    vendors = vendor_spend(rows)
    anomalies = score_anomalies(rows)
    statuses = status_breakdown(rows)
    categories = category_spend(rows)

    st.subheader("Spend and Workflow")
    left, middle, right = st.columns([1.35, 1, 1])
    with left:
        if not trend.empty:
            current_month = str(pd.Timestamp.today().to_period("M"))
            has_partial_month = bool((trend["month"] == current_month).any())
            trend = trend[trend["month"] < current_month].tail(24)
            fig = go.Figure()
            fig.add_trace(
                go.Bar(
                    x=trend["month"],
                    y=trend["total_amount"],
                    name="Spend",
                    marker_color="#38bdf8" if theme == "Dark" else "#2563eb",
                    hovertemplate="Month: %{x}<br>Spend: %{y:,.2f}<extra></extra>",
                )
            )
            fig.add_trace(
                go.Scatter(
                    x=trend["month"],
                    y=trend["invoice_count"],
                    name="Invoices",
                    mode="lines+markers",
                    yaxis="y2",
                    line=dict(color="#f97316", width=3),
                    marker=dict(size=7),
                    hovertemplate="Month: %{x}<br>Invoices: %{y}<extra></extra>",
                )
            )
            fig.update_layout(
                title="Monthly spend and invoice volume",
                yaxis=dict(title="Spend"),
                yaxis2=dict(title="Invoices", overlaying="y", side="right", showgrid=False),
                xaxis_title="",
                barmode="group",
            )
            st.plotly_chart(polish_chart(fig, theme), use_container_width=True)
            if has_partial_month:
                st.caption(f"{current_month} is a partial month and is excluded from this chart.")
    with middle:
        if not statuses.empty:
            st.write("Approval aging")
            st.dataframe(
                workflow_aging(rows),
                use_container_width=True,
                hide_index=True,
                column_config={
                    "Count": st.column_config.NumberColumn("Count"),
                    "Share": st.column_config.TextColumn("Share"),
                },
            )
    with right:
        if not categories.empty:
            fig = px.bar(
                categories,
                x="Amount",
                y="Category",
                orientation="h",
                title="Spend by category",
                color="Category",
                color_discrete_map=CATEGORY_COLORS,
            )
            fig.update_traces(hovertemplate="Category: %{y}<br>Spend: %{x:,.2f}<extra></extra>")
            fig.update_layout(showlegend=False, yaxis_title="", xaxis_title="Spend")
            st.plotly_chart(polish_chart(fig, theme), use_container_width=True)

    st.subheader("Vendor Exposure")
    left, right = st.columns([1.3, 1])
    with left:
        if not vendors.empty:
            fig = px.bar(
                vendors.head(10),
                x="total_amount",
                y="vendor_name",
                orientation="h",
                title="Top vendors by spend",
                color="total_amount",
                color_continuous_scale=["#38bdf8", "#2563eb", "#f97316"],
            )
            fig.update_traces(hovertemplate="Vendor: %{y}<br>Spend: %{x:,.2f}<extra></extra>")
            fig.update_layout(yaxis_title="", xaxis_title="Spend", coloraxis_colorbar_title="Spend")
            st.plotly_chart(polish_chart(fig, theme), use_container_width=True)
    with right:
        queue = invoice_view_dataframe(rows)
        queue = queue[(queue["Status"].isin(["Submitted", "Reviewed", "Flagged"])) | (queue["Risk score"] >= 50)].head(10)
        st.write("Priority queue")
        show_invoice_table(queue)

    st.subheader("Invoice Register")
    show_invoice_table(invoice_view_dataframe(rows), hide_score=True)

    if not anomalies.empty:
        st.subheader("Flagged Invoice Reasons")
        st.caption("Risk score is a rules-based review priority, not a fraud verdict.")
        st.dataframe(
            anomalies.rename(
                columns={
                    "invoice_number": "Invoice",
                    "vendor_name": "Vendor",
                    "total_amount": "Amount",
                    "risk_score": "Risk",
                    "risk_labels": "Risk labels",
                }
            ),
            use_container_width=True,
            hide_index=True,
            column_config={"Amount": st.column_config.NumberColumn("Amount", format="€ %.2f")},
        )


def render_intake(
    language: str,
    groq_api_key: Optional[str],
    models: List[str],
    local_model: Optional[str],
    local_base_url: str,
) -> None:
    st.subheader("Document Queue")
    st.caption("Supports JPG, PNG, and multi-page PDF invoices. Native PDF text is used first; OCR/vision fallback is used when available.")
    uploads = st.file_uploader(
        "Upload invoices",
        type=["pdf", "png", "jpg", "jpeg"],
        accept_multiple_files=True,
    )
    col1, col2 = st.columns([1, 3])
    with col1:
        start = st.button("Process queue", type="primary", disabled=not uploads)
    with col2:
        if local_model:
            st.info(f"Local AI extraction is enabled through Ollama model `{local_model}`. If Ollama is unavailable, the parser falls back safely.")
        elif not groq_api_key:
            st.info("Demo extraction is available. Add a local Ollama model later for higher-accuracy extraction on messy scans.")

    if start:
        tasks = []
        if uploads:
            tasks.extend(list(uploads))
        progress = st.progress(0)
        results = []
        errors = []
        total = len(tasks)
        completed = 0
        with ThreadPoolExecutor(max_workers=min(4, max(total, 1))) as executor:
            futures = [
                executor.submit(process_upload, upload, language, groq_api_key, models, local_model, local_base_url)
                for upload in tasks
            ]
            for future in as_completed(futures):
                completed += 1
                progress.progress(completed / total)
                try:
                    results.append(future.result())
                except Exception as exc:
                    errors.append(str(exc))
        if results:
            st.success(f"Processed {len(results)} invoice(s).")
        if errors:
            st.error("Some documents failed: " + " | ".join(errors[:3]))

    jobs = list_jobs()
    if jobs:
        job_df = pd.DataFrame(jobs).drop(columns=["created_at", "updated_at", "file_hash"], errors="ignore")
        st.dataframe(job_df, use_container_width=True, hide_index=True)


def render_review(rows: List[Dict]) -> None:
    st.subheader("Review and Approval")
    if not rows:
        st.info("No invoices are available for review.")
        return
    df = invoice_view_dataframe(rows)
    show_invoice_table(df)

    invoice_options = {invoice_option_label(row): row["id"] for row in rows}
    selected_labels = st.multiselect("Select invoices", options=list(invoice_options.keys()))
    selected_ids = [invoice_options[label] for label in selected_labels]
    c1, c2, c3 = st.columns([1, 2, 1])
    with c1:
        status = st.selectbox("Bulk status", STATUS_OPTIONS)
    with c2:
        notes = st.text_input("Review note")
    with c3:
        if st.button("Apply", disabled=not selected_ids):
            changed = bulk_update_status(selected_ids, status, notes)
            st.success(f"Updated {changed} invoice(s).")
            st.rerun()

    detail_label = st.selectbox("Open invoice detail", options=list(invoice_options.keys()))
    detail_id = invoice_options[detail_label]
    row = next((item for item in rows if item["id"] == detail_id), None)
    if row:
        data = row["data"]
        st.markdown(render_status_badge(row["status"]), unsafe_allow_html=True)
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Vendor", data.get("vendor_name") or "Unknown")
        c2.metric("Invoice", data.get("invoice_number") or "Missing")
        c3.metric("Amount", format_money(float(data.get("total_amount") or 0), data.get("currency") or "EUR"))
        c4.metric("Risk score", f"{row.get('risk_score') or 0:.0f} / 100")

        action, reason = recommended_action(float(row.get("risk_score") or 0), row["fraud_flags"])
        st.info(f"Recommended action: {action}\n\nReason: {reason}")

        left, right = st.columns([1.15, 1])
        with left:
            st.write("Invoice fields")
            field_rows = pd.DataFrame(
                [
                    {"Field": "Invoice date", "Value": data.get("invoice_date") or "-"},
                    {"Field": "Due date", "Value": data.get("due_date") or "-"},
                    {"Field": "Customer", "Value": data.get("customer_name") or "-"},
                    {"Field": "Vendor VAT ID", "Value": data.get("vendor_vat_id") or "Missing"},
                    {"Field": "Billing address", "Value": data.get("billing_address") or "-"},
                    {"Field": "Subtotal", "Value": format_money(float(data.get("subtotal") or 0), data.get("currency") or "EUR")},
                    {"Field": "Tax", "Value": format_money(float(data.get("tax") or 0), data.get("currency") or "EUR")},
                ]
            )
            st.dataframe(field_rows, use_container_width=True, hide_index=True)
            line_items = line_items_dataframe(data)
            if not line_items.empty:
                st.write("Line items")
                st.dataframe(
                    line_items,
                    use_container_width=True,
                    hide_index=True,
                    column_config={
                        "Unit price": st.column_config.NumberColumn("Unit price", format=f"{data.get('currency') or 'EUR'} %.2f"),
                        "Line total": st.column_config.NumberColumn("Line total", format=f"{data.get('currency') or 'EUR'} %.2f"),
                    },
                )
            if row["fraud_flags"]:
                st.write("Risk score breakdown")
                st.dataframe(risk_reason_dataframe(row["fraud_flags"]), use_container_width=True, hide_index=True)
        with right:
            st.write("Extraction source")
            source_rows = pd.DataFrame(
                [
                    {"Field": "Source", "Value": data.get("source") or "Upload"},
                    {"Field": "Source file", "Value": data.get("source_file") or row.get("filename") or "-"},
                    {"Field": "OCR confidence", "Value": f"{float(data.get('ocr_confidence') or 0) * 100:.0f}%"},
                    {"Field": "Risk model", "Value": data.get("risk_model") or "rules_v1"},
                ]
            )
            st.dataframe(source_rows, use_container_width=True, hide_index=True)
            st.write("Extraction confidence")
            confidence = confidence_dataframe(row["confidence_scores"])
            if not confidence.empty:
                st.dataframe(
                    confidence,
                    use_container_width=True,
                    hide_index=True,
                    column_config={
                        "Confidence": st.column_config.ProgressColumn(
                            "Confidence",
                            min_value=0,
                            max_value=100,
                            format="%d%%",
                        )
                    },
                )
            if row["warnings"]:
                st.info(" ".join(str(item) for item in row["warnings"]))
            new_status = st.selectbox("Status", STATUS_OPTIONS, index=STATUS_OPTIONS.index(row["status"]) if row["status"] in STATUS_OPTIONS else 0)
            detail_notes = st.text_area("Notes", value=row.get("notes", ""))
            if st.button("Save status"):
                update_invoice_status(row["id"], new_status, detail_notes)
                st.success("Status saved.")
                st.rerun()


def render_vendors() -> None:
    st.subheader("Vendor Management")
    vendors = vendor_summary()
    if not vendors:
        st.info("Vendor history will appear after invoices are processed.")
        return
    st.dataframe(pd.DataFrame(vendors), use_container_width=True, hide_index=True)


def render_rules(rows: List[Dict]) -> None:
    st.subheader("Invoice Risk Detection")
    st.caption("Rules-based review priority. It highlights invoices to inspect; it does not label fraud.")
    st.caption("Workspace policy expects EUR invoices. Rare non-EUR invoices are not fraud, but they require review.")
    rules = pd.DataFrame(
        [
            {"Points": 30, "Rule": "Duplicate invoice number for the same vendor"},
            {"Points": 25, "Rule": "Amount is more than 3x the normal vendor pattern"},
            {"Points": 20, "Rule": "Missing vendor VAT ID"},
            {"Points": 15, "Rule": "Due date is before the invoice date"},
            {"Points": 10, "Rule": "New vendor with a high first invoice"},
            {"Points": 10, "Rule": "Currency differs from the EUR workspace policy"},
        ]
    )
    st.dataframe(rules, use_container_width=True, hide_index=True)

    flagged = score_anomalies(rows).head(20)
    if not flagged.empty:
        st.write("Highest priority invoices")
        st.dataframe(
            flagged.rename(
                columns={
                    "invoice_number": "Invoice",
                    "vendor_name": "Vendor",
                    "total_amount": "Amount",
                    "risk_score": "Risk",
                    "risk_labels": "Risk labels",
                }
            ),
            use_container_width=True,
            hide_index=True,
            column_config={
                "Amount": st.column_config.NumberColumn("Amount", format="EUR %.2f"),
                "Risk": st.column_config.ProgressColumn("Risk", min_value=0, max_value=100, format="%d"),
            },
        )


def render_assistant(
    rows: List[Dict],
    groq_api_key: Optional[str],
    models: List[str],
    local_model: Optional[str],
    local_base_url: str,
) -> None:
    st.subheader("Invoice Assistant")
    if local_model:
        st.caption(f"Uses local Ollama model `{local_model}` with invoice data as context. If it is unavailable, it falls back to built-in analytics.")
    elif groq_api_key:
        st.caption("Uses the configured cloud AI model with the invoice table as context. If the model is unavailable, it falls back to built-in demo analytics.")
    else:
        st.caption("Demo mode: answers are calculated directly from the invoice database, so it works without any API key.")
    quick = st.selectbox(
        "Quick question",
        [
            "",
            "What is the total spend?",
            "Which invoice is the largest?",
            "Who are the top vendors?",
            "How many invoices need approval?",
            "What should I review first?",
        ],
    )
    question = st.text_input("Ask about stored invoices", value=quick)
    if st.button("Ask", disabled=not question):
        context = invoice_dataframe(rows).head(100).to_json(orient="records")
        prompt = f"Use this invoice table to answer the question. Table: {context}\nQuestion: {question}"
        if local_model:
            try:
                answer = OllamaClient(local_model, local_base_url).run_chatbot_query(prompt)
                st.markdown(answer)
                return
            except Exception:
                pass
        if not groq_api_key:
            st.markdown(answer_from_demo_data(rows, question))
            return
        from utils import GroqClient

        try:
            answer = GroqClient(groq_api_key, models=models or None).run_chatbot_query(prompt)
            st.markdown(answer)
        except Exception:
            st.markdown(answer_from_demo_data(rows, question))


def render_reports(rows: List[Dict]) -> None:
    st.subheader("Reports")
    st.caption("Download board-ready files or review recent workflow activity.")
    c1, c2, c3 = st.columns(3)
    with c1:
        st.download_button("Download CSV", invoices_to_csv(rows), "smart_invoices.csv", "text/csv")
    with c2:
        st.download_button(
            "Download Excel",
            invoices_to_excel(rows),
            "smart_invoices.xlsx",
            "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        )
    with c3:
        st.download_button("Download PDF report", invoices_to_pdf(rows), "smart_invoices_report.pdf", "application/pdf")

    audit = list_audit_events()
    if audit:
        st.write("Recent activity")
        audit_df = pd.DataFrame(audit).drop(columns=["created_at"], errors="ignore")
        audit_df = audit_df.rename(columns={"invoice_id": "Invoice", "action": "Activity", "detail": "Notes"})
        audit_df = audit_df[["Invoice", "Activity", "Notes"]].head(20)
        audit_df["Activity"] = audit_df["Activity"].str.replace("_", " ", regex=False).str.title()
        st.dataframe(audit_df, use_container_width=True, hide_index=True)


def main() -> None:
    init_db()
    if os.getenv("SMARTINVOICEAI_DEMO_DATA", "1") != "0":
        ensure_demo_data()
        normalize_demo_dates()
    st.set_page_config(page_title="SmartInvoiceAI", layout="wide")
    st.title("SmartInvoiceAI")
    st.caption("Invoice approvals, vendor risk, and spend insights in one workspace.")

    with st.sidebar:
        st.header("Workspace")
        theme = st.radio("Theme", ["Dark", "Light"], horizontal=True)
        language = st.selectbox("Invoice language", ["English", "German", "Spanish", "French", "Tamil", "Other"])
        st.divider()
        st.subheader("Local AI")
        local_model_text = st.text_input("Ollama model", value=os.getenv("OLLAMA_MODEL", "llama3.2:3b"))
        local_base_url = st.text_input("Ollama URL", value=os.getenv("OLLAMA_BASE_URL", "http://localhost:11434"))
        local_demo_limit = st.slider("Local AI demo invoices", min_value=4, max_value=12, value=12, step=4)
        if st.button("Run local AI extraction demo", type="primary"):
            with st.spinner("Extracting sample invoices with the local model..."):
                inserted = seed_local_ai_demo_data(local_model_text.strip(), local_base_url, limit=local_demo_limit)
            if inserted:
                st.success(f"Created {inserted} local-AI invoice(s).")
                st.rerun()
            else:
                st.info("Local-AI demo invoices already exist, or the model is unavailable.")
        default_models = "meta-llama/llama-4-scout-17b-16e-instruct,llama-3.3-70b-versatile,llama-3.1-8b-instant"
        model_text = os.getenv("GROQ_MODELS", default_models)
        groq_api_key = get_secret("GROQ_API_KEY")
        stored_rows = list_invoices()
        st.metric("Stored invoices", len(stored_rows))
        st.metric("Local AI invoices", len(local_ai_rows(stored_rows)))
    apply_theme(theme)

    local_model = local_model_text.strip() or None
    models = [model.strip() for model in model_text.split(",") if model.strip()]
    all_rows = list_invoices()
    rows = local_ai_rows(all_rows) or all_rows
    if local_ai_rows(all_rows):
        st.caption("Dashboard is using local Ollama-extracted invoices. Synthetic portfolio rows remain available as fallback data.")
    else:
        st.caption("Dashboard is using the synthetic fallback dataset. Run the local AI extraction demo to create model-extracted invoices.")
    tab_dashboard, tab_intake, tab_review, tab_vendors, tab_rules, tab_assistant, tab_reports = st.tabs(
        ["Overview", "Add Invoices", "Approvals", "Vendors", "Risk Rules", "Assistant", "Reports"]
    )

    with tab_dashboard:
        render_dashboard(rows, theme)
    with tab_intake:
        render_intake(language, groq_api_key, models, local_model, local_base_url)
    with tab_review:
        render_review(rows)
    with tab_vendors:
        render_vendors()
    with tab_rules:
        render_rules(rows)
    with tab_assistant:
        render_assistant(rows, groq_api_key, models, local_model, local_base_url)
    with tab_reports:
        render_reports(rows)


if __name__ == "__main__":
    main()
