import io
from typing import Any, Dict, List

import pandas as pd
from reportlab.lib import colors
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.platypus import Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle

from utils import invoice_dataframe


def invoices_to_csv(rows: List[Dict[str, Any]]) -> str:
    return invoice_dataframe(rows).to_csv(index=False)


def invoices_to_excel(rows: List[Dict[str, Any]]) -> bytes:
    output = io.BytesIO()
    with pd.ExcelWriter(output, engine="openpyxl") as writer:
        df = invoice_dataframe(rows)
        df.to_excel(writer, index=False, sheet_name="Invoices")
        if df.empty:
            vendors = pd.DataFrame(columns=["vendor_name", "invoice_count", "total_spend"])
        else:
            vendors = (
                df.groupby("vendor_name", dropna=False)["total_amount"]
                .agg(["count", "sum"])
                .reset_index()
                .rename(columns={"count": "invoice_count", "sum": "total_spend"})
            )
        vendors.to_excel(writer, index=False, sheet_name="Vendors")
        workbook = writer.book
        for worksheet in workbook.worksheets:
            for column_cells in worksheet.columns:
                width = max(len(str(cell.value or "")) for cell in column_cells[:50])
                worksheet.column_dimensions[column_cells[0].column_letter].width = min(max(width + 2, 12), 42)
    return output.getvalue()


def invoices_to_pdf(rows: List[Dict[str, Any]]) -> bytes:
    output = io.BytesIO()
    doc = SimpleDocTemplate(output, pagesize=letter, rightMargin=36, leftMargin=36, topMargin=36, bottomMargin=36)
    styles = getSampleStyleSheet()
    story = [Paragraph("SmartInvoiceAI Invoice Report", styles["Title"]), Spacer(1, 12)]

    df = invoice_dataframe(rows)
    total_spend = float(df["total_amount"].fillna(0).sum()) if not df.empty and "total_amount" in df else 0.0
    story.append(Paragraph(f"Invoices: {len(df)}", styles["Normal"]))
    story.append(Paragraph(f"Total spend: {total_spend:,.2f}", styles["Normal"]))
    story.append(Spacer(1, 12))

    columns = ["invoice_number", "vendor_name", "status", "category", "total_amount", "risk_score"]
    table_rows = [columns]
    if df.empty:
        table_rows.append(["No invoices yet", "", "", "", "", ""])
    else:
        for _, row in df[columns].fillna("").head(40).iterrows():
            table_rows.append([str(row[col]) for col in columns])

    table = Table(table_rows, repeatRows=1)
    table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#1f2937")),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                ("GRID", (0, 0), (-1, -1), 0.25, colors.HexColor("#cbd5e1")),
                ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                ("FONTSIZE", (0, 0), (-1, -1), 8),
                ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#f8fafc")]),
            ]
        )
    )
    story.append(table)
    doc.build(story)
    return output.getvalue()
