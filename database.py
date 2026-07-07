import json
import os
import re
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

from sqlalchemy import (
    JSON,
    Column,
    DateTime,
    Float,
    ForeignKey,
    Integer,
    String,
    Text,
    create_engine,
    func,
    select,
)
from sqlalchemy.orm import Session, declarative_base, relationship, sessionmaker


DEFAULT_DB_PATH = Path("data") / "smart_invoice_ai.sqlite3"
DATABASE_URL = os.getenv("DATABASE_URL", f"sqlite:///{DEFAULT_DB_PATH.as_posix()}")


def _engine_kwargs() -> Dict[str, Any]:
    if DATABASE_URL.startswith("sqlite"):
        DEFAULT_DB_PATH.parent.mkdir(parents=True, exist_ok=True)
        return {"connect_args": {"check_same_thread": False}}
    return {"pool_pre_ping": True}


engine = create_engine(DATABASE_URL, future=True, **_engine_kwargs())
SessionLocal = sessionmaker(bind=engine, expire_on_commit=False, future=True)
Base = declarative_base()


class InvoiceRecord(Base):
    __tablename__ = "invoice_records"

    id = Column(Integer, primary_key=True)
    filename = Column(String(255), nullable=False)
    file_hash = Column(String(64), nullable=False, index=True)
    page_count = Column(Integer, default=1)
    status = Column(String(32), default="Submitted", index=True)
    category = Column(String(64), default="general", index=True)
    model_used = Column(String(128), default="regex-fallback")
    risk_score = Column(Float, default=0.0)
    extracted_text = Column(Text, default="")
    data = Column(JSON, nullable=False)
    confidence_scores = Column(JSON, default=dict)
    fraud_flags = Column(JSON, default=list)
    warnings = Column(JSON, default=list)
    notes = Column(Text, default="")
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow, nullable=False)

    audit_events = relationship("AuditEvent", back_populates="invoice", cascade="all, delete-orphan")


class ProcessingJob(Base):
    __tablename__ = "processing_jobs"

    id = Column(Integer, primary_key=True)
    filename = Column(String(255), nullable=False)
    file_hash = Column(String(64), nullable=False, index=True)
    status = Column(String(32), default="Queued", index=True)
    detail = Column(Text, default="")
    invoice_id = Column(Integer, ForeignKey("invoice_records.id"), nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow, nullable=False)


class AuditEvent(Base):
    __tablename__ = "audit_events"

    id = Column(Integer, primary_key=True)
    invoice_id = Column(Integer, ForeignKey("invoice_records.id"), nullable=True)
    action = Column(String(80), nullable=False)
    detail = Column(Text, default="")
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)

    invoice = relationship("InvoiceRecord", back_populates="audit_events")


def init_db() -> None:
    Base.metadata.create_all(bind=engine)


@contextmanager
def session_scope():
    init_db()
    session = SessionLocal()
    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()


def create_job(filename: str, file_hash: str, status: str = "Queued", detail: str = "") -> int:
    with session_scope() as session:
        job = ProcessingJob(filename=filename, file_hash=file_hash, status=status, detail=detail)
        session.add(job)
        session.flush()
        return job.id


def update_job(job_id: int, status: str, detail: str = "", invoice_id: Optional[int] = None) -> None:
    with session_scope() as session:
        job = session.get(ProcessingJob, job_id)
        if not job:
            return
        job.status = status
        job.detail = detail
        if invoice_id is not None:
            job.invoice_id = invoice_id


def duplicate_count(session: Session, file_hash: str, invoice_number: Optional[str], vendor_name: Optional[str]) -> int:
    count = session.scalar(select(func.count()).select_from(InvoiceRecord).where(InvoiceRecord.file_hash == file_hash)) or 0
    if invoice_number:
        query = select(func.count()).select_from(InvoiceRecord).where(
            func.lower(func.json_extract(InvoiceRecord.data, "$.invoice_number")) == invoice_number.lower()
        )
        if vendor_name:
            query = query.where(func.lower(func.json_extract(InvoiceRecord.data, "$.vendor_name")) == vendor_name.lower())
        try:
            count += session.scalar(query) or 0
        except Exception:
            rows = session.scalars(select(InvoiceRecord)).all()
            count += sum(
                1
                for row in rows
                if (row.data or {}).get("invoice_number", "").lower() == invoice_number.lower()
                and (not vendor_name or (row.data or {}).get("vendor_name", "").lower() == vendor_name.lower())
            )
    return int(count)


def save_invoice(
    *,
    filename: str,
    file_hash: str,
    page_count: int,
    extracted_text: str,
    data: Dict[str, Any],
    confidence_scores: Dict[str, Any],
    category: str,
    model_used: str,
    fraud_flags: List[str],
    warnings: List[str],
) -> int:
    risk_score = _risk_score_from_flags(fraud_flags, confidence_scores)
    with session_scope() as session:
        record = InvoiceRecord(
            filename=filename,
            file_hash=file_hash,
            page_count=page_count,
            status="Submitted",
            category=category,
            model_used=model_used,
            risk_score=risk_score,
            extracted_text=extracted_text,
            data=data,
            confidence_scores=confidence_scores,
            fraud_flags=fraud_flags,
            warnings=warnings,
        )
        session.add(record)
        session.flush()
        audit = AuditEvent(invoice_id=record.id, action="created", detail=f"Created from {filename}")
        session.add(audit)
        return record.id


def _risk_score_from_flags(fraud_flags: List[str], confidence_scores: Dict[str, Any]) -> float:
    scored_reasons = 0
    score = 0
    for flag in fraud_flags:
        match = re.match(r"\+?(\d+)\s*(?:points|pts)\s*:", str(flag), re.IGNORECASE)
        if match:
            score += int(match.group(1))
            scored_reasons += 1
    if scored_reasons:
        return float(min(score, 100))
    low_confidence = 0
    for value in confidence_scores.values():
        try:
            numeric = float(value)
        except (TypeError, ValueError):
            continue
        if numeric and numeric < 0.7:
            low_confidence += 1
    return min(100.0, len(fraud_flags) * 25.0 + low_confidence * 8.0)


def record_audit(invoice_id: Optional[int], action: str, detail: str = "") -> None:
    with session_scope() as session:
        session.add(AuditEvent(invoice_id=invoice_id, action=action, detail=detail))


def list_invoices(status: Optional[str] = None) -> List[Dict[str, Any]]:
    with session_scope() as session:
        stmt = select(InvoiceRecord).order_by(InvoiceRecord.created_at.desc())
        if status:
            stmt = stmt.where(InvoiceRecord.status == status)
        return [_record_to_dict(record) for record in session.scalars(stmt).all()]


def get_invoice(invoice_id: int) -> Optional[Dict[str, Any]]:
    with session_scope() as session:
        record = session.get(InvoiceRecord, invoice_id)
        return _record_to_dict(record) if record else None


def update_invoice_status(invoice_id: int, status: str, notes: str = "") -> None:
    with session_scope() as session:
        record = session.get(InvoiceRecord, invoice_id)
        if not record:
            return
        old_status = record.status
        record.status = status
        if notes:
            record.notes = notes
        session.add(AuditEvent(invoice_id=invoice_id, action="status_changed", detail=f"{old_status} -> {status}. {notes}"))


def bulk_update_status(invoice_ids: Iterable[int], status: str, notes: str = "") -> int:
    changed = 0
    for invoice_id in invoice_ids:
        update_invoice_status(int(invoice_id), status, notes)
        changed += 1
    return changed


def list_jobs(limit: int = 30) -> List[Dict[str, Any]]:
    with session_scope() as session:
        rows = session.scalars(select(ProcessingJob).order_by(ProcessingJob.created_at.desc()).limit(limit)).all()
        return [
            {
                "id": row.id,
                "filename": row.filename,
                "file_hash": row.file_hash,
                "status": row.status,
                "detail": row.detail,
                "invoice_id": row.invoice_id,
                "created_at": row.created_at.isoformat(),
                "updated_at": row.updated_at.isoformat(),
            }
            for row in rows
        ]


def list_audit_events(limit: int = 50) -> List[Dict[str, Any]]:
    with session_scope() as session:
        rows = session.scalars(select(AuditEvent).order_by(AuditEvent.created_at.desc()).limit(limit)).all()
        return [
            {
                "id": row.id,
                "invoice_id": row.invoice_id,
                "action": row.action,
                "detail": row.detail,
                "created_at": row.created_at.isoformat(),
            }
            for row in rows
        ]


def _record_to_dict(record: InvoiceRecord) -> Dict[str, Any]:
    return {
        "id": record.id,
        "filename": record.filename,
        "file_hash": record.file_hash,
        "page_count": record.page_count,
        "status": record.status,
        "category": record.category,
        "model_used": record.model_used,
        "risk_score": record.risk_score,
        "extracted_text": record.extracted_text,
        "data": record.data or {},
        "confidence_scores": record.confidence_scores or {},
        "fraud_flags": record.fraud_flags or [],
        "warnings": record.warnings or [],
        "notes": record.notes or "",
        "created_at": record.created_at.isoformat(),
        "updated_at": record.updated_at.isoformat(),
    }


def vendor_summary() -> List[Dict[str, Any]]:
    rows = list_invoices()
    vendors: Dict[str, Dict[str, Any]] = {}
    for row in rows:
        data = row["data"]
        vendor = data.get("vendor_name") or "Unknown"
        total = float(data.get("total_amount") or 0)
        summary = vendors.setdefault(vendor, {"vendor_name": vendor, "invoice_count": 0, "total_spend": 0.0, "risk_score": 0.0})
        summary["invoice_count"] += 1
        summary["total_spend"] += total
        summary["risk_score"] = max(summary["risk_score"], row.get("risk_score") or 0.0)
    return sorted(vendors.values(), key=lambda item: item["total_spend"], reverse=True)


def database_info() -> Dict[str, str]:
    redacted = DATABASE_URL
    if "@" in redacted:
        redacted = redacted.split("://", 1)[0] + "://***@" + redacted.rsplit("@", 1)[-1]
    return {"database_url": redacted}


def export_json_snapshot(path: str = "data/invoices_snapshot.json") -> str:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    payload = {"invoices": list_invoices(), "audit_events": list_audit_events(500), "jobs": list_jobs(500)}
    Path(path).write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return path
