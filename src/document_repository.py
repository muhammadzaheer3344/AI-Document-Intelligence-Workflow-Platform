"""SQLite-backed document metadata and organized file storage."""

from __future__ import annotations

import hashlib
import json
import sqlite3
import uuid
from contextlib import closing
from datetime import datetime
from pathlib import Path
from typing import Any

DOCUMENT_TYPES = ("Invoice", "Resume", "Other")
STATUSES = ("New", "Processing", "Needs Review", "Approved", "Rejected", "Completed", "Failed")
VALID_TRANSITIONS = {
    "New": {"Processing", "Failed"},
    "Processing": {"Needs Review", "Completed", "Failed"},
    "Needs Review": {"Processing", "Approved", "Rejected", "Failed"},
    "Approved": {"Completed"},
    "Rejected": set(),
    "Completed": set(),
    "Failed": {"Processing"},
}


class WorkflowTransitionError(ValueError):
    """Raised when a document is moved along an unsupported workflow edge."""


class DuplicateDocumentError(Exception):
    def __init__(self, document: dict[str, Any]) -> None:
        self.document = document
        super().__init__("This file has already been uploaded.")


class DocumentRepository:
    """Owns the SQLite database and files referenced by its records."""

    def __init__(self, db_path: str | Path, storage_root: str | Path | None = None) -> None:
        self.db_path = Path(db_path)
        self.storage_root = Path(storage_root) if storage_root else self.db_path.parent / "storage"
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.storage_root.mkdir(parents=True, exist_ok=True)
        self._initialize()

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.db_path)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA foreign_keys = ON")
        return connection

    def _initialize(self) -> None:
        with closing(self._connect()) as connection:
            with connection:
                connection.execute(
                    """
                    CREATE TABLE IF NOT EXISTS documents (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        original_filename TEXT NOT NULL,
                        stored_filename TEXT NOT NULL,
                        document_type TEXT NOT NULL,
                        upload_date TEXT NOT NULL,
                        company TEXT,
                        invoice_number TEXT,
                        total_amount TEXT,
                        file_path TEXT NOT NULL,
                        text_preview TEXT,
                        file_hash TEXT NOT NULL UNIQUE,
                        status TEXT NOT NULL,
                        extracted_fields TEXT NOT NULL DEFAULT '{}'
                    )
                    """
                )
                existing_columns = {row[1] for row in connection.execute("PRAGMA table_info(documents)")}
                for column_name, column_type in (
                    ("predicted_type", "TEXT"),
                    ("confidence", "REAL"),
                    ("validation_errors", "TEXT NOT NULL DEFAULT '{}'"),
                    ("workflow_reason", "TEXT NOT NULL DEFAULT ''"),
                ):
                    if column_name not in existing_columns:
                        connection.execute(f"ALTER TABLE documents ADD COLUMN {column_name} {column_type}")
                connection.execute("UPDATE documents SET status = 'Completed' WHERE status = 'Processed'")
                connection.execute(
                    """
                    CREATE TABLE IF NOT EXISTS audit_events (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        document_id INTEGER NOT NULL,
                        action TEXT NOT NULL,
                        previous_status TEXT,
                        new_status TEXT NOT NULL,
                        timestamp TEXT NOT NULL,
                        reason TEXT NOT NULL DEFAULT '',
                        FOREIGN KEY(document_id) REFERENCES documents(id) ON DELETE CASCADE
                    )
                    """
                )
                connection.execute("CREATE INDEX IF NOT EXISTS idx_audit_document_time ON audit_events(document_id, timestamp)")
                connection.execute(
                    """
                    INSERT INTO audit_events (document_id, action, previous_status, new_status, timestamp, reason)
                    SELECT d.id, 'Week 5 migration',
                           CASE WHEN d.status = 'Completed' THEN 'Processed' ELSE NULL END,
                           d.status, d.upload_date,
                           'Existing Week 4 record imported; earlier workflow events were not available.'
                    FROM documents AS d
                    WHERE NOT EXISTS (
                        SELECT 1 FROM audit_events AS a WHERE a.document_id = d.id
                    )
                    """
                )
                connection.execute("CREATE INDEX IF NOT EXISTS idx_documents_upload_date ON documents(upload_date)")
                connection.execute("CREATE INDEX IF NOT EXISTS idx_documents_type ON documents(document_type)")
                connection.execute("CREATE INDEX IF NOT EXISTS idx_documents_status ON documents(status)")

    @staticmethod
    def hash_bytes(file_bytes: bytes) -> str:
        return hashlib.sha256(file_bytes).hexdigest()

    def get_by_hash(self, file_hash: str) -> dict[str, Any] | None:
        with closing(self._connect()) as connection:
            row = connection.execute("SELECT * FROM documents WHERE file_hash = ?", (file_hash,)).fetchone()
        return self._row_to_dict(row) if row else None

    def create_document(
        self,
        *,
        file_bytes: bytes,
        original_filename: str,
        document_type: str,
        status: str,
        fields: dict[str, Any] | None = None,
        text_preview: str = "",
        predicted_type: str | None = None,
        confidence: float | None = None,
        validation_errors: dict[str, str] | None = None,
        workflow_reason: str = "",
    ) -> dict[str, Any]:
        document_type = document_type if document_type in DOCUMENT_TYPES else "Other"
        if status == "Processed":
            status = "Completed"
        if status not in STATUSES:
            raise ValueError(f"Unsupported workflow status: {status}")
        file_hash = self.hash_bytes(file_bytes)
        existing = self.get_by_hash(file_hash)
        if existing:
            raise DuplicateDocumentError(existing)

        safe_original = Path(original_filename).name or "document"
        stored_filename = f"{uuid.uuid4().hex}{Path(safe_original).suffix.lower()}"
        folder = self.storage_root / document_type.lower()
        folder.mkdir(parents=True, exist_ok=True)
        stored_path = folder / stored_filename
        fields = fields or {}
        record = (
            safe_original, stored_filename, document_type,
            datetime.now().astimezone().isoformat(timespec="seconds"),
            fields.get("Company Name"), fields.get("Invoice Number"), fields.get("Total Amount"),
            str(stored_path.resolve()), text_preview[:2000], file_hash, status,
            json.dumps(fields, ensure_ascii=True), predicted_type or document_type, confidence,
            json.dumps(validation_errors or {}, ensure_ascii=True), workflow_reason,
        )
        try:
            stored_path.write_bytes(file_bytes)
            with closing(self._connect()) as connection:
                with connection:
                    document_id = connection.execute(
                        """
                        INSERT INTO documents (
                            original_filename, stored_filename, document_type, upload_date,
                            company, invoice_number, total_amount, file_path, text_preview,
                            file_hash, status, extracted_fields, predicted_type, confidence,
                            validation_errors, workflow_reason
                        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                        """,
                        record,
                    ).lastrowid
                    connection.execute(
                        "INSERT INTO audit_events (document_id, action, previous_status, new_status, timestamp, reason) "
                        "VALUES (?, ?, ?, ?, ?, ?)",
                        (document_id, "Uploaded", None, status,
                         datetime.now().astimezone().isoformat(timespec="seconds"), "Document uploaded."),
                    )
        except sqlite3.IntegrityError:
            stored_path.unlink(missing_ok=True)
            existing = self.get_by_hash(file_hash)
            if existing:
                raise DuplicateDocumentError(existing) from None
            raise
        except Exception:
            stored_path.unlink(missing_ok=True)
            raise
        return self.get_document(document_id)  # type: ignore[return-value]

    def get_document(self, document_id: int) -> dict[str, Any] | None:
        with closing(self._connect()) as connection:
            row = connection.execute("SELECT * FROM documents WHERE id = ?", (document_id,)).fetchone()
        return self._row_to_dict(row) if row else None

    def transition_document(
        self,
        document_id: int,
        new_status: str,
        *,
        action: str,
        reason: str = "",
        reviewer_note: str = "",
        validation_errors: dict[str, str] | None = None,
    ) -> dict[str, Any]:
        """Atomically validate and apply a state transition, then audit it."""
        with closing(self._connect()) as connection:
            with connection:
                row = connection.execute("SELECT status FROM documents WHERE id = ?", (document_id,)).fetchone()
                if row is None:
                    raise KeyError(f"Document {document_id} does not exist.")
                previous_status = row["status"]
                if new_status not in VALID_TRANSITIONS.get(previous_status, set()):
                    raise WorkflowTransitionError(f"Cannot transition document from {previous_status} to {new_status}.")
                full_reason = "; ".join(part for part in (reason.strip(), reviewer_note.strip()) if part)
                connection.execute(
                    "UPDATE documents SET status = ?, workflow_reason = ?, validation_errors = ? WHERE id = ?",
                    (new_status, full_reason, json.dumps(validation_errors or {}, ensure_ascii=True), document_id),
                )
                connection.execute(
                    "INSERT INTO audit_events (document_id, action, previous_status, new_status, timestamp, reason) "
                    "VALUES (?, ?, ?, ?, ?, ?)",
                    (document_id, action, previous_status, new_status,
                     datetime.now().astimezone().isoformat(timespec="seconds"), full_reason),
                )
        return self.get_document(document_id)  # type: ignore[return-value]

    def _insert_audit_event(
        self, document_id: int, action: str, previous_status: str | None,
        new_status: str, reason: str,
    ) -> None:
        with closing(self._connect()) as connection:
            with connection:
                connection.execute(
                    "INSERT INTO audit_events (document_id, action, previous_status, new_status, timestamp, reason) "
                    "VALUES (?, ?, ?, ?, ?, ?)",
                    (document_id, action, previous_status, new_status,
                     datetime.now().astimezone().isoformat(timespec="seconds"), reason),
                )

    def list_audit_events(self, document_id: int) -> list[dict[str, Any]]:
        with closing(self._connect()) as connection:
            rows = connection.execute(
                "SELECT action, previous_status, new_status, timestamp, reason "
                "FROM audit_events WHERE document_id = ? ORDER BY id DESC", (document_id,),
            ).fetchall()
        return [dict(row) for row in rows]

    def record_workflow_event(self, document_id: int, action: str, reason: str) -> None:
        """Record an action that does not change the document's current state."""
        with closing(self._connect()) as connection:
            with connection:
                row = connection.execute("SELECT status FROM documents WHERE id = ?", (document_id,)).fetchone()
                if row is None:
                    raise KeyError(f"Document {document_id} does not exist.")
                connection.execute(
                    "INSERT INTO audit_events (document_id, action, previous_status, new_status, timestamp, reason) "
                    "VALUES (?, ?, ?, ?, ?, ?)",
                    (document_id, action, row["status"], row["status"],
                     datetime.now().astimezone().isoformat(timespec="seconds"), reason),
                )

    def get_metrics(self) -> dict[str, Any]:
        with closing(self._connect()) as connection:
            status_rows = connection.execute("SELECT status, COUNT(*) AS count FROM documents GROUP BY status").fetchall()
            type_rows = connection.execute("SELECT document_type, COUNT(*) AS count FROM documents GROUP BY document_type").fetchall()
        statuses = {row["status"]: row["count"] for row in status_rows}
        return {
            "total": sum(statuses.values()),
            "statuses": statuses,
            "by_type": {row["document_type"]: row["count"] for row in type_rows},
        }

    def list_documents(
        self, *, search: str = "", document_type: str = "All", status: str = "All",
        start_date: str | None = None, end_date: str | None = None, newest_first: bool = True,
    ) -> list[dict[str, Any]]:
        clauses: list[str] = []
        params: list[Any] = []
        if search.strip():
            term = f"%{search.strip()}%"
            clauses.append("(original_filename LIKE ? OR company LIKE ? OR invoice_number LIKE ? OR document_type LIKE ? OR text_preview LIKE ?)")
            params.extend([term] * 5)
        if document_type != "All":
            clauses.append("document_type = ?")
            params.append(document_type)
        if status != "All":
            clauses.append("status = ?")
            params.append(status)
        if start_date:
            clauses.append("date(upload_date) >= date(?)")
            params.append(start_date)
        if end_date:
            clauses.append("date(upload_date) <= date(?)")
            params.append(end_date)
        query = """
            SELECT documents.*,
                   (SELECT action FROM audit_events WHERE document_id = documents.id ORDER BY id DESC LIMIT 1) AS latest_workflow_action,
                   (SELECT timestamp FROM audit_events WHERE document_id = documents.id ORDER BY id DESC LIMIT 1) AS latest_workflow_timestamp
            FROM documents
        """
        if clauses:
            query += " WHERE " + " AND ".join(clauses)
        query += " ORDER BY upload_date " + ("DESC" if newest_first else "ASC")
        with closing(self._connect()) as connection:
            rows = connection.execute(query, params).fetchall()
        return [self._row_to_dict(row) for row in rows]

    def update_document(self, document_id: int, **changes: Any) -> dict[str, Any] | None:
        allowed = {
            "document_type", "company", "invoice_number", "total_amount", "text_preview",
            "extracted_fields", "predicted_type", "confidence", "validation_errors", "workflow_reason",
        }
        updates = {key: value for key, value in changes.items() if key in allowed}
        if "extracted_fields" in updates and isinstance(updates["extracted_fields"], dict):
            updates["extracted_fields"] = json.dumps(updates["extracted_fields"], ensure_ascii=True)
        if "validation_errors" in updates and isinstance(updates["validation_errors"], dict):
            updates["validation_errors"] = json.dumps(updates["validation_errors"], ensure_ascii=True)
        if updates:
            assignments = ", ".join(f"{key} = ?" for key in updates)
            with closing(self._connect()) as connection:
                with connection:
                    connection.execute(f"UPDATE documents SET {assignments} WHERE id = ?", [*updates.values(), document_id])
        return self.get_document(document_id)

    def delete_document(self, document_id: int) -> bool:
        document = self.get_document(document_id)
        if not document:
            return False
        with closing(self._connect()) as connection:
            with connection:
                connection.execute("DELETE FROM documents WHERE id = ?", (document_id,))
        Path(document["file_path"]).unlink(missing_ok=True)
        return True

    @staticmethod
    def _row_to_dict(row: sqlite3.Row) -> dict[str, Any]:
        document = dict(row)
        try:
            document["extracted_fields"] = json.loads(document["extracted_fields"] or "{}")
        except (TypeError, json.JSONDecodeError):
            document["extracted_fields"] = {}
        try:
            document["validation_errors"] = json.loads(document.get("validation_errors") or "{}")
        except (TypeError, json.JSONDecodeError):
            document["validation_errors"] = {}
        return document
