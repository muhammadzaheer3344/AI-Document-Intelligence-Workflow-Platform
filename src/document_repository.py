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
STATUSES = ("Processed", "Needs Review", "Failed")


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
    ) -> dict[str, Any]:
        document_type = document_type if document_type in DOCUMENT_TYPES else "Other"
        status = status if status in STATUSES else "Failed"
        file_hash = self.hash_bytes(file_bytes)
        existing = self.get_by_hash(file_hash)
        if existing:
            raise DuplicateDocumentError(existing)

        safe_original = Path(original_filename).name or "document"
        stored_filename = f"{uuid.uuid4().hex}{Path(safe_original).suffix.lower()}"
        folder = self.storage_root / document_type.lower()
        folder.mkdir(parents=True, exist_ok=True)
        stored_path = folder / stored_filename
        stored_path.write_bytes(file_bytes)
        fields = fields or {}
        record = (
            safe_original, stored_filename, document_type,
            datetime.now().astimezone().isoformat(timespec="seconds"),
            fields.get("Company Name"), fields.get("Invoice Number"), fields.get("Total Amount"),
            str(stored_path.resolve()), text_preview[:2000], file_hash, status,
            json.dumps(fields, ensure_ascii=True),
        )
        try:
            with closing(self._connect()) as connection:
                with connection:
                    document_id = connection.execute(
                        """
                        INSERT INTO documents (
                            original_filename, stored_filename, document_type, upload_date,
                            company, invoice_number, total_amount, file_path, text_preview,
                            file_hash, status, extracted_fields
                        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                        """,
                        record,
                    ).lastrowid
        except sqlite3.IntegrityError:
            stored_path.unlink(missing_ok=True)
            existing = self.get_by_hash(file_hash)
            if existing:
                raise DuplicateDocumentError(existing) from None
            raise
        return self.get_document(document_id)  # type: ignore[return-value]

    def get_document(self, document_id: int) -> dict[str, Any] | None:
        with closing(self._connect()) as connection:
            row = connection.execute("SELECT * FROM documents WHERE id = ?", (document_id,)).fetchone()
        return self._row_to_dict(row) if row else None

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
        query = "SELECT * FROM documents"
        if clauses:
            query += " WHERE " + " AND ".join(clauses)
        query += " ORDER BY upload_date " + ("DESC" if newest_first else "ASC")
        with closing(self._connect()) as connection:
            rows = connection.execute(query, params).fetchall()
        return [self._row_to_dict(row) for row in rows]

    def update_document(self, document_id: int, **changes: Any) -> dict[str, Any] | None:
        allowed = {"document_type", "status", "company", "invoice_number", "total_amount", "text_preview", "extracted_fields"}
        updates = {key: value for key, value in changes.items() if key in allowed}
        if "extracted_fields" in updates and isinstance(updates["extracted_fields"], dict):
            updates["extracted_fields"] = json.dumps(updates["extracted_fields"], ensure_ascii=True)
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
        return document
