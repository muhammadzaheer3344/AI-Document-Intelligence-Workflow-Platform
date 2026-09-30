import tempfile
import unittest
from unittest.mock import patch
from pathlib import Path

from src.document_repository import DocumentRepository, DuplicateDocumentError, WorkflowTransitionError
from src.workflow import process_batch


class DocumentRepositoryTests(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        root = Path(self.temp_dir.name)
        self.repository = DocumentRepository(root / "documents.db", root / "storage")

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_create_stores_safe_file_and_metadata(self):
        document = self.repository.create_document(
            file_bytes=b"invoice contents",
            original_filename="../../invoice.pdf",
            document_type="Invoice",
            status="Processed",
            fields={"Invoice Number": "INV-100", "Company Name": "Acme"},
            text_preview="Acme INV-100",
        )
        stored_path = Path(document["file_path"])
        self.assertEqual(document["original_filename"], "invoice.pdf")
        self.assertEqual(stored_path.parent.name, "invoice")
        self.assertTrue(stored_path.is_file())
        self.assertEqual(document["invoice_number"], "INV-100")

    def test_duplicate_hash_returns_existing_record(self):
        first = self.repository.create_document(
            file_bytes=b"same file", original_filename="one.png", document_type="Other", status="Processed"
        )
        with self.assertRaises(DuplicateDocumentError) as context:
            self.repository.create_document(
                file_bytes=b"same file", original_filename="two.png", document_type="Other", status="Processed"
            )
        self.assertEqual(context.exception.document["id"], first["id"])
        self.assertEqual(len(self.repository.list_documents()), 1)

    def test_search_filters_update_and_delete(self):
        first = self.repository.create_document(
            file_bytes=b"one", original_filename="invoice.pdf", document_type="Invoice",
            status="Needs Review", fields={"Company Name": "Northwind"}, text_preview="Northwind total",
        )
        self.repository.create_document(
            file_bytes=b"two", original_filename="resume.pdf", document_type="Resume", status="Processed",
            text_preview="Python experience",
        )
        self.assertEqual(len(self.repository.list_documents(search="Northwind")), 1)
        self.assertEqual(len(self.repository.list_documents(document_type="Invoice", status="Needs Review")), 1)
        with self.assertRaises(WorkflowTransitionError):
            self.repository.transition_document(first["id"], "Completed", action="Invalid completion")
        approved = self.repository.transition_document(first["id"], "Approved", action="Approved", reviewer_note="Verified")
        completed = self.repository.transition_document(approved["id"], "Completed", action="Completed")
        self.assertEqual(completed["status"], "Completed")
        history = self.repository.list_audit_events(first["id"])
        self.assertEqual([event["action"] for event in history[:2]], ["Completed", "Approved"])
        self.assertEqual(history[1]["reason"], "Verified")
        self.assertTrue(self.repository.delete_document(first["id"]))
        self.assertIsNone(self.repository.get_document(first["id"]))

    def test_week_four_status_is_migrated_and_audited(self):
        document = self.repository.create_document(
            file_bytes=b"legacy invoice", original_filename="legacy.pdf",
            document_type="Invoice", status="Completed",
        )
        connection = self.repository._connect()
        with connection:
            connection.execute("UPDATE documents SET status = 'Processed' WHERE id = ?", (document["id"],))
            connection.execute("DELETE FROM audit_events WHERE document_id = ?", (document["id"],))
        connection.close()

        migrated_repository = DocumentRepository(self.repository.db_path, self.repository.storage_root)
        migrated = migrated_repository.get_document(document["id"])
        self.assertEqual(migrated["status"], "Completed")
        event = migrated_repository.list_audit_events(document["id"])[0]
        self.assertEqual(event["action"], "Week 5 migration")
        self.assertEqual(event["previous_status"], "Processed")

    def test_mixed_batch_keeps_processing_after_one_document_fails(self):
        valid_fields = {
            "Invoice Number": "INV-9", "Date": "2026-09-01",
            "Company Name": "Acme", "Total Amount": "$99.00",
        }
        documents = [
            self.repository.create_document(
                file_bytes=f"batch-{index}".encode(), original_filename=f"invoice-{index}.pdf",
                document_type="Invoice", status="New", fields=valid_fields,
            )
            for index in range(3)
        ]
        original_transition = self.repository.transition_document

        def fail_second_start(document_id, new_status, **kwargs):
            if document_id == documents[1]["id"] and new_status == "Processing":
                raise OSError("simulated per-document failure")
            return original_transition(document_id, new_status, **kwargs)

        with patch.object(self.repository, "transition_document", side_effect=fail_second_start):
            results = process_batch(self.repository, [document["id"] for document in documents])

        self.assertEqual([result["Result"] for result in results], ["Completed", "Failed", "Completed"])
        self.assertEqual(self.repository.get_document(documents[2]["id"])["status"], "Completed")
        self.assertEqual(self.repository.list_audit_events(documents[1]["id"])[0]["action"], "Batch processing failed")


if __name__ == "__main__":
    unittest.main()
