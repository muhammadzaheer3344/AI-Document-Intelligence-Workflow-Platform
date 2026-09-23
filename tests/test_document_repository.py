import tempfile
import unittest
from pathlib import Path

from src.document_repository import DocumentRepository, DuplicateDocumentError


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
        self.assertEqual(self.repository.update_document(first["id"], status="Processed")["status"], "Processed")
        self.assertTrue(self.repository.delete_document(first["id"]))
        self.assertIsNone(self.repository.get_document(first["id"]))


if __name__ == "__main__":
    unittest.main()
