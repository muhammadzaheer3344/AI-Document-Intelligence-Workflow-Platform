"""
test_week6_integration.py
--------------------------
Week 6 end-to-end integration tests.

Covers all 14 required test scenarios:
  1.  Clean digital PDF
  2.  Scanned document (OCR)
  3.  Supported document category (Invoice / Resume)
  4.  Document with missing required fields
  5.  Duplicate document / repeated identifier
  6.  Unsupported file type
  7.  Corrupted / unreadable file
  8.  Low-confidence classification
  9.  Financial anomaly (zero / negative amount)
  10. Invalid workflow transition
  11. Mixed-success batch
  12. Application restart (persistence)
  13. AI assistant — answerable question
  14. AI assistant — unanswerable question
"""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from src.anomaly import detect_anomalies, AnomalyFlag
from src.classifier import classify_document
from src.document_repository import DocumentRepository, DuplicateDocumentError, WorkflowTransitionError
from src.extract_text import extract_text
from src.field_extraction import extract_fields, get_missing_fields
from src.preprocess import clean_text, is_usable, normalize_for_classification
from src.rag import DocumentIndex, answer_question, chunk_text
from src.validator import validate_fields
from src.workflow import decide_next_action, process_batch, require_human_review, LOW_CONFIDENCE_THRESHOLD
from app import validate_upload


ROOT = Path(__file__).resolve().parent.parent
SAMPLE_DOCS = ROOT / "sample_docs"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_repo(tmp_dir: Path) -> DocumentRepository:
    return DocumentRepository(tmp_dir / "test.db", tmp_dir / "storage")


def _invoice_fields(overrides: dict | None = None) -> dict:
    base = {
        "Invoice Number": "INV-001",
        "Date": "2026-01-15",
        "Company Name": "Acme Corp",
        "Total Amount": "$500.00",
    }
    if overrides:
        base.update(overrides)
    return base


def _resume_fields(overrides: dict | None = None) -> dict:
    base = {"Name": "Ada Lovelace", "Email": "ada@example.com", "Skills": "Python, SQL"}
    if overrides:
        base.update(overrides)
    return base


# ---------------------------------------------------------------------------
# Test 1 — Clean digital PDF
# ---------------------------------------------------------------------------

class Test01CleanPDF(unittest.TestCase):
    def test_native_invoice_full_pipeline(self):
        pdf_path = SAMPLE_DOCS / "native_invoice.pdf"
        if not pdf_path.exists():
            self.skipTest("native_invoice.pdf not found — run tests/generate_sample_docs.py")
        data = pdf_path.read_bytes()
        result = extract_text(data, "pdf")
        self.assertTrue(result.success)
        self.assertEqual(result.method, "pymupdf")
        self.assertFalse(result.used_ocr)
        cleaned = clean_text(result.text)
        self.assertTrue(is_usable(cleaned))
        normalized = normalize_for_classification(cleaned)
        cls = classify_document(cleaned, normalized)
        self.assertEqual(cls.label, "Invoice")
        fields = extract_fields(cleaned, cls.label)
        self.assertIn("Invoice Number", fields)
        self.assertIn("Total Amount", fields)


# ---------------------------------------------------------------------------
# Test 2 — Scanned document (OCR)
# ---------------------------------------------------------------------------

class Test02ScannedOCR(unittest.TestCase):
    def test_scanned_invoice_uses_ocr(self):
        img_path = SAMPLE_DOCS / "scanned_invoice.png"
        if not img_path.exists():
            self.skipTest("scanned_invoice.png not found")
        data = img_path.read_bytes()
        result = extract_text(data, "png")
        self.assertTrue(result.success)
        self.assertTrue(result.used_ocr)
        self.assertEqual(result.method, "ocr")


# ---------------------------------------------------------------------------
# Test 3 — Supported document categories
# ---------------------------------------------------------------------------

class Test03DocumentCategories(unittest.TestCase):
    def _classify(self, text: str) -> str:
        cleaned = clean_text(text)
        normalized = normalize_for_classification(cleaned)
        return classify_document(cleaned, normalized).label

    def test_invoice_text_classified_as_invoice(self):
        text = "INVOICE\nInvoice Number: INV-42\nBill To: Northwind\nTotal Amount: $1,250.00\nDate: 2026-01-01"
        self.assertEqual(self._classify(text), "Invoice")

    def test_resume_text_classified_as_resume(self):
        text = "RESUME\nAda Lovelace\nEmail: ada@example.com\nSkills: Python, Machine Learning\nExperience: 5 years"
        self.assertEqual(self._classify(text), "Resume")


# ---------------------------------------------------------------------------
# Test 4 — Missing required fields
# ---------------------------------------------------------------------------

class Test04MissingFields(unittest.TestCase):
    def test_missing_invoice_fields_route_to_review(self):
        fields = _invoice_fields({"Invoice Number": "Not Found", "Company Name": "Not Found"})
        decision = decide_next_action("Invoice", fields)
        self.assertEqual(decision.action, "Needs Review")
        self.assertIn("Invoice Number", decision.validation_errors)
        self.assertIn("Company Name", decision.validation_errors)

    def test_missing_resume_skills_route_to_review(self):
        fields = _resume_fields({"Skills": "Not Found"})
        decision = decide_next_action("Resume", fields)
        self.assertEqual(decision.action, "Needs Review")
        self.assertIn("Skills", decision.validation_errors)


# ---------------------------------------------------------------------------
# Test 5 — Duplicate document / repeated identifier
# ---------------------------------------------------------------------------

class Test05Duplicates(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.repo = _make_repo(Path(self.tmp.name))

    def tearDown(self):
        self.tmp.cleanup()

    def test_exact_duplicate_raises(self):
        self.repo.create_document(
            file_bytes=b"same content", original_filename="inv.pdf",
            document_type="Invoice", status="New",
        )
        with self.assertRaises(DuplicateDocumentError):
            self.repo.create_document(
                file_bytes=b"same content", original_filename="inv_copy.pdf",
                document_type="Invoice", status="New",
            )

    def test_repeated_invoice_number_flagged(self):
        self.repo.create_document(
            file_bytes=b"invoice one", original_filename="inv1.pdf",
            document_type="Invoice", status="New",
            fields={"Invoice Number": "INV-999", "Company Name": "Acme",
                    "Date": "2026-01-01", "Total Amount": "$100"},
        )
        flags = detect_anomalies("Invoice", {"Invoice Number": "INV-999"}, self.repo, current_doc_id=99)
        codes = [f.code for f in flags]
        self.assertIn("DUPLICATE_INVOICE_NUMBER", codes)


# ---------------------------------------------------------------------------
# Test 6 — Unsupported file type
# ---------------------------------------------------------------------------

class Test06UnsupportedFileType(unittest.TestCase):
    def test_docx_rejected(self):
        err = validate_upload(b"PK\x03\x04fake docx", "document.docx")
        self.assertIsNotNone(err)
        self.assertIn("docx", err.lower())

    def test_empty_file_rejected(self):
        err = validate_upload(b"", "empty.pdf")
        self.assertIsNotNone(err)

    def test_valid_pdf_accepted(self):
        err = validate_upload(b"%PDF-1.4 content", "invoice.pdf")
        self.assertIsNone(err)


# ---------------------------------------------------------------------------
# Test 7 — Corrupted / unreadable file
# ---------------------------------------------------------------------------

class Test07CorruptFile(unittest.TestCase):
    def test_corrupt_pdf_does_not_crash(self):
        corrupt_path = SAMPLE_DOCS / "corrupt.pdf"
        if not corrupt_path.exists():
            self.skipTest("corrupt.pdf not found")
        data = corrupt_path.read_bytes()
        result = extract_text(data, "pdf")
        # Must not raise; success may be False or text may be empty
        self.assertIsInstance(result.success, bool)
        self.assertIsInstance(result.warnings, list)

    def test_blank_image_returns_no_usable_text(self):
        blank_path = SAMPLE_DOCS / "blank.png"
        if not blank_path.exists():
            self.skipTest("blank.png not found")
        data = blank_path.read_bytes()
        result = extract_text(data, "png")
        cleaned = clean_text(result.text)
        self.assertFalse(is_usable(cleaned))


# ---------------------------------------------------------------------------
# Test 8 — Low-confidence classification
# ---------------------------------------------------------------------------

class Test08LowConfidence(unittest.TestCase):
    def test_low_confidence_routes_to_review(self):
        fields = _resume_fields()
        decision = decide_next_action("Resume", fields, LOW_CONFIDENCE_THRESHOLD - 0.01)
        self.assertEqual(decision.action, "Needs Review")
        self.assertIn("confidence", decision.reason.lower())

    def test_no_confidence_does_not_invent_review(self):
        fields = _resume_fields()
        decision = decide_next_action("Resume", fields, None)
        self.assertEqual(decision.action, "Completed")


# ---------------------------------------------------------------------------
# Test 9 — Financial anomaly
# ---------------------------------------------------------------------------

class Test09FinancialAnomaly(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.repo = _make_repo(Path(self.tmp.name))

    def tearDown(self):
        self.tmp.cleanup()

    def test_zero_amount_flagged(self):
        flags = detect_anomalies("Invoice", {"Invoice Number": "INV-1", "Total Amount": "0"}, self.repo)
        self.assertTrue(any(f.code == "ZERO_AMOUNT" for f in flags))

    def test_negative_amount_flagged_as_error(self):
        flags = detect_anomalies("Invoice", {"Invoice Number": "INV-2", "Total Amount": "-100"}, self.repo)
        error_flags = [f for f in flags if f.code == "NEGATIVE_AMOUNT"]
        self.assertTrue(error_flags)
        self.assertEqual(error_flags[0].severity, "error")

    def test_large_amount_flagged_as_warning(self):
        flags = detect_anomalies("Invoice", {"Invoice Number": "INV-3", "Total Amount": "$2,000,000"}, self.repo)
        self.assertTrue(any(f.code == "UNUSUALLY_LARGE_AMOUNT" for f in flags))

    def test_normal_amount_no_flags(self):
        flags = detect_anomalies("Invoice", _invoice_fields(), self.repo)
        self.assertEqual(flags, [])


# ---------------------------------------------------------------------------
# Test 10 — Invalid workflow transition
# ---------------------------------------------------------------------------

class Test10InvalidTransition(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.repo = _make_repo(Path(self.tmp.name))

    def tearDown(self):
        self.tmp.cleanup()

    def test_new_to_completed_rejected(self):
        doc = self.repo.create_document(
            file_bytes=b"test", original_filename="t.pdf",
            document_type="Invoice", status="New",
        )
        with self.assertRaises(WorkflowTransitionError):
            self.repo.transition_document(doc["id"], "Completed", action="Invalid")

    def test_rejected_to_approved_rejected(self):
        doc = self.repo.create_document(
            file_bytes=b"test2", original_filename="t2.pdf",
            document_type="Invoice", status="New",
        )
        self.repo.transition_document(doc["id"], "Processing", action="Start")
        self.repo.transition_document(doc["id"], "Needs Review", action="Review")
        self.repo.transition_document(doc["id"], "Rejected", action="Reject", reviewer_note="bad")
        with self.assertRaises(WorkflowTransitionError):
            self.repo.transition_document(doc["id"], "Approved", action="Invalid")

    def test_valid_full_path_succeeds(self):
        doc = self.repo.create_document(
            file_bytes=b"test3", original_filename="t3.pdf",
            document_type="Invoice", status="New",
        )
        self.repo.transition_document(doc["id"], "Processing", action="Start")
        self.repo.transition_document(doc["id"], "Needs Review", action="Review")
        self.repo.transition_document(doc["id"], "Approved", action="Approve")
        final = self.repo.transition_document(doc["id"], "Completed", action="Complete")
        self.assertEqual(final["status"], "Completed")


# ---------------------------------------------------------------------------
# Test 11 — Mixed-success batch
# ---------------------------------------------------------------------------

class Test11MixedBatch(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.repo = _make_repo(Path(self.tmp.name))

    def tearDown(self):
        self.tmp.cleanup()

    def test_one_failure_does_not_stop_batch(self):
        docs = [
            self.repo.create_document(
                file_bytes=f"batch-{i}".encode(), original_filename=f"doc{i}.pdf",
                document_type="Invoice", status="New", fields=_invoice_fields(),
            )
            for i in range(3)
        ]
        original = self.repo.transition_document

        def fail_middle(doc_id, new_status, **kw):
            if doc_id == docs[1]["id"] and new_status == "Processing":
                raise OSError("simulated failure")
            return original(doc_id, new_status, **kw)

        with patch.object(self.repo, "transition_document", side_effect=fail_middle):
            results = process_batch(self.repo, [d["id"] for d in docs])

        result_statuses = [r["Result"] for r in results]
        self.assertEqual(result_statuses[0], "Completed")
        self.assertEqual(result_statuses[1], "Failed")
        self.assertEqual(result_statuses[2], "Completed")


# ---------------------------------------------------------------------------
# Test 12 — Application restart / persistence
# ---------------------------------------------------------------------------

class Test12Persistence(unittest.TestCase):
    def test_records_survive_repository_reconnect(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            repo1 = _make_repo(tmp_path)
            doc = repo1.create_document(
                file_bytes=b"persistent", original_filename="persist.pdf",
                document_type="Invoice", status="New",
            )
            doc_id = doc["id"]

            # Simulate restart by creating a new repository instance
            repo2 = _make_repo(tmp_path)
            retrieved = repo2.get_document(doc_id)
            self.assertIsNotNone(retrieved)
            self.assertEqual(retrieved["original_filename"], "persist.pdf")

            events = repo2.list_audit_events(doc_id)
            self.assertTrue(len(events) >= 1)


# ---------------------------------------------------------------------------
# RAG availability check (sentence-transformers may fail on Keras 3 systems)
# ---------------------------------------------------------------------------

def _rag_available() -> bool:
    try:
        from src.rag import get_embed_model
        get_embed_model()
        return True
    except Exception:
        return False


_RAG_SKIP_MSG = (
    "sentence-transformers unavailable (likely Keras 3 conflict — "
    "install tf-keras to resolve). RAG works in the Streamlit app "
    "when the model loads successfully."
)


# ---------------------------------------------------------------------------
# Test 13 — AI assistant: answerable question
# ---------------------------------------------------------------------------

class Test13RAGAnswerable(unittest.TestCase):
    def setUp(self):
        if not _rag_available():
            self.skipTest(_RAG_SKIP_MSG)

    def _build_index(self, docs: list[dict]) -> DocumentIndex:
        idx = DocumentIndex()
        idx.build(docs)
        return idx

    def test_relevant_passage_retrieved(self):
        docs = [
            {"id": 1, "original_filename": "invoice.pdf",
             "text_preview": "Invoice Number INV-42. Company: Northwind. Total Amount: $1,250.00. Date: 2026-01-15."},
        ]
        idx = self._build_index(docs)
        result = answer_question("What is the total amount?", idx)
        self.assertTrue(len(result.sources) > 0)
        self.assertNotIn("do not contain enough information", result.answer.lower()[:50])

    def test_answer_references_source_document(self):
        docs = [
            {"id": 2, "original_filename": "resume.pdf",
             "text_preview": "Ada Lovelace. Email: ada@example.com. Skills: Python, Machine Learning."},
        ]
        idx = self._build_index(docs)
        result = answer_question("What are the candidate's skills?", idx)
        self.assertTrue(len(result.sources) > 0)
        self.assertEqual(result.sources[0]["filename"], "resume.pdf")


# ---------------------------------------------------------------------------
# Test 14 — AI assistant: unanswerable question
# ---------------------------------------------------------------------------

class Test14RAGUnanswerable(unittest.TestCase):
    def test_low_relevance_returns_cannot_answer(self):
        if not _rag_available():
            self.skipTest(_RAG_SKIP_MSG)
        docs = [
            {"id": 3, "original_filename": "invoice.pdf",
             "text_preview": "Invoice Number INV-1. Total Amount: $100. Date: 2026-01-01."},
        ]
        idx = DocumentIndex()
        idx.build(docs)
        result = answer_question("What is the weather forecast for tomorrow?", idx)
        self.assertIn("do not contain enough information", result.answer.lower())

    def test_empty_index_returns_no_documents_message(self):
        idx = DocumentIndex()
        result = answer_question("Any question", idx)
        self.assertIn("no documents", result.answer.lower())


if __name__ == "__main__":
    unittest.main(verbosity=2)
