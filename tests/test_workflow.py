import unittest

from src.workflow import LOW_CONFIDENCE_THRESHOLD, decide_next_action, get_workflow_metrics
from src.validator import validate_fields


class WorkflowTests(unittest.TestCase):
    def test_valid_invoice_completes_without_confidence(self):
        fields = {
            "Invoice Number": "INV-42", "Date": "2026-08-19",
            "Company Name": "Northwind", "Total Amount": "USD 1,250.00",
        }
        decision = decide_next_action("Invoice", fields)
        self.assertEqual(decision.action, "Completed")
        self.assertEqual(decision.validation_errors, {})

    def test_missing_invoice_fields_are_reported_by_name(self):
        errors = validate_fields("Invoice", {"Invoice Number": "Not Found"})
        self.assertEqual(set(errors), {"Invoice Number", "Date", "Company Name", "Total Amount"})

    def test_invalid_formats_are_reported(self):
        errors = validate_fields("Invoice", {
            "Invoice Number": "I-1", "Date": "2026-99-99",
            "Company Name": "Northwind", "Total Amount": "many dollars",
        })
        self.assertEqual(set(errors), {"Date", "Total Amount"})

    def test_invalid_resume_email_requires_review(self):
        decision = decide_next_action("Resume", {"Name": "Ada Lovelace", "Email": "invalid", "Skills": "Python"})
        self.assertEqual(decision.action, "Needs Review")
        self.assertIn("Email", decision.validation_errors)

    def test_valid_resume_completes(self):
        decision = decide_next_action("Resume", {"Name": "Ada Lovelace", "Email": "ada@example.com", "Skills": "Python"})
        self.assertEqual(decision.action, "Completed")

    def test_low_confidence_routes_to_review(self):
        fields = {"Name": "Ada Lovelace", "Email": "ada@example.com", "Skills": "Python"}
        decision = decide_next_action("Resume", fields, LOW_CONFIDENCE_THRESHOLD - 0.01)
        self.assertEqual(decision.action, "Needs Review")
        self.assertIn("confidence", decision.reason)

    def test_absent_confidence_is_not_invented_or_used(self):
        fields = {"Name": "Ada Lovelace", "Email": "ada@example.com", "Skills": "Python"}
        self.assertEqual(decide_next_action("Resume", fields, None).action, "Completed")

    def test_recorded_readability_failure_routes_other_document_to_review(self):
        decision = decide_next_action(
            "Other", {}, None, {"Readable text": "OCR returned no usable text."},
        )
        self.assertEqual(decision.action, "Needs Review")
        self.assertIn("Readable text", decision.validation_errors)

    def test_twelve_document_workflow_matrix(self):
        valid_invoice = {
            "Invoice Number": "INV-42", "Date": "2026-08-19",
            "Company Name": "Northwind", "Total Amount": "$125.00",
        }
        valid_resume = {"Name": "Ada Lovelace", "Email": "ada@example.com", "Skills": "Python, SQL"}
        scenarios = [
            ("native invoice", "Invoice", valid_invoice, 0.98, {}, "Completed"),
            ("scanned OCR invoice", "Invoice", {**valid_invoice, "Date": "28-08-2025"}, 0.91, {}, "Completed"),
            ("invoice with PKR amount", "Invoice", {**valid_invoice, "Total Amount": "PKR 3,000"}, None, {}, "Completed"),
            ("invoice with date slash format", "Invoice", {**valid_invoice, "Date": "08/19/2026"}, 0.82, {}, "Completed"),
            ("invoice missing number", "Invoice", {**valid_invoice, "Invoice Number": "Not Found"}, 0.95, {}, "Needs Review"),
            ("invoice invalid date and amount", "Invoice", {**valid_invoice, "Date": "2026-99-99", "Total Amount": "many"}, None, {}, "Needs Review"),
            ("complete resume", "Resume", valid_resume, 0.96, {}, "Completed"),
            ("resume without model confidence", "Resume", valid_resume, None, {}, "Completed"),
            ("resume with invalid email", "Resume", {**valid_resume, "Email": "bad-address"}, 0.9, {}, "Needs Review"),
            ("resume missing skills", "Resume", {**valid_resume, "Skills": "Not Found"}, None, {}, "Needs Review"),
            ("low-confidence resume", "Resume", valid_resume, 0.4, {}, "Needs Review"),
            ("unreadable scanned document", "Other", {}, None,
             {"Readable text": "OCR returned no usable text."}, "Needs Review"),
        ]
        actual = [
            decide_next_action(doc_type, fields, confidence, recorded_errors).action
            for _, doc_type, fields, confidence, recorded_errors, _ in scenarios
        ]
        expected = [expected for _, _, _, _, _, expected in scenarios]
        self.assertEqual(actual, expected, [name for name, *_ in scenarios])

    def test_metrics_fallback_supports_week_four_repository(self):
        class WeekFourRepository:
            def list_documents(self):
                return [
                    {"status": "Completed", "document_type": "Invoice"},
                    {"status": "Needs Review", "document_type": "Invoice"},
                    {"status": "Completed", "document_type": "Resume"},
                ]

        self.assertEqual(get_workflow_metrics(WeekFourRepository()), {
            "total": 3,
            "statuses": {"Completed": 2, "Needs Review": 1},
            "by_type": {"Invoice": 2, "Resume": 1},
        })

    def test_metrics_use_repository_aggregate_when_available(self):
        class CurrentRepository:
            def get_metrics(self):
                return {"total": 0, "statuses": {}, "by_type": {}}

            def list_documents(self):
                raise AssertionError("fallback should not be used")

        self.assertEqual(get_workflow_metrics(CurrentRepository()), {
            "total": 0, "statuses": {}, "by_type": {},
        })


if __name__ == "__main__":
    unittest.main()