"""Pure rule-based workflow decisions, independent of the Streamlit UI."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from src.validator import validate_fields

# Classifications below this value are routed for human review only when the
# classifier supplied a real confidence score.
LOW_CONFIDENCE_THRESHOLD = 0.55


@dataclass(frozen=True)
class WorkflowDecision:
    action: str
    reason: str
    validation_errors: dict[str, str]


def decide_next_action(
    document_type: str,
    fields: dict,
    confidence: float | None = None,
    recorded_validation_errors: dict[str, str] | None = None,
) -> WorkflowDecision:
    errors = validate_fields(document_type, fields)
    for field_name, reason in (recorded_validation_errors or {}).items():
        errors.setdefault(field_name, reason)
    if errors:
        details = ", ".join(f"{name}: {reason}" for name, reason in errors.items())
        return WorkflowDecision("Needs Review", f"Validation failed: {details}", errors)
    if confidence is not None and confidence < LOW_CONFIDENCE_THRESHOLD:
        return WorkflowDecision(
            "Needs Review",
            f"Classification confidence {confidence:.0%} is below the {LOW_CONFIDENCE_THRESHOLD:.0%} review threshold.",
            errors,
        )
    return WorkflowDecision("Completed", "Required fields passed validation and workflow rules.", errors)


def process_batch(repository: Any, document_ids: list[int]) -> list[dict[str, Any]]:
    """Process each stored document independently and return one result per ID."""
    results: list[dict[str, Any]] = []
    for document_id in document_ids:
        filename = f"Document #{document_id}"
        try:
            document = repository.get_document(document_id)
            if document is None:
                raise KeyError(f"Document {document_id} does not exist.")
            filename = document["original_filename"]
            current = repository.transition_document(
                document_id, "Processing", action="Batch processing started",
            )
            decision = decide_next_action(
                current["document_type"], current.get("extracted_fields") or {},
                current.get("confidence"), current.get("validation_errors") or {},
            )
            final = repository.transition_document(
                document_id, decision.action, action="Batch workflow decision",
                reason=decision.reason, validation_errors=decision.validation_errors,
            )
            results.append({"Document": filename, "Result": final["status"], "Reason": decision.reason})
        except Exception as exc:  # Continue even if one row cannot be processed.
            try:
                latest = repository.get_document(document_id)
                if latest and latest["status"] == "Processing":
                    repository.transition_document(
                        document_id, "Failed", action="Batch processing failed",
                        reason=f"{type(exc).__name__}: {exc}",
                    )
                elif latest:
                    repository.record_workflow_event(
                        document_id, "Batch processing failed", f"{type(exc).__name__}: {exc}",
                    )
            except Exception:
                pass
            results.append({"Document": filename, "Result": "Failed", "Reason": str(exc)})
    return results