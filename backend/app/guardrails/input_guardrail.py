import re

from pydantic import BaseModel

from app.guardrails.pii_guardrail import mask_pii
from app.guardrails.prompt_injection import detect_injection, has_hidden_chars
from app.guardrails.result import GuardrailFinding, GuardrailResult

NAME = "input"
DOCUMENT_ID = re.compile(r"[A-Za-z0-9_-]{1,100}")
CONTROL = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")

GUARDRAIL_ERROR_MESSAGE = "The request could not be checked, so it was not processed."
INJECTION_MESSAGE = (
    "This question looks like an attempt to change how the assistant behaves, so it was not processed. "
    "Rephrase it as a question about the contract."
)
USER_MESSAGES = {
    "invalid_document_id": "The document reference is not valid.",
    "control_characters": "The question contains characters that cannot be processed.",
    "guardrail_error": GUARDRAIL_ERROR_MESSAGE,
}


def _finding(code: str, field: str, action: str, detail: str) -> GuardrailFinding:
    return GuardrailFinding(guardrail=NAME, action=action, code=code, field=field, detail=detail)


class InputCheck(BaseModel):
    result: GuardrailResult
    masked_question: str | None = None

    @property
    def action(self) -> str:
        return self.result.action

    @property
    def user_message(self) -> str | None:
        for f in self.result.findings:
            if f.action == "block":
                return USER_MESSAGES.get(f.code, INJECTION_MESSAGE)
        return None


def check_request(question: str | None, document_id: str | None) -> InputCheck:
    findings: list[GuardrailFinding] = []

    if document_id is not None and not DOCUMENT_ID.fullmatch(document_id):
        findings.append(_finding("invalid_document_id", "document_id", "block", "The document reference has an invalid format"))

    masked: str | None = None
    if question is not None:
        masked = question
        if CONTROL.search(question):
            findings.append(_finding("control_characters", "question", "block", "Contains control characters"))
        for hit in detect_injection(question):
            action = "block" if hit.severity == "high" else "flag"
            findings.append(_finding(hit.code, "question", action, f"Matched the '{hit.code}' pattern group"))
        if has_hidden_chars(question):
            findings.append(_finding("hidden_characters", "question", "flag", "Contains invisible characters"))
        masked, counts = mask_pii(question)
        if counts:
            summary = ", ".join(f"{kind} x{n}" for kind, n in sorted(counts.items()))
            findings.append(_finding("pii_masked", "question", "flag", f"Masked in the stored copy: {summary}"))

    return InputCheck(result=GuardrailResult(guardrail=NAME, findings=findings), masked_question=masked)