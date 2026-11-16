from typing import Literal

from pydantic import BaseModel

Action = Literal["allow", "flag", "block"]


class GuardrailFinding(BaseModel):
    guardrail: str
    action: Literal["flag", "block"]
    code: str
    field: str
    detail: str  # never contains the checked text


class GuardrailResult(BaseModel):
    guardrail: str
    findings: list[GuardrailFinding] = []

    @property
    def action(self) -> Action:
        if any(f.action == "block" for f in self.findings):
            return "block"
        return "flag" if self.findings else "allow"

class GuardrailBlocked(Exception):
    """A request was stopped by a guardrail. `message` is safe to show the user."""

    def __init__(self, message: str, findings: list[GuardrailFinding]) -> None:
        super().__init__(message)
        self.message = message
        self.findings = findings