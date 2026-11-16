from typing import Literal

from pydantic import BaseModel, Field, model_validator

from app.agents.contract.schemas import ClauseType, EvidenceRef

FactField = Literal[
    "effective_date", "term_or_expiry", "termination_notice", "payment_terms", "governing_law"
]
CheckType = Literal[
    "field_present", "duration_at_least", "duration_at_most", "value_in_list", "clause_present"
]
Severity = Literal["low", "medium", "high"]
Status = Literal["compliant", "non_compliant", "not_found", "needs_review"]


class DurationValue(BaseModel):
    amount: float = Field(gt=0)
    unit: Literal["days", "weeks", "months", "years"]


class ComplianceRule(BaseModel):
    id: str
    name: str
    source: str = "unspecified"
    check: CheckType
    field: FactField | None = None
    clause_type: ClauseType | None = None
    value: DurationValue | list[str] | None = None
    severity: Severity

    @model_validator(mode="after")
    def _check_shape(self):
        if self.check == "clause_present":
            if self.clause_type is None:
                raise ValueError("clause_present needs clause_type")
        elif self.field is None:
            raise ValueError(f"{self.check} needs field")

        if self.check in ("duration_at_least", "duration_at_most") and not isinstance(
            self.value, DurationValue
        ):
            raise ValueError(f"{self.check} needs value: {{amount, unit}}")
        if self.check == "value_in_list" and not (isinstance(self.value, list) and self.value):
            raise ValueError("value_in_list needs a non-empty list as value")
        return self


class ComplianceFinding(BaseModel):
    rule_id: str
    rule_name: str
    source: str
    severity: Severity
    status: Status
    detail: str
    evidence: list[EvidenceRef] = []