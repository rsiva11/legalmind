import logging
import re
from collections import Counter
from dataclasses import dataclass

from app.agents.base_agent import BaseAgent
from app.agents.contract.schemas import ContractAnalysis, EvidenceRef
from app.agents.risk.schemas import CandidateRisk
from app.agents.verification.prompts import build_messages
from app.agents.verification.schemas import (
    ItemVerification,
    SemanticDraft,
    SemanticVerdict,
    VerificationInput,
    VerificationIssue,
    VerificationReport,
)
from app.business_logic.compliance_rules import _WORDS, evaluate, load_rules
from app.business_logic.risk_rules import evaluate_risks, load_risk_rules
from app.business_logic.scoring_engine import score_risks
from app.utils.text_utils import norm

logger = logging.getLogger(__name__)

FACT_FIELDS = ("effective_date", "term_or_expiry", "termination_notice", "payment_terms", "governing_law")
MAX_OUTPUT_TOKENS = 3000
_RANK = {"supported": 0, "unclear": 1, "unsupported": 2}

_DIGITS = re.compile(r"\d+(?:,\d{3})*(?:\.\d+)?")
_WORD_NUM = re.compile(r"\b(" + "|".join(_WORDS) + r")\b", re.I)


def numbers_in(text: str) -> set[float]:
    found = {float(m.group(0).replace(",", "")) for m in _DIGITS.finditer(text)}
    found |= {float(_WORDS[m.group(1).lower()]) for m in _WORD_NUM.finditer(text)}
    return found


@dataclass
class Claim:
    item_id: str
    kind: str
    label: str
    claim: str
    refs: list[EvidenceRef]
    value: str | None = None  # numbers in it must appear in the quotes
    name: str | None = None  # must appear in the quotes
    field: str | None = None  # the fact field, for mapping to rules
    clause_type: str | None = None  # for mapping clause_present rules


def build_claims(analysis: ContractAnalysis, candidates: list[CandidateRisk]) -> list[Claim]:
    claims: list[Claim] = []
    for field in FACT_FIELDS:
        fact = getattr(analysis, field)
        if fact is not None:
            claims.append(Claim(
                f"fact:{field}", "fact", f"{field}: {fact.value}",
                f"The contract states that {field.replace('_', ' ')} is: {fact.value}",
                fact.evidence, value=fact.value, field=field,
            ))
    for i, p in enumerate(analysis.parties):
        role = f" (role: {p.role})" if p.role else ""
        claims.append(Claim(
            f"party:{i}", "party", f"party: {p.name}",
            f"{p.name}{role} is a party to the contract", p.evidence, name=p.name,
        ))
    for i, c in enumerate(analysis.clauses):
        claims.append(Claim(
            f"clause:{i}", "clause", f"{c.clause_type} clause",
            f"The contract contains a {c.clause_type} clause: {c.summary}", c.evidence,
            clause_type=c.clause_type,
        ))
    for i, o in enumerate(analysis.obligations):
        deadline = f", deadline: {o.deadline}" if o.deadline else ""
        claims.append(Claim(
            f"obligation:{i}", "obligation", f"obligation of {o.obligated_party}",
            f"{o.obligated_party} must {o.action}{deadline}", o.evidence, value=o.deadline,
        ))
    for i, c in enumerate(candidates):
        claims.append(Claim(
            f"candidate:{i}", "candidate", f"candidate risk: {c.title}",
            f"Risk: {c.title}. {c.description}", c.evidence,
        ))
    return claims


def code_problems(c: Claim, lookup: dict[str, str]) -> list[str]:
    problems: list[str] = []
    for r in c.refs:
        if r.label not in lookup:
            problems.append(f"evidence label {r.label} is not in the retrieved evidence")
        elif norm(r.quote) not in lookup[r.label]:
            problems.append(f"quote for {r.label} was not found in that evidence")

    quotes = " ".join(r.quote for r in c.refs)
    if c.value:
        missing = numbers_in(c.value) - numbers_in(quotes)
        if missing:
            problems.append(
                "number(s) in the value are not in its quote: " + ", ".join(f"{n:g}" for n in sorted(missing))
            )
    if c.name and norm(c.name) not in norm(quotes):
        problems.append("name is not in its quote")
    return problems


def _diff(expected: dict, actual: dict) -> list[str]:
    return sorted(k for k in expected.keys() | actual.keys() if expected.get(k) != actual.get(k))


def _recompute(data: VerificationInput) -> tuple[list[VerificationIssue], list, list]:
    """Re-run the deterministic engines and compare with the reports we were given."""
    issues: list[VerificationIssue] = []
    c_rules: list = []
    r_rules: list = []

    if data.compliance is not None:
        c_rules = data.compliance_rules if data.compliance_rules is not None else load_rules()
        expected = {f.rule_id: f.status for f in evaluate(data.analysis, c_rules)}
        actual = {f.rule_id: f.status for f in data.compliance.findings}
        bad = _diff(expected, actual)
        if bad:
            issues.append(VerificationIssue(
                code="compliance_mismatch", severity="error",
                detail=f"Compliance statuses differ from a fresh evaluation for: {', '.join(bad)}",
            ))

    if data.risk is not None:
        r_rules = data.risk_rules if data.risk_rules is not None else load_risk_rules()
        findings = evaluate_risks(data.analysis, r_rules)
        expected = {f.rule_id: f.outcome for f in findings}
        actual = {f.rule_id: f.outcome for f in data.risk.findings}
        bad = _diff(expected, actual)
        if score_risks(findings) != data.risk.score:
            bad.append("score")
        if bad:
            issues.append(VerificationIssue(
                code="risk_mismatch", severity="error",
                detail=f"Risk results differ from a fresh evaluation for: {', '.join(bad)}",
            ))
    return issues, c_rules, r_rules


def _warning(code: str, detail: str) -> VerificationIssue:
    return VerificationIssue(code=code, severity="warning", detail=detail)


def _coverage(data: VerificationInput, claims: list[Claim]) -> list[VerificationIssue]:
    out: list[VerificationIssue] = []
    if not data.context.evidence:
        out.append(_warning("no_evidence", "No evidence was retrieved, so nothing can be confirmed"))
    if data.context.truncated or data.analysis.context_truncated:
        out.append(_warning("context_truncated", "Some retrieved evidence was cut to fit the size budget"))
    dropped = data.analysis.dropped_unverified + (data.risk.candidates_dropped if data.risk else 0)
    if dropped:
        out.append(_warning("items_dropped_earlier", f"{dropped} item(s) were already dropped earlier because their quotes were not found"))
    if data.compliance is not None:
        unresolved = {s: n for s, n in data.compliance.status_counts.items() if s in ("not_found", "needs_review") and n}
        if unresolved:
            out.append(_warning("compliance_unresolved", f"Compliance rules not fully resolved: {unresolved}"))
    if data.risk is not None and not data.risk.score.complete:
        out.append(_warning("risk_coverage_incomplete", f"Only {data.risk.score.coverage:.0%} of risk rules could be assessed, so the level is a lower bound"))
    if not claims:
        out.append(_warning("nothing_to_verify", "The analysis contains no extracted items"))
    return out


class VerificationAgent(BaseAgent[VerificationInput, "VerificationReport"]):
    name = "verification"
    description = "Quality gate: rechecks quotes, values, engine results and evidence support"
    input_schema = VerificationInput
    output_schema = VerificationReport

    async def execute(self, data: VerificationInput) -> VerificationReport:
        lookup = {e.label: norm(e.text) for e in data.context.evidence}
        candidates = data.risk.candidates if data.risk else []
        claims = build_claims(data.analysis, candidates)
        problems = {c.item_id: code_problems(c, lookup) for c in claims}

        issues, c_rules, r_rules = _recompute(data)
        issues += _coverage(data, claims)

        # One batched semantic call, only for items that passed every code check
        semantic: dict[str, SemanticVerdict] = {}
        semantic_checked = False
        to_check = [c for c in claims if not problems[c.item_id]]
        if data.include_semantic and to_check:
            try:
                draft = await self.ask_structured(
                    build_messages([
                        {"item_id": c.item_id, "kind": c.kind, "claim": c.claim, "quotes": [r.quote for r in c.refs]}
                        for c in to_check
                    ]),
                    SemanticDraft,
                    max_tokens=MAX_OUTPUT_TOKENS,
                )
                allowed = {c.item_id for c in to_check}
                for v in draft.verdicts:
                    # Keep the WORST verdict if the model repeats an item: downgrade only
                    if v.item_id in allowed and (
                        v.item_id not in semantic or _RANK[v.verdict] > _RANK[semantic[v.item_id].verdict]
                    ):
                        semantic[v.item_id] = v
                semantic_checked = True
            except Exception as exc:  # one narrow call; recorded as an issue, never hidden
                logger.error("Verification semantic check failed: %s", type(exc).__name__)
                issues.append(_warning(
                    "semantic_check_failed",
                    f"The semantic check could not run ({type(exc).__name__}); only code checks were applied",
                ))
        elif not data.include_semantic and claims:
            issues.append(_warning("semantic_check_skipped", "Semantic check was turned off; only code checks were applied"))

        items: list[ItemVerification] = []
        for c in claims:
            reasons = problems[c.item_id]
            if reasons:
                verdict = "failed_check"
            elif semantic_checked:
                s = semantic.get(c.item_id)
                if s is None:
                    verdict, reasons = "unclear", ["not assessed by the semantic check"]
                elif s.verdict == "supported":
                    verdict = "verified"
                else:
                    verdict, reasons = s.verdict, [s.reason]
            else:
                verdict = "verified"
            items.append(ItemVerification(item_id=c.item_id, kind=c.kind, label=c.label, verdict=verdict, reasons=reasons))

        withheld = [i.item_id for i in items if i.verdict != "verified"]
        bad_fields = {c.field for c, i in zip(claims, items) if c.field and i.verdict != "verified"}
        verified_types = {c.clause_type for c, i in zip(claims, items) if c.clause_type and i.verdict == "verified"}
        # A clause type is withheld only if no verified clause of that type remains
        bad_types = {
            c.clause_type for c, i in zip(claims, items) if c.clause_type and i.verdict != "verified"
        } - verified_types

        def hit(rule) -> bool:
            return (rule.field is not None and rule.field in bad_fields) or (
                rule.clause_type is not None and rule.clause_type in bad_types
            )

        withheld_findings = (
            [f"compliance:{r.id}" for r in c_rules if hit(r)]
            + [f"risk:{r.id}" for r in r_rules if hit(r)]
        )

        counts = dict(Counter(i.verdict for i in items))
        if any(i.severity == "error" for i in issues) or (items and counts.get("verified", 0) == 0):
            result = "fail"
        elif withheld or issues:
            result = "pass_with_warnings"
        else:
            result = "pass"

        logger.info("Verification result=%s items=%d withheld=%d issues=%d", result, len(items), len(withheld), len(issues))
        return VerificationReport(
            document_id=data.document_id,
            result=result,
            items=items,
            withheld_item_ids=withheld,
            withheld_finding_ids=withheld_findings,
            issues=issues,
            semantic_checked=semantic_checked,
            counts=counts,
        )