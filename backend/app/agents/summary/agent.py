import logging
import re

from app.agents.base_agent import BaseAgent
from app.agents.contract.schemas import ContractAnalysis, EvidenceRef
from app.agents.summary.prompts import build_messages
from app.agents.summary.schemas import (
    OverviewDraft,
    RiskSummary,
    SummaryInput,
    SummaryLine,
    SummaryReport,
)
from app.agents.verification.agent import numbers_in  # belongs in utils; moved when a third user appears
from app.agents.verification.schemas import VerificationReport
from app.business_logic.scoring_engine import score_risks
from app.knowledge.citation_manager import extract_citations
from app.schemas.document import BuiltContext

logger = logging.getLogger(__name__)

DISCLAIMER = (
    "This is an automated analysis of the uploaded document, not legal advice. "
    "Findings depend on the text that was retrieved and should be reviewed by a qualified person."
)
UNVERIFIED_OVERVIEW = (
    "This analysis could not be verified, so no findings are shown. "
    "Review the reasons below or have the document reviewed manually."
)
FACT_LABELS = {
    "effective_date": "Effective date",
    "term_or_expiry": "Term or expiry",
    "termination_notice": "Termination notice",
    "payment_terms": "Payment terms",
    "governing_law": "Governing law",
}
MAX_OUTPUT_TOKENS = 1500
MAX_FEED_LINES = 25
# Already covered by the lines and the risk note, so not repeated as gaps
REDUNDANT_ISSUES = {"compliance_unresolved", "risk_coverage_incomplete"}

_CITE = re.compile(r"\[E\d+\]")
# Stopgap until the output guardrail exists (Phase 8)
_BANNED = re.compile(
    r"\b(illegal|unlawful|unenforceable|enforceable|void|voidable|invalid|"
    r"you should (?:sign|not sign)|do not sign)\b",
    re.I,
)


def _cite(refs: list[EvidenceRef], pages: dict[str, int]) -> list[str]:
    out: list[str] = []
    for r in refs:
        s = r.label + (f" p.{pages[r.label]}" if r.label in pages else "")
        if s not in out:
            out.append(s)
    return out


def _line(text: str, refs: list[EvidenceRef], pages: dict[str, int]) -> SummaryLine:
    return SummaryLine(text=text, citations=_cite(refs, pages))


def _feed(line: SummaryLine) -> str:
    return line.text + " " + "".join(f"[{c.split()[0]}]" for c in line.citations)


def _describe(item_id: str, analysis: ContractAnalysis) -> str:
    """Name a withheld item WITHOUT showing its unverified value."""
    kind, _, key = item_id.partition(":")
    if kind == "fact":
        return key.replace("_", " ")
    if kind == "party":
        return "a party entry"
    if kind == "clause":
        try:
            return f"a {analysis.clauses[int(key)].clause_type.replace('_', ' ')} clause entry"
        except (ValueError, IndexError):
            return "a clause entry"
    if kind == "obligation":
        return "an obligation entry"
    return "a suggested risk"


def _item_gaps(v: VerificationReport, analysis: ContractAnalysis) -> list[str]:
    bad = [i for i in v.items if i.verdict != "verified"]
    if not bad:
        return []
    parts = [f"{_describe(i.item_id, analysis)} ({i.verdict.replace('_', ' ')})" for i in bad]
    return ["Withheld because they could not be verified: " + "; ".join(parts)]


def _key_facts(a: ContractAnalysis, verified: set[str], pages: dict[str, int]) -> list[SummaryLine]:
    lines: list[SummaryLine] = []
    for field, label in FACT_LABELS.items():
        fact = getattr(a, field)
        if fact is not None and f"fact:{field}" in verified:
            lines.append(_line(f"{label}: {fact.value}", fact.evidence, pages))
    for i, p in enumerate(a.parties):
        if f"party:{i}" in verified:
            role = f" ({p.role})" if p.role else ""
            lines.append(_line(f"Party: {p.name}{role}", p.evidence, pages))
    for i, c in enumerate(a.clauses):
        if f"clause:{i}" in verified:
            lines.append(_line(f"{c.clause_type.replace('_', ' ').title()} clause: {c.summary}", c.evidence, pages))
    for i, o in enumerate(a.obligations):
        if f"obligation:{i}" in verified:
            deadline = f" (deadline: {o.deadline})" if o.deadline else ""
            lines.append(_line(f"Obligation of {o.obligated_party}: {o.action}{deadline}", o.evidence, pages))
    return lines


def _compliance(report, withheld: set[str], pages: dict[str, int]):
    if report is None:
        return [], []
    lines: list[SummaryLine] = []
    not_found: list[str] = []
    review: list[str] = []
    held: list[str] = []
    for f in report.findings:
        if f"compliance:{f.rule_id}" in withheld:
            held.append(f.rule_name)
        elif f.status in ("compliant", "non_compliant"):
            text = f"{'Met' if f.status == 'compliant' else 'Not met'}: {f.rule_name}. {f.detail}."
            if f.status == "non_compliant" and f.suggested_action:
                text += f" Suggested: {f.suggested_action}"
            lines.append(_line(text, f.evidence, pages))
        elif f.status == "not_found":
            not_found.append(f.rule_name)
        else:
            review.append(f"{f.rule_name}: {f.detail}")
    gaps: list[str] = []
    if not_found:
        gaps.append(
            "Not found in the retrieved text (this does not mean the contract lacks it; "
            "a targeted search or manual check is needed): " + "; ".join(not_found)
        )
    if review:
        gaps.append("Needs a person to check: " + " | ".join(review))
    if held:
        gaps.append("Compliance rules withheld because the fact behind them could not be verified: " + "; ".join(held))
    return lines, gaps


def _risk(report, withheld: set[str], pages: dict[str, int]):
    if report is None:
        return None, []
    # A withheld finding must not add points: treat it as not assessed, then rescore
    adjusted = [
        f.model_copy(update={"outcome": "unassessed", "points": 0})
        if f"risk:{f.rule_id}" in withheld
        else f
        for f in report.findings
    ]
    score = score_risks(adjusted)
    lines: list[SummaryLine] = []
    unassessed: list[str] = []
    review: list[str] = []
    for f in adjusted:
        if f.outcome == "triggered":
            text = f"{f.rule_name} ({f.category}, {f.severity} severity): {f.detail}."
            if f.suggested_action:
                text += f" Suggested: {f.suggested_action}"
            lines.append(_line(text, f.evidence, pages))
        elif f.outcome == "unassessed":
            unassessed.append(f.rule_name)
        elif f.outcome == "needs_review":
            review.append(f"{f.rule_name}: {f.detail}")

    if score.level == "not_assessed":
        note = "No risk rule could be assessed, so no level is given."
    elif score.complete:
        note = f"Based on all {score.rules_total} risk rules, using verified items only."
    else:
        note = (
            f"Lower bound: only {score.coverage:.0%} of the {score.rules_total} risk rules could be "
            "assessed using verified items, so the true level could be higher."
        )
    gaps: list[str] = []
    if unassessed:
        gaps.append("Risk rules that could not be assessed (no evidence found or evidence not verified): " + "; ".join(unassessed))
    if review:
        gaps.append("Risk rules needing a person to check: " + " | ".join(review))
    summary = RiskSummary(
        level=score.level, total_points=score.total_points, coverage=score.coverage,
        complete=score.complete, note=note, lines=lines,
    )
    return summary, gaps


def _overview_ok(text: str, feed_text: str, used: set[str], context: BuiltContext) -> bool:
    ctx = BuiltContext(text="", evidence=[e for e in context.evidence if e.label in used], truncated=False)
    report = extract_citations(text, ctx)
    if not report.has_citations or report.invalid_labels or _BANNED.search(text):
        return False
    return numbers_in(_CITE.sub("", text)) <= numbers_in(_CITE.sub("", feed_text))


def _template_overview(v: VerificationReport, n_verified: int, risk: RiskSummary | None) -> str:
    text = (
        f"Verification result: {v.result.replace('_', ' ')}. "
        f"{n_verified} extracted items verified; {len(v.withheld_item_ids)} could not be confirmed."
    )
    if risk is not None:
        text += f" Risk level from verified items: {risk.level.replace('_', ' ')}. {risk.note}"
    return text


def render_markdown(r: SummaryReport) -> str:
    def fmt(line: SummaryLine) -> str:
        cites = f" [{'; '.join(line.citations)}]" if line.citations else ""
        return f"- {line.text}{cites}"

    out = ["## Overview", r.overview, ""]
    for title, lines in (("Key facts", r.key_facts), ("Compliance", r.compliance)):
        if lines:
            out += [f"## {title}", *[fmt(x) for x in lines], ""]
    if r.risk is not None:
        out += [
            "## Risk",
            f"Level: {r.risk.level.replace('_', ' ')} ({r.risk.total_points} points). {r.risk.note}",
            *[fmt(x) for x in r.risk.lines],
            "",
        ]
    if r.suggestions:
        out += ["## Suggested risks for human review (unverified suggestions)", *[fmt(x) for x in r.suggestions], ""]
    if r.could_not_confirm:
        out += ["## Could not confirm", *[f"- {g}" for g in r.could_not_confirm], ""]
    out.append(f"_{r.disclaimer}_")
    return "\n".join(out)


class SummaryAgent(BaseAgent[SummaryInput, SummaryReport]):
    name = "summary"
    description = "Builds the user-facing report from verified findings only"
    input_schema = SummaryInput
    output_schema = SummaryReport

    async def execute(self, data: SummaryInput) -> SummaryReport:
        v = data.verification
        pages = {e.label: e.page_number for e in data.context.evidence}

        if v.result == "fail":
            report = SummaryReport(
                document_id=data.document_id, status="unverified", verification_result=v.result,
                overview=UNVERIFIED_OVERVIEW, overview_source="template",
                key_facts=[], compliance=[], risk=None, suggestions=[],
                could_not_confirm=[i.detail for i in v.issues] + _item_gaps(v, data.analysis),
                disclaimer=DISCLAIMER, markdown="",
            )
            return report.model_copy(update={"markdown": render_markdown(report)})

        verified = {i.item_id for i in v.items if i.verdict == "verified"}
        withheld = set(v.withheld_finding_ids)

        key_facts = _key_facts(data.analysis, verified, pages)
        comp_lines, comp_gaps = _compliance(data.compliance, withheld, pages)
        risk, risk_gaps = _risk(data.risk, withheld, pages)
        suggestions = [
            _line(f"{c.title}: {c.description}", c.evidence, pages)
            for i, c in enumerate(data.risk.candidates if data.risk else [])
            if f"candidate:{i}" in verified
        ]
        issue_gaps = [i.detail for i in v.issues if i.code not in REDUNDANT_ISSUES]
        gaps = _item_gaps(v, data.analysis) + comp_gaps + risk_gaps + issue_gaps

        overview, source = _template_overview(v, len(verified), risk), "template"
        cited = (key_facts + comp_lines + (risk.lines if risk else []))[:MAX_FEED_LINES]
        used = {c.split()[0] for line in cited for c in line.citations}

        if data.include_overview and used:
            feed = [_feed(line) for line in cited]
            if risk is not None:
                lower = " (a lower bound)" if risk.level != "not_assessed" and not risk.complete else ""
                feed.append(f"Risk level from verified items: {risk.level.replace('_', ' ')}{lower}")
            try:
                draft = await self.ask_structured(
                    build_messages(feed), OverviewDraft, max_tokens=MAX_OUTPUT_TOKENS
                )
                text = draft.overview.strip()
                if _overview_ok(text, "\n".join(feed), used, data.context):
                    overview, source = text, "model"
                else:
                    logger.warning("Summary overview rejected by code checks; using template")
            except Exception as exc:  # the report is still complete without the overview
                logger.error("Summary overview failed: %s", type(exc).__name__)

        report = SummaryReport(
            document_id=data.document_id,
            status="ok" if not gaps else "ok_with_gaps",
            verification_result=v.result,
            overview=overview, overview_source=source,
            key_facts=key_facts, compliance=comp_lines, risk=risk,
            suggestions=suggestions, could_not_confirm=gaps,
            disclaimer=DISCLAIMER, markdown="",
        )
        return report.model_copy(update={"markdown": render_markdown(report)})