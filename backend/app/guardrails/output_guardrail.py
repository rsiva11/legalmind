import re
from typing import Literal

from app.guardrails.result import GuardrailFinding
from app.utils.text_utils import norm

NAME = "output"
Profile = Literal["general", "document", "commentary"]


def _compile(*patterns: str) -> list[re.Pattern]:
    return [re.compile(p, re.I) for p in patterns]


CONCLUSION = _compile(
    r"\b(?:un)?enforceable\b", r"\billegal\b", r"\bunlawful\b", r"\bvoid(?:able)?\b", r"\binvalid\b"
)
ADVICE = _compile(
    r"\byou (?:should|must|need to|ought to) (?:not )?(?:sign|accept|reject|sue|settle|terminate|renew)\b",
    r"\b(?:do not|don't|never) sign\b",
    r"\bI (?:recommend|advise|suggest) (?:that you )?(?:not )?(?:sign|signing|accept|accepting|reject|rejecting|suing|settling)\b",
    r"\b(?:safe|fine) to sign\b",
)
PREDICTION = _compile(
    r"\b(?:a|the) court (?:will|would|is likely to)\b",
    r"\byou (?:will|would) (?:win|lose|prevail)\b",
    r"\b(?:likely|certain|bound) to (?:win|lose|succeed|fail)\b",
    r"\b(?:strong|weak|winning) (?:case|claim)\b",
)
IMAGE = re.compile(r"!\[[^\]]*\]\([^)]*\)")
URL = re.compile(r"(?:https?://|www\.)[^\s)\]>\"']+", re.I)
LABEL = re.compile(r"\[(E\d+)\]")

# code -> (message when blocked, message when only flagged)
_MESSAGES = {
    "legal_conclusion": (
        "States a legal conclusion (such as validity or enforceability) that the source text does not use",
        "Uses legal-conclusion wording that also appears in the source text; confirm it is quoted, not asserted",
    ),
    "legal_advice": ("Gives directive advice about the user's own decision", ""),
    "outcome_prediction": (
        "Predicts a legal outcome",
        "Uses outcome wording that also appears in the source text; confirm it is quoted, not asserted",
    ),
}


def _finding(code: str, field: str, action: str, detail: str) -> GuardrailFinding:
    return GuardrailFinding(guardrail=NAME, action=action, code=code, field=field, detail=detail)


def _in_source(phrase: str, source_norm: str) -> bool:
    """Whole-word match, so 'void' is not found inside 'avoided'."""
    if not source_norm:
        return False
    return re.search(rf"(?<!\w){re.escape(norm(phrase))}(?!\w)", source_norm) is not None


def check_text(
    field: str,
    text: str,
    profile: Profile,
    source: str = "",
    valid_labels: set[str] | None = None,
) -> list[GuardrailFinding]:
    """Pure function: findings for one piece of model-written text."""
    if not text or not text.strip():
        return []
    src = norm(source)
    findings: list[GuardrailFinding] = []

    # (code, patterns, may wording that appears in the source be downgraded to a flag?)
    groups = [("legal_advice", ADVICE, False), ("outcome_prediction", PREDICTION, True)]
    if profile != "general":
        groups.insert(0, ("legal_conclusion", CONCLUSION, True))
    for code, patterns, downgradable in groups:
        matches = [m.group(0) for p in patterns for m in p.finditer(text)]
        if not matches:
            continue
        block_msg, flag_msg = _MESSAGES[code]
        if any(not (downgradable and _in_source(m, src)) for m in matches):
            findings.append(_finding(code, field, "block", block_msg))
        else:
            findings.append(_finding(code, field, "flag", flag_msg))

    if IMAGE.search(text):
        findings.append(_finding("markdown_image", field, "block", "Contains an embedded image link, which can leak data"))
    urls = {m.group(0).rstrip(".,;:") for m in URL.finditer(text)}
    if any(not _in_source(u, src) for u in urls):
        findings.append(_finding("unverified_link", field, "block", "Contains a web link that is not in the source text"))

    if valid_labels is not None:
        bad = sorted({label for label in LABEL.findall(text) if label not in valid_labels})
        if bad:
            findings.append(
                _finding("unknown_citation", field, "block", "Cites evidence that was not provided: " + ", ".join(bad))
            )
    return findings