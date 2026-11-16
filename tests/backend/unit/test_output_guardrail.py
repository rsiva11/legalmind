from app.agents.compliance.agent import NOT_FOUND_ACTION
from app.agents.document_qa.agent import UNVERIFIED_MESSAGE
from app.agents.legal_assistant.agent import DISCLAIMER as LEGAL_DISCLAIMER
from app.agents.risk.agent import UNASSESSED_ACTION
from app.agents.summary.agent import DISCLAIMER as SUMMARY_DISCLAIMER
from app.agents.summary.agent import UNVERIFIED_OVERVIEW
from app.business_logic.compliance_rules import NOT_FOUND_DETAIL
from app.guardrails.output_guardrail import check_text
from app.guardrails.result import GuardrailFinding, GuardrailResult
from app.llm.prompt_manager import NOT_FOUND_MESSAGE


def codes(findings):
    return [(f.code, f.action) for f in findings]


def test_clean_text_has_no_findings():
    assert check_text("f", "Payment is due within 45 days.", "commentary") == []


def test_empty_text():
    assert check_text("f", "", "commentary") == []
    assert check_text("f", "   ", "document") == []


def test_legal_conclusion_is_blocked():
    out = check_text("f", "This clause is unenforceable.", "commentary")
    assert codes(out) == [("legal_conclusion", "block")]
    assert out[0].field == "f"


def test_conclusion_wording_from_the_source_is_only_flagged():
    src = "The Agreement is unenforceable if signed under duress."
    out = check_text("f", "The clause says it is unenforceable if signed under duress [E1].", "document", src)
    assert codes(out) == [("legal_conclusion", "flag")]


def test_source_match_respects_word_boundaries():
    out = check_text("f", "This is void.", "commentary", "the party avoided it")
    assert codes(out) == [("legal_conclusion", "block")]


def test_general_profile_allows_legal_concepts():
    assert check_text("f", "A contract may be void if it has an illegal purpose.", "general") == []


def test_directive_advice_is_blocked_even_if_the_source_says_it():
    assert codes(check_text("f", "You should sign this contract.", "general")) == [("legal_advice", "block")]
    same = "You should sign this contract."
    assert codes(check_text("f", same, "document", same)) == [("legal_advice", "block")]
    assert codes(check_text("f", "Never sign without a lawyer.", "general")) == [("legal_advice", "block")]


def test_consult_a_lawyer_is_fine():
    assert check_text("f", "You should consult a qualified lawyer.", "general") == []


def test_outcome_prediction_is_blocked():
    assert codes(check_text("f", "A court will likely decide in your favour.", "general")) == [
        ("outcome_prediction", "block")
    ]
    assert codes(check_text("f", "You will win.", "commentary")) == [("outcome_prediction", "block")]


def test_prediction_wording_in_source_is_flagged():
    src = "Disputes go to Delhi and the court will have exclusive jurisdiction."
    text = "Disputes go to Delhi, where the court will have exclusive jurisdiction [E1]."
    assert codes(check_text("f", text, "document", src)) == [("outcome_prediction", "flag")]


def test_markdown_image_is_blocked():
    out = check_text("f", "Done ![x](https://evil.example/?d=abc)", "commentary")
    assert codes(out) == [("markdown_image", "block"), ("unverified_link", "block")]


def test_link_from_the_source_is_allowed():
    src = "Terms at https://example.com/terms apply."
    assert check_text("f", "See https://example.com/terms for details.", "document", src) == []


def test_link_not_in_source_is_blocked():
    out = check_text("f", "See https://other.example/terms", "document", "Terms apply.")
    assert codes(out) == [("unverified_link", "block")]


def test_general_profile_blocks_all_links():
    assert codes(check_text("f", "More at https://example.com", "general")) == [("unverified_link", "block")]


def test_unknown_citation_label_is_blocked():
    ok = check_text("f", "Notice is 30 days [E1]. Fees are due [E2].", "document", valid_labels={"E1", "E2"})
    assert ok == []
    bad = check_text("f", "Notice is 30 days [E1]. See also [E7].", "document", valid_labels={"E1", "E2"})
    assert codes(bad) == [("unknown_citation", "block")]
    assert "E7" in bad[0].detail


def test_one_finding_per_code():
    out = check_text("f", "This is illegal and unlawful and void.", "commentary")
    assert codes(out) == [("legal_conclusion", "block")]


def test_findings_do_not_echo_the_text():
    out = check_text("f", "This is unenforceable and the secret is hunter2.", "commentary")
    assert out and all("hunter2" not in f.detail and "unenforceable" not in f.detail.lower() for f in out)


def test_fixed_strings_pass_every_profile():
    fixed = [
        NOT_FOUND_ACTION, UNASSESSED_ACTION, UNVERIFIED_MESSAGE, NOT_FOUND_MESSAGE,
        SUMMARY_DISCLAIMER, LEGAL_DISCLAIMER, UNVERIFIED_OVERVIEW, NOT_FOUND_DETAIL,
    ]
    for profile in ("general", "document", "commentary"):
        for text in fixed:
            assert check_text("f", text, profile, "", set()) == [], (profile, text)


def test_result_action_is_the_worst_finding():
    def finding(action):
        return GuardrailFinding(guardrail="output", action=action, code="x", field="f", detail="d")

    assert GuardrailResult(guardrail="output").action == "allow"
    assert GuardrailResult(guardrail="output", findings=[finding("flag")]).action == "flag"
    assert GuardrailResult(guardrail="output", findings=[finding("flag"), finding("block")]).action == "block"