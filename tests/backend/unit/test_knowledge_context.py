from knowledge_fixture import passage
from app.knowledge.knowledge_context import build_context, fit_passages


def test_context_format():
    p1 = passage("K1", "First text.", title="Source A", page=2)
    p2 = passage("K2", "Second text.", title="Source B", page=5).model_copy(
        update={"version_note": "as on 1 June 2026"}
    )
    assert build_context([p1, p2]) == (
        "<knowledge>\n[K1] Source A (page 2)\nFirst text.\n\n"
        "[K2] Source B, as on 1 June 2026 (page 5)\nSecond text.\n</knowledge>"
    )


def test_empty_context():
    assert build_context([]) == ""


def test_budget_drops_whole_passages_from_the_tail():
    passages = [passage(f"K{i}", "x" * 100) for i in (1, 2, 3)]  # each block is 133 characters
    kept, truncated = fit_passages(passages, 300)
    assert [p.label for p in kept] == ["K1", "K2"] and truncated is True
    kept, truncated = fit_passages(passages, 1000)
    assert len(kept) == 3 and truncated is False