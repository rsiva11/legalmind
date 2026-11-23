from app.schemas.knowledge import KnowledgePassage


def _block(p: KnowledgePassage) -> str:
    version = f", {p.version_note}" if p.version_note else ""
    return f"[{p.label}] {p.source_title}{version} (page {p.page_number})\n{p.text}"


def fit_passages(passages: list[KnowledgePassage], max_chars: int) -> tuple[list[KnowledgePassage], bool]:
    """Keep whole passages from the top until the budget runs out. Labels stay contiguous."""
    kept: list[KnowledgePassage] = []
    used = 0
    for p in passages:
        size = len(_block(p))
        if used + size > max_chars:
            return kept, True
        kept.append(p)
        used += size + 2
    return kept, False


def build_context(passages: list[KnowledgePassage]) -> str:
    if not passages:
        return ""
    return "<knowledge>\n" + "\n\n".join(_block(p) for p in passages) + "\n</knowledge>"