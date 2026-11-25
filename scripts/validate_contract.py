"""Measure the full analysis on ONE contract. Prints counts and codes, not contract text.

Run from the backend folder:
    python ../scripts/validate_contract.py "D:\\path\\to\\contract.pdf"
    python ../scripts/validate_contract.py "D:\\path\\to\\contract.pdf" --show-facts

--show-facts also prints the extracted values so YOU can check them against the PDF.
Do not paste that output anywhere. Everything retrieved from the contract is sent to Groq.
"""
import argparse
import asyncio
import sys
import time
import uuid
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "backend"))

from app.agents.verification.agent import FACT_FIELDS  # noqa: E402
from app.business_logic.contract_rules import analyze_dates  # noqa: E402
from app.config import get_settings  # noqa: E402
from app.database.mongodb import close_mongo_connection, connect_to_mongo  # noqa: E402
from app.ingestion.ingestion_pipeline import ingest_document  # noqa: E402
from app.knowledge import vector_store  # noqa: E402
from app.knowledge.analysis_retrieval import build_analysis_context  # noqa: E402
from app.knowledge.embedding_service import embed_documents, embed_query  # noqa: E402
from app.orchestration.router import route  # noqa: E402
from app.orchestration.state import WorkflowRequest, WorkflowState  # noqa: E402
from app.orchestration.workflow import WorkflowDeps, run_workflow  # noqa: E402
from app.core.logging import setup_logging  # noqa: E402


def section(title: str) -> None:
    print(f"\n== {title} ==")


def show(label: str, value) -> None:
    print(f"  {label}: {value}")


async def wait_for_index(doc_id: str, seconds: int = 120) -> float | None:
    probe = embed_query("termination")
    started = time.perf_counter()
    while time.perf_counter() - started < seconds:
        if await vector_store.search(probe, doc_id, 1):
            return round(time.perf_counter() - started, 1)
        await asyncio.sleep(5)
    return None


def report(state: WorkflowState, settings, show_facts: bool) -> None:
    section("3. Job")
    show("status / failed_step", f"{state.status} / {state.failed_step}")
    show("duration", f"{state.duration_ms / 1000:.1f}s (job timeout {settings.workflow_timeout_seconds:.0f}s)")
    budget = settings.analysis_token_budget
    show("tokens", f"{state.total_tokens} of {budget} budget ({state.total_tokens / budget:.0%}), "
                   f"{state.total_llm_calls} LLM calls")
    for s in state.steps:
        print(f"    {s.agent}: {s.status} {s.duration_ms}ms llm_calls={s.llm_calls} tokens={s.tokens} {s.error_type or ''}".rstrip())
    show("warnings", state.warnings)
    show("guardrail findings", [(f.guardrail, f.action, f.code, f.field) for f in state.guardrail_findings])

    c, v = state.contract, state.verification
    if c:
        section("4. Contract Agent")
        present = [f for f in FACT_FIELDS if getattr(c, f) is not None]
        show("facts found", f"{len(present)}/{len(FACT_FIELDS)}: {present}")
        show("parties / clauses / obligations", f"{len(c.parties)} / {len(c.clauses)} / {len(c.obligations)}")
        show("clause types", dict(Counter(x.clause_type for x in c.clauses)))
        show("items dropped (quote not found)", c.dropped_unverified)
    if v:
        section("5. Verification")
        show("result / semantic checked", f"{v.result} / {v.semantic_checked}")
        show("verdicts", v.counts)
        show("withheld items", v.withheld_item_ids)
        show("issue codes", sorted(i.code for i in v.issues))

    comp = state.compliance
    if comp:
        section("6. Compliance (uses the EXAMPLE policy rules)")
        show("statuses", comp.status_counts)
        share = sum(1 for f in comp.findings if f.status in ("not_found", "needs_review")) / max(len(comp.findings), 1)
        show("unresolved share", f"{share:.0%} (escalates at >= {settings.escalation_max_unresolved_share:.0%})")

    r = state.risk
    if r:
        section("7. Risk (uses the EXAMPLE risk rules)")
        show("level / points", f"{r.score.level} / {r.score.total_points}")
        show("coverage / complete", f"{r.score.coverage:.0%} / {r.score.complete} "
                                    f"(escalates below {settings.escalation_min_risk_coverage:.0%})")
        show("triggered rules", [f.rule_id for f in r.findings if f.outcome == "triggered"])
        show("candidates kept / dropped", f"{len(r.candidates)} / {r.candidates_dropped}")

    if c and v:
        verified = {i.item_id for i in v.items if i.verdict == "verified"}
        d = analyze_dates(c, None, verified)
        section("8. Dates")
        show("status / basis / reasons", f"{d.status} / {d.expiry_basis} / {d.reasons}")

    section("9. Result")
    if state.summary:
        s = state.summary
        show("summary", f"{s.status}, overview from {s.overview_source}, "
                        f"{len(s.key_facts)} key facts, {len(s.could_not_confirm)} gaps")
    show("human review", f"{state.needs_human_review} {state.review_codes}")

    if show_facts and c:
        section("FACTS (check these against the PDF yourself; do not paste)")
        verdicts = {i.item_id: i.verdict for i in v.items} if v else {}
        for f in FACT_FIELDS:
            fact = getattr(c, f)
            print(f"  {f}: {fact.value if fact else None} [{verdicts.get(f'fact:{f}', 'n/a')}]")
        for i, p in enumerate(c.parties):
            print(f"  party: {p.name} ({p.role}) [{verdicts.get(f'party:{i}', 'n/a')}]")


async def main(path: Path, show_facts: bool) -> int:
    settings = get_settings()

    section("1. Ingestion")
    try:
        ingest = ingest_document(path)
    except ValueError as exc:
        print("  ingestion failed:", exc)
        return 1
    sizes = sorted(c.char_count for c in ingest.chunks)
    show("pages / chunks", f"{ingest.page_count} / {ingest.chunk_count}")
    show("chunk size min/median/max", f"{sizes[0]} / {sizes[len(sizes) // 2]} / {sizes[-1]} (limit {settings.chunk_max_chars})")
    show("withheld by document guardrail", f"{ingest.withheld_count} (needs_review={ingest.needs_review})")
    show("clause-type tags", dict(Counter(t for c in ingest.chunks for t in c.metadata.clause_types)))
    show("chunks with a date / an amount", f"{sum(1 for c in ingest.chunks if c.metadata.dates)} / "
                                          f"{sum(1 for c in ingest.chunks if c.metadata.money)}")

    await connect_to_mongo()
    doc_id = f"validate-{uuid.uuid4().hex[:12]}"
    try:
        await vector_store.ensure_vector_index()
        started = time.perf_counter()
        vectors = embed_documents([c.text for c in ingest.chunks])
        embed_s = round(time.perf_counter() - started, 1)
        await vector_store.add_chunks(doc_id, ingest.chunks, vectors)

        section("2. Index and retrieval")
        show("embedding time (s)", embed_s)
        ready = await wait_for_index(doc_id)
        show("index ready after (s)", ready)
        if ready is None:
            print("  the index did not become ready; stopping")
            return 1
        started = time.perf_counter()
        ctx = await build_analysis_context(doc_id)
        show("analysis retrieval (s)", round(time.perf_counter() - started, 1))
        show("evidence items / characters", f"{len(ctx.evidence)} / {len(ctx.text)} "
                                           f"(budget {settings.analysis_context_max_chars}, truncated={ctx.truncated})")
        show("retrieval guardrail findings", [(f.code, f.action) for f in ctx.guardrail_findings])

        async def cached(_document_id: str):
            return ctx  # exactly what was measured above

        req = WorkflowRequest(document_id=doc_id, mode="full_analysis")
        state = await run_workflow(WorkflowState.start(req, route(req)), WorkflowDeps(analysis_context=cached))
        report(state, settings, show_facts)
        return 0
    finally:
        deleted = await vector_store.delete_document(doc_id)
        print(f"\ncleanup: removed {deleted} stored chunks")
        await close_mongo_connection()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("pdf", type=Path)
    parser.add_argument("--show-facts", action="store_true")
    args = parser.parse_args()
    raise SystemExit(asyncio.run(main(args.pdf, args.show_facts)))