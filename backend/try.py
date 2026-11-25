import asyncio
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, "../tests/backend/unit")
from knowledge_fixture import JURISDICTION, write_fixture  # noqa: E402

from app.agents.research.agent import ResearchAgent  # noqa: E402
from app.config import get_settings  # noqa: E402
from app.database.mongodb import close_mongo_connection, connect_to_mongo  # noqa: E402
from app.guardrails.guardrail_manager import GuardrailManager  # noqa: E402
from app.knowledge.embedding_service import embed_query  # noqa: E402
from app.knowledge.knowledge_ingestion import ingest_knowledge  # noqa: E402
from app.knowledge.knowledge_manifest import load_manifest  # noqa: E402
from app.knowledge.knowledge_store import MongoKnowledgeBackend  # noqa: E402
from app.knowledge.knowledge_tools import build_tool_guardrail  # noqa: E402
from app.orchestration.router import route  # noqa: E402
from app.orchestration.state import WorkflowRequest, WorkflowState  # noqa: E402
from app.orchestration.workflow import WorkflowDeps, run_workflow  # noqa: E402

QUESTIONS = [
    "How much notice must a Zorbian Garden Club member give to resign?",
    "What is the penalty for building a shed?",
    "What is the weather in Chennai today?",
]


async def main():
    settings = get_settings()
    settings.chunk_max_chars, settings.chunk_overlap_chars = 140, 20  # small chunks, like the unit tests

    await connect_to_mongo()
    backend = MongoKnowledgeBackend()
    await backend.ensure_index()
    folder = Path(tempfile.mkdtemp())
    write_fixture(folder)
    manifest = load_manifest(folder, allowed_jurisdictions={JURISDICTION})

    report = await ingest_knowledge(manifest, backend)  # real embeddings
    print("ingested:", [(r.source_id, r.status, r.chunks, r.withheld, r.error) for r in report.sources])
    print("ignored:", report.unregistered_files)
    print("waiting 60s for the vector index...")
    await asyncio.sleep(60)

    guard = GuardrailManager(tools=build_tool_guardrail(manifest, backend, embed_query))
    deps = WorkflowDeps(research=ResearchAgent(manifest=manifest, guardrails=guard))
    for question in QUESTIONS:
        req = WorkflowRequest(question=question)
        state = await run_workflow(WorkflowState.start(req, route(req)), deps)
        print("\nQ:", question)
        print("  job:", state.status, "| failed_step:", state.failed_step, "| tokens:", state.total_tokens,
              "| ms:", state.duration_ms)
        if state.research:
            print("  research:", state.research.status, [p.label for p in state.research.passages],
                  "| calls:", [(c.tool, c.status) for c in state.research.tool_calls])
        a = state.legal_answer
        if a:
            print("  answer:", a.status, "|", a.answer[:160])
            print("  citations:", [(c.label, c.source_id, c.page_number) for c in a.citations],
                  "| quotes:", len(a.quotes), "| rejected:", a.rejected_reason)
        print("  warnings:", state.warnings, "| review:", state.review_codes)

    for source in manifest.sources:  # remove the synthetic chunks; the index itself stays
        await backend.replace_source(source.id, [])
    await close_mongo_connection()


asyncio.run(main())