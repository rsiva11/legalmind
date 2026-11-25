"""Probe the real knowledge base. Scores only; --answer also runs a few full questions (uses Groq).

Run from the backend folder:
    python ../scripts/probe_knowledge.py
    python ../scripts/probe_knowledge.py --answer
"""
import argparse
import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "backend"))

from app.config import get_settings  # noqa: E402
from app.database.mongodb import close_mongo_connection, connect_to_mongo  # noqa: E402
from app.knowledge.embedding_service import embed_query  # noqa: E402
from app.knowledge.knowledge_manifest import load_manifest  # noqa: E402
from app.knowledge.knowledge_retrieval import search_knowledge  # noqa: E402
from app.knowledge.knowledge_store import MongoKnowledgeBackend  # noqa: E402
from app.orchestration.router import route  # noqa: E402
from app.orchestration.state import WorkflowRequest, WorkflowState  # noqa: E402
from app.orchestration.workflow import run_workflow  # noqa: E402

CONTRACT, ARBITRATION, DATA = "indian-contract-act-1872", "arbitration-and-conciliation-act-1996", "dpdp-act-2023"
QUERIES = [
    (CONTRACT, "What makes the consent of the parties to a contract free?"),
    (CONTRACT, "What compensation can be claimed for breach of contract?"),
    (ARBITRATION, "How is an arbitral tribunal appointed?"),
    (ARBITRATION, "On what grounds can an arbitral award be set aside?"),
    (DATA, "What are the obligations of a data fiduciary?"),
    (DATA, "How is consent given for processing personal data?"),
    (None, "What is the weather in Chennai today?"),
    (None, "Give me a recipe for pasta."),
    (None, "Who won the cricket match yesterday?"),
]
ANSWER_QUERIES = [QUERIES[0], QUERIES[2], QUERIES[4], QUERIES[6]]


async def main(answer: bool) -> int:
    manifest = load_manifest(Path(get_settings().knowledge_dir))
    await connect_to_mongo()
    try:
        backend = MongoKnowledgeBackend()
        on_topic, off_topic = [], []
        for expected, query in QUERIES:
            res = await search_knowledge(query, manifest, backend, embed_query, top_k=4)
            tag = "expected" if expected else "off-topic"
            print(f"\n[{tag}] {query}")
            print("  hits:", [(h.source_id, round(h.score, 3)) for h in res.hits])
            print(f"  withheld={res.withheld} dropped={res.dropped_reasons}")
            if res.hits:
                (on_topic if expected else off_topic).append(res.hits[0].score)
                if expected:
                    print("  top source is the expected one:", res.hits[0].source_id == expected)
        if on_topic and off_topic:
            print(f"\non-topic top-1 scores: {min(on_topic):.3f} to {max(on_topic):.3f}")
            print(f"off-topic top-1 scores: {min(off_topic):.3f} to {max(off_topic):.3f}")
            print(f"gap (lowest on-topic minus highest off-topic): {min(on_topic) - max(off_topic):.3f}")

        if answer:
            for expected, query in ANSWER_QUERIES:
                req = WorkflowRequest(question=query)
                state = await run_workflow(WorkflowState.start(req, route(req)))
                print(f"\nQ: {query}")
                print(f"  job: {state.status} {state.failed_step or ''} | tokens={state.total_tokens} | ms={state.duration_ms}")
                if state.research:
                    print(f"  research: {state.research.status} {[p.label for p in state.research.passages]}")
                a = state.legal_answer
                if a:
                    print(f"  answer: {a.status} | {a.answer[:400]}")
                    print("  citations:", [(c.label, c.source_id, c.page_number) for c in a.citations])
                    print("  quotes:", [q.quote[:120] for q in a.quotes], "| rejected:", a.rejected_reason)
                print("  warnings:", state.warnings, "| review:", state.review_codes)
        return 0
    finally:
        await close_mongo_connection()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--answer", action="store_true")
    raise SystemExit(asyncio.run(main(parser.parse_args().answer)))