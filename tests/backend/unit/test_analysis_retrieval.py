import asyncio
from typing import get_args

from app.config import get_settings

from app.agents.contract.schemas import ClauseType
from app.agents.verification.agent import FACT_FIELDS
from app.knowledge.analysis_retrieval import (
    CLAUSE_QUERIES, FACT_QUERIES, build_analysis_context, retrieve_for_analysis,
)
from app.schemas.document import RetrievalResult, RetrievedChunk



DISTINCT_QUERIES = set(FACT_QUERIES.values()) | set(CLAUSE_QUERIES.values())


def chunk(i, score, text="t"):
    return RetrievedChunk(document_id="d", chunk_index=i, page_number=i + 1, text=text, score=score)


def fake_retrieve(table, calls=None):
    async def fn(question, document_id, top_k=None):
        if calls is not None:
            calls.append((question, document_id, top_k))
        chunks = table.get(question, [])
        return RetrievalResult(question=question, found=bool(chunks), chunks=chunks)
    return fn


def test_clause_queries_match_clause_types():
    assert set(CLAUSE_QUERIES) == set(get_args(ClauseType))


def test_fact_queries_match_contract_facts():
    assert set(FACT_QUERIES) == {"parties", *FACT_FIELDS}


def test_one_search_per_distinct_query():
    calls = []
    asyncio.run(retrieve_for_analysis("d", retrieve_fn=fake_retrieve({}, calls), top_k=2))
    assert len(DISTINCT_QUERIES) == 17
    assert {c[0] for c in calls} == DISTINCT_QUERIES
    assert len(calls) == len(DISTINCT_QUERIES)  # no query is run twice
    assert {c[1:] for c in calls} == {("d", 2)}

def test_merge_dedupes_and_ranks_by_matches():
    table = {
        FACT_QUERIES["termination_notice"]: [chunk(5, 0.9), chunk(2, 0.5)],
        CLAUSE_QUERIES["termination"]: [chunk(5, 0.8), chunk(7, 0.99)],
        CLAUSE_QUERIES["payment"]: [chunk(2, 0.6)],
    }
    res = asyncio.run(retrieve_for_analysis("d", retrieve_fn=fake_retrieve(table)))
    assert res.found is True
    assert [c.chunk_index for c in res.chunks] == [5, 2, 7]  # two matches, two matches, one match
    assert res.chunks[1].score == 0.6  # chunk 2 keeps its best score


def test_nothing_found():
    res = asyncio.run(retrieve_for_analysis("d", retrieve_fn=fake_retrieve({})))
    assert (res.found, res.chunks) == (False, [])


def test_budget_drops_whole_chunks_and_flags_truncation():
    big = "x" * 5000
    table = {
        FACT_QUERIES["parties"]: [chunk(1, 0.9, big)],
        FACT_QUERIES["effective_date"]: [chunk(2, 0.8, big)],
        FACT_QUERIES["governing_law"]: [chunk(3, 0.7, big)],
    }
    ctx = asyncio.run(build_analysis_context("d", retrieve_fn=fake_retrieve(table)))
    assert [e.label for e in ctx.evidence] == ["E1", "E2"]
    assert ctx.truncated is True