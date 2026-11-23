from collections.abc import Callable

from app.guardrails.tool_guardrail import ToolGuardrail, ToolPolicy, ToolRegistry, ToolSpec
from app.knowledge.knowledge_manifest import LoadedManifest
from app.knowledge.knowledge_retrieval import search_knowledge
from app.knowledge.knowledge_store import KnowledgeBackend
from app.schemas.knowledge import SearchKnowledgeOutput, SearchKnowledgeParams

SEARCH_TOOL = "search_knowledge"
RESEARCH_AGENT = "research"


def build_tool_guardrail(
    manifest: LoadedManifest, backend: KnowledgeBackend, embed_query_fn: Callable[[str], list[float]]
) -> ToolGuardrail:
    async def handler(params: SearchKnowledgeParams) -> SearchKnowledgeOutput:
        result = await search_knowledge(params.query, manifest, backend, embed_query_fn, top_k=params.top_k)
        return SearchKnowledgeOutput(hits=result.hits, withheld=result.withheld, dropped=result.dropped)

    registry = ToolRegistry()
    registry.register(
        ToolSpec(
            name=SEARCH_TOOL,
            description="Search the registered legal knowledge base",
            params=SearchKnowledgeParams,
            output=SearchKnowledgeOutput,
            handler=handler,
            max_calls=2,
            max_output_chars=30_000,
        )
    )
    return ToolGuardrail(registry, ToolPolicy({RESEARCH_AGENT: {SEARCH_TOOL}}))