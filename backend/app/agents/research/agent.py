import logging
from pathlib import Path

from app.agents.base_agent import BaseAgent
from app.agents.research.schemas import ResearchInput, ResearchResult
from app.config import get_settings
from app.guardrails.guardrail_manager import GuardrailManager
from app.guardrails.result import GuardrailFinding
from app.knowledge.knowledge_context import build_context, fit_passages
from app.knowledge.knowledge_manifest import LoadedManifest, load_manifest
from app.knowledge.knowledge_store import MongoKnowledgeBackend
from app.knowledge.knowledge_tools import SEARCH_TOOL, build_tool_guardrail
from app.schemas.knowledge import KnowledgePassage, SourceRef

logger = logging.getLogger(__name__)


class ResearchAgent(BaseAgent[ResearchInput, ResearchResult]):
    name = "research"
    description = "Searches the registered legal knowledge base through the tool guardrail"
    input_schema = ResearchInput
    output_schema = ResearchResult

    def __init__(self, manifest: LoadedManifest | None = None, guardrails: GuardrailManager | None = None) -> None:
        self._manifest = manifest
        self._guardrails = guardrails

    def _resolve(self) -> tuple[LoadedManifest, GuardrailManager]:
        """Build the production wiring on first use, so creating the agent costs nothing."""
        if self._manifest is None:
            self._manifest = load_manifest(Path(get_settings().knowledge_dir))  # ManifestError fails the run
        if self._guardrails is None:
            from app.knowledge.embedding_service import embed_query  # lazy: loads the model library

            tools = build_tool_guardrail(self._manifest, MongoKnowledgeBackend(), embed_query)
            self._guardrails = GuardrailManager(tools=tools)
        return self._manifest, self._guardrails

    async def execute(self, data: ResearchInput) -> ResearchResult:
        settings = get_settings()
        manifest, guardrails = self._resolve()
        session = guardrails.tool_session(self.name)
        call = await session.call(SEARCH_TOOL, {"query": data.question, "top_k": settings.knowledge_top_k})

        if call.status != "ok" or call.output is None:
            return ResearchResult(
                status="unavailable",
                unavailable_reason=call.record.code or call.status,
                tool_calls=list(session.records),
                findings=list(session.findings),
            )

        passages: list[KnowledgePassage] = []
        local_dropped = 0
        for hit in call.output.hits:
            source = manifest.get(hit.source_id)
            if source is None:  # the tool already filters, but one layer is never enough
                local_dropped += 1
                continue
            passages.append(
                KnowledgePassage(
                    label=f"K{len(passages) + 1}",
                    source_id=source.id,
                    source_title=source.title,  # from the manifest, never from the database
                    source_url=source.source_url,
                    jurisdiction=source.jurisdiction,
                    version_note=source.version_note,
                    chunk_index=hit.chunk_index,
                    page_number=hit.page_number,
                    text=hit.text,
                    score=hit.score,
                )
            )
        kept, truncated = fit_passages(passages, settings.knowledge_context_max_chars)

        sources: list[SourceRef] = []
        for p in kept:
            if all(s.source_id != p.source_id for s in sources):
                src = manifest.get(p.source_id)
                sources.append(
                    SourceRef(
                        source_id=src.id, title=src.title, publisher=src.publisher, url=src.source_url,
                        jurisdiction=src.jurisdiction, version_note=src.version_note,
                    )
                )

        findings = list(session.findings)
        withheld = call.output.withheld
        if withheld:
            findings.append(
                GuardrailFinding(
                    guardrail="retrieval", action="block", code="suspicious_content", field="knowledge",
                    detail=f"{withheld} knowledge passage(s) were withheld from the model",
                )
            )
        return ResearchResult(
            status="found" if kept else "not_found",
            passages=kept,
            sources=sources,
            context=build_context(kept),
            truncated=truncated,
            withheld_count=withheld,
            dropped_count=call.output.dropped + local_dropped,
            tool_calls=list(session.records),
            findings=findings,
        )