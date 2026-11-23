import asyncio
import hashlib
import json
import logging
import time
from collections.abc import Awaitable, Callable, Iterator
from dataclasses import dataclass
from typing import Any, Literal
from urllib.parse import urlsplit

from pydantic import BaseModel, SerializeAsAny, ValidationError

from app.config import get_settings
from app.guardrails.prompt_injection import detect_injection
from app.guardrails.result import GuardrailFinding

logger = logging.getLogger(__name__)

NAME = "tool"
Handler = Callable[[Any], Awaitable[Any]]
CallStatus = Literal["ok", "blocked", "error"]


@dataclass(frozen=True)
class ToolSpec:
    name: str
    description: str
    params: type[BaseModel]  # must set extra="forbid"
    output: type[BaseModel]
    handler: Handler
    url_fields: tuple[str, ...] = ()  # str fields that hold a URL; checked against allowed_hosts
    allowed_hosts: frozenset[str] = frozenset()
    document_id_field: str | None = None  # if set, must equal the session's document id
    max_calls: int = 3  # per request
    timeout_seconds: float | None = None  # None = config default
    max_output_chars: int = 20_000


class ToolRegistry:
    def __init__(self) -> None:
        self._tools: dict[str, ToolSpec] = {}

    def register(self, spec: ToolSpec) -> None:
        if spec.name in self._tools:
            raise ValueError(f"Tool already registered: {spec.name}")
        if spec.params.model_config.get("extra") != "forbid":
            raise ValueError(f"{spec.name}: the params model must set extra='forbid'")
        if spec.url_fields and not spec.allowed_hosts:
            raise ValueError(f"{spec.name}: url_fields need allowed_hosts")
        unknown = [f for f in (*spec.url_fields, spec.document_id_field) if f and f not in spec.params.model_fields]
        if unknown:
            raise ValueError(f"{spec.name}: not a field of the params model: {', '.join(unknown)}")
        self._tools[spec.name] = spec

    def get(self, name: str) -> ToolSpec | None:
        return self._tools.get(name)


class ToolPolicy:
    """Which agent may call which tool. Deny by default."""

    def __init__(self, allowed: dict[str, set[str]] | None = None) -> None:
        self._allowed = {agent: frozenset(tools) for agent, tools in (allowed or {}).items()}

    def allows(self, agent: str, tool: str) -> bool:
        return tool in self._allowed.get(agent, frozenset())


class ToolCallRecord(BaseModel):
    agent: str
    tool: str  # "unknown" when the name is not a registered tool
    status: CallStatus
    code: str | None = None
    duration_ms: int
    params_digest: str


class ToolResult(BaseModel):
    status: CallStatus
    output: SerializeAsAny[BaseModel] | None = None
    findings: list[GuardrailFinding] = []
    record: ToolCallRecord


def host_allowed(url: str, allowed_hosts: frozenset[str]) -> bool:
    """https only, exact host match, default port, no credentials in the URL."""
    try:
        parts = urlsplit(url.strip())
        host, port = parts.hostname, parts.port
    except ValueError:
        return False
    return (
        parts.scheme == "https"
        and host is not None
        and host.lower() in {h.lower() for h in allowed_hosts}
        and parts.username is None
        and parts.password is None
        and port in (None, 443)
    )


def _strings(obj: Any) -> Iterator[str]:
    if isinstance(obj, str):
        yield obj
    elif isinstance(obj, dict):
        for value in obj.values():
            yield from _strings(value)
    elif isinstance(obj, (list, tuple)):
        for value in obj:
            yield from _strings(value)


def _digest(params: Any) -> str:
    try:
        raw = json.dumps(params, sort_keys=True, default=str)
    except Exception:
        raw = repr(type(params))
    return hashlib.sha256(raw.encode()).hexdigest()[:12]


def _finding(code: str, label: str, detail: str, action: str = "block") -> GuardrailFinding:
    return GuardrailFinding(guardrail=NAME, action=action, code=code, field=f"tool:{label}", detail=detail)


class ToolSession:
    """Per-request, per-agent gate. Create one with ToolGuardrail.session()."""

    def __init__(
        self,
        registry: ToolRegistry,
        policy: Any,
        agent: str,
        document_id: str | None,
        max_calls: int,
        default_timeout: float,
    ) -> None:
        self._registry, self._policy = registry, policy
        self._agent, self._document_id = agent, document_id
        self._max_calls, self._default_timeout = max_calls, default_timeout
        self._attempts = 0
        self._per_tool: dict[str, int] = {}
        self.records: list[ToolCallRecord] = []
        self.findings: list[GuardrailFinding] = []

    async def call(self, tool: str, params: Any) -> ToolResult:
        start = time.perf_counter()
        self._attempts += 1  # every attempt counts, including blocked ones
        spec = self._registry.get(tool) if isinstance(tool, str) else None
        label = spec.name if spec else "unknown"
        try:
            status, code, output, findings = await self._run(spec, label, params)
        except Exception as exc:  # fail closed: an unchecked call is not made
            logger.error("Tool guardrail error: %s", type(exc).__name__)
            status, code, output = "blocked", "guardrail_error", None
            findings = [_finding("guardrail_error", label, "The tool check failed, so the call was blocked")]

        record = ToolCallRecord(
            agent=self._agent, tool=label, status=status, code=code,
            duration_ms=int((time.perf_counter() - start) * 1000), params_digest=_digest(params),
        )
        self.records.append(record)
        self.findings.extend(findings)
        logger.info("Tool call agent=%s tool=%s status=%s code=%s", self._agent, label, status, code)
        return ToolResult(status=status, output=output, findings=findings, record=record)

    async def _run(self, spec: ToolSpec | None, label: str, params: Any):
        def stop(code: str, detail: str, status: CallStatus = "blocked"):
            return status, code, None, [_finding(code, label, detail)]

        if spec is None or not self._policy.allows(self._agent, spec.name):
            return stop("tool_not_allowed", "This agent may not call that tool")

        used = self._per_tool.get(spec.name, 0) + 1
        self._per_tool[spec.name] = used
        if self._attempts > self._max_calls:
            return stop("call_limit", "Tool call limit for this request reached")
        if used > spec.max_calls:
            return stop("call_limit", "Call limit for this tool reached")

        try:
            p = spec.params.model_validate(params)
        except ValidationError:
            return stop("invalid_params", "The tool parameters did not match the expected shape")

        for name in spec.url_fields:
            value = getattr(p, name)
            if value is not None and not host_allowed(str(value), spec.allowed_hosts):
                return stop("host_not_allowed", "The address is not on the allowed list for this tool")

        if spec.document_id_field is not None:
            if self._document_id is None or getattr(p, spec.document_id_field) != self._document_id:
                return stop("wrong_document", "The tool may only be used on the document selected for this request")

        timeout = spec.timeout_seconds if spec.timeout_seconds is not None else self._default_timeout
        try:
            raw = await asyncio.wait_for(spec.handler(p), timeout)
        except TimeoutError:
            return stop("tool_timeout", "The tool did not respond in time", "error")
        except Exception as exc:  # the message may hold secrets, so log only the type
            logger.error("Tool %s raised %s", label, type(exc).__name__)
            return stop("tool_error", "The tool failed", "error")

        try:
            out = spec.output.model_validate(raw)
        except ValidationError:
            return stop("invalid_output", "The tool returned data in an unexpected shape")
        dumped = out.model_dump(mode="json")
        if len(json.dumps(dumped)) > spec.max_output_chars:
            return stop("output_too_large", "The tool output was larger than allowed")

        hits = [h for text in _strings(dumped) for h in detect_injection(text)]
        names = ", ".join(sorted({h.code for h in hits}))
        if any(h.severity == "high" for h in hits):
            return stop("suspicious_output", f"Tool output looked like instructions and was not used; matched: {names}")
        flags = [_finding("suspicious_output", label, f"Kept; matched: {names}", "flag")] if hits else []
        return "ok", None, out, flags


class ToolGuardrail:
    def __init__(self, registry: ToolRegistry | None = None, policy: Any = None) -> None:
        self.registry = registry if registry is not None else ToolRegistry()
        self.policy = policy if policy is not None else ToolPolicy()

    def session(self, agent: str, document_id: str | None = None) -> ToolSession:
        s = get_settings()
        return ToolSession(
            self.registry, self.policy, agent, document_id, s.tool_max_calls_per_request, s.tool_timeout_seconds
        )