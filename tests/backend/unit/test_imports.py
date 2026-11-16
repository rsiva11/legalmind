import importlib

import pytest

MODULES = [
    "app.agents.base_agent",
    "app.agents.legal_assistant.agent",
    "app.agents.contract.agent",
    "app.agents.compliance.agent",
    "app.agents.risk.agent",
    "app.agents.verification.agent",
    "app.agents.summary.agent",
    "app.agents.document_qa.agent",
    "app.business_logic.compliance_rules",
    "app.business_logic.risk_rules",
    "app.business_logic.scoring_engine",
    "app.knowledge.analysis_retrieval",
    "app.knowledge.context_builder",
    "app.knowledge.citation_manager",
    "app.orchestration.router",
    "app.orchestration.state",
    "app.main",
    "app.orchestration.execution_context",
    "app.orchestration.workflow",
    "app.orchestration.task_manager",
    "app.orchestration.orchestrator",
    "app.database.repositories.job_repository",
    "app.guardrails.result",
    "app.guardrails.output_guardrail",
    "app.guardrails.guardrail_manager",
    "app.guardrails.prompt_injection",
    "app.guardrails.pii_guardrail",
    "app.guardrails.input_guardrail",
]


@pytest.mark.parametrize("module", MODULES)
def test_module_imports(module):
    importlib.import_module(module)