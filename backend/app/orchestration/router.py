from app.orchestration.state import RouteDecision, WorkflowRequest


class RoutingError(ValueError):
    """The request cannot be mapped to a workflow. The message is safe to show the user."""


def route(request: WorkflowRequest) -> RouteDecision:
    question = (request.question or "").strip()

    if request.document_id is None:
        if request.mode not in (None, "legal_question"):
            raise RoutingError("This mode needs a document: select one first")
        if not question:
            raise RoutingError("A question is required")
        return RouteDecision(
            workflow="legal_question",
            reason="No document selected: general legal question",
        )

    if request.mode is None:
        raise RoutingError("Choose a mode: document_qa, full_analysis or legal_question")

    if request.mode == "legal_question":
        if not question:
            raise RoutingError("A question is required")
        return RouteDecision(
            workflow="legal_question",
            reason="General question; the selected document is not used",
        )

    if request.mode == "document_qa":
        if not question:
            raise RoutingError("A question is required for document questions")
        return RouteDecision(workflow="document_qa", reason="Question about the selected document")

    return RouteDecision(workflow="full_analysis", reason="Full analysis of the selected document")