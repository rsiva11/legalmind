from app.config import get_settings
from app.schemas.risk import RiskFinding, RiskLevel, RiskScore


def score_risks(findings: list[RiskFinding]) -> RiskScore:
    settings = get_settings()
    counts = {"triggered": 0, "not_triggered": 0, "unassessed": 0, "needs_review": 0}
    by_category: dict[str, int] = {}
    for f in findings:
        counts[f.outcome] += 1
        if f.outcome == "triggered":
            by_category[f.category] = by_category.get(f.category, 0) + f.points

    total = min(100, sum(by_category.values()))
    assessed = counts["triggered"] + counts["not_triggered"]

    level: RiskLevel
    if assessed == 0:
        level = "not_assessed"
    elif total >= settings.risk_threshold_critical:
        level = "critical"
    elif total >= settings.risk_threshold_high:
        level = "high"
    elif total >= settings.risk_threshold_medium:
        level = "medium"
    else:
        level = "low"

    return RiskScore(
        total_points=total,
        level=level,
        rules_total=len(findings),
        coverage=round(assessed / len(findings), 2) if findings else 0.0,
        complete=bool(findings) and assessed == len(findings),
        by_category=by_category,
        **counts,
    )