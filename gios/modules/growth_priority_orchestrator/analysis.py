"""Deterministic parts of the Growth Priority Orchestrator (growth-priority-orchestrator.md):
merging confirmed duplicates, portfolio buckets, dependency minimums, horizon defaults, and
the portfolio balance check."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable, Optional

from gios.core.schemas import Opportunity

MODULE = "growth_priority_orchestrator"
BUCKETS = {
    "foundational_fix": "Foundational fixes",
    "high_confidence_optimization": "High-confidence optimizations",
    "strategic_experiment": "Strategic experiments",
    "research_bet": "Research / learning bets",
}
HORIZONS = {"now": "Now", "next": "Next", "later": "Later"}
TOP_FUNNEL = {"acquisition", "conversion"}
MAX_BUCKET_SHARE = 0.50
DEPENDENCY_LABELS = {"data": "Data", "engineering": "Engineering", "design": "Design", "content": "Content",
                     "sales": "Sales", "product": "Product", "operations": "Operations"}
CATEGORY_LABELS = {
    "customer_problem": "Customer problem", "acquisition": "Acquisition", "conversion": "Conversion",
    "activation": "Activation", "qualification": "Qualification", "retention": "Retention",
    "revenue": "Revenue", "operations_measurement": "Operations / measurement",
}


def overlap(a: Opportunity, b: Opportunity) -> float:
    """Jaccard overlap of linked signal ids (0–1), shown next to proposed duplicates."""
    sa, sb = set(a.signal_ids), set(b.signal_ids)
    return len(sa & sb) / len(sa | sb) if sa | sb else 0.0


def cluster_overlap(items: list[Opportunity]) -> float:
    pairs = [(a, b) for i, a in enumerate(items) for b in items[i + 1:]]
    return min((overlap(a, b) for a, b in pairs), default=0.0)


def merge(items: list[Opportunity], clusters: Iterable[tuple[list[str], Optional[str]]]) -> list[Opportunity]:
    """Apply confirmed merges. The highest-priority member stays as the primary (ties: title),
    absorbs the others' signals and dependencies, and the rest point to it via merged_into."""
    by_id = {o.id: o.model_copy(deep=True) for o in items}
    for member_ids, title in clusters:
        members = [by_id[i] for i in member_ids if i in by_id and not by_id[i].merged_into]
        if len(members) < 2:
            continue
        primary = sorted(members, key=lambda o: (-(o.priority_score or 0), o.title, o.id))[0]
        for o in members:
            if o is primary:
                continue
            primary.signal_ids = list(dict.fromkeys(primary.signal_ids + o.signal_ids))
            primary.dependencies = list(dict.fromkeys(primary.dependencies + o.dependencies))
            primary.merged_ids = list(dict.fromkeys(primary.merged_ids + [o.id] + o.merged_ids))
            o.merged_into = primary.id
            o.horizon = None
        if title:
            primary.title = title
    return list(by_id.values())


def active(items: Iterable[Opportunity]) -> list[Opportunity]:
    return [o for o in items if not o.merged_into]


# --- Classification defaults ----------------------------------------------------------------------


def default_bucket(o: Opportunity) -> str:
    """Rules, in order: research first or H ≤ 2 → research bet; fixes something broken,
    instrument first, or an operations/measurement item → foundational fix; H ≥ 4 and T ≤ 3 →
    high-confidence optimization; otherwise strategic experiment."""
    rec = o.final_recommendation
    if rec == "research_first" or (o.hypothesis_confidence or 0) <= 2:
        return "research_bet"
    if o.is_fix or rec == "instrument_first" or o.category == "operations_measurement":
        return "foundational_fix"
    if (o.hypothesis_confidence or 0) >= 4 and (o.technical_effort or 5) <= 3:
        return "high_confidence_optimization"
    return "strategic_experiment"


def rule_dependencies(o: Opportunity) -> list[str]:
    """Minimum dependency flags implied by the item itself (the LLM may add more)."""
    deps = []
    if o.final_recommendation == "instrument_first" or "measurement_readiness" in o.risk_flags:
        deps.append("data")
    if o.is_fix or (o.technical_effort or 0) >= 4 or "implementation_dependency" in o.risk_flags:
        deps.append("engineering")
    if o.test_unit == "web" and not o.is_fix:
        deps += ["design", "content"]
    if o.category == "qualification":
        deps += ["sales", "operations"]
    if o.category == "operations_measurement":
        deps.append("operations")
    return list(dict.fromkeys(deps))


def default_horizon(o: Opportunity) -> Optional[str]:
    """Run now / Instrument first → Now; Research first → Next; Defer → Later; Reject → off the board."""
    return {"run_now": "now", "instrument_first": "now", "research_first": "next", "defer": "later"}.get(
        o.final_recommendation or "", None)


def assign_defaults(items: list[Opportunity]) -> list[Opportunity]:
    """Fill bucket, horizon, rule dependencies and order where the user has not set them."""
    out = []
    for o in items:
        o = o.model_copy(deep=True)
        if not o.merged_into:
            o.portfolio_bucket = o.portfolio_bucket or default_bucket(o)
            if o.horizon is None and o.horizon_order is None:
                o.horizon = default_horizon(o)
            o.dependencies = list(dict.fromkeys(o.dependencies + rule_dependencies(o)))
        out.append(o)
    for h in HORIZONS:
        lane = sorted([o for o in out if o.horizon == h and not o.merged_into],
                      key=lambda o: (o.horizon_order if o.horizon_order is not None else 10_000,
                                     -(o.priority_score or 0), o.title))
        for i, o in enumerate(lane, 1):
            o.horizon_order = i
    return out


# --- Portfolio balance -------------------------------------------------------------------------------


@dataclass
class Balance:
    shares: dict[str, float]
    counts: dict[str, int]
    warnings: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)


def balance(items: Iterable[Opportunity], now_capacity: Optional[int] = None) -> Balance:
    """Shares of the planned roadmap (active items on the board) by bucket, with the spec's
    balance warnings: no bucket above 50%, not all top-funnel, Now within capacity."""
    plan = [o for o in active(items) if o.horizon]
    n = len(plan)
    counts = {b: sum(o.portfolio_bucket == b for o in plan) for b in BUCKETS}
    shares = {b: (c / n if n else 0.0) for b, c in counts.items()}
    result = Balance(shares, counts)
    if not n:
        result.warnings.append("The roadmap is empty.")
        return result
    for b, share in shares.items():
        if share > MAX_BUCKET_SHARE:
            result.warnings.append(f"{BUCKETS[b]} make up {share:.0%} of the roadmap (more than "
                                   f"{MAX_BUCKET_SHARE:.0%}). Rebalance the portfolio.")
    if all(o.category in TOP_FUNNEL for o in plan):
        result.warnings.append("Every roadmap item is top-funnel (acquisition or conversion). Add qualification, "
                               "activation, retention or revenue work.")
    if now_capacity is not None:
        now = sum(o.horizon == "now" for o in plan)
        if now > now_capacity:
            result.warnings.append(f"Now holds {now} items but capacity is {now_capacity}. Move some to Next.")
    for b, c in counts.items():
        if c == 0:
            result.notes.append(f"No {BUCKETS[b].lower()} in the plan.")
    return result
