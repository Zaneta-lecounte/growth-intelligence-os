"""Deterministic anomaly detection and funnel sizing for the Behavioral Friction Analyzer.

Each check compares a slice of web behavior with a standardized baseline, so mix effects
(e.g. paid social being mostly mobile) don't masquerade as friction:
- device divergence: (page, device) vs the other device on the same page, standardized by source
- source divergence: (page, source) vs other sources on the same page, standardized by device
- load time: (page, device) vs the same device's other pages (the site baseline)
- trend: (page, source) first vs last month
A metric is flagged only in its "bad" direction when |z| >= z threshold AND the relative
deviation >= the minimum AND the slice has enough volume.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import pandas as pd

from gios.core import stats

MODULE = "behavioral_friction_analyzer"
COUNT_COLUMNS = ["sessions", "exits", "scroll_75", "cta_clicks", "form_starts", "form_completes", "rage_clicks"]


@dataclass(frozen=True)
class Metric:
    name: str
    label: str
    numerator: str
    denominator: str
    bad: str  # "high" or "low"


RATE_METRICS = {
    "exit_rate": Metric("exit_rate", "Exit rate", "exits", "sessions", "high"),
    "cta_rate": Metric("cta_rate", "CTA click rate", "cta_clicks", "sessions", "low"),
    "form_completion": Metric("form_completion", "Form start→complete", "form_completes", "form_starts", "low"),
    "rage_rate": Metric("rage_rate", "Rage clicks per session", "rage_clicks", "sessions", "high"),
}
LOAD_LABEL = "Avg load time"
PAGE_STAGE = {"home": "awareness", "features": "consideration", "compare": "evaluation",
              "pricing": "evaluation", "demo": "evaluation"}


@dataclass(frozen=True)
class Thresholds:
    z: float = 3.0
    min_deviation: float = 0.20     # relative deviation from the baseline
    min_volume: int = 500           # denominator (sessions or form starts) in the slice
    load_deviation: float = 0.30    # relative load-time deviation from the site baseline
    trend_min_volume: int = 200     # per month


@dataclass
class MetricFlag:
    check: str            # device | source | load | trend
    metric: str
    label: str
    observed: float
    expected: float
    deviation: float
    z: Optional[float]
    volume: int
    numerator: int = 0

    def describe(self) -> str:
        if self.metric == "load_ms":
            return (f"{self.label} {self.observed:,.0f} ms vs {self.expected:,.0f} ms baseline "
                    f"({self.deviation:+.0%}, {self.volume:,} sessions)")
        what = "first→last month" if self.check == "trend" else "baseline"
        return (f"{self.label} {self.observed:.1%} vs {self.expected:.1%} {what} "
                f"({self.deviation:+.0%}, z = {self.z:+.1f}, n = {self.volume:,})")


@dataclass
class Finding:
    id: str
    page: str
    dimension: str        # device | source
    value: str
    flags: list[MetricFlag] = field(default_factory=list)
    context: dict = field(default_factory=dict)
    sizing: dict = field(default_factory=dict)

    @property
    def title(self) -> str:
        who = f"{self.value} visitors" if self.dimension == "device" else f"{self.value.replace('_', ' ')} traffic"
        return f"{self.page.title()} page, {who}"

    @property
    def max_abs_z(self) -> float:
        zs = [abs(f.z) for f in self.flags if f.z is not None]
        return max(zs) if zs else 0.0


def aggregate(web: pd.DataFrame, by: list[str]) -> pd.DataFrame:
    g = web.assign(load_x_sessions=web.avg_load_ms * web.sessions).groupby(by, as_index=False)
    out = g[COUNT_COLUMNS + ["load_x_sessions"]].sum()
    out["avg_load_ms"] = out.load_x_sessions / out.sessions.where(out.sessions > 0)
    return out


def _is_bad(metric: Metric, deviation: float) -> bool:
    return deviation > 0 if metric.bad == "high" else deviation < 0


def standardized_comparison(cells: pd.DataFrame, dim: str, value: str, stratum: str,
                            num: str, den: str) -> tuple[int, int, float, float]:
    """Observed (x, n, rate) for dim == value vs expected rate if each stratum behaved like
    the rest (dim != value). `cells` must be aggregated by [dim, stratum]."""
    slice_ = cells[cells[dim] == value].set_index(stratum)
    rest = cells[cells[dim] != value].groupby(stratum)[[num, den]].sum()
    rest_total = stats.safe_rate(rest[num].sum(), rest[den].sum())
    ref = (rest[num] / rest[den].where(rest[den] > 0)).reindex(slice_.index).fillna(rest_total)
    x, n = int(slice_[num].sum()), int(slice_[den].sum())
    expected = stats.standardized_rate(slice_[den].tolist(), ref.tolist())
    return x, n, stats.safe_rate(x, n), expected


def _rate_flag(check, metric: Metric, x, n, observed, expected, th: Thresholds) -> Optional[MetricFlag]:
    if n < th.min_volume or expected <= 0:
        return None
    deviation = stats.pct_deviation(observed, expected)
    z = stats.one_sample_z(x, n, expected)
    if abs(z) >= th.z and abs(deviation) >= th.min_deviation and _is_bad(metric, deviation):
        return MetricFlag(check, metric.name, metric.label, observed, expected, deviation, z, n, x)
    return None


def divergence_flags(web: pd.DataFrame, th: Thresholds) -> dict[tuple, list[MetricFlag]]:
    """Device and source divergence per page. Keys: (page, dimension, value)."""
    flags: dict[tuple, list[MetricFlag]] = {}
    cells = aggregate(web, ["page", "source", "device"])
    for page, pc in cells.groupby("page"):
        for dim, stratum in (("device", "source"), ("source", "device")):
            for value in sorted(pc[dim].unique()):
                for metric in RATE_METRICS.values():
                    x, n, obs, exp = standardized_comparison(pc, dim, value, stratum,
                                                             metric.numerator, metric.denominator)
                    flag = _rate_flag(dim, metric, x, n, obs, exp, th)
                    if flag:
                        flags.setdefault((page, dim, value), []).append(flag)
    return flags


def load_flags(web: pd.DataFrame, th: Thresholds) -> dict[tuple, list[MetricFlag]]:
    """(page, device) load time vs the same device's other pages."""
    flags: dict[tuple, list[MetricFlag]] = {}
    cells = aggregate(web, ["device", "page"])
    for device, dc in cells.groupby("device"):
        for r in dc.itertuples():
            if r.sessions < th.min_volume:
                continue
            rest = dc[dc.page != r.page]
            baseline = stats.safe_rate(rest.load_x_sessions.sum(), rest.sessions.sum())
            deviation = stats.pct_deviation(r.avg_load_ms, baseline)
            if deviation >= th.load_deviation:
                flags.setdefault((r.page, "device", device), []).append(MetricFlag(
                    "load", "load_ms", LOAD_LABEL, r.avg_load_ms, baseline, deviation, None, int(r.sessions)))
    return flags


def trend_flags(web: pd.DataFrame, th: Thresholds) -> dict[tuple, list[MetricFlag]]:
    """(page, source) first vs last month, flagged when worsening."""
    flags: dict[tuple, list[MetricFlag]] = {}
    months = sorted(web.month.unique())
    if len(months) < 2:
        return flags
    cells = aggregate(web[web.month.isin([months[0], months[-1]])], ["page", "source", "month"])
    for (page, source), g in cells.groupby(["page", "source"]):
        first, last = g.set_index("month").reindex([months[0], months[-1]]).fillna(0).itertuples()
        for metric in RATE_METRICS.values():
            n1, n2 = int(getattr(first, metric.denominator)), int(getattr(last, metric.denominator))
            x1, x2 = int(getattr(first, metric.numerator)), int(getattr(last, metric.numerator))
            if min(n1, n2) < th.trend_min_volume:
                continue
            p1, p2 = stats.safe_rate(x1, n1), stats.safe_rate(x2, n2)
            deviation = stats.pct_deviation(p2, p1)
            z = stats.two_proportion_z(x2, n2, x1, n1)
            if abs(z) >= th.z and abs(deviation) >= th.min_deviation and _is_bad(metric, deviation):
                flags.setdefault((page, "source", source), []).append(MetricFlag(
                    "trend", metric.name, f"{metric.label} trend", p2, p1, deviation, z, n2, x2))
    return flags


def slice_context(web: pd.DataFrame, page: str, dim: str, value: str) -> dict:
    s = web[(web.page == page) & (web[dim] == value)]
    rest = web[(web.page == page) & (web[dim] != value)]
    n_months = max(web.month.nunique(), 1)
    ctx = {
        "sessions": int(s.sessions.sum()),
        "sessions_per_month": int(round(s.sessions.sum() / n_months)),
        "share_of_page_sessions": stats.safe_rate(s.sessions.sum(), web[web.page == page].sessions.sum()),
        "months": n_months,
        "source_mix": (s.groupby("source").sessions.sum() / s.sessions.sum()).round(3).to_dict()
        if s.sessions.sum() else {},
    }
    if s.form_starts.sum():
        ctx["form_start_rate"] = stats.safe_rate(s.form_starts.sum(), s.sessions.sum())
        ctx["form_start_rate_rest"] = stats.safe_rate(rest.form_starts.sum(), rest.sessions.sum())
    return ctx


def detect(web: pd.DataFrame, th: Thresholds = Thresholds()) -> list[Finding]:
    merged: dict[tuple, list[MetricFlag]] = {}
    for part in (divergence_flags(web, th), load_flags(web, th), trend_flags(web, th)):
        for key, flags in part.items():
            merged.setdefault(key, []).extend(flags)
    findings = [
        Finding(id=f"{page}|{dim}={value}", page=page, dimension=dim, value=value, flags=flags,
                context=slice_context(web, page, dim, value))
        for (page, dim, value), flags in merged.items()
    ]
    return sorted(findings, key=lambda f: (-f.max_abs_z, f.id))


# --- Funnel consequence --------------------------------------------------------------------

SIZED_METRICS = ("form_completion", "cta_rate", "exit_rate")


def downstream_rates(funnel: pd.DataFrame) -> pd.DataFrame:
    """Per-source visit→lead, lead→win, and revenue per win from funnel_by_source."""
    g = funnel.groupby("source")[["visits", "leads", "mqls", "sqls", "opps", "wins", "revenue"]].sum()
    return pd.DataFrame({
        "visit_to_lead": g.leads / g.visits,
        "lead_to_win": g.wins / g.leads,
        "revenue_per_win": g.revenue / g.wins.where(g.wins > 0),
    }).fillna(0.0)


def _mix_rate(mix: dict[str, float], rates: pd.Series) -> float:
    total = sum(mix.values())
    return sum(w * rates.get(s, 0.0) for s, w in mix.items()) / total if total else 0.0


def size_finding(finding: Finding, rates: pd.DataFrame, demo_completion_rate: float) -> dict:
    """Monthly lost leads → wins → revenue from the flag with the largest estimated loss.

    Assumptions (shown in the report):
    - form completion: each lost completion is a lost demo request (lead)
    - CTA rate: lost clicks convert at the site-wide demo-page sessions→completion rate
    - exit rate: retained sessions convert at the traffic's visit→lead rate
    Losses from different metrics on the same slice overlap, so only the largest is used.
    """
    mix = finding.context.get("source_mix", {})
    months = finding.context.get("months", 1)
    lead_to_win = _mix_rate(mix, rates.lead_to_win)
    rev_per_win = _mix_rate(mix, rates.revenue_per_win)
    visit_to_lead = _mix_rate(mix, rates.visit_to_lead)
    candidates = []
    for f in finding.flags:
        if f.metric not in SIZED_METRICS or f.check == "trend":
            continue
        gap = abs(f.expected - f.observed) * f.volume / months  # lost events per month
        if f.metric == "form_completion":
            leads, basis = gap, "lost form completions (demo requests)"
        elif f.metric == "cta_rate":
            leads, basis = gap * demo_completion_rate, "lost CTA clicks × demo-page completion rate"
        else:
            leads, basis = gap * visit_to_lead, "excess exits × the traffic's visit→lead rate"
        candidates.append({"metric": f.metric, "basis": basis, "lost_events_per_month": gap,
                           "lost_leads_per_month": leads})
    if not candidates:
        return {"sized": False}
    best = max(candidates, key=lambda c: c["lost_leads_per_month"])
    wins = best["lost_leads_per_month"] * lead_to_win
    return {"sized": True, **best, "lead_to_win": lead_to_win, "revenue_per_win": rev_per_win,
            "lost_wins_per_month": wins, "lost_revenue_per_month": wins * rev_per_win,
            "affected_sessions_per_month": finding.context.get("sessions_per_month", 0)}


def size_all(findings: list[Finding], web: pd.DataFrame, funnel: pd.DataFrame) -> None:
    rates = downstream_rates(funnel)
    demo = web[web.page == "demo"]
    demo_completion_rate = stats.safe_rate(demo.form_completes.sum(), demo.sessions.sum())
    for f in findings:
        f.sizing = size_finding(f, rates, demo_completion_rate)
