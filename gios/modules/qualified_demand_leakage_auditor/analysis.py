"""Fully deterministic leakage analysis (qualified-demand-leakage-auditor.md).

Stage map Visit→Lead→MQL→SQL→Opp→Win per source and source×segment, Wilson 95% CIs,
benchmarks pooled from the other sources, rule-based flags, opportunity sizing, and owner
classification. The LLM only writes narrative around these results.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import pandas as pd

from gios.core import stats

MODULE = "qualified_demand_leakage_auditor"
COUNTS = ["visits", "leads", "mqls", "sqls", "opps", "wins"]
# (denominator, numerator, name)
STAGES = [
    ("visits", "leads", "Visit→Lead"),
    ("leads", "mqls", "Lead→MQL"),
    ("mqls", "sqls", "MQL→SQL"),
    ("sqls", "opps", "SQL→Opp"),
    ("opps", "wins", "Opp→Win"),
]
SUMMARY_STAGE = ("leads", "wins", "Lead→Win")
DOWNSTREAM = {"MQL→SQL", "SQL→Opp", "Opp→Win"}
# The funnel stage each flag is about (used by downstream modules to place experiments).
FLAG_STAGE = {"qualification_mismatch": "MQL→SQL", "strong_top_weak_downstream": "MQL→SQL",
              "sla_loss": "MQL→SQL", "high_volume_low_quality": "Lead→MQL", "rising_cac": "Visit→Lead",
              "routing_loss": "MQL→SQL"}
QUALIFICATION_REASONS = {"no_budget", "student_or_researcher", "not_decision_maker", "wrong_use_case"}
STAGE_JOURNEY = {"Visit→Lead": "consideration", "Lead→MQL": "consideration", "MQL→SQL": "evaluation",
                 "SQL→Opp": "evaluation", "Opp→Win": "purchase", "Lead→Win": "evaluation"}

# Owner categories (spec step 5)
OWNERS = {
    "acquisition": "Acquisition", "web_conversion": "Web / conversion", "qualification": "Qualification",
    "operations_routing": "Operations / routing", "sales_handoff": "Sales handoff",
    "product_offering": "Product / offering", "measurement": "Measurement",
}


@dataclass(frozen=True)
class Settings:
    min_volume: int = 30              # stage denominator needed before flagging
    gap_closure: float = 0.50         # share of the gap to benchmark closed when sizing
    sla_hours: float = 24.0
    sla_share_threshold: float = 0.20  # share of MQLs followed up after the SLA
    qualification_share_threshold: float = 0.50  # share of rejections with qualification reasons
    cac_rise_threshold: float = 0.25  # CAC in the later half vs the earlier half of the period


@dataclass
class Flag:
    kind: str           # high_volume_low_quality | strong_top_weak_downstream | sla_loss | routing_loss | qualification_mismatch | rising_cac
    source: str
    owner: str
    evidence: str
    details: dict = field(default_factory=dict)

    @property
    def label(self) -> str:
        return {
            "high_volume_low_quality": "High-volume / low-quality source",
            "strong_top_weak_downstream": "Strong top-funnel, weak downstream",
            "sla_loss": "Follow-up SLA loss",
            "routing_loss": "Routing loss",
            "qualification_mismatch": "Qualification mismatch",
            "rising_cac": "Rising CAC",
        }[self.kind]


def _rows(group: pd.DataFrame, keys: dict, stages) -> list[dict]:
    out = []
    for den, num, name in stages:
        x, n = int(group[num].sum()), int(group[den].sum())
        lo, hi = stats.wilson_ci(x, n)
        out.append({**keys, "stage": name, "numerator": x, "denominator": n,
                    "rate": stats.safe_rate(x, n), "ci_low": lo, "ci_high": hi})
    return out


def stage_map(funnel: pd.DataFrame, by: list[str], min_volume: int = 30) -> pd.DataFrame:
    """Conversion per stage for each group with Wilson CIs and a benchmark pooled from the
    other groups (within the same segment when grouping by segment)."""
    stages = STAGES + [SUMMARY_STAGE]
    rows = []
    for keys, g in funnel.groupby(by):
        keys = dict(zip(by, keys if isinstance(keys, tuple) else (keys,)))
        rows += _rows(g, keys, stages)
    df = pd.DataFrame(rows)
    peer_keys = [k for k in by if k != "source"]
    bench = []
    for r in df.itertuples():
        peers = funnel[funnel.source != r.source]
        for k in peer_keys:
            peers = peers[peers[k] == getattr(r, k)]
        den, num = next((d, n) for d, n, name in stages if name == r.stage)
        bench.append(stats.safe_rate(peers[num].sum(), peers[den].sum()))
    df["benchmark"] = bench
    df["sufficient"] = df.denominator >= min_volume
    df["status"] = "in line"
    df.loc[df.sufficient & (df.ci_high < df.benchmark), "status"] = "below"
    df.loc[df.sufficient & (df.ci_low > df.benchmark), "status"] = "above"
    df.loc[~df.sufficient, "status"] = "insufficient volume"
    df["vs_benchmark"] = df.rate / df.benchmark.where(df.benchmark > 0) - 1
    return df


def economics(funnel: pd.DataFrame) -> pd.DataFrame:
    g = funnel.groupby("source")[COUNTS + ["revenue", "spend"]].sum()
    per = lambda col: g.spend / g[col].where(g[col] > 0)  # noqa: E731
    return pd.DataFrame({
        "spend": g.spend, "revenue": g.revenue, "cpl": per("leads"), "cost_per_mql": per("mqls"),
        "cost_per_sql": per("sqls"), "cost_per_opp": per("opps"), "cac": per("wins"),
        "roas": g.revenue / g.spend.where(g.spend > 0),
        "revenue_per_win": g.revenue / g.wins.where(g.wins > 0),
        "lead_share": g.leads / g.leads.sum(),
    })


def cac_trend(funnel: pd.DataFrame) -> pd.DataFrame:
    """Per source: CAC (spend / wins) in the earlier vs later half of the period."""
    months = sorted(funnel.month.unique())
    half = len(months) // 2
    early, late = months[:half], months[len(months) - half:]
    rows = []
    for source, g in funnel.groupby("source"):
        e, l = g[g.month.isin(early)], g[g.month.isin(late)]
        cac_e = stats.safe_rate(e.spend.sum(), e.wins.sum())
        cac_l = stats.safe_rate(l.spend.sum(), l.wins.sum())
        rows.append({"source": source, "early_months": f"{early[0]}..{early[-1]}" if early else "",
                     "late_months": f"{late[0]}..{late[-1]}" if late else "",
                     "early_wins": int(e.wins.sum()), "late_wins": int(l.wins.sum()),
                     "early_cac": cac_e, "late_cac": cac_l,
                     "change": stats.pct_deviation(cac_l, cac_e) if cac_e else 0.0,
                     "early_spend": float(e.spend.sum()), "late_spend": float(l.spend.sum())})
    return pd.DataFrame(rows).set_index("source")


def follow_up(sales: pd.DataFrame, settings: Settings) -> pd.DataFrame:
    """Per source: follow-up timing vs SLA and acceptance within vs beyond SLA."""
    s = sales.assign(late=sales.hours_to_first_follow_up > settings.sla_hours,
                     accepted=sales.rejection_reason == "")
    rows = []
    fu_cols = ["source", "mqls", "median_hours", "late_share", "late", "accept_on_time", "accept_late",
               "acceptance_z", "unassigned"]
    for source, g in s.groupby("source"):
        on, late = g[~g.late], g[g.late]
        rows.append({
            "source": source, "mqls": len(g), "median_hours": g.hours_to_first_follow_up.median(),
            "late_share": g.late.mean(), "late": int(g.late.sum()),
            "accept_on_time": stats.safe_rate(on.accepted.sum(), len(on)),
            "accept_late": stats.safe_rate(late.accepted.sum(), len(late)),
            "acceptance_z": stats.two_proportion_z(int(late.accepted.sum()), len(late),
                                                   int(on.accepted.sum()), len(on)),
            "unassigned": int((g.routed_to == "unassigned").sum()),
        })
    return pd.DataFrame(rows, columns=fu_cols).set_index("source")


def rejection_mix(sales: pd.DataFrame) -> pd.DataFrame:
    rejected = sales[sales.rejection_reason != ""]
    if rejected.empty:
        return pd.DataFrame(columns=["rejections", "qualification_share"], index=pd.Index([], name="source"))
    mix = pd.crosstab(rejected.source, rejected.rejection_reason)
    mix["rejections"] = mix.sum(axis=1)
    quals = [c for c in mix.columns if c in QUALIFICATION_REASONS]
    mix["qualification_share"] = mix[quals].sum(axis=1) / mix.rejections
    return mix


def _get(sm: pd.DataFrame, source: str, stage: str) -> pd.Series:
    return sm[(sm.source == source) & (sm.stage == stage)].iloc[0]


def detect_flags(sm: pd.DataFrame, econ: pd.DataFrame, fu: pd.DataFrame, rej: pd.DataFrame,
                 settings: Settings, cac: Optional[pd.DataFrame] = None) -> list[Flag]:
    flags: list[Flag] = []
    n_sources = sm.source.nunique()
    for source in sorted(sm.source.unique()):
        l2w = _get(sm, source, "Lead→Win")
        if econ.loc[source, "lead_share"] >= 1 / n_sources and l2w.status == "below":
            flags.append(Flag("high_volume_low_quality", source, "acquisition",
                              f"{econ.loc[source, 'lead_share']:.0%} of all leads but Lead→Win {l2w.rate:.1%} "
                              f"[{l2w.ci_low:.1%}–{l2w.ci_high:.1%}] vs {l2w.benchmark:.1%} for other sources",
                              {"lead_share": econ.loc[source, "lead_share"], "lead_to_win": l2w.rate,
                               "benchmark": l2w.benchmark}))
        top = _get(sm, source, "Lead→MQL")
        weak = [_get(sm, source, s) for s in DOWNSTREAM]
        weak = [w for w in weak if w.status == "below"]
        if top.status == "above" and weak:
            w = min(weak, key=lambda r: r.vs_benchmark)
            flags.append(Flag("strong_top_weak_downstream", source, "qualification",
                              f"Lead→MQL {top.rate:.1%} vs {top.benchmark:.1%} benchmark, but {w.stage} "
                              f"{w.rate:.1%} [{w.ci_low:.1%}–{w.ci_high:.1%}] vs {w.benchmark:.1%}",
                              {"top_rate": top.rate, "weak_stage": w.stage, "weak_rate": w.rate,
                               "weak_benchmark": w.benchmark}))
        if source in fu.index:
            f = fu.loc[source]
            if f.mqls >= settings.min_volume and f.late_share >= settings.sla_share_threshold:
                impact = ("acceptance is lower when late" if f.acceptance_z <= -2
                          else "no measurable acceptance difference when late")
                flags.append(Flag("sla_loss", source, "operations_routing",
                                  f"{f.late_share:.0%} of MQLs followed up after {settings.sla_hours:g}h "
                                  f"(median {f.median_hours:.1f}h); {impact} "
                                  f"({f.accept_late:.1%} vs {f.accept_on_time:.1%}, z = {f.acceptance_z:+.1f})",
                                  {"late_share": f.late_share, "median_hours": f.median_hours,
                                   "acceptance_z": f.acceptance_z}))
        if source in rej.index:
            r = rej.loc[source]
            m2s = _get(sm, source, "MQL→SQL")
            if (r.rejections >= settings.min_volume and m2s.status == "below"
                    and r.qualification_share >= settings.qualification_share_threshold):
                top_reasons = r.drop(["rejections", "qualification_share"]).astype(int).sort_values(ascending=False)
                reasons = ", ".join(f"{k} {v / r.rejections:.0%}" for k, v in top_reasons.head(3).items())
                flags.append(Flag("qualification_mismatch", source, "qualification",
                                  f"{r.qualification_share:.0%} of {int(r.rejections):,} sales rejections cite "
                                  f"qualification reasons ({reasons}); MQL→SQL {m2s.rate:.1%} vs {m2s.benchmark:.1%}",
                                  {"qualification_share": r.qualification_share, "rejections": int(r.rejections)}))
        if cac is not None and source in cac.index:
            c = cac.loc[source]
            if (min(c.early_wins, c.late_wins) >= settings.min_volume
                    and c.change >= settings.cac_rise_threshold):
                spend_change = stats.pct_deviation(c.late_spend, c.early_spend)
                wins_change = stats.pct_deviation(c.late_wins, c.early_wins)
                flags.append(Flag("rising_cac", source, "acquisition",
                                  f"CAC ${c.late_cac:,.0f} in {c.late_months} vs ${c.early_cac:,.0f} in "
                                  f"{c.early_months} ({c.change:+.0%}); spend {spend_change:+.0%}, wins {wins_change:+.0%}",
                                  {"early_cac": c.early_cac, "late_cac": c.late_cac, "change": c.change}))
    unassigned = int(fu.unassigned.sum()) if len(fu) else 0
    if unassigned >= settings.min_volume:
        flags.append(Flag("routing_loss", "all", "operations_routing",
                          f"{unassigned:,} MQLs ({unassigned / fu.mqls.sum():.1%}) were never routed to an owner",
                          {"unassigned": unassigned}))
    return flags


def owner_for(source: str, stage: str, flags: list[Flag], sufficient: bool = True) -> str:
    """Spec step 5: classify the most likely owner of a leaking stage."""
    if not sufficient:
        return "measurement"
    kinds = {f.kind for f in flags if f.source == source}
    if stage == "Visit→Lead":
        return "web_conversion"
    if stage == "Lead→MQL":
        return "acquisition"
    if stage == "MQL→SQL":
        if "qualification_mismatch" in kinds:
            return "qualification"
        if "sla_loss" in kinds:
            return "operations_routing"
        return "sales_handoff"
    if stage == "SQL→Opp":
        return "sales_handoff"
    if stage == "Opp→Win":
        return "product_offering"
    return "measurement"


def size_leaks(sm: pd.DataFrame, econ: pd.DataFrame, flags: list[Flag], settings: Settings,
               months: int) -> pd.DataFrame:
    """For every below-benchmark stage with enough volume: close `gap_closure` of the gap to
    benchmark and propagate the extra conversions through the source's own downstream rates."""
    rows = []
    names = [name for _, _, name in STAGES]
    for r in sm[(sm.stage.isin(names)) & (sm.status == "below")].itertuples():
        lift = settings.gap_closure * (r.benchmark - r.rate)
        extra = r.denominator * lift
        downstream = 1.0
        for later in names[names.index(r.stage) + 1:]:
            downstream *= _get(sm, r.source, later).rate
        wins = extra * downstream
        rpw = econ.loc[r.source, "revenue_per_win"]
        rpw = 0.0 if pd.isna(rpw) else rpw
        rows.append({
            "source": r.source, "stage": r.stage, "rate": r.rate, "ci_low": r.ci_low, "ci_high": r.ci_high,
            "benchmark": r.benchmark, "volume": r.denominator, "vs_benchmark": r.vs_benchmark,
            "lifted_rate": r.rate + lift, "extra_conversions": extra, "extra_wins": wins,
            "extra_revenue": wins * rpw,
            "extra_conversions_per_month": extra / months, "extra_wins_per_month": wins / months,
            "extra_revenue_per_month": wins * rpw / months,
            "owner": owner_for(r.source, r.stage, flags),
        })
    cols = ["source", "stage", "rate", "ci_low", "ci_high", "benchmark", "volume", "vs_benchmark", "lifted_rate",
            "extra_conversions", "extra_wins", "extra_revenue", "extra_conversions_per_month",
            "extra_wins_per_month", "extra_revenue_per_month", "owner"]
    out = pd.DataFrame(rows, columns=cols)
    return out.sort_values(["extra_revenue", "source"], ascending=[False, True]).reset_index(drop=True)


@dataclass
class LeakageAnalysis:
    settings: Settings
    stage_map: pd.DataFrame
    segment_stage_map: pd.DataFrame
    economics: pd.DataFrame
    follow_up: pd.DataFrame
    rejections: pd.DataFrame
    flags: list[Flag]
    leaks: pd.DataFrame
    months: int
    cac: Optional[pd.DataFrame] = None
    period: str = ""

    @property
    def top_leak(self) -> Optional[pd.Series]:
        return None if self.leaks.empty else self.leaks.iloc[0]

    def owners(self) -> list[str]:
        owners = []
        if self.top_leak is not None:
            owners.append(self.top_leak.owner)
            top_kinds = [f for f in self.flags if f.source == self.top_leak.source]
            owners += [f.owner for f in top_kinds]
        owners += [f.owner for f in self.flags]
        return list(dict.fromkeys(owners))


def analyze(funnel: pd.DataFrame, sales: pd.DataFrame, settings: Settings = Settings()) -> LeakageAnalysis:
    sm = stage_map(funnel, ["source"], settings.min_volume)
    seg = stage_map(funnel, ["source", "segment"], settings.min_volume)
    econ = economics(funnel)
    fu = follow_up(sales, settings)
    rej = rejection_mix(sales)
    cac = cac_trend(funnel)
    flags = detect_flags(sm, econ, fu, rej, settings, cac)
    months = max(funnel.month.nunique(), 1)
    leaks = size_leaks(sm, econ, flags, settings, months)
    period = f"{funnel.month.min()}..{funnel.month.max()}"
    return LeakageAnalysis(settings, sm, seg, econ, fu, rej, flags, leaks, months, cac, period)
