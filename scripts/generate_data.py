"""Generate seeded, reproducible synthetic data for EchoAI (fictional B2B SaaS, AI meeting
transcription), covering six months (2026-03 .. 2026-08).

Embedded stories that GIOS modules must later surface:
  1. Paid search -> pricing page: rising pricing-clarity verbatims, rising pricing-page exits
     for paid-search traffic, rising paid-search CAC.
  2. Webinar leads: highest lead volume and strongest Lead->MQL, but the weakest MQL->SQL
     (qualification mismatch; follow-up speed is NOT the problem).
  3. Mobile demo form: high mobile form starts, low mobile completion, slow mobile load.
Red herring: "dark mode / cosmetic" requests are the MOST frequent customer theme but low
severity and concentrated post-purchase.

Usage:  python scripts/generate_data.py [--seed 42] [--out data/synthetic]
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

SEED = 42
MONTHS = pd.period_range("2026-03", "2026-08", freq="M")
SOURCES = ["paid_search", "paid_social", "organic", "webinar", "partner", "email"]
SEGMENTS = ["smb", "mid_market"]
PAGES = ["home", "pricing", "demo", "features", "compare"]
DEVICES = ["desktop", "mobile"]
EVIDENCE_SOURCES = ["interview", "survey", "chat", "sales_call", "support", "win_loss", "review"]

FUNNEL_COLUMNS = ["month", "source", "segment", "visits", "leads", "mqls", "sqls", "opps",
                  "wins", "revenue", "spend"]
WEB_COLUMNS = ["date", "page", "source", "device", "sessions", "exits", "scroll_75", "cta_clicks",
               "form_starts", "form_completes", "rage_clicks", "avg_load_ms"]
EVIDENCE_COLUMNS = ["date", "source", "segment", "journey_stage", "verbatim"]
SALES_COLUMNS = ["lead_id", "source", "rejection_reason", "hours_to_first_follow_up", "routed_to"]
EXPERIMENT_COLUMNS = ["experiment_id", "experiment_name", "hypothesis", "primary_metric",
                      "start_date", "end_date", "variant", "visitors", "conversions", "mqls",
                      "sqls", "opps", "wins"]

# --- Funnel / acquisition assumptions ----------------------------------------------------

DAILY_VISITS = {"paid_search": 1800, "paid_social": 1100, "organic": 2200,
                "webinar": 500, "partner": 250, "email": 500}
MID_MARKET_SHARE = {"paid_search": 0.30, "paid_social": 0.20, "organic": 0.30,
                    "webinar": 0.35, "partner": 0.50, "email": 0.30}
VISIT_TO_LEAD = {"paid_search": 0.032, "paid_social": 0.015, "organic": 0.022,
                 "webinar": 0.120, "partner": 0.050, "email": 0.040}
PAID_SEARCH_LEAD_DECAY = 0.0026  # per month: pricing-page friction erodes lead rate (story 1)
LEAD_TO_MQL = {"paid_search": 0.45, "paid_social": 0.35, "organic": 0.40,
               "webinar": 0.72, "partner": 0.55, "email": 0.42}
MQL_TO_SQL = {"paid_search": 0.42, "paid_social": 0.30, "organic": 0.40,
              "webinar": 0.11, "partner": 0.50, "email": 0.36}
SQL_TO_OPP = {"paid_search": 0.55, "paid_social": 0.50, "organic": 0.55,
              "webinar": 0.45, "partner": 0.60, "email": 0.52}
OPP_TO_WIN = {"smb": 0.26, "mid_market": 0.20}
ACV = {"smb": 6_000, "mid_market": 24_000}
MONTHLY_SPEND = {"paid_search": 95_000, "paid_social": 45_000, "organic": 8_000,
                 "webinar": 18_000, "partner": 12_000, "email": 3_000}
PAID_SEARCH_SPEND_GROWTH = 0.09  # per month: bids rising to defend volume (story 1)
ORGANIC_GROWTH = 0.03

QUALIFICATION_REASONS = ["no_budget", "student_or_researcher", "not_decision_maker", "wrong_use_case"]
REJECTION_MIX = {
    "webinar": {"no_budget": 0.30, "student_or_researcher": 0.20, "not_decision_maker": 0.25,
                "wrong_use_case": 0.10, "no_response": 0.10, "duplicate": 0.05},
    "default": {"no_budget": 0.15, "not_decision_maker": 0.15, "wrong_use_case": 0.10,
                "no_response": 0.30, "too_small": 0.15, "duplicate": 0.10,
                "competitor_contract": 0.05},
}

# --- Customer evidence pools (untagged in the CSV; theme kept only in memory for tests) ----

PRICING_CLARITY = [
    "I couldn't tell what's included in {plan} versus the tier above it.",
    "Your pricing page lists seats and minutes, but I can't work out what we'd actually pay.",
    "Clicked a search ad, landed on pricing, and still had no idea if transcription hours are capped.",
    "Is the AI summary feature part of {plan} or an add-on? The pricing page doesn't say.",
    "We stalled because nobody could explain the difference between {plan} and Enterprise.",
    "The pricing table has too many asterisks. What's actually in {plan}?",
    "I had to book a call just to find out whether {plan} includes the CRM integration.",
    "Pricing felt hidden. I went to compare a cheaper tool because I couldn't estimate cost.",
    "Lost the deal on price confusion: the buyer thought {plan} excluded speaker identification.",
    "Not sure if overage minutes are billed on {plan}. That uncertainty slowed approval.",
    "Came in from Google, read the pricing page twice, and still couldn't tell the plans apart.",
]
WEBINAR_MISMATCH = [
    "I joined the webinar to learn about AI note-taking trends. We have no budget this year.",
    "I'm a grad student researching meeting productivity; the webinar was great for my thesis.",
    "Attended the webinar for ideas, but purchasing decisions sit with IT, not me.",
    "Rep called after the webinar, but we're a two-person team and don't need a sales demo.",
    "Webinar attendee: just exploring, no active project or timeline.",
    "Our team already has a transcription contract through next year; the webinar was for research.",
    "I signed up for the webinar for the free template, not to evaluate tools.",
]
MOBILE_FORM = [
    "Tried to book a demo on my phone and the form just kept spinning.",
    "The demo request form froze on my {phone} after I hit submit.",
    "On mobile the demo page took forever to load, then the date picker wouldn't open.",
    "Started the demo form on my phone twice and gave up; I'll try from a laptop.",
    "Submit button on the demo form did nothing on {phone}. Tapped it several times.",
    "Demo form on mobile kept clearing my company name when I scrolled.",
]
RED_HERRING = [
    "Would love a dark mode for the transcript view.",
    "Dark mode please! My eyes at night.",
    "Minor thing: export file names are long and ugly.",
    "Can you add more color themes? Not a big deal, just a nice-to-have.",
    "The app is great. A dark theme would make it perfect.",
    "Small request: let me pick the font in transcripts.",
    "Would be nice to have emoji reactions on transcript highlights.",
]
NOISE = [
    "Transcription accuracy on accented speakers is impressive.",
    "Do you support Microsoft Teams recordings natively?",
    "Security asked whether you're SOC 2 Type II; we need the report before signing.",
    "Setup took longer than expected because SSO configuration was unclear.",
    "We chose EchoAI because the summaries are better than the tool we used before.",
    "The Zoom integration dropped a recording last week.",
    "Speaker labels get confused when people talk over each other.",
    "We'd buy more seats if there was an admin usage dashboard.",
    "Onboarding videos were helpful, the team was up and running in a day.",
    "Our legal team needs data residency in the EU.",
    "Compared three tools; yours had the cleanest action-item extraction.",
    "Search across past meetings is slow for large workspaces.",
]
PLANS = ["Starter", "Pro", "Business"]
PHONES = ["iPhone", "Android phone", "Pixel", "Galaxy"]

# theme -> monthly counts (6 months), allowed sources, segment mix (smb share), journey stages
EVIDENCE_PLAN = {
    "pricing_clarity": dict(pool=PRICING_CLARITY, monthly=[6, 9, 14, 20, 27, 35],
                            sources=["sales_call", "chat", "survey", "win_loss", "interview"],
                            smb_share=0.75, stages={"evaluation": 0.6, "consideration": 0.25, "purchase": 0.15}),
    "webinar_mismatch": dict(pool=WEBINAR_MISMATCH, monthly=[9, 10, 9, 11, 10, 10],
                             sources=["sales_call", "win_loss", "survey"],
                             smb_share=0.7, stages={"consideration": 0.7, "awareness": 0.3}),
    "mobile_form": dict(pool=MOBILE_FORM, monthly=[5, 6, 5, 7, 6, 6],
                        sources=["chat", "support", "survey"],
                        smb_share=0.65, stages={"evaluation": 0.8, "consideration": 0.2}),
    "dark_mode_red_herring": dict(pool=RED_HERRING, monthly=[44, 46, 45, 47, 45, 46],
                                  sources=["review", "support", "survey", "chat"],
                                  smb_share=0.6, stages={"retention": 0.6, "onboarding": 0.4}),
    "noise": dict(pool=NOISE, monthly=[30, 32, 31, 33, 32, 31],
                  sources=EVIDENCE_SOURCES, smb_share=0.55,
                  stages={"evaluation": 0.25, "onboarding": 0.25, "retention": 0.3, "purchase": 0.2}),
}

EXPERIMENTS = [
    # id, name, hypothesis, metric, start, end, {variant: (visitors, cvr, mql, sql, opp, win)}
    ("EXP-01", "Homepage hero: outcome headline",
     "If we lead with 'never take meeting notes again' instead of feature copy, more evaluators "
     "will request a demo because the value is clearer.",
     "demo_request_rate", "2026-03-02", "2026-03-29",
     {"control": (21000, 0.031, 0.48, 0.40, 0.55, 0.24), "outcome_headline": (21000, 0.036, 0.48, 0.40, 0.55, 0.24)}),
    ("EXP-02", "Webinar registration: 3 fields instead of 7",
     "If we shorten webinar registration, more attendees will register because the form feels lighter.",
     "registration_rate", "2026-04-06", "2026-05-03",
     {"control": (9000, 0.110, 0.72, 0.13, 0.45, 0.22), "short_form": (9000, 0.150, 0.70, 0.08, 0.45, 0.22)}),
    ("EXP-03", "Pricing page FAQ accordion",
     "If we add an FAQ under the pricing table, paid-search visitors will exit less because "
     "common plan questions are answered.",
     "pricing_cta_click_rate", "2026-06-01", "2026-06-14",
     {"control": (2600, 0.080, 0.45, 0.40, 0.55, 0.25), "faq_accordion": (2600, 0.085, 0.45, 0.40, 0.55, 0.25)}),
    ("EXP-04", "Demo page customer logos",
     "If we show customer logos on the demo page, more visitors will start the form because of social proof.",
     "form_start_rate", "2026-05-11", "2026-06-07",
     {"control": (16000, 0.300, 0.45, 0.40, 0.55, 0.24), "logos": (16000, 0.330, 0.42, 0.37, 0.55, 0.24)}),
    ("EXP-05", "Paid social long-form landing page",
     "If paid social lands on a long-form page, visitors will convert better because they arrive cold.",
     "lead_rate", "2026-07-06", "2026-08-02",
     {"control": (15000, 0.016, 0.35, 0.30, 0.50, 0.22), "long_form": (15000, 0.012, 0.35, 0.30, 0.50, 0.22)}),
    ("EXP-06", "Nurture email CTA: 2-minute tour vs book demo",
     "If nurture emails offer a 2-minute product tour instead of a demo, more leads will engage and "
     "later qualify because the ask is smaller.",
     "click_to_lead_rate", "2026-07-13", "2026-08-16",
     {"control": (12000, 0.040, 0.42, 0.36, 0.52, 0.24), "product_tour": (12000, 0.052, 0.46, 0.40, 0.52, 0.24)}),
]


def _noise(rng: np.random.Generator, size=None, sd: float = 0.05):
    return np.clip(rng.normal(1.0, sd, size), 0.7, 1.3)


def _stage(rng: np.random.Generator, n: int, rate: float, sd: float = 0.06) -> int:
    """Downstream stage count: expected value with mild multiplicative noise.

    Chained binomials on small monthly counts (wins) bury the stories in sampling noise,
    so later stages use expected value x noise, capped at the stage above.
    """
    return int(min(n, round(n * rate * float(_noise(rng, sd=sd)))))


# --- Generators ----------------------------------------------------------------------------


def generate_funnel(rng: np.random.Generator) -> pd.DataFrame:
    rows = []
    for m, month in enumerate(MONTHS):
        for source in SOURCES:
            source_visits = DAILY_VISITS[source] * month.days_in_month
            if source == "organic":
                source_visits *= (1 + ORGANIC_GROWTH) ** m
            spend_total = MONTHLY_SPEND[source]
            if source == "paid_search":
                spend_total *= (1 + PAID_SEARCH_SPEND_GROWTH) ** m
            for segment in SEGMENTS:
                share = MID_MARKET_SHARE[source] if segment == "mid_market" else 1 - MID_MARKET_SHARE[source]
                visits = int(source_visits * share * _noise(rng))
                lead_rate = VISIT_TO_LEAD[source]
                if source == "paid_search":
                    lead_rate -= PAID_SEARCH_LEAD_DECAY * m
                mql_rate = LEAD_TO_MQL[source] + (0.03 if segment == "mid_market" else 0)
                leads = rng.binomial(visits, lead_rate)
                mqls = rng.binomial(leads, mql_rate)
                sqls = _stage(rng, mqls, MQL_TO_SQL[source])
                opps = _stage(rng, sqls, SQL_TO_OPP[source])
                wins = _stage(rng, opps, OPP_TO_WIN[segment])
                revenue = int(round(sum(ACV[segment] * _noise(rng, wins, sd=0.12))))
                spend = int(round(spend_total * share * _noise(rng, sd=0.03)))
                rows.append([str(month), source, segment, visits, leads, mqls, sqls, opps, wins,
                             revenue, spend])
    return pd.DataFrame(rows, columns=FUNNEL_COLUMNS)


def generate_web_behavior(rng: np.random.Generator) -> pd.DataFrame:
    dates = pd.date_range(MONTHS[0].start_time, MONTHS[-1].end_time.normalize(), freq="D")
    grid = pd.MultiIndex.from_product([dates, PAGES, SOURCES, DEVICES],
                                      names=["date", "page", "source", "device"]).to_frame(index=False)
    n = len(grid)
    month_idx = ((grid["date"].dt.year - 2026) * 12 + grid["date"].dt.month - MONTHS[0].month).to_numpy()
    is_paid_search = (grid["source"] == "paid_search").to_numpy()
    is_mobile = (grid["device"] == "mobile").to_numpy()
    page = grid["page"].to_numpy()
    is_pricing, is_demo = page == "pricing", page == "demo"

    page_share = {"home": 0.35, "pricing": 0.20, "demo": 0.15, "features": 0.20, "compare": 0.10}
    ps_page_share = {"home": 0.25, "pricing": 0.30, "demo": 0.20, "features": 0.15, "compare": 0.10}
    mobile_share = {"paid_social": 0.60, "email": 0.45}
    base = grid["source"].map(DAILY_VISITS).to_numpy(dtype=float)
    shares = np.where(is_paid_search, grid["page"].map(ps_page_share), grid["page"].map(page_share))
    mshare = grid["source"].map(lambda s: mobile_share.get(s, 0.35)).to_numpy()
    dshare = np.where(is_mobile, mshare, 1 - mshare)
    weekday = np.where(grid["date"].dt.dayofweek.to_numpy() >= 5, 0.45, 1.15)
    sessions = rng.poisson(base * shares * dshare * weekday * _noise(rng, n, sd=0.08))

    exit_base = grid["page"].map({"home": 0.40, "pricing": 0.40, "demo": 0.30,
                                  "features": 0.35, "compare": 0.45}).to_numpy()
    # Story 1: paid-search pricing exits climb ~4.5pp per month (0.46 -> ~0.69)
    exit_rate = np.where(is_paid_search & is_pricing, 0.46 + 0.045 * month_idx, exit_base)
    exit_rate = np.clip(exit_rate + rng.normal(0, 0.02, n), 0.05, 0.95)
    cta_rate = np.where(is_paid_search & is_pricing, 0.10 - 0.008 * month_idx, 0.12)
    form_start_rate = np.where(is_demo, np.where(is_mobile, 0.36, 0.30), 0.0)  # story 3
    complete_rate = np.where(is_mobile, 0.22, 0.62)  # story 3
    rage_rate = np.where(is_demo & is_mobile, 0.030, 0.004)
    load_base = np.where(is_mobile, np.where(is_demo, 4800, 2100), np.where(is_demo, 1400, 1300))

    exits = rng.binomial(sessions, exit_rate)
    form_starts = rng.binomial(sessions, form_start_rate)
    grid["date"] = grid["date"].dt.strftime("%Y-%m-%d")
    grid["sessions"] = sessions
    grid["exits"] = exits
    grid["scroll_75"] = rng.binomial(sessions - exits, 0.45)
    grid["cta_clicks"] = rng.binomial(sessions, np.clip(cta_rate, 0.01, 1))
    grid["form_starts"] = form_starts
    grid["form_completes"] = rng.binomial(form_starts, complete_rate)
    grid["rage_clicks"] = rng.poisson(sessions * rage_rate)
    grid["avg_load_ms"] = np.round(load_base * _noise(rng, n, sd=0.08)).astype(int)
    return grid[WEB_COLUMNS]


def generate_customer_evidence(rng: np.random.Generator) -> pd.DataFrame:
    """Return evidence with a hidden `_theme` column (dropped before writing the CSV)."""
    rows = []
    for theme, spec in EVIDENCE_PLAN.items():
        stages, stage_p = list(spec["stages"]), list(spec["stages"].values())
        for month, count in zip(MONTHS, spec["monthly"]):
            for _ in range(count):
                day = int(rng.integers(1, month.days_in_month + 1))
                text = str(rng.choice(spec["pool"])).format(
                    plan=rng.choice(PLANS), phone=rng.choice(PHONES))
                rows.append([
                    f"{month}-{day:02d}",
                    str(rng.choice(spec["sources"])),
                    "smb" if rng.random() < spec["smb_share"] else "mid_market",
                    str(rng.choice(stages, p=stage_p)),
                    text,
                    theme,
                ])
    df = pd.DataFrame(rows, columns=EVIDENCE_COLUMNS + ["_theme"])
    return df.sort_values(["date", "source", "verbatim"], kind="stable").reset_index(drop=True)


def generate_sales_feedback(rng: np.random.Generator, funnel: pd.DataFrame) -> pd.DataFrame:
    """One row per MQL handed to sales; accepted rows (SQLs) have no rejection reason."""
    rows = []
    for rec in funnel.itertuples(index=False):
        mix = REJECTION_MIX.get(rec.source, REJECTION_MIX["default"])
        reasons = rng.choice(list(mix), size=rec.mqls - rec.sqls, p=list(mix.values()))
        outcomes = [""] * rec.sqls + [str(r) for r in reasons]
        rng.shuffle(outcomes)
        for reason in outcomes:
            if rec.source == "partner":
                routed, hours = "partner_team", rng.lognormal(np.log(20), 0.5)
            elif rng.random() < 0.02:
                routed, hours = "unassigned", rng.lognormal(np.log(60), 0.4)
            else:
                routed = "smb_sdr" if rec.segment == "smb" else "mid_market_ae"
                hours = rng.lognormal(np.log(5), 0.6)
            rows.append([rec.source, reason, round(float(hours), 1), routed])
    df = pd.DataFrame(rows, columns=SALES_COLUMNS[1:])
    df.insert(0, "lead_id", [f"L-{i:06d}" for i in range(1, len(df) + 1)])
    return df


def generate_experiments(rng: np.random.Generator) -> pd.DataFrame:
    rows = []
    for exp_id, name, hyp, metric, start, end, variants in EXPERIMENTS:
        for variant, (visitors, cvr, mql, sql, opp, win) in variants.items():
            conversions = rng.binomial(visitors, cvr)
            mqls = rng.binomial(conversions, mql)
            sqls = rng.binomial(mqls, sql)
            opps = rng.binomial(sqls, opp)
            wins = rng.binomial(opps, win)
            rows.append([exp_id, name, hyp, metric, start, end, variant, visitors, conversions,
                         mqls, sqls, opps, wins])
    return pd.DataFrame(rows, columns=EXPERIMENT_COLUMNS)


def generate_all(seed: int = SEED) -> dict[str, pd.DataFrame]:
    """All datasets keyed by file stem. customer_evidence keeps its hidden `_theme` column."""
    rng = np.random.default_rng(seed)
    funnel = generate_funnel(rng)
    return {
        "funnel_by_source": funnel,
        "web_behavior": generate_web_behavior(rng),
        "customer_evidence": generate_customer_evidence(rng),
        "sales_feedback": generate_sales_feedback(rng, funnel),
        "experiments": generate_experiments(rng),
    }


def write_all(out_dir: Path, seed: int = SEED) -> dict[str, Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = {}
    for name, df in generate_all(seed).items():
        public = df.drop(columns=[c for c in df.columns if c.startswith("_")])
        path = out_dir / f"{name}.csv"
        public.to_csv(path, index=False)
        paths[name] = path
    return paths


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--out", type=Path, default=Path(__file__).resolve().parent.parent / "data" / "synthetic")
    args = parser.parse_args()
    for name, path in write_all(args.out, args.seed).items():
        print(f"wrote {path} ({sum(1 for _ in path.open()) - 1} rows)")


if __name__ == "__main__":
    main()
