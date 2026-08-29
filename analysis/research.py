"""Descriptive research summaries separate from hard acceptance checks."""

from analysis.statistics import distribution_summary, safe_mean


def build_research_summary(context):
    households = context.tables["household_snapshot"]
    firms = context.tables["weekly_firm"]
    final_step = max((row.get("step") for row in firms), default=None)
    final_firms = [row for row in firms if row.get("step") == final_step]
    wealth = [float(row["wealth"]) for row in households if row.get("wealth") not in (None, "NA")]
    shares = [float(row["market_share"]) for row in final_firms if row.get("market_share") not in (None, "NA")]
    return {
        "household_wealth_distribution": distribution_summary(wealth),
        "firm_count_final": len(final_firms),
        "firm_market_share_mean": safe_mean(shares),
        "firm_price_dispersion": distribution_summary([float(row["price"]) for row in final_firms if row.get("price") not in (None, "NA")]),
        "credit_binding_firm_weeks": sum(str(row.get("financing_state")) in {"2", "3", "2.0", "3.0"} for row in firms),
        "interest_shortfall_firm_weeks": sum(float(row.get("current_interest_unpaid", 0) or 0) > 1e-9 for row in firms),
    }


def write_research(context, summary):
    directory = context.analysis_dir / "research"
    directory.mkdir(parents=True, exist_ok=True)
    lines = ["# Analysis v2 Research Summary", "", "These are descriptive research metrics, not economic acceptance verdicts.", ""]
    for key, value in summary.items():
        lines.append(f"- **{key}**: `{value}`")
    (directory / "research_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
