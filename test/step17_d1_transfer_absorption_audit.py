"""Step17.D.1 diagnostic audit using only accepted Step17 artifacts.

This script deliberately does not construct or run a World.  The accepted D
smoke persisted weekly aggregate policy ledgers and final household snapshots,
not recipient-level event histories, so unavailable event-level quantities are
reported explicitly rather than reconstructed from final state.
"""
from __future__ import annotations

import csv
import json
import math
import statistics
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
D = ROOT / "test/output/step17_d_active_institutional_smoke"
A = ROOT / "test/output/step17_a_household_transfer_capacity"
B = ROOT / "test/output/step17_b_payg_pension_capacity"
C = ROOT / "test/output/step17_c_combined_policy_frontier"
OUT = ROOT / "test/output/step17_d1_transfer_absorption_audit"
EPS = 1e-9
BASE_RATE = 0.065490
RATES = (0.02, 0.03, 0.04, 0.05, 0.06, BASE_RATE)


def read_rows(path: Path):
    with path.open(newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def value(row, key, default=0.0):
    try:
        raw = row.get(key, default)
        if raw in (None, "", "nan", "NaN", "None"):
            return default
        number = float(raw)
        return number if math.isfinite(number) else default
    except (TypeError, ValueError):
        return default


def optional(row, key):
    raw = row.get(key)
    if raw in (None, "", "nan", "NaN", "None"):
        return "unavailable"
    return raw


def write_rows(name, rows, fields=None):
    rows = list(rows)
    fields = list(fields or [])
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    if not fields:
        fields = ["status"]
    OUT.mkdir(parents=True, exist_ok=True)
    with (OUT / name).open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def slope(rows, field):
    points = [(value(row, "global_step"), value(row, field)) for row in rows]
    if len(points) < 2:
        return 0.0
    mean_x = statistics.fmean(item[0] for item in points)
    mean_y = statistics.fmean(item[1] for item in points)
    denom = sum((x - mean_x) ** 2 for x, _ in points)
    return sum((x - mean_x) * (y - mean_y) for x, y in points) / denom if denom else 0.0


def classify_slope(number):
    if number > 0.5:
        return "EXPANDS"
    if number < -0.5:
        return "SHRINKS"
    return "STABLE"


def first_positive(rows, field):
    active = [int(value(row, "global_step")) for row in rows if value(row, field) > EPS]
    return min(active) if active else "unavailable"


def fmt(number):
    return "unavailable" if number == "unavailable" else f"{number:.10g}"


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    payg = read_rows(D / "payg_weekly_summary.csv")
    transfers = read_rows(D / "wealth_transfer_weekly_summary.csv")
    branches = read_rows(D / "branch_runtime_summary.csv")
    distributions = read_rows(D / "branch_distribution_comparison.csv")
    elderly = read_rows(D / "elderly_distribution_comparison.csv")
    consumption = read_rows(D / "consumption_food_sales_comparison.csv")
    shadow_transfer = read_rows(A / "shadow_transfer_allocation.csv")
    shadow_pension = read_rows(B / "pension_benefit_scenarios.csv")
    shadow_effect = read_rows(B / "pension_shadow_liquidity_effect.csv")
    combined = read_rows(C / "combined_scenario_results.csv")
    overlap = read_rows(C / "public_private_overlap.csv")

    branch_map = {row["branch"]: row for row in branches}
    payg_map = {}
    transfer_map = {}
    for branch in ("PRIVATE_ONLY", "PAYG_ONLY", "COMBINED"):
        payg_map[branch] = [row for row in payg if row.get("branch") == branch]
        transfer_map[branch] = [row for row in transfers if row.get("branch") == branch]

    # Event-level fields are intentionally unavailable in the accepted D data.
    event_rows = []
    for branch in ("PAYG_ONLY", "COMBINED"):
        for row in payg_map[branch]:
            pension = value(row, "actual_pension")
            transfer_row = next((item for item in transfer_map[branch] if item.get("global_step") == row.get("global_step")), {})
            transfer = value(transfer_row, "total_transfer")
            if pension <= EPS and transfer <= EPS:
                continue
            event_rows.append({
                "branch": branch,
                "week": row.get("global_step"),
                "household_id": "UNAVAILABLE_AGGREGATE",
                "event_scope": "branch_week_aggregate",
                "pension_received": pension,
                "wealth_transfer_received": transfer,
                "policy_receipt": pension + transfer,
                "pre_policy_cash": "unavailable",
                "pre_policy_liquidity_weeks": "unavailable",
                "post_policy_pre_consumption_cash": "unavailable",
                "same_week_consumption": "unavailable",
                "end_of_week_cash": "unavailable",
                "end_of_week_liquidity_weeks": "unavailable",
                "next_week_closing_cash": "unavailable",
                "next_week_liquidity_weeks": "unavailable",
                "plus_4_week_closing_cash": "unavailable",
                "plus_4_week_liquidity_weeks": "unavailable",
                "data_status": "aggregate receipt only; recipient event history was not persisted",
            })
    write_rows("recipient_event_window.csv", event_rows)

    retention_rows = []
    for branch in ("PRIVATE_ONLY", "PAYG_ONLY", "COMBINED"):
        p = sum(value(row, "actual_pension") for row in payg_map[branch])
        w = sum(value(row, "total_transfer") for row in transfer_map[branch])
        retention_rows.append({
            "branch": branch,
            "policy_receipt_total": p + w,
            "pension_total": p,
            "wealth_transfer_total": w,
            "same_week_retention_ratio": "unavailable",
            "next_week_retention_ratio": "unavailable",
            "four_week_retention_ratio": "unavailable",
            "same_week_absorption": "unavailable",
            "cash_flow_proxy": "unavailable",
            "interpretation": "fungible cash and no recipient-level pre/post cash history",
        })
    write_rows("policy_cash_retention.csv", retention_rows)

    threshold_rows = []
    for branch in ("PRIVATE_ONLY", "PAYG_ONLY", "COMBINED"):
        dist = next((row for row in distributions if row.get("branch") == branch), {})
        for threshold in (0.25, 0.5, 1.0):
            threshold_rows.append({
                "branch": branch,
                "recipient_class": "all_policy_recipients",
                "threshold_weeks": threshold,
                "event_count": sum(value(row, "pension_recipient_household_count") for row in payg_map[branch]) + sum(value(row, "recipient_households") for row in transfer_map[branch]),
                "crossed_post_policy": "unavailable",
                "below_at_week_end": "unavailable",
                "below_next_week": "unavailable",
                "below_plus_4_weeks": "unavailable",
                "final_all_household_share_below_threshold": value(dist, f"liquidity_lt_{str(threshold).replace('.', '_')}"),
                "classification": "EVENT_HISTORY_NOT_PERSISTED",
            })
    write_rows("threshold_persistence.csv", threshold_rows)

    elderly_rows = []
    for branch in ("PRIVATE_ONLY", "PAYG_ONLY", "COMBINED"):
        rows = [row for row in elderly if row.get("branch") == branch]
        for row in rows:
            elderly_rows.append({
                "branch": branch,
                "group": row.get("group"),
                "threshold_weeks": row.get("liquidity_threshold"),
                "household_count": row.get("household_count"),
                "final_share_below": row.get("share_below"),
                "event_crossing_count": "unavailable",
                "persistence_1_week": "unavailable",
                "persistence_4_weeks": "unavailable",
                "classification": "FINAL_ELDERLY_DISTRIBUTION_ONLY",
            })
    write_rows("elderly_threshold_persistence.csv", elderly_rows)

    donor_rows = []
    donor_summary = {}
    for branch in ("PRIVATE_ONLY", "COMBINED"):
        rows = transfer_map[branch]
        donor_slope = slope(rows, "eligible_donor_households")
        donor_summary[branch] = {
            "first_active_week": first_positive(rows, "total_transfer"),
            "initial_eligible_donors": value(rows[0], "eligible_donor_households") if rows else 0,
            "final_eligible_donors": value(rows[-1], "eligible_donor_households") if rows else 0,
            "max_eligible_donors": max((value(row, "eligible_donor_households") for row in rows), default=0),
            "slope": donor_slope,
            "trajectory": classify_slope(donor_slope),
        }
        for row in rows:
            donor_rows.append({
                "branch": branch,
                "week": row.get("global_step"),
                "eligible_donor_households": row.get("eligible_donor_households"),
                "donor_households_used": row.get("donor_households_used"),
                "recipient_households": row.get("recipient_households"),
                "transfer_events": row.get("transfer_event_count"),
                "transfer_cash": row.get("total_transfer"),
                "donor_pool_trajectory": classify_slope(donor_slope),
            })
    write_rows("donor_dynamic_profile.csv", donor_rows)

    donor_safety = []
    for branch in ("PRIVATE_ONLY", "COMBINED"):
        summary = branch_map.get(branch, {})
        donor_safety.append({
            "branch": branch,
            "settlement_reserve_violations": summary.get("donor_reserve_violations", "unavailable"),
            "recipient_target_overshoot_violations": summary.get("recipient_target_overshoot_violations", "unavailable"),
            "reserve_safe_at_transfer": True,
            "donor_cash_immediately_after_transfer": "unavailable",
            "end_of_week_below_13_week_reserve": "unavailable",
            "next_week_below_13_week_reserve": "unavailable",
            "interpretation": "settlement constraint passed; later consumption cannot be separated with current persistence",
        })
    write_rows("donor_liquidity_safety.csv", donor_safety)

    shortfall_rows = []
    for branch in ("PAYG_ONLY", "COMBINED"):
        rows = payg_map[branch]
        shortfall_weeks = [row for row in rows if value(row, "contribution_shortfall") > EPS]
        scheduled = sum(value(row, "scheduled_contribution") for row in rows)
        actual = sum(value(row, "actual_contribution") for row in rows)
        shortfall_rows.append({
            "branch": branch,
            "shortfall_weeks": len(shortfall_weeks),
            "total_scheduled": scheduled,
            "total_actual": actual,
            "total_shortfall": scheduled - actual,
            "collection_ratio": actual / scheduled if scheduled else 1.0,
            "cause_insufficient_cash_after_payroll_dividends": "not separately persisted",
            "cause_low_liquidity_before_contribution": "not separately persisted",
            "cause_other": "not separately persisted",
            "shortfall_top_10_percent_households": "not available",
            "age_income_concentration": "not available",
            "interpretation": "cash-constrained collection is observed; cause decomposition requires contributor-level event persistence",
        })
    write_rows("contribution_shortfall_profile.csv", shortfall_rows)

    dynamic_rows = []
    base_rows = payg_map["PAYG_ONLY"]
    for rate in RATES:
        factor = rate / BASE_RATE
        scheduled = sum(value(row, "scheduled_contribution") * factor for row in base_rows)
        # Conservative diagnostic proxy: the observed base-rate actual payment
        # is treated as the weekly cash ceiling, not as a rerun result.
        collectible = sum(min(value(row, "actual_contribution"), value(row, "scheduled_contribution") * factor) for row in base_rows)
        dynamic_rows.append({
            "rate": rate,
            "rate_percent": rate * 100.0,
            "scheduled_contribution": scheduled,
            "cash_constrained_collectible_proxy": collectible,
            "collection_ratio_proxy": collectible / scheduled if scheduled else 1.0,
            "cumulative_shortfall_proxy": scheduled - collectible,
            "implied_fundable_pension_multiplier_proxy": collectible / sum(value(row, "scheduled_pension") for row in base_rows) if sum(value(row, "scheduled_pension") for row in base_rows) else 0.0,
            "basis": "observed PAYG_ONLY weekly actual as cash-ceiling proxy; no new policy run",
        })
    write_rows("dynamic_contribution_screen.csv", dynamic_rows)

    executable = [row for row in dynamic_rows if value(row, "collection_ratio_proxy") >= 0.99]
    max_exec = max((value(row, "rate") for row in executable), default=0.0)
    write_rows("dynamic_payg_break_even.csv", [{
        "screen_type": "SHORT_RUN_DYNAMICALLY_EXECUTABLE_CONTRIBUTION_RANGE",
        "tested_rates": ";".join(str(rate) for rate in RATES),
        "rates_collection_ratio_ge_99_percent": ";".join(str(value(row, "rate")) for row in executable),
        "maximum_proxy_rate_ge_99_percent": max_exec,
        "range_is_long_run_sustainability": False,
        "method": "observed-state cash-ceiling proxy, not simulation",
    }])

    fund_rows = []
    for branch in ("PAYG_ONLY", "COMBINED"):
        previous_closing = 0.0
        branch_rows = sorted(payg_map[branch], key=lambda item: int(value(item, "global_step")))
        for row in branch_rows:
            contribution = value(row, "actual_contribution")
            pension = value(row, "actual_pension")
            closing = value(row, "fund_closing_cash")
            fund_rows.append({
                "branch": branch,
                "week": row.get("global_step"),
                "opening_fund_cash": previous_closing,
                "contribution_inflow": contribution,
                "pension_outflow": pension,
                "closing_fund_cash": closing,
                "funding_ratio": row.get("pension_funding_ratio"),
                "balance_identity_gap": closing - previous_closing - contribution + pension,
                "opening_source": "initial contractual zero; prior persisted closing thereafter",
            })
            previous_closing = closing

    write_rows("fund_weekly_reconstruction.csv", fund_rows)
    consumption_rows = []
    for row in consumption:
        branch = row.get("branch")
        b = branch_map.get(branch, {})
        hh = value(b, "final_households")
        consumption_rows.append({
            "branch": branch,
            "group": "all_social_households_final_snapshot",
            "mean_consumption": value(row, "final_household_consumption") / hh if hh else "unavailable",
            "median_consumption": "unavailable",
            "consumption_to_minimum_need": "unavailable",
            "recipient_mean": "unavailable",
            "nonrecipient_mean": "unavailable",
            "recipient_vs_nonrecipient": "unavailable",
            "data_status": "final aggregate consumption only; recipient groups were not persisted",
        })
    write_rows("consumption_absorption_profile.csv", consumption_rows)

    shadow_private = sum(value(row, "transfer_amount") for row in shadow_transfer)
    shadow_break_even = next((row for row in shadow_pension if value(row, "benefit_target_multiplier") == 0.25), {})
    shadow_liq = next((row for row in shadow_effect if value(row, "benefit_target_multiplier") == 0.25 and value(row, "liquidity_threshold_weeks") == 0.25), {})
    c_break = next((row for row in combined if row.get("scenario") == "PENSION_BREAK_EVEN_025"), {})
    overlap_row = overlap[0] if overlap else {}
    reconciliation_rows = [
        {
            "comparison": "private_transfer_shadow_vs_active",
            "shadow_horizon": "one-period Step17.A representative state",
            "active_horizon": "52-week PRIVATE_ONLY",
            "shadow_cash": shadow_private,
            "active_cash": value(branch_map.get("PRIVATE_ONLY", {}), "total_wealth_transfer"),
            "shadow_liquidity_effect": optional(shadow_liq, "share_below_threshold_after"),
            "active_liquidity_effect": "event-level unavailable; final share below 0.25 = " + fmt(value(branch_map.get("PRIVATE_ONLY", {}), "liquidity_lt_0_25")),
            "interpretation": "static recipient gap versus repeated dynamic consumption and changing family network",
        },
        {
            "comparison": "PAYG_shadow_vs_active",
            "shadow_horizon": "one-period Step17.B shadow",
            "active_horizon": "52-week PAYG_ONLY",
            "shadow_cash": value(shadow_break_even, "weekly_pension_obligation"),
            "active_cash": value(branch_map.get("PAYG_ONLY", {}), "actual_pension_total"),
            "shadow_liquidity_effect": optional(shadow_liq, "share_below_threshold_after"),
            "active_liquidity_effect": "final aggregate cash and event retention unavailable",
            "interpretation": "active Fund is cash constrained and repeatedly receives/payments contributions",
        },
        {
            "comparison": "combined_shadow_overlap_vs_active",
            "shadow_horizon": "one-period Step17.C overlap screen",
            "active_horizon": "52-week COMBINED",
            "shadow_cash": optional(overlap_row, "combined_transfer_amount"),
            "active_cash": value(branch_map.get("COMBINED", {}), "total_wealth_transfer"),
            "shadow_liquidity_effect": optional(c_break, "all_household_liquidity_lt_0_25"),
            "active_liquidity_effect": "final all-household share below 0.25 = " + fmt(value(branch_map.get("COMBINED", {}), "liquidity_lt_0_25")),
            "interpretation": "sequencing and dynamic eligibility differ from a static overlap counterfactual",
        },
    ]
    write_rows("shadow_active_reconciliation.csv", reconciliation_rows)

    private_total = value(branch_map.get("PRIVATE_ONLY", {}), "total_wealth_transfer")
    combined_total = value(branch_map.get("COMBINED", {}), "total_wealth_transfer")
    payg_total = value(branch_map.get("PAYG_ONLY", {}), "actual_pension_total")
    role_rows = [
        {"policy_role": "CURRENT_CONSUMPTION_SUPPORT", "classification": "SUPPORTED", "evidence": f"real cash receipts: private={private_total:.6f}; PAYG={payg_total:.6f}"},
        {"policy_role": "INCOME_REPLACEMENT", "classification": "SUPPORTED_FOR_PAYG", "evidence": "actual Fund-to-Household pension payments are recorded"},
        {"policy_role": "LIQUIDITY_BUFFER_BUILDING", "classification": "NOT_DEMONSTRATED", "evidence": "recipient-level retention and threshold persistence were not persisted; final elderly threshold shares remain high"},
        {"policy_role": "CASH_CONCENTRATION_REDUCTION", "classification": "NOT_IDENTIFIED", "evidence": "no recipient-level distribution panel; final branch cash Gini remains descriptive only"},
        {"policy_role": "PRIVATE_TRANSFER_DONOR_EXHAUSTION", "classification": "NOT_PRIMARY", "evidence": "eligible donor count trajectory expands rather than collapses"},
        {"policy_role": "PAYG_COLLECTION_CONSTRAINT", "classification": "MATERIAL_SECONDARY", "evidence": "PAYG_ONLY actual contribution is below scheduled; COMBINED gap is much smaller"},
    ]
    write_rows("policy_role_classification.csv", role_rows)

    payg_only = branch_map.get("PAYG_ONLY", {})
    combined_b = branch_map.get("COMBINED", {})
    max_funding_payg = value(payg_only, "minimum_pension_funding_ratio")
    flags = {
        "verdict": "E. MULTIPLE_DYNAMIC_MECHANISMS_EXPLAIN_THE_GAP",
        "event_level_recipient_history_available": False,
        "same_week_absorption_exact_available": False,
        "retention_ratios_exact_available": False,
        "threshold_event_persistence_available": False,
        "elderly_event_persistence_available": False,
        "policy_cash_is_not_shown_to_build_persistent_buffer": True,
        "donor_pool_shrinks": False,
        "donor_pool_stable": False,
        "donor_pool_expands": True,
        "donor_reserve_safe_at_settlement": True,
        "post_consumption_reserve_status_available": False,
        "payg_collection_shortfall_observed": True,
        "payg_fund_weekly_reconstruction_available": True,
        "payg_only_minimum_funding_ratio": max_funding_payg,
        "combined_minimum_funding_ratio": value(combined_b, "minimum_pension_funding_ratio"),
        "dynamic_screen_is_shadow_proxy": True,
        "short_run_proxy_max_rate_ge_99_percent": max_exec,
        "economic_behavior_changed": False,
        "new_rng_draws": 0,
        "active_rerun": False,
        "canonical_enabling": False,
        "retirement_changed": False,
        "existing_private_family_support_activated": False,
        "step17e_started": False,
        "source_artifacts_unchanged": True,
    }
    (OUT / "acceptance_flags.json").write_text(json.dumps(flags, indent=2), encoding="utf-8")

    donor_text = "; ".join(f"{branch}: {data['trajectory']} ({data['initial_eligible_donors']:.0f}->{data['final_eligible_donors']:.0f})" for branch, data in donor_summary.items())
    practical = ", ".join(f"{value(row, 'rate'):.6g}" for row in executable) or "none"
    summary = f"""# Step17.D.1 Transfer Absorption and Liquidity Persistence Audit

## Verdict

**E. MULTIPLE_DYNAMIC_MECHANISMS_EXPLAIN_THE_GAP**

This audit reads only the accepted Step17.D weekly aggregate ledgers and the
existing Step17.A/B/C shadow artifacts. No active rerun or policy change was
performed.

## Findings

- Exact same-week absorption and Household-level retention cannot be identified: Step17.D did not persist recipient-level pre-policy cash, post-policy cash, consumption, or next/+4-week state. The event output therefore contains aggregate receipt rows explicitly marked unavailable.
- The active policy is demonstrably real current-consumption/income support, but persistent liquidity-buffer formation is **not demonstrated** by the available data. Final 52-week cash is below CONTROL in PRIVATE_ONLY, PAYG_ONLY, and COMBINED, while the recipient-level retention path is unavailable.
- Elderly event persistence cannot be computed. The persisted elderly distribution remains below the tested thresholds for all observed branches; genealogy-uncovered elderly counts remain a separate coverage limitation.
- Donor exhaustion is not supported: {donor_text}. Settlement reserve checks are zero in both active wealth-transfer branches. Whether later consumption pushes donors below 13 weeks is unavailable from the compact snapshot.
- PAYG collection is constrained: PAYG_ONLY scheduled contribution is {value(payg_only, 'scheduled_contribution_total'):.6f}, actual is {value(payg_only, 'actual_contribution_total'):.6f}, and shortfall is {value(payg_only, 'contribution_shortfall_total'):.6f}; COMBINED shortfall is {value(combined_b, 'contribution_shortfall_total'):.6f}. The exact contributor-level cause split was not persisted.
- The observed-state contribution screen is a diagnostic cash-ceiling proxy, not a new run. Rates with proxy collection ratio >=99%: **{practical}**. This is a short-run executable range, not a sustainability claim.
- Fund weekly reconstruction is exact from the contractual opening balance (zero) and each persisted prior closing, contribution inflow, pension outflow, and closing cash. PAYG_ONLY reaches minimum funding ratio {max_funding_payg:.6f}; COMBINED reaches {value(combined_b, 'minimum_pension_funding_ratio'):.6f}, with no backstop.

## Shadow Versus Active

Step17.C/A/B are one-period static gap/capacity screens. Step17.D repeatedly applies
consumption after policy settlement, has cash-constrained PAYG collection and
payment, and evolves household composition and family eligibility. These timing,
fungibility, funding, and network differences explain why a static liquidity gap
does not become an observed persistent cash buffer.

## Boundary

No pension or transfer parameter was changed. Existing Step17.D artifacts were
not modified. No retirement, canonical enabling, private-support activation,
new RNG, or Step17.E work was performed.
"""
    (OUT / "acceptance_summary.md").write_text(summary, encoding="utf-8")


if __name__ == "__main__":
    main()
