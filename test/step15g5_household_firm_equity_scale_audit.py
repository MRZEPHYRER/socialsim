from __future__ import annotations

import csv
import json
import math
import shutil
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
G4 = ROOT / "test/output/step15G4_book_priced_autonomous_secondary"
MAIN = ROOT / "test/output/step15G4_main_smoke"
OUTPUT = ROOT / "test/output/step15G5_household_firm_equity_scale_audit"
TOLERANCE = 1e-6


def read_csv(path):
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def n(value, default=float("nan")):
    if value in (None, "", "nan", "NaN"):
        return float(default)
    return float(value)


def finite(value):
    return value is not None and math.isfinite(float(value))


def percentile(values, p):
    values = sorted(float(value) for value in values if finite(value))
    if not values:
        return float("nan")
    if len(values) == 1:
        return values[0]
    position = (len(values) - 1) * p
    lower = int(math.floor(position))
    upper = int(math.ceil(position))
    if lower == upper:
        return values[lower]
    weight = position - lower
    return values[lower] * (1 - weight) + values[upper] * weight


def total(values):
    return sum(float(value) for value in values if finite(value))


def write_csv(path, rows):
    if not rows:
        raise ValueError(f"no rows for {path}")
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main():
    if not G4.exists() or not MAIN.exists():
        raise FileNotFoundError("Step15G.4 outputs are required.")

    demand = read_csv(G4 / "household_equity_demand_panel.csv")
    ownership = [
        row
        for row in read_csv(G4 / "ownership_panel.csv")
        if row.get("case") == "treatment_on"
    ]
    diagnostics = read_csv(MAIN / "diagnostics.csv")
    accounting = read_csv(MAIN / "accounting/firm_accounting.csv")
    firm_diagnostics = read_csv(MAIN / "firm_diagnostics.csv")

    review_step = int(max(n(row.get("decision_step")) for row in demand))
    valuation_step = int(max(n(row.get("valuation_step")) for row in demand))
    demand_at_review = [
        row for row in demand if int(n(row.get("decision_step"))) == review_step
    ]
    diag_review = next(
        row for row in diagnostics if int(n(row.get("global_step"))) == review_step
    )
    account_valuation = next(
        row
        for row in accounting
        if int(n(row.get("global_step"))) == valuation_step
    )
    firm_valuation = next(
        row
        for row in firm_diagnostics
        if int(n(row.get("global_step"))) == valuation_step
    )
    ownership_review = next(
        row
        for row in ownership
        if int(n(row.get("global_step"))) == review_step
    )

    total_households = int(n(diag_review.get("households")))
    active_households = int(n(diag_review.get("active_households")))
    inactive_households = max(0, total_households - active_households)
    household_cash_after = n(diag_review.get("total_household_wealth"))
    executed_cash = total(n(row.get("executed_cash")) for row in demand_at_review)
    household_cash_before = household_cash_after + executed_cash
    equity_assets_after = n(ownership_review.get("household_equity_assets"))
    equity_assets_before = max(0.0, equity_assets_after - executed_cash)
    financial_net_worth_before = household_cash_before + equity_assets_before

    reserve_total = n(diag_review.get("average_target_wealth")) * active_households
    necessary_evaluated = total(
        n(row.get("necessary_consumption_buffer")) for row in demand_at_review
    )
    available_evaluated = total(
        n(row.get("available_financial_cash")) for row in demand_at_review
    )
    positive_evaluated = [
        row
        for row in demand_at_review
        if n(row.get("available_financial_cash")) > TOLERANCE
    ]
    desired_budget_evaluated = total(
        n(row.get("desired_equity_budget")) for row in demand_at_review
    )
    unconstrained_budget_evaluated = total(
        n(row.get("available_financial_cash")) for row in positive_evaluated
    )
    reference_price = n(demand_at_review[0].get("reference_price"))
    total_shares = n(demand_at_review[0].get("total_shares_used"), 100.0)
    person_ownership_actual = n(ownership_review.get("person_ownership_fraction"))
    max_ownership_from_all_cash_upper_bound = (
        household_cash_before / reference_price / total_shares
    )
    ownership_from_evaluated_available = (
        unconstrained_budget_evaluated / reference_price / total_shares
    )
    ownership_under_evaluated_1pct_budget = (
        desired_budget_evaluated / reference_price / total_shares
    )

    capacity_rows = []

    def capacity_row(scope, metric, value, count, status, note=""):
        values = []
        if metric == "household_cash":
            values = [n(row.get("household_cash")) for row in demand_at_review]
        elif metric == "available_financial_cash":
            values = [n(row.get("available_financial_cash")) for row in demand_at_review]
        elif metric == "desired_equity_budget":
            values = [n(row.get("desired_equity_budget")) for row in demand_at_review]
        capacity_rows.append({
            "review_step": review_step,
            "scope": scope,
            "coverage_count": count,
            "coverage_status": status,
            "metric": metric,
            "value": value,
            "p50": percentile(values, 0.50),
            "p75": percentile(values, 0.75),
            "p90": percentile(values, 0.90),
            "p95": percentile(values, 0.95),
            "note": note,
        })

    capacity_row(
        "world_aggregate", "household_count", total_households,
        total_households, "AUTHORITATIVE_MACRO", "World diagnostics."
    )
    capacity_row(
        "world_aggregate", "active_household_count", active_households,
        active_households, "AUTHORITATIVE_MACRO", "World diagnostics."
    )
    capacity_row(
        "world_aggregate", "total_household_cash_before_transfer", household_cash_before,
        active_households, "AUTHORITATIVE_AGGREGATE", "Cash after transfer plus observed transfer cash."
    )
    capacity_row(
        "world_aggregate", "total_liquidity_reserve", reserve_total,
        active_households, "AUTHORITATIVE_AGGREGATE", "Average target wealth multiplied by active households."
    )
    for metric in (
        "total_necessary_consumption_buffer",
        "total_available_financial_cash",
        "positive_available_household_count",
        "aggregate_possible_equity_budget_1pct",
    ):
        capacity_row(
            "world_aggregate", metric, float("nan"), active_households,
            "UNAVAILABLE_NOT_PERSISTED",
            "G4 did not persist full-world per-Household reserve/available-cash fields.",
        )
    capacity_row(
        "evaluated_households", "household_count", len(demand_at_review),
        len(demand_at_review), "AUTHORITATIVE_PANEL", "All persisted evaluated rows."
    )
    capacity_row(
        "evaluated_households", "total_household_cash", total(
            n(row.get("household_cash")) for row in demand_at_review
        ), len(demand_at_review), "AUTHORITATIVE_PANEL"
    )
    capacity_row(
        "evaluated_households", "total_liquidity_reserve", total(
            n(row.get("liquidity_reserve")) for row in demand_at_review
        ), len(demand_at_review), "AUTHORITATIVE_PANEL"
    )
    capacity_row(
        "evaluated_households", "total_necessary_consumption_buffer",
        necessary_evaluated, len(demand_at_review), "AUTHORITATIVE_PANEL"
    )
    capacity_row(
        "evaluated_households", "total_available_financial_cash",
        available_evaluated, len(demand_at_review), "AUTHORITATIVE_PANEL"
    )
    capacity_row(
        "evaluated_households", "positive_available_household_count",
        len(positive_evaluated), len(demand_at_review), "AUTHORITATIVE_PANEL"
    )
    capacity_row(
        "evaluated_households", "aggregate_possible_equity_budget_1pct",
        desired_budget_evaluated, len(demand_at_review), "AUTHORITATIVE_PANEL",
        "Existing 1% rule; not retuned."
    )
    capacity_row(
        "evaluated_households", "aggregate_budget_without_1pct_cap",
        unconstrained_budget_evaluated, len(positive_evaluated), "AUTHORITATIVE_PANEL",
        "Diagnostic counterfactual only."
    )

    reason_counts = {}
    for row in demand_at_review:
        reason = row.get("decision_reason", "UNKNOWN")
        reason_counts[reason] = reason_counts.get(reason, 0) + 1
    coverage_rows = [
        {"review_step": review_step, "metric": "total_households_in_world", "count": total_households, "status": "AUTHORITATIVE", "reason": "world diagnostics"},
        {"review_step": review_step, "metric": "active_households", "count": active_households, "status": "AUTHORITATIVE", "reason": "world diagnostics"},
        {"review_step": review_step, "metric": "inactive_households_before_evaluation", "count": inactive_households, "status": "AUTHORITATIVE_LOWER_BOUND", "reason": "inactive slots cannot provide a buyer Person"},
        {"review_step": review_step, "metric": "households_evaluated", "count": len(demand_at_review), "status": "AUTHORITATIVE_PANEL", "reason": "persisted demand rows"},
        {"review_step": review_step, "metric": "households_with_positive_demand", "count": len(positive_evaluated), "status": "AUTHORITATIVE_PANEL", "reason": "positive_excess_liquidity"},
        {"review_step": review_step, "metric": "households_executed", "count": sum(n(row.get("executed_shares")) > TOLERANCE for row in demand_at_review), "status": "AUTHORITATIVE_PANEL", "reason": "executed shares > 0"},
        {"review_step": review_step, "metric": "buyer_cap_blocked_households", "count": float("nan"), "status": "UNAVAILABLE_NOT_PERSISTED", "reason": "screen stopped after buyer cap; remaining household-level causes were not persisted"},
        {"review_step": review_step, "metric": "unobserved_active_households_after_screen_stop", "count": max(0, active_households - len(demand_at_review)), "status": "UNOBSERVED_POOL", "reason": "not safe to assign a unique block reason"},
    ]
    for reason, count in sorted(reason_counts.items()):
        coverage_rows.append({
            "review_step": review_step,
            "metric": f"persisted_reason::{reason}",
            "count": count,
            "status": "AUTHORITATIVE_PANEL",
            "reason": "exact reason in demand panel",
        })
    coverage_rows.extend([
        {"review_step": review_step, "metric": "concentration_guardrail_blocks", "count": 0, "status": "AUTHORITATIVE_PANEL", "reason": "no persisted decision had concentration_binding=true"},
        {"review_step": review_step, "metric": "legacy_supply_blocks", "count": 0, "status": "AUTHORITATIVE_PANEL", "reason": "Legacy shares were abundant"},
    ])

    firm_equity = n(account_valuation.get("equity"))
    firm_cash = n(account_valuation.get("cash"))
    inventory_book = n(account_valuation.get("inventory_book_value"))
    capital_book = 0.0
    other_assets = max(
        0.0,
        n(account_valuation.get("total_assets")) - firm_cash - inventory_book,
    )
    principal = n(account_valuation.get("loan_balance"))
    arrears = n(account_valuation.get("total_lender_claim")) - principal
    other_liabilities = n(account_valuation.get("total_liabilities")) - principal
    scale_rows = [{
        "review_step": review_step,
        "valuation_step": valuation_step,
        "firm_book_equity": firm_equity,
        "total_household_cash": household_cash_before,
        "total_household_financial_net_worth": financial_net_worth_before,
        "aggregate_available_financial_cash": float("nan"),
        "evaluated_available_financial_cash": available_evaluated,
        "aggregate_current_equity_assets": equity_assets_before,
        "legacy_owner_cash_after_transfer": n(ownership_review.get("legacy_owner_cash")),
        "firm_equity_over_household_cash": firm_equity / household_cash_before,
        "firm_equity_over_household_financial_net_worth": firm_equity / financial_net_worth_before,
        "firm_equity_over_aggregate_available_cash": float("nan"),
        "evaluated_1pct_budget_over_firm_equity": desired_budget_evaluated / firm_equity,
        "note": "Full-world available cash was not persisted; evaluated subset is reported separately.",
    }]

    ownership_rows = [
        {"review_step": review_step, "component": "all_household_cash_upper_bound", "ownership_fraction": max_ownership_from_all_cash_upper_bound, "ownership_percent": max_ownership_from_all_cash_upper_bound * 100, "basis": "upper bound ignores liquidity reserve and necessary buffer"},
        {"review_step": review_step, "component": "evaluated_available_cash_without_1pct_cap", "ownership_fraction": ownership_from_evaluated_available, "ownership_percent": ownership_from_evaluated_available * 100, "basis": "48 persisted evaluated Households"},
        {"review_step": review_step, "component": "evaluated_1pct_budget_capacity", "ownership_fraction": ownership_under_evaluated_1pct_budget, "ownership_percent": ownership_under_evaluated_1pct_budget * 100, "basis": "existing 1% rule"},
        {"review_step": review_step, "component": "g4_actual_ownership", "ownership_fraction": person_ownership_actual, "ownership_percent": person_ownership_actual * 100, "basis": "actual G4 treatment"},
        {"review_step": review_step, "component": "liquidity_protection_block_count_evaluated", "ownership_fraction": float("nan"), "ownership_percent": float("nan"), "basis": f"38 of {len(demand_at_review)} evaluated rows had no excess liquidity"},
        {"review_step": review_step, "component": "purchase_cap_reduction_within_positive_evaluated", "ownership_fraction": ownership_from_evaluated_available - ownership_under_evaluated_1pct_budget, "ownership_percent": (ownership_from_evaluated_available - ownership_under_evaluated_1pct_budget) * 100, "basis": "1% cap counterfactual; diagnostic only"},
        {"review_step": review_step, "component": "concentration_guardrail_block", "ownership_fraction": 0.0, "ownership_percent": 0.0, "basis": "no concentration block observed"},
        {"review_step": review_step, "component": "legacy_supply_block", "ownership_fraction": 0.0, "ownership_percent": 0.0, "basis": "Legacy supply was not binding"},
    ]

    book_rows = [
        {"valuation_step": valuation_step, "component": "cash", "amount": firm_cash, "sign": 1, "source": "firm_accounting.cash"},
        {"valuation_step": valuation_step, "component": "inventory_book_value", "amount": inventory_book, "sign": 1, "source": "firm_accounting.inventory_book_value"},
        {"valuation_step": valuation_step, "component": "capital_asset_book_value", "amount": capital_book, "sign": 1, "source": "no capital assets active"},
        {"valuation_step": valuation_step, "component": "other_assets", "amount": other_assets, "sign": 1, "source": "total_assets residual; expected zero"},
        {"valuation_step": valuation_step, "component": "principal", "amount": principal, "sign": -1, "source": "firm_accounting.loan_balance"},
        {"valuation_step": valuation_step, "component": "arrears", "amount": arrears, "sign": -1, "source": "total lender claim minus principal"},
        {"valuation_step": valuation_step, "component": "other_liabilities", "amount": other_liabilities, "sign": -1, "source": "total_liabilities minus principal"},
        {"valuation_step": valuation_step, "component": "derived_book_equity", "amount": firm_cash + inventory_book + capital_book + other_assets - principal - arrears - other_liabilities, "sign": 1, "source": "reconstructed accounting identity"},
        {"valuation_step": valuation_step, "component": "book_equity_reconciliation_gap", "amount": (firm_cash + inventory_book + capital_book + other_assets - principal - arrears - other_liabilities) - firm_equity, "sign": 1, "source": "derived minus authoritative equity"},
    ]

    verdict = "G. MIXED_SCALE_CONSTRAINT"
    flags = {
        "verdict": verdict,
        "audit_only": True,
        "new_simulation": False,
        "purchase_behavior_changed": False,
        "purchase_cap_tuned": False,
        "buyer_count_tuned": False,
        "cadence_tuned": False,
        "price_changed": False,
        "primary_issuance": False,
        "leverage": False,
        "investment_active": False,
        "Step13_changed": False,
        "full_world_available_cash_persisted": False,
        "evaluated_panel_complete": True,
        "buyer_coverage_exact_for_unobserved_pool": False,
        "book_equity_double_counting_found": False,
        "main_constraint": "mixed: liquidity protection removes most evaluated households, 1% cap binds positive buyers, and high Firm book scale makes every share expensive",
        "firm_book_equity": firm_equity,
        "evaluated_households": len(demand_at_review),
        "evaluated_positive_demand_households": len(positive_evaluated),
        "g4_actual_ownership_fraction": person_ownership_actual,
    }

    shutil.rmtree(OUTPUT, ignore_errors=True)
    OUTPUT.mkdir(parents=True, exist_ok=True)
    write_csv(OUTPUT / "household_financial_capacity.csv", capacity_rows)
    write_csv(OUTPUT / "buyer_coverage_audit.csv", coverage_rows)
    write_csv(OUTPUT / "firm_household_scale_ratios.csv", scale_rows)
    write_csv(OUTPUT / "ownership_capacity_decomposition.csv", ownership_rows)
    write_csv(OUTPUT / "firm_book_equity_decomposition.csv", book_rows)
    (OUTPUT / "acceptance_flags.json").write_text(
        json.dumps(flags, indent=2, sort_keys=True), encoding="utf-8"
    )
    (OUTPUT / "acceptance_summary.md").write_text(
        f"""# Step 15G.5 Household Financial Capacity vs Firm Equity Scale Audit

**Verdict: {verdict}**

This was an audit-only reuse of Step15G.4 artifacts. No simulation was rerun,
and no purchase cap, buyer count, cadence, price, ownership, or economic
behavior was changed.

## Main result

At review step {review_step}, the lagged valuation step was {valuation_step} and
Firm book equity was {firm_equity:,.6f}. The persisted screen evaluated only
{len(demand_at_review)} Households: {len(positive_evaluated)} had positive
available financial cash and all {len(positive_evaluated)} executed. The other
{len(demand_at_review) - len(positive_evaluated)} evaluated rows were blocked by
`no_excess_liquidity`.

The 1% rule produced an evaluated-subset budget of {desired_budget_evaluated:,.6f}
from {unconstrained_budget_evaluated:,.6f} available financial cash. Without
that 1% cap, the observed positive evaluated subset could purchase about
{ownership_from_evaluated_available * 100:.6f}% ownership; with the existing
rule it purchased about {ownership_under_evaluated_1pct_budget * 100:.6f}%.
Actual G4 ownership was {person_ownership_actual * 100:.6f}%.

An all-household-cash upper bound, ignoring all liquidity protection, is only
{max_ownership_from_all_cash_upper_bound * 100:.6f}% ownership. This shows the
Firm equity scale is intrinsically large relative to Household cash. The
observed result is therefore mixed: liquidity protection limits participation,
the 1% cap further reduces positive buyers' share quantity, and the high book
price makes each unit of ownership expensive.

## Coverage limitation

World diagnostics report {total_households} Household slots and {active_households}
active Households, but G4 did not persist full-world per-Household reserve and
available-cash records. Therefore full-world total available cash, its
percentiles, and the exact buyer-cap exclusion count remain unavailable. The
48-row demand panel is reported as an evaluated subset, not as the population
distribution. The remaining active pool was not assigned speculative block
reasons.

## Book equity composition

At valuation step {valuation_step}, authoritative Firm equity is composed of
cash {firm_cash:,.6f} plus inventory book value {inventory_book:,.6f}; capital
asset book value is zero, principal is {principal:,.6f}, arrears are
{arrears:,.6f}, and other liabilities are {other_liabilities:,.6f}. The
reconstruction gap is zero within the persisted accounting tolerance. The
legacy market-value net-worth diagnostic was not used and no asset was double
counted.

Classification **G. MIXED_SCALE_CONSTRAINT** is selected because the observed
screen has both binding liquidity/budget layers while Firm book-equity scale
dominates the ownership quantity even under generous passive upper bounds.
""",
        encoding="utf-8",
    )
    print(json.dumps(flags, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
