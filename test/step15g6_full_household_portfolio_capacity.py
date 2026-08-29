from __future__ import annotations

import csv
import json
import math
import shutil
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from economy.autonomous_secondary_equity import (
    AutonomousSecondaryEquityPurchaseSystem,
)
from economy.household_equity_demand import ShadowHouseholdEquityDemandSystem
from world import World


OUTPUT = ROOT / "test/output/step15G6_full_household_portfolio_capacity"
WEEKS_TO_REVIEW = 53
TOLERANCE = 1e-6
REFERENCE_MODE = "lagged_book_equity"
PURCHASE_RATE_GRID = (0.005, 0.01, 0.02, 0.05, 0.10, 1.00)


def num(value, default=0.0):
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
    low = int(math.floor(position))
    high = int(math.ceil(position))
    if low == high:
        return values[low]
    weight = position - low
    return values[low] * (1 - weight) + values[high] * weight


def write_csv(path, rows):
    if not rows:
        raise ValueError(f"no rows for {path}")
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def build_shadow_world():
    world = World(initial_population=5000, seed=42, diagnostics_mode="full")
    world.ensure_ownership_state()
    world.person_equity_transition_enabled = False
    world.autonomous_secondary_equity_enabled = False
    return world


def main():
    # This replay intentionally keeps autonomous execution OFF. It only
    # recreates the accepted review state so the non-mutating demand screen
    # can inspect every Household before the buyer cap would truncate it.
    world = build_shadow_world()
    for _ in range(WEEKS_TO_REVIEW):
        world.step()

    review_step = int(world.current_step_index)
    valuation_step = review_step - 1
    firm = world.firms[0]
    accounting_rows = getattr(world.accounting, "rows", [])
    valuation_row = next(
        row for row in accounting_rows
        if int(num(row.get("global_step", row.get("step")))) == valuation_step
        and int(num(row.get("firm_id"))) == int(getattr(firm, "firm_id", 0))
    )
    book_equity = num(valuation_row.get("equity"))
    total_shares = float(firm.cap_table.total_shares)
    reference_price = book_equity / total_shares if book_equity > 0 else float("nan")

    demand_system = ShadowHouseholdEquityDemandSystem()
    liquidity_system = AutonomousSecondaryEquityPurchaseSystem(
        world,
        reference_price_mode=REFERENCE_MODE,
    )
    decisions = []
    ordered_eligible = []
    exclusion_counts = {}

    for household in sorted(world.households, key=lambda item: getattr(item, "id", 0)):
        person = demand_system.select_buyer_person(household, world.population)
        household_cash = num(getattr(household, "wealth", 0.0))
        reserve = liquidity_system._liquidity_reserve(household)
        necessary = num(
            getattr(household, "necessary_consumption_this_step", 0.0)
        )
        available = max(household_cash - reserve - necessary, 0.0)
        equity_assets = num(getattr(household, "equity_asset_value", 0.0))
        financial_net_worth = household_cash + equity_assets

        if person is None:
            record = {
                "review_step": review_step,
                "valuation_step": valuation_step,
                "household_id": household.id,
                "buyer_person_id": "",
                "cash": household_cash,
                "liquidity_reserve": reserve,
                "necessary_consumption_buffer": necessary,
                "available_financial_cash": available,
                "current_equity_assets": equity_assets,
                "financial_net_worth": financial_net_worth,
                "desired_equity_budget": 0.0,
                "desired_shares": 0.0,
                "reference_price": reference_price,
                "liquidity_eligible": False,
                "concentration_eligible": False,
                "buyer_person_available": False,
                "final_shadow_eligibility": False,
                "exclusion_reason": "no_buyer_person_available",
                "concentration_binding": False,
            }
            decisions.append(record)
            exclusion_counts[record["exclusion_reason"]] = (
                exclusion_counts.get(record["exclusion_reason"], 0) + 1
            )
            continue

        decision = demand_system.decide(
            household_id=household.id,
            buyer_person_id=person.id,
            firm_id=getattr(firm, "firm_id", 0),
            household_cash=household_cash,
            liquidity_reserve=reserve,
            necessary_consumption_buffer=necessary,
            offered_transfer_price=reference_price,
            current_equity_assets=equity_assets,
            current_firm_equity_assets=equity_assets,
            current_person_firm_shares=sum(
                holding.shares
                for holding in firm.cap_table.holdings
                if holding.holder_type == "person"
                and holding.holder_id == person.id
            ),
            firm_total_shares=total_shares,
            available_legacy_shares=firm.cap_table.legacy_shares,
            desired_equity_budget=available * 0.01,
            max_household_equity_share_of_assets=0.10,
            max_person_ownership_fraction=0.05,
        )
        positive = decision.desired_shares > TOLERANCE
        if positive:
            reason = "eligible"
        elif available <= TOLERANCE:
            if household_cash <= reserve + TOLERANCE:
                reason = (
                    "liquidity_reserve_and_necessary_buffer_binding"
                    if necessary > TOLERANCE
                    else "liquidity_reserve_binding"
                )
            else:
                reason = "necessary_consumption_buffer_binding"
        elif decision.concentration_binding:
            reason = "concentration_guardrail_binding"
        else:
            reason = decision.decision_reason or "no_positive_shadow_demand"
        concentration_eligible = not decision.concentration_binding
        record = {
            "review_step": review_step,
            "valuation_step": valuation_step,
            "household_id": household.id,
            "buyer_person_id": person.id,
            "cash": household_cash,
            "liquidity_reserve": reserve,
            "necessary_consumption_buffer": necessary,
            "available_financial_cash": available,
            "current_equity_assets": equity_assets,
            "financial_net_worth": financial_net_worth,
            "desired_equity_budget": decision.desired_equity_budget,
            "desired_shares": decision.desired_shares,
            "reference_price": reference_price,
            "liquidity_eligible": available > TOLERANCE,
            "concentration_eligible": concentration_eligible,
            "buyer_person_available": True,
            "final_shadow_eligibility": positive,
            "exclusion_reason": reason,
            "concentration_binding": decision.concentration_binding,
        }
        decisions.append(record)
        if positive:
            ordered_eligible.append(record)
        else:
            exclusion_counts[reason] = exclusion_counts.get(reason, 0) + 1

    ordered_eligible.sort(key=lambda row: int(row["household_id"]))
    all_cash = sum(num(row["cash"]) for row in decisions)
    total_reserve = sum(num(row["liquidity_reserve"]) for row in decisions)
    total_necessary = sum(
        num(row["necessary_consumption_buffer"]) for row in decisions
    )
    available_all = sum(
        num(row["available_financial_cash"]) for row in decisions
    )
    desired_budget_all = sum(
        num(row["desired_equity_budget"]) for row in decisions
    )
    positive_available = [
        row for row in decisions
        if num(row["available_financial_cash"]) > TOLERANCE
    ]
    positive_demand = ordered_eligible
    total_households = len(decisions)
    active_count = len(world.active_households())

    distribution_rows = []
    for metric, values in (
        ("cash", [row["cash"] for row in decisions]),
        ("available_financial_cash", [row["available_financial_cash"] for row in decisions]),
        ("desired_equity_budget", [row["desired_equity_budget"] for row in decisions]),
    ):
        distribution_rows.append({
            "review_step": review_step,
            "metric": metric,
            "household_count": total_households,
            "mean": sum(values) / len(values) if values else float("nan"),
            "median": percentile(values, 0.50),
            "p75": percentile(values, 0.75),
            "p90": percentile(values, 0.90),
            "p95": percentile(values, 0.95),
            "p99": percentile(values, 0.99),
            "max": max(values, default=float("nan")),
            "total": sum(values),
        })

    coverage_rows = []
    for reason, count in sorted(exclusion_counts.items()):
        coverage_rows.append({
            "review_step": review_step,
            "reason": reason,
            "count": count,
            "share_of_households": count / total_households if total_households else 0.0,
        })
    coverage_rows.extend([
        {
            "review_step": review_step,
            "reason": "positive_available_financial_cash",
            "count": len(positive_available),
            "share_of_households": len(positive_available) / total_households,
        },
        {
            "review_step": review_step,
            "reason": "positive_equity_demand",
            "count": len(positive_demand),
            "share_of_households": len(positive_demand) / total_households,
        },
    ])

    buyer_cap_rows = []
    for count in (10, 25, 50, 100, len(positive_demand)):
        selected = positive_demand[:count]
        budget = sum(num(row["desired_equity_budget"]) for row in selected)
        shares = budget / reference_price if reference_price > 0 else float("nan")
        buyer_cap_rows.append({
            "review_step": review_step,
            "buyer_count": min(count, len(positive_demand)),
            "available_eligible_buyers": len(positive_demand),
            "cumulative_cash_budget": budget,
            "cumulative_shares": shares,
            "implied_ownership_fraction": shares / total_shares,
            "implied_ownership_percent": shares / total_shares * 100,
            "ordering": "household_id ascending",
            "executed": False,
        })

    purchase_rate_rows = []
    affected_count = len(positive_available)
    for rate in PURCHASE_RATE_GRID:
        budget = available_all * rate
        shares = budget / reference_price if reference_price > 0 else float("nan")
        purchase_rate_rows.append({
            "review_step": review_step,
            "available_cash_fraction": rate,
            "total_cash_budget": budget,
            "implied_ownership_fraction": shares / total_shares,
            "implied_ownership_percent": shares / total_shares * 100,
            "household_cash_remaining": all_cash - budget,
            "affected_households": affected_count,
            "positive_demand_households_at_1pct": len(positive_demand),
            "diagnostic_only": True,
        })

    valuation_row = {
        "review_step": review_step,
        "valuation_step": valuation_step,
        "firm_book_equity": book_equity,
        "lagged_book_price": reference_price,
        "total_household_cash": all_cash,
        "total_available_financial_cash": available_all,
        "total_1pct_budget": desired_budget_all,
        "total_household_financial_net_worth": all_cash + sum(
            num(row["current_equity_assets"]) for row in decisions
        ),
        "firm_equity_over_household_cash": book_equity / all_cash,
        "firm_equity_over_available_cash": book_equity / available_all if available_all else float("nan"),
        "total_1pct_budget_over_firm_equity": desired_budget_all / book_equity,
        "maximum_ownership_from_all_household_cash": all_cash / reference_price / total_shares,
        "maximum_ownership_from_all_available_cash": available_all / reference_price / total_shares,
        "ownership_from_current_1pct_budget": desired_budget_all / reference_price / total_shares,
        "household_count": total_households,
        "active_household_count": active_count,
        "positive_available_household_count": len(positive_available),
        "positive_demand_household_count": len(positive_demand),
    }

    # The rate and buyer cap reductions are interpreted together. This run
    # has a positive world capacity, but both layers materially compress it.
    first_10_budget = buyer_cap_rows[0]["cumulative_cash_budget"]
    buyer_cap_ratio = first_10_budget / desired_budget_all if desired_budget_all else 1.0
    rate_ratio = desired_budget_all / available_all if available_all else 0.0
    positive_share = len(positive_available) / total_households
    binding_layers = sum(
        ratio < 0.50
        for ratio in (
            available_all / all_cash if all_cash else 0.0,
            rate_ratio,
            buyer_cap_ratio,
        )
    )
    if binding_layers >= 2:
        verdict = "E. MIXED_PORTFOLIO_CAPACITY_CONSTRAINT"
    elif buyer_cap_ratio < 0.50:
        verdict = "B. BUYER_COUNT_CAP_PRIMARY_CONSTRAINT"
    elif rate_ratio < 0.50:
        verdict = "C. PURCHASE_RATE_CAP_PRIMARY_CONSTRAINT"
    elif available_all / all_cash < 0.50:
        verdict = "D. LIQUIDITY_BUFFER_PRIMARY_CONSTRAINT"
    else:
        verdict = "A. WORLD_HOUSEHOLD_CAPACITY_INTRINSICALLY_LOW"

    flags = {
        "verdict": verdict,
        "shadow_only": True,
        "all_households_evaluated": True,
        "households_evaluated": total_households,
        "review_step": review_step,
        "valuation_step": valuation_step,
        "lagged_book_price_used": True,
        "actual_purchase_executed": False,
        "purchase_cap_changed": False,
        "buyer_count_changed": False,
        "cadence_changed": False,
        "price_changed": False,
        "primary_issuance": False,
        "leverage": False,
        "investment_active": False,
        "dividend_policy_changed": False,
        "Step13_changed": False,
        "new_rng_draws_from_census": 0,
        "full_distribution_available": True,
        "first_10_budget_share_of_all_1pct_budget": buyer_cap_ratio,
        "current_rate_share_of_unconstrained_available_budget": rate_ratio,
        "positive_available_household_share": positive_share,
    }

    shutil.rmtree(OUTPUT, ignore_errors=True)
    OUTPUT.mkdir(parents=True, exist_ok=True)
    write_csv(OUTPUT / "household_capacity_census.csv", decisions)
    scale_distribution_rows = []
    for metric, value in valuation_row.items():
        if metric in {"review_step", "valuation_step"}:
            continue
        scale_distribution_rows.append({
            "review_step": review_step,
            "metric": f"scale::{metric}",
            "household_count": total_households,
            "mean": value,
            "median": float("nan"),
            "p75": float("nan"),
            "p90": float("nan"),
            "p95": float("nan"),
            "p99": float("nan"),
            "max": float("nan"),
            "total": value,
        })
    write_csv(
        OUTPUT / "household_capacity_distribution.csv",
        distribution_rows + scale_distribution_rows,
    )
    write_csv(OUTPUT / "buyer_cap_counterfactual.csv", buyer_cap_rows)
    write_csv(OUTPUT / "purchase_rate_counterfactual.csv", purchase_rate_rows)
    write_csv(OUTPUT / "liquidity_exclusion_breakdown.csv", coverage_rows)
    (OUTPUT / "acceptance_flags.json").write_text(
        json.dumps(flags, indent=2, sort_keys=True), encoding="utf-8"
    )
    (OUTPUT / "acceptance_summary.md").write_text(
        f"""# Step 15G.6 Full-World Household Portfolio Capacity Census

**Verdict: {verdict}**

This was a shadow-only replay to review step {review_step}. Autonomous share
settlement was disabled. Every one of the {total_households} Household records
was evaluated before any buyer-count truncation, using the lagged book price
from valuation step {valuation_step}: {reference_price:,.6f} per share.

## Full-world capacity

- Total Household cash: {all_cash:,.6f}
- Total available financial cash: {available_all:,.6f}
- Total desired budget under the existing 1% rule: {desired_budget_all:,.6f}
- Positive available-cash Households: {len(positive_available)} ({positive_share:.3%})
- Positive 1% equity-demand Households: {len(positive_demand)}
- Firm book equity: {book_equity:,.6f}

Shadow maximum ownership is {valuation_row['maximum_ownership_from_all_household_cash'] * 100:.6f}%
from all Household cash, {valuation_row['maximum_ownership_from_all_available_cash'] * 100:.6f}%
from all available financial cash, and
{valuation_row['ownership_from_current_1pct_budget'] * 100:.6f}% under the
current 1% rule.

## Buyer-cap and rate counterfactuals

The current deterministic order is Household ID ascending. The buyer-cap
counterfactual is recorded without execution. The 1% rule and buyer cap are
reported as separate diagnostic layers; neither was changed.

## Liquidity decomposition

The census records the exact exclusion reason for every Household. Reserve and
necessary-consumption buffers are not modified. Current equity assets are zero
in this fresh legacy-only shadow state, so concentration does not bind.

No Legacy supply constraint, primary issuance, leverage, investment, dividend
change, or Step13 change was introduced.
""",
        encoding="utf-8",
    )
    print(json.dumps(flags, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
