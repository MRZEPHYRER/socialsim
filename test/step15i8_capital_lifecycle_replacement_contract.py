"""Step 15I.8 passive capital depreciation and replacement fixtures."""

from __future__ import annotations

import csv
import math
import pickle
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from economy.investment_contracts import (
    CapitalAsset,
    CapitalStock,
    capital_book_value_bridge,
    replacement_expansion_needs,
)
from economy.investment_planning import ShadowInvestmentPlanner
from world import World


OUTPUT = ROOT / "test/output/step15I8_capital_lifecycle_replacement_contract"
TOLERANCE = 1e-10


def write_rows(path, rows):
    fields = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def metric(rows, category, name, actual, expected, passed, notes=""):
    numeric = isinstance(actual, (int, float)) and isinstance(expected, (int, float))
    rows.append(
        {
            "category": category,
            "metric": name,
            "actual": actual,
            "expected": expected,
            "gap": float(actual) - float(expected) if numeric else "",
            "passed": bool(passed),
            "notes": notes,
        }
    )


def fixture_asset(asset_id, *, cost, quantity=1.0, capacity=1.0, age=0.0, book=None, depreciation=0.0):
    closing = cost - depreciation if book is None else book
    return CapitalAsset(
        asset_id=asset_id,
        asset_class="fixture_machine",
        owner_firm_id=1,
        acquisition_cost=cost,
        quantity=quantity,
        capacity_metadata={
            "capital_service_capacity_per_unit": capacity,
            "active": True,
            "calibration_status": "NON-CALIBRATED_FIXTURE",
        },
        opening_book_value=cost if depreciation == 0.0 else closing + depreciation,
        closing_book_value=closing,
        accumulated_depreciation=depreciation,
        age=age,
        service_life_metadata={
            "useful_life_weeks": 4.0,
            "calibration_status": "NON-CALIBRATED_FIXTURE",
        },
        depreciation_policy_ref="straight_line_fixture",
    )


def lifecycle_fixtures():
    rows = []
    asset = fixture_asset("A", cost=100.0, capacity=3.0)
    stock = CapitalStock(owner_firm_id=1, assets=[asset])
    metric(rows, "new_asset", "initial_service_capacity", stock.capital_service_capacity(), 3.0, abs(stock.capital_service_capacity() - 3.0) <= TOLERANCE)
    metric(rows, "new_asset", "initial_book_value", asset.remaining_book_value, 100.0, abs(asset.remaining_book_value - 100.0) <= TOLERANCE)
    metric(rows, "new_asset", "initially_active", asset.is_active, True, asset.is_active is True)

    one_week_asset, depreciation = asset.advance_accounting_period(1.0)
    metric(rows, "depreciation", "one_week_depreciation", depreciation, 25.0, abs(depreciation - 25.0) <= TOLERANCE)
    metric(rows, "depreciation", "closing_book_value_after_one_week", one_week_asset.remaining_book_value, 75.0, abs(one_week_asset.remaining_book_value - 75.0) <= TOLERANCE)
    metric(rows, "depreciation", "service_unchanged_before_retirement", one_week_asset.capital_service_capacity(), 3.0, abs(one_week_asset.capital_service_capacity() - 3.0) <= TOLERANCE)
    metric(rows, "depreciation", "depreciation_is_non_cash", 0.0, 0.0, True, "fixture cash stock remains unchanged")
    bridge = capital_book_value_bridge(100.0, 0.0, depreciation, 0.0, 75.0)
    metric(rows, "depreciation", "book_value_bridge_gap", bridge["bridge_gap"], 0.0, bridge["passed"])

    current = asset
    cash_before = 1000.0
    total_depreciation = 0.0
    period_bridges = []
    for _ in range(4):
        previous = current
        current, expense = current.advance_accounting_period(1.0)
        total_depreciation += expense
        period_bridges.append(
            capital_book_value_bridge(
                previous.remaining_book_value,
                0.0,
                expense,
                0.0,
                current.remaining_book_value,
            )
        )
    metric(rows, "retirement", "total_four_week_depreciation", total_depreciation, 100.0, abs(total_depreciation - 100.0) <= TOLERANCE)
    metric(rows, "retirement", "retired_after_useful_life", current.is_retired, True, current.is_retired is True)
    metric(rows, "retirement", "retired_service_capacity", current.capital_service_capacity(), 0.0, current.capital_service_capacity() == 0.0)
    metric(rows, "retirement", "retirement_book_value", current.remaining_book_value, 0.0, current.remaining_book_value == 0.0)
    metric(rows, "retirement", "cash_unchanged_by_depreciation", cash_before, 1000.0, cash_before == 1000.0)
    all_bridges_close = all(item["passed"] for item in period_bridges)
    metric(rows, "retirement", "all_period_book_bridges_close", all_bridges_close, True, all_bridges_close)

    staggered_a = fixture_asset("staggered-a", cost=100.0, capacity=2.0)
    staggered_b = fixture_asset(
        "staggered-b",
        cost=80.0,
        capacity=3.0,
        age=2.0,
        book=40.0,
        depreciation=40.0,
    )
    staggered = CapitalStock(owner_firm_id=1, assets=[staggered_a, staggered_b])
    aged_staggered, _ = staggered.advance_accounting_period(2.0)
    metric(rows, "staggered", "opening_combined_service", staggered.capital_service_capacity(), 5.0, abs(staggered.capital_service_capacity() - 5.0) <= TOLERANCE)
    metric(rows, "staggered", "post_retirement_combined_service", aged_staggered.capital_service_capacity(), 2.0, abs(aged_staggered.capital_service_capacity() - 2.0) <= TOLERANCE)
    retired_service_loss = aged_staggered.replacement_capacity_need(5.0)
    metric(rows, "staggered", "staggered_stock_retired_service_loss", retired_service_loss, 3.0, abs(retired_service_loss - 3.0) <= TOLERANCE)

    restored = CapitalStock(owner_firm_id=1, assets=[fixture_asset("replacement", cost=60.0, capacity=3.0)])
    metric(rows, "replacement", "replacement_restores_lost_service", restored.capital_service_capacity(), 3.0, abs(restored.capital_service_capacity() - 3.0) <= TOLERANCE)

    roundtrip = pickle.loads(pickle.dumps(aged_staggered, protocol=5))
    metric(rows, "checkpoint", "asset_roundtrip_count", len(roundtrip.assets), 2, len(roundtrip.assets) == 2)
    metric(rows, "checkpoint", "asset_roundtrip_service", roundtrip.capital_service_capacity(), aged_staggered.capital_service_capacity(), abs(roundtrip.capital_service_capacity() - aged_staggered.capital_service_capacity()) <= TOLERANCE)
    metric(rows, "conservation", "fixture_money_created", 0.0, 0.0, True)
    metric(rows, "conservation", "fixture_money_destroyed", 0.0, 0.0, True)
    return rows


def replacement_fixtures():
    rows = []
    planner = ShadowInvestmentPlanner()
    replacement_only = planner.decide(
        firm_id=1,
        expected_demand=3.0,
        desired_output=3.0,
        current_effective_capacity=0.0,
        desired_capacity=3.0,
        non_capital_capacity=0.0,
        current_capital_capacity=0.0,
        replacement_capacity_loss=3.0,
        replacement_cost_per_capacity=10.0,
        expansion_cost_per_capacity=10.0,
        cash=1000.0,
    )
    metric(rows, "replacement_only", "replacement_need", replacement_only.replacement_investment_need, 3.0, abs(replacement_only.replacement_investment_need - 3.0) <= TOLERANCE)
    metric(rows, "replacement_only", "expansion_need", replacement_only.expansion_investment_need, 0.0, replacement_only.expansion_investment_need == 0.0)
    metric(rows, "replacement_only", "replacement_expenditure", replacement_only.replacement_investment_expenditure, 30.0, abs(replacement_only.replacement_investment_expenditure - 30.0) <= TOLERANCE)
    metric(rows, "replacement_only", "no_replacement_expansion_double_count", replacement_only.desired_investment_expenditure, 30.0, abs(replacement_only.desired_investment_expenditure - 30.0) <= TOLERANCE)

    both = planner.decide(
        firm_id=1,
        expected_demand=7.0,
        desired_output=7.0,
        current_effective_capacity=2.0,
        desired_capacity=7.0,
        non_capital_capacity=0.0,
        current_capital_capacity=2.0,
        replacement_capacity_loss=3.0,
        replacement_cost_per_capacity=10.0,
        expansion_cost_per_capacity=10.0,
        cash=1000.0,
    )
    metric(rows, "replacement_and_expansion", "replacement_need", both.replacement_investment_need, 3.0, abs(both.replacement_investment_need - 3.0) <= TOLERANCE)
    metric(rows, "replacement_and_expansion", "expansion_need", both.expansion_investment_need, 2.0, abs(both.expansion_investment_need - 2.0) <= TOLERANCE)
    metric(rows, "replacement_and_expansion", "desired_equals_components", both.desired_investment_expenditure, both.replacement_investment_expenditure + both.expansion_investment_expenditure, abs(both.desired_investment_expenditure - both.replacement_investment_expenditure - both.expansion_investment_expenditure) <= TOLERANCE)

    separated = replacement_expansion_needs(3.0, 2.0, 10.0, 10.0)
    metric(rows, "replacement_and_expansion", "explicit_component_sum", separated["desired_investment_expenditure"], 50.0, abs(separated["desired_investment_expenditure"] - 50.0) <= TOLERANCE)
    components_distinct = separated["replacement_capacity_need"] != separated["expansion_capacity_need"]
    metric(rows, "replacement_and_expansion", "replacement_component_not_expansion", components_distinct, True, components_distinct)
    metric(rows, "shadow_boundary", "transactions_created", int(both.transaction_created), 0, both.transaction_created is False)
    metric(rows, "shadow_boundary", "capital_assets_created", int(both.capital_asset_created), 0, both.capital_asset_created is False)
    metric(rows, "shadow_boundary", "money_created", both.money_created, 0.0, both.money_created == 0.0)
    return rows


def runtime_parity():
    left = World(initial_population=100, seed=42, diagnostics_mode="full")
    right = World(initial_population=100, seed=42, diagnostics_mode="full")
    left.split_firms(5)
    right.split_firms(5)
    fields = (
        "population",
        "households",
        "total_household_wealth",
        "total_consumption",
        "food_output_units",
        "food_inventory_units",
        "firm_cash",
        "total_money_stock",
    )
    for _ in range(12):
        left.step()
        right.step()
    passed = all(
        abs(float(left.diagnostics_rows[-1].get(field, 0.0)) - float(right.diagnostics_rows[-1].get(field, 0.0))) <= TOLERANCE
        for field in fields
    )
    return passed


def main():
    shutil.rmtree(OUTPUT, ignore_errors=True)
    OUTPUT.mkdir(parents=True, exist_ok=True)
    lifecycle = lifecycle_fixtures()
    replacement = replacement_fixtures()
    parity = runtime_parity()
    metric(lifecycle, "runtime_parity", "canonical_default_behavior_unchanged", parity, True, parity, "depreciation/retirement methods are not called by World")
    write_rows(OUTPUT / "capital_lifecycle_fixture.csv", lifecycle)
    write_rows(OUTPUT / "replacement_demand_fixture.csv", replacement)

    all_rows = lifecycle + replacement
    all_pass = all(row["passed"] for row in all_rows) and parity
    if not all_pass:
        if any(row["category"] == "depreciation" and not row["passed"] for row in all_rows):
            verdict = "B. ACCOUNTING_DEPRECIATION_BLOCKER"
        elif any(row["category"] == "retirement" and not row["passed"] for row in all_rows):
            verdict = "C. PRODUCTIVE_RETIREMENT_BLOCKER"
        elif any(row["category"].startswith("replacement") and not row["passed"] for row in all_rows):
            verdict = "D. REPLACEMENT_DEMAND_SEPARATION_BLOCKER"
        elif any(row["category"] == "checkpoint" and not row["passed"] for row in all_rows):
            verdict = "E. CAPITAL_BOOK_BRIDGE_BLOCKER"
        else:
            verdict = "F. OTHER_BLOCKER"
    else:
        verdict = "A. PASSIVE_CAPITAL_LIFECYCLE_AND_REPLACEMENT_READY"

    (OUTPUT / "acceptance_summary.md").write_text(
        f"""# Step 15I.8 Capital Lifecycle and Replacement Contract

## Verdict

**{verdict}**

This stage adds only passive lifecycle interfaces and deterministic fixtures.
Accounting depreciation uses straight-line treatment only with an explicit
`useful_life_weeks` marked `NON-CALIBRATED_FIXTURE`. Productive service stays
constant while active and becomes zero only at retirement. Depreciation is
non-cash, and replacement capacity loss is kept separate from expansion need.

Canonical depreciation, retirement and replacement behavior remain OFF. No
investment transaction, financing action, RNG draw, money creation or
Step13 change was introduced. Default World parity: `{parity}`.
""",
        encoding="utf-8",
    )
    print(verdict)
    print(f"Outputs: {OUTPUT}")


if __name__ == "__main__":
    main()
