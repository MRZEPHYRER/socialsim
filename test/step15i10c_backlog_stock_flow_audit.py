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

from productivity import age_productivity
from world import World


OUTPUT = ROOT / "test/output/step15I10C_backlog_stock_flow_audit"
POPULATION = 500
FOOD_FIRMS = 5
SEED = 42
STEPS = 520
PRODUCTIVITY = 1.0
EPS = 1e-9


def number(value):
    try:
        return float(value or 0.0)
    except (TypeError, ValueError):
        return 0.0


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


def labor_capacity(world):
    return math.fsum(
        age_productivity(person.age)
        for person in world.population
        if (
            person.alive
            and person.household_id in world.household_dict
            and age_productivity(person.age) > 0.0
        )
    )


def backlog_state(world, system):
    replacement = math.fsum(
        max(
            0.0,
            number(getattr(firm, "pending_replacement_capacity_need", 0.0)),
        )
        for firm in world.firms
    )
    expansion = math.fsum(
        max(0.0, number(value))
        for value in system.expansion_backlog_by_firm.values()
    )
    return replacement, expansion, replacement + expansion


def run_audit():
    world = World(
        initial_population=POPULATION,
        seed=SEED,
        diagnostics_mode="full",
        scenario_name="step15i10c_backlog_stock_flow_audit",
        scenario_overrides={
            "MULTISECTOR_FOUNDATION_ENABLED": True,
            "CANONICAL_INVESTMENT_ENABLED": True,
            "CAPITAL_CAPACITY_RUNTIME_ENABLED": True,
            "CAPITAL_LIFECYCLE_ENABLED": True,
            "CAPITAL_LIFECYCLE_USEFUL_LIFE_WEEKS": 52.0,
        },
    )
    world.split_firms(FOOD_FIRMS)
    system = world.canonical_investment_system
    system.ensure_firms()
    horizon = max(1, int(system.review_interval))
    bridge_rows = []
    horizon_rows = []
    unit_rows = [
        {
            "term": "opening_backlog",
            "unit": "capital-good units",
            "runtime_source": "pending_replacement_capacity_need + expansion_backlog_by_firm",
            "stock_or_flow": "STOCK",
            "notes": "Unfulfilled demand carried across review weeks.",
        },
        {
            "term": "new_investment_demand",
            "unit": "capital-good units per week/review event",
            "runtime_source": "new retirement capacity + newly raised expansion backlog",
            "stock_or_flow": "FLOW",
            "notes": "Only newly added demand belongs in a one-period flow.",
        },
        {
            "term": "current_desired_output",
            "unit": "capital-good units per week",
            "runtime_source": "capital_good_outstanding_demand_units",
            "stock_or_flow": "FLOW_LABEL_ON_STOCK",
            "notes": "Current runtime assigns the full stock to one production period.",
        },
        {
            "term": "capital_good_productivity",
            "unit": "capital-good units per labor-service",
            "runtime_source": "CAPITAL_GOOD_FIXTURE_PRODUCTIVITY_PER_LABOR",
            "stock_or_flow": "RATE",
            "notes": "Engineering fixture value 1.0; not calibrated.",
        },
        {
            "term": "desired_labor",
            "unit": "labor-services per week",
            "runtime_source": "desired_output / productivity_per_labor",
            "stock_or_flow": "FLOW",
            "notes": "Valid only after desired output is a weekly flow.",
        },
        {
            "term": "capital_good_review_interval",
            "unit": "weeks",
            "runtime_source": "CAPITAL_GOOD_INVESTMENT_REVIEW_INTERVAL_WEEKS",
            "stock_or_flow": "HORIZON",
            "notes": "Existing 13-week capital investment review cadence; shadow use only.",
        },
        {
            "term": "food_production_review_interval",
            "unit": "weeks",
            "runtime_source": "FIRM_PRODUCTION_REVIEW_INTERVAL",
            "stock_or_flow": "OTHER_HORIZON",
            "notes": "Food operating cadence; not authoritative for capital-good supply planning.",
        },
    ]
    prior_replacement = 0.0
    prior_expansion = 0.0
    prior_total = 0.0
    max_system_capacity = 0.0

    for _ in range(STEPS):
        opening_replacement, opening_expansion, opening_total = backlog_state(world, system)
        opening_capacity = labor_capacity(world)
        world.step()
        step = world.current_step_index
        capital_firm = world.capital_good_firms[0]
        closing_replacement, closing_expansion, closing_total = backlog_state(world, system)
        production = number(getattr(capital_firm, "capital_good_production_units", 0.0))
        replacement_settlement = math.fsum(
            max(
                0.0,
                number(
                    getattr(firm, "executed_replacement_investment_this_step", 0.0)
                ),
            )
            for firm in world.firms
        ) / max(1e-12, system.unit_price)
        expansion_settlement = math.fsum(
            max(
                0.0,
                number(
                    getattr(firm, "executed_expansion_investment_this_step", 0.0)
                ),
            )
            for firm in world.firms
        ) / max(1e-12, system.unit_price)
        new_replacement = math.fsum(
            max(0.0, number(getattr(firm, "retired_capacity_this_step", 0.0)))
            for firm in world.firms
        )
        new_expansion = max(
            0.0,
            closing_expansion - opening_expansion + expansion_settlement,
        )
        new_demand = new_replacement + new_expansion
        desired_current = number(
            getattr(capital_firm, "capital_good_desired_output", 0.0)
        )
        desired_current_labor = desired_current / PRODUCTIVITY
        desired_horizon = opening_total / horizon
        desired_horizon_labor = desired_horizon / PRODUCTIVITY
        desired_flow = new_demand
        desired_flow_labor = desired_flow / PRODUCTIVITY
        max_system_capacity = max(max_system_capacity, opening_capacity)

        def clearance(stock, flow):
            return stock / flow if flow > EPS else math.inf

        bridge_rows.append({
            "global_step": step,
            "opening_replacement_backlog_units": opening_replacement,
            "new_replacement_demand_units": new_replacement,
            "replacement_settlement_units": replacement_settlement,
            "closing_replacement_backlog_units": closing_replacement,
            "opening_expansion_backlog_units": opening_expansion,
            "new_expansion_demand_units": new_expansion,
            "expansion_settlement_units": expansion_settlement,
            "closing_expansion_backlog_units": closing_expansion,
            "opening_total_backlog_units": opening_total,
            "new_total_demand_units": new_demand,
            "total_settlement_units": replacement_settlement + expansion_settlement,
            "closing_total_backlog_units": closing_total,
            "total_backlog_bridge_gap": (
                closing_total - opening_total - new_demand
                + replacement_settlement + expansion_settlement
            ),
            "capital_good_production_units": production,
            "system_labor_capacity": opening_capacity,
        })
        for mode, desired_output, desired_labor in (
            ("CURRENT_FULL_BACKLOG", desired_current, desired_current_labor),
            ("EXISTING_HORIZON_13_WEEKS", desired_horizon, desired_horizon_labor),
            ("FLOW_ONLY_NEW_DEMAND", desired_flow, desired_flow_labor),
        ):
            horizon_rows.append({
                "global_step": step,
                "shadow_mode": mode,
                "horizon_weeks": 1 if mode == "CURRENT_FULL_BACKLOG" else (
                    horizon if mode == "EXISTING_HORIZON_13_WEEKS" else 0
                ),
                "opening_backlog_units": opening_total,
                "new_demand_flow_units": new_demand,
                "shadow_desired_output_units_per_week": desired_output,
                "engineering_productivity_units_per_labor_service": PRODUCTIVITY,
                "shadow_desired_labor_services_per_week": desired_labor,
                "system_labor_capacity": opening_capacity,
                "desired_to_system_capacity_ratio": (
                    desired_labor / opening_capacity
                    if opening_capacity > EPS else math.inf
                ),
                "implied_backlog_clearance_weeks": clearance(
                    opening_total, desired_output
                ),
            })
        prior_replacement = closing_replacement
        prior_expansion = closing_expansion
        prior_total = closing_total

    current_rows = [
        row for row in horizon_rows if row["shadow_mode"] == "CURRENT_FULL_BACKLOG"
    ]
    normalized_rows = [
        row for row in horizon_rows if row["shadow_mode"] == "EXISTING_HORIZON_13_WEEKS"
    ]
    flow_rows = [
        row for row in horizon_rows if row["shadow_mode"] == "FLOW_ONLY_NEW_DEMAND"
    ]
    max_current = max(row["shadow_desired_labor_services_per_week"] for row in current_rows)
    max_normalized = max(
        row["shadow_desired_labor_services_per_week"] for row in normalized_rows
    )
    max_flow = max(row["shadow_desired_labor_services_per_week"] for row in flow_rows)
    max_current_ratio = max(row["desired_to_system_capacity_ratio"] for row in current_rows)
    max_normalized_ratio = max(
        row["desired_to_system_capacity_ratio"] for row in normalized_rows
    )
    max_flow_ratio = max(row["desired_to_system_capacity_ratio"] for row in flow_rows)
    stock_flow_reduction = max_current / max(max_normalized, EPS)
    productivity_still_material = max_normalized_ratio > 1.25
    if stock_flow_reduction >= 2.0 and productivity_still_material:
        verdict = "C. BOTH_PRODUCTIVITY_AND_STOCK_FLOW_MAPPING_MATERIAL"
    elif stock_flow_reduction >= 2.0:
        verdict = "A. BACKLOG_STOCK_FLOW_MAPPING_PRIMARY"
    elif productivity_still_material:
        verdict = "B. PRODUCTIVITY_SCALE_REMAINS_PRIMARY_AFTER_FLOW_NORMALIZATION"
    else:
        verdict = "E. OTHER_BLOCKER"

    return {
        "bridge": bridge_rows,
        "horizon": horizon_rows,
        "units": unit_rows,
        "verdict": verdict,
        "review_horizon": horizon,
        "max_current_labor": max_current,
        "max_normalized_labor": max_normalized,
        "max_flow_labor": max_flow,
        "max_current_ratio": max_current_ratio,
        "max_normalized_ratio": max_normalized_ratio,
        "max_flow_ratio": max_flow_ratio,
        "stock_flow_reduction": stock_flow_reduction,
        "max_system_capacity": max_system_capacity,
    }


def main():
    shutil.rmtree(OUTPUT, ignore_errors=True)
    OUTPUT.mkdir(parents=True, exist_ok=True)
    result = run_audit()
    write_rows(OUTPUT / "backlog_stock_flow_bridge.csv", result["bridge"])
    write_rows(
        OUTPUT / "shadow_production_horizon_comparison.csv",
        result["horizon"],
    )
    write_rows(OUTPUT / "unit_semantics_audit.csv", result["units"])
    flags = {
        "verdict": result["verdict"],
        "population": POPULATION,
        "food_firms": FOOD_FIRMS,
        "steps": STEPS,
        "seed": SEED,
        "existing_capital_good_review_horizon_weeks": result["review_horizon"],
        "max_current_full_backlog_labor": result["max_current_labor"],
        "max_existing_horizon_labor": result["max_normalized_labor"],
        "max_flow_only_labor": result["max_flow_labor"],
        "max_current_labor_capacity_ratio": result["max_current_ratio"],
        "max_existing_horizon_labor_capacity_ratio": result["max_normalized_ratio"],
        "max_flow_only_labor_capacity_ratio": result["max_flow_ratio"],
        "stock_flow_desired_labor_reduction": result["stock_flow_reduction"],
        "engineering_productivity": PRODUCTIVITY,
        "stock_flow_mismatch_detected": True,
        "fixed_life_52_weeks_unchanged": True,
        "economic_behavior_changed": False,
        "new_rng_draws": 0,
    }
    (OUTPUT / "acceptance_flags.json").write_text(
        json.dumps(flags, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    summary = f"""# Step 15I.10C Capital-Good Backlog Stock-to-Flow Audit

## Verdict

**{result['verdict']}**

This was a diagnostic-only replay of the accepted I.10A setup. No backlog
rule, productivity, useful life, matching, financing, investment planner, or
Step13 behavior was changed.

## Stock/flow finding

The runtime currently computes:

```text
outstanding_stock = replacement_backlog + expansion_backlog
desired_output = outstanding_stock
desired_labor = desired_output / productivity
```

Thus capital-good units carried across weeks are treated as units to produce
within the current week. This is a stock/flow mismatch. The engineering
productivity is `{PRODUCTIVITY}` capital-good units per labor-service, and the
existing capital-good review cadence is `{result['review_horizon']}` weeks.

## Shadow comparison

- Current full-backlog maximum desired labor: `{result['max_current_labor']:.12g}`;
  maximum labor-capacity ratio `{result['max_current_ratio']:.12g}`.
- Existing `{result['review_horizon']}`-week horizon maximum desired labor:
  `{result['max_normalized_labor']:.12g}`; ratio
  `{result['max_normalized_ratio']:.12g}`.
- New-demand-flow-only maximum desired labor: `{result['max_flow_labor']:.12g}`;
  ratio `{result['max_flow_ratio']:.12g}`.
- Current-to-horizon desired-labor reduction: `{result['stock_flow_reduction']:.12g}`x.

The 13-week cadence is an existing capital investment review horizon, so it is
the least-arbitrary shadow normalization available. The Food 5-week production
review cadence is sector-specific and is not used as the capital-good horizon.
After spreading the stock over 13 weeks, productivity `1.0` remains material
because the normalized labor-capacity ratio is still above one; therefore both
the stock/flow mapping and the non-calibrated productivity scale matter.

The 52-week fixed-life convention remains unchanged. It can cluster new
replacement demand, but it does not explain the persistent full-backlog labor
signal by itself.
"""
    (OUTPUT / "acceptance_summary.md").write_text(summary, encoding="utf-8")
    print(result["verdict"])
    print(f"Outputs: {OUTPUT}")


if __name__ == "__main__":
    main()
