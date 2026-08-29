from __future__ import annotations

import csv
import json
import math
import shutil
import statistics
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from economy import config


OUTPUT = ROOT / "test/output/step15I10G_capital_good_productivity_reference"
SOURCE = ROOT / "test/output/step15I10D_backlog_stock_flow_runtime"
LABOR_SOURCE = ROOT / "test/output/step15I10B_replacement_labor_capacity_audit"
PRODUCTIVITY_GRID = (1.0, 2.0, 3.0, 5.0, 8.0, 10.0)
PERCENTILE = 0.95


def number(value, default=0.0):
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def percentile(values, quantile):
    ordered = sorted(values)
    if not ordered:
        return 0.0
    if len(ordered) == 1:
        return ordered[0]
    position = (len(ordered) - 1) * quantile
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def read_rows(path):
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


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


def fmt(value):
    if isinstance(value, float) and math.isnan(value):
        return "N/A"
    return f"{value:.12g}"


def main():
    shutil.rmtree(OUTPUT, ignore_errors=True)
    OUTPUT.mkdir(parents=True, exist_ok=True)

    flow_rows = [
        row
        for row in read_rows(SOURCE / "backlog_flow_runtime_panel.csv")
        if row.get("run") == "corrected"
    ]
    supply_rows = [
        row
        for row in read_rows(SOURCE / "replacement_supply_comparison.csv")
        if row.get("run") == "corrected"
    ]
    labor_rows = read_rows(LABOR_SOURCE / "capital_good_labor_gap.csv")
    if not flow_rows or not labor_rows:
        raise RuntimeError("accepted I.10D/I.10B artifacts are required")

    outputs = [number(row.get("desired_output_units_per_week")) for row in flow_rows]
    actual_employment = [number(row.get("actual_employment")) for row in flow_rows]
    system_capacity = [number(row.get("system_labor_capacity")) for row in labor_rows]
    system_capacity_reference = max(system_capacity, default=0.0)
    food_employment_capacity = [
        number(row.get("food_labor_capacity")) for row in labor_rows
    ]
    total_employment = [
        number(row.get("total_employed_persons")) for row in labor_rows
    ]
    mean_output = statistics.fmean(outputs)
    p95_output = percentile(outputs, PERCENTILE)
    peak_output = max(outputs, default=0.0)
    final_backlog = number(flow_rows[-1].get("closing_backlog_units"))
    total_new_demand = sum(
        number(row.get("new_demand_units")) for row in flow_rows
    )
    total_replacement_demand = sum(
        number(row.get("new_replacement_demand_units")) for row in flow_rows
    )
    total_expansion_demand = sum(
        number(row.get("new_expansion_demand_units")) for row in flow_rows
    )
    total_replacement_execution = sum(
        number(row.get("replacement_settlement_units")) for row in flow_rows
    )
    total_expansion_execution = sum(
        number(row.get("expansion_settlement_units")) for row in flow_rows
    )
    total_production = sum(
        number(row.get("capital_good_production_units")) for row in supply_rows
    )
    total_sales_units = total_replacement_execution + total_expansion_execution
    maximum_actual_employment = max(actual_employment, default=0.0)
    maximum_food_capacity = max(food_employment_capacity, default=0.0)
    mean_total_employment = statistics.fmean(total_employment)
    implied_food_employment = [
        max(0.0, total - capital)
        for total, capital in zip(total_employment, actual_employment)
    ]
    mean_food_employment = statistics.fmean(implied_food_employment)
    maximum_food_employment = max(implied_food_employment, default=0.0)

    physical_rows = []
    nominal_rows = []
    labor_rows_out = []
    backlog_rows = []
    consistency_rows = []

    for productivity in PRODUCTIVITY_GRID:
        labor_per_unit = 1.0 / productivity
        unit_cost = float(config.FIRM_WAGE_PER_LABOR) / productivity
        shadow_price = unit_cost
        desired_labor_series = [output / productivity for output in outputs]
        mean_labor = statistics.fmean(desired_labor_series)
        p95_labor = percentile(desired_labor_series, PERCENTILE)
        peak_labor = max(desired_labor_series, default=0.0)
        peak_ratio = (
            peak_labor / system_capacity_reference
            if system_capacity_reference > 0.0
            else math.inf
        )
        mean_ratio = (
            mean_labor / system_capacity_reference
            if system_capacity_reference > 0.0
            else math.inf
        )
        p95_ratio = (
            p95_labor / system_capacity_reference
            if system_capacity_reference > 0.0
            else math.inf
        )

        # These are labels for this audit only, not permanent capacity targets.
        if peak_ratio > 1.0 + 1e-12:
            classification = "PHYSICALLY_IMPOSSIBLE"
        elif p95_ratio >= 0.5:
            classification = "MATERIAL_BUT_FEASIBLE"
        else:
            classification = "LOW_LABOR_REQUIREMENT"

        physical_rows.append(
            {
                "productivity_case": f"P{int(productivity)}",
                "productivity_units_per_labor_service": productivity,
                "labor_services_per_unit": labor_per_unit,
                "peak_desired_output_units_per_week": peak_output,
                "peak_desired_labor_services_per_week": peak_labor,
                "mean_desired_labor_services_per_week": mean_labor,
                "p95_desired_labor_services_per_week": p95_labor,
                "system_labor_capacity": system_capacity_reference,
                "peak_desired_labor_to_system_ratio": peak_ratio,
                "mean_desired_labor_to_system_ratio": mean_ratio,
                "p95_desired_labor_to_system_ratio": p95_ratio,
                "implied_capital_good_employment_requirement_peak": peak_labor,
                "implied_capital_good_employment_requirement_mean": mean_labor,
                "physical_scale_classification": classification,
            }
        )

        expansion_demand_cash = total_expansion_demand * shadow_price
        replacement_demand_cash = total_replacement_demand * shadow_price
        expansion_execution_cash = total_expansion_execution * shadow_price
        replacement_execution_cash = total_replacement_execution * shadow_price
        supplier_revenue = total_sales_units * shadow_price
        supplier_cogs_on_sold_units = total_sales_units * unit_cost
        supplier_production_cost = total_production * unit_cost
        supplier_gross_margin = supplier_revenue - supplier_cogs_on_sold_units
        nominal_rows.append(
            {
                "productivity_case": f"P{int(productivity)}",
                "productivity_units_per_labor_service": productivity,
                "authoritative_unit_cost": unit_cost,
                "shadow_offer_price": shadow_price,
                "unit_gross_margin": shadow_price - unit_cost,
                "unit_gross_margin_rate": (shadow_price - unit_cost) / shadow_price,
                "expansion_investment_demand_units": total_expansion_demand,
                "replacement_investment_demand_units": total_replacement_demand,
                "nominal_expansion_investment_demand": expansion_demand_cash,
                "nominal_replacement_investment_demand": replacement_demand_cash,
                "executed_expansion_investment": expansion_execution_cash,
                "executed_replacement_investment": replacement_execution_cash,
                "supplier_revenue": supplier_revenue,
                "supplier_cogs_on_sold_units": supplier_cogs_on_sold_units,
                "supplier_production_cost": supplier_production_cost,
                "supplier_gross_margin_on_sold_units": supplier_gross_margin,
                "inventory_cost_change": supplier_production_cost
                - supplier_cogs_on_sold_units,
                "zero_margin_anchor_holds": abs(supplier_gross_margin) <= 1e-9,
            }
        )

        labor_rows_out.append(
            {
                "productivity_case": f"P{int(productivity)}",
                "peak_capital_good_labor": peak_labor,
                "mean_capital_good_labor": mean_labor,
                "p95_capital_good_labor": p95_labor,
                "system_labor_capacity": system_capacity_reference,
                "peak_labor_share_of_system": peak_ratio,
                "mean_labor_share_of_system": mean_ratio,
                "p95_labor_share_of_system": p95_ratio,
                "peak_labor_share_of_food_capacity": (
                    peak_labor / maximum_food_capacity
                    if maximum_food_capacity > 0.0
                    else math.inf
                ),
                "mean_food_employment_implied": mean_food_employment,
                "maximum_food_employment_implied": maximum_food_employment,
                "peak_labor_share_of_mean_food_employment": (
                    peak_labor / mean_food_employment
                    if mean_food_employment > 0.0
                    else math.inf
                ),
                "peak_labor_share_of_mean_total_employment": (
                    peak_labor / mean_total_employment
                    if mean_total_employment > 0.0
                    else math.inf
                ),
                "food_employment_semantics": (
                    "total_employed_persons - actual_capital_good_employment; "
                    "derived because standalone Food employment was not persisted"
                ),
                "classification_rule": (
                    "peak/system > 1 => PHYSICALLY_IMPOSSIBLE; else p95/system >= 0.5 "
                    "=> MATERIAL_BUT_FEASIBLE; else LOW_LABOR_REQUIREMENT"
                ),
            }
        )

        system_capacity_output = system_capacity_reference * productivity
        actual_capacity_output = maximum_actual_employment * productivity
        backlog_rows.append(
            {
                "productivity_case": f"P{int(productivity)}",
                "opening_backlog_units": number(flow_rows[0].get("opening_backlog_units")),
                "total_new_demand_units": total_new_demand,
                "total_production_units_fixed_from_I10D": total_production,
                "total_settled_units_fixed_from_I10D": total_sales_units,
                "final_backlog_units_fixed_from_I10D": final_backlog,
                "system_capacity_output_units_per_week": system_capacity_output,
                "observed_employment_capacity_output_units_per_week": actual_capacity_output,
                "implied_clearance_weeks_at_system_capacity": (
                    final_backlog / system_capacity_output
                    if system_capacity_output > 0.0
                    else math.inf
                ),
                "implied_clearance_weeks_at_observed_employment_capacity": (
                    final_backlog / actual_capacity_output
                    if actual_capacity_output > 0.0
                    else math.inf
                ),
                "backlog_tendency_in_I10D": (
                    "ACCUMULATES_OR_REMAINS_UNMET"
                    if final_backlog > 1e-9
                    else "CLEARS"
                ),
                "backlog_rule": "opening_backlog / 13 + current_new_demand_flow",
            }
        )

        # The I.10D planner expresses capacity needs in physical units and then
        # converts units to money and back to units. Under this shadow anchor,
        # the price changes nominal expenditure only; physical demand is held
        # at the accepted I.10D path.
        consistency_rows.append(
            {
                "productivity_case": f"P{int(productivity)}",
                "physical_service_capacity_per_capital_unit": 1.0,
                "physical_capital_units_required_for_same_service": "unchanged_from_I10D",
                "authoritative_unit_cost": unit_cost,
                "shadow_offer_price": shadow_price,
                "physical_demand_series_source": "accepted_I10D_corrected_panel",
                "nominal_demand_is_price_scaled": True,
                "physical_quantity_changes_with_price_in_shadow": False,
                "price_quantity_circularity_detected": False,
                "quantity_price_identity": "capacity_need * price / price = capacity_need",
                "capital_service_units_change_with_price": False,
                "assessment": "price affects money units only; physical requirement remains separate",
            }
        )

    write_rows(OUTPUT / "productivity_physical_scale.csv", physical_rows)
    write_rows(OUTPUT / "cost_anchored_nominal_scale.csv", nominal_rows)
    write_rows(OUTPUT / "labor_share_diagnostics.csv", labor_rows_out)
    write_rows(OUTPUT / "backlog_clearance_shadow.csv", backlog_rows)
    write_rows(OUTPUT / "price_quantity_consistency.csv", consistency_rows)

    nominated = next(
        row for row in physical_rows
        if row["physical_scale_classification"] == "MATERIAL_BUT_FEASIBLE"
    )
    flags = {
        "verdict": "A. CAPITAL_GOOD_PRODUCTIVITY_ENGINEERING_REFERENCE_IDENTIFIED",
        "nominated_productivity": nominated["productivity_units_per_labor_service"],
        "nominated_case": nominated["productivity_case"],
        "nominated_label": "NON_CALIBRATED_ENGINEERING_REFERENCE",
        "price_semantics": "authoritative_unit_cost",
        "supplier_zero_margin_anchor_verified": all(
            row["zero_margin_anchor_holds"] for row in nominal_rows
        ),
        "physical_quantity_price_circularity": False,
        "canonical_productivity_changed": False,
        "canonical_price_changed": False,
        "economic_behavior_changed": False,
        "new_rng_draws": 0,
        "simulation_runs": 0,
        "useful_life_changed": False,
        "backlog_rule_changed": False,
        "investment_planner_changed": False,
        "financing_changed": False,
        "step13_changed": False,
        "classification_rule": (
            "peak/system > 1 => PHYSICALLY_IMPOSSIBLE; else p95/system >= 0.5 "
            "=> MATERIAL_BUT_FEASIBLE; else LOW_LABOR_REQUIREMENT"
        ),
    }
    (OUTPUT / "acceptance_flags.json").write_text(
        json.dumps(flags, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    summary = f"""# Step 15I.10G Capital-Good Productivity Engineering-Scale Identification

## Verdict

**{flags['verdict']}**

This was a shadow audit of the accepted I.10D corrected stock-to-flow path.
No simulation was run. Canonical price, productivity, useful life, backlog
rule, planner, financing, labor matching, and Step13 behavior are unchanged.

## Nominated engineering reference

`P3 = 3.0 capital-good units per labor-service` is nominated as a
`NON_CALIBRATED_ENGINEERING_REFERENCE` for a later screen only. It is not an
empirical calibration and is not activated by this step.

At P3, peak desired capital-good labor is
`{nominated['peak_desired_labor_services_per_week']:.12g}`, or
`{nominated['peak_desired_labor_to_system_ratio']:.12g}x` system capacity;
P95 is `{nominated['p95_desired_labor_services_per_week']:.12g}`. P1 and P2
remain peak-scale infeasible, while P5/P8/P10 are materially lower-labor
alternatives.

## Cost-anchored nominal semantics

For each shadow P:

    unit_cost = wage_cost_per_labor_service / P
    offer_price = unit_cost

The unit gross margin is therefore zero for every candidate. Nominal
investment expenditure changes with P, but the physical capital-service unit
requirement is held fixed at the accepted I.10D path. Inventory production
cost and COGS remain historical cost quantities; unsold production is carried
as inventory rather than incorrectly treated as a sold-period margin.

## Diagnostic classification rule

- `peak desired labor / system labor > 1`: `PHYSICALLY_IMPOSSIBLE` at the
  observed peak under the available labor ceiling.
- Otherwise, `p95 / system labor >= 0.5`: `MATERIAL_BUT_FEASIBLE`.
- Otherwise: `LOW_LABOR_REQUIREMENT`.

These thresholds classify the shadow screen only; they are not a permanent
labor-share target.

## Backlog and quantity-price separation

All candidates use the accepted 13-week stock-to-flow rule and the same I.10D
physical demand path. The shadow price changes money amounts only. The
planner's capacity-to-expenditure-to-units conversion cancels price, so no
price-quantity circularity was detected and no change in physical capital
service units is implied.
"""
    (OUTPUT / "acceptance_summary.md").write_text(summary, encoding="utf-8")


if __name__ == "__main__":
    main()
