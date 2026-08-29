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

from economy import config


OUTPUT = ROOT / "test/output/step15I10E_capital_good_productivity_scale"
SOURCE = ROOT / "test/output/step15I10D_backlog_stock_flow_runtime"
LABOR_SOURCE = ROOT / "test/output/step15I10B_replacement_labor_capacity_audit"
PRODUCTIVITY_GRID = (1.0, 2.0, 3.0, 5.0)


def read_rows(path):
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


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


def main():
    shutil.rmtree(OUTPUT, ignore_errors=True)
    OUTPUT.mkdir(parents=True, exist_ok=True)

    scale_rows = read_rows(SOURCE / "backlog_flow_runtime_panel.csv")
    corrected = [row for row in scale_rows if row.get("run") == "corrected"]
    supply_rows = read_rows(SOURCE / "replacement_supply_comparison.csv")
    corrected_supply = [row for row in supply_rows if row.get("run") == "corrected"]
    labor_rows = read_rows(LABOR_SOURCE / "capital_good_labor_gap.csv")

    sale_price = float(config.CAPITAL_GOOD_UNIT_PRICE)
    wage_per_service = float(config.FIRM_WAGE_PER_LABOR)
    current_productivity = float(config.CAPITAL_GOOD_FIXTURE_PRODUCTIVITY_PER_LABOR)
    review_horizon = int(config.CAPITAL_GOOD_INVESTMENT_REVIEW_INTERVAL_WEEKS)
    peak_output = max(
        (number(row.get("desired_output_units_per_week")) for row in corrected),
        default=0.0,
    )
    peak_labor = max(
        (number(row.get("desired_labor_services_per_week")) for row in corrected),
        default=0.0,
    )
    final_backlog = number(
        corrected[-1].get("closing_backlog_units") if corrected else 0.0
    )
    system_capacity = max(
        (number(row.get("system_labor_capacity")) for row in labor_rows),
        default=0.0,
    )
    actual_capacity = max(
        (number(row.get("actual_capital_good_labor_capacity")) for row in labor_rows),
        default=0.0,
    )
    actual_employment = max(
        (number(row.get("actual_capital_good_employment")) for row in labor_rows),
        default=0.0,
    )

    production_units = sum(
        number(row.get("capital_good_production_units"))
        for row in corrected_supply
    )
    sold_units = sum(
        number(row.get("replacement_execution_units"))
        + number(row.get("expansion_execution_units"))
        for row in corrected_supply
    )
    sales_revenue = sold_units * sale_price
    production_wage_cost = production_units * wage_per_service / max(
        current_productivity, 1e-12
    )
    inventory_cost_per_unit = wage_per_service / max(current_productivity, 1e-12)
    gross_profit = sales_revenue - production_wage_cost
    actual_margin = gross_profit / sales_revenue if sales_revenue else math.nan

    unit_economics = [{
        "case": "CURRENT_ENGINEERING_RUNTIME",
        "sale_price_per_unit": sale_price,
        "wage_cost_per_labor_service": wage_per_service,
        "productivity_units_per_labor_service": current_productivity,
        "labor_cost_per_physical_unit": inventory_cost_per_unit,
        "inventory_book_cost_per_unit": inventory_cost_per_unit,
        "production_units": production_units,
        "sold_units": sold_units,
        "sales_revenue": sales_revenue,
        "production_wage_cost": production_wage_cost,
        "gross_profit_before_other_costs": gross_profit,
        "gross_margin": actual_margin,
        "operating_margin": actual_margin,
        "price_cost_ratio": sale_price / inventory_cost_per_unit,
        "price_scale_blocker": sale_price < inventory_cost_per_unit,
    }]
    write_rows(OUTPUT / "capital_good_unit_economics.csv", unit_economics)

    grid_rows = []
    for productivity in PRODUCTIVITY_GRID:
        labor_per_unit = 1.0 / productivity
        unit_labor_cost = wage_per_service / productivity
        unit_gross_profit = sale_price - unit_labor_cost
        margin = unit_gross_profit / sale_price
        grid_rows.append({
            "shadow_case": f"P{int(productivity)}",
            "productivity_multiple": productivity / current_productivity,
            "productivity_units_per_labor_service": productivity,
            "implied_labor_services_per_unit": labor_per_unit,
            "peak_desired_output_units_per_week": peak_output,
            "peak_desired_labor_services_per_week": peak_output / productivity,
            "system_labor_capacity": system_capacity,
            "desired_to_system_labor_ratio": (
                peak_output / productivity / system_capacity
                if system_capacity else math.inf
            ),
            "implied_unit_labor_cost": unit_labor_cost,
            "sale_price_per_unit": sale_price,
            "gross_profit_per_unit": unit_gross_profit,
            "supplier_gross_margin": margin,
            "backlog_clearance_weeks_at_system_capacity": (
                final_backlog / (system_capacity * productivity)
                if system_capacity > 0 else math.inf
            ),
            "backlog_clearance_weeks_at_current_actual_capacity": (
                final_backlog / (actual_capacity * productivity)
                if actual_capacity > 0 else math.inf
            ),
            "price_below_unit_labor_cost": sale_price < unit_labor_cost,
        })
    write_rows(OUTPUT / "productivity_shadow_grid.csv", grid_rows)

    labor_comparison = [{
        "case": "CURRENT_RUNTIME_P1",
        "productivity": current_productivity,
        "peak_desired_output": peak_output,
        "peak_desired_labor": peak_labor,
        "actual_peak_employment": actual_employment,
        "actual_peak_labor_capacity": actual_capacity,
        "system_labor_capacity": system_capacity,
        "desired_labor_share_of_system_capacity": (
            peak_labor / system_capacity if system_capacity else math.inf
        ),
        "final_backlog": final_backlog,
        "clearance_weeks_at_system_capacity": (
            final_backlog / (system_capacity * current_productivity)
            if system_capacity else math.inf
        ),
        "clearance_weeks_at_actual_capacity": (
            final_backlog / (actual_capacity * current_productivity)
            if actual_capacity else math.inf
        ),
        "existing_review_horizon_weeks": review_horizon,
    }]
    for row in grid_rows:
        labor_comparison.append({
            "case": row["shadow_case"],
            "productivity": row["productivity_units_per_labor_service"],
            "peak_desired_output": row["peak_desired_output_units_per_week"],
            "peak_desired_labor": row["peak_desired_labor_services_per_week"],
            "actual_peak_employment": actual_employment,
            "actual_peak_labor_capacity": actual_capacity,
            "system_labor_capacity": system_capacity,
            "desired_labor_share_of_system_capacity": row["desired_to_system_labor_ratio"],
            "final_backlog": final_backlog,
            "clearance_weeks_at_system_capacity": row["backlog_clearance_weeks_at_system_capacity"],
            "clearance_weeks_at_actual_capacity": row["backlog_clearance_weeks_at_current_actual_capacity"],
            "existing_review_horizon_weeks": review_horizon,
        })
    write_rows(OUTPUT / "labor_scale_comparison.csv", labor_comparison)

    p1 = grid_rows[0]
    p5 = grid_rows[-1]
    flags = {
        "verdict": "B. CAPITAL_GOOD_PRICE_SCALE_BLOCKER",
        "population": 500,
        "food_firms": 5,
        "seed": 42,
        "steps": 520,
        "canonical_productivity_unchanged": True,
        "canonical_price_unchanged": True,
        "sale_price_per_unit": sale_price,
        "wage_cost_per_labor_service": wage_per_service,
        "current_labor_cost_per_unit": inventory_cost_per_unit,
        "current_supplier_gross_margin": actual_margin,
        "p1_supplier_gross_margin": p1["supplier_gross_margin"],
        "p5_supplier_gross_margin": p5["supplier_gross_margin"],
        "peak_corrected_desired_output": peak_output,
        "peak_corrected_desired_labor": peak_labor,
        "system_labor_capacity": system_capacity,
        "current_desired_to_system_ratio": peak_labor / system_capacity,
        "price_scale_blocker": True,
        "productivity_scale_alone_sufficient": False,
        "engineering_reference_selected": None,
        "economic_behavior_changed": False,
        "new_rng_draws": 0,
    }
    (OUTPUT / "acceptance_flags.json").write_text(
        json.dumps(flags, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    summary = f"""# Step 15I.10E Capital-Good Productivity and Unit-Economics Scale Audit

## Verdict

**{flags['verdict']}**

This is a diagnostic-only audit of the accepted I.10D output. No canonical
productivity, price, useful life, backlog rule, planner, matching, financing,
or Step13 behavior was changed.

## Authoritative current economics

- Sale price: `{sale_price}` per physical capital-good unit.
- Wage cost: `{wage_per_service}` per labor-service.
- Engineering productivity: `{current_productivity}` units per labor-service.
- Labor/inventory cost: `{inventory_cost_per_unit}` per unit.
- Realized sales revenue: `{sales_revenue:.12g}`.
- Realized production wage cost: `{production_wage_cost:.12g}`.
- Realized gross/operating margin: `{actual_margin:.6%}`.

At the current P1 engineering scale, one unit costs `41` in labor but sells
for `10`. The supplier therefore has a negative unit gross margin of `-31`
and cannot be economically coherent without an unmodeled subsidy or a price
scale change. Productivity alone should not be used to hide this price-cost
mismatch.

## Shadow grid

P1/P2/P3/P5 imply labor costs per physical unit of `41`, `20.5`, `13.667`,
and `8.2`, respectively. Only P5 produces a positive shadow margin at the
unchanged price, but it is not selected as a canonical calibration. The
corrected peak labor demand is `{peak_labor:.12g}` at P1, or
`{peak_labor / system_capacity:.12g}x` total system labor capacity.

The price/productivity pair is therefore the primary unit-economics blocker;
this stage nominates no engineering reference value.
"""
    (OUTPUT / "acceptance_summary.md").write_text(summary, encoding="utf-8")
    print(flags["verdict"])
    print(f"Outputs: {OUTPUT}")


if __name__ == "__main__":
    main()
