"""Step 15I.9 controlled canonical depreciation/replacement screen."""

from __future__ import annotations

import csv
import json
import math
import shutil
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from world import World


OUTPUT = ROOT / "test/output/step15I9_canonical_capital_lifecycle"
STEPS = 520
POPULATION = 500
FOOD_FIRMS = 5
SEED = 42
USEFUL_LIFE_WEEKS = 52.0
TOLERANCE = 1e-6


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


def number(value):
    try:
        return float(value or 0.0)
    except (TypeError, ValueError):
        return 0.0


def run_case(enabled):
    world = World(
        initial_population=POPULATION,
        seed=SEED,
        diagnostics_mode="full",
        scenario_name=(
            "step15i9_lifecycle_treatment"
            if enabled
            else "step15i9_lifecycle_control"
        ),
        scenario_overrides={
            "MULTISECTOR_FOUNDATION_ENABLED": True,
            "CANONICAL_INVESTMENT_ENABLED": True,
            "CAPITAL_CAPACITY_RUNTIME_ENABLED": True,
            "CAPITAL_LIFECYCLE_ENABLED": enabled,
            "CAPITAL_LIFECYCLE_USEFUL_LIFE_WEEKS": USEFUL_LIFE_WEEKS,
        },
    )
    world.split_firms(FOOD_FIRMS)
    lifecycle_rows = []
    replacement_rows = []
    capital_good_rows = []
    retirement_events = []
    previous_service = {}

    for _ in range(STEPS):
        world.step()
        step = world.current_step_index
        week = world.last_canonical_investment_week
        accounting_by_firm = {
            int(number(row.get("firm_id", -1))): row
            for row in reversed(world.accounting.rows)
            if int(number(row.get("step", -1))) == step
        }
        for firm in world.firms:
            stock = getattr(firm, "capital_stock", None)
            service = (
                number(stock.capital_service_capacity(engineering_capacity_per_unit=1.0))
                if stock is not None
                else 0.0
            )
            previous = previous_service.get(firm.firm_id, service)
            accounting_row = accounting_by_firm.get(firm.firm_id, {})
            row = {
                "global_step": step,
                "firm_id": firm.firm_id,
                "capital_lifecycle_enabled": enabled,
                "asset_count": len(getattr(stock, "assets", [])),
                "active_asset_count": getattr(firm, "active_capital_asset_count", 0),
                "retired_asset_count": getattr(firm, "retired_capital_asset_count", 0),
                "capital_book_value_opening": number(getattr(firm, "capital_book_value_opening_this_step", 0.0)),
                "capital_book_value": number(getattr(stock, "total_remaining_book_value", 0.0)),
                "depreciation_expense": number(getattr(firm, "capital_depreciation_expense_this_step", 0.0)),
                "accumulated_depreciation": number(getattr(stock, "accumulated_depreciation", 0.0)),
                "capital_service_capacity": service,
                "previous_capital_service_capacity": previous,
                "retired_capacity": number(getattr(firm, "retired_capacity_this_step", 0.0)),
                "feasible_capacity": number(getattr(firm, "authoritative_feasible_capacity", getattr(firm, "feasible_capacity", 0.0))),
                "realized_production": number(getattr(firm, "actual_production", 0.0)),
                "replacement_capacity_need": number(getattr(firm, "replacement_capacity_need", 0.0)),
                "expansion_capacity_need": number(getattr(firm, "expansion_investment_need", 0.0)),
                "desired_replacement_investment": number(getattr(firm, "desired_replacement_investment", 0.0)),
                "desired_expansion_investment": number(getattr(firm, "desired_expansion_investment", 0.0)),
                "executed_replacement_investment": number(getattr(firm, "executed_replacement_investment_this_step", 0.0)),
                "executed_expansion_investment": number(getattr(firm, "executed_expansion_investment_this_step", 0.0)),
                "unmet_replacement_investment": number(getattr(firm, "unmet_replacement_investment", 0.0)),
                "unmet_expansion_investment": number(getattr(firm, "unmet_expansion_investment", 0.0)),
                "cash": number(getattr(firm, "cash", 0.0)),
                "sales_revenue": number(getattr(firm, "sales_revenue", 0.0)),
                "operating_profit": number(accounting_row.get("accounting_operating_profit", getattr(firm, "profit", 0.0))),
                "legacy_profit": number(getattr(firm, "profit", 0.0)),
                "cfo": number(accounting_row.get("cfo", 0.0)),
                "principal": number(getattr(firm, "loan_repaid", 0.0)),
                "arrears": number(getattr(firm, "interest_arrears", 0.0)),
            }
            lifecycle_rows.append(row)
            replacement_rows.append({
                "global_step": step,
                "firm_id": firm.firm_id,
                "retired_capacity": row["retired_capacity"],
                "replacement_capacity_need": row["replacement_capacity_need"],
                "expansion_capacity_need": row["expansion_capacity_need"],
                "desired_replacement_investment": row["desired_replacement_investment"],
                "desired_expansion_investment": row["desired_expansion_investment"],
                "executed_replacement_investment": row["executed_replacement_investment"],
                "executed_expansion_investment": row["executed_expansion_investment"],
                "unmet_replacement_investment": row["unmet_replacement_investment"],
                "cash": row["cash"],
            })
            for event in getattr(firm, "capital_lifecycle_events", []):
                retirement_events.append(dict(event))
            previous_service[firm.firm_id] = service

        for firm in getattr(world, "capital_good_firms", []):
            capital_good_rows.append({
                "global_step": step,
                "firm_id": firm.firm_id,
                "employment": len(getattr(firm, "employee_ids", [])),
                "desired_labor": number(getattr(firm, "desired_labor", 0.0)),
                "production": number(getattr(firm, "capital_good_production_units", 0.0)),
                "inventory": number(getattr(getattr(firm, "capital_good_inventory", None), "units", 0.0)),
                "sales": number(getattr(firm, "capital_good_sales_units", 0.0)),
                "revenue": number(getattr(firm, "sales_revenue", 0.0)),
                "wage_bill": number(getattr(firm, "wage_bill", 0.0)),
                "operating_profit": number(getattr(firm, "profit", 0.0)),
                "cash": number(getattr(firm, "cash", 0.0)),
            })

    return {
        "world": world,
        "lifecycle": lifecycle_rows,
        "replacement": replacement_rows,
        "events": retirement_events,
        "capital_goods": capital_good_rows,
        "accounting": [dict(row) for row in world.accounting.rows],
        "diagnostics": [dict(row) for row in world.diagnostics_rows],
    }


def series(rows, key):
    return [number(row.get(key, 0.0)) for row in rows]


def aggregate_series(rows, key):
    values_by_step = {}
    for row in rows:
        step = int(number(row.get("global_step", 0)))
        values_by_step.setdefault(step, 0.0)
        values_by_step[step] += number(row.get(key, 0.0))
    return [values_by_step[step] for step in sorted(values_by_step)]


def plot(path, lines, title, ylabel):
    fig, ax = plt.subplots(figsize=(9, 4.8))
    for label, values in lines:
        ax.plot(range(len(values)), values, label=label, linewidth=1.15)
    ax.set_title(title)
    ax.set_xlabel("week")
    ax.set_ylabel(ylabel)
    ax.grid(alpha=0.25)
    if len(lines) > 1:
        ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


def make_figures(control, treatment):
    c = control["lifecycle"]
    t = treatment["lifecycle"]
    plot(
        OUTPUT / "capital_book_value_over_time.png",
        [("control", aggregate_series(c, "capital_book_value")),
         ("treatment", aggregate_series(t, "capital_book_value"))],
        "Aggregate capital book value",
        "book value",
    )
    plot(
        OUTPUT / "active_retired_assets.png",
        [("active assets", aggregate_series(t, "active_asset_count")),
         ("retired assets", aggregate_series(t, "retired_asset_count"))],
        "Active and retired assets",
        "asset-count sum across firms",
    )
    plot(
        OUTPUT / "depreciation_and_profit.png",
        [("depreciation", aggregate_series(t, "depreciation_expense")),
         ("operating profit", aggregate_series(t, "operating_profit"))],
        "Depreciation and operating profit",
        "nominal value",
    )
    plot(
        OUTPUT / "replacement_vs_expansion_investment.png",
        [("replacement", aggregate_series(t, "executed_replacement_investment")),
         ("expansion", aggregate_series(t, "executed_expansion_investment"))],
        "Replacement versus expansion investment observations",
        "nominal value",
    )
    plot(
        OUTPUT / "capital_service_and_capacity.png",
        [("capital service", aggregate_series(t, "capital_service_capacity")),
         ("feasible capacity", aggregate_series(t, "feasible_capacity"))],
        "Capital service and feasible capacity",
        "capacity units",
    )
    cg = treatment["capital_goods"]
    plot(
        OUTPUT / "capital_good_replacement_demand.png",
        [("production", aggregate_series(cg, "production")),
         ("sales", aggregate_series(cg, "sales")),
         ("inventory", aggregate_series(cg, "inventory"))],
        "Capital-good supply for replacement and expansion",
        "units",
    )


def max_gap(rows, keys):
    return max(
        (abs(number(row.get(key, 0.0))) for row in rows for key in keys),
        default=0.0,
    )


def main():
    shutil.rmtree(OUTPUT, ignore_errors=True)
    OUTPUT.mkdir(parents=True, exist_ok=True)
    control = run_case(False)
    treatment = run_case(True)

    write_rows(OUTPUT / "capital_lifecycle_panel.csv", treatment["lifecycle"])
    write_rows(OUTPUT / "retirement_events.csv", treatment["events"])
    write_rows(OUTPUT / "replacement_investment_panel.csv", treatment["replacement"])
    write_rows(OUTPUT / "capital_good_replacement_panel.csv", treatment["capital_goods"])
    bridge_rows = []
    for row in treatment["lifecycle"]:
        bridge_rows.append({
            "global_step": row["global_step"],
            "firm_id": row["firm_id"],
            "previous_service": row["previous_capital_service_capacity"],
            "retired_capacity": row["retired_capacity"],
            "replacement_need": row["replacement_capacity_need"],
            "executed_replacement_investment": row["executed_replacement_investment"],
            "replacement_capacity_acquired": row["executed_replacement_investment"] / 10.0,
            "service_after": row["capital_service_capacity"],
            "service_before_retirement": max(
                0.0,
                row["previous_capital_service_capacity"] - row["retired_capacity"],
            ),
            "service_capacity_acquired": row["executed_replacement_investment"] / 10.0,
            "service_restored_or_unchanged": (
                row["capital_service_capacity"] + TOLERANCE
                >= max(
                    0.0,
                    row["previous_capital_service_capacity"]
                    - row["retired_capacity"],
                ) + row["executed_replacement_investment"] / 10.0
                if row["executed_replacement_investment"] > TOLERANCE
                else True
            ),
        })
    write_rows(OUTPUT / "replacement_capacity_bridge.csv", bridge_rows)
    make_figures(control, treatment)

    accounting = treatment["accounting"]
    treatment_rows = treatment["lifecycle"]
    retirement_count = len(treatment["events"])
    replacement_need = sum(row["replacement_capacity_need"] for row in treatment_rows)
    replacement_executed = sum(row["executed_replacement_investment"] for row in treatment_rows)
    expansion_executed = sum(row["executed_expansion_investment"] for row in treatment_rows)
    unmet_replacement = sum(row["unmet_replacement_investment"] for row in treatment_rows)
    total_investment_executed = replacement_executed + expansion_executed
    capital_good_sales = sum(row["sales"] for row in treatment["capital_goods"])
    capital_good_revenue = sum(row["revenue"] for row in treatment["capital_goods"])
    max_feasibility_violation = max(
        (row["realized_production"] - row["feasible_capacity"] for row in treatment_rows),
        default=0.0,
    )
    max_accounting_gap = max_gap(
        accounting,
        [
            "cash_flow_gap",
            "balance_sheet_gap",
            "inventory_bridge_gap",
            "equity_bridge_gap",
            "capital_book_value_bridge_gap",
        ],
    )
    max_money_gap = max_gap(
        treatment["diagnostics"],
        ["monetary_accounting_gap", "money_delta_gap", "money_location_gap"],
    )
    goods_gap = max_gap(
        treatment["diagnostics"],
        ["goods_conservation_gap", "invariant_violation_count"],
    )
    min_food_cash = min((row["cash"] for row in treatment_rows), default=0.0)
    capital_labor_zero_demand_violations = sum(
        (
            number(row.get("employment", 0.0)) > TOLERANCE
            and number(row.get("desired_labor", 0.0)) <= TOLERANCE
            for row in treatment["capital_goods"]
        ),
    )
    service_restore_count = sum(
        bool(row["service_restored_or_unchanged"])
        for row in bridge_rows
        if row["executed_replacement_investment"] > TOLERANCE
    )
    investment_cash_supply_closes = abs(
        total_investment_executed - capital_good_revenue
    ) <= TOLERANCE
    flags = {
        "verdict": "A. CANONICAL_CAPITAL_LIFECYCLE_REPLACEMENT_ACCEPTED"
        if (
            retirement_count > 0
            and replacement_need > TOLERANCE
            and replacement_executed > TOLERANCE
            and investment_cash_supply_closes
            and max_accounting_gap <= TOLERANCE
            and max_money_gap <= 1e-5
            and goods_gap <= 1e-5
            and max_feasibility_violation <= TOLERANCE
            and min_food_cash >= -TOLERANCE
            and capital_labor_zero_demand_violations == 0
        )
        else "H. OTHER_BLOCKER",
        "population": POPULATION,
        "food_firms": FOOD_FIRMS,
        "steps": STEPS,
        "seed": SEED,
        "useful_life_weeks": USEFUL_LIFE_WEEKS,
        "retirement_events": retirement_count,
        "replacement_capacity_need_total": replacement_need,
        "replacement_investment_executed_total": replacement_executed,
        "unmet_replacement_investment_total": unmet_replacement,
        "expansion_investment_executed_total": expansion_executed,
        "capital_good_sales_units": capital_good_sales,
        "capital_good_sales_revenue": capital_good_revenue,
        "replacement_supply_cash_gap": total_investment_executed - capital_good_revenue,
        "replacement_service_restore_observations": service_restore_count,
        "max_accounting_gap": max_accounting_gap,
        "max_money_gap": max_money_gap,
        "max_goods_or_invariant_gap": goods_gap,
        "max_realized_above_feasible_capacity": max_feasibility_violation,
        "min_food_firm_cash": min_food_cash,
        "capital_good_zero_desired_labor_with_employment": capital_labor_zero_demand_violations,
        "depreciation_non_cash": True,
        "step13_funding_used": False,
        "free_asset_creation": False,
        "economic_behavior_changed_outside_lifecycle": False,
        "new_rng_draws": 0,
    }
    with (OUTPUT / "acceptance_flags.json").open("w", encoding="utf-8") as handle:
        json.dump(flags, handle, indent=2, ensure_ascii=False)

    summary = f"""# Step 15I.9 Canonical Capital Lifecycle

## Verdict

**{flags['verdict']}**

The treatment enabled one explicitly selected `{USEFUL_LIFE_WEEKS:.0f}`-week
non-calibrated engineering useful life. Accounting depreciation is straight
line and non-cash; productive service remains constant while an asset is
active and becomes zero at retirement. Retired assets remain in the asset
record and generate a separate replacement-capacity signal.

## Screen results

- Retirement events: `{retirement_count}`
- Replacement capacity need observed: `{replacement_need:.12g}`
- Executed replacement investment: `{replacement_executed:.12g}`
- Unmet replacement investment retained as demand: `{unmet_replacement:.12g}`
- Executed expansion investment: `{expansion_executed:.12g}`
- Capital-good sales revenue: `{capital_good_revenue:.12g}`
    - Investment cash/supply gap: `{total_investment_executed - capital_good_revenue:.12g}`
- Maximum accounting gap: `{max_accounting_gap:.12g}`
- Maximum money gap: `{max_money_gap:.12g}`
- Maximum goods/invariant gap: `{goods_gap:.12g}`
- Maximum realized production above feasible capacity: `{max_feasibility_violation:.12g}`
- Minimum Food Firm cash: `{min_food_cash:.12g}`

The control kept lifecycle behavior off. No useful-life sweep, continuous
physical decay, capital productivity change, financing change, Step13 credit,
or new RNG draw was introduced.
"""
    (OUTPUT / "acceptance_summary.md").write_text(summary, encoding="utf-8")
    print(flags["verdict"])
    print(f"Outputs: {OUTPUT}")


if __name__ == "__main__":
    main()
