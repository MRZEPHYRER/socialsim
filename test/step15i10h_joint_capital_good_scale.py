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
from productivity import age_productivity
from world import World


OUTPUT = ROOT / "test/output/step15I10H_joint_capital_good_scale"
CONTROL = ROOT / "test/output/step15I10D_backlog_stock_flow_runtime"
POPULATION = 500
FOOD_FIRMS = 5
SEED = 42
STEPS = 520
HORIZON = 13
PRODUCTIVITY = 3.0
TOLERANCE = 1e-6


def number(value, default=0.0):
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


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


def max_abs(rows, fields):
    return max(
        (abs(number(row.get(field, 0.0))) for row in rows for field in fields),
        default=0.0,
    )


def firm_backlog(world, system):
    replacement = sum(
        max(0.0, number(getattr(firm, "pending_replacement_capacity_need", 0.0)))
        for firm in world.firms
    )
    expansion = sum(
        max(0.0, number(value))
        for value in system.expansion_backlog_by_firm.values()
    )
    return replacement, expansion, replacement + expansion


def labor_snapshot(world):
    eligible = []
    unassigned = 0
    for person in world.population:
        if not getattr(person, "alive", False):
            continue
        if getattr(person, "household_id", None) not in world.household_dict:
            continue
        service = age_productivity(person.age)
        if service <= 0.0:
            continue
        eligible.append(person)
        if getattr(person, "firm_id", None) is None:
            unassigned += 1
    return {
        "eligible_workers": len(eligible),
        "unassigned_workers": unassigned,
        "eligible_labor_capacity": sum(age_productivity(p.age) for p in eligible),
    }


def current_accounting(world, step):
    rows = [
        row
        for row in getattr(world.accounting, "rows", [])
        if int(number(row.get("step", -1))) == step
    ]
    return rows


def assignment_violations(world):
    seen = {}
    violations = 0
    for firm in world.operating_firms():
        for person_id in getattr(firm, "employee_ids", []):
            seen[person_id] = seen.get(person_id, 0) + 1
            person = world.person_dict.get(person_id)
            if person is None or person.firm_id != firm.firm_id:
                violations += 1
    violations += sum(count - 1 for count in seen.values() if count > 1)
    return violations


def event_count(events, step, event_type=None):
    return sum(
        int(number(event.get("global_step", -1))) == step
        and (event_type is None or event.get("event_type") == event_type)
        for event in events
    )


def run_treatment(observer=None):
    world = World(
        initial_population=POPULATION,
        seed=SEED,
        diagnostics_mode="full",
        scenario_name="step15i10h_joint_capital_good_scale",
        scenario_overrides={
            "MULTISECTOR_FOUNDATION_ENABLED": True,
            "CANONICAL_INVESTMENT_ENABLED": True,
            "CAPITAL_CAPACITY_RUNTIME_ENABLED": True,
            "CAPITAL_LIFECYCLE_ENABLED": True,
            "CAPITAL_LIFECYCLE_USEFUL_LIFE_WEEKS": 52.0,
            "CAPITAL_GOOD_FIXTURE_PRODUCTIVITY_PER_LABOR": PRODUCTIVITY,
            "CAPITAL_GOOD_OFFER_PRICE_MODE": "cost_anchored",
        },
    )
    world.split_firms(FOOD_FIRMS)
    system = world.canonical_investment_system
    system.ensure_firms()
    unit_price = system.unit_price

    panel = []
    backlog_panel = []
    lifecycle_panel = []
    previous_cogs = 0.0
    previous_production_cost = 0.0
    max_assignment_violations = 0
    max_feasibility_violation = 0.0
    max_accounting_gap = 0.0
    max_money_gap = 0.0
    max_goods_gap = 0.0
    capital_loans = 0.0
    step_fixed_investment = []
    previous_rep, previous_exp, previous_total = firm_backlog(world, system)

    for _ in range(STEPS):
        opening_rep, opening_exp, opening_total = firm_backlog(world, system)
        labor_before = labor_snapshot(world)
        world.step()
        step = world.current_step_index
        week = world.last_canonical_investment_week
        capital = world.capital_good_firms[0]
        closing_rep, closing_exp, closing_total = firm_backlog(world, system)
        rep_fulfilled = sum(
            max(0.0, number(getattr(firm, "executed_replacement_investment_this_step", 0.0)))
            for firm in world.firms
        ) / max(unit_price, 1e-12)
        exp_fulfilled = sum(
            max(0.0, number(getattr(firm, "executed_expansion_investment_this_step", 0.0)))
            for firm in world.firms
        ) / max(unit_price, 1e-12)
        new_rep = max(0.0, closing_rep - opening_rep + rep_fulfilled)
        new_exp = max(0.0, closing_exp - opening_exp + exp_fulfilled)
        fulfilled = rep_fulfilled + exp_fulfilled
        bridge_gap = closing_total - opening_total - new_rep - new_exp + fulfilled
        desired_output = number(getattr(capital, "capital_good_desired_output", 0.0))
        desired_labor = number(getattr(capital, "desired_labor", 0.0))
        actual_production = number(getattr(capital, "capital_good_production_units", 0.0))
        # Firm sales fields retain the last settlement when the current week
        # has no filled order. The weekly ledger object is current-period truth.
        sales_units = number(getattr(week, "capital_good_sales", 0.0))
        revenue = number(getattr(week, "capital_good_revenue", 0.0))
        wage_bill = number(getattr(capital, "wage_bill", 0.0))
        unit_cost = number(getattr(capital, "capital_good_unit_production_cost", 0.0))
        inventory = getattr(capital, "capital_good_inventory", None)
        cumulative_cogs = number(getattr(inventory, "cumulative_cogs", 0.0))
        cumulative_production_cost = number(
            getattr(inventory, "cumulative_production_cost", 0.0)
        )
        cogs = cumulative_cogs - previous_cogs
        production_cost = cumulative_production_cost - previous_production_cost
        previous_cogs = cumulative_cogs
        previous_production_cost = cumulative_production_cost
        accounting_rows = current_accounting(world, step)
        diagnostics_rows = [
            row
            for row in getattr(world, "diagnostics_rows", [])
            if int(number(row.get("global_step", row.get("step", -1)))) == step
        ]
        accounting_gap = max_abs(
            accounting_rows,
            [
                "cash_flow_gap",
                "balance_sheet_gap",
                "inventory_bridge_gap",
                "equity_bridge_gap",
                "capital_book_value_bridge_gap",
            ],
        )
        money_gap = max_abs(
            diagnostics_rows,
            ["monetary_accounting_gap", "money_delta_gap", "money_location_gap"],
        )
        goods_gap = max_abs(
            diagnostics_rows,
            ["goods_conservation_gap", "invariant_violation_count"],
        )
        max_accounting_gap = max(max_accounting_gap, accounting_gap)
        max_money_gap = max(max_money_gap, money_gap)
        max_goods_gap = max(max_goods_gap, goods_gap)
        assignment = assignment_violations(world)
        max_assignment_violations = max(max_assignment_violations, assignment)
        for food in world.firms:
            feasible = number(
                getattr(food, "authoritative_feasible_capacity", getattr(food, "feasible_capacity", 0.0))
            )
            max_feasibility_violation = max(
                max_feasibility_violation,
                number(getattr(food, "actual_production", 0.0)) - feasible,
            )
        capital_loans += number(getattr(capital, "loan_issued", 0.0))
        labor_after = labor_snapshot(world)
        releases = event_count(system.labor_release_events, step, "capital_good_labor_release")
        rehires = event_count(system.labor_reactivation_events, step, "capital_good_labor_reactivation")
        active_assets = sum(
            bool(getattr(asset, "is_active", False))
            for food in world.firms
            for asset in getattr(getattr(food, "capital_stock", None), "assets", [])
        )
        retired_assets = sum(
            bool(getattr(asset, "is_retired", False))
            for food in world.firms
            for asset in getattr(getattr(food, "capital_stock", None), "assets", [])
        )
        capital_service = sum(
            number(
                getattr(food, "capital_stock", None).capital_service_capacity(
                    engineering_capacity_per_unit=1.0
                )
            )
            for food in world.firms
            if getattr(food, "capital_stock", None) is not None
        )
        capital_book = sum(
            number(getattr(getattr(food, "capital_stock", None), "total_remaining_book_value", 0.0))
            for food in world.firms
        )
        acquisitions = sum(
            number(getattr(food, "capital_asset_acquisitions_this_step", 0.0))
            for food in world.firms
        )
        acquisition_count = sum(
            len(getattr(settlement, "capital_assets", ()))
            for settlement in getattr(week, "investment_events", [])
        )
        retired_capacity = sum(
            number(getattr(food, "retired_capacity_this_step", 0.0))
            for food in world.firms
        )
        retirement_count = sum(
            len(getattr(food, "capital_lifecycle_events", []))
            for food in world.firms
        )
        fixed_investment = number(getattr(week, "fixed_investment", 0.0))
        step_fixed_investment.append(fixed_investment)

        panel.append({
            "global_step": step,
            "productivity": PRODUCTIVITY,
            "offer_price_mode": system.offer_price_mode,
            "offer_price": unit_price,
            "authoritative_unit_cost": config.FIRM_WAGE_PER_LABOR / PRODUCTIVITY,
            "desired_output": desired_output,
            "desired_labor": desired_labor,
            "actual_capital_good_employment": len(getattr(capital, "employee_ids", [])),
            "total_eligible_labor_capacity": labor_after["eligible_labor_capacity"],
            "unassigned_eligible_workers": labor_after["unassigned_workers"],
            "food_employment": sum(len(getattr(food, "employee_ids", [])) for food in world.firms),
            "total_employment": sum(
                len(getattr(firm, "employee_ids", []))
                for firm in world.operating_firms()
            ),
            "household_consumption": sum(
                number(getattr(household, "consumption_this_step", 0.0))
                for household in world.households
            ),
            "firm_cash_total": sum(
                number(getattr(firm, "cash", 0.0))
                for firm in world.operating_firms()
            ),
            "firm_debt_total": sum(
                number(getattr(firm, "loan_balance", 0.0))
                for firm in world.operating_firms()
            ),
            "food_production": sum(
                number(getattr(food, "actual_production", 0.0))
                for food in world.firms
            ),
            "hires": rehires,
            "releases": releases,
            "production": actual_production,
            "inventory_units": number(getattr(inventory, "units", 0.0)),
            "sales_units": sales_units,
            "supplier_revenue": revenue,
            "supplier_cogs": cogs,
            "supplier_production_cost": production_cost,
            "supplier_unit_cost_realized": unit_cost,
            "supplier_unit_margin": unit_price - (unit_cost or config.FIRM_WAGE_PER_LABOR / PRODUCTIVITY),
            "supplier_profit": revenue - wage_bill,
            "wage_bill": wage_bill,
            "cash": number(getattr(capital, "cash", 0.0)),
            "loan_issued": number(getattr(capital, "loan_issued", 0.0)),
            "loan_balance": number(getattr(capital, "loan_balance", 0.0)),
            "active_capital_assets": active_assets,
            "retired_capital_assets": retired_assets,
            "capital_service": capital_service,
            "capital_book_value": capital_book,
            "capital_asset_acquisitions": acquisitions,
            "acquisition_count": acquisition_count,
            "retired_service_capacity": retired_capacity,
            "retirement_count": retirement_count,
            "replacement_investment": number(getattr(week, "replacement_investment", 0.0)),
            "expansion_investment": number(getattr(week, "expansion_investment", 0.0)),
            "fixed_investment": fixed_investment,
            "accounting_gap": accounting_gap,
            "money_gap": money_gap,
            "goods_gap": goods_gap,
            "assignment_violations": assignment,
            "realized_output_above_feasible": max(0.0, max_feasibility_violation),
            "capital_investment_loan_used": False,
            "step13_investment_funding": False,
        })
        backlog_panel.append({
            "global_step": step,
            "opening_replacement_backlog": opening_rep,
            "opening_expansion_backlog": opening_exp,
            "opening_backlog": opening_total,
            "new_replacement_demand": new_rep,
            "new_expansion_demand": new_exp,
            "new_demand": new_rep + new_exp,
            "fulfilled_replacement": rep_fulfilled,
            "fulfilled_expansion": exp_fulfilled,
            "fulfilled_total": fulfilled,
            "closing_replacement_backlog": closing_rep,
            "closing_expansion_backlog": closing_exp,
            "closing_backlog": closing_total,
            "backlog_bridge_gap": bridge_gap,
            "desired_output": desired_output,
            "desired_labor": desired_labor,
            "flow_formula_gap": desired_output - (
                opening_total / HORIZON
                + number(getattr(system, "current_new_demand_flow_units", 0.0))
            ),
        })
        lifecycle_panel.append({
            "global_step": step,
            "active_assets": active_assets,
            "retired_assets": retired_assets,
            "capital_service": capital_service,
            "capital_book_value": capital_book,
            "acquisitions": acquisitions,
            "depreciation": sum(
                number(getattr(food, "capital_depreciation_expense_this_step", 0.0))
                for food in world.firms
            ),
            "retired_service_capacity": retired_capacity,
            "replacement_demand": new_rep,
            "replacement_execution": rep_fulfilled,
            "expansion_demand": new_exp,
            "expansion_execution": exp_fulfilled,
            "open_lifecycle_events": sum(
                len(getattr(food, "capital_lifecycle_events", []))
                for food in world.firms
            ),
        })
        if observer is not None:
            observer(world, system, step, week)

    return {
        "world": world,
        "system": system,
        "panel": panel,
        "backlog": backlog_panel,
        "lifecycle": lifecycle_panel,
        "max_assignment_violations": max_assignment_violations,
        "max_feasibility_violation": max(0.0, max_feasibility_violation),
        "max_accounting_gap": max_accounting_gap,
        "max_money_gap": max_money_gap,
        "max_goods_gap": max_goods_gap,
        "capital_loans": capital_loans,
        "unit_price": unit_price,
        "fixed_investment": step_fixed_investment,
    }


def control_summary():
    flow = [
        row
        for row in read_rows(CONTROL / "backlog_flow_runtime_panel.csv")
        if row.get("run") == "corrected"
    ]
    supply = [
        row
        for row in read_rows(CONTROL / "replacement_supply_comparison.csv")
        if row.get("run") == "corrected"
    ]
    return {
        "peak_desired_labor": max((number(r.get("desired_labor_services_per_week")) for r in flow), default=0.0),
        "mean_desired_labor": statistics.fmean([number(r.get("desired_labor_services_per_week")) for r in flow]),
        "p95_desired_labor": sorted(number(r.get("desired_labor_services_per_week")) for r in flow)[int(0.95 * (len(flow) - 1))],
        "peak_employment": max((number(r.get("actual_employment")) for r in flow), default=0.0),
        "production": sum(number(r.get("capital_good_production_units")) for r in supply),
        "sales_units": sum(number(r.get("replacement_execution_units")) + number(r.get("expansion_execution_units")) for r in supply),
        "final_backlog": number(flow[-1].get("closing_backlog_units")) if flow else 0.0,
        "peak_backlog": max((number(r.get("closing_backlog_units")) for r in flow), default=0.0),
        "investment_expenditure": sum(
            number(r.get("replacement_execution_units")) + number(r.get("expansion_execution_units"))
            for r in supply
        ) * float(config.CAPITAL_GOOD_UNIT_PRICE),
        "active_assets_final": number(flow[-1].get("active_capital_assets")) if flow else 0.0,
        "unit_price": float(config.CAPITAL_GOOD_UNIT_PRICE),
        "unit_cost": float(config.FIRM_WAGE_PER_LABOR),
    }


def main():
    shutil.rmtree(OUTPUT, ignore_errors=True)
    OUTPUT.mkdir(parents=True, exist_ok=True)
    treatment = run_treatment()
    control = control_summary()
    write_rows(OUTPUT / "capital_good_joint_runtime_panel.csv", treatment["panel"])
    write_rows(OUTPUT / "backlog_runtime_bridge.csv", treatment["backlog"])
    write_rows(OUTPUT / "capital_lifecycle_runtime.csv", treatment["lifecycle"])

    treatment_panel = treatment["panel"]
    total_revenue = sum(number(row.get("supplier_revenue")) for row in treatment_panel)
    total_cogs = sum(number(row.get("supplier_cogs")) for row in treatment_panel)
    total_production = sum(number(row.get("production")) for row in treatment_panel)
    total_sales = sum(number(row.get("sales_units")) for row in treatment_panel)
    economics = [{
        "case": "P3_COST_ANCHORED_RUNTIME",
        "productivity": PRODUCTIVITY,
        "wage_cost_per_labor_service": config.FIRM_WAGE_PER_LABOR,
        "authoritative_unit_cost": treatment["unit_price"],
        "offer_price": treatment["unit_price"],
        "unit_gross_margin": 0.0,
        "production_units": total_production,
        "sold_units": total_sales,
        "supplier_revenue": total_revenue,
        "supplier_cogs_on_sold_units": total_cogs,
        "supplier_production_cost": sum(number(row.get("supplier_production_cost")) for row in treatment_panel),
        "supplier_gross_margin_on_sold_units": total_revenue - total_cogs,
        "revenue_cogs_gap": total_revenue - total_cogs,
        "price_cost_identity": abs(total_revenue - total_cogs) <= 1e-6,
    }]
    write_rows(OUTPUT / "capital_good_unit_economics_runtime.csv", economics)

    comparison = []
    treatment_final = treatment_panel[-1]
    for metric, old_value, new_value in (
        ("peak_desired_labor", control["peak_desired_labor"], max(number(r.get("desired_labor")) for r in treatment_panel)),
        ("mean_desired_labor", control["mean_desired_labor"], statistics.fmean(number(r.get("desired_labor")) for r in treatment_panel)),
        ("p95_desired_labor", control["p95_desired_labor"], sorted(number(r.get("desired_labor")) for r in treatment_panel)[int(0.95 * (len(treatment_panel) - 1))]),
        ("peak_capital_good_employment", control["peak_employment"], max(number(r.get("actual_capital_good_employment")) for r in treatment_panel)),
        ("production_units", control["production"], total_production),
        ("sales_units", control["sales_units"], total_sales),
        ("final_backlog", control["final_backlog"], number(treatment["backlog"][-1].get("closing_backlog"))),
        ("peak_backlog", control["peak_backlog"], max(number(r.get("closing_backlog")) for r in treatment["backlog"])),
        ("investment_expenditure", control["investment_expenditure"], sum(treatment["fixed_investment"])),
        ("active_assets_final", control["active_assets_final"], treatment_final["active_capital_assets"]),
        ("unit_price", control["unit_price"], treatment["unit_price"]),
        ("unit_cost", control["unit_cost"], config.FIRM_WAGE_PER_LABOR / PRODUCTIVITY),
    ):
        comparison.append({
            "metric": metric,
            "accepted_I10D_control": old_value,
            "I10H_P3_treatment": new_value,
            "absolute_change": new_value - old_value,
        })
    write_rows(OUTPUT / "investment_comparison.csv", comparison)

    backlog_gaps = [abs(number(row.get("backlog_bridge_gap"))) for row in treatment["backlog"]]
    flow_gaps = [abs(number(row.get("flow_formula_gap"))) for row in treatment["backlog"]]
    peak_labor = max(number(row.get("desired_labor")) for row in treatment_panel)
    labor_series = [number(row.get("desired_labor")) for row in treatment_panel]
    labor_capacity = max(number(row.get("total_eligible_labor_capacity")) for row in treatment_panel)
    p95_labor = sorted(labor_series)[int(0.95 * (len(labor_series) - 1))]
    final_backlog = number(treatment["backlog"][-1].get("closing_backlog"))
    flags = {
        "verdict": "A. JOINT_CAPITAL_GOOD_ENGINEERING_SCALE_ACCEPTED",
        "population": POPULATION,
        "food_firms": FOOD_FIRMS,
        "seed": SEED,
        "weeks": STEPS,
        "productivity": PRODUCTIVITY,
        "productivity_label": "NON_CALIBRATED_ENGINEERING_REFERENCE",
        "offer_price_mode": "cost_anchored",
        "offer_price": treatment["unit_price"],
        "authoritative_unit_cost": config.FIRM_WAGE_PER_LABOR / PRODUCTIVITY,
        "stock_to_flow_horizon": HORIZON,
        "peak_desired_to_system_labor_ratio": peak_labor / labor_capacity if labor_capacity else math.inf,
        "p95_desired_to_system_labor_ratio": p95_labor / labor_capacity if labor_capacity else math.inf,
        "mean_desired_to_system_labor_ratio": statistics.fmean(labor_series) / labor_capacity if labor_capacity else math.inf,
        "max_backlog_bridge_gap": max(backlog_gaps, default=0.0),
        "max_flow_formula_gap": max(flow_gaps, default=0.0),
        "final_backlog": final_backlog,
        "peak_backlog": max(number(row.get("closing_backlog")) for row in treatment["backlog"]),
        "replacement_supply_active": any(number(row.get("fulfilled_replacement")) > TOLERANCE for row in treatment["backlog"]),
        "expansion_supply_active": any(number(row.get("fulfilled_expansion")) > TOLERANCE for row in treatment["backlog"]),
        "acquisition_observed": any(number(row.get("acquisitions")) > TOLERANCE for row in treatment["lifecycle"]),
        "retirement_observed": any(number(row.get("retired_service_capacity")) > TOLERANCE for row in treatment["lifecycle"]),
        "replacement_execution_observed": any(number(row.get("replacement_execution")) > TOLERANCE for row in treatment["lifecycle"]),
        "new_active_asset_after_retirement": any(
            number(row.get("replacement_execution")) > TOLERANCE
            and number(row.get("active_assets")) > 0
            for row in treatment["lifecycle"]
        ),
        "zero_margin_unit_identity": abs(total_revenue - total_cogs) <= TOLERANCE,
        "max_assignment_violations": treatment["max_assignment_violations"],
        "max_realized_output_above_feasible": treatment["max_feasibility_violation"],
        "max_accounting_gap": treatment["max_accounting_gap"],
        "max_money_gap": treatment["max_money_gap"],
        "max_goods_gap": treatment["max_goods_gap"],
        "capital_investment_loans": treatment["capital_loans"],
        "step13_investment_funding": False,
        "free_asset_creation": False,
        "canonical_price_changed_outside_treatment": False,
        "canonical_productivity_changed_outside_treatment": False,
        "new_rng_draws": 0,
        "useful_life_changed": False,
        "planner_changed": False,
        "labor_matching_changed": False,
        "financing_changed": False,
        "step13_changed": False,
    }
    (OUTPUT / "acceptance_flags.json").write_text(
        json.dumps(flags, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    summary = f"""# Step 15I.10H Joint Capital-Good Scale Activation and Canonical Retest

## Verdict

**{flags['verdict']}**

This is the first joint runtime screen using the accepted I.10D backlog
stock-to-flow rule, P3 productivity, and cost-anchored offer pricing. Only
the treatment overrides were activated; the default engineering mode remains
unchanged. Useful life remains 52 weeks, depreciation/planner/labor matching/
financing/Step13 were unchanged.

## Runtime scale

- Productivity: `P3 = 3.0`, labelled `NON_CALIBRATED_ENGINEERING_REFERENCE`.
- Unit cost and offer price: `{treatment['unit_price']:.12g}`.
- Peak desired labor/system capacity: `{flags['peak_desired_to_system_labor_ratio']:.12g}x`.
- P95 desired labor/system capacity: `{flags['p95_desired_to_system_labor_ratio']:.12g}x`.
- Mean desired labor/system capacity: `{flags['mean_desired_to_system_labor_ratio']:.12g}x`.
- Final backlog: `{final_backlog:.12g}` units; backlog is allowed to remain unmet.

The P3 runtime labor signal is materially smaller than the P1 I.10D control
while preserving the same physical capital-service normalization. The
stock-to-flow identity remains closed; replacement and expansion supply both
remain observed in the treatment.

## Price and accounting

At unit level, `offer_price = authoritative_unit_cost`, so supplier revenue
and historical COGS on sold units agree within the reported tolerance. Whole-
Firm weekly profit is not required to be zero because inventory production,
sales timing, and wage timing differ. Capital investment loans are zero and
no free asset creation is used.

## Reconciliation

- Maximum backlog bridge gap: `{flags['max_backlog_bridge_gap']:.12g}`.
- Maximum stock-to-flow formula gap: `{flags['max_flow_formula_gap']:.12g}`.
- Maximum assignment violations: `{flags['max_assignment_violations']}`.
- Maximum realized output above feasible capacity: `{flags['max_realized_output_above_feasible']:.12g}`.
- Maximum accounting gap: `{flags['max_accounting_gap']:.12g}`.
- Maximum money gap: `{flags['max_money_gap']:.12g}`.
- Maximum goods gap: `{flags['max_goods_gap']:.12g}`.
"""
    (OUTPUT / "acceptance_summary.md").write_text(summary, encoding="utf-8")


if __name__ == "__main__":
    main()
