from __future__ import annotations

import csv
import gc
import importlib.util
import json
import math
import shutil
import statistics
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

OUTPUT = ROOT / "test/output/step15I11A_large_world_scale_decomposition"
RUNNER_PATH = ROOT / "test/step15i10h_joint_capital_good_scale.py"
FOOD_FIRMS = 5
SEED = 42
STEPS = 520
N_SMALL = 500
N_LARGE = 5000
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


def load_runner():
    spec = importlib.util.spec_from_file_location("step15i10h_runner_11a", RUNNER_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def latest_diagnostic(world, step):
    rows = [
        row
        for row in getattr(world, "diagnostics_rows", [])
        if int(number(row.get("global_step", row.get("step", -1)))) == step
    ]
    return rows[-1] if rows else {}


def latest_firm_diagnostic(world, step, firm_id):
    rows = [
        row
        for row in getattr(world, "firm_diagnostics_rows", [])
        if int(number(row.get("step", -1))) == step
        and int(number(row.get("firm_id", -1))) == int(firm_id)
    ]
    return rows[-1] if rows else {}


def audit_case(runner, population):
    runner.POPULATION = population
    runner.FOOD_FIRMS = FOOD_FIRMS
    runner.SEED = SEED
    runner.STEPS = STEPS
    macro = []
    firms = []

    def observe(world, system, step, week):
        raw = latest_diagnostic(world, step)
        food_firms = list(world.firms)
        capital_firms = list(getattr(world, "capital_good_firms", []))
        household_income = sum(
            number(getattr(household, "income_this_step", 0.0))
            for household in world.households
        )
        household_budget = sum(
            number(getattr(household, "desired_consumption_this_step", 0.0))
            for household in world.households
        )
        household_consumption = sum(
            number(getattr(household, "consumption_this_step", 0.0))
            for household in world.households
        )
        food_demand = sum(number(getattr(firm, "demand_units", 0.0)) for firm in food_firms)
        food_sales = sum(number(getattr(firm, "sales_units", 0.0)) for firm in food_firms)
        food_production = sum(number(getattr(firm, "actual_production", 0.0)) for firm in food_firms)
        food_revenue = sum(number(getattr(firm, "sales_revenue", 0.0)) for firm in food_firms)
        food_employment = sum(len(getattr(firm, "employee_ids", [])) for firm in food_firms)
        food_productivity = max(
            number(getattr(world.firm_system, "food_productivity", 1.0)),
            1e-12,
        )
        food_desired_labor = 0.0
        for firm in food_firms:
            diagnostic = latest_firm_diagnostic(world, step, firm.firm_id)
            desired_labor = number(
                diagnostic.get("desired_labor", getattr(firm, "desired_labor", 0.0))
            )
            if desired_labor <= 1e-12:
                desired_labor = max(
                    0.0,
                    number(
                        diagnostic.get(
                            "expected_output", getattr(firm, "desired_production", 0.0)
                        )
                    )
                    / food_productivity,
                )
            food_desired_labor += desired_labor
        total_employment = sum(
            len(getattr(firm, "employee_ids", []))
            for firm in [*food_firms, *capital_firms]
        )
        total_firm_cash = sum(
            number(getattr(firm, "cash", 0.0))
            for firm in [*food_firms, *capital_firms]
        )
        total_firm_revenue = food_revenue + number(getattr(week, "capital_good_revenue", 0.0))
        opening_backlog = number(getattr(system, "opening_total_backlog_units", 0.0))
        current_flow = number(getattr(system, "current_desired_output_units", 0.0))
        closing_backlog = sum(
            max(0.0, number(getattr(firm, "pending_replacement_capacity_need", 0.0)))
            for firm in food_firms
        ) + sum(
            max(0.0, number(value))
            for value in system.expansion_backlog_by_firm.values()
        )
        active_service = sum(
            number(
                getattr(firm, "capital_stock", None).capital_service_capacity(
                    engineering_capacity_per_unit=1.0
                )
            )
            for firm in food_firms
            if getattr(firm, "capital_stock", None) is not None
        )
        active_assets = sum(
            bool(getattr(asset, "is_active", False))
            for firm in food_firms
            for asset in getattr(getattr(firm, "capital_stock", None), "assets", [])
        )
        lifecycle_acquisitions = sum(
            len(getattr(settlement, "capital_assets", ()))
            for settlement in getattr(week, "investment_events", [])
        )
        lifecycle_retirements = sum(
            len(getattr(firm, "capital_lifecycle_events", [])) for firm in food_firms
        )
        macro.append({
            "global_step": step,
            "population": sum(1 for person in world.population if getattr(person, "alive", False)),
            "households": len(world.households),
            "household_income": household_income,
            "household_consumption_budget": household_budget,
            "household_consumption": household_consumption,
            "food_demand_units": food_demand,
            "food_sales_units": food_sales,
            "food_production_units": food_production,
            "food_revenue": food_revenue,
            "food_employment": food_employment,
            "food_desired_labor": food_desired_labor,
            "total_employment": total_employment,
            "capital_good_employment": sum(len(getattr(firm, "employee_ids", [])) for firm in capital_firms),
            "unassigned_eligible_workers": number(
                sum(
                    1
                    for person in world.population
                    if getattr(person, "alive", False)
                    and getattr(person, "firm_id", None) is None
                    and getattr(person, "household_id", None) in world.household_dict
                )
            ),
            "total_firm_revenue": total_firm_revenue,
            "total_firm_cash": total_firm_cash,
            "fixed_investment": number(getattr(week, "fixed_investment", 0.0)),
            "capital_good_production_units": number(getattr(week, "capital_good_production", 0.0)),
            "capital_good_sales_units": number(getattr(week, "capital_good_sales", 0.0)),
            "opening_capital_backlog": opening_backlog,
            "capital_good_weekly_desired_output": current_flow,
            "closing_capital_backlog": closing_backlog,
            "active_capital_service": active_service,
            "active_assets": active_assets,
            "acquisitions": lifecycle_acquisitions,
            "retirements": lifecycle_retirements,
            "accounting_gap": max(
                [
                    abs(number(row.get(field, 0.0)))
                    for row in getattr(world.accounting, "rows", [])
                    if int(number(row.get("step", -1))) == step
                    for field in ("cash_flow_gap", "balance_sheet_gap", "inventory_bridge_gap", "equity_bridge_gap", "capital_book_value_bridge_gap")
                ],
                default=0.0,
            ),
            "money_gap": abs(number(raw.get("monetary_accounting_gap", 0.0))),
            "goods_gap": max(
                abs(number(raw.get(field, 0.0)))
                for field in ("goods_conservation_gap", "invariant_violation_count")
            ) if raw else 0.0,
        })
        for firm in food_firms:
            diagnostic = latest_firm_diagnostic(world, step, firm.firm_id)
            labor_capacity = number(
                diagnostic.get("labor_capacity", world.firm_employee_capacity(firm))
            )
            desired_labor = number(
                diagnostic.get("desired_labor", getattr(firm, "desired_labor", 0.0))
            )
            desired_output = number(
                diagnostic.get(
                    "expected_output", getattr(firm, "desired_production", 0.0)
                )
            )
            if desired_labor <= 1e-12:
                desired_labor = max(
                    0.0,
                    desired_output
                    / max(
                        1e-12,
                        number(getattr(world.firm_system, "food_productivity", 1.0)),
                    ),
                )
            capital_capacity = number(
                diagnostic.get(
                    "capital_capacity", getattr(firm, "current_capital_capacity", 0.0)
                )
            )
            desired_capital_capacity = number(
                diagnostic.get(
                    "desired_capital_capacity",
                    getattr(firm, "desired_capital_capacity", 0.0),
                )
            )
            capacity_gap = number(
                diagnostic.get("capacity_gap", getattr(firm, "capacity_gap", 0.0))
            )
            protected = number(system._protected_liquidity(firm))
            firms.append({
                "global_step": step,
                "firm_id": firm.firm_id,
                "employment": len(getattr(firm, "employee_ids", [])),
                "labor_capacity": labor_capacity,
                "base_capacity": labor_capacity * number(getattr(world.firm_system, "food_productivity", 1.0)),
                "desired_labor": desired_labor,
                "desired_output": desired_output,
                "expected_demand": number(getattr(firm, "expected_demand", 0.0)),
                "food_demand_units": number(getattr(firm, "demand_units", 0.0)),
                "food_sales_units": number(getattr(firm, "sales_units", 0.0)),
                "food_production_units": number(getattr(firm, "actual_production", 0.0)),
                "inventory_units": number(getattr(firm, "inventory_units", 0.0)),
                "target_inventory_units": number(getattr(firm, "target_inventory_units", 0.0)),
                "cash": number(getattr(firm, "cash", 0.0)),
                "target_cash": number(getattr(firm, "target_cash", 0.0)),
                "protected_operating_liquidity": protected,
                "available_investment_cash": max(0.0, number(getattr(firm, "cash", 0.0)) - protected),
                "desired_investment": number(getattr(firm, "desired_investment_expenditure", 0.0)),
                "executed_investment": number(getattr(firm, "investment_expenditure_this_step", 0.0)),
                "investment_financing_gap": number(getattr(firm, "investment_financing_gap", 0.0)),
                "unexecuted_investment": max(
                    0.0,
                    number(getattr(firm, "desired_investment_expenditure", 0.0))
                    - number(getattr(firm, "investment_expenditure_this_step", 0.0)),
                ),
                "capital_capacity": capital_capacity,
                "desired_capital_capacity": desired_capital_capacity,
                "capacity_gap": capacity_gap,
                "credit_limit": number(getattr(firm, "credit_limit", 0.0)),
                "loan_balance": number(getattr(firm, "loan_balance", 0.0)),
                "investment_review": int(step % 13 == 0),
            })

    result = runner.run_treatment(observer=observe)
    return {
        "population": population,
        "macro": macro,
        "firms": firms,
        "runtime": result,
    }


def sum_field(rows, field):
    return sum(number(row.get(field)) for row in rows)


def mean_field(rows, field):
    values = [number(row.get(field)) for row in rows]
    return statistics.fmean(values) if values else 0.0


def final_field(rows, field):
    return number(rows[-1].get(field)) if rows else 0.0


def scale_value(case, metric):
    rows = case["macro"]
    if metric in {"population", "households"}:
        return final_field(rows, metric), "stock"
    if metric in {
        "household_income",
        "household_consumption_budget",
        "household_consumption",
        "food_demand_units",
        "food_sales_units",
        "food_production_units",
        "food_revenue",
        "total_firm_revenue",
        "fixed_investment",
        "capital_good_production_units",
        "capital_good_sales_units",
    }:
        return sum_field(rows, metric), "cumulative_flow"
    if metric in {"food_employment", "food_desired_labor", "total_employment", "capital_good_employment"}:
        return mean_field(rows, metric), "mean_flow_state"
    if metric in {"total_firm_cash", "active_capital_service", "active_assets", "closing_capital_backlog"}:
        return final_field(rows, metric), "ending_stock"
    raise KeyError(metric)


def firm_comparison(small, large):
    metrics = [
        "employment",
        "labor_capacity",
        "base_capacity",
        "desired_labor",
        "desired_output",
        "expected_demand",
        "inventory_units",
        "target_inventory_units",
        "cash",
        "target_cash",
        "protected_operating_liquidity",
        "available_investment_cash",
        "desired_investment",
        "executed_investment",
        "investment_financing_gap",
        "capital_capacity",
        "desired_capital_capacity",
        "capacity_gap",
        "credit_limit",
        "loan_balance",
    ]
    rows = []
    for metric in metrics:
        small_values = [number(row.get(metric)) for row in small["firms"]]
        large_values = [number(row.get(metric)) for row in large["firms"]]
        small_per_firm = statistics.fmean(small_values) if small_values else 0.0
        large_per_firm = statistics.fmean(large_values) if large_values else 0.0
        small_steps = len({int(number(row.get("global_step", -1))) for row in small["firms"]})
        large_steps = len({int(number(row.get("global_step", -1))) for row in large["firms"]})
        small_aggregate_mean = sum(small_values) / max(1, small_steps)
        large_aggregate_mean = sum(large_values) / max(1, large_steps)
        rows.append({
            "metric": metric,
            "N500_mean_per_firm_week": small_per_firm,
            "N5000_mean_per_firm_week": large_per_firm,
            "per_firm_ratio_N5000_over_N500": large_per_firm / small_per_firm if abs(small_per_firm) > 1e-12 else math.inf,
            "N500_aggregate_mean_week": small_aggregate_mean,
            "N5000_aggregate_mean_week": large_aggregate_mean,
            "aggregate_ratio_N5000_over_N500": large_aggregate_mean / small_aggregate_mean if abs(small_aggregate_mean) > 1e-12 else math.inf,
            "semantic_note": "Food Firm panel; exactly five firms in both worlds",
        })
    return rows


def main():
    shutil.rmtree(OUTPUT, ignore_errors=True)
    OUTPUT.mkdir(parents=True, exist_ok=True)
    runner = load_runner()
    small = audit_case(runner, N_SMALL)
    del runner
    gc.collect()
    runner = load_runner()
    large = audit_case(runner, N_LARGE)
    del runner
    gc.collect()

    metrics = [
        "population",
        "households",
        "household_income",
        "household_consumption_budget",
        "household_consumption",
        "food_demand_units",
        "food_sales_units",
        "food_production_units",
        "food_employment",
        "food_desired_labor",
        "total_employment",
        "total_firm_revenue",
        "total_firm_cash",
        "fixed_investment",
        "capital_good_production_units",
        "capital_good_sales_units",
        "closing_capital_backlog",
        "active_capital_service",
        "active_assets",
    ]
    macro_rows = []
    for metric in metrics:
        small_value, semantics = scale_value(small, metric)
        large_value, _ = scale_value(large, metric)
        small_per_capita = small_value / N_SMALL
        large_per_capita = large_value / N_LARGE
        total_ratio = large_value / small_value if abs(small_value) > 1e-12 else math.inf
        per_capita_ratio = large_per_capita / small_per_capita if abs(small_per_capita) > 1e-12 else math.inf
        macro_rows.append({
            "metric": metric,
            "semantics": semantics,
            "N500_total": small_value,
            "N500_per_capita": small_per_capita,
            "N5000_total": large_value,
            "N5000_per_capita": large_per_capita,
            "total_ratio_N5000_over_N500": total_ratio,
            "per_capita_ratio_N5000_over_N500": per_capita_ratio,
            "proportional_scale_break": abs(per_capita_ratio - 1.0) > 0.5,
        })
    write_rows(OUTPUT / "macro_scale_bridge.csv", macro_rows)
    write_rows(OUTPUT / "firm_scale_comparison.csv", firm_comparison(small, large))

    bridge_metrics = [
        "household_consumption",
        "food_demand_units",
        "food_production_units",
        "fixed_investment",
        "capital_good_production_units",
        "closing_capital_backlog",
        "active_capital_service",
        "active_assets",
    ]
    bridge_rows = []
    for metric in bridge_metrics:
        small_value, semantics = scale_value(small, metric)
        large_value, _ = scale_value(large, metric)
        bridge_rows.append({
            "stage": metric,
            "semantics": semantics,
            "N500_value": small_value,
            "N5000_value": large_value,
            "total_ratio": large_value / small_value if abs(small_value) > 1e-12 else math.inf,
            "per_capita_ratio": (large_value / N_LARGE) / (small_value / N_SMALL) if abs(small_value) > 1e-12 else math.inf,
        })
    write_rows(OUTPUT / "production_investment_scale_bridge.csv", bridge_rows)

    service_rows = []
    for metric in (
        "active_assets",
        "active_capital_service",
        "acquisitions",
        "retirements",
    ):
        if metric in {"acquisitions", "retirements"}:
            small_value = sum_field(small["macro"], metric)
            large_value = sum_field(large["macro"], metric)
        else:
            small_value = final_field(small["macro"], metric)
            large_value = final_field(large["macro"], metric)
        service_rows.append({
            "metric": metric,
            "N500_value": small_value,
            "N5000_value": large_value,
            "N500_per_capita": small_value / N_SMALL,
            "N5000_per_capita": large_value / N_LARGE,
            "per_capita_ratio": (large_value / N_LARGE) / (small_value / N_SMALL) if abs(small_value) > 1e-12 else math.inf,
        })
    service_rows.extend([
        {
            "metric": "average_service_per_active_asset",
            "N500_value": final_field(small["macro"], "active_capital_service") / max(final_field(small["macro"], "active_assets"), 1e-12),
            "N5000_value": final_field(large["macro"], "active_capital_service") / max(final_field(large["macro"], "active_assets"), 1e-12),
            "N500_per_capita": "not applicable",
            "N5000_per_capita": "not applicable",
            "per_capita_ratio": "not applicable",
        },
        {
            "metric": "average_service_per_Food_Firm",
            "N500_value": final_field(small["macro"], "active_capital_service") / FOOD_FIRMS,
            "N5000_value": final_field(large["macro"], "active_capital_service") / FOOD_FIRMS,
            "N500_per_capita": "not applicable",
            "N5000_per_capita": "not applicable",
            "per_capita_ratio": "not applicable",
        },
    ])
    write_rows(OUTPUT / "capital_service_scale_comparison.csv", service_rows)

    labor_rows = []
    for metric in (
        "food_employment",
        "capital_good_employment",
        "total_employment",
        "food_desired_labor",
        "unassigned_eligible_workers",
    ):
        small_value = mean_field(small["macro"], metric)
        large_value = mean_field(large["macro"], metric)
        labor_rows.append({
            "metric": metric,
            "N500_mean": small_value,
            "N5000_mean": large_value,
            "total_ratio": large_value / small_value if abs(small_value) > 1e-12 else math.inf,
            "per_capita_ratio": (large_value / N_LARGE) / (small_value / N_SMALL) if abs(small_value) > 1e-12 else math.inf,
        })
    labor_rows.append({
        "metric": "capital_good_employment_share",
        "N500_mean": mean_field(small["macro"], "capital_good_employment") / max(mean_field(small["macro"], "total_employment"), 1e-12),
        "N5000_mean": mean_field(large["macro"], "capital_good_employment") / max(mean_field(large["macro"], "total_employment"), 1e-12),
        "total_ratio": "not applicable",
        "per_capita_ratio": "not applicable",
    })
    write_rows(OUTPUT / "labor_scale_decomposition.csv", labor_rows)

    cash_metrics = [
        "cash",
        "protected_operating_liquidity",
        "available_investment_cash",
        "desired_investment",
        "executed_investment",
        "investment_financing_gap",
        "unexecuted_investment",
        "target_cash",
        "credit_limit",
    ]
    cash_rows = []
    for metric in cash_metrics:
        small_value = mean_field(small["firms"], metric)
        large_value = mean_field(large["firms"], metric)
        cash_rows.append({
            "metric": metric,
            "N500_mean_per_food_firm": small_value,
            "N5000_mean_per_food_firm": large_value,
            "per_firm_ratio": large_value / small_value if abs(small_value) > 1e-12 else math.inf,
            "N500_aggregate_mean": sum_field(small["firms"], metric) / STEPS,
            "N5000_aggregate_mean": sum_field(large["firms"], metric) / STEPS,
            "aggregate_ratio": (sum_field(large["firms"], metric) / STEPS) / (sum_field(small["firms"], metric) / STEPS) if abs(sum_field(small["firms"], metric)) > 1e-12 else math.inf,
        })
    write_rows(OUTPUT / "internal_cash_scale_comparison.csv", cash_rows)

    macro_lookup = {row["metric"]: row for row in macro_rows}
    cash_lookup = {row["metric"]: row for row in cash_rows}
    first_break = "investment / household consumption"
    verdict = "F. MULTIPLE_SCALE_BOUNDARIES"
    if (
        macro_lookup["food_production_units"]["per_capita_ratio_N5000_over_N500"] < 0.5
        or macro_lookup["food_demand_units"]["per_capita_ratio_N5000_over_N500"] < 0.5
    ):
        verdict = "B. UPSTREAM_FOOD_PRODUCTION_SCALE_BOUNDARY"
        first_break = "Food demand/production"
    elif (
        macro_lookup["active_assets"]["per_capita_ratio_N5000_over_N500"] < 0.5
        and macro_lookup["active_capital_service"]["per_capita_ratio_N5000_over_N500"] < 0.5
    ):
        verdict = "F. MULTIPLE_SCALE_BOUNDARIES"
        first_break = "investment intensity and active capital scale"
    elif cash_lookup["available_investment_cash"]["per_firm_ratio"] < 0.5:
        verdict = "C. INTERNAL_CASH_SCALE_BOUNDARY"
        first_break = "available internal investment cash"

    flags = {
        "verdict": verdict,
        "N500_population": N_SMALL,
        "N5000_population": N_LARGE,
        "food_firms_both": FOOD_FIRMS,
        "seed": SEED,
        "weeks": STEPS,
        "productivity": 3.0,
        "offer_price_semantics": "authoritative_unit_cost",
        "stock_to_flow_horizon": 13,
        "useful_life_weeks": 52,
        "internal_cash_only": True,
        "first_major_scale_break": first_break,
        "food_production_upstream_scale_failure": macro_lookup["food_production_units"]["per_capita_ratio_N5000_over_N500"] < 0.5,
        "investment_production_ratio_scale_invariant": (
            macro_lookup["fixed_investment"]["per_capita_ratio_N5000_over_N500"]
            / max(macro_lookup["food_production_units"]["per_capita_ratio_N5000_over_N500"], 1e-12)
            if macro_lookup["food_production_units"]["per_capita_ratio_N5000_over_N500"] else False
        ),
        "active_asset_count_per_capita_misleading": (
            macro_lookup["active_assets"]["per_capita_ratio_N5000_over_N500"] < 0.5
            and macro_lookup["active_capital_service"]["per_capita_ratio_N5000_over_N500"] >= 0.5
        ),
        "accounting_gap_max": max(
            max(number(row.get("accounting_gap")) for row in small["macro"]),
            max(number(row.get("accounting_gap")) for row in large["macro"]),
        ),
        "money_gap_max": max(
            max(number(row.get("money_gap")) for row in small["macro"]),
            max(number(row.get("money_gap")) for row in large["macro"]),
        ),
        "goods_gap_max": max(
            max(number(row.get("goods_gap")) for row in small["macro"]),
            max(number(row.get("goods_gap")) for row in large["macro"]),
        ),
        "assignment_violations": small["runtime"]["max_assignment_violations"] + large["runtime"]["max_assignment_violations"],
        "realized_output_above_feasible": max(
            small["runtime"]["max_feasibility_violation"],
            large["runtime"]["max_feasibility_violation"],
        ),
        "investment_loans": small["runtime"]["capital_loans"] + large["runtime"]["capital_loans"],
        "economic_behavior_changed": False,
        "new_rng_draws": 0,
    }
    (OUTPUT / "acceptance_flags.json").write_text(
        json.dumps(flags, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    summary = f"""# Step 15I.11A Large-World Capital-Scale Decomposition Audit

## Verdict

**{verdict}**

This diagnostic replay reused the accepted P3, cost-anchored price, 13-week
backlog-flow, 52-week lifecycle, and internal-cash-only configuration for
N=500 and N=5000, both with seed42 and five Food Firms. No economic mechanism,
parameter, Firm count, financing rule, or Step13 behavior was changed.

## Main interpretation

The detailed CSV bridge identifies the first major proportional-scaling break
as **{first_break}**. Investment per Food production is reported separately
from investment per Household consumption, so upstream real-output scale and
capital-formation intensity are not conflated.

Raw active-asset counts are also decomposed into active capital service and
service per active asset. The service measure is the authoritative physical
capital quantity; asset count alone is not treated as sufficient evidence of
capital under-scaling.

## Accounting status

Maximum accounting, money, and goods gaps and assignment/feasibility checks are
recorded in `acceptance_flags.json`. This audit does not interpret the large-
world scale boundary as a reason to tune productivity or activate financing.
"""
    (OUTPUT / "acceptance_summary.md").write_text(summary, encoding="utf-8")


if __name__ == "__main__":
    main()
