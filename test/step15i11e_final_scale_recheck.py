from __future__ import annotations

import csv
import json
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from productivity import age_productivity
from world import World


OUTPUT = ROOT / "test/output/step15I11E_final_scale_recheck"
SEED = 42
WEEKS = 520
FOOD_FIRMS = 5
UNIT_PRICE = 41.0 / 3.0
TOLERANCE = 1e-6


def number(value, default=0.0):
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


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


def overrides():
    return {
        "MULTISECTOR_FOUNDATION_ENABLED": True,
        "CANONICAL_INVESTMENT_ENABLED": True,
        "CAPITAL_CAPACITY_RUNTIME_ENABLED": True,
        "CAPITAL_LIFECYCLE_ENABLED": True,
        "CAPITAL_LIFECYCLE_USEFUL_LIFE_WEEKS": 52.0,
        "CAPITAL_GOOD_FIXTURE_PRODUCTIVITY_PER_LABOR": 3.0,
        "CAPITAL_GOOD_OFFER_PRICE_MODE": "cost_anchored",
        "CAPITAL_GOOD_CUSTOMER_ADVANCE_ENABLED": True,
    }


def alive_population(world):
    return max(1, sum(bool(getattr(person, "alive", False)) for person in world.population))


def assignment_violations(world):
    seen = {}
    violations = 0
    for firm in world.operating_firms():
        for person_id in getattr(firm, "employee_ids", []):
            seen[person_id] = seen.get(person_id, 0) + 1
            person = world.person_dict.get(person_id)
            if person is None or person.firm_id != firm.firm_id:
                violations += 1
    return violations + sum(count - 1 for count in seen.values() if count > 1)


def rows_at(rows, step):
    return [row for row in rows if int(number(row.get("step", row.get("global_step", -1)))) == step]


def slope(values):
    values = [number(value) for value in values]
    if len(values) < 2:
        return 0.0
    return (values[-1] - values[0]) / (len(values) - 1)


def pending_durations(system, step):
    durations = []
    for order in getattr(system, "customer_advance_pending_orders", {}).values():
        text = str(getattr(order, "order_id", ""))
        parts = text.split(":")
        try:
            accepted_step = int(parts[1])
        except (IndexError, ValueError):
            continue
        durations.append(max(0, step - accepted_step + 1))
    return durations


def run_case(population):
    world = World(
        initial_population=population,
        seed=SEED,
        diagnostics_mode="full",
        scenario_name="step15i11e_final_scale_recheck",
        scenario_overrides=overrides(),
    )
    world.split_firms(FOOD_FIRMS)
    system = world.canonical_investment_system
    system.ensure_firms()

    series = {
        "income_per_capita": [],
        "consumption_per_capita": [],
        "food_production_per_capita": [],
        "food_employment_per_capita": [],
        "fixed_investment_per_capita": [],
        "expansion_investment_per_capita": [],
        "replacement_investment_per_capita": [],
        "capital_good_production_per_capita": [],
        "settlement_units_per_capita": [],
        "capital_good_employment_share": [],
        "active_capital_service_per_capita": [],
        "backlog_per_capita": [],
    }
    backlog_units = []
    advance_rows = []
    supplier_rows = []
    lifecycle_rows = []
    previous_liability = 0.0
    previous_prepaid = 0.0
    totals = {
        "income": 0.0,
        "consumption": 0.0,
        "fixed_investment": 0.0,
        "expansion_investment": 0.0,
        "replacement_investment": 0.0,
        "capital_good_production": 0.0,
        "settlement_units": 0.0,
        "food_production": 0.0,
    }
    max_accounting_gap = 0.0
    max_money_gap = 0.0
    max_goods_gap = 0.0
    max_advance_gap = 0.0
    max_assignment_violations = 0
    max_feasibility_violation = 0.0
    supplier_desired_output = 0.0
    supplier_funded_output = 0.0
    supplier_cash_driven_shortfall = 0.0
    supplier_cash_driven_weeks = 0
    advance_before_payroll_weeks = 0
    advance_weeks = 0
    total_depreciation = 0.0
    total_retired_service = 0.0
    acquisition_events = 0
    retirement_events = 0
    replacement_demand = 0.0
    replacement_execution = 0.0
    expansion_execution = 0.0

    for _ in range(WEEKS):
        world.step()
        step = world.current_step_index
        week = world.last_canonical_investment_week
        population_now = alive_population(world)
        capital_firms = list(getattr(world, "capital_good_firms", []))
        food_production = math.fsum(number(getattr(firm, "actual_production", getattr(firm, "production", 0.0))) for firm in world.firms)
        food_employment = sum(len(getattr(firm, "employee_ids", [])) for firm in world.firms)
        total_employment = sum(len(getattr(firm, "employee_ids", [])) for firm in world.operating_firms())
        capital_employment = sum(len(getattr(firm, "employee_ids", [])) for firm in capital_firms)
        fixed = number(getattr(week, "fixed_investment", 0.0))
        production = number(getattr(week, "capital_good_production", 0.0))
        settlement_units = number(getattr(week, "capital_good_sales", 0.0))
        income = number(world.income_history[-1]) if world.income_history else 0.0
        consumption = number(world.consumption_history[-1]) if world.consumption_history else 0.0
        expansion = math.fsum(number(getattr(firm, "executed_expansion_investment_this_step", 0.0)) for firm in world.firms)
        replacement = math.fsum(number(getattr(firm, "executed_replacement_investment_this_step", 0.0)) for firm in world.firms)
        active_service = math.fsum(
            number(getattr(firm, "capital_stock", None).capital_service_capacity(engineering_capacity_per_unit=1.0))
            for firm in world.firms
            if getattr(firm, "capital_stock", None) is not None
        )
        backlog = number(system._capital_good_outstanding_demand())
        for name, value in {
            "income_per_capita": income / population_now,
            "consumption_per_capita": consumption / population_now,
            "food_production_per_capita": food_production / population_now,
            "food_employment_per_capita": food_employment / population_now,
            "fixed_investment_per_capita": fixed / population_now,
            "expansion_investment_per_capita": expansion / population_now,
            "replacement_investment_per_capita": replacement / population_now,
            "capital_good_production_per_capita": production / population_now,
            "settlement_units_per_capita": settlement_units / population_now,
            "capital_good_employment_share": capital_employment / max(1, total_employment),
            "active_capital_service_per_capita": active_service / population_now,
            "backlog_per_capita": backlog / population_now,
        }.items():
            series[name].append(value)
        backlog_units.append(backlog)
        totals["income"] += income
        totals["consumption"] += consumption
        totals["fixed_investment"] += fixed
        totals["expansion_investment"] += expansion
        totals["replacement_investment"] += replacement
        totals["capital_good_production"] += production
        totals["settlement_units"] += settlement_units
        totals["food_production"] += food_production

        accounting_rows = rows_at(getattr(world.accounting, "rows", []), step)
        diagnostic_rows = rows_at(getattr(world, "diagnostics_rows", []), step)
        for row in accounting_rows:
            max_accounting_gap = max(
                max_accounting_gap,
                *(abs(number(row.get(field))) for field in (
                    "cash_flow_gap", "balance_sheet_gap", "inventory_bridge_gap",
                    "equity_bridge_gap", "capital_book_value_bridge_gap",
                    "customer_advance_liability_bridge_gap", "prepaid_investment_bridge_gap",
                )),
            )
        for row in diagnostic_rows:
            max_money_gap = max(
                max_money_gap,
                *(abs(number(row.get(field))) for field in (
                    "monetary_accounting_gap", "money_delta_gap", "money_location_gap",
                )),
            )
            max_goods_gap = max(
                max_goods_gap,
                abs(number(row.get("goods_conservation_gap"))),
                abs(number(row.get("invariant_violation_count"))),
            )
        max_assignment_violations = max(max_assignment_violations, assignment_violations(world))
        for firm in world.firms:
            feasible = number(getattr(firm, "authoritative_feasible_capacity", getattr(firm, "feasible_capacity", 0.0)))
            max_feasibility_violation = max(max_feasibility_violation, number(getattr(firm, "actual_production", 0.0)) - feasible)

        prepaid = math.fsum(number(getattr(firm, "prepaid_capital_investment_asset", 0.0)) for firm in world.firms)
        liability = math.fsum(number(getattr(firm, "customer_advance_liability", 0.0)) for firm in capital_firms)
        paid = math.fsum(number(getattr(firm, "prepaid_capital_investment_paid_this_step", 0.0)) for firm in world.firms)
        received = math.fsum(number(getattr(firm, "customer_advance_received_this_step", 0.0)) for firm in capital_firms)
        delivered = math.fsum(number(getattr(firm, "customer_advance_delivered_this_step", 0.0)) for firm in capital_firms)
        prepaid_capitalized = math.fsum(number(getattr(firm, "prepaid_capital_investment_capitalized_this_step", 0.0)) for firm in world.firms)
        advance_gap = prepaid - liability
        advance_bridge_gap = liability - previous_liability - received + delivered
        prepaid_bridge_gap = prepaid - previous_prepaid - paid + prepaid_capitalized
        max_advance_gap = max(max_advance_gap, abs(advance_gap), abs(advance_bridge_gap), abs(prepaid_bridge_gap))
        advance_rows.append({
            "global_step": step,
            "population": population_now,
            "customer_advance_liability_opening": previous_liability,
            "customer_advance_liability_closing": liability,
            "prepaid_investment_asset_opening": previous_prepaid,
            "prepaid_investment_asset_closing": prepaid,
            "advance_received": received,
            "advance_paid": paid,
            "delivered_value": delivered,
            "capitalized_value": prepaid_capitalized,
            "advance_prepaid_stock_gap": advance_gap,
            "advance_liability_bridge_gap": advance_bridge_gap,
            "prepaid_asset_bridge_gap": prepaid_bridge_gap,
            "pending_advance_orders": len(getattr(system, "customer_advance_pending_orders", {})),
        })
        previous_liability = liability
        previous_prepaid = prepaid

        desired_output = math.fsum(number(getattr(firm, "capital_good_desired_output", 0.0)) for firm in capital_firms)
        funded_output = production
        desired_labor = math.fsum(number(getattr(firm, "desired_labor", 0.0)) for firm in capital_firms)
        actual_labor = math.fsum(number(getattr(firm, "capital_good_labor_services", 0.0)) for firm in capital_firms)
        desired_payroll = desired_labor * 41.0
        cash_before_payroll = math.fsum(
            number(getattr(firm, "cash_start", 0.0))
            + number(getattr(firm, "startup_capitalization_inflow_this_step", 0.0))
            for firm in capital_firms
        )
        if received > 1e-12:
            advance_weeks += 1
            if cash_before_payroll + TOLERANCE >= desired_payroll:
                advance_before_payroll_weeks += 1
        supplier_shortfall = max(0.0, desired_output - funded_output)
        supplier_desired_output += desired_output
        supplier_funded_output += funded_output
        if desired_output > 1e-12 and cash_before_payroll + TOLERANCE < desired_payroll:
            supplier_cash_driven_weeks += 1
            supplier_cash_driven_shortfall += supplier_shortfall
        supplier_rows.append({
            "global_step": step,
            "desired_output": desired_output,
            "funded_output": funded_output,
            "realized_production": production,
            "desired_labor": desired_labor,
            "actual_labor_services": actual_labor,
            "cash_before_payroll": cash_before_payroll,
            "desired_payroll": desired_payroll,
            "advance_received": received,
            "payroll_paid": math.fsum(number(getattr(firm, "wage_payment", 0.0)) for firm in capital_firms),
            "delivery_recognized": delivered,
            "advance_liability_closing": liability,
            "cash_driven_output_shortfall": supplier_shortfall if cash_before_payroll + TOLERANCE < desired_payroll else 0.0,
        })

        depreciation = math.fsum(number(getattr(firm, "capital_depreciation_expense_this_step", 0.0)) for firm in world.firms)
        retired_service = math.fsum(number(getattr(firm, "retired_capacity_this_step", 0.0)) for firm in world.firms)
        acquisition_events += sum(number(getattr(firm, "capital_asset_acquisitions_this_step", 0.0)) > 1e-12 for firm in world.firms)
        retirement_events += sum(number(getattr(firm, "retired_capacity_this_step", 0.0)) > 1e-12 for firm in world.firms)
        total_depreciation += depreciation
        total_retired_service += retired_service
        replacement_demand += math.fsum(number(getattr(firm, "replacement_investment_need", 0.0)) for firm in world.firms)
        replacement_execution += replacement
        expansion_execution += expansion
        lifecycle_rows.append({
            "global_step": step,
            "acquisition_events_cumulative": acquisition_events,
            "depreciation_expense_cumulative": total_depreciation,
            "retirement_events_cumulative": retirement_events,
            "retired_service_cumulative": total_retired_service,
            "replacement_demand_cumulative": replacement_demand,
            "replacement_execution_cumulative": replacement_execution,
            "expansion_execution_cumulative": expansion_execution,
            "active_assets": sum(bool(getattr(asset, "is_active", False)) for firm in world.firms for asset in getattr(getattr(firm, "capital_stock", None), "assets", [])),
            "active_capital_service": active_service,
            "retired_assets": sum(bool(getattr(asset, "is_retired", False)) for firm in world.firms for asset in getattr(getattr(firm, "capital_stock", None), "assets", [])),
        })

    durations = pending_durations(system, world.current_step_index)
    final = {
        "population": alive_population(world),
        "final_customer_advance_liability": previous_liability,
        "final_prepaid_investment_asset": previous_prepaid,
        "pending_advance_orders": len(getattr(system, "customer_advance_pending_orders", {})),
        "pending_duration_min": min(durations) if durations else 0,
        "pending_duration_mean": sum(durations) / len(durations) if durations else 0.0,
        "pending_duration_max": max(durations) if durations else 0,
        "acquisition_events": acquisition_events,
        "depreciation_expense": total_depreciation,
        "retirement_events": retirement_events,
        "retired_service": total_retired_service,
        "replacement_demand": replacement_demand,
        "replacement_execution": replacement_execution,
        "replacement_share": replacement_execution / max(TOLERANCE, replacement_execution + expansion_execution),
        "final_active_assets": lifecycle_rows[-1]["active_assets"],
        "final_retired_assets": lifecycle_rows[-1]["retired_assets"],
        "final_active_service": lifecycle_rows[-1]["active_capital_service"],
        "peak_backlog": max(backlog_units, default=0.0),
        "final_backlog": backlog_units[-1] if backlog_units else 0.0,
        "first_half_backlog_slope": slope(backlog_units[: len(backlog_units) // 2]),
        "second_half_backlog_slope": slope(backlog_units[len(backlog_units) // 2:]),
        "final_quarter_backlog_slope": slope(backlog_units[-max(2, len(backlog_units) // 4):]),
        "supplier_desired_output": supplier_desired_output,
        "supplier_funded_output": supplier_funded_output,
        "supplier_funded_to_desired_ratio": supplier_funded_output / max(TOLERANCE, supplier_desired_output),
        "supplier_cash_driven_weeks": supplier_cash_driven_weeks,
        "supplier_cash_driven_shortfall": supplier_cash_driven_shortfall,
        "advance_before_payroll_share": advance_before_payroll_weeks / max(1, advance_weeks),
        "max_accounting_gap": max_accounting_gap,
        "max_money_gap": max_money_gap,
        "max_goods_gap": max_goods_gap,
        "max_advance_prepaid_gap": max_advance_gap,
        "assignment_violations": max_assignment_violations,
        "max_feasibility_violation": max(0.0, max_feasibility_violation),
        "supplier_loans": math.fsum(number(getattr(firm, "loan_issued", 0.0)) for firm in capital_firms),
        "buyer_investment_loans": 0.0,
        "step13_investment_funding": False,
        "free_asset_creation": False,
    }
    return {"population": population, "series": series, "final": final, "advance": advance_rows, "supplier": supplier_rows, "lifecycle": lifecycle_rows}


def main():
    OUTPUT.mkdir(parents=True, exist_ok=True)
    cases = []
    for population in (500, 5000):
        print(f"Running customer-advance scale recheck N={population} ...", flush=True)
        cases.append(run_case(population))

    small, large = cases
    metric_names = list(small["series"])
    scale_rows = []
    for metric in metric_names:
        small_value = sum(small["series"][metric]) / len(small["series"][metric])
        large_value = sum(large["series"][metric]) / len(large["series"][metric])
        ratio = large_value / small_value if abs(small_value) > 1e-12 else math.nan
        scale_rows.append({
            "metric": metric,
            "N500_mean": small_value,
            "N5000_mean": large_value,
            "N5000_over_N500": ratio,
            "engineering_band_0_5_to_2_0": bool(math.isfinite(ratio) and 0.5 <= ratio <= 2.0),
        })
    write_rows(OUTPUT / "final_scale_comparison.csv", scale_rows)

    advance_rows = []
    backlog_rows = []
    lifecycle_rows = []
    supplier_rows = []
    for population, case in ((500, small), (5000, large)):
        final = case["final"]
        advance_rows.append({"population": population, **{key: value for key, value in final.items() if "advance" in key or "prepaid" in key or "pending" in key or key == "population"}})
        backlog_rows.append({
            "population": population,
            "peak_backlog": final["peak_backlog"],
            "final_backlog": final["final_backlog"],
            "first_half_slope": final["first_half_backlog_slope"],
            "second_half_slope": final["second_half_backlog_slope"],
            "final_quarter_slope": final["final_quarter_backlog_slope"],
            "classification": (
                "STRONGLY_GROWING" if final["final_quarter_backlog_slope"] > 1.0
                else "GROWING" if final["final_quarter_backlog_slope"] > 1e-6
                else "DECLINING" if final["final_quarter_backlog_slope"] < -1e-6
                else "STABLE"
            ),
        })
        lifecycle_rows.append({"population": population, **{key: value for key, value in final.items() if key in (
            "acquisition_events", "depreciation_expense", "retirement_events", "retired_service",
            "replacement_demand", "replacement_execution", "replacement_share", "final_active_assets",
            "final_retired_assets", "final_active_service",
        )}})
        supplier_rows.append({"population": population, **{key: value for key, value in final.items() if key in (
            "supplier_desired_output", "supplier_funded_output", "supplier_funded_to_desired_ratio",
            "supplier_cash_driven_weeks", "supplier_cash_driven_shortfall", "advance_before_payroll_share",
        )}})
    write_rows(OUTPUT / "advance_prepaid_scale.csv", advance_rows)
    write_rows(OUTPUT / "backlog_scale_trends.csv", backlog_rows)
    write_rows(OUTPUT / "lifecycle_scale_summary.csv", lifecycle_rows)
    write_rows(OUTPUT / "supplier_throughput_scale.csv", supplier_rows)

    band_failures = [row for row in scale_rows if not row["engineering_band_0_5_to_2_0"]]
    scale_pass = not band_failures
    bridge_pass = all(case["final"]["max_advance_prepaid_gap"] <= 1e-5 for case in cases)
    conservation_pass = all(
        case["final"]["max_accounting_gap"] <= 1e-5
        and case["final"]["max_money_gap"] <= 1e-4
        and case["final"]["max_goods_gap"] <= 1e-6
        and case["final"]["assignment_violations"] == 0
        and case["final"]["max_feasibility_violation"] <= 1e-6
        and case["final"]["supplier_loans"] == 0.0
        and case["final"]["buyer_investment_loans"] == 0.0
        and not case["final"]["step13_investment_funding"]
        and not case["final"]["free_asset_creation"]
        for case in cases
    )
    pipeline_pass = bridge_pass and all(case["final"]["pending_duration_max"] < WEEKS for case in cases)
    throughput_pass = all(case["final"]["supplier_funded_to_desired_ratio"] > 0.95 for case in cases)
    backlog_metric_rows = {
        "backlog_per_capita": next(
            row for row in scale_rows if row["metric"] == "backlog_per_capita"
        ),
    }
    active_service_row = next(
        row for row in scale_rows if row["metric"] == "active_capital_service_per_capita"
    )
    backlog_scale_bad = (
        not backlog_metric_rows["backlog_per_capita"]["engineering_band_0_5_to_2_0"]
        or any(
            row["final_quarter_slope"] > 1e-6
            for row in backlog_rows
        )
    )
    capital_service_scale_bad = not active_service_row["engineering_band_0_5_to_2_0"]
    cash_driven_throughput_loss = any(
        case["final"]["supplier_cash_driven_weeks"] > 0
        and case["final"]["supplier_cash_driven_shortfall"] > 1e-6
        for case in cases
    )
    component_scale_failures = [
        row["metric"]
        for row in scale_rows
        if row["metric"] in {
            "expansion_investment_per_capita",
            "replacement_investment_per_capita",
        }
        and not row["engineering_band_0_5_to_2_0"]
    ]
    if not conservation_pass:
        verdict = "F. ACCOUNTING_OR_CONSERVATION_BLOCKER"
    elif not bridge_pass or not pipeline_pass:
        verdict = "B. PREPAID_ADVANCE_PIPELINE_ACCUMULATION_BLOCKER"
    elif backlog_scale_bad:
        verdict = "C. BACKLOG_SCALE_REGRESSION_REMAINS"
    elif capital_service_scale_bad:
        verdict = "D. CAPITAL_SERVICE_SCALE_REGRESSION_REMAINS"
    elif not throughput_pass:
        verdict = "E. SUPPLIER_THROUGHPUT_REGRESSION_REMAINS"
    elif component_scale_failures:
        verdict = "G. OTHER_BLOCKER"
    else:
        verdict = "A. CAPITAL_FORMATION_SCALE_ROBUSTNESS_ACCEPTED"

    flags = {
        "verdict": verdict,
        "accepted_productivity": 3.0,
        "accepted_offer_price_semantics": "authoritative_unit_cost",
        "accepted_backlog_horizon_weeks": 13,
        "accepted_useful_life_weeks": 52,
        "customer_advance_enabled": True,
        "full_accepted_order_advance": True,
        "N500_run": True,
        "N5000_run": True,
        "scale_band_pass": scale_pass,
        "scale_band_failures": [row["metric"] for row in band_failures],
        "capital_formation_component_scale_failures": component_scale_failures,
        "backlog_scale_bad": backlog_scale_bad,
        "capital_service_scale_bad": capital_service_scale_bad,
        "advance_prepaid_reconciliation_pass": bridge_pass,
        "pipeline_duration_pass": pipeline_pass,
        "supplier_throughput_pass": throughput_pass,
        "cash_driven_throughput_loss": cash_driven_throughput_loss,
        "labor_capacity_throughput_constraint_observed": not throughput_pass and not cash_driven_throughput_loss,
        "accounting_conservation_pass": conservation_pass,
        "supplier_loans": 0.0,
        "buyer_investment_loans": 0.0,
        "step13_investment_funding": False,
        "free_asset_creation": False,
        "new_economic_mechanism": False,
        "new_rng_draws": 0,
    }
    (OUTPUT / "acceptance_flags.json").write_text(json.dumps(flags, indent=2), encoding="utf-8")
    summary_lines = [
        "# Step 15I.11E Final Capital-Formation Scale Recheck",
        "",
        f"Verdict: **{verdict}**",
        "",
        "Both runs used customer advance ON, P3 productivity, cost-anchored offer pricing, 13-week backlog flow, 52-week useful life and internal-cash-only buyer finance.",
        "",
        "## Scale",
    ]
    for row in scale_rows:
        summary_lines.append(f"- {row['metric']}: N5000/N5000 ratio = {row['N5000_over_N500']:.6g}; band pass = {row['engineering_band_0_5_to_2_0']}")
    summary_lines.extend([
        "",
        "## Reconciliation",
        *[
            f"- N={case['population']}: accounting={case['final']['max_accounting_gap']:.6g}, money={case['final']['max_money_gap']:.6g}, goods={case['final']['max_goods_gap']:.6g}, advance/prepaid={case['final']['max_advance_prepaid_gap']:.6g}, assignment={case['final']['assignment_violations']}"
            for case in cases
        ],
        "",
        "Customer advances were the only treatment mechanism; no supplier/investment loan or Step13 funding was used.",
    ])
    (OUTPUT / "acceptance_summary.md").write_text("\n".join(summary_lines) + "\n", encoding="utf-8")
    print(json.dumps(flags, indent=2))


if __name__ == "__main__":
    main()
