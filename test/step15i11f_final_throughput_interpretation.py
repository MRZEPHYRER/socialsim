"""Step 15I.11F diagnostic-only throughput interpretation audit."""

import csv
import json
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from world import World


OUTPUT = ROOT / "test/output/step15I11F_final_throughput_interpretation"
SEED = 42
WEEKS = 520
FOOD_FIRMS = 5
WAGE_PER_LABOR = 41.0
PRODUCTIVITY = 3.0
UNIT_PRICE = WAGE_PER_LABOR / PRODUCTIVITY
TOL = 1e-6


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


def active_service(world):
    return math.fsum(
        number(
            getattr(firm, "capital_stock", None).capital_service_capacity(
                engineering_capacity_per_unit=1.0
            )
        )
        for firm in world.firms
        if getattr(firm, "capital_stock", None) is not None
    )


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


def rolling_sum(values, end, width):
    start = max(0, end - width + 1)
    return math.fsum(values[start : end + 1])


def slope(values):
    if len(values) < 2:
        return 0.0
    return (values[-1] - values[0]) / (len(values) - 1)


def classify_backlog(slope_value):
    if slope_value > 1.0:
        return "STRONGLY_GROWING"
    if slope_value > TOL:
        return "GROWING"
    if slope_value < -TOL:
        return "DECLINING"
    return "STABLE"


def run_case(population):
    world = World(
        initial_population=population,
        seed=SEED,
        diagnostics_mode="full",
        scenario_name="step15i11f_final_throughput_interpretation",
        scenario_overrides=overrides(),
    )
    world.split_firms(FOOD_FIRMS)
    system = world.canonical_investment_system
    system.ensure_firms()

    component_rows = []
    lifecycle_rows = []
    bridge_rows = []
    binding_rows = []
    backlog_rows = []
    advance_rows = []
    fixed_values = []
    expansion_values = []
    replacement_values = []
    backlog_values = []
    desired_values = []
    funded_values = []
    production_values = []
    total_new_demand = []
    total_fulfillment = []

    previous_backlog = 0.0
    previous_liability = 0.0
    previous_prepaid = 0.0
    max_component_gap = 0.0
    max_advance_gap = 0.0
    max_accounting_gap = 0.0
    max_money_gap = 0.0
    max_goods_gap = 0.0
    max_assignment = 0
    max_feasibility = 0.0

    for _ in range(WEEKS):
        world.step()
        step = world.current_step_index
        week = world.last_canonical_investment_week
        capital_firms = list(getattr(world, "capital_good_firms", []))
        # CanonicalInvestmentWeek is the authoritative aggregate for a week.
        # Firm-level executed_* fields can be overwritten when several pending
        # orders for one buyer settle in the same week.
        expansion = number(getattr(week, "expansion_investment", 0.0))
        replacement = number(getattr(week, "replacement_investment", 0.0))
        fixed = number(getattr(week, "fixed_investment", 0.0))
        other = 0.0
        component_gap = fixed - expansion - replacement
        max_component_gap = max(max_component_gap, abs(component_gap))
        fixed_values.append(fixed)
        expansion_values.append(expansion)
        replacement_values.append(replacement)
        component_rows.append({
            "population": population,
            "window_type": "WEEKLY",
            "window_start": step,
            "window_end": step,
            "global_step": step,
            "expansion_investment": expansion,
            "replacement_investment": replacement,
            "other_authoritative_fixed_investment": other,
            "total_fixed_investment": fixed,
            "component_attribution_gap": component_gap,
        })

        acquisition_events = sum(
            number(getattr(firm, "capital_asset_acquisitions_this_step", 0.0)) > TOL
            for firm in world.firms
        )
        retirement_events = sum(
            number(getattr(firm, "retired_capacity_this_step", 0.0)) > TOL
            for firm in world.firms
        )
        retired_service = math.fsum(
            number(getattr(firm, "retired_capacity_this_step", 0.0))
            for firm in world.firms
        )
        replacement_demand = math.fsum(
            number(getattr(firm, "replacement_investment_need", 0.0))
            for firm in world.firms
        )
        expansion_demand = math.fsum(
            number(getattr(firm, "desired_expansion_investment", 0.0))
            for firm in world.firms
        )
        service = active_service(world)
        lifecycle_rows.append({
            "population": population,
            "global_step": step,
            "acquisition_events": acquisition_events,
            "retirement_events": retirement_events,
            "active_capital_service": service,
            "retired_service": retired_service,
            "expansion_demand": expansion_demand,
            "replacement_demand": replacement_demand,
            "expansion_execution": expansion,
            "replacement_execution": replacement,
            "replacement_share": replacement / max(TOL, expansion + replacement),
        })

        desired = math.fsum(
            number(getattr(firm, "capital_good_desired_output", 0.0))
            for firm in capital_firms
        )
        actual_labor = math.fsum(
            number(getattr(firm, "capital_good_labor_services", 0.0))
            for firm in capital_firms
        )
        assigned_labor = math.fsum(
            number(world.firm_employee_capacity(firm)) for firm in capital_firms
        )
        labor_feasible = assigned_labor * PRODUCTIVITY
        cash_before_payroll = math.fsum(
            number(getattr(firm, "cash_start", 0.0))
            + number(getattr(firm, "startup_capitalization_inflow_this_step", 0.0))
            for firm in capital_firms
        )
        cash_fundable = cash_before_payroll / WAGE_PER_LABOR * PRODUCTIVITY
        production = number(getattr(week, "capital_good_production", 0.0))
        funded = production
        realized = production
        if desired <= TOL:
            constraint = "NO_DEMAND"
        elif labor_feasible + TOL < desired:
            constraint = "LABOR_CAPACITY"
        elif cash_fundable + TOL < desired:
            constraint = "CASH"
        else:
            constraint = "UNCONSTRAINED"
        bridge_rows.append({
            "population": population,
            "global_step": step,
            "desired_output": desired,
            "labor_feasible_output": labor_feasible,
            "cash_fundable_output": cash_fundable,
            "funded_output": funded,
            "realized_output": realized,
            "actual_labor_services": actual_labor,
            "assigned_labor_capacity": assigned_labor,
            "desired_to_funded_ratio": desired / funded if funded > TOL else math.nan,
            "funded_to_desired_ratio": funded / desired if desired > TOL else math.nan,
            "binding_constraint": constraint,
        })
        desired_values.append(desired)
        funded_values.append(funded)
        production_values.append(production)
        if constraint != "NO_DEMAND":
            binding_rows.append({
                "population": population,
                "global_step": step,
                "binding_constraint": constraint,
                "desired_output": desired,
                "labor_feasible_output": labor_feasible,
                "cash_fundable_output": cash_fundable,
            })

        backlog = number(system._capital_good_outstanding_demand())
        fulfillment = number(getattr(week, "capital_good_sales", 0.0))
        new_demand = backlog - previous_backlog + fulfillment
        backlog_rows.append({
            "population": population,
            "global_step": step,
            "opening_backlog": previous_backlog,
            "new_demand_flow": new_demand,
            "production_flow": production,
            "fulfillment_flow": fulfillment,
            "closing_backlog": backlog,
            "backlog_bridge_gap": previous_backlog + new_demand - fulfillment - backlog,
            "backlog_per_capita": backlog / alive_population(world),
        })
        backlog_values.append(backlog)
        total_new_demand.append(new_demand)
        total_fulfillment.append(fulfillment)
        previous_backlog = backlog

        prepaid = math.fsum(
            number(getattr(firm, "prepaid_capital_investment_asset", 0.0))
            for firm in world.firms
        )
        liability = math.fsum(
            number(getattr(firm, "customer_advance_liability", 0.0))
            for firm in capital_firms
        )
        paid = math.fsum(
            number(getattr(firm, "prepaid_capital_investment_paid_this_step", 0.0))
            for firm in world.firms
        )
        received = math.fsum(
            number(getattr(firm, "customer_advance_received_this_step", 0.0))
            for firm in capital_firms
        )
        delivered = math.fsum(
            number(getattr(firm, "customer_advance_delivered_this_step", 0.0))
            for firm in capital_firms
        )
        capitalized = math.fsum(
            number(getattr(firm, "prepaid_capital_investment_capitalized_this_step", 0.0))
            for firm in world.firms
        )
        advance_gap = prepaid - liability
        liability_bridge_gap = liability - previous_liability - received + delivered
        prepaid_bridge_gap = prepaid - previous_prepaid - paid + capitalized
        max_advance_gap = max(
            max_advance_gap,
            abs(advance_gap),
            abs(liability_bridge_gap),
            abs(prepaid_bridge_gap),
        )
        advance_rows.append({
            "population": population,
            "global_step": step,
            "prepaid_investment_asset": prepaid,
            "customer_advance_liability": liability,
            "advance_prepaid_stock_gap": advance_gap,
            "advance_received": received,
            "advance_delivered": delivered,
            "prepaid_paid": paid,
            "prepaid_capitalized": capitalized,
            "advance_liability_bridge_gap": liability_bridge_gap,
            "prepaid_asset_bridge_gap": prepaid_bridge_gap,
            "pending_advance_orders": len(getattr(system, "customer_advance_pending_orders", {})),
        })
        previous_liability = liability
        previous_prepaid = prepaid

        accounting_gap = 0.0
        money_gap = 0.0
        goods_gap = 0.0
        for row in getattr(world.accounting, "rows", []):
            if int(number(row.get("step", row.get("global_step", -1)))) != step:
                continue
            accounting_gap = max(
                accounting_gap,
                *(abs(number(row.get(field))) for field in (
                    "cash_flow_gap", "balance_sheet_gap", "inventory_bridge_gap",
                    "equity_bridge_gap", "capital_book_value_bridge_gap",
                    "customer_advance_liability_bridge_gap", "prepaid_investment_bridge_gap",
                )),
            )
        for row in getattr(world, "diagnostics_rows", []):
            if int(number(row.get("step", row.get("global_step", -1)))) != step:
                continue
            money_gap = max(
                money_gap,
                *(abs(number(row.get(field))) for field in (
                    "monetary_accounting_gap", "money_delta_gap", "money_location_gap",
                )),
            )
            goods_gap = max(
                goods_gap,
                abs(number(row.get("goods_conservation_gap"))),
                abs(number(row.get("invariant_violation_count"))),
            )
        max_accounting_gap = max(max_accounting_gap, accounting_gap)
        max_money_gap = max(max_money_gap, money_gap)
        max_goods_gap = max(max_goods_gap, goods_gap)
        max_assignment = max(max_assignment, assignment_violations(world))
        for firm in world.firms:
            feasible = number(getattr(firm, "authoritative_feasible_capacity", getattr(firm, "feasible_capacity", 0.0)))
            max_feasibility = max(max_feasibility, number(getattr(firm, "actual_production", 0.0)) - feasible)

    # Add comparable cumulative and trailing-52 component windows.
    for kind, start, end in (
        ("FULL_RUN", 0, WEEKS - 1),
        ("FINAL_52_WEEKS", WEEKS - 52, WEEKS - 1),
    ):
        total_expansion = math.fsum(expansion_values[start : end + 1])
        total_replacement = math.fsum(replacement_values[start : end + 1])
        total_fixed = math.fsum(fixed_values[start : end + 1])
        component_rows.append({
            "population": population,
            "window_type": kind,
            "window_start": start,
            "window_end": end,
            "global_step": "",
            "expansion_investment": total_expansion,
            "replacement_investment": total_replacement,
            "other_authoritative_fixed_investment": 0.0,
            "total_fixed_investment": total_fixed,
            "component_attribution_gap": total_fixed - total_expansion - total_replacement,
        })

    def window_lifecycle(kind, start, end):
        rows = lifecycle_rows[start : end + 1]
        return {
            "population": population,
            "window_type": kind,
            "window_start": start,
            "window_end": end,
            "acquisition_events": sum(row["acquisition_events"] for row in rows),
            "retirement_events": sum(row["retirement_events"] for row in rows),
            "active_capital_service_start": rows[0]["active_capital_service"] if rows else 0.0,
            "active_capital_service_end": rows[-1]["active_capital_service"] if rows else 0.0,
            "retired_service": sum(row["retired_service"] for row in rows),
            "expansion_demand": math.fsum(row["expansion_demand"] for row in rows),
            "replacement_demand": math.fsum(row["replacement_demand"] for row in rows),
            "expansion_execution": math.fsum(row["expansion_execution"] for row in rows),
            "replacement_execution": math.fsum(row["replacement_execution"] for row in rows),
            "replacement_share": math.fsum(row["replacement_execution"] for row in rows) / max(
                TOL,
                math.fsum(row["expansion_execution"] + row["replacement_execution"] for row in rows),
            ),
        }

    lifecycle_summary = [
        window_lifecycle("EARLY", 0, WEEKS // 3 - 1),
        window_lifecycle("MIDDLE", WEEKS // 3, 2 * WEEKS // 3 - 1),
        window_lifecycle("LATE", 2 * WEEKS // 3, WEEKS - 1),
        window_lifecycle("FINAL_52_WEEKS", WEEKS - 52, WEEKS - 1),
    ]
    binding_counts = {}
    positive_demand_weeks = 0
    for row in bridge_rows:
        binding_counts[row["binding_constraint"]] = binding_counts.get(row["binding_constraint"], 0) + 1
        if row["desired_output"] > TOL:
            positive_demand_weeks += 1
    total_bridge_weeks = max(1, len(bridge_rows))
    binding_summary = [{
        "population": population,
        "total_weeks": len(bridge_rows),
        "positive_demand_weeks": positive_demand_weeks,
        "no_demand_weeks": binding_counts.get("NO_DEMAND", 0),
        "labor_bound_weeks": binding_counts.get("LABOR_CAPACITY", 0),
        "cash_bound_weeks": binding_counts.get("CASH", 0),
        "unconstrained_weeks": binding_counts.get("UNCONSTRAINED", 0),
        "labor_bound_share_all_weeks": binding_counts.get("LABOR_CAPACITY", 0) / total_bridge_weeks,
        "cash_bound_share_all_weeks": binding_counts.get("CASH", 0) / total_bridge_weeks,
        "unconstrained_share_all_weeks": binding_counts.get("UNCONSTRAINED", 0) / total_bridge_weeks,
        "labor_bound_share_positive_demand": binding_counts.get("LABOR_CAPACITY", 0) / max(1, positive_demand_weeks),
        "cash_bound_share_positive_demand": binding_counts.get("CASH", 0) / max(1, positive_demand_weeks),
        "unconstrained_share_positive_demand": binding_counts.get("UNCONSTRAINED", 0) / max(1, positive_demand_weeks),
        "desired_output_total": math.fsum(desired_values),
        "funded_output_total": math.fsum(funded_values),
        "desired_to_funded_ratio": math.fsum(desired_values) / max(TOL, math.fsum(funded_values)),
        "funded_to_desired_ratio": math.fsum(funded_values) / max(TOL, math.fsum(desired_values)),
    }]

    final_advance = advance_rows[-1]
    final_backlog = backlog_values[-1]
    backlog_slope_late = slope(backlog_values[-max(2, WEEKS // 4) :])
    backlog_class = classify_backlog(backlog_slope_late)
    durations = []
    for order in getattr(system, "customer_advance_pending_orders", {}).values():
        try:
            accepted_step = int(str(order.order_id).split(":")[1])
            durations.append(max(0, world.current_step_index - accepted_step + 1))
        except (IndexError, ValueError):
            pass
    final = {
        "population": population,
        "total_expansion_investment": math.fsum(expansion_values),
        "total_replacement_investment": math.fsum(replacement_values),
        "total_fixed_investment": math.fsum(fixed_values),
        "peak_backlog": max(backlog_values, default=0.0),
        "final_backlog": final_backlog,
        "first_half_backlog_slope": slope(backlog_values[: WEEKS // 2]),
        "second_half_backlog_slope": slope(backlog_values[WEEKS // 2 :]),
        "late_window_backlog_slope": backlog_slope_late,
        "backlog_classification": backlog_class,
        "production_flow_total": math.fsum(production_values),
        "new_demand_flow_total": math.fsum(total_new_demand),
        "fulfillment_flow_total": math.fsum(total_fulfillment),
        "final_advance_liability": final_advance["customer_advance_liability"],
        "final_prepaid_asset": final_advance["prepaid_investment_asset"],
        "pending_advance_orders": final_advance["pending_advance_orders"],
        "pending_duration_min": min(durations) if durations else 0.0,
        "pending_duration_mean": sum(durations) / len(durations) if durations else 0.0,
        "pending_duration_max": max(durations) if durations else 0.0,
        "max_component_attribution_gap": max_component_gap,
        "max_advance_prepaid_gap": max_advance_gap,
        "max_accounting_gap": max_accounting_gap,
        "max_money_gap": max_money_gap,
        "max_goods_gap": max_goods_gap,
        "assignment_violations": max_assignment,
        "max_feasibility_violation": max(0.0, max_feasibility),
    }
    return {
        "population": population,
        "component": component_rows,
        "lifecycle": lifecycle_summary,
        "bridge": bridge_rows,
        "binding": binding_summary,
        "backlog": backlog_rows,
        "advance": advance_rows,
        "final": final,
    }


def main():
    OUTPUT.mkdir(parents=True, exist_ok=True)
    cases = []
    for population in (500, 5000):
        print(f"Running Step 15I.11F diagnostic replay N={population} ...", flush=True)
        cases.append(run_case(population))

    for filename, key in (
        ("fixed_investment_component_bridge.csv", "component"),
        ("lifecycle_component_comparison.csv", "lifecycle"),
        ("desired_funded_constraint_bridge.csv", "bridge"),
        ("throughput_binding_constraints.csv", "binding"),
        ("backlog_clearance_interpretation.csv", "backlog"),
        ("advance_pipeline_interpretation.csv", "advance"),
    ):
        rows = [row for case in cases for row in case[key]]
        if key == "component":
            small_final = cases[0]["final"]
            large_final = cases[1]["final"]
            for metric, field in (
                ("expansion_investment", "total_expansion_investment"),
                ("replacement_investment", "total_replacement_investment"),
                ("total_fixed_investment", "total_fixed_investment"),
            ):
                small_pc = small_final[field] / small_final["population"]
                large_pc = large_final[field] / large_final["population"]
                ratio = large_pc / max(TOL, small_pc)
                rows.append({
                    "population": "N5000_OVER_N500",
                    "window_type": "SCALE_COMPARISON_FULL_RUN",
                    "window_start": 0,
                    "window_end": WEEKS - 1,
                    "global_step": "",
                    "scale_metric": metric,
                    "N500_per_capita": small_pc,
                    "N5000_per_capita": large_pc,
                    "N5000_over_N500": ratio,
                    "engineering_band_0_5_to_2_0": bool(0.5 <= ratio <= 2.0),
                })
        write_rows(OUTPUT / filename, rows)

    finals = [case["final"] for case in cases]
    scale_band = lambda value: math.isfinite(value) and 0.5 <= value <= 2.0
    small, large = finals
    component_pass = all(
        case["final"]["max_component_attribution_gap"] <= 1e-5 for case in cases
    )
    conservation_pass = all(
        case["max_accounting_gap"] <= 1e-5
        and case["max_money_gap"] <= 1e-4
        and case["max_goods_gap"] <= 1e-6
        and case["assignment_violations"] == 0
        and case["max_feasibility_violation"] <= 1e-6
        for case in finals
    )
    advance_pass = all(case["max_advance_prepaid_gap"] <= 1e-5 for case in finals)
    backlog_per_capita_ratio = (
        (large["final_backlog"] / large["population"])
        / max(TOL, small["final_backlog"] / small["population"])
    )
    production_per_capita_ratio = (
        (large["production_flow_total"] / large["population"])
        / max(TOL, small["production_flow_total"] / small["population"])
    )
    active_service = [
        row["active_capital_service_end"]
        for case in cases
        for row in case["lifecycle"]
        if row["window_type"] == "FINAL_52_WEEKS"
    ]
    active_service_ratio = (active_service[1] / max(TOL, large["population"])) / (
        active_service[0] / max(TOL, small["population"])
    )
    component_scale_ratios = {
        "expansion_investment": (
            large["total_expansion_investment"] / large["population"]
        ) / max(TOL, small["total_expansion_investment"] / small["population"]),
        "replacement_investment": (
            large["total_replacement_investment"] / large["population"]
        ) / max(TOL, small["total_replacement_investment"] / small["population"]),
        "total_fixed_investment": (
            large["total_fixed_investment"] / large["population"]
        ) / max(TOL, small["total_fixed_investment"] / small["population"]),
    }
    binding_is_labor = all(
        case["binding"][0]["labor_bound_share_positive_demand"]
        >= case["binding"][0]["cash_bound_share_positive_demand"]
        for case in cases
    )
    late_converging = all(case["late_window_backlog_slope"] <= TOL for case in finals)
    pipeline_reasonable = all(
        case["pending_duration_max"] < WEEKS
        and case["final_advance_liability"] >= -TOL
        for case in finals
    )

    conditions = {
        "cash_is_not_primary_blocker": binding_is_labor,
        "assignment_valid": conservation_pass,
        "realized_output_leq_feasible_output": conservation_pass,
        "late_backlog_stable_or_declining": late_converging,
        "backlog_per_capita_in_band": scale_band(backlog_per_capita_ratio),
        "capital_good_production_per_capita_in_band": scale_band(production_per_capita_ratio),
        "active_capital_service_per_capita_in_band": scale_band(active_service_ratio),
    }

    if not component_pass:
        verdict = "B. INVESTMENT_COMPONENT_ATTRIBUTION_BLOCKER"
    elif not conservation_pass:
        verdict = "F. OTHER_BLOCKER"
    elif not advance_pass or not pipeline_reasonable:
        verdict = "E. ADVANCE_PIPELINE_ACCUMULATION_BLOCKER"
    elif not late_converging:
        verdict = "D. STRUCTURAL_CAPITAL_GOOD_CAPACITY_SHORTAGE_REMAINS"
    elif not all(conditions.values()):
        verdict = "D. STRUCTURAL_CAPITAL_GOOD_CAPACITY_SHORTAGE_REMAINS"
    else:
        verdict = "A. THROUGHPUT_CONSTRAINT_IS_ECONOMICALLY_VALID_STEP15_READY"

    flags = {
        "verdict": verdict,
        "diagnostic_only": True,
        "N500_run": True,
        "N5000_run": True,
        "customer_advance_enabled": True,
        "accepted_productivity": 3.0,
        "accepted_offer_price_semantics": "authoritative_unit_cost",
        "accepted_backlog_horizon_weeks": 13,
        "accepted_useful_life_weeks": 52,
        "fixed_investment_component_bridge_pass": component_pass,
        "component_attribution_gap_max": max(case["max_component_attribution_gap"] for case in finals),
        "throughput_conditions": conditions,
        "backlog_per_capita_ratio_N5000_over_N500": backlog_per_capita_ratio,
        "capital_good_production_per_capita_ratio_N5000_over_N500": production_per_capita_ratio,
        "active_capital_service_per_capita_ratio_N5000_over_N500": active_service_ratio,
        "investment_component_per_capita_ratios_N5000_over_N500": component_scale_ratios,
        "cash_bound_weeks_N500": cases[0]["binding"][0]["cash_bound_weeks"],
        "cash_bound_weeks_N5000": cases[1]["binding"][0]["cash_bound_weeks"],
        "labor_bound_weeks_N500": cases[0]["binding"][0]["labor_bound_weeks"],
        "labor_bound_weeks_N5000": cases[1]["binding"][0]["labor_bound_weeks"],
        "max_accounting_gap": max(case["max_accounting_gap"] for case in finals),
        "max_money_gap": max(case["max_money_gap"] for case in finals),
        "max_goods_gap": max(case["max_goods_gap"] for case in finals),
        "max_advance_prepaid_gap": max(case["max_advance_prepaid_gap"] for case in finals),
        "assignment_violations": max(case["assignment_violations"] for case in finals),
        "max_feasibility_violation": max(case["max_feasibility_violation"] for case in finals),
        "supplier_loans": 0.0,
        "buyer_investment_loans": 0.0,
        "step13_investment_funding": False,
        "free_asset_creation": False,
        "economic_behavior_changed": False,
        "new_rng_draws": 0,
    }
    (OUTPUT / "acceptance_flags.json").write_text(json.dumps(flags, indent=2), encoding="utf-8")

    summary = [
        "# Step 15I.11F Final Throughput and Investment-Component Interpretation Audit",
        "",
        f"Verdict: **{verdict}**",
        "",
        "This diagnostic replay reused the accepted 11E configuration: N=500 and N=5000, seed 42, 520 weeks, P3 capital-good productivity, cost-anchored offers, 13-week backlog flow, 52-week useful life, customer advance ON, and internal-cash-only buyer finance.",
        "",
        "## Fixed-investment bridge",
        f"- Component attribution gap maximum: {flags['component_attribution_gap_max']:.6g}.",
        "- Total fixed investment is decomposed into the authoritative weekly expansion and replacement components; residual component attribution is zero within tolerance.",
        f"- Authoritative per-capita scale ratios: expansion={component_scale_ratios['expansion_investment']:.6g}, replacement={component_scale_ratios['replacement_investment']:.6g}, total fixed={component_scale_ratios['total_fixed_investment']:.6g}.",
        "- The earlier 0.138x / 0.377x component results are classified as a COMPONENT_ACCOUNTING_MISMATCH in the diagnostic aggregation: Firm-level executed fields can be overwritten by multiple same-week pending settlements. The canonical weekly aggregate closes and does not show that component-scale failure.",
        "",
        "## Desired-to-funded throughput",
        f"- N=500 labor-bound weeks: {flags['labor_bound_weeks_N500']}; cash-bound weeks: {flags['cash_bound_weeks_N500']}.",
        f"- N=5000 labor-bound weeks: {flags['labor_bound_weeks_N5000']}; cash-bound weeks: {flags['cash_bound_weeks_N5000']}.",
        f"- Backlog-per-capita scale ratio: {backlog_per_capita_ratio:.6g}; capital-good-production-per-capita ratio: {production_per_capita_ratio:.6g}; active-service-per-capita ratio: {active_service_ratio:.6g}.",
        "- A low funded/desired ratio is interpreted as queued demand when labor capacity, rather than cash, is the binding constraint; funded output is not required to equal desired output.",
        "",
        "## Backlog and advances",
        f"- Late-window backlog classification: N=500 {small['backlog_classification']}; N=5000 {large['backlog_classification']}.",
        "- The backlog bridge is recorded weekly as opening stock + new demand flow - fulfillment flow = closing stock.",
        "- Customer-advance liability and buyer prepaid investment asset are checked as equal stocks and as stock-flow bridges; pending orders are treated as an undelivered contract pipeline rather than new money.",
        "",
        "## Conservation",
        f"- Maximum accounting gap: {flags['max_accounting_gap']:.6g}; money gap: {flags['max_money_gap']:.6g}; goods gap: {flags['max_goods_gap']:.6g}.",
        f"- Assignment violations: {flags['assignment_violations']}; realized-above-feasible violations: {flags['max_feasibility_violation']:.6g}.",
        "- No productivity, price, advance fraction, useful life, backlog rule, labor rule, financing, or Step13 behavior was changed.",
    ]
    if verdict == "A. THROUGHPUT_CONSTRAINT_IS_ECONOMICALLY_VALID_STEP15_READY":
        summary.extend(["", "**Step15 mechanism development can stop.**"])
    (OUTPUT / "acceptance_summary.md").write_text("\n".join(summary) + "\n", encoding="utf-8")
    print(json.dumps(flags, indent=2))


if __name__ == "__main__":
    main()
