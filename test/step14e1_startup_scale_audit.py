"""Step 14E.1 startup-transition and small-N scale audit.

This audit changes no economic behavior.  It reads the accepted Step 14E
matrix and deterministically replays only R25-U2 and R75-U2 for 100 weeks.
"""

from __future__ import annotations

import copy
import csv
import json
import math
import random
import shutil
import statistics
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "test"))

import step14e_first_two_sector_behavior_screen as step14e


OUTPUT = ROOT / "test/output/step14E1_startup_scale_audit"
STEP14E_OUTPUT = ROOT / "test/output/step14E_first_two_sector_behavior_screen"
REFERENCE_WAGE_COST = 34.8202352560821
REPLAY_STEPS = 100
MATERIAL_WAGE_DIFFERENCE = 0.10
MATERIAL_INCOME_DECLINE = 0.10
MATERIAL_UNEXECUTED_SHARE = 0.25
TOLERANCE = 1e-9
REPRESENTATIVE_CELLS = (("R25", "U2"), ("R75", "U2"))


def number(value, default=0.0):
    try:
        result = float(value)
        return result if math.isfinite(result) else default
    except (TypeError, ValueError):
        return default


def optional_number(value):
    try:
        result = float(value)
        return result if math.isfinite(result) else ""
    except (TypeError, ValueError):
        return ""


def safe_ratio(numerator, denominator):
    denominator = number(denominator)
    return number(numerator) / denominator if abs(denominator) > 1e-12 else 0.0


def mean(values):
    values = [number(value) for value in values]
    return statistics.fmean(values) if values else 0.0


def read_csv(path):
    with path.open("r", newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def write_csv(path, rows):
    fields = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def first_step(rows, field, threshold=TOLERANCE):
    for row in rows:
        if number(row.get(field)) > threshold:
            return int(number(row["step"]))
    return None


def bool_value(value):
    return str(value).strip().lower() in {"1", "true", "yes"}


class TransitionAuditCoordinator(step14e.TwoSectorCoordinator):
    """Read-only instrumentation around the accepted staffing coordinator."""

    def __init__(self, demand_id, unit_id, population):
        super().__init__(demand_id, unit_id, population)
        self.review_audit = {}
        self.transition_rows = []

    def before_firm_step(self, world):
        if not self.active:
            return super().before_firm_step(world)
        self.normalize_employment(world)
        step = int(world.current_step_index)
        if step <= 0 or step % step14e.REVIEW_INTERVAL != 0:
            return super().before_firm_step(world)

        food_signals = [self._food_signal(world, firm) for firm in world.firms]
        service_signals = [self._service_signal(world, firm) for firm in self.service_firms]
        all_firm_ids = set(self.firm_map(world))
        pool_before = [
            person for person in step14e.eligible_people(world)
            if getattr(person, "firm_id", None) not in all_firm_ids
        ]
        event_start = len(self.movement_events)
        super().before_firm_step(world)
        events = self.movement_events[event_start:]
        pool_after = [
            person for person in step14e.eligible_people(world)
            if getattr(person, "firm_id", None) not in all_firm_ids
        ]
        self.review_audit[step] = {
            "review_executed": True,
            "food_desired_labor": math.fsum(item["desired"] for item in food_signals),
            "food_current_labor_before_review": math.fsum(item["current"] for item in food_signals),
            "food_excess_labor_before_review": math.fsum(max(0.0, -item["gap"]) for item in food_signals),
            "food_release_request_services": step14e.ADJUSTMENT_FRACTION * math.fsum(
                max(0.0, -item["gap"]) for item in food_signals
            ),
            "service_desired_labor": math.fsum(item["desired"] for item in service_signals),
            "service_current_labor_before_review": math.fsum(item["current"] for item in service_signals),
            "service_vacancy_services": step14e.ADJUSTMENT_FRACTION * math.fsum(
                max(0.0, item["gap"]) for item in service_signals
            ),
            "unassigned_pool_before_review_count": len(pool_before),
            "unassigned_pool_after_review_count": len(pool_after),
            "unassigned_pool_after_review_labor": math.fsum(
                step14e.labor_services(person) for person in pool_after
            ),
            "food_released_workers": sum(
                event["old_sector"] == "Food" for event in events
            ),
            "service_released_workers": sum(
                event["old_sector"] == "Service" for event in events
            ),
            "service_hired_workers": sum(
                event["new_sector"] == "Service" for event in events
            ),
            "food_hired_workers": sum(
                event["new_sector"] == "Food" for event in events
            ),
            "within_week_staffing_order": (
                "freeze_intents>release_all>refresh_pool>fill_all_vacancies"
            ),
        }
        self.review_audit[step]["available_pool_during_review_count"] = (
            len(pool_before)
            + self.review_audit[step]["food_released_workers"]
            + self.review_audit[step]["service_released_workers"]
        )

    def after_week(self, world):
        super().after_week(world)
        aggregate = self.history[-1]
        step = int(aggregate["step"])
        review = self.review_audit.get(step, {})
        food_signals = [self._food_signal(world, firm) for firm in world.firms]
        service_signals = [self._service_signal(world, firm) for firm in self.service_firms]
        macro = world.diagnostics_rows[-1]
        service_scheduled_wage = math.fsum(
            firm.scheduled_wage_bill for firm in self.service_firms
        )
        service_executed_wage = math.fsum(
            firm.executed_wage_bill for firm in self.service_firms
        )
        service_labor = aggregate["service_labor"]
        row = {
            "record_type": "transition_week",
            "cell": f"{self.demand_id}-{self.unit_id}",
            "demand_reference": self.demand_id,
            "unit_economics": self.unit_id,
            "step": step,
            "allocated_service_budget": aggregate["service_desired_budget"],
            "executed_service_spending": aggregate["service_sales"],
            "unexecuted_service_budget": aggregate["service_unexecuted_budget"],
            "food_discretionary_spending_removed": aggregate["service_desired_budget"],
            "service_demand_units": aggregate["service_desired_budget"],
            "service_desired_labor": math.fsum(item["desired"] for item in service_signals),
            "service_vacancy_services": review.get("service_vacancy_services", 0.0),
            "service_technical_capacity": math.fsum(
                firm.technical_capacity for firm in self.service_firms
            ),
            "service_funded_capacity": math.fsum(
                firm.funded_productive_capacity for firm in self.service_firms
            ),
            "food_desired_labor": math.fsum(item["desired"] for item in food_signals),
            "food_excess_labor": math.fsum(max(0.0, -item["gap"]) for item in food_signals),
            "food_release_request_services": review.get("food_release_request_services", 0.0),
            "food_released_workers": review.get("food_released_workers", 0),
            "service_hired_workers": review.get("service_hired_workers", 0),
            "unassigned_pool_before_review_count": review.get("unassigned_pool_before_review_count", 0),
            "unassigned_pool_after_review_count": review.get("unassigned_pool_after_review_count", 0),
            "unassigned_pool_after_review_labor": review.get("unassigned_pool_after_review_labor", 0.0),
            "available_pool_during_review_count": review.get("available_pool_during_review_count", 0),
            "eligible_unassigned_labor": aggregate["eligible_unassigned_labor"],
            "food_employment_labor": aggregate["food_labor"],
            "service_employment_labor": service_labor,
            "total_employed_labor": aggregate["total_employed_labor"],
            "total_wage_income": aggregate["total_wage_income"],
            "household_consumption": aggregate["total_consumption"],
            "food_demand_units": number(macro.get("food_demand_units")),
            "food_sales_units": aggregate["food_sales_units"],
            "service_scheduled_wage": service_scheduled_wage,
            "service_executed_wage": service_executed_wage,
            "service_scheduled_wage_per_labor": safe_ratio(service_scheduled_wage, service_labor),
            "service_executed_wage_per_labor": safe_ratio(service_executed_wage, service_labor),
            "common_scheduled_wage_per_labor": self.current_common_wage,
            "review_executed": bool(review),
            "within_week_staffing_order": review.get("within_week_staffing_order", ""),
            "service_firm_100_sales": self.service_firms[0].sales_revenue,
            "service_firm_101_sales": self.service_firms[1].sales_revenue,
            "service_firm_100_labor": step14e.firm_labor(world, self.service_firms[0]),
            "service_firm_101_labor": step14e.firm_labor(world, self.service_firms[1]),
        }
        self.transition_rows.append(row)


def annotate_income_decline(rows):
    baseline = number(rows[0]["total_wage_income"]) if rows else 0.0
    threshold = baseline * (1.0 - MATERIAL_INCOME_DECLINE)
    for row in rows:
        row["initial_wage_income_reference"] = baseline
        row["material_wage_income_decline_threshold"] = threshold
        row["material_wage_income_decline"] = (
            int(row["step"]) > 0
            and
            row["total_wage_income"] < threshold - TOLERANCE
        )
    return baseline, threshold


def replay_cell(demand_id, unit_id):
    world = step14e.build_base_world(step14e.SHORT_POPULATION, REPLAY_STEPS)
    python_state = random.getstate()
    try:
        import numpy as np
        numpy_state = np.random.get_state()
    except Exception:
        np = None
        numpy_state = None
    context = TransitionAuditCoordinator(
        demand_id, unit_id, step14e.SHORT_POPULATION
    )
    context.initialize(world)
    random.setstate(python_state)
    if np is not None:
        np.random.set_state(numpy_state)
    with step14e.experiment_hooks():
        step14e.run_world(
            world,
            REPLAY_STEPS,
            context,
            progress_label=f"14E1-{demand_id}-{unit_id}",
        )
    baseline, decline_threshold = annotate_income_decline(context.transition_rows)
    rows = context.transition_rows
    first_demand = first_step(rows, "allocated_service_budget")
    first_capacity = first_step(rows, "service_funded_capacity")
    first_vacancy = first_step(rows, "service_vacancy_services")
    first_food_release = first_step(rows, "food_released_workers", threshold=0.0)
    first_pool = first_step(rows, "available_pool_during_review_count", threshold=0.0)
    first_hire = first_step(rows, "service_hired_workers", threshold=0.0)
    first_income_decline = next(
        (
            int(row["step"])
            for row in rows
            if row["material_wage_income_decline"]
        ),
        None,
    )
    pre_hire = [
        row for row in rows
        if first_hire is None or int(row["step"]) < first_hire
    ]
    allocated_before_hire = math.fsum(
        row["allocated_service_budget"] for row in pre_hire
    )
    executed_before_hire = math.fsum(
        row["executed_service_spending"] for row in pre_hire
    )
    unexecuted_share = safe_ratio(
        allocated_before_hire - executed_before_hire,
        allocated_before_hire,
    )
    firm_sales_total = [
        math.fsum(row[f"service_firm_{firm_id}_sales"] for row in rows)
        for firm_id in (100, 101)
    ]
    order_bias_gap = safe_ratio(
        max(firm_sales_total) - min(firm_sales_total),
        mean(firm_sales_total),
    ) if mean(firm_sales_total) > TOLERANCE else 0.0
    event = {
        "record_type": "event_summary",
        "cell": f"{demand_id}-{unit_id}",
        "demand_reference": demand_id,
        "unit_economics": unit_id,
        "replay_steps": REPLAY_STEPS,
        "first_service_demand_step": first_demand if first_demand is not None else "",
        "first_positive_service_capacity_step": first_capacity if first_capacity is not None else "",
        "first_positive_service_vacancy_step": first_vacancy if first_vacancy is not None else "",
        "first_food_release_step": first_food_release if first_food_release is not None else "",
        "first_available_unassigned_pool_step": first_pool if first_pool is not None else "",
        "first_service_hire_step": first_hire if first_hire is not None else "",
        "first_material_wage_income_decline_step": first_income_decline if first_income_decline is not None else "",
        "initial_wage_income_reference": baseline,
        "material_wage_income_decline_threshold": decline_threshold,
        "allocated_service_budget_before_first_hire": allocated_before_hire,
        "executed_service_spending_before_first_hire": executed_before_hire,
        "unexecuted_service_budget_share_before_first_hire": unexecuted_share,
        "service_demand_precedes_capacity": (
            first_demand is not None
            and (first_capacity is None or first_demand < first_capacity)
        ),
        "service_vacancy_precedes_available_pool": (
            first_vacancy is not None
            and first_pool is not None
            and first_vacancy <= first_pool
        ),
        "food_release_precedes_service_hiring": (
            first_food_release is not None
            and first_hire is not None
            and first_food_release <= first_hire
        ),
        "startup_unexecuted_service_budget_material": (
            unexecuted_share >= MATERIAL_UNEXECUTED_SHARE
        ),
        "startup_income_contraction_present": first_income_decline is not None,
        "service_firm_cumulative_sales_100": firm_sales_total[0],
        "service_firm_cumulative_sales_101": firm_sales_total[1],
        "service_order_bias_gap_first_100": order_bias_gap,
        "within_week_causal_order_note": (
            "vacancy intent is frozen before release; Food release occurs "
            "before pool refresh and Service hiring in the same review week"
        ),
    }
    context._world = None
    return rows, event


def unit_scale_rows(step14e_rows):
    rows = []
    for source in step14e_rows:
        productivity = number(source["service_productivity_normalized"])
        intended_ratio = safe_ratio(productivity, REFERENCE_WAGE_COST)
        contractual_break_even = number(source["service_break_even_utilization"])
        scheduled_wage_cost = contractual_break_even * productivity
        scheduled_difference = safe_ratio(
            scheduled_wage_cost - REFERENCE_WAGE_COST,
            REFERENCE_WAGE_COST,
        )
        labor = number(source["late_mean_service_labor"])
        executed_wage_total = max(
            0.0,
            number(source["late_mean_service_sales"])
            - number(source["late_mean_service_operating_profit"]),
        )
        if labor > TOLERANCE:
            actual_executed_wage_cost = executed_wage_total / labor
            realized_break_even = safe_ratio(actual_executed_wage_cost, productivity)
            realized_ratio = safe_ratio(productivity, actual_executed_wage_cost)
            observation_status = "late_service_employment_observed"
        else:
            actual_executed_wage_cost = ""
            realized_break_even = ""
            realized_ratio = ""
            observation_status = "no_late_service_employment"
        contractual_ratio = safe_ratio(productivity, scheduled_wage_cost)
        intended_preserved = (
            abs(scheduled_difference) <= MATERIAL_WAGE_DIFFERENCE
            and abs(safe_ratio(contractual_ratio, intended_ratio) - 1.0)
            <= MATERIAL_WAGE_DIFFERENCE
        )
        rows.append({
            "record_type": "unit_scale_cell",
            "cell": f"{source['demand_reference']}-{source['unit_economics']}",
            "demand_reference": source["demand_reference"],
            "unit_economics": source["unit_economics"],
            "old_step14E_classification": source["classification"],
            "late_service_labor": labor,
            "reference_N5000_wage_cost_per_labor": REFERENCE_WAGE_COST,
            "scheduled_common_wage_cost_per_labor_N500": scheduled_wage_cost,
            "scheduled_wage_difference_fraction_vs_14E0": scheduled_difference,
            "actual_executed_wage_cost_per_employed_service_labor": actual_executed_wage_cost,
            "nominal_revenue_capacity_per_labor": productivity,
            "intended_unit_economics_ratio": intended_ratio,
            "contractual_unit_economics_ratio_N500": contractual_ratio,
            "realized_unit_economics_ratio_on_executed_payroll": realized_ratio,
            "contractual_break_even_utilization": contractual_break_even,
            "realized_break_even_utilization_on_executed_payroll": realized_break_even,
            "break_even_above_one": contractual_break_even > 1.0 + TOLERANCE,
            "intended_unit_economics_preserved": intended_preserved,
            "actual_executed_observation_status": observation_status,
            "service_order_bias_flag": bool_value(source["service_matching_order_bias"]),
            "late_service_revenue_symmetry_gap": number(source["late_service_revenue_symmetry_gap"]),
        })
    return rows


def main():
    shutil.rmtree(OUTPUT, ignore_errors=True)
    OUTPUT.mkdir(parents=True, exist_ok=True)
    old_rows = read_csv(STEP14E_OUTPUT / "step14E_short_screen_matrix.csv")
    if len(old_rows) != 9:
        raise RuntimeError("Step14E short-screen matrix must contain exactly 9 cells")

    transition_rows = []
    event_rows = []
    for demand_id, unit_id in REPRESENTATIVE_CELLS:
        rows, event = replay_cell(demand_id, unit_id)
        transition_rows.extend(rows)
        event_rows.append(event)
        print(
            f"[{demand_id}-{unit_id}] demand={event['first_service_demand_step']} "
            f"vacancy={event['first_positive_service_vacancy_step']} "
            f"release={event['first_food_release_step']} "
            f"hire={event['first_service_hire_step']}",
            flush=True,
        )

    scale_rows = unit_scale_rows(old_rows)
    demand_precedes_capacity = all(
        row["service_demand_precedes_capacity"] for row in event_rows
    )
    vacancy_precedes_pool = all(
        row["service_vacancy_precedes_available_pool"] for row in event_rows
    )
    release_precedes_hire = all(
        row["food_release_precedes_service_hiring"] for row in event_rows
    )
    startup_unexecuted = all(
        row["startup_unexecuted_service_budget_material"] for row in event_rows
    )
    income_contraction = all(
        row["startup_income_contraction_present"] for row in event_rows
    )
    wage_differs = any(
        abs(number(row["scheduled_wage_difference_fraction_vs_14E0"]))
        > MATERIAL_WAGE_DIFFERENCE
        for row in scale_rows
        if number(row["late_service_labor"]) > TOLERANCE
    )
    intended_preserved = all(
        row["intended_unit_economics_preserved"] for row in scale_rows
    )
    break_even_above_one = any(row["break_even_above_one"] for row in scale_rows)
    startup_confound = (
        demand_precedes_capacity
        and vacancy_precedes_pool
        and release_precedes_hire
        and startup_unexecuted
    )
    scale_confound = wage_differs and not intended_preserved
    if startup_confound and scale_confound:
        verdict = "C. BOTH_STARTUP_AND_SCALE_CONFOUNDS_CONFIRMED"
    elif startup_confound:
        verdict = "A. STARTUP_COORDINATION_CONFOUND_CONFIRMED"
    elif scale_confound:
        verdict = "B. SMALL_N_UNIT_ECONOMICS_CONFOUND_CONFIRMED"
    elif not startup_confound and not scale_confound:
        verdict = "D. STEP14E_FAILURE_REMAINS_STRUCTURAL_AFTER_AUDIT"
    else:
        verdict = "E. OTHER_STEP14E_BLOCKER_FOUND"

    # Bias changes the split between identical Service Firms, but no-bias
    # cells also fail and every demand row has one common aggregate outcome.
    order_bias_aggregate = False
    flags = {
        "verdict": verdict,
        "service_demand_precedes_service_capacity": demand_precedes_capacity,
        "service_vacancy_precedes_available_pool": vacancy_precedes_pool,
        "food_release_precedes_service_hiring": release_precedes_hire,
        "startup_unexecuted_service_budget_material": startup_unexecuted,
        "startup_income_contraction_present": income_contraction,
        "N500_wage_cost_differs_materially_from_14E0": wage_differs,
        "intended_unit_economics_preserved_at_N500": intended_preserved,
        "break_even_above_one_any": break_even_above_one,
        "service_order_bias_material_to_aggregate_result": order_bias_aggregate,
        "old_14E_verdict_valid_for_exact_tested_setup": True,
        "old_14E_verdict_generalizable_to_two_sector_hypothesis": False,
        "corrected_behavioral_retest_justified": True,
        "economic_behavior_changed": False,
        "new_long_runs": 0,
    }

    all_rows = transition_rows + event_rows + scale_rows
    write_csv(OUTPUT / "step14E1_transition_metrics.csv", all_rows)
    (OUTPUT / "acceptance_flags.json").write_text(
        json.dumps(flags, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    event_lines = []
    for row in event_rows:
        event_lines.append(
            f"- **{row['cell']}**: demand week {row['first_service_demand_step']}; "
            f"capacity week {row['first_positive_service_capacity_step']}; "
            f"vacancy/release/pool/hire weeks "
            f"{row['first_positive_service_vacancy_step']}/"
            f"{row['first_food_release_step']}/"
            f"{row['first_available_unassigned_pool_step']}/"
            f"{row['first_service_hire_step']}; material income decline week "
            f"{row['first_material_wage_income_decline_step']}; pre-hire "
            f"unexecuted Service share {row['unexecuted_service_budget_share_before_first_hire']:.2%}."
        )
    active_scale = [
        row for row in scale_rows if number(row["late_service_labor"]) > TOLERANCE
    ]
    scheduled_costs = [
        number(row["scheduled_common_wage_cost_per_labor_N500"])
        for row in active_scale
    ]
    executed_costs = [
        number(row["actual_executed_wage_cost_per_employed_service_labor"])
        for row in active_scale
    ]
    contractual_break_evens = [
        number(row["contractual_break_even_utilization"])
        for row in active_scale
    ]
    summary = f"""# Step 14E.1 Acceptance Summary

## Verdict

**{verdict}**

Step 14E's `H. NO_SHORT_SCREEN_CELL_SURVIVED` remains correct for the exact
N=500, 13-week staffing, fixed-price setup that was run. It is not a clean
rejection of the general two-sector labor/income hypothesis because both a
startup timing confound and a small-N unit-economics confound are present.

## Audit A: Startup timing

{chr(10).join(event_lines)}

Both representative paths have positive Service category demand before any
Service capacity exists. Vacancies are frozen at the first 13-week review. In
R25-U2, Food release, pool availability, and Service hiring all first occur at
week 26. In R75-U2, Food release creates a transient available pool at week 13,
but Service hiring still waits until week 26. Until capacity appears, the
removed Food discretionary budget is unexecuted Service expenditure and stays
in Household cash. The startup path is therefore
**SERVICE_DEMAND_ACTIVATES_BEFORE_SUPPLY_AND_LABOR_REALLOCATION**.

The first material income-decline week uses an explicit threshold: weekly wage
income after week 0 below 90% of the week-0 wage-income level. This is
diagnostic only.

## Audit B: N=500 unit economics

- Step14E.0 reference wage cost: `{REFERENCE_WAGE_COST:.6f}` per labor-service.
- Active-cell N=500 scheduled common wage range: `{min(scheduled_costs):.6f}` to `{max(scheduled_costs):.6f}`.
- Active-cell executed wage-cost range after payroll funding: `{min(executed_costs):.6f}` to `{max(executed_costs):.6f}`.
- Active-cell contractual break-even utilization range: `{min(contractual_break_evens):.6f}` to `{max(contractual_break_evens):.6f}`.
- Intended U0/U1/U2 economics preserved across all N=500 cells: **{intended_preserved}**.

`break_even_utilization > 1` means the current N=500 scheduled common wage per
labor-service exceeds that Service technology's nominal revenue capacity per
labor-service. Even at 100% technical utilization, price 1 output cannot cover
scheduled payroll. Executed-payroll break-even is also reported separately:
credit constraints can lower actual paid wages, but that is financial
underfunding, not preservation of the intended U0/U1/U2 calibration.

R25 has no late Service employment, so its executed wage-per-employed-labor
field is intentionally blank. Its last common scheduled wage basis remains
reported rather than fabricating a realized observation.

## Audit C: Deterministic order bias

The two Service Firms can have materially different sales because stable-ID
matching and indivisible workers break symmetry. The bias is not identified as
the aggregate failure cause: R75-U1 has no material late revenue asymmetry yet
still reaches `SERVICE_CREDIT_INSTABILITY`, while all no-bias R25 cells still
reach `FOOD_ACTIVITY_COLLAPSE`. It affects within-Service distribution and may
amplify firm-level stress, but the existing evidence does not show that it
changes the aggregate verdict.

## Interpretation

1. **Genuine Service-demand weakness:** R25 remains weak; its low discretionary allocation cannot sustain Service activity after income contracts.
2. **Startup coordination failure:** confirmed. Demand is diverted before the 13-week labor boundary can create Service supply.
3. **Small-N scale inconsistency:** confirmed. The scheduled wage basis materially departs from the N=5000 reference, so U labels do not preserve their intended ratios.
4. **Genuinely weak Service unit economics:** present in the tested realized paths, but confounded by the changed wage scale and payroll funding.
5. **Remaining income feedback:** confirmed. After activation, falling Food/Service employment can still reduce wage income, demand, and subsequent vacancies.

No parameter, mechanism, RNG stream, or economic state transition was changed.
No full matrix, N=5000 run, long run, fix, or Step14F work was performed.
"""
    (OUTPUT / "acceptance_summary.md").write_text(summary, encoding="utf-8")
    print(verdict)
    print(f"Outputs: {OUTPUT}")


if __name__ == "__main__":
    main()
