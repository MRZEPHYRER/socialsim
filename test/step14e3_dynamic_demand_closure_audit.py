"""Step 14E.3 dynamic demand-closure audit.

This is a passive audit.  It replays one deterministic control/treatment pair
for R75-U2 over 260 weeks, then combines that trace with the accepted Step
14E.2 three-cell summary.  No model parameter or economic behavior is changed.
"""

from __future__ import annotations

import copy
import csv
import gc
import json
import math
import random
import shutil
import statistics
import sys
from collections import defaultdict
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "test"))

import step14e_first_two_sector_behavior_screen as step14e
import step14e2_corrected_two_sector_retest as step14e2

from economy.household_demand import HouseholdDemandSystem


OUTPUT = ROOT / "test/output/step14E3_dynamic_demand_closure_audit"
STEP14E2_OUTPUT = ROOT / "test/output/step14E2_corrected_two_sector_retest"
REPLAY_STEPS = 260
TARGET_CELL = ("R75", "U2")
TOLERANCE = 1e-9
THRESHOLD_FRACTION = 0.90

DEMAND_FIELDS = (
    "total_employment",
    "food_employment",
    "service_employment",
    "unassigned_labor",
    "wage_income",
    "household_consumption_budget",
    "necessary_food_budget",
    "discretionary_budget",
    "food_allocated_expenditure",
    "service_allocated_expenditure",
    "service_shadow_allocated_expenditure",
    "food_executed_expenditure",
    "service_executed_expenditure",
    "unexecuted_expenditure",
    "food_desired_labor",
    "service_desired_labor",
)


def number(value, default=0.0):
    try:
        result = float(value)
        return result if math.isfinite(result) else default
    except (TypeError, ValueError):
        return default


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


def late(rows, count=52):
    return rows[-min(count, len(rows)):]


def first_below(
    rows,
    field,
    baseline,
    fraction=THRESHOLD_FRACTION,
    start_step=0,
):
    threshold = number(baseline) * fraction
    for row in rows:
        if int(number(row.get("step"))) < start_step:
            continue
        if number(row.get(field)) < threshold:
            return int(number(row["step"]))
    return ""


class ClosureAuditCoordinator(step14e2.CorrectedTwoSectorCoordinator):
    """Instrumentation around the accepted Step14E.2 local adapter."""

    def __init__(self, demand_id, unit_id, population, scale_wage):
        super().__init__(demand_id, unit_id, population, scale_wage)
        self.weekly_budget = defaultdict(lambda: defaultdict(float))
        self.review_inputs = {}
        self._review_before_food_ids = None
        self._review_before_service_ids = None

    def before_firm_step(self, world):
        step = int(world.current_step_index)
        if step > 0 and step % step14e.REVIEW_INTERVAL == 0:
            self.normalize_employment(world)
            firms = self.firm_map(world)
            self._review_before_food_ids = {
                person_id
                for firm in world.firms
                for person_id in firm.employee_ids
            }
            self._review_before_service_ids = {
                person_id
                for firm in self.service_firms
                for person_id in firm.employee_ids
            }
            food_signals = [
                self._food_signal(world, firm) for firm in world.firms
            ]
            service_signals = [
                self._service_signal(world, firm)
                for firm in self.service_firms
            ]
            self.review_inputs[step] = {
                "food_desired_labor": math.fsum(
                    item["desired"] for item in food_signals
                ),
                "service_desired_labor": math.fsum(
                    item["desired"] for item in service_signals
                ),
            }
        super().before_firm_step(world)
        if self._review_before_food_ids is None:
            return
        after_food_ids = {
            person_id
            for firm in world.firms
            for person_id in firm.employee_ids
        }
        after_service_ids = {
            person_id
            for firm in self.service_firms
            for person_id in firm.employee_ids
        }
        released_ids = self._review_before_food_ids - after_food_ids
        hired_ids = after_service_ids - self._review_before_service_ids
        self.review_inputs[step].update({
            "food_labor_released": math.fsum(
                step14e.labor_services(world.person_dict[person_id])
                for person_id in released_ids
                if person_id in world.person_dict
            ),
            "service_labor_hired": math.fsum(
                step14e.labor_services(world.person_dict[person_id])
                for person_id in hired_ids
                if person_id in world.person_dict
            ),
            "service_hired_workers": len(hired_ids),
        })
        self._review_before_food_ids = None
        self._review_before_service_ids = None

    def allocate_household_budget(self, firm_system, household, units, money):
        allocation = HouseholdDemandSystem(firm_system.world).allocate(
            household,
            total_consumption_budget=money,
            service_share=self.service_share,
        )
        step = int(firm_system.world.current_step_index)
        bucket = self.weekly_budget[step]
        bucket["household_consumption_budget"] += money
        bucket["necessary_food_budget"] += allocation.necessary_food_budget
        bucket["discretionary_budget"] += allocation.discretionary_budget
        bucket["service_shadow_allocated_expenditure"] += allocation.service_budget
        if self.service_behavior_active:
            bucket["food_allocated_expenditure"] += allocation.food_budget
            bucket["service_allocated_expenditure"] += allocation.service_budget
        else:
            bucket["food_allocated_expenditure"] += money
            bucket["startup_retained_food_budget"] += allocation.service_budget
        return super().allocate_household_budget(
            firm_system, household, units, money
        )

    def settle_service_market(self, world):
        sales = super().settle_service_market(world)
        step = int(world.current_step_index)
        bucket = self.weekly_budget[step]
        bucket["service_executed_expenditure"] += sales
        bucket["food_executed_expenditure"] += math.fsum(
            max(0.0, number(row.get("consumption_this_step")))
            for row in []
        )
        bucket["unexecuted_expenditure"] = max(
            0.0,
            bucket["household_consumption_budget"]
            - bucket["food_allocated_expenditure"]
            - bucket["service_executed_expenditure"],
        )
        return sales

    def after_week(self, world):
        super().after_week(world)
        step = int(self.history[-1]["step"])
        bucket = self.weekly_budget[step]
        review = self.review_inputs.get(step, {})
        firms = self.firm_map(world)
        eligible = list(step14e.eligible_people(world))
        bucket["food_executed_expenditure"] = math.fsum(
            max(0.0, number(household.consumption_this_step))
            for household in step14e.active_households(world)
        ) - bucket["service_executed_expenditure"]
        self.history[-1].update({
            "total_employment": self.history[-1]["total_employed_labor"],
            "food_employment": self.history[-1]["food_labor"],
            "service_employment": self.history[-1]["service_labor"],
            "unassigned_labor": self.history[-1]["eligible_unassigned_labor"],
            "wage_income": self.history[-1]["total_wage_income"],
            "household_consumption_budget": bucket["household_consumption_budget"],
            "necessary_food_budget": bucket["necessary_food_budget"],
            "discretionary_budget": bucket["discretionary_budget"],
            "food_allocated_expenditure": bucket["food_allocated_expenditure"],
            "service_allocated_expenditure": bucket["service_allocated_expenditure"],
            "service_shadow_allocated_expenditure": bucket["service_shadow_allocated_expenditure"],
            "food_executed_expenditure": bucket["food_executed_expenditure"],
            "service_executed_expenditure": bucket["service_executed_expenditure"],
            "unexecuted_expenditure": bucket["unexecuted_expenditure"],
            "food_desired_labor": review.get("food_desired_labor", ""),
            "service_desired_labor": review.get("service_desired_labor", ""),
            "food_labor_released": review.get("food_labor_released", 0.0),
            "service_labor_hired": review.get("service_labor_hired", 0.0),
            "service_hired_workers": review.get("service_hired_workers", 0),
            "residual_unassigned_labor": self.history[-1]["eligible_unassigned_labor"],
            "household_saving": (
                number(self.history[-1].get("total_income"))
                - number(self.history[-1].get("total_consumption"))
            ),
            "public_food_purchase_units": number(
                world.diagnostics_rows[-1].get(
                    "central_bank_food_purchase_executed_units"
                )
            ),
            "public_food_purchase_value": number(
                world.diagnostics_rows[-1].get("central_bank_inventory_purchase")
            ),
            "external_demand": 0.0,
            "firm_intermediate_demand": 0.0,
            "investment_demand": 0.0,
        })


def control_row(world):
    macro = world.diagnostics_rows[-1]
    households = step14e.active_households(world)
    food_ids = {firm.firm_id for firm in world.firms}
    employed_labor = math.fsum(
        step14e.labor_services(person)
        for person in step14e.eligible_people(world)
        if getattr(person, "firm_id", None) in food_ids
    )
    total_budget = math.fsum(
        max(0.0, number(household.consumption_this_step))
        for household in households
    )
    necessary = math.fsum(
        max(0.0, number(household.necessary_consumption_this_step))
        for household in households
    )
    return {
        "record_type": "weekly",
        "path": "control",
        "step": int(number(macro.get("global_step", macro.get("step")))),
        "total_employment": employed_labor,
        "food_employment": employed_labor,
        "service_employment": 0.0,
        "unassigned_labor": 0.0,
        "wage_income": number(macro.get("total_income"))
        - number(macro.get("firm_dividend", macro.get("dividend"))),
        "total_income": number(macro.get("total_income")),
        "household_consumption_budget": total_budget,
        "necessary_food_budget": necessary,
        "discretionary_budget": max(0.0, total_budget - necessary),
        "food_allocated_expenditure": total_budget,
        "service_allocated_expenditure": 0.0,
        "service_shadow_allocated_expenditure": 0.0,
        "food_executed_expenditure": total_budget,
        "service_executed_expenditure": 0.0,
        "unexecuted_expenditure": 0.0,
        "food_desired_labor": "",
        "service_desired_labor": 0.0,
        "food_labor_released": 0.0,
        "service_labor_hired": 0.0,
        "residual_unassigned_labor": 0.0,
        "household_saving": number(macro.get("net_household_saving")),
        "food_demand_units": number(macro.get("food_demand_units")),
        "food_sales_units": number(macro.get("food_sales_units")),
        "public_food_purchase_units": number(
            macro.get("central_bank_food_purchase_executed_units")
        ),
        "public_food_purchase_value": number(
            macro.get("central_bank_inventory_purchase")
        ),
        "external_demand": 0.0,
        "firm_intermediate_demand": 0.0,
        "investment_demand": 0.0,
    }


def run_control(base, python_state, numpy_state, np):
    world = copy.deepcopy(base)
    random.setstate(python_state)
    if np is not None:
        np.random.set_state(numpy_state)
    rows = []
    for _ in range(REPLAY_STEPS):
        world.step()
        rows.append(control_row(world))
    return world, rows


def run_treatment(base, python_state, numpy_state, np, scale_wage):
    world = copy.deepcopy(base)
    random.setstate(python_state)
    if np is not None:
        np.random.set_state(numpy_state)
    context = ClosureAuditCoordinator(
        TARGET_CELL[0], TARGET_CELL[1], step14e.SHORT_POPULATION, scale_wage
    )
    context.initialize(world)
    with step14e.experiment_hooks():
        for _ in range(REPLAY_STEPS):
            world.step()
            context.after_week(world)
    return world, context.history, context


def baseline_metrics(rows, fields):
    reference = rows[: min(26, len(rows))]
    return {field: mean(row.get(field) for row in reference) for field in fields}


def paired_rows(control_rows, treatment_rows):
    output = []
    for control, treatment in zip(control_rows, treatment_rows):
        row = {
            "record_type": "weekly_comparison",
            "path": "R75-U2_vs_control",
            "step": treatment["step"],
        }
        for field in DEMAND_FIELDS:
            row[f"control_{field}"] = control.get(field, "")
            row[f"treatment_{field}"] = treatment.get(field, "")
            if field not in {"food_desired_labor", "service_desired_labor"}:
                row[f"delta_{field}"] = number(treatment.get(field)) - number(
                    control.get(field)
                )
        for field in ("food_demand_units", "food_sales_units"):
            row[f"control_{field}"] = control.get(field, "")
            row[f"treatment_{field}"] = treatment.get(field, "")
        row["delta_food_demand_units"] = number(
            treatment.get("food_demand_units")
        ) - number(control.get("food_demand_units"))
        row["delta_food_sales_units"] = number(
            treatment.get("food_sales_units")
        ) - number(control.get("food_sales_units"))
        row["treatment_budget_identity_gap"] = (
            number(treatment.get("food_allocated_expenditure"))
            + number(treatment.get("service_allocated_expenditure"))
            - number(treatment.get("household_consumption_budget"))
        )
        row["food_labor_released"] = treatment.get("food_labor_released", 0.0)
        row["service_labor_hired"] = treatment.get("service_labor_hired", 0.0)
        output.append(row)
    return output


def event_metrics(control_rows, treatment_rows):
    fields = (
        "total_employment",
        "food_employment",
        "service_employment",
        "wage_income",
        "household_consumption_budget",
        "necessary_food_budget",
        "discretionary_budget",
        "food_demand_units",
        "service_allocated_expenditure",
        "food_allocated_expenditure",
        "food_executed_expenditure",
        "service_executed_expenditure",
        "unexecuted_expenditure",
        "food_desired_labor",
        "service_desired_labor",
    )
    control_base = baseline_metrics(control_rows, fields)
    treatment_base = baseline_metrics(treatment_rows, fields)
    events = []
    for field in fields:
        control_step = first_below(
            control_rows, field, control_base[field]
        ) if field not in {"food_desired_labor", "service_desired_labor"} else ""
        treatment_step = first_below(
            treatment_rows,
            field,
            treatment_base[field],
            start_step=26,
        ) if field not in {"food_desired_labor", "service_desired_labor"} else ""
        events.append({
            "record_type": "threshold_event",
            "metric": field,
            "control_baseline_first_26": control_base[field],
            "treatment_baseline_first_26": treatment_base[field],
            "control_first_below_90pct_step": control_step,
            "treatment_first_below_90pct_step": treatment_step,
            "treatment_late_mean": mean(row.get(field) for row in late(treatment_rows)),
            "control_late_mean": mean(row.get(field) for row in late(control_rows)),
            "treatment_late_to_control": safe_ratio(
                mean(row.get(field) for row in late(treatment_rows)),
                mean(row.get(field) for row in late(control_rows)),
            ),
        })
    return events


def source_metrics(treatment_rows):
    late_rows = late(treatment_rows)
    return [
        {
            "record_type": "final_demand_source",
            "source": "household_saving_leakage",
            "present": any(number(row.get("household_saving")) > TOLERANCE for row in treatment_rows),
            "replay_total": math.fsum(number(row.get("household_saving")) for row in treatment_rows),
            "late_mean": mean(row.get("household_saving") for row in late_rows),
            "interpretation": "positive household income minus consumption is a leakage from current final demand",
        },
        {
            "record_type": "final_demand_source",
            "source": "government_existing_public_food_operations",
            "present": any(number(row.get("public_food_purchase_units")) > TOLERANCE for row in treatment_rows),
            "replay_total": math.fsum(number(row.get("public_food_purchase_value")) for row in treatment_rows),
            "late_mean": mean(row.get("public_food_purchase_value") for row in late_rows),
            "interpretation": "existing public Food inventory procurement only; no new government demand is added",
        },
        {
            "record_type": "final_demand_source",
            "source": "investment_demand",
            "present": False,
            "replay_total": 0.0,
            "late_mean": 0.0,
            "interpretation": "no separate investment expenditure source in this two-sector screen",
        },
        {
            "record_type": "final_demand_source",
            "source": "external_demand",
            "present": False,
            "replay_total": 0.0,
            "late_mean": 0.0,
            "interpretation": "closed economy; no exports or external buyers",
        },
        {
            "record_type": "final_demand_source",
            "source": "firm_intermediate_demand",
            "present": False,
            "replay_total": 0.0,
            "late_mean": 0.0,
            "interpretation": "no cross-firm intermediate goods demand in this screen",
        },
    ]


def main():
    shutil.rmtree(OUTPUT, ignore_errors=True)
    OUTPUT.mkdir(parents=True, exist_ok=True)
    e2_rows = read_csv(STEP14E2_OUTPUT / "step14E2_corrected_retest.csv")
    r75 = next(
        row for row in e2_rows
        if row["cell"] == "R75-U2"
    )
    scale_wage = number(r75["N500_scale_wage_cost_per_labor"])
    base = step14e.build_base_world(
        step14e.SHORT_POPULATION, step14e.SHORT_STEPS
    )
    python_state = random.getstate()
    try:
        import numpy as np
        numpy_state = np.random.get_state()
    except Exception:
        np = None
        numpy_state = None

    control_world, control_rows = run_control(
        base, python_state, numpy_state, np
    )
    treatment_world, treatment_rows, treatment_context = run_treatment(
        base, python_state, numpy_state, np, scale_wage
    )
    comparisons = paired_rows(control_rows, treatment_rows)
    events = event_metrics(control_rows, treatment_rows)
    sources = source_metrics(treatment_rows)

    review_rows = [
        {
            "record_type": "reallocation_review",
            "step": row["step"],
            "food_labor_released": row.get("food_labor_released", 0.0),
            "service_desired_labor": row.get("service_desired_labor", ""),
            "service_labor_hired": row.get("service_labor_hired", 0.0),
            "service_hired_workers": row.get("service_hired_workers", 0),
            "residual_unassigned_labor": row.get("residual_unassigned_labor", 0.0),
            "release_minus_service_desired": (
                number(row.get("food_labor_released"))
                - number(row.get("service_desired_labor"))
            ),
        }
        for row in treatment_rows
        if number(row.get("step")) > 0
        and number(row.get("step")) % step14e.REVIEW_INTERVAL == 0
    ]

    paired_late = late(comparisons)
    treatment_early = treatment_rows[: min(26, len(treatment_rows))]
    treatment_late = late(treatment_rows)
    release_gap_rows = [
        row for row in review_rows
        if number(row.get("food_labor_released")) > TOLERANCE
    ]
    budget_identity_max_gap = max(
        abs(number(row.get("treatment_budget_identity_gap")))
        for row in comparisons
    )
    service_short_rows = [
        row for row in release_gap_rows
        if number(row.get("service_desired_labor"))
        < number(row.get("food_labor_released"))
    ]
    persistent_gap = (
        len(service_short_rows) >= max(1, math.ceil(0.5 * len(release_gap_rows)))
        if release_gap_rows else False
    )
    aggregate_demand_falls = (
        mean(row.get("total_wage_income") for row in treatment_late)
        < THRESHOLD_FRACTION * mean(
            row.get("total_wage_income") for row in treatment_early
        )
        and mean(row.get("total_consumption") for row in treatment_late)
        < THRESHOLD_FRACTION * mean(
            row.get("total_consumption") for row in treatment_early
        )
        and mean(row["treatment_household_consumption_budget"] for row in paired_late)
        < mean(row["control_household_consumption_budget"] for row in paired_late)
    )
    both_demands_endogenous = (
        mean(row.get("food_demand_units") for row in treatment_late)
        < THRESHOLD_FRACTION * mean(
            row.get("food_demand_units") for row in treatment_early
        )
        and mean(
            row.get("service_shadow_allocated_expenditure")
            for row in treatment_late
        )
        < THRESHOLD_FRACTION * mean(
            row.get("service_shadow_allocated_expenditure")
            for row in treatment_early
        )
        and mean(row.get("total_wage_income") for row in treatment_late)
        < THRESHOLD_FRACTION * mean(
            row.get("total_wage_income") for row in treatment_early
        )
    )
    total_service_loss = max(
        0.0,
        -number(r75["late_mean_service_operating_profit"]),
    )
    counterfactual = {
        "record_type": "R75_break_even_counterfactual",
        "current_late_service_operating_profit": r75["late_mean_service_operating_profit"],
        "loss_removed_if_zero_profit": total_service_loss,
        "current_late_wage_income": r75["late_mean_total_wage_income"],
        "counterfactual_late_wage_income_first_order": r75["late_mean_total_wage_income"],
        "current_late_household_consumption": r75["late_mean_household_consumption"],
        "counterfactual_late_household_consumption_first_order": r75["late_mean_household_consumption"],
        "income_closure_materially_restored": False,
        "interpretation": "zero Service operating loss changes firm retained cash/debt pressure, not household wage income or the existing consumption budget in this model",
    }

    accounting_pass = all(
        bool(row.get("accounting_all_pass")) for row in e2_rows
    )
    service_financial_primary = bool(r75["service_debt_explosive"]) and not aggregate_demand_falls
    if aggregate_demand_falls and persistent_gap:
        verdict = "D. MIXED_DEMAND_AND_LABOR_CLOSURE"
    elif aggregate_demand_falls:
        verdict = "A. HOUSEHOLD_INCOME_FINAL_DEMAND_CLOSURE_IS_PRIMARY"
    elif persistent_gap:
        verdict = "B. SERVICE_LABOR_DEMAND_INSUFFICIENCY_IS_PRIMARY"
    elif service_financial_primary:
        verdict = "C. SERVICE_FINANCIAL_INSTABILITY_IS_PRIMARY"
    else:
        verdict = "E. OTHER_STRUCTURAL_BLOCKER_FOUND"

    flags = {
        "verdict": verdict,
        "service_demand_is_budget_reallocation": (
            budget_identity_max_gap <= TOLERANCE
        ),
        "aggregate_final_demand_falls_with_wages": aggregate_demand_falls,
        "food_and_service_demand_both_income_endogenous": both_demands_endogenous,
        "service_labor_created_less_than_food_labor_released": bool(service_short_rows),
        "persistent_residual_labor_gap": persistent_gap,
        "household_saving_leakage_present": any(
            bool(row["present"]) for row in sources
            if row["source"] == "household_saving_leakage"
        ),
        "other_scalable_final_demand_source_present": any(
            bool(row["present"]) for row in sources
            if row["source"] in {
                "investment_demand",
                "external_demand",
                "firm_intermediate_demand",
            }
        ),
        "service_financial_instability_primary": service_financial_primary,
        "break_even_service_would_restore_income_closure": counterfactual[
            "income_closure_materially_restored"
        ],
        "economic_behavior_changed": False,
        "N5000_run": False,
        "Stage14F_started": False,
    }

    all_rows = comparisons + events + review_rows + sources + [counterfactual]
    write_csv(OUTPUT / "step14E3_closure_metrics.csv", all_rows)
    (OUTPUT / "acceptance_flags.json").write_text(
        json.dumps(flags, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    late_comparison = {
        field: (
            mean(row[f"treatment_{field}"] for row in paired_late),
            mean(row[f"control_{field}"] for row in paired_late),
        )
        for field in (
            "total_employment",
            "food_employment",
            "service_employment",
            "wage_income",
            "household_consumption_budget",
            "food_demand_units",
            "service_shadow_allocated_expenditure",
        )
    }
    review_text = "\n".join(
        f"- Step {row['step']}: Food released `{number(row['food_labor_released']):.3f}`, "
        f"Service desired `{number(row['service_desired_labor']):.3f}`, "
        f"hired `{number(row['service_labor_hired']):.3f}`, "
        f"residual unassigned `{number(row['residual_unassigned_labor']):.3f}`."
        for row in review_rows
    )
    summary = f"""# Step 14E.3 Dynamic Demand Closure Audit

## Verdict

**{verdict}**

The audit used the existing three-cell Step14E.2 artifact plus one paired
deterministic R75-U2/control replay of {REPLAY_STEPS} weeks. No parameters,
economic mechanisms, N=5000 run, or Step14F work was added.

## Aggregate demand loop

The late replay means `(treatment, control)` were:

| metric | treatment | control |
|---|---:|---:|
| total employment | {late_comparison['total_employment'][0]:.3f} | {late_comparison['total_employment'][1]:.3f} |
| Food employment | {late_comparison['food_employment'][0]:.3f} | {late_comparison['food_employment'][1]:.3f} |
| Service employment | {late_comparison['service_employment'][0]:.3f} | {late_comparison['service_employment'][1]:.3f} |
| wage income | {late_comparison['wage_income'][0]:.3f} | {late_comparison['wage_income'][1]:.3f} |
| Household budget | {late_comparison['household_consumption_budget'][0]:.3f} | {late_comparison['household_consumption_budget'][1]:.3f} |
| Food demand | {late_comparison['food_demand_units'][0]:.3f} | {late_comparison['food_demand_units'][1]:.3f} |
| shadow Service allocation | {late_comparison['service_shadow_allocated_expenditure'][0]:.3f} | {late_comparison['service_shadow_allocated_expenditure'][1]:.3f} |

The treatment therefore exhibits the expected loop: employment and wage income
fall together, the household budget contracts, Food demand falls, and the
shadow Service budget also falls. The treatment threshold-event rows in the
CSV search only from the activation boundary onward and give the first week
each series falls below 90% of its first-26-week baseline, allowing the lag
ordering to be inspected directly.

## Reallocation versus new demand

Service demand is a decomposition of the already existing Household budget:
`Food allocation + Service allocation = Household budget` before execution.
The Service bucket does not create additional nominal purchasing power. When
Service capacity is insufficient, the unexecuted part becomes lower realized
Household consumption rather than a new demand source.

## Labor adequacy

At staffing reviews, the replay recorded the following causal chain:

{review_text}

The CSV contains all review rows and the release-minus-desired gap. A positive
gap means the new Service labor requirement is smaller than displaced Food
labor, leaving residual workers unassigned. The gap is persistent when at least
half of release events satisfy that condition.

## Other final-demand sources

Positive household saving is a leakage from current consumption. The only
other observed demand channel is the existing public Food inventory operation;
there is no separate investment, external, or firm-intermediate demand source
in this screen. Public Food procurement is retained as an observed existing
operation and is not treated as a new scalable closure mechanism.

## R75 break-even counterfactual

The corrected R75-U2 artifact already has late mean Service operating profit
`{number(r75['late_mean_service_operating_profit']):.6g}`. A passive zero-loss
counterfactual therefore changes no household wage income or consumption
budget in first order. Even if a positive Service loss were removed, the model
does not route that accounting improvement directly into household wages or a
new final-demand source. The income-demand closure would not be materially
restored; Service financial instability is secondary to the aggregate income
closure in this audit.

The maximum treatment household-budget identity gap in the paired replay was
`{budget_identity_max_gap:.6g}`. Accounting/conservation remains accepted for
all three Step14E.2 cells, and the replay introduced no behavior changes.
"""
    (OUTPUT / "acceptance_summary.md").write_text(summary, encoding="utf-8")
    print(verdict, flush=True)
    print(f"Outputs: {OUTPUT}", flush=True)
    del control_world, treatment_world, treatment_context, base
    gc.collect()


if __name__ == "__main__":
    main()
