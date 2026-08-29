"""Step 14E.2 corrected Food + Generic Services short retest.

The implementation is experiment-local.  It removes the two confounds found
in Step 14E.1 without changing production code, credit, wages, staffing
cadence, demand families, or RNG use.
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
import time
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "test"))

import step14e_first_two_sector_behavior_screen as step14e

from economy.household_demand import HouseholdDemandSystem


OUTPUT = ROOT / "test/output/step14E2_corrected_two_sector_retest"
STEP14E_OUTPUT = ROOT / "test/output/step14E_first_two_sector_behavior_screen"
STEP14E1_OUTPUT = ROOT / "test/output/step14E1_startup_scale_audit"
CELLS = (("R25", "U2"), ("R50", "U2"), ("R75", "U2"))
INTENDED_RATIOS = {
    "U0": 1.00000,
    "U1": 1.04464756,
    "U2": 1.16042102,
}
WAGE_PRESERVATION_FLOOR = 0.80
CONSUMPTION_PRESERVATION_FLOOR = 0.80
FOOD_ACTIVITY_FLOOR = 0.25
BREAK_EVEN_APPROACH_FRACTION = 0.90
TOLERANCE = 1e-9


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


def late(rows, count=104):
    return rows[-min(count, len(rows)):]


def old_cell_map():
    rows = read_csv(STEP14E_OUTPUT / "step14E_short_screen_matrix.csv")
    return {
        (row["demand_reference"], row["unit_economics"]): row
        for row in rows
    }


def scale_wage_map():
    rows = read_csv(STEP14E1_OUTPUT / "step14E1_transition_metrics.csv")
    return {
        (row["demand_reference"], row["unit_economics"]): number(
            row["scheduled_common_wage_cost_per_labor_N500"]
        )
        for row in rows
        if row.get("record_type") == "unit_scale_cell"
    }


class CorrectedTwoSectorCoordinator(step14e.TwoSectorCoordinator):
    """Startup-synchronized and scale-normalized experiment adapter."""

    def __init__(self, demand_id, unit_id, population, scale_wage):
        super().__init__(demand_id, unit_id, population)
        self.intended_unit_ratio = INTENDED_RATIOS[unit_id]
        self.scale_wage_cost_per_labor = float(scale_wage)
        self.service_productivity = (
            self.scale_wage_cost_per_labor * self.intended_unit_ratio
        )
        self.current_common_wage = self.scale_wage_cost_per_labor
        self.service_behavior_active = False
        self.service_activation_week = None
        self.shadow_service_allocations = {}
        self.last_shadow_service_budget = 0.0
        self.last_startup_retained_food_budget = 0.0

    def prepare_credit(self, world, food_wage_bill, food_labor):
        super().prepare_credit(world, food_wage_bill, food_labor)
        if self.service_behavior_active:
            return
        employed_labor = self.service_labor(world)
        funded_capacity = math.fsum(
            max(0.0, number(firm.funded_productive_capacity))
            for firm in self.service_firms
        )
        if employed_labor > TOLERANCE and funded_capacity > TOLERANCE:
            self.service_behavior_active = True
            self.service_activation_week = int(world.current_step_index)

    def allocate_household_budget(self, firm_system, household, units, money):
        if self.service_behavior_active:
            return super().allocate_household_budget(
                firm_system, household, units, money
            )

        # The shadow allocation is read-only and feeds only the staffing
        # forecast.  The actual purchase remains the unmodified Food control
        # transaction until Service labor and funded capacity both exist.
        allocation = HouseholdDemandSystem(firm_system.world).allocate(
            household,
            total_consumption_budget=money,
            service_share=self.service_share,
        )
        self.shadow_service_allocations[household.id] = allocation.service_budget
        return units, money

    def settle_service_market(self, world):
        if self.service_behavior_active:
            self.last_shadow_service_budget = 0.0
            self.last_startup_retained_food_budget = 0.0
            self.shadow_service_allocations = {}
            return super().settle_service_market(world)

        shadow_total = math.fsum(self.shadow_service_allocations.values())
        prior_forecasts = [
            getattr(firm, "service_demand_forecast", None)
            for firm in self.service_firms
        ]
        self.service_allocations = {}
        sales = super().settle_service_market(world)
        observed = shadow_total / step14e.SERVICE_FIRM_COUNT
        for firm, prior in zip(self.service_firms, prior_forecasts):
            firm.service_demand_forecast = (
                observed
                if prior is None
                else (1.0 - step14e.DEMAND_ALPHA) * prior
                + step14e.DEMAND_ALPHA * observed
            )
            firm.latent_allocated_demand = observed
            firm.unmet_demand = observed
        self.last_shadow_service_budget = shadow_total
        self.last_startup_retained_food_budget = shadow_total
        self.last_service_market["shadow_desired_budget"] = shadow_total
        self.last_service_market["startup_retained_food_budget"] = shadow_total
        self.shadow_service_allocations = {}
        return sales

    def after_week(self, world):
        super().after_week(world)
        row = self.history[-1]
        macro = world.diagnostics_rows[-1]
        row.update({
            "food_demand_units": number(macro.get("food_demand_units")),
            "service_shadow_budget": self.last_shadow_service_budget,
            "startup_retained_food_budget": self.last_startup_retained_food_budget,
            "service_behavior_active": self.service_behavior_active,
            "service_activation_week": (
                self.service_activation_week
                if self.service_activation_week is not None else ""
            ),
            "service_funded_capacity": math.fsum(
                max(0.0, number(firm.funded_productive_capacity))
                for firm in self.service_firms
            ),
        })


def summarize_corrected(world, context, elapsed, control_history, old):
    summary = step14e.summarize_run(
        world, context, elapsed, control_history
    )
    mature = late(context.history)
    mature_firms = late(
        context.firm_history,
        step14e.SERVICE_FIRM_COUNT * min(104, len(context.history)),
    )
    control_mature = late(control_history)
    pre_activation = [
        row for row in context.history
        if context.service_activation_week is None
        or number(row["step"]) < context.service_activation_week
    ]
    pre_activation_actual_allocation = math.fsum(
        row["service_desired_budget"] for row in pre_activation
    )
    pre_activation_executed_spending = math.fsum(
        row["service_sales"] for row in pre_activation
    )
    pre_activation_retained_food_budget = math.fsum(
        row["startup_retained_food_budget"] for row in pre_activation
    )
    intended_break_even = 1.0 / context.intended_unit_ratio
    configured_ratio = safe_ratio(
        context.service_productivity,
        context.scale_wage_cost_per_labor,
    )
    wage_ratio = safe_ratio(
        mean(row["total_wage_income"] for row in mature),
        mean(row["total_wage_income"] for row in control_mature),
    )
    consumption_ratio = safe_ratio(
        mean(row["total_consumption"] for row in mature),
        mean(row["total_consumption"] for row in control_mature),
    )
    food_labor_ratio = safe_ratio(
        mean(row["food_labor"] for row in mature),
        mean(row["food_labor"] for row in control_mature),
    )
    food_sales_ratio = safe_ratio(
        mean(row["food_sales_units"] for row in mature),
        mean(row["food_sales_units"] for row in control_mature),
    )
    service_employment_positive = (
        mean(row["service_labor"] for row in mature) > TOLERANCE
    )
    wage_preserved = wage_ratio >= WAGE_PRESERVATION_FLOOR
    consumption_preserved = (
        consumption_ratio >= CONSUMPTION_PRESERVATION_FLOOR
    )
    food_preserved = (
        summary["food_activity_survives"]
        and food_labor_ratio >= FOOD_ACTIVITY_FLOOR
        and food_sales_ratio >= FOOD_ACTIVITY_FLOOR
    )
    service_utilization = mean(row["utilization"] for row in mature_firms)
    utilization_approaches_break_even = (
        service_utilization
        >= BREAK_EVEN_APPROACH_FRACTION * intended_break_even
    )
    legacy_credit_instability = summary["service_credit_instability"]
    debt_growth_limit = max(
        1.0,
        0.10 * summary["late_mean_service_sales"],
    )
    service_debt_explosive = (
        summary["late_service_debt_slope"] > debt_growth_limit
    )
    service_credit_instability = (
        legacy_credit_instability or service_debt_explosive
    )
    structural_pass = all((
        context.service_activation_week is not None,
        service_employment_positive,
        food_preserved,
        wage_preserved,
        consumption_preserved,
        utilization_approaches_break_even,
        not service_credit_instability,
        not summary["worker_ping_pong_material"],
        summary["accounting_all_pass"],
    ))
    corrected_classification = (
        "A. CORRECTED_STRUCTURAL_PASS"
        if structural_pass else summary["classification"]
    )
    row = {
        "cell": f"{context.demand_id}-{context.unit_id}",
        "demand_reference": context.demand_id,
        "unit_economics": context.unit_id,
        "old_step14E_classification": old["classification"],
        "corrected_classification": corrected_classification,
        "structural_pass": structural_pass,
        "service_activation_week": (
            context.service_activation_week
            if context.service_activation_week is not None else ""
        ),
        "startup_synchronized": context.service_activation_week is not None,
        "pre_activation_actual_service_allocation": pre_activation_actual_allocation,
        "pre_activation_executed_service_spending": pre_activation_executed_spending,
        "pre_activation_food_budget_retained": pre_activation_retained_food_budget,
        "N500_scale_wage_cost_per_labor": context.scale_wage_cost_per_labor,
        "normalized_service_price": step14e.NORMALIZED_SERVICE_PRICE,
        "nominal_service_revenue_capacity_per_labor": context.service_productivity,
        "intended_unit_economics_ratio": context.intended_unit_ratio,
        "configured_unit_economics_ratio": configured_ratio,
        "unit_economics_ratio_gap": configured_ratio - context.intended_unit_ratio,
        "intended_break_even_utilization": intended_break_even,
        "expected_break_even_utilization_U2": 1.0 / INTENDED_RATIOS["U2"],
        "late_runtime_break_even_utilization": summary["service_break_even_utilization"],
        "late_mean_service_utilization": service_utilization,
        "utilization_approaches_intended_break_even": utilization_approaches_break_even,
        "late_mean_food_employment": summary["late_mean_food_labor"],
        "old_late_mean_food_employment": number(old["late_mean_food_labor"]),
        "late_mean_service_employment": summary["late_mean_service_labor"],
        "old_late_mean_service_employment": number(old["late_mean_service_labor"]),
        "late_mean_unassigned_labor": summary["late_mean_eligible_unassigned_labor"],
        "old_late_mean_unassigned_labor": number(old["late_mean_eligible_unassigned_labor"]),
        "food_to_service_moves": summary["food_to_service_moves"],
        "old_food_to_service_moves": number(old["food_to_service_moves"]),
        "cross_sector_moves": summary["cross_sector_moves"],
        "workers_moved_more_than_once": summary["workers_moved_more_than_once"],
        "A_B_A_sequences": summary["A_B_A_sequences"],
        "worker_ping_pong_material": summary["worker_ping_pong_material"],
        "old_worker_ping_pong_material": old["worker_ping_pong_material"],
        "late_mean_total_wage_income": summary["late_mean_total_wage_income"],
        "old_late_mean_total_wage_income": number(old["late_mean_total_wage_income"]),
        "late_wage_income_to_control": wage_ratio,
        "wage_income_preserved": wage_preserved,
        "late_mean_household_consumption": summary["late_mean_household_consumption"],
        "old_late_mean_household_consumption": number(old["late_mean_household_consumption"]),
        "late_household_consumption_to_control": consumption_ratio,
        "household_consumption_preserved": consumption_preserved,
        "late_mean_food_demand_units": mean(row["food_demand_units"] for row in mature),
        "late_mean_food_sales_units": summary["late_mean_food_sales_units"],
        "old_late_mean_food_sales_units": number(old["late_mean_food_sales_units"]),
        "late_food_labor_to_control": food_labor_ratio,
        "late_food_sales_to_control": food_sales_ratio,
        "food_activity_preserved": food_preserved,
        "startup_shadow_service_budget_total": math.fsum(
            row["service_shadow_budget"] for row in context.history
        ),
        "startup_food_discretionary_budget_retained_total": math.fsum(
            row["startup_retained_food_budget"] for row in context.history
        ),
        "late_mean_service_allocated_expenditure": summary["late_mean_service_desired_budget"],
        "old_late_mean_service_allocated_expenditure": number(old["late_mean_service_desired_budget"]),
        "late_mean_service_executed_expenditure": summary["late_mean_service_sales"],
        "old_late_mean_service_executed_expenditure": number(old["late_mean_service_sales"]),
        "late_service_execution_ratio": summary["late_mean_service_execution_ratio"],
        "late_mean_service_operating_profit": summary["late_mean_service_operating_profit"],
        "old_late_mean_service_operating_profit": number(old["late_mean_service_operating_profit"]),
        "final_service_principal": summary["final_service_debt"],
        "old_final_service_principal": number(old["final_service_debt"]),
        "final_service_interest_arrears": summary["final_service_arrears"],
        "old_final_service_interest_arrears": number(old["final_service_arrears"]),
        "late_service_debt_slope": summary["late_service_debt_slope"],
        "old_late_service_debt_slope": number(old["late_service_debt_slope"]),
        "service_debt_growth_limit": debt_growth_limit,
        "service_debt_explosive": service_debt_explosive,
        "late_mean_service_loan_issued": mean(row["service_loan_issued"] for row in mature),
        "late_mean_service_principal_repaid": mean(row["service_principal_repaid"] for row in mature),
        "cumulative_service_loan_issued": math.fsum(row["service_loan_issued"] for row in context.history),
        "cumulative_service_principal_repaid": math.fsum(row["service_principal_repaid"] for row in context.history),
        "legacy_service_credit_instability": legacy_credit_instability,
        "service_credit_instability": service_credit_instability,
        "old_service_credit_instability": old["service_credit_instability"],
        "firm_cash_flow_gap": summary["firm_cash_flow_gap"],
        "inventory_bridge_gap": summary["inventory_bridge_gap"],
        "equity_bridge_gap": summary["equity_bridge_gap"],
        "household_wealth_bridge_gap": summary["household_wealth_bridge_gap"],
        "public_cash_flow_gap": summary["public_cash_flow_gap"],
        "credit_money_stock_flow_gap": summary["credit_money_stock_flow_gap"],
        "money_location_gap": summary["money_location_gap"],
        "loan_reconciliation_gap": summary["loan_reconciliation_gap"],
        "monetary_accounting_gap": summary["monetary_accounting_gap"],
        "goods_conservation_gap": summary["goods_conservation_gap"],
        "service_settlement_gap": summary["service_settlement_gap"],
        "service_no_inventory_gap": summary["service_no_inventory_gap"],
        "service_cash_bridge_gap": summary["service_cash_bridge_gap"],
        "max_accounting_or_conservation_gap": summary["max_accounting_or_conservation_gap"],
        "invariant_violation_count": summary["invariant_violation_count"],
        "accounting_all_pass": summary["accounting_all_pass"],
        "elapsed_seconds": elapsed,
    }
    return row


def choose_verdict(rows):
    passed = sum(bool(row["structural_pass"]) for row in rows)
    if passed == len(rows):
        return "A. CORRECTED_TWO_SECTOR_MECHANISM_SUPPORTED"
    if passed > 0:
        return "B. CORRECTED_TWO_SECTOR_MECHANISM_PARTIALLY_SUPPORTED"
    if all(not bool(row["wage_income_preserved"]) for row in rows):
        return "E. INCOME_COLLAPSE_PERSISTS_AFTER_CONFOUND_REMOVAL"
    if any(bool(row["service_credit_instability"]) for row in rows):
        return "D. SERVICE_FINANCIAL_INSTABILITY_PERSISTS"
    if all(
        number(row["late_mean_service_employment"]) <= TOLERANCE
        or not bool(row["utilization_approaches_intended_break_even"])
        for row in rows
    ):
        return "C. SERVICE_DEMAND_STILL_TOO_WEAK"
    return "F. OTHER_BLOCKER_FOUND"


def build_summary(verdict, rows, nominated):
    lines = []
    for row in rows:
        activation = row["service_activation_week"]
        activation = activation if activation != "" else "none"
        lines.append(
            f"| {row['cell']} | {row['old_step14E_classification']} | "
            f"{row['corrected_classification']} | {activation} | "
            f"{row['late_mean_food_employment']:.3f} | "
            f"{row['late_mean_service_employment']:.3f} | "
            f"{row['late_wage_income_to_control']:.3f} | "
            f"{row['late_household_consumption_to_control']:.3f} | "
            f"{row['late_mean_service_utilization']:.3f} | "
            f"{row['intended_break_even_utilization']:.3f} | "
            f"{row['late_service_debt_slope']:.3f} | "
            f"{row['service_debt_explosive']} | "
            f"{row['structural_pass']} |"
        )
    nomination_text = ", ".join(nominated) if nominated else "none"
    passed = sum(bool(row["structural_pass"]) for row in rows)
    max_gap = max(number(row["max_accounting_or_conservation_gap"]) for row in rows)
    return f"""# Step 14E.2 Acceptance Summary

## Verdict

**{verdict}**

The corrected retest ran exactly three N=500, seed-42, 520-week cells. No
N=5000/1820 confirmation, 9-cell matrix, parameter tuning, or Step14F work was
performed.

## Corrected contracts

- Before activation, the Household executes the unchanged Food control
  purchase. A read-only shadow Service allocation informs staffing forecasts
  without removing Food expenditure or creating Service sales.
- Treatment allocation activates permanently in the first week with both
  positive Service employed labor and positive funded capacity.
- Each cell uses its audited N=500 scheduled wage scale. Revenue capacity is
  `scale wage x 1.16042102`, preserving U2 exactly; normalized Service price
  remains 1.0 and intended break-even utilization is
  `{1.0 / INTENDED_RATIOS['U2']:.9f}`.

## Results

| cell | old result | corrected result | activation | Food labor | Service labor | wage/control | consumption/control | Service util. | intended BE | debt slope | debt explosive | pass |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---|---|
{chr(10).join(lines)}

Structural passes: **{passed}/3**. Later full-scale nominations are limited to
at most two corrected structural passes: **{nomination_text}**. They were not
run in this stage.

## Interpretation rules

The predeclared structural gate requires positive Service employment,
material Food operation, at least 80% of control wage income and Household
consumption, Service utilization at least 90% of intended break-even, no
explosive Service principal growth, no material worker ping-pong, and closed
accounting. Debt growth is marked explosive when its late weekly slope exceeds
`max(1, 10% of late Service sales)`, independently of the sign of current
operating profit. Residual unemployment is allowed.

All cells reproduce the intended unit-economics ratio within floating
tolerance. Maximum accounting/conservation gap was `{max_gap:.6g}`. Existing
staffing cadence, adjustment fraction, capitalization, target-cash adapter,
credit, wages, demand families, and RNG streams were retained.
"""


def main():
    shutil.rmtree(OUTPUT, ignore_errors=True)
    OUTPUT.mkdir(parents=True, exist_ok=True)
    old_map = old_cell_map()
    wages = scale_wage_map()
    if any(cell not in old_map or cell not in wages for cell in CELLS):
        raise RuntimeError("Missing accepted Step14E/Step14E.1 source cell")

    started = time.perf_counter()
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

    rows = []
    with step14e.experiment_hooks():
        random.setstate(python_state)
        if np is not None:
            np.random.set_state(numpy_state)
        control_world = copy.deepcopy(base)
        control_history, _ = step14e.run_world(
            control_world,
            step14e.SHORT_STEPS,
            progress_label="CONTROL",
        )

        for demand_id, unit_id in CELLS:
            random.setstate(python_state)
            if np is not None:
                np.random.set_state(numpy_state)
            world = copy.deepcopy(base)
            context = CorrectedTwoSectorCoordinator(
                demand_id,
                unit_id,
                step14e.SHORT_POPULATION,
                wages[(demand_id, unit_id)],
            )
            context.initialize(world)
            _, elapsed = step14e.run_world(
                world,
                step14e.SHORT_STEPS,
                context,
                f"{demand_id}-{unit_id}",
            )
            row = summarize_corrected(
                world,
                context,
                elapsed,
                control_history,
                old_map[(demand_id, unit_id)],
            )
            rows.append(row)
            print(
                f"[{row['cell']}] activation={row['service_activation_week']} "
                f"result={row['corrected_classification']}",
                flush=True,
            )
            context._world = None
            del world, context
            gc.collect()

    ratio_preserved = all(
        abs(number(row["unit_economics_ratio_gap"])) <= TOLERANCE
        for row in rows
    )
    startup_removed = all(
        row["service_activation_week"] != ""
        and abs(number(row["pre_activation_actual_service_allocation"]))
        <= TOLERANCE
        and abs(number(row["pre_activation_executed_service_spending"]))
        <= TOLERANCE
        and number(row["pre_activation_food_budget_retained"])
        > TOLERANCE
        for row in rows
    )
    scale_removed = ratio_preserved and all(
        abs(
            number(row["intended_break_even_utilization"])
            - 1.0 / INTENDED_RATIOS["U2"]
        ) <= TOLERANCE
        for row in rows
    )
    verdict = choose_verdict(rows)
    passed_rows = [row for row in rows if row["structural_pass"]]
    nominated = [
        row["cell"]
        for row in sorted(
            passed_rows,
            key=lambda item: (
                -number(item["late_wage_income_to_control"]),
                -number(item["late_mean_service_utilization"]),
            ),
        )[:2]
    ]
    for row in rows:
        row["nominated_for_later_N5000_confirmation"] = (
            row["cell"] in nominated
        )

    flags = {
        "startup_confound_removed": startup_removed,
        "scale_confound_removed": scale_removed,
        "intended_unit_economics_preserved": ratio_preserved,
        "cells_run": 3,
        "cells_structurally_passed": len(passed_rows),
        "Service_employment_positive_any": any(
            number(row["late_mean_service_employment"]) > TOLERANCE
            for row in rows
        ),
        "Food_to_Service_moves_positive_any": any(
            number(row["food_to_service_moves"]) > 0 for row in rows
        ),
        "wage_income_preserved_any": any(
            bool(row["wage_income_preserved"]) for row in rows
        ),
        "Household_consumption_preserved_any": any(
            bool(row["household_consumption_preserved"]) for row in rows
        ),
        "Food_activity_preserved_any": any(
            bool(row["food_activity_preserved"]) for row in rows
        ),
        "Service_credit_instability_any": any(
            bool(row["service_credit_instability"]) for row in rows
        ),
        "worker_ping_pong_material_any": any(
            bool(row["worker_ping_pong_material"]) for row in rows
        ),
        "accounting_all_pass": all(
            bool(row["accounting_all_pass"]) for row in rows
        ),
        "N5000_run": False,
        "Stage14F_started": False,
    }

    write_csv(OUTPUT / "step14E2_corrected_retest.csv", rows)
    (OUTPUT / "acceptance_flags.json").write_text(
        json.dumps(
            {"verdict": verdict, **flags},
            indent=2,
            ensure_ascii=False,
        ) + "\n",
        encoding="utf-8",
    )
    (OUTPUT / "acceptance_summary.md").write_text(
        build_summary(verdict, rows, nominated),
        encoding="utf-8",
    )
    print(verdict, flush=True)
    print(f"Elapsed: {time.perf_counter() - started:.1f}s", flush=True)
    print(f"Outputs: {OUTPUT}", flush=True)


if __name__ == "__main__":
    main()
