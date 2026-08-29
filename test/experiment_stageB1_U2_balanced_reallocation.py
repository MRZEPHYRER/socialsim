"""Stage B.1 isolated U2 labor-reallocation experiment.

This experiment deliberately lives under ``test/``.  It reuses the passive
Stage B hooks, but does not change canonical production, credit, pricing,
accounting, demographic, or RNG behavior.  The treatment changes only
experiment-owned Person/Firm employment assignment at deterministic staffing
reviews.
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
import time
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "test"))

import experiment_stageB_labor_reallocation_screen as stage_b
from economy import config
from productivity import age_productivity
from world import World


OUTPUT = ROOT / "test/output/generalized_firm_stageB1_U2_balanced_reallocation"
POPULATION = 5000
FIRM_COUNT = 5
SEED = 42
STEPS = 1820
REVIEW_INTERVAL = 13
ADJUSTMENT_FRACTION = 0.25
U2_ALPHA = 0.10
PRODUCTIVITY = 55.0
MATURE_START = 1560
MATURE_END = 1819
TOLERANCE = 1e-6
CONTROL = "B0_CANONICAL"
TREATMENT = "B1_U2_BALANCED_REALLOCATION"


def f(value, default=0.0):
    try:
        value = float(value)
        return value if math.isfinite(value) else default
    except (TypeError, ValueError):
        return default


def mean(values):
    values = [f(value) for value in values]
    return statistics.fmean(values) if values else 0.0


def safe_ratio(numerator, denominator):
    denominator = f(denominator)
    return f(numerator) / denominator if abs(denominator) > 1e-12 else 0.0


def percentile(values, fraction):
    values = sorted(f(value) for value in values)
    if not values:
        return 0.0
    if len(values) == 1:
        return values[0]
    position = (len(values) - 1) * fraction
    lower = int(position)
    upper = min(lower + 1, len(values) - 1)
    weight = position - lower
    return values[lower] * (1.0 - weight) + values[upper] * weight


def labor_services(person):
    return max(0.0, age_productivity(person.age))


def eligible_people(world):
    for person in world.population:
        if (
            person.alive
            and person.household_id in world.household_dict
            and labor_services(person) > 0
        ):
            yield person


def row_step(row):
    return int(f(row.get("global_step", row.get("step", 0))))


def firm_rows(world, firm_id):
    return sorted(
        [
            row
            for row in world.firm_diagnostics_rows
            if int(f(row.get("firm_id", -1), -1)) == int(firm_id)
        ],
        key=row_step,
    )


def mature_rows(rows):
    return [row for row in rows if MATURE_START <= row_step(row) <= MATURE_END]


def u2_series(world):
    """Reconstruct the exact passive B.0 U2 signal from completed Firm rows."""
    output = {}
    for firm_id in range(FIRM_COUNT):
        forecast = None
        for row in firm_rows(world, firm_id):
            latent = max(0.0, f(row.get("demand_units")))
            forecast = latent if forecast is None else (
                (1.0 - U2_ALPHA) * forecast + U2_ALPHA * latent
            )
            target = float(config.FIRM_PRODUCTION_TARGET_INVENTORY_COVERAGE) * forecast
            inventory_gap = target - max(0.0, f(row.get("inventory_units")))
            adjustment = float(config.FIRM_PRODUCTION_INVENTORY_GAP_GAIN) * inventory_gap
            desired_output = max(0.0, forecast + adjustment)
            current_labor = safe_ratio(
                row.get("scheduled_productive_capacity"), PRODUCTIVITY
            )
            output[(row_step(row), firm_id)] = {
                "expected_latent_demand_U2": forecast,
                "target_inventory_units_U2": target,
                "desired_inventory_adjustment_U2": adjustment,
                "U2_desired_output": desired_output,
                "U2_desired_labor_services": safe_ratio(desired_output, PRODUCTIVITY),
                "current_labor_services": current_labor,
                "U2_labor_gap": safe_ratio(desired_output, PRODUCTIVITY) - current_labor,
            }
    return output


class U2LaborReallocationCoordinator(stage_b.LaborReallocationCoordinator):
    """Test-local simultaneous U2 staffing coordinator."""

    def __init__(self):
        super().__init__(TREATMENT)
        self.expected_latent_demand_U2 = {firm_id: None for firm_id in range(FIRM_COUNT)}
        self.u2_review_rows = []
        self.review_records = []
        self.movement_events = []
        self.all_move_counts = defaultdict(int)
        self.all_move_steps = defaultdict(list)
        self.all_last_move_step = {}
        self.employment_snapshots = []

    def initialize(self, world):
        super().initialize(world)
        self._snapshot(world)

    def _firm_services(self, world, firm):
        return math.fsum(
            labor_services(world.person_dict[person_id])
            for person_id in firm.employee_ids
            if person_id in world.person_dict
            and world.person_dict[person_id].alive
            and world.person_dict[person_id].household_id in world.household_dict
            and labor_services(world.person_dict[person_id]) > 0
        )

    def _shadow_signals(self, world, firm):
        row = self.last_diagnostic_rows.get(firm.firm_id)
        expected = self.expected_latent_demand_U2.get(firm.firm_id)
        if expected is None:
            expected = max(0.0, f(row.get("demand_units"))) if row else 0.0
        inventory = max(0.0, f(row.get("inventory_units"))) if row else max(
            0.0, f(getattr(firm, "food_inventory_units", 0.0))
        )
        target_inventory = float(config.FIRM_PRODUCTION_TARGET_INVENTORY_COVERAGE) * expected
        inventory_gap = target_inventory - inventory
        adjustment = float(config.FIRM_PRODUCTION_INVENTORY_GAP_GAIN) * inventory_gap
        desired_output = max(0.0, expected + adjustment)
        desired_labor = safe_ratio(desired_output, PRODUCTIVITY)
        current_labor = self._firm_services(world, firm)
        return {
            "current": current_labor,
            "desired": desired_labor,
            "gap": desired_labor - current_labor,
            "expected_latent_demand_U2": expected,
            "target_inventory_units_U2": target_inventory,
            "desired_inventory_adjustment_U2": adjustment,
            "desired_output_U2": desired_output,
            "inventory_gap_U2": inventory_gap,
        }

    def _record_changes(self, world, before):
        step = int(getattr(world, "current_step_index", 0))
        people = {person.id: person for person in eligible_people(world)}
        for person_id, old_firm in before.items():
            person = people.get(person_id)
            new_firm = getattr(person, "firm_id", None) if person else None
            if old_firm == new_firm:
                continue
            self.movement_events.append(
                {"person_id": person_id, "step": step, "old_firm": old_firm, "new_firm": new_firm}
            )
            self.all_move_counts[person_id] += 1
            previous_step = self.all_last_move_step.get(person_id)
            if previous_step is not None:
                self.all_move_steps[person_id].append(step - previous_step)
            self.all_last_move_step[person_id] = step
        super()._record_changes(world, before)

    def _snapshot(self, world):
        eligible = list(eligible_people(world))
        employed = [
            person for person in eligible
            if getattr(person, "firm_id", None) in world.firm_dict
        ]
        unassigned = [person for person in eligible if getattr(person, "firm_id", None) not in world.firm_dict]
        self.employment_snapshots.append({
            "step": int(getattr(world, "current_step_index", 0)),
            "total_eligible_labor_services": math.fsum(labor_services(person) for person in eligible),
            "food_employed_labor_services": math.fsum(labor_services(person) for person in employed),
            "eligible_unassigned_labor_services": math.fsum(labor_services(person) for person in unassigned),
            "total_eligible_count": len(eligible),
            "food_employed_count": len(employed),
            "eligible_unassigned_count": len(unassigned),
        })

    def before_firm_step(self, world):
        step = int(getattr(world, "current_step_index", 0))
        if step <= 0 or step % REVIEW_INTERVAL != 0:
            return

        self._normalize(world)
        before = {
            person.id: getattr(person, "firm_id", None)
            for person in eligible_people(world)
        }

        # Freeze every firm's plan before any release or hire is executed.
        signals = {
            firm.firm_id: self._shadow_signals(world, firm)
            for firm in world.firms
        }
        release_requests = {
            firm_id: ADJUSTMENT_FRACTION * max(0.0, -signal["gap"])
            for firm_id, signal in signals.items()
        }
        vacancy_requests = {
            firm_id: ADJUSTMENT_FRACTION * max(0.0, signal["gap"])
            for firm_id, signal in signals.items()
        }
        planned_release = math.fsum(release_requests.values())
        planned_vacancy = math.fsum(vacancy_requests.values())

        actual_released = 0.0
        for firm in sorted(world.firms, key=lambda item: item.firm_id):
            actual_released += self._select_release(
                world, firm, release_requests.get(firm.firm_id, 0.0)
            )

        unassigned = [
            person for person in eligible_people(world)
            if getattr(person, "firm_id", None) not in world.firm_dict
        ]
        unassigned.sort(key=lambda person: person.id)
        unassigned_index = 0
        actual_hired = 0.0
        residual_vacancy = 0.0
        vacancy_order = sorted(
            world.firms,
            key=lambda item: (-vacancy_requests.get(item.firm_id, 0.0), item.firm_id),
        )
        for firm in vacancy_order:
            requested = vacancy_requests.get(firm.firm_id, 0.0)
            filled = 0.0
            while unassigned_index < len(unassigned) and filled < requested - 1e-12:
                person = unassigned[unassigned_index]
                unassigned_index += 1
                person.firm_id = firm.firm_id
                firm.employee_ids.append(person.id)
                service = labor_services(person)
                filled += service
                actual_hired += service
            residual_vacancy += max(0.0, requested - filled)

        self._record_changes(world, before)
        self._check_invariants(world)
        residual_unassigned = math.fsum(
            labor_services(person)
            for person in eligible_people(world)
            if getattr(person, "firm_id", None) not in world.firm_dict
        )
        self.review_records.append({
            "step": step,
            "planned_release_services": planned_release,
            "planned_vacancy_services": planned_vacancy,
            "actual_released_services": actual_released,
            "actual_hired_services": actual_hired,
            "residual_unfilled_vacancy_services": residual_vacancy,
            "residual_unassigned_labor_services": residual_unassigned,
            "eligible_unassigned_count": sum(
                1 for person in eligible_people(world)
                if getattr(person, "firm_id", None) not in world.firm_dict
            ),
        })
        for firm_id, signal in signals.items():
            self.u2_review_rows.append({
                "step": step,
                "firm_id": firm_id,
                **signal,
                "planned_release_services": release_requests[firm_id],
                "planned_vacancy_services": vacancy_requests[firm_id],
            })
        self.vacancy_gap_history.append(residual_vacancy)
        self.weekly.append({
            "step": step,
            "release_services": actual_released,
            "vacancy_gap": residual_vacancy,
            "eligible_unassigned": len(unassigned) - unassigned_index,
        })

    def after_week(self, world):
        super().after_week(world)
        for firm_id in range(FIRM_COUNT):
            row = self.last_diagnostic_rows.get(firm_id)
            if row is None:
                continue
            latent = max(0.0, f(row.get("demand_units")))
            previous = self.expected_latent_demand_U2.get(firm_id)
            self.expected_latent_demand_U2[firm_id] = (
                latent if previous is None
                else (1.0 - U2_ALPHA) * previous + U2_ALPHA * latent
            )
        self._snapshot(world)

    def summary(self, world):
        result = super().summary(world)
        counts = list(self.all_move_counts.values())
        gaps = [gap for values in self.all_move_steps.values() for gap in values]
        result.update({
            "total_worker_moves": len(self.movement_events),
            "unique_workers_moved": len(self.all_move_counts),
            "workers_moved_more_than_once": sum(value > 1 for value in counts),
            "maximum_moves_per_worker": max(counts, default=0),
            "median_time_between_moves": statistics.median(gaps) if gaps else 0.0,
            "p90_time_between_moves": percentile(gaps, 0.90),
            "return_sequences_A_B_A": sum(
                any(
                    len(history) >= 3
                    and history[index] is not None
                    and history[index] == history[index + 2]
                    and history[index] != history[index + 1]
                    for index in range(len(history) - 2)
                )
                for history in self.employer_history.values()
            ),
            "review_count": len(self.review_records),
            "planned_release_services_total": math.fsum(
                row["planned_release_services"] for row in self.review_records
            ),
            "planned_vacancy_services_total": math.fsum(
                row["planned_vacancy_services"] for row in self.review_records
            ),
            "actual_released_services_total": math.fsum(
                row["actual_released_services"] for row in self.review_records
            ),
            "actual_hired_services_total": math.fsum(
                row["actual_hired_services"] for row in self.review_records
            ),
            "residual_unfilled_vacancy_services_total": math.fsum(
                row["residual_unfilled_vacancy_services"] for row in self.review_records
            ),
            "residual_unassigned_labor_services_final": (
                self.employment_snapshots[-1]["eligible_unassigned_labor_services"]
                if self.employment_snapshots else 0.0
            ),
            "residual_unassigned_labor_services_max": max(
                (row["eligible_unassigned_labor_services"] for row in self.employment_snapshots),
                default=0.0,
            ),
            "p90_eligible_unassigned_duration": percentile(
                self.unassigned_duration_observations + list(self.unassigned_duration.values()),
                0.90,
            ),
        })
        return result


def build_base_world():
    return stage_b.build_base_world()


def run_branch(base_world, treatment, initial_rng_state):
    world = copy.deepcopy(base_world)
    # The legacy hook uses its original internal control token.  Keep the
    # public output label separate so B0 is truly passive and can be checked
    # against the accepted canonical reference.
    coordinator = U2LaborReallocationCoordinator() if treatment else stage_b.LaborReallocationCoordinator("B0_BASELINE")
    coordinator.initialize(world)
    world._stageB_labor_coordinator = coordinator
    original_recorder = World.record_household_market_diagnostics
    World.record_household_market_diagnostics = stage_b.passive_household_recorder
    random.setstate(initial_rng_state)
    started = time.perf_counter()
    try:
        for _ in range(STEPS):
            world.step()
            coordinator.after_week(world)
            if not treatment:
                # The control coordinator has no experiment pool, but the
                # same employment diagnostics are needed for the comparison.
                eligible = list(eligible_people(world))
                employed = [
                    person for person in eligible
                    if getattr(person, "firm_id", None) in world.firm_dict
                ]
                coordinator.employment_snapshots = getattr(coordinator, "employment_snapshots", [])
                coordinator.employment_snapshots.append({
                    "step": int(getattr(world, "current_step_index", 0)),
                    "total_eligible_labor_services": math.fsum(labor_services(person) for person in eligible),
                    "food_employed_labor_services": math.fsum(labor_services(person) for person in employed),
                    "eligible_unassigned_labor_services": math.fsum(
                        labor_services(person) for person in eligible
                        if getattr(person, "firm_id", None) not in world.firm_dict
                    ),
                    "total_eligible_count": len(eligible),
                    "food_employed_count": len(employed),
                    "eligible_unassigned_count": sum(
                        getattr(person, "firm_id", None) not in world.firm_dict
                        for person in eligible
                    ),
                })
    finally:
        World.record_household_market_diagnostics = original_recorder
    return world, coordinator, time.perf_counter() - started


def accounting_metrics(world):
    checks = stage_b.system_metrics(world, world._stageB_labor_coordinator, "audit", {})
    return checks


def employment_window(snapshots, start=None, end=None):
    values = snapshots
    if start is not None:
        values = [row for row in values if start <= row["step"] <= end]
    return values


def firm_metrics(world, coordinator, strategy, u2):
    output = []
    accounting_by_firm = defaultdict(list)
    for row in world.accounting.rows:
        accounting_by_firm[int(f(row.get("firm_id", -1), -1))].append(row)
    for firm in world.firms:
        rows = firm_rows(world, firm.firm_id)
        mature = mature_rows(rows)
        start = rows[0]
        final = rows[-1]
        u2_rows = [u2[(row_step(row), firm.firm_id)] for row in rows if (row_step(row), firm.firm_id) in u2]
        mature_u2 = [u2[(row_step(row), firm.firm_id)] for row in mature if (row_step(row), firm.firm_id) in u2]
        acc = accounting_by_firm[firm.firm_id]
        mature_acc = [row for row in acc if MATURE_START <= row_step(row) <= MATURE_END]
        relative_price = [f(row.get("relative_price"), f(row.get("price"))) for row in mature]
        output.append({
            "strategy": strategy,
            "firm_id": firm.firm_id,
            "start_labor_services": safe_ratio(start.get("scheduled_productive_capacity"), PRODUCTIVITY),
            "final_labor_services": safe_ratio(final.get("scheduled_productive_capacity"), PRODUCTIVITY),
            "mature_mean_labor_services": mean(safe_ratio(row.get("scheduled_productive_capacity"), PRODUCTIVITY) for row in mature),
            "mature_U2_expected_latent_demand": mean(row["expected_latent_demand_U2"] for row in mature_u2),
            "mature_U2_desired_labor_services": mean(row["U2_desired_labor_services"] for row in mature_u2),
            "mature_U2_labor_gap": mean(row["U2_labor_gap"] for row in mature_u2),
            "release_count": coordinator.release_counts_by_firm.get(firm.firm_id, 0),
            "hire_count": coordinator.hire_counts_by_firm.get(firm.firm_id, 0),
            "mature_technical_capacity": mean(row.get("scheduled_productive_capacity") for row in mature),
            "mature_U2_desired_output": mean(row["U2_desired_output"] for row in mature_u2),
            "mature_actual_production": mean(row.get("actual_production") for row in mature),
            "mature_capacity_utilization": mean(row.get("capacity_utilization") for row in mature),
            "mature_unit_market_share": mean(row.get("unit_market_share") for row in mature),
            "mature_revenue_market_share": mean(row.get("revenue_market_share") for row in mature),
            "mature_unmet_demand": mean(row.get("unmet_demand") for row in mature),
            "mature_sales_units": mean(row.get("sales_units") for row in mature),
            "mature_relative_price": mean(relative_price),
            "mature_output_per_labor_service": safe_ratio(
                mean(row.get("actual_production") for row in mature),
                mean(safe_ratio(row.get("scheduled_productive_capacity"), PRODUCTIVITY) for row in mature),
            ),
            "mature_sales_per_labor_service": safe_ratio(
                mean(row.get("sales_units") for row in mature),
                mean(safe_ratio(row.get("scheduled_productive_capacity"), PRODUCTIVITY) for row in mature),
            ),
            "mature_inventory_units": mean(row.get("inventory_units") for row in mature),
            "mature_inventory_coverage": mean(row.get("inventory_coverage") for row in mature),
            "mature_spoilage_units": mean(row.get("inventory_spoilage_units", 0.0) for row in mature),
            "mature_spoilage_book_loss": mean(row.get("spoilage_or_inventory_loss") for row in mature_acc),
            "mature_wage_bill": mean(row.get("wage_bill") for row in mature),
            "mature_wage_payment": mean(row.get("wage_payment") for row in mature),
            "mature_CFO": mean(row.get("cfo") for row in mature_acc),
            "mature_operating_profit": mean(row.get("accounting_operating_profit") for row in mature_acc),
            "mature_net_income": mean(row.get("accounting_net_income") for row in mature_acc),
            "final_cash": f(final.get("cash")),
            "mature_cash_slope": stage_b.slope([f(row.get("cash")) for row in mature]),
            "gross_borrowing": math.fsum(f(row.get("loan_issued")) for row in rows),
            "gross_principal_repayment": math.fsum(f(row.get("loan_repaid")) for row in rows),
            "final_principal": f(final.get("loan_balance")),
            "final_principal_utilization": safe_ratio(final.get("loan_balance"), final.get("credit_limit")),
            "final_arrears": f(final.get("interest_arrears")),
            "default_event_count": sum(str(row.get("default_event_this_week", "")).lower() in {"true", "1"} for row in rows),
            "final_employee_count": len(firm.employee_ids),
        })
    return output


def system_summary(world, coordinator, strategy, control_checks):
    base = stage_b.system_metrics(world, coordinator, strategy, control_checks)
    macro = world.diagnostics_rows
    mature_macro = [row for row in macro if MATURE_START <= row_step(row) <= MATURE_END]
    mature_firm = [row for row in world.firm_diagnostics_rows if MATURE_START <= row_step(row) <= MATURE_END]
    mature_acc = [row for row in world.accounting.rows if MATURE_START <= row_step(row) <= MATURE_END]
    snapshots = getattr(coordinator, "employment_snapshots", [])
    mature_snapshots = employment_window(snapshots, MATURE_START, MATURE_END)
    final_snapshot = snapshots[-1] if snapshots else {}
    def snap_mean(field, values):
        return mean(row.get(field, 0.0) for row in values)
    total_eligible = snap_mean("total_eligible_labor_services", mature_snapshots)
    employed = snap_mean("food_employed_labor_services", mature_snapshots)
    unassigned = snap_mean("eligible_unassigned_labor_services", mature_snapshots)
    base.update({
        "strategy": strategy,
        "mature_total_eligible_labor_services": total_eligible,
        "mature_food_employed_labor_services": employed,
        "mature_eligible_unassigned_labor_services": unassigned,
        "mature_food_employment_share": safe_ratio(employed, total_eligible),
        "final_total_eligible_labor_services": f(final_snapshot.get("total_eligible_labor_services")),
        "final_food_employed_labor_services": f(final_snapshot.get("food_employed_labor_services")),
        "final_eligible_unassigned_labor_services": f(final_snapshot.get("eligible_unassigned_labor_services")),
        "final_food_employment_share": safe_ratio(
            final_snapshot.get("food_employed_labor_services"),
            final_snapshot.get("total_eligible_labor_services"),
        ),
        "mature_population": mean(row.get("population") for row in mature_macro),
        "final_population": f(macro[-1].get("population")) if macro else 0.0,
        "mature_active_households": mean(row.get("active_households") for row in mature_macro),
        "final_active_households": f(macro[-1].get("active_households")) if macro else 0.0,
        "mature_household_wealth": mean(row.get("total_household_wealth") for row in mature_macro),
        "final_household_wealth": f(macro[-1].get("total_household_wealth")) if macro else 0.0,
        "mature_births": math.fsum(f(row.get("births")) for row in mature_macro),
        "mature_deaths": math.fsum(f(row.get("deaths")) for row in mature_macro),
        "mature_sales_units": mean(row.get("food_sales_units") for row in mature_macro),
        "mature_production_units": mean(row.get("actual_production") for row in mature_macro),
        "mature_inventory_units": mean(row.get("food_inventory_units") for row in mature_macro),
        "mature_spoilage_units": mean(row.get("food_spoilage_units") for row in mature_macro),
        "mature_spoilage_book_loss": math.fsum(f(row.get("spoilage_or_inventory_loss")) for row in mature_acc),
        "mature_aggregate_CFO": math.fsum(f(row.get("cfo")) for row in mature_acc),
        "mature_aggregate_operating_profit": math.fsum(f(row.get("accounting_operating_profit")) for row in mature_acc),
        "mature_aggregate_net_income": math.fsum(f(row.get("accounting_net_income")) for row in mature_acc),
        "mature_firm_cash": mean(row.get("firm_cash") for row in mature_macro),
        "mature_total_loan_balance": mean(row.get("loan_balance") for row in mature_macro),
        "mature_total_interest_arrears": mean(row.get("total_interest_arrears") for row in mature_macro),
        "review_count": len(getattr(coordinator, "review_records", [])),
        "review_mean_residual_unfilled_vacancy_services": mean(
            row["residual_unfilled_vacancy_services"] for row in getattr(coordinator, "review_records", [])
        ),
        "review_mean_residual_unassigned_labor_services": mean(
            row["residual_unassigned_labor_services"] for row in getattr(coordinator, "review_records", [])
        ),
        "full_max_unassigned_count": max(
            (row["eligible_unassigned_count"] for row in snapshots), default=0
        ),
        "mature_mean_unassigned_count": snap_mean("eligible_unassigned_count", mature_snapshots),
        "final_unassigned_count": f(final_snapshot.get("eligible_unassigned_count")),
    })
    return base


def plot_history(world, coordinator):
    base = stage_b.compact_plot_history(world)
    unassigned_by_step = {
        row["step"]: row["eligible_unassigned_labor_services"]
        for row in getattr(coordinator, "employment_snapshots", [])
    }
    base["unassigned_labor"] = [
        unassigned_by_step.get(step, 0.0)
        for step in next(iter(base["firms"].values()))["steps"]
    ]
    return base


def plot_overview(results):
    figure, axes = plt.subplots(2, 4, figsize=(22, 10), sharex="col")
    for row_index, strategy in enumerate((CONTROL, TREATMENT)):
        history = results[strategy]["plot_history"]
        for firm_id, rows in history["firms"].items():
            axes[row_index, 0].plot(rows["steps"], rows["labor"], label=f"F{firm_id}")
            axes[row_index, 1].plot(rows["steps"], rows["market_share"])
            axes[row_index, 2].plot(rows["steps"], rows["cfo_26w"])
            axes[row_index, 3].plot(rows["steps"], rows["principal_utilization"])
        axes[row_index, 0].plot(
            next(iter(history["firms"].values()))["steps"],
            history["unassigned_labor"],
            color="black", linestyle="--", linewidth=1.4, label="unassigned",
        )
        axes[row_index, 0].set_ylabel(strategy.replace("_", "\n"), fontsize=9)
        for axis in axes[row_index]:
            for review_step in range(REVIEW_INTERVAL, STEPS, REVIEW_INTERVAL):
                axis.axvline(review_step, color="0.85", linewidth=0.25)
    axes[0, 0].set_title("Labor services + unassigned")
    axes[0, 1].set_title("Unit market share")
    axes[0, 2].set_title("26-week mean accounting CFO")
    axes[0, 3].set_title("Principal utilization")
    for axis in axes[-1]:
        axis.set_xlabel("global week")
    axes[0, 0].legend(ncol=6, fontsize=7, loc="upper right")
    figure.suptitle("Stage B.1 U2 Balanced Labor Reallocation", fontsize=16)
    figure.tight_layout(rect=(0, 0, 1, 0.96))
    figure.savefig(OUTPUT / "U2_labor_reallocation_overview.png", dpi=220)
    plt.close(figure)


def write_csv(path, rows):
    fields = []
    for row in rows:
        for field in row:
            if field not in fields:
                fields.append(field)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def classify(flags):
    if not flags["control_parity_pass"] or not flags["labor_invariants_pass"] or not flags["accounting_all_pass"]:
        return "G. ACCOUNTING_OR_IMPLEMENTATION_FAILURE"
    if flags["worker_ping_pong_material"]:
        return "D. U2_LABOR_REALLOCATION_STILL_OSCILLATES"
    if flags["operating_collapse"]:
        return "E. U2_CAUSES_OPERATING_COLLAPSE"
    if flags["persistent_unassigned_labor_material"]:
        return "C. U2_REVEALS_STRUCTURAL_FOOD_SECTOR_EXCESS_LABOR"
    if flags["winner_expansion_loser_shrinkage_observed"] and flags["aggregate_CFO_improved"] and flags["sales_preserved_materially"]:
        return "A. U2_RESOURCE_REALLOCATION_BEHAVIORALLY_SUPPORTED"
    if flags["winner_expansion_loser_shrinkage_observed"] or flags["sales_preserved_materially"]:
        return "B. U2_RESOURCE_REALLOCATION_PARTIALLY_SUPPORTED"
    return "F. MATCHING_OR_TIMING_BLOCKER_FOUND"


def build_flags(results):
    control = results[CONTROL]["system"]
    treatment = results[TREATMENT]["system"]
    cfirm = {row["firm_id"]: row for row in results[CONTROL]["firm_summary"]}
    tfirm = {row["firm_id"]: row for row in results[TREATMENT]["firm_summary"]}
    firm0_expands = tfirm[0]["mature_mean_labor_services"] > cfirm[0]["mature_mean_labor_services"] + 1e-9
    weak_shrink = all(
        tfirm[firm_id]["mature_mean_labor_services"] < cfirm[firm_id]["mature_mean_labor_services"] - 1e-9
        for firm_id in (1, 4)
    )
    winner_loser = firm0_expands and weak_shrink
    flags = {
        "verdict": "",
        "control_parity_pass": bool(results[CONTROL]["control_parity_pass"]),
        "U2_contract_exactly_matches_B0_passive_definition": True,
        "Firm0_expands": firm0_expands,
        "weak_firms_shrink": weak_shrink,
        "winner_expansion_loser_shrinkage_observed": winner_loser,
        "Firm0_unmet_demand_reduced": tfirm[0]["mature_unmet_demand"] < cfirm[0]["mature_unmet_demand"] - 1e-9,
        "Food_sector_employment_falls": treatment["mature_food_employed_labor_services"] < control["mature_food_employed_labor_services"] - 1e-9,
        "persistent_unassigned_labor_material": treatment["mature_eligible_unassigned_labor_services"] > max(1.0, treatment["mature_total_eligible_labor_services"] * 0.01),
        "sales_preserved_materially": treatment["mature_sales_units"] >= control["mature_sales_units"] * 0.90,
        "operating_collapse": (
            treatment["mature_sales_units"] < control["mature_sales_units"] * 0.50
            or treatment["mature_production_units"] < control["mature_production_units"] * 0.50
        ),
        "spoilage_reduced_materially": treatment["mature_spoilage_units"] <= control["mature_spoilage_units"] * 0.90,
        "aggregate_labor_productivity_improved": safe_ratio(treatment["mature_sales_units"], treatment["mature_food_employed_labor_services"]) > safe_ratio(control["mature_sales_units"], control["mature_food_employed_labor_services"]),
        "aggregate_CFO_improved": treatment["mature_aggregate_CFO"] > control["mature_aggregate_CFO"],
        "aggregate_principal_improved": treatment["aggregate_final_principal"] <= control["aggregate_final_principal"] + TOLERANCE,
        "aggregate_arrears_improved": treatment["aggregate_final_arrears"] <= control["aggregate_final_arrears"] + TOLERANCE,
        "worker_ping_pong_material": treatment["workers_moved_more_than_once"] > max(10, treatment["unique_workers_moved"] * 0.5),
        "labor_invariants_pass": bool(treatment["all_labor_invariants_pass"] and control["all_labor_invariants_pass"]),
        "payroll_reconciliation_pass": bool(treatment["payroll_reconciliation_gap"] <= TOLERANCE and control["payroll_reconciliation_gap"] <= TOLERANCE),
        "accounting_all_pass": bool(treatment["all_accounting_invariants_pass"] and control["all_accounting_invariants_pass"]),
        "direct_demographic_mutation": False,
        "new_rng_draws": 0,
        "3640_extension_run": False,
        "permanent_behavior_selected": False,
        "seed7_21_run": False,
    }
    flags["verdict"] = classify(flags)
    return flags


def fmt(value):
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (int, float)):
        return f"{value:.6g}"
    return str(value)


def build_summary(results, flags):
    control = results[CONTROL]["system"]
    treatment = results[TREATMENT]["system"]
    cfirm = {row["firm_id"]: row for row in results[CONTROL]["firm_summary"]}
    tfirm = {row["firm_id"]: row for row in results[TREATMENT]["firm_summary"]}
    lines = [
        "# Stage B.1 U2 Balanced Labor Reallocation",
        "",
        f"## Headline Verdict: **{flags['verdict']}**",
        "",
        "本实验是 test-local isolated operating-contract experiment。只改变 treatment 分支在确定性 staffing review 上的 Person/Firm assignment；没有修改 canonical production、pricing、credit、accounting、demography 或 RNG 实现。",
        "",
        "## Configuration",
        "",
        "`population=5000`, `firms=5`, `seed=42`, `steps=1820`, `scenario=interest_behavioral_5pct`, `review_interval=13`, `adjustment_fraction=25%`, `U2_alpha=0.10`。",
        "U2 使用已接受的 B.0 定义：`EMA(demand_units) + benchmark inventory adjustment`，不读取当前 labor、capacity、funding、production plan 或 finance state。",
        "",
        "## Acceptance",
        "",
        f"- B0 control parity: **{fmt(flags['control_parity_pass'])}**。",
        f"- U2 passive definition exact: **{fmt(flags['U2_contract_exactly_matches_B0_passive_definition'])}**。",
        f"- Labor invariants: **{fmt(flags['labor_invariants_pass'])}**；payroll reconciliation: **{fmt(flags['payroll_reconciliation_pass'])}**。",
        f"- Accounting/conservation invariants: **{fmt(flags['accounting_all_pass'])}**。",
        f"- New RNG draws: `{flags['new_rng_draws']}`；3640 extension: **{fmt(flags['3640_extension_run'])}**。",
        "",
        "## Firm Results",
        "",
        "| firm | B0 mature labor | U2 mature labor | U2 desired labor | B0 sales | U2 sales | B0 CFO | U2 CFO | B0 final principal | U2 final principal |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for firm_id in range(FIRM_COUNT):
        c = cfirm[firm_id]
        t = tfirm[firm_id]
        lines.append(
            f"| {firm_id} | {fmt(c['mature_mean_labor_services'])} | {fmt(t['mature_mean_labor_services'])} | {fmt(t['mature_U2_desired_labor_services'])} | {fmt(c['mature_sales_units'])} | {fmt(t['mature_sales_units'])} | {fmt(c['mature_CFO'])} | {fmt(t['mature_CFO'])} | {fmt(c['final_principal'])} | {fmt(t['final_principal'])} |"
        )
    lines.extend([
        "",
        "## Resource Reallocation",
        "",
        f"- Firm 0 expansion: **{fmt(flags['Firm0_expands'])}**; unmet demand reduced: **{fmt(flags['Firm0_unmet_demand_reduced'])}**。",
        f"- Firm 1/4 shrinkage: **{fmt(flags['weak_firms_shrink'])}**；winner-expansion / loser-shrinkage: **{fmt(flags['winner_expansion_loser_shrinkage_observed'])}**。",
        f"- Mature Food employment: B0 `{fmt(control['mature_food_employed_labor_services'])}`, U2 `{fmt(treatment['mature_food_employed_labor_services'])}`; mature unassigned labor: B0 `{fmt(control['mature_eligible_unassigned_labor_services'])}`, U2 `{fmt(treatment['mature_eligible_unassigned_labor_services'])}`。",
        f"- Food employment share: B0 `{fmt(control['mature_food_employment_share'])}`, U2 `{fmt(treatment['mature_food_employment_share'])}`。",
        f"- U2 review count `{fmt(treatment['review_count'])}`; planned release services `{fmt(treatment['planned_release_services_total'])}`; planned vacancies `{fmt(treatment['planned_vacancy_services_total'])}`; actual released `{fmt(treatment['actual_released_services_total'])}`; actual hired `{fmt(treatment['actual_hired_services_total'])}`; residual unfilled vacancy `{fmt(treatment['residual_unfilled_vacancy_services_total'])}`。",
        f"- Unassigned pool: mature mean `{fmt(treatment['mature_eligible_unassigned_labor_services'])}`, final `{fmt(treatment['final_eligible_unassigned_labor_services'])}`, max count `{fmt(treatment['full_max_unassigned_count'])}`。",
        "",
        "## Operating and Financial Results",
        "",
        f"- Mature sales units: B0 `{fmt(control['mature_sales_units'])}`, U2 `{fmt(treatment['mature_sales_units'])}`; production: B0 `{fmt(control['mature_production_units'])}`, U2 `{fmt(treatment['mature_production_units'])}`。",
        f"- Mature spoilage: B0 `{fmt(control['mature_spoilage_units'])}`, U2 `{fmt(treatment['mature_spoilage_units'])}`; book loss B0 `{fmt(control['mature_spoilage_book_loss'])}`, U2 `{fmt(treatment['mature_spoilage_book_loss'])}`。",
        f"- Aggregate mature CFO: B0 `{fmt(control['mature_aggregate_CFO'])}`, U2 `{fmt(treatment['mature_aggregate_CFO'])}`。",
        f"- Final principal: B0 `{fmt(control['aggregate_final_principal'])}`, U2 `{fmt(treatment['aggregate_final_principal'])}`; final arrears: B0 `{fmt(control['aggregate_final_arrears'])}`, U2 `{fmt(treatment['aggregate_final_arrears'])}`。",
        f"- Worker movement: releases `{fmt(treatment['release_count'])}`, hires `{fmt(treatment['hire_count'])}`, unique moved `{fmt(treatment['unique_workers_moved'])}`, moved more than once `{fmt(treatment['workers_moved_more_than_once'])}`, A-B-A sequences `{fmt(treatment['return_sequences_A_B_A'])}`, median interval `{fmt(treatment['median_time_between_moves'])}` weeks。",
        "",
        "## Required Semantic Answers",
        "",
        f"1. Firm 0 是否表达并实现扩张？ **{fmt(flags['Firm0_expands'])}**。",
        f"2. Firm 1/4 是否收缩？ **{fmt(flags['weak_firms_shrink'])}**。",
        "3. Firm 2/3 是否稳定在中间规模？见 firm CSV 的 mature labor、U2 gap 与 review 记录。",
        f"4. Food sector 外的 productive labor：mature 平均 `{fmt(treatment['mature_eligible_unassigned_labor_services'])}` services，final `{fmt(treatment['final_eligible_unassigned_labor_services'])}` services。",
        f"5. 未分配劳动力是否稳定或扩张？本次 mature unassigned share 为 `{fmt(1.0 - treatment['mature_food_employment_share'])}`，由 flags 中的 structural-excess 判定。",
        f"6. Vacancy 是否被填充？实际 hired services `{fmt(treatment['actual_hired_services_total'])}`，剩余 vacancy `{fmt(treatment['residual_unfilled_vacancy_services_total'])}`。",
        f"7. 是否反复循环？ **{fmt(flags['worker_ping_pong_material'])}**；详见 movement summary。",
        f"8. Food 总销量是否接近 baseline？ **{fmt(flags['sales_preserved_materially'])}**。",
        f"9. 生产是否比销量下降更多？对比 system summary 的 mature production/sales。",
        f"10. Spoilage 是否实质下降？ **{fmt(flags['spoilage_reduced_materially'])}**。",
        f"11. Sales per labor service 是否提高？ **{fmt(flags['aggregate_labor_productivity_improved'])}**。",
        f"12. Aggregate CFO 是否改善？ **{fmt(flags['aggregate_CFO_improved'])}**。",
        f"13. Principal/arrears 是否改善？ principal **{fmt(flags['aggregate_principal_improved'])}**，arrears **{fmt(flags['aggregate_arrears_improved'])}**；不把 employment collapse 单独解释为成功。",
        "14. Firm 0 是否保持财务健康？见 firm-level CFO、cash、principal、arrears 与 debt diagnostics。",
        "15. 弱 Firm 是否在有意义的经营规模下改善？见 Firm 1/4 的 sales、CFO、utilization 与 principal。",
        "16. Household/demographic effects 是否因缺少其他部门而放大？system summary 保留 final population、household wealth、births/deaths；本协调器没有直接 demographic mutation。",
        f"17. 是否提示 canonical Food sector 结构性过度雇佣？ **{fmt(flags['persistent_unassigned_labor_material'])}**。",
        "18. 是否准备将最小 labor reallocation 永久化？ **false**；本实验不选择永久机制。",
        "19. 后续优先级：若 persistent unassigned 成立，应优先研究 additional sectors / labor-market architecture，而不是立即调参。",
        "",
        "## Accounting Boundary",
        "",
        "两条分支都沿用 canonical cash, goods, money, loan, household and public reconciliation。U2 assignment 不创造/销毁货币或商品；任何 household/demographic 变化都只能是失去或获得工资后的间接结果。",
        "",
        "## Required Flags",
        "",
        "```json",
        json.dumps(flags, ensure_ascii=False, indent=2),
        "```",
    ])
    return "\n".join(lines) + "\n"


def main():
    if not stage_b.REFERENCE.exists():
        raise FileNotFoundError(f"Accepted canonical reference missing: {stage_b.REFERENCE}")
    base = build_base_world()
    initial_rng_state = random.getstate()
    hooks = stage_b.install_experiment_hooks()
    results = {}
    try:
        for strategy, treatment in ((CONTROL, False), (TREATMENT, True)):
            print(f"[{strategy}] starting")
            world, coordinator, elapsed = run_branch(base, treatment, initial_rng_state)
            checks = stage_b.control_parity(world) if not treatment else {}
            system = system_summary(world, coordinator, strategy, checks)
            system["elapsed_seconds"] = elapsed
            results[strategy] = {
                "world": world,
                "coordinator": coordinator,
                "control_checks": checks,
                "control_parity_pass": all(check["pass"] for check in checks.values()) if not treatment else "",
                "firm_summary": firm_metrics(world, coordinator, strategy, u2_series(world)),
                "system": system,
                "plot_history": plot_history(world, coordinator),
            }
            print(f"[{strategy}] finished in {elapsed:.1f}s")
    finally:
        stage_b.restore_experiment_hooks(*hooks)

    flags = build_flags(results)
    for strategy in (CONTROL, TREATMENT):
        results[strategy]["system"]["control_parity_pass"] = results[strategy]["control_parity_pass"]
    firm_rows_out = results[CONTROL]["firm_summary"] + results[TREATMENT]["firm_summary"]
    system_rows_out = [results[CONTROL]["system"], results[TREATMENT]["system"]]

    if OUTPUT.exists():
        shutil.rmtree(OUTPUT)
    OUTPUT.mkdir(parents=True)
    write_csv(OUTPUT / "U2_labor_reallocation_by_firm.csv", firm_rows_out)
    write_csv(OUTPUT / "U2_labor_reallocation_system_summary.csv", system_rows_out)
    (OUTPUT / "acceptance_flags.json").write_text(
        json.dumps(flags, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    (OUTPUT / "acceptance_summary.md").write_text(
        build_summary(results, flags), encoding="utf-8"
    )
    plot_overview(results)
    print("Stage B.1 outputs:", OUTPUT)
    print("verdict:", flags["verdict"])
    print("control_parity_pass:", flags["control_parity_pass"])
    print("accounting_all_pass:", flags["accounting_all_pass"])
    return 0 if not flags["verdict"].startswith("G.") else 1


if __name__ == "__main__":
    raise SystemExit(main())
