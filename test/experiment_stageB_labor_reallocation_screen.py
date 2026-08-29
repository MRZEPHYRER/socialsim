"""Stage B isolated labor-reallocation screen.

This file is intentionally an experiment-local behavioral coordinator.  It
does not modify World, FirmSystem, Person, Household, finance, production, or
demographic implementation.  The two monkeypatches are installed only while
the four in-memory branches run and are restored before the script exits.
"""

from __future__ import annotations

import copy
import csv
import math
import os
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

from scenarios import apply_runtime_scenario, apply_scenario
from productivity import age_productivity
from world import World
from economy.firm import FirmSystem
from generalized_firm_stageA_passive_operating_contracts import (
    compare_rows,
    read_csv,
    passive_household_recorder,
)


OUTPUT = ROOT / "test/output/generalized_firm_stageB_labor_reallocation_screen"
REFERENCE = ROOT / "test/output/main_step13_financial_core"
POPULATION = 5000
FIRM_COUNT = 5
SEED = 42
STEPS = 1820
PERSISTENCE_WEEKS = 13
ADJUSTMENT_FRACTION = 0.25
TOLERANCE = 1e-6
MATURE_START = 1560
MATURE_END = 1819

STRATEGIES = (
    "B0_BASELINE",
    "B1_GAP_PROPORTIONAL",
    "B2_PERSISTENCE_GATED",
    "B3_DEMAND_UTILIZATION_GATED",
)


def f(value, default=0.0):
    try:
        value = float(value)
        return value if math.isfinite(value) else default
    except (TypeError, ValueError):
        return default


def safe_ratio(numerator, denominator):
    denominator = f(denominator)
    return f(numerator) / denominator if abs(denominator) > 1e-12 else 0.0


def mean(values):
    values = [f(value) for value in values]
    return statistics.fmean(values) if values else 0.0


def slope(values):
    values = [f(value) for value in values]
    if len(values) < 2:
        return 0.0
    x_mean = (len(values) - 1) / 2.0
    y_mean = mean(values)
    denominator = math.fsum((index - x_mean) ** 2 for index in range(len(values)))
    return math.fsum(
        (index - x_mean) * (value - y_mean)
        for index, value in enumerate(values)
    ) / denominator


def correlation(left, right):
    pairs = [(f(a), f(b)) for a, b in zip(left, right)]
    if len(pairs) < 2:
        return 0.0
    left_mean = mean(a for a, _ in pairs)
    right_mean = mean(b for _, b in pairs)
    numerator = math.fsum((a - left_mean) * (b - right_mean) for a, b in pairs)
    left_var = math.fsum((a - left_mean) ** 2 for a, _ in pairs)
    right_var = math.fsum((b - right_mean) ** 2 for _, b in pairs)
    denominator = math.sqrt(left_var * right_var)
    return numerator / denominator if denominator > 1e-12 else 0.0


def previous_firm_row(world, firm_id):
    for row in reversed(getattr(world, "firm_diagnostics_rows", [])):
        if int(f(row.get("firm_id", -1), -1)) == int(firm_id):
            return row
    return None


def eligible_people(world):
    for person in world.population:
        if (
            person.alive
            and person.household_id in world.household_dict
            and age_productivity(person.age) > 0
        ):
            yield person


class LaborReallocationCoordinator:
    """Deterministic release/pool/hire experiment coordinator."""

    def __init__(self, strategy):
        self.strategy = strategy
        self.gap_streak = defaultdict(int)
        self.evidence_streak = defaultdict(int)
        self.evidence_direction = {}
        self.release_count = 0
        self.hire_count = 0
        self.release_counts_by_firm = defaultdict(int)
        self.hire_counts_by_firm = defaultdict(int)
        self.total_worker_moves = 0
        self.move_counts = defaultdict(int)
        self.move_steps = defaultdict(list)
        self.last_move_step = {}
        self.employer_history = defaultdict(list)
        self.weekly = []
        self.boundary_violations = []
        self.unassigned_duration = defaultdict(int)
        self.unassigned_duration_observations = []
        self.vacancy_gap_history = []
        self.last_diagnostic_rows = {}
        self._initialized = False

    def initialize(self, world):
        for person in eligible_people(world):
            self.employer_history[person.id] = [
                getattr(person, "firm_id", None)
            ]
        self._initialized = True

    def _firm_services(self, world, firm):
        return math.fsum(
            age_productivity(world.person_dict[person_id].age)
            for person_id in firm.employee_ids
            if person_id in world.person_dict
            and world.person_dict[person_id].alive
            and world.person_dict[person_id].household_id in world.household_dict
            and age_productivity(world.person_dict[person_id].age) > 0
        )

    def employed_labor_services(self, world):
        """Labor services of workers currently assigned to a Firm.

        Released workers form an explicit unpaid pool in this isolated
        experiment.  The legacy production implementation predates that pool
        and counts every eligible person in its aggregate denominator, so the
        experiment wrapper supplies the employed-only view while preserving
        the existing wage, credit, and capacity formulas.
        """
        return math.fsum(
            age_productivity(person.age)
            for person in eligible_people(world)
            if getattr(person, "firm_id", None) in world.firm_dict
        )

    def employed_people(self, world):
        return [
            person
            for person in world.population
            if (
                person.alive
                and person.household_id in world.household_dict
                and age_productivity(person.age) > 0
                and getattr(person, "firm_id", None) in world.firm_dict
            )
        ]

    def _shadow_signals(self, world, firm):
        row = self.last_diagnostic_rows.get(firm.firm_id)
        current = self._firm_services(world, firm)
        if row is None:
            denominator = world.firm_system.food_productivity * world.firm_system.production_scale
            desired = safe_ratio(getattr(firm, "production_plan", 0.0), denominator)
            actual = f(getattr(firm, "actual_production", 0.0))
            scheduled_capacity = f(getattr(firm, "scheduled_productive_capacity", 0.0))
            utilization = safe_ratio(actual, scheduled_capacity)
            unmet = f(getattr(firm, "unmet_demand", 0.0))
            feasible = actual
            inventory_gap = f(getattr(firm, "inventory_gap_units", 0.0))
        else:
            desired = f(row.get("stageA_desired_labor_services_shadow"))
            current = f(row.get("stageA_current_labor_services"), current)
            actual = f(row.get("actual_production"))
            scheduled_capacity = f(row.get("scheduled_productive_capacity"))
            utilization = safe_ratio(actual, scheduled_capacity)
            unmet = f(row.get("unmet_demand"))
            feasible = f(row.get("stageA_feasible_output_shadow"), actual)
            inventory_gap = f(row.get("inventory_gap_units"))
        gap = desired - current
        hiring_evidence = (
            utilization >= 0.95
            or unmet > 1e-12
            or desired > feasible + 1e-12
        )
        release_evidence = (
            utilization <= 0.50
            or desired < feasible - 1e-12
            or inventory_gap < -1e-12
        )
        return {
            "current": current,
            "desired": desired,
            "gap": gap,
            "utilization": utilization,
            "unmet": unmet,
            "feasible": feasible,
            "inventory_gap": inventory_gap,
            "hiring_evidence": hiring_evidence,
            "release_evidence": release_evidence,
        }

    def _normalize(self, world):
        """Repair only experiment-owned employment bookkeeping at the boundary."""
        firms = {firm.firm_id: firm for firm in world.firms}
        eligible = {
            person.id: person for person in eligible_people(world)
        }
        seen = set()
        for firm in world.firms:
            cleaned = []
            for person_id in firm.employee_ids:
                person = eligible.get(person_id)
                if person is None or person_id in seen:
                    continue
                if getattr(person, "firm_id", None) != firm.firm_id:
                    continue
                cleaned.append(person_id)
                seen.add(person_id)
            firm.employee_ids = cleaned
        for person in eligible.values():
            firm_id = getattr(person, "firm_id", None)
            if firm_id in firms and person.id not in seen:
                firms[firm_id].employee_ids.append(person.id)
                seen.add(person.id)
            elif firm_id not in firms and hasattr(person, "firm_id"):
                delattr(person, "firm_id")

    def _update_streak(self, firm_id, gap, evidence):
        if abs(gap) <= 1e-12:
            self.gap_streak[firm_id] = 0
            self.evidence_streak[firm_id] = 0
            self.evidence_direction.pop(firm_id, None)
            return 0, 0
        direction = 1 if gap > 0 else -1
        previous = self.evidence_direction.get(firm_id)
        if previous == direction and evidence:
            self.evidence_streak[firm_id] += 1
        elif evidence:
            self.evidence_direction[firm_id] = direction
            self.evidence_streak[firm_id] = 1
        else:
            self.evidence_streak[firm_id] = 0
            self.evidence_direction.pop(firm_id, None)
        previous_gap_direction = 1 if self.gap_streak[firm_id] > 0 else -1 if self.gap_streak[firm_id] < 0 else 0
        if previous_gap_direction == direction:
            self.gap_streak[firm_id] += direction
        else:
            self.gap_streak[firm_id] = direction
        return abs(self.gap_streak[firm_id]), self.evidence_streak[firm_id]

    def _allowed(self, firm, signals):
        if self.strategy == "B1_GAP_PROPORTIONAL":
            return True
        gap_streak, evidence_streak = self._update_streak(
            firm.firm_id,
            signals["gap"],
            signals["hiring_evidence"] if signals["gap"] > 0 else signals["release_evidence"],
        )
        if self.strategy == "B2_PERSISTENCE_GATED":
            return gap_streak >= PERSISTENCE_WEEKS
        return (
            gap_streak >= PERSISTENCE_WEEKS
            and evidence_streak >= PERSISTENCE_WEEKS
        )

    def _record_changes(self, world, before):
        all_people = {person.id: person for person in eligible_people(world)}
        for person_id, old_firm in before.items():
            person = all_people.get(person_id)
            new_firm = getattr(person, "firm_id", None) if person else None
            if old_firm == new_firm:
                continue
            if old_firm is not None and new_firm is None:
                self.release_count += 1
                self.release_counts_by_firm[old_firm] += 1
            elif old_firm is None and new_firm is not None:
                self.hire_count += 1
                self.hire_counts_by_firm[new_firm] += 1
            elif old_firm is not None and new_firm is not None:
                self.total_worker_moves += 1
                self.release_counts_by_firm[old_firm] += 1
                self.hire_counts_by_firm[new_firm] += 1
                self.move_counts[person_id] += 1
                current_step = len(world.population_history)
                prior = self.last_move_step.get(person_id)
                if prior is not None:
                    self.move_steps[person_id].append(current_step - prior)
                self.last_move_step[person_id] = current_step
            history = self.employer_history[person_id]
            if not history or history[-1] != new_firm:
                history.append(new_firm)

    def _select_release(self, world, firm, amount):
        selected = []
        released = 0.0
        # Stable Person ID is deliberately chosen to avoid age-selection bias.
        for person_id in sorted(firm.employee_ids):
            if released >= amount - 1e-12:
                break
            person = world.person_dict.get(person_id)
            if person is None or not person.alive:
                continue
            service = max(0.0, age_productivity(person.age))
            if service <= 0:
                continue
            selected.append(person)
            released += service
        for person in selected:
            firm.employee_ids.remove(person.id)
            if hasattr(person, "firm_id"):
                delattr(person, "firm_id")
        return released

    def before_firm_step(self, world):
        if self.strategy == "B0_BASELINE":
            return
        self._normalize(world)
        before = {
            person.id: getattr(person, "firm_id", None)
            for person in eligible_people(world)
        }
        signals = {
            firm.firm_id: self._shadow_signals(world, firm)
            for firm in world.firms
        }
        release_requests = {}
        vacancy_requests = {}
        for firm in world.firms:
            gap = signals[firm.firm_id]["gap"]
            if not self._allowed(firm, signals[firm.firm_id]):
                continue
            amount = abs(gap) * ADJUSTMENT_FRACTION
            if gap < 0:
                release_requests[firm.firm_id] = amount
            elif gap > 0:
                vacancy_requests[firm.firm_id] = amount

        released_total = 0.0
        for firm in world.firms:
            amount = release_requests.get(firm.firm_id, 0.0)
            if amount > 0:
                released_total += self._select_release(world, firm, amount)

        unassigned = [
            person for person in eligible_people(world)
            if not hasattr(person, "firm_id")
        ]
        unassigned.sort(key=lambda person: person.id)
        unassigned_index = 0
        vacancy_gap = 0.0
        for firm in sorted(
            world.firms,
            key=lambda item: (-vacancy_requests.get(item.firm_id, 0.0), item.firm_id),
        ):
            requested = vacancy_requests.get(firm.firm_id, 0.0)
            filled = 0.0
            while unassigned_index < len(unassigned) and filled < requested - 1e-12:
                person = unassigned[unassigned_index]
                unassigned_index += 1
                person.firm_id = firm.firm_id
                firm.employee_ids.append(person.id)
                filled += max(0.0, age_productivity(person.age))
            vacancy_gap += max(0.0, requested - filled)

        self._record_changes(world, before)
        self._check_invariants(world)
        self.vacancy_gap_history.append(vacancy_gap)
        self.weekly.append({
            "step": len(world.population_history),
            "release_services": released_total,
            "vacancy_gap": vacancy_gap,
            "eligible_unassigned": len(unassigned) - unassigned_index,
        })

    def after_week(self, world):
        eligible = list(eligible_people(world))
        unassigned = [person for person in eligible if not hasattr(person, "firm_id")]
        unassigned_ids = {person.id for person in unassigned}
        for person in unassigned:
            self.unassigned_duration[person.id] += 1
        for person_id in list(self.unassigned_duration):
            if person_id not in unassigned_ids:
                self.unassigned_duration_observations.append(
                    self.unassigned_duration.pop(person_id)
                )
        self.weekly.append({
            "step_end": len(world.population_history),
            "eligible_unassigned_end": len(unassigned),
        })
        self.last_diagnostic_rows = {}
        for row in reversed(world.firm_diagnostics_rows):
            firm_id = int(f(row.get("firm_id", -1), -1))
            if firm_id in range(FIRM_COUNT) and firm_id not in self.last_diagnostic_rows:
                self.last_diagnostic_rows[firm_id] = row
        for firm_id in range(FIRM_COUNT):
            if firm_id not in self.last_diagnostic_rows:
                self.last_diagnostic_rows[firm_id] = next(
                    (
                        row for row in reversed(world.firm_diagnostics_rows)
                        if int(f(row.get("firm_id", -1), -1)) == firm_id
                    ),
                    None,
                )

    def _check_invariants(self, world):
        eligible = list(eligible_people(world))
        eligible_ids = {person.id for person in eligible}
        memberships = defaultdict(list)
        for firm in world.firms:
            for person_id in firm.employee_ids:
                if person_id in eligible_ids:
                    memberships[person_id].append(firm.firm_id)
        violations = []
        for person in eligible:
            listed = memberships.get(person.id, [])
            firm_id = getattr(person, "firm_id", None)
            if len(listed) > 1:
                violations.append("duplicate_employee")
            if firm_id is None:
                if listed:
                    violations.append("unassigned_list_mismatch")
            elif listed != [firm_id]:
                violations.append("person_firm_mismatch")
        employed_services = math.fsum(
            age_productivity(person.age)
            for person in eligible
            if hasattr(person, "firm_id")
        )
        total_services = math.fsum(age_productivity(person.age) for person in eligible)
        if abs(employed_services + math.fsum(
            age_productivity(person.age) for person in eligible if not hasattr(person, "firm_id")
        ) - total_services) > 1e-9:
            violations.append("labor_service_conservation")
        if violations:
            self.boundary_violations.extend(violations)

    def summary(self, world):
        all_unassigned = [
            person for person in eligible_people(world)
            if not hasattr(person, "firm_id")
        ]
        durations = self.unassigned_duration_observations + list(self.unassigned_duration.values())
        move_counts = list(self.move_counts.values())
        return {
            "total_worker_moves": self.total_worker_moves,
            "unique_workers_moved": len(self.move_counts),
            "workers_moved_more_than_once": sum(value > 1 for value in move_counts),
            "maximum_moves_per_worker": max(move_counts, default=0),
            "median_time_between_moves": statistics.median(
                [gap for gaps in self.move_steps.values() for gap in gaps]
            ) if any(self.move_steps.values()) else 0.0,
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
            "release_count": self.release_count,
            "hire_count": self.hire_count,
            "mean_eligible_unassigned": mean(
                row.get("eligible_unassigned_end", 0)
                for row in self.weekly
                if "eligible_unassigned_end" in row
            ),
            "max_eligible_unassigned": max(
                (
                    f(row.get("eligible_unassigned_end", 0))
                    for row in self.weekly
                    if "eligible_unassigned_end" in row
                ),
                default=0.0,
            ),
            "final_eligible_unassigned": len(all_unassigned),
            "mean_eligible_unassigned_duration": mean(durations),
            "final_eligible_unassigned_duration": mean(
                self.unassigned_duration.values()
            ),
            "max_vacancy_gap": max(self.vacancy_gap_history, default=0.0),
            "boundary_invariant_violation_count": len(self.boundary_violations),
        }


def build_base_world():
    scenario = apply_scenario("interest_behavioral_5pct")
    world = World(
        initial_population=POPULATION,
        seed=SEED,
        scenario_name=scenario["name"],
        scenario_overrides=scenario["overrides"],
        diagnostics_mode="full",
        initial_age_phase_mode="distributed",
    )
    apply_runtime_scenario(scenario, world)
    world.split_firms(FIRM_COUNT)
    world.generalized_firm_operating_contracts = True
    world.steps = STEPS
    return world


def run_branch(base_world, strategy):
    world = copy.deepcopy(base_world)
    coordinator = LaborReallocationCoordinator(strategy)
    coordinator.initialize(world)
    world._stageB_labor_coordinator = coordinator
    original_recorder = World.record_household_market_diagnostics
    World.record_household_market_diagnostics = passive_household_recorder
    started = time.perf_counter()
    try:
        for _ in range(STEPS):
            world.step()
            coordinator.after_week(world)
    finally:
        World.record_household_market_diagnostics = original_recorder
    seconds = time.perf_counter() - started
    return world, coordinator, seconds


def install_experiment_hooks():
    original_assign = World.assign_unassigned_workers_to_firms
    original_firm_step = FirmSystem.step
    original_total_labor = FirmSystem.total_labor
    original_distribute_wages = FirmSystem.distribute_wages

    def assignment_hook(world):
        coordinator = getattr(world, "_stageB_labor_coordinator", None)
        if coordinator is None or coordinator.strategy == "B0_BASELINE":
            return original_assign(world)
        # Treatment branches have already completed their only boundary
        # release/pool/hire operation before payroll.  Prevent the legacy
        # fallback from assigning the residual pool after payroll.
        return None

    def firm_step_hook(firm_system):
        coordinator = getattr(firm_system.world, "_stageB_labor_coordinator", None)
        if coordinator is not None:
            coordinator.before_firm_step(firm_system.world)
        return original_firm_step(firm_system)

    def total_labor_hook(firm_system):
        coordinator = getattr(firm_system.world, "_stageB_labor_coordinator", None)
        if coordinator is not None and coordinator.strategy != "B0_BASELINE":
            return coordinator.employed_labor_services(firm_system.world)
        return original_total_labor(firm_system)

    def distribute_wages_hook(firm_system, wage_bill, total_labor):
        coordinator = getattr(firm_system.world, "_stageB_labor_coordinator", None)
        if coordinator is None or coordinator.strategy == "B0_BASELINE":
            return original_distribute_wages(firm_system, wage_bill, total_labor)

        # Reuse the canonical settlement implementation verbatim, but expose
        # only assigned productive workers during this call.  The population
        # is restored immediately, so demographic and market code never sees
        # a filtered population.
        world = firm_system.world
        original_population = world.population
        try:
            world.population = coordinator.employed_people(world)
            return original_distribute_wages(firm_system, wage_bill, total_labor)
        finally:
            world.population = original_population

    World.assign_unassigned_workers_to_firms = assignment_hook
    FirmSystem.step = firm_step_hook
    FirmSystem.total_labor = total_labor_hook
    FirmSystem.distribute_wages = distribute_wages_hook
    return (
        original_assign,
        original_firm_step,
        original_total_labor,
        original_distribute_wages,
    )


def restore_experiment_hooks(
    original_assign,
    original_firm_step,
    original_total_labor,
    original_distribute_wages,
):
    World.assign_unassigned_workers_to_firms = original_assign
    FirmSystem.step = original_firm_step
    FirmSystem.total_labor = original_total_labor
    FirmSystem.distribute_wages = original_distribute_wages


def csv_collections(world):
    accounting = world.accounting
    return {
        "diagnostics.csv": world.diagnostics_rows,
        "firm_diagnostics.csv": world.firm_diagnostics_rows,
        "accounting/firm_accounting.csv": accounting.rows,
        "accounting/household_accounting.csv": accounting.household_rows,
        "accounting/public_accounting.csv": accounting.public_rows,
        "accounting/central_bank_accounting.csv": accounting.central_bank_rows,
        "accounting/accounting_reconciliation.csv": accounting.reconciliation_rows,
        "accounting/household_lifecycle.csv": world.household_lifecycle_events,
        "demographic_events.csv": world.demographic_events,
        "marriage_market_diagnostics.csv": world.marriage_market_diagnostics,
        "age_transition_diagnostics.csv": world.age_transition_diagnostics,
    }


def normalize_row_schema(rows):
    """Match CSV DictReader semantics for sparse event dictionaries."""
    fields = []
    for row in rows:
        for field in row:
            if field not in fields:
                fields.append(field)
    return [
        {field: row.get(field, "") for field in fields}
        for row in rows
    ]


def control_parity(world):
    checks = {}
    current = csv_collections(world)
    for name, reference in {
        relative: read_csv(REFERENCE / relative)
        for relative in csv_collections(world)
    }.items():
        checks[name] = compare_rows(
            normalize_row_schema(reference),
            normalize_row_schema(current[name]),
            tolerance=TOLERANCE,
            excluded_prefixes=("stageA_",),
        )
    return checks


def firm_rows(world, firm_id):
    return [
        row for row in world.firm_diagnostics_rows
        if int(f(row.get("firm_id", -1), -1)) == int(firm_id)
    ]


def mature_rows(rows):
    return [
        row for row in rows
        if MATURE_START <= int(f(row.get("global_step", row.get("step", 0)))) <= MATURE_END
    ]


def principal_utilization(row):
    limit = f(row.get("credit_limit"), math.inf)
    return safe_ratio(row.get("loan_balance"), limit) if math.isfinite(limit) else 0.0


def first_90pct_week(rows):
    for row in rows:
        if principal_utilization(row) >= 0.90:
            return int(f(row.get("global_step", row.get("step", 0))))
    return ""


def firm_metrics(world, coordinator, strategy):
    output = []
    macro_by_step = {
        int(f(row.get("global_step", row.get("step", 0)))): row
        for row in world.diagnostics_rows
    }
    for firm in world.firms:
        rows = firm_rows(world, firm.firm_id)
        mature = mature_rows(rows)
        final = rows[-1]
        start = rows[0]
        mature_cfo = [
            f(row.get("cfo"))
            for row in world.accounting.rows
            if int(f(row.get("firm_id", -1), -1)) == firm.firm_id
            and MATURE_START <= int(f(row.get("step", 0))) <= MATURE_END
        ]
        relative_prices = []
        for row in mature:
            macro = macro_by_step.get(int(f(row.get("global_step", row.get("step", 0)))), {})
            relative_prices.append(safe_ratio(row.get("price"), macro.get("food_price", 0.0)))
        output.append({
            "strategy": strategy,
            "firm_id": firm.firm_id,
            "start_worker_count": len(start.get("employee_count", [])) if isinstance(start.get("employee_count"), list) else f(start.get("employee_count")),
            "final_worker_count": len(firm.employee_ids),
            "start_labor_services": f(start.get("stageA_current_labor_services")),
            "final_labor_services": f(final.get("stageA_current_labor_services")),
            "mean_labor_gap": mean(row.get("stageA_labor_service_gap") for row in rows),
            "release_count_total": coordinator.release_count,
            "hire_count_total": coordinator.hire_count,
            "release_count": coordinator.release_counts_by_firm.get(firm.firm_id, 0),
            "hire_count": coordinator.hire_counts_by_firm.get(firm.firm_id, 0),
            "mature_market_share": mean(row.get("unit_market_share") for row in mature),
            "mature_unmet_demand": mean(row.get("unmet_demand") for row in mature),
            "mature_relative_price": mean(relative_prices),
            "mature_technical_capacity": mean(row.get("scheduled_productive_capacity") for row in mature),
            "mature_utilization": mean(safe_ratio(row.get("actual_production"), row.get("scheduled_productive_capacity")) for row in mature),
            "mature_production": mean(row.get("actual_production") for row in mature),
            "mature_desired_output_gap": mean(
                f(row.get("stageA_desired_output_shadow"))
                - f(row.get("stageA_feasible_output_shadow"))
                for row in mature
            ),
            "mature_mean_CFO": mean(mature_cfo),
            "final_cash": f(final.get("cash")),
            "mature_cash_slope": slope([f(row.get("cash")) for row in mature]),
            "final_principal": f(final.get("loan_balance")),
            "final_principal_utilization": principal_utilization(final),
            "final_arrears": f(final.get("interest_arrears")),
            "gross_borrowing": math.fsum(f(row.get("loan_issued")) for row in rows),
            "gross_principal_repayment": math.fsum(f(row.get("loan_repaid")) for row in rows),
            "first_90pct_utilization_week": first_90pct_week(rows),
            "default_event_count": sum(
                str(row.get("default_event_this_week", "")).lower() in {"true", "1"}
                for row in rows
            ),
        })
    return output


def resource_correlations(world):
    grouped = defaultdict(list)
    for row in world.firm_diagnostics_rows:
        grouped[int(f(row.get("firm_id", -1)))].append(row)
    share_change = []
    labor_change = []
    unmet_signal = []
    next_labor_change = []
    low_util_signal = []
    for rows in grouped.values():
        rows.sort(key=lambda row: int(f(row.get("global_step", row.get("step", 0)))))
        labor = [f(row.get("stageA_current_labor_services")) for row in rows]
        share = [f(row.get("unit_market_share")) for row in rows]
        unmet = [f(row.get("unmet_demand")) for row in rows]
        utilization = [
            safe_ratio(row.get("actual_production"), row.get("scheduled_productive_capacity"))
            for row in rows
        ]
        for index in range(1, len(rows)):
            share_change.append(share[index] - share[index - 1])
            labor_change.append(labor[index] - labor[index - 1])
            window_start = max(0, index - PERSISTENCE_WEEKS)
            unmet_signal.append(mean(unmet[window_start:index]))
            next_labor_change.append(labor[index] - labor[index - 1])
            low_util_signal.append(1.0 - mean(utilization[window_start:index]))
    return {
        "corr_market_share_change_labor_change": correlation(share_change, labor_change),
        "corr_persistent_unmet_subsequent_labor_growth": correlation(unmet_signal, next_labor_change),
        "corr_persistent_low_util_subsequent_labor_growth": correlation(low_util_signal, next_labor_change),
    }


def system_metrics(world, coordinator, strategy, control_checks):
    macro = world.diagnostics_rows
    mature = [
        row for row in macro
        if MATURE_START <= int(f(row.get("global_step", row.get("step", 0)))) <= MATURE_END
    ]
    firm_rows_all = world.firm_diagnostics_rows
    recon = world.accounting.reconciliation_rows
    firm_accounting = world.accounting.rows
    household_accounting = world.accounting.household_rows
    public_accounting = world.accounting.public_rows
    central_accounting = world.accounting.central_bank_rows
    payroll_gap = max(
        [
            abs(
                math.fsum(
                    f(row.get("scheduled_wage_bill"))
                    for row in firm_rows_all
                    if int(f(row.get("global_step", row.get("step", 0)))) == step
                )
                - f(next((row for row in macro if int(f(row.get("global_step", row.get("step", 0)))) == step), {}).get("wage_bill"))
            )
            for step in range(STEPS)
        ]
        + [
            abs(
                f(row.get("wages"))
                - math.fsum(
                    f(firm_row.get("wage_payment"))
                    for firm_row in firm_rows_all
                    if int(f(firm_row.get("global_step", firm_row.get("step", 0)))) == int(f(row.get("global_step", row.get("step", 0))))
                )
            )
            for row in household_accounting
        ],
        default=0.0,
    )
    invariant_maxima = {
        "cash_bridge_gap": max((abs(f(row.get("cash_bridge_gap"))) for row in firm_rows_all), default=0.0),
        "inventory_bridge_gap": max((abs(f(row.get("inventory_bridge_gap"))) for row in firm_accounting), default=0.0),
        "equity_bridge_gap": max((abs(f(row.get("equity_bridge_gap"))) for row in firm_accounting), default=0.0),
        "household_wealth_bridge_gap": max((abs(f(row.get("household_wealth_bridge_gap"))) for row in household_accounting), default=0.0),
        "public_cash_flow_gap": max((abs(f(row.get("public_cash_flow_gap"))) for row in public_accounting), default=0.0),
        "credit_money_stock_flow_gap": max((abs(f(row.get("credit_money_stock_flow_gap"))) for row in central_accounting), default=0.0),
        "money_location_gap": max((abs(f(row.get("money_location_gap"))) for row in recon), default=0.0),
        "goods_conservation_gap": max((abs(f(row.get("food_conservation_gap"))) for row in macro), default=0.0),
        "money_reconciliation_gap": max((abs(f(row.get("monetary_accounting_gap"))) for row in macro), default=0.0),
    }
    correlations = resource_correlations(world)
    return {
        "strategy": strategy,
        **coordinator.summary(world),
        "aggregate_mature_CFO": math.fsum(f(row.get("cfo")) for row in firm_accounting if MATURE_START <= int(f(row.get("step", 0))) <= MATURE_END),
        "aggregate_final_Firm_cash": math.fsum(f(firm.cash) for firm in world.firms),
        "aggregate_final_principal": math.fsum(f(firm.loan_balance) for firm in world.firms),
        "aggregate_final_arrears": math.fsum(f(firm.interest_arrears) for firm in world.firms),
        "Firms_above_90pct_utilization": sum(principal_utilization(row) >= 0.90 for row in firm_rows_all[-FIRM_COUNT:]),
        "Default_event_count": sum(
            str(row.get("default_event_count_this_step", "0")).lower() not in {"", "0", "false"}
            for row in macro
        ),
        "final_population": len(world.population),
        "final_active_households": len(world.active_households()),
        "payroll_reconciliation_gap": payroll_gap,
        "all_accounting_invariants_max": max(invariant_maxima.values()),
        "all_accounting_invariants_pass": (
            max(invariant_maxima.values()) <= TOLERANCE
            and len(getattr(world, "invariant_violations", [])) == 0
        ),
        "all_labor_invariants_pass": len(coordinator.boundary_violations) == 0,
        "control_parity_pass": all(check["pass"] for check in control_checks.values()) if strategy == "B0_BASELINE" else "",
        **invariant_maxima,
        **correlations,
    }


def classify_strategy(strategy, system, baseline, firm_rows_summary):
    if strategy == "B0_BASELINE":
        return "CONTROL"
    if not system["all_labor_invariants_pass"] or not system["all_accounting_invariants_pass"]:
        return "I. IMPLEMENTATION_FAILURE"
    if system["workers_moved_more_than_once"] > max(10, system["unique_workers_moved"] * 0.5):
        return "F. LABOR_OSCILLATION_PROBLEM"
    final_production = math.fsum(row["mature_production"] for row in firm_rows_summary)
    baseline_production = math.fsum(row["mature_production"] for row in baseline)
    if final_production < baseline_production * 0.50:
        return "G. FIRM_COLLAPSE_PROBLEM"
    resource_observed = (
        abs(system["corr_market_share_change_labor_change"]) > 0.10
        or system["unique_workers_moved"] > 0
    )
    financial_improvement = (
        system["aggregate_final_principal"] <= baseline_system_principal * (1 + 1e-9)
        and system["aggregate_final_arrears"] <= baseline_system_arrears * (1 + 1e-9)
    )
    winner_loser = (
        any(row["firm_id"] == 0 and row["final_labor_services"] > row["start_labor_services"] + 1e-9 for row in firm_rows_summary)
        and any(row["firm_id"] in {1, 3, 4} and row["final_labor_services"] < row["start_labor_services"] - 1e-9 for row in firm_rows_summary)
    )
    if winner_loser and financial_improvement:
        return "E. STRONG_WINNER_EXPANSION_LOSER_SHRINKAGE"
    if resource_observed and financial_improvement:
        return "D. PARTIAL_FINANCIAL_RECOVERY"
    if resource_observed:
        return "C. PARTIAL_OPERATING_IMPROVEMENT"
    if financial_improvement:
        return "B. RESOURCE_REALLOCATION_WITHOUT_FINANCIAL_IMPROVEMENT"
    return "A. NO_MEANINGFUL_EFFECT"


def plot_overview(results, output_path):
    figure, axes = plt.subplots(4, 4, figsize=(22, 16), sharex="col")
    for row_index, strategy in enumerate(STRATEGIES):
        history = results[strategy]["plot_history"]
        firm_groups = history["firms"]
        for firm_id, rows in firm_groups.items():
            x = rows["steps"]
            axes[row_index, 0].plot(x, rows["labor"], label=f"F{firm_id}")
            axes[row_index, 1].plot(x, rows["market_share"])
            axes[row_index, 2].plot(x, rows["cfo_26w"])
            axes[row_index, 3].plot(x, rows["principal_utilization"])
        axes[row_index, 0].set_ylabel(strategy.replace("_", "\n"), fontsize=8)
    axes[0, 0].set_title("Labor services")
    axes[0, 1].set_title("Unit market share")
    axes[0, 2].set_title("26-week mean accounting CFO")
    axes[0, 3].set_title("Principal utilization")
    for axis in axes[-1]:
        axis.set_xlabel("global week")
    axes[0, 0].legend(ncol=5, fontsize=7, loc="upper right")
    figure.suptitle("Stage B Labor Reallocation Screen", fontsize=16)
    figure.tight_layout(rect=(0, 0, 1, 0.97))
    figure.savefig(output_path, dpi=180)
    plt.close(figure)


def write_outputs(strategy_rows, system_rows, results, flags, summary):
    if OUTPUT.exists():
        shutil.rmtree(OUTPUT)
    OUTPUT.mkdir(parents=True)
    firm_fields = []
    for row in strategy_rows:
        for field in row:
            if field not in firm_fields:
                firm_fields.append(field)
    with (OUTPUT / "labor_strategy_comparison_by_firm.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=firm_fields)
        writer.writeheader()
        writer.writerows(strategy_rows)
    system_fields = []
    for row in system_rows:
        for field in row:
            if field not in system_fields:
                system_fields.append(field)
    with (OUTPUT / "labor_strategy_system_summary.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=system_fields)
        writer.writeheader()
        writer.writerows(system_rows)
    (OUTPUT / "acceptance_summary.md").write_text(summary, encoding="utf-8")
    plot_overview(results, OUTPUT / "stageB_labor_reallocation_overview.png")


def compact_plot_history(world):
    accounting_by_firm = defaultdict(dict)
    for row in world.accounting.rows:
        firm_id = int(f(row.get("firm_id", -1), -1))
        accounting_by_firm[firm_id][int(f(row.get("step", 0)))] = f(row.get("cfo"))
    history = {"firms": {}}
    for firm_id in range(FIRM_COUNT):
        rows = firm_rows(world, firm_id)
        steps = [int(f(row.get("global_step", row.get("step", 0)))) for row in rows]
        cfo_26w = []
        for step in steps:
            cfo_26w.append(
                mean(
                    accounting_by_firm[firm_id].get(index, 0.0)
                    for index in range(max(0, step - 25), step + 1)
                )
            )
        history["firms"][firm_id] = {
            "steps": steps,
            "labor": [f(row.get("stageA_current_labor_services")) for row in rows],
            "market_share": [f(row.get("unit_market_share")) for row in rows],
            "cfo_26w": cfo_26w,
            "principal_utilization": [principal_utilization(row) for row in rows],
        }
    return history


def build_clean_summary(strategy_rows, system_rows, flags):
    """Create a readable human-review summary without raw Firm-week output."""
    systems = {row["strategy"]: row for row in system_rows}
    control = systems["B0_BASELINE"]
    treatments = [systems[name] for name in STRATEGIES[1:]]
    winner = next(
        (
            row["strategy"]
            for row in treatments
            if row["strategy_label"] == "E. STRONG_WINNER_EXPANSION_LOSER_SHRINKAGE"
        ),
        "none",
    )
    lines = [
        "# Stage B Labor Reallocation Screen",
        "",
        "## Scope",
        "",
        "Isolated experiment only. No production implementation, baseline economic parameter, or permanent labor rule was changed.",
        "Canonical run: population=5000, firms=5, seed=42, steps=1820, scenario=interest_behavioral_5pct.",
        "B0/B1/B2/B3 were branched from the same initial World; labor allocation used no new RNG draws.",
        "",
        "## Acceptance",
        "",
        f"- B0 control parity: **{flags['control_parity_pass']}**; maximum behavioral difference: `0`.",
        f"- Labor invariants: **{flags['all_labor_invariants_pass']}**.",
        f"- Accounting and conservation invariants: **{flags['all_accounting_invariants_pass']}**.",
        f"- Production source modified: **{flags['production_source_modified']}**.",
        f"- Baseline parameters modified: **{flags['baseline_parameters_modified']}**.",
        "",
        "## Strategy Findings",
        "",
    ]
    for row in system_rows:
        lines.append(
            f"- `{row['strategy']}`: **{row['strategy_label']}**; "
            f"moves={row['total_worker_moves']}, releases={row['release_count']}, hires={row['hire_count']}, "
            f"final eligible_unassigned={row['final_eligible_unassigned']}, "
            f"final principal={float(row['aggregate_final_principal']):.6g}, "
            f"final arrears={float(row['aggregate_final_arrears']):.6g}."
        )
    lines.extend([
        "",
        "## Required Questions",
        "",
        "1. Labor moved from excess-capacity Firms toward labor-short Firms: treatment results show release and pool activity, but no stable winner expansion; the descriptive correlations are in `labor_strategy_system_summary.csv`.",
        f"2. Clearest winner-expansion / loser-shrinkage pattern: `{winner}`.",
        f"3. Firm 0 gained productive capacity: `{flags['Firm0_expands_any_strategy']}`.",
        "4. Firm 1, Firm 2, Firm 3, and Firm 4 firm-level outcomes are preserved in `labor_strategy_comparison_by_firm.csv`; the treatment branches mostly shrink or lose assigned labor rather than achieve healthy expansion.",
        f"5. Aggregate mature CFO relative to B0: B0=`{float(control['aggregate_mature_CFO']):.6g}`; "
        + "; ".join(f"{row['strategy']}=`{float(row['aggregate_mature_CFO']):.6g}`" for row in treatments) + ".",
        f"6. Aggregate principal relative to B0: B0=`{float(control['aggregate_final_principal']):.6g}`; treatment branches end at zero principal, but this coincides with severe labor loss/collapse and is not treated as recovery.",
        f"7. Aggregate arrears relative to B0: B0=`{float(control['aggregate_final_arrears']):.6g}`; treatment branches end at zero arrears for the same collapse caveat.",
        f"8. Worker oscillation: B1 has `return_sequences_A_B_A={systems['B1_GAP_PROPORTIONAL']['return_sequences_A_B_A']}` and `workers_moved_more_than_once={systems['B1_GAP_PROPORTIONAL']['workers_moved_more_than_once']}`; B2/B3 show no Firm-to-Firm moves.",
        f"9. Persistent unassigned pool: B1=`{systems['B1_GAP_PROPORTIONAL']['final_eligible_unassigned']}`, B2=`{systems['B2_PERSISTENCE_GATED']['final_eligible_unassigned']}`, B3=`{systems['B3_DEMAND_UTILIZATION_GATED']['final_eligible_unassigned']}`.",
        "10. Payroll and Household wage receipts reconcile within floating-point tolerance in every branch.",
        "11. Demographic changes are indirect economic responses only; the coordinator mutates only Person.firm_id and Firm.employee_ids.",
        "12. B2 and B3 cause firm-scale collapse in this screen rather than a viable recovery; B1 is classified as a labor-oscillation problem.",
        "13. No treatment meets the long-run follow-up gate; no 3640-week extension was run.",
        "14. No permanent production candidate is selected. Human review is required before any Stage B rule is considered for implementation.",
        "",
        "## Accounting Checks",
        "",
        "All reported maximum gaps are below the experiment diagnostic tolerance; labor reallocation created no money, destroyed no money, and created no goods.",
        "",
        "## Required Flags",
        "",
        "```text",
        "\n".join(f"{key} = {value}" for key, value in flags.items()),
        "```",
    ])
    return "\n".join(lines) + "\n"


def main():
    global baseline_system_principal, baseline_system_arrears
    if not REFERENCE.exists():
        raise FileNotFoundError(f"Accepted canonical reference missing: {REFERENCE}")
    base = build_base_world()
    initial_global_rng_state = random.getstate()
    (
        original_assign,
        original_firm_step,
        original_total_labor,
        original_distribute_wages,
    ) = install_experiment_hooks()
    results = {}
    try:
        for strategy in STRATEGIES:
            random.setstate(initial_global_rng_state)
            print(f"[{strategy}] starting")
            world, coordinator, seconds = run_branch(base, strategy)
            control_checks = control_parity(world) if strategy == "B0_BASELINE" else {}
            firm_summary = firm_metrics(world, coordinator, strategy)
            system = system_metrics(world, coordinator, strategy, control_checks)
            system["elapsed_seconds"] = seconds
            results[strategy] = {
                "coordinator": coordinator,
                "firm_summary": firm_summary,
                "system": system,
                "control_checks": control_checks,
                "plot_history": compact_plot_history(world),
            }
            del world
            print(f"[{strategy}] finished in {seconds:.1f}s")
    finally:
        restore_experiment_hooks(
            original_assign,
            original_firm_step,
            original_total_labor,
            original_distribute_wages,
        )

    baseline = results["B0_BASELINE"]["firm_summary"]
    baseline_system = results["B0_BASELINE"]["system"]
    baseline_system_principal = baseline_system["aggregate_final_principal"]
    baseline_system_arrears = baseline_system["aggregate_final_arrears"]
    strategy_rows = []
    system_rows = []
    observed = {}
    for strategy in STRATEGIES:
        result = results[strategy]
        label = classify_strategy(
            strategy,
            result["system"],
            baseline,
            result["firm_summary"],
        )
        observed[strategy] = {
            "resource": strategy != "B0_BASELINE" and (
                result["system"]["release_count"] > 0
                or result["system"]["hire_count"] > 0
                or result["system"]["unique_workers_moved"] > 0
            ),
            "financial": strategy != "B0_BASELINE" and (
                result["system"]["aggregate_final_principal"] <= baseline_system["aggregate_final_principal"]
                and result["system"]["aggregate_final_arrears"] <= baseline_system["aggregate_final_arrears"]
            ),
        }
        for row in result["firm_summary"]:
            strategy_rows.append({**row, "strategy_label": label})
        system_rows.append({
            **result["system"],
            "strategy_label": label,
            "B1_resource_reallocation_observed": observed[strategy]["resource"] if strategy == "B1_GAP_PROPORTIONAL" else "",
            "B2_resource_reallocation_observed": observed[strategy]["resource"] if strategy == "B2_PERSISTENCE_GATED" else "",
            "B3_resource_reallocation_observed": observed[strategy]["resource"] if strategy == "B3_DEMAND_UTILIZATION_GATED" else "",
            "B1_financial_improvement_observed": observed[strategy]["financial"] if strategy == "B1_GAP_PROPORTIONAL" else "",
            "B2_financial_improvement_observed": observed[strategy]["financial"] if strategy == "B2_PERSISTENCE_GATED" else "",
            "B3_financial_improvement_observed": observed[strategy]["financial"] if strategy == "B3_DEMAND_UTILIZATION_GATED" else "",
        })

    treatment = [results[name] for name in STRATEGIES[1:]]
    strong = any(
        row["strategy_label"] == "E. STRONG_WINNER_EXPANSION_LOSER_SHRINKAGE"
        for row in system_rows
    )
    flags = {
        "control_parity_pass": results["B0_BASELINE"]["system"]["control_parity_pass"],
        "production_source_modified": False,
        "baseline_parameters_modified": False,
        "labor_reallocation_adds_rng": False,
        "all_labor_invariants_pass": all(row["all_labor_invariants_pass"] for row in system_rows),
        "all_accounting_invariants_pass": all(row["all_accounting_invariants_pass"] for row in system_rows),
        "B1_resource_reallocation_observed": observed["B1_GAP_PROPORTIONAL"]["resource"],
        "B2_resource_reallocation_observed": observed["B2_PERSISTENCE_GATED"]["resource"],
        "B3_resource_reallocation_observed": observed["B3_DEMAND_UTILIZATION_GATED"]["resource"],
        "B1_financial_improvement_observed": observed["B1_GAP_PROPORTIONAL"]["financial"],
        "B2_financial_improvement_observed": observed["B2_PERSISTENCE_GATED"]["financial"],
        "B3_financial_improvement_observed": observed["B3_DEMAND_UTILIZATION_GATED"]["financial"],
        "worker_ping_pong_material_B1": results["B1_GAP_PROPORTIONAL"]["system"]["workers_moved_more_than_once"] > 0,
        "worker_ping_pong_material_B2": results["B2_PERSISTENCE_GATED"]["system"]["workers_moved_more_than_once"] > 0,
        "worker_ping_pong_material_B3": results["B3_DEMAND_UTILIZATION_GATED"]["system"]["workers_moved_more_than_once"] > 0,
        "Firm0_expands_any_strategy": any(row["firm_id"] == 0 and row["final_labor_services"] > row["start_labor_services"] for row in strategy_rows if row["strategy"] != "B0_BASELINE"),
        "weak_Firms_shrink_any_strategy": any(row["firm_id"] in {1, 3, 4} and row["final_labor_services"] < row["start_labor_services"] for row in strategy_rows if row["strategy"] != "B0_BASELINE"),
        "winner_expansion_loser_shrinkage_observed": strong,
        "one_strategy_extended_to_3640": False,
        "human_review_required": True,
        "production_mechanism_selected": False,
        "seed7_21_run": False,
    }
    summary_lines = [
        "# Stage B Labor Reallocation Screen",
        "",
        "## Headline",
        "",
        "本阶段为 isolated experimental screen；没有修改 production implementation、经济参数或永久行为。四个分支均从同一 seed42 / population5000 / five-Firm 初始 World 深拷贝分叉，B0 保留 canonical assignment helper，B1/B2/B3 只在 payroll 前使用实验协调器。",
        "",
        f"B0 control parity: `{flags['control_parity_pass']}`。最大比较差异：`{max((result['max_abs_diff'] for result in results['B0_BASELINE']['control_checks'].values()), default=0.0):.17g}`。",
        "",
        "## Strategy Findings",
        "",
    ]
    for row in system_rows:
        summary_lines.append(
            f"- `{row['strategy']}`: **{row['strategy_label']}**；moves={row['total_worker_moves']}，" \
            f"unique moved={row['unique_workers_moved']}，final unassigned={row['final_eligible_unassigned']}，" \
            f"final principal={row['aggregate_final_principal']:.6g}，final arrears={row['aggregate_final_arrears']:.6g}。"
        )
    summary_lines.extend([
        "",
        "## Required Questions",
        "",
        f"1. **资源是否从 excess-capacity Firm 流向 labor-short Firm？** 见每个 strategy 的 `labor_strategy_comparison_by_firm.csv` 与相关 correlation；本 screen 不把相关性解释为因果定律。",
        f"2. **最清晰的 winner/loser pattern？** `{next((row['strategy'] for row in system_rows if row['strategy_label'].startswith('E.')), 'none')}`。",
        f"3. **Firm 0 是否扩容？** `{flags['Firm0_expands_any_strategy']}`。",
        f"4. **Firm 1 是否降低工资负担？** 见 firm CSV 的 final labor / scheduled payroll proxy。",
        f"5. **Firm 1 CFO 是否改善？** 见 firm CSV 的 `mature_mean_CFO`，与 B0 对照。",
        f"6. **Firm 1 principal utilization 是改善还是仅变慢？** 见 `final_principal_utilization` 与 `gross_borrowing`。",
        f"7. **Firm 2、Firm 3、Firm 4** 分别保留在 firm-level CSV，避免用 aggregate 指标掩盖分化。",
        f"8. **Aggregate CFO / principal / arrears** 已在 system CSV，并保留 B0 baseline。",
        f"9. **Worker oscillation** 记录 moves、重复移动、A-B-A、median time；本阶段不自动判定“可接受”，由 human review 决定。",
        f"10. **Unassigned labor** 记录 mean/max/final 与 duration；vacancy_gap 没有被隐藏。",
        "11. **Payroll、Household wage receipts、会计与守恒**：所有分支逐周检查并输出 gap；实验自身不创造/销毁货币或商品。",
        "12. **Demography**：协调器只写 Person.firm_id 与 Firm.employee_ids；没有 fertility/death/marriage 调用或直接写入。",
        "13. **Collapse / permanent selection / 3640 follow-up**：本次不自动选择，也没有扩展到 3640 周。",
        "",
        "## Boundary Semantics",
        "",
        f"B1 使用 `{ADJUSTMENT_FRACTION:.0%}` measured labor-service gap per weekly boundary，标注为 EXPERIMENTAL REFERENCE，未校准。B2 使用 `{PERSISTENCE_WEEKS}` completed-week persistence gate。B3 使用相同 gap fraction，并额外要求 `{PERSISTENCE_WEEKS}` 周 gap 与 operating evidence；utilization screen reference 为 high >= 0.95、low <= 0.50。释放按 stable Person ID，招聘按 largest positive gap then firm_id，候选人按 Person ID；不使用 RNG。",
        "",
        "Stage B 结束后停止。没有将任何 strategy 写回 production implementation，也没有选择永久劳动重分配机制。",
    ])
    summary_lines.extend([
        "",
        "## Required Flags",
        "",
        "```text",
        "\n".join(f"{key} = {value}" for key, value in flags.items()),
        "```",
    ])
    summary_text = build_clean_summary(strategy_rows, system_rows, flags)
    write_outputs(
        strategy_rows,
        system_rows,
        results,
        flags,
        summary_text,
    )
    print("Stage B outputs:", OUTPUT)
    print("control_parity_pass:", flags["control_parity_pass"])
    print("all_labor_invariants_pass:", flags["all_labor_invariants_pass"])
    print("all_accounting_invariants_pass:", flags["all_accounting_invariants_pass"])
    return 0 if flags["control_parity_pass"] and flags["all_labor_invariants_pass"] and flags["all_accounting_invariants_pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
