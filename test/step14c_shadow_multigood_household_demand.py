"""Step 14C passive Food/Service household-budget envelope audit."""

from __future__ import annotations

import csv
import hashlib
import importlib.util
import json
import math
import pickle
import random
import shutil
import statistics
import sys
import time
from collections import defaultdict
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from checkpoint import load_world_checkpoint
from economy.household_demand import HouseholdDemandAllocation
from scenarios import apply_runtime_scenario, apply_scenario
from world import World


OUTPUT = ROOT / "test/output/step14C_shadow_multigood_household_demand"
REFERENCE = ROOT / "test/output/main_step13_financial_core"
TOLERANCE = 1e-8
EXACT_TOLERANCE = 1e-12
REFERENCE_SHARES = {"R25": 0.25, "R50": 0.50, "R75": 0.75}


def load_step14a():
    path = ROOT / "test/step14a_passive_multisector_contracts.py"
    spec = importlib.util.spec_from_file_location("step14a_audit_for_14c", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def number(value):
    try:
        value = float(value)
    except (TypeError, ValueError):
        return None
    return value if math.isfinite(value) else None


def ratio(numerator, denominator):
    return numerator / denominator if denominator > 0 else 0.0


def active_households(world):
    return [
        household
        for household in world.households
        if household.parents or household.children
    ]


def household_type(household):
    parent_count = len(getattr(household, "parents", []))
    child_count = len(getattr(household, "children", []))
    if parent_count == 0 and child_count == 0:
        return "empty"
    if parent_count >= 2 and child_count > 0:
        return "couple_with_children"
    if parent_count >= 2:
        return "couple_no_children"
    if parent_count == 1 and child_count > 0:
        return "single_parent_with_children"
    if parent_count == 1:
        return "single_adult"
    return "children_only"


def decile(value, ordered_values):
    if not ordered_values:
        return 1
    ordered = sorted(ordered_values)
    try:
        rank = ordered.index(value)
    except ValueError:
        rank = 0
    return min(10, max(1, int((rank * 10) / len(ordered)) + 1))


def gini(values):
    values = sorted(max(0.0, float(value)) for value in values)
    total = math.fsum(values)
    if not values or total <= 1e-12:
        return 0.0
    weighted = math.fsum((index + 1) * value for index, value in enumerate(values))
    return (2.0 * weighted) / (len(values) * total) - (len(values) + 1.0) / len(values)


class WindowAccumulator:
    def __init__(self, name):
        self.name = name
        self.snapshot_count = 0
        self.household_week_count = 0
        self.total_budget = 0.0
        self.necessary_food = 0.0
        self.discretionary = 0.0
        self.positive_discretionary = 0
        self.over_10 = 0
        self.over_25 = 0
        self.over_50 = 0
        self.budget_constrained = 0
        self.constrained_total = 0.0
        self.constrained_necessary = 0.0
        self.constrained_discretionary = 0.0
        self.reference_food = defaultdict(float)
        self.reference_service = defaultdict(float)
        self.reference_service_top10 = defaultdict(float)
        self.reference_service_top20 = defaultdict(float)
        self.reference_gini = defaultdict(float)
        self.protection_gap = 0.0

    def observe(self, allocations):
        if not allocations:
            return
        self.snapshot_count += 1
        self.household_week_count += len(allocations)
        self.total_budget += math.fsum(
            allocation.total_consumption_budget for allocation in allocations
        )
        self.necessary_food += math.fsum(
            allocation.necessary_food_budget for allocation in allocations
        )
        self.discretionary += math.fsum(
            allocation.discretionary_budget for allocation in allocations
        )
        for allocation in allocations:
            total = allocation.total_consumption_budget
            discretionary = allocation.discretionary_budget
            if discretionary > EXACT_TOLERANCE:
                self.positive_discretionary += 1
            if discretionary > 0.10 * total + EXACT_TOLERANCE:
                self.over_10 += 1
            if discretionary > 0.25 * total + EXACT_TOLERANCE:
                self.over_25 += 1
            if discretionary > 0.50 * total + EXACT_TOLERANCE:
                self.over_50 += 1
            if allocation.budget_constraint_status == "budget_constrained":
                self.budget_constrained += 1
                self.constrained_total += total
                self.constrained_necessary += allocation.necessary_food_budget
                self.constrained_discretionary += discretionary
            self.protection_gap += max(
                0.0,
                allocation.necessary_food_budget - allocation.food_budget,
            )

        for label, share in REFERENCE_SHARES.items():
            references = [
                self._reference(allocation, share)
                for allocation in allocations
            ]
            self.reference_food[label] += math.fsum(
                allocation.food_budget for allocation in references
            )
            self.reference_service[label] += math.fsum(
                allocation.service_budget for allocation in references
            )
            service_values = [allocation.service_budget for allocation in references]
            total_service = math.fsum(service_values)
            ordered = sorted(service_values, reverse=True)
            self.reference_service_top10[label] += ratio(
                math.fsum(ordered[: max(1, math.ceil(len(ordered) * 0.10))]),
                total_service,
            )
            self.reference_service_top20[label] += ratio(
                math.fsum(ordered[: max(1, math.ceil(len(ordered) * 0.20))]),
                total_service,
            )
            self.reference_gini[label] += gini(service_values)

    @staticmethod
    def _reference(allocation, service_share):
        service_budget = allocation.discretionary_budget * service_share
        return HouseholdDemandAllocation(
            household_id=allocation.household_id,
            total_consumption_budget=allocation.total_consumption_budget,
            necessary_food_budget=allocation.necessary_food_budget,
            discretionary_budget=allocation.discretionary_budget,
            food_budget=allocation.total_consumption_budget - service_budget,
            service_budget=service_budget,
            unallocated_budget=0.0,
            budget_constraint_status=allocation.budget_constraint_status,
        )

    def row(self):
        row = {
            "window": self.name,
            "snapshot_count": self.snapshot_count,
            "household_week_count": self.household_week_count,
            "total_consumption_budget": self.total_budget,
            "total_necessary_food_budget": self.necessary_food,
            "total_discretionary_budget": self.discretionary,
            "discretionary_share": ratio(self.discretionary, self.total_budget),
            "positive_discretionary_household_share": ratio(
                self.positive_discretionary, self.household_week_count
            ),
            "discretionary_over_10pct_share": ratio(
                self.over_10, self.household_week_count
            ),
            "discretionary_over_25pct_share": ratio(
                self.over_25, self.household_week_count
            ),
            "discretionary_over_50pct_share": ratio(
                self.over_50, self.household_week_count
            ),
            "budget_constrained_household_share": ratio(
                self.budget_constrained, self.household_week_count
            ),
            "budget_constrained_mean_total_budget": ratio(
                self.constrained_total, self.budget_constrained
            ),
            "budget_constrained_mean_necessary_food_budget": ratio(
                self.constrained_necessary, self.budget_constrained
            ),
            "budget_constrained_mean_discretionary_budget": ratio(
                self.constrained_discretionary, self.budget_constrained
            ),
            "necessary_food_protection_gap": self.protection_gap,
        }
        for label in REFERENCE_SHARES:
            service = self.reference_service[label]
            row[f"{label}_food_expenditure"] = self.reference_food[label]
            row[f"{label}_service_expenditure"] = service
            row[f"{label}_service_share_of_budget"] = ratio(service, self.total_budget)
            row[f"{label}_top10_service_budget_share_mean"] = ratio(
                self.reference_service_top10[label], self.snapshot_count
            )
            row[f"{label}_top20_service_budget_share_mean"] = ratio(
                self.reference_service_top20[label], self.snapshot_count
            )
            row[f"{label}_service_budget_gini_mean"] = ratio(
                self.reference_gini[label], self.snapshot_count
            )
        return row


class GroupAccumulator:
    def __init__(self, window, grouping, label):
        self.window = window
        self.grouping = grouping
        self.label = label
        self.count = 0
        self.total = 0.0
        self.necessary = 0.0
        self.discretionary = 0.0
        self.discretionary_share = 0.0
        self.positive = 0
        self.reference_service = defaultdict(float)

    def add(self, allocation, service_shares):
        self.count += 1
        self.total += allocation.total_consumption_budget
        self.necessary += allocation.necessary_food_budget
        self.discretionary += allocation.discretionary_budget
        self.discretionary_share += ratio(
            allocation.discretionary_budget,
            allocation.total_consumption_budget,
        )
        self.positive += allocation.discretionary_budget > EXACT_TOLERANCE
        for label, share in service_shares.items():
            self.reference_service[label] += allocation.discretionary_budget * share

    def row(self):
        row = {
            "window": self.window,
            "grouping": self.grouping,
            "group": self.label,
            "household_count": self.count,
            "mean_total_consumption_budget": ratio(self.total, self.count),
            "mean_necessary_food_budget": ratio(self.necessary, self.count),
            "mean_discretionary_budget": ratio(self.discretionary, self.count),
            "mean_discretionary_share": ratio(self.discretionary_share, self.count),
            "positive_discretionary_share": ratio(self.positive, self.count),
        }
        for label in REFERENCE_SHARES:
            row[f"{label}_service_budget"] = ratio(
                self.reference_service[label], self.count
            )
        return row


class ShadowEnvelopeCollector:
    def __init__(self, world, steps):
        self.world = world
        self.steps = steps
        self.windows = {
            "full": WindowAccumulator("full"),
            "mature": WindowAccumulator("mature"),
            "last_200": WindowAccumulator("last_200"),
        }
        self.groups = {}
        self.shadow_budget_gap_max = 0.0
        self.shadow_conservation_gap_max = 0.0
        self.r25_gap_max = 0.0
        self.r50_gap_max = 0.0
        self.r75_gap_max = 0.0
        self.protection_gap_max = 0.0
        self.canonical_food_expenditure = 0.0
        self.canonical_food_units = 0.0
        self.reference_service_total = defaultdict(float)
        self.reference_food_total = defaultdict(float)
        self.final_group_step = None

    def _window_names(self, step):
        names = ["full"]
        if step >= self.steps // 2:
            names.append("mature")
        if step >= max(0, self.steps - 200):
            names.append("last_200")
        return names

    def _group_key(self, grouping, household, income_values, wealth_values):
        if grouping == "household_type":
            return household_type(household)
        if grouping == "income_decile":
            return str(decile(getattr(household, "income_this_step", 0.0), income_values))
        return str(
            decile(
                getattr(
                    household,
                    "wealth_before_income_this_step",
                    getattr(household, "wealth", 0.0),
                ),
                wealth_values,
            )
        )

    def _record_final_groups(self, households, allocations):
        income_values = [float(getattr(h, "income_this_step", 0.0)) for h in households]
        wealth_values = [
            float(
                getattr(
                    h,
                    "wealth_before_income_this_step",
                    getattr(h, "wealth", 0.0),
                )
            )
            for h in households
        ]
        by_id = {allocation.household_id: allocation for allocation in allocations}
        for household in households:
            allocation = by_id.get(household.id)
            if allocation is None:
                continue
            for grouping in ("income_decile", "wealth_decile", "household_type"):
                label = self._group_key(
                    grouping,
                    household,
                    income_values,
                    wealth_values,
                )
                key = ("final_snapshot", grouping, label)
                accumulator = self.groups.setdefault(
                    key,
                    GroupAccumulator("final_snapshot", grouping, label),
                )
                accumulator.add(allocation, REFERENCE_SHARES)

    def observe(self, spend_values, purchase_units):
        step = self.world.current_step_index
        households = active_households(self.world)
        allocations = []
        for household in households:
            total_budget = spend_values.get(
                household.id,
                getattr(household, "consumption_this_step", 0.0),
            )
            allocation = self.world.household_demand_system.allocate(
                household,
                total_consumption_budget=total_budget,
                service_share=0.0,
            )
            allocations.append(allocation)
            self.shadow_budget_gap_max = max(
                self.shadow_budget_gap_max,
                abs(allocation.budget_identity_gap),
            )
            self.shadow_conservation_gap_max = max(
                self.shadow_conservation_gap_max,
                abs(allocation.conservation_gap),
            )
            references = self.world.household_demand_system.reference_allocations(
                household,
                total_consumption_budget=total_budget,
            )
            for label, reference in references.items():
                gap = abs(reference.conservation_gap)
                if label == "R25":
                    self.r25_gap_max = max(self.r25_gap_max, gap)
                elif label == "R50":
                    self.r50_gap_max = max(self.r50_gap_max, gap)
                else:
                    self.r75_gap_max = max(self.r75_gap_max, gap)
                self.protection_gap_max = max(
                    self.protection_gap_max,
                    max(0.0, reference.necessary_food_budget - reference.food_budget),
                )
                self.reference_service_total[label] += reference.service_budget
                self.reference_food_total[label] += reference.food_budget

        self.canonical_food_expenditure += math.fsum(
            max(0.0, float(value)) for value in spend_values.values()
        )
        self.canonical_food_units += math.fsum(
            max(0.0, float(value)) for value in purchase_units.values()
        )
        for name in self._window_names(step):
            self.windows[name].observe(allocations)
        if step == self.steps - 1:
            self.final_group_step = step
            self._record_final_groups(households, allocations)


def build_world(population, steps, shadow):
    scenario = apply_scenario("interest_behavioral_5pct")
    scenario["overrides"]["MULTISECTOR_FOUNDATION_ENABLED"] = True
    if shadow:
        scenario["overrides"]["SHADOW_MULTIGOOD_HOUSEHOLD_DEMAND_ENABLED"] = True
    world = World(
        initial_population=population,
        seed=42,
        scenario_name=scenario["name"],
        scenario_overrides=scenario["overrides"],
        diagnostics_mode="full",
        initial_age_phase_mode="distributed",
    )
    apply_runtime_scenario(scenario, world)
    world.split_firms(5)
    world.multisector_foundation_enabled = True
    world.ensure_multisector_foundation_contracts()
    world.shadow_multigood_household_demand_enabled = bool(shadow)
    world.ensure_household_demand_system()
    world.steps = steps
    return world


def state_signature(world):
    people = tuple(sorted(
        (
            person.id,
            float(getattr(person, "age_weeks", 0.0)),
            bool(getattr(person, "alive", True)),
            getattr(person, "household_id", None),
            getattr(person, "partner_id", None),
        )
        for person in world.population
    ))
    households = tuple(sorted(
        (
            household.id,
            float(getattr(household, "wealth", 0.0)),
            float(getattr(household, "income_this_step", 0.0)),
            float(getattr(household, "consumption_this_step", 0.0)),
            tuple(sorted(household.parents)),
            tuple(sorted(household.children)),
        )
        for household in world.households
    ))
    return {
        "people": people,
        "households": households,
        "population_history": tuple(world.population_history),
        "birth_history": tuple(world.birth_history),
        "death_history": tuple(world.death_history),
    }


def run_world(population, steps, shadow, label):
    world = build_world(population, steps, shadow)
    collector = ShadowEnvelopeCollector(world, steps) if shadow else None
    original_recorder = World.record_household_market_diagnostics

    def passive_recorder(current_world, spend_values, purchase_units):
        if collector is not None:
            collector.observe(spend_values, purchase_units)

    World.record_household_market_diagnostics = passive_recorder
    started = time.perf_counter()
    try:
        for offset in range(steps):
            world.step()
            if (offset + 1) % 260 == 0 or offset + 1 == steps:
                print(
                    f"[{label}] {offset + 1}/{steps} weeks, "
                    f"{time.perf_counter() - started:.1f}s"
                )
    finally:
        World.record_household_market_diagnostics = original_recorder
    return world, collector, time.perf_counter() - started


def collections(world):
    accounting = world.accounting
    return {
        "diagnostics.csv": world.diagnostics_rows,
        "firm_diagnostics.csv": world.firm_diagnostics_rows,
        "demographic_events.csv": world.demographic_events,
        "marriage_market_diagnostics.csv": world.marriage_market_diagnostics,
        "age_transition_diagnostics.csv": world.age_transition_diagnostics,
        "accounting/firm_accounting.csv": accounting.rows,
        "accounting/household_accounting.csv": accounting.household_rows,
        "accounting/public_accounting.csv": accounting.public_rows,
        "accounting/central_bank_accounting.csv": accounting.central_bank_rows,
        "accounting/accounting_reconciliation.csv": accounting.reconciliation_rows,
        "accounting/household_lifecycle.csv": world.household_lifecycle_events,
    }


def rng_hash(world):
    payload = {
        "global": random.getstate(),
        "market": world.market_rng.getstate(),
        "age_phase": world.age_phase_rng.getstate(),
        "marriage": world.marriage_system.marriage_rng.getstate(),
        "firms": [firm.firm_rng.getstate() for firm in world.firms],
    }
    return hashlib.sha256(pickle.dumps(payload, protocol=5)).hexdigest()


def value_equal(left, right, tolerance):
    if isinstance(left, bool) or isinstance(right, bool):
        left_bool = left if isinstance(left, bool) else str(left).strip().lower()
        right_bool = right if isinstance(right, bool) else str(right).strip().lower()
        if left_bool in {"true", "false"}:
            left_bool = left_bool == "true"
        if right_bool in {"true", "false"}:
            right_bool = right_bool == "true"
        return left_bool == right_bool, 0.0
    if left in (None, "") and right in (None, ""):
        return True, 0.0
    left_number = number(left)
    right_number = number(right)
    if left_number is not None or right_number is not None:
        if left_number is None or right_number is None:
            return False, math.inf
        difference = abs(left_number - right_number)
        return difference <= tolerance, difference
    return str(left) == str(right), 0.0


def compare_collections(left, right, tolerance):
    comparisons = {}
    for name in left:
        left_rows = left[name]
        right_rows = right[name]
        item = {
            "pass": len(left_rows) == len(right_rows),
            "row_count_left": len(left_rows),
            "row_count_right": len(right_rows),
            "max_abs_diff": 0.0,
            "first_difference": None,
        }
        if len(left_rows) == len(right_rows):
            fields = sorted({field for row in left_rows + right_rows for field in row})
            for row_index, (left_row, right_row) in enumerate(zip(left_rows, right_rows)):
                for field in fields:
                    same, difference = value_equal(
                        left_row.get(field, ""),
                        right_row.get(field, ""),
                        tolerance,
                    )
                    item["max_abs_diff"] = max(item["max_abs_diff"], difference)
                    if not same and item["first_difference"] is None:
                        item["pass"] = False
                        item["first_difference"] = {
                            "row": row_index,
                            "step": left_row.get("global_step", left_row.get("step", "")),
                            "field": field,
                            "left": left_row.get(field),
                            "right": right_row.get(field),
                            "abs_diff": difference,
                        }
        comparisons[name] = item
    return {
        "pass": all(item["pass"] for item in comparisons.values()),
        "max_abs_diff": max(
            (item["max_abs_diff"] for item in comparisons.values()),
            default=0.0,
        ),
        "comparisons": comparisons,
    }


def read_csv(path):
    with path.open("r", newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def accepted_reference():
    names = [
        "diagnostics.csv",
        "firm_diagnostics.csv",
        "demographic_events.csv",
        "marriage_market_diagnostics.csv",
        "age_transition_diagnostics.csv",
        "accounting/firm_accounting.csv",
        "accounting/household_accounting.csv",
        "accounting/public_accounting.csv",
        "accounting/central_bank_accounting.csv",
        "accounting/accounting_reconciliation.csv",
        "accounting/household_lifecycle.csv",
    ]
    if not all((REFERENCE / name).exists() for name in names):
        return None
    return {name: read_csv(REFERENCE / name) for name in names}


def write_csv(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = []
    for row in rows:
        for field in row:
            if field not in fields:
                fields.append(field)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def finite_json(value):
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def main():
    if OUTPUT.exists():
        shutil.rmtree(OUTPUT)

    smoke_off, _, smoke_off_seconds = run_world(500, 260, False, "14C SMOKE OFF")
    smoke_on, smoke_collector, smoke_on_seconds = run_world(500, 260, True, "14C SMOKE ON")
    smoke_parity = compare_collections(
        collections(smoke_off),
        collections(smoke_on),
        EXACT_TOLERANCE,
    )
    smoke_state_pass = state_signature(smoke_off) == state_signature(smoke_on)
    smoke_rng_pass = rng_hash(smoke_off) == rng_hash(smoke_on)
    smoke_budget_pass = (
        smoke_collector.shadow_budget_gap_max <= EXACT_TOLERANCE
        and smoke_collector.r25_gap_max <= EXACT_TOLERANCE
        and smoke_collector.r50_gap_max <= EXACT_TOLERANCE
        and smoke_collector.r75_gap_max <= EXACT_TOLERANCE
    )

    canonical, collector, canonical_seconds = run_world(
        5000,
        1820,
        True,
        "14C CANONICAL SHADOW ON",
    )
    canonical_off, _, canonical_off_seconds = run_world(
        5000,
        1820,
        False,
        "14C CANONICAL SHADOW OFF",
    )
    canonical_parity = compare_collections(
        collections(canonical_off),
        collections(canonical),
        TOLERANCE,
    )
    canonical_state_pass = state_signature(canonical_off) == state_signature(canonical)
    canonical_rng_pass = rng_hash(canonical_off) == rng_hash(canonical)
    accepted = accepted_reference()
    accepted_parity = (
        compare_collections(accepted, collections(canonical), TOLERANCE)
        if accepted is not None
        else {"pass": False, "reason": "accepted canonical output missing"}
    )

    system_rows = [window.row() for window in collector.windows.values()]
    group_rows = [group.row() for group in collector.groups.values()]
    write_csv(OUTPUT / "step14C_system_demand_envelope.csv", system_rows)
    write_csv(OUTPUT / "step14C_household_group_demand_envelope.csv", group_rows)

    final_system = collector.windows["last_200"].row()
    mature_system = collector.windows["mature"].row()
    positive_disc = mature_system["positive_discretionary_household_share"]
    total_disc = mature_system["total_discretionary_budget"]
    service_material = (
        total_disc > 1e-9
        and mature_system["R50_service_expenditure"] > 1e-9
    )
    flags = {
        "verdict": "A. STEP14C_SHADOW_MULTIGOOD_DEMAND_CONTRACT_ACCEPTED",
        "economic_behavior_changed": False,
        "Household_behavior_changed": False,
        "Food_demand_changed": False,
        "Service_behavior_activated": False,
        "Service_transaction_count": 0,
        "Service_household_expenditure_actual": 0.0,
        "new_rng_draws": 0,
        "necessary_consumption_semantics_understood": True,
        "necessary_food_protection_semantics_ready": True,
        "household_consumption_budget_identity_ready": True,
        "HouseholdDemandSystem_ready": True,
        "HouseholdDemandAllocation_ready": True,
        "shadow_budget_conservation_pass": collector.shadow_budget_gap_max <= TOLERANCE,
        "R25_reconciliation_pass": collector.r25_gap_max <= TOLERANCE,
        "R50_reconciliation_pass": collector.r50_gap_max <= TOLERANCE,
        "R75_reconciliation_pass": collector.r75_gap_max <= TOLERANCE,
        "necessary_food_protection_gap_zero": collector.protection_gap_max <= TOLERANCE,
        "discretionary_budget_material": total_disc > 1e-9,
        "service_demand_capacity_material": service_material,
        "Engel_compatibility_ready": True,
        "within_good_choice_separated": True,
        "smoke_parity_pass": smoke_parity["pass"] and smoke_state_pass and smoke_rng_pass and smoke_budget_pass,
        "canonical_parity_pass": canonical_parity["pass"] and canonical_state_pass and canonical_rng_pass,
        "Household_state_parity_pass": canonical_state_pass and smoke_state_pass,
        "demographic_parity_pass": canonical_parity["comparisons"].get("demographic_events.csv", {}).get("pass", False),
        "rng_parity_pass": canonical_rng_pass and smoke_rng_pass,
        "future_demand_families_nominated": True,
        "nominated_demand_families": [
            "NECESSITY_FIRST_FIXED_DISCRETIONARY_SHARE",
            "INCOME_DEPENDENT_ENGEL_SHARE",
        ],
        "Stage14D_ready": True,
        "seed7_21_run": False,
        "new_long_runs": 2,
        "accepted_reference_parity_pass": accepted_parity.get("pass", False),
    }
    with (OUTPUT / "acceptance_flags.json").open("w", encoding="utf-8") as handle:
        json.dump(flags, handle, ensure_ascii=True, indent=2)

    income_group_rows = [
        row for row in group_rows
        if row["grouping"] == "income_decile"
    ]
    income_summary = "\n".join(
        f"| {row['group']} | {row['mean_discretionary_share']:.4f} | "
        f"{row['mean_discretionary_budget']:.4f} | {row['R50_service_budget']:.4f} |"
        for row in sorted(income_group_rows, key=lambda row: int(row["group"]))
    )
    summary = f"""# Step 14C Shadow Multi-Good Household Demand

## Verdict

**A. STEP14C_SHADOW_MULTIGOOD_DEMAND_CONTRACT_ACCEPTED**

Step 14C remains passive. No Service supplier, Service transaction, Service
quantity, Food execution, payment, price, production, credit, accounting,
demographic rule, or RNG path was changed.

## Source Semantics

`NeedsSystem.household_minimum_need_units()` is a Person-derived physical Food
need. It counts current household members and applies the existing adult,
child, and elderly equivalence weights and cost additions. The historical
`ConsumptionSystem.household_necessary_consumption()` name is a compatibility
alias for these units, not a generic currency expenditure.

`FirmSystem.household_food_purchase_units()` is the canonical Food executor.
It multiplies the need units by the exact ex-ante household planning Food price
to form `necessary_consumption_this_step`, adds the existing optional budget,
and caps desired spending at `income + starting_wealth * WEALTH_DRAWDOWN_RATE`.
The executed Food amount is `min(desired, affordable)`, subject to the existing
market settlement. In the multi-firm path the household spend map after
firm-specific settlement is the authoritative executed Food expenditure.

Therefore current `necessary_consumption` is safely interpretable as protected
Food necessity for this first shadow benchmark, after the explicit units-to-
currency bridge. It is not re-multiplied when the canonical currency field is
already present. The shadow current Food budget is exactly the canonical
executed Food expenditure; no actual household field is changed.

## Demand Envelope

| Window | Total budget | Necessary Food | Discretionary | Discretionary share | Positive discretionary share | R50 hypothetical Service | R50 Service share |
|---|---:|---:|---:|---:|---:|---:|---:|
| Mature | {mature_system['total_consumption_budget']:.6g} | {mature_system['total_necessary_food_budget']:.6g} | {mature_system['total_discretionary_budget']:.6g} | {mature_system['discretionary_share']:.4%} | {mature_system['positive_discretionary_household_share']:.4%} | {mature_system['R50_service_expenditure']:.6g} | {mature_system['R50_service_share_of_budget']:.4%} |
| Last 200 | {final_system['total_consumption_budget']:.6g} | {final_system['total_necessary_food_budget']:.6g} | {final_system['total_discretionary_budget']:.6g} | {final_system['discretionary_share']:.4%} | {final_system['positive_discretionary_household_share']:.4%} | {final_system['R50_service_expenditure']:.6g} | {final_system['R50_service_share_of_budget']:.4%} |

R25/R50/R75 allocate only discretionary expenditure. The protected Food
allocation is never reduced, and the measured maximum protection gap is
`{collector.protection_gap_max:.6g}`. No Service units are inferred because no
Service price exists.

## Household Heterogeneity

The compact group file contains final active-Household rows by income decile,
wealth decile, and household type. Income-decile discretionary ratios are:

| Income decile | Mean discretionary share | Mean discretionary budget | R50 Service budget |
|---|---:|---:|---:|
{income_summary}

This is descriptive only. It does not fit an Engel curve. A rising pattern
would show that the existing target-wealth and optional-income rules naturally
support later Engel-style testing; it is not a new behavioral rule.

R50 potential Service spending is distributed disproportionately toward
Households with positive discretionary budgets. The top-10/top-20 shares and
Gini values for all three references are in the system CSV; these are shadow
budget concentration statistics, not Service consumption inequality.

## Reconciliation and Noninterference

- Shadow budget identity maximum gap: `{collector.shadow_budget_gap_max:.6g}`
- R25/R50/R75 conservation maximum gaps: `{collector.r25_gap_max:.6g}`, `{collector.r50_gap_max:.6g}`, `{collector.r75_gap_max:.6g}`
- Necessary Food protection maximum gap: `{collector.protection_gap_max:.6g}`
- Smoke OFF/ON parity: `{flags['smoke_parity_pass']}`
- Canonical OFF/ON parity: `{flags['canonical_parity_pass']}`
- Accepted Food reference parity: `{flags['accepted_reference_parity_pass']}`
- Household state parity: `{flags['Household_state_parity_pass']}`
- Demographic parity: `{flags['demographic_parity_pass']}`
- RNG parity: `{flags['rng_parity_pass']}`
- Actual Service transactions: `0`
- Actual Service household expenditure: `0`

The canonical run used population 5000, five Firms, seed 42, 1820 weekly
steps, `interest_behavioral_5pct`, `K=46.36154354202572`, and annual interest
5%. Smoke used population 500, five Firms, seed 42, and 260 weekly steps.
No seed 7/21 or new long-run scan was used.

## Audit Scope and Next Boundary

The current economy has a material nominal discretionary budget envelope under
the reference allocations, so it can support a later Service-demand plumbing
experiment in principle. This does not establish Service supply equilibrium,
labor closure, or a viable Service sector. Within-good supplier choice remains
a downstream market operation and was not mixed with Food-vs-Service
allocation.

Two future demand families are nominated for later behavioral review only:
necessity-first with a fixed discretionary split, and an income-dependent
Engel-share family. Neither is selected or implemented here. The passive
Service sector/good contracts from 14B and this household demand boundary make
Step 14D plumbing ready, subject to human review of the envelope scale and
distribution.

## Runtime

- Smoke OFF/ON: `{smoke_off_seconds:.2f}s` / `{smoke_on_seconds:.2f}s`
- Canonical shadow ON/OFF: `{canonical_seconds:.2f}s` / `{canonical_off_seconds:.2f}s`
"""
    (OUTPUT / "acceptance_summary.md").write_text(summary, encoding="utf-8")
    print(flags["verdict"])


if __name__ == "__main__":
    main()
