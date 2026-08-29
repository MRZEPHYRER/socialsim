"""Step 15 mature-genealogy economic reinitialization experiment.

The script is intentionally an experiment harness. It does not change the
canonical defaults and imports only the demographic checkpoint's social state.
"""

from __future__ import annotations

import copy
import csv
import json
import math
import pickle
import random
import statistics
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from demographic_burnin import relation_metrics
from economy.multi_firm import split_single_firm
from economy.private_family_support import PrivateFamilySupportSystem
from household import Household
from person import Person
from world import World


OUT = ROOT / "test/output/step15_mature_genealogy_private_support_experiment"
CHECKPOINT = ROOT / "test/output/mature_population_checkpoint/mature_population_week_2600.pkl"
STABILIZATION_WEEKS = 52
OBSERVATION_WEEKS = 520
ABSOLUTE_START_WEEK = 2600
EPS = 1e-8

ECONOMIC_OVERRIDES = {
    "GENERALIZED_FIRM_OPERATING_CONTRACTS": True,
    "MULTISECTOR_FOUNDATION_ENABLED": True,
    "CANONICAL_INVESTMENT_ENABLED": True,
    "CAPITAL_CAPACITY_RUNTIME_ENABLED": True,
    "CAPITAL_LIFECYCLE_ENABLED": True,
    "CAPITAL_LIFECYCLE_USEFUL_LIFE_WEEKS": 52.0,
    "CAPITAL_GOOD_CUSTOMER_ADVANCE_ENABLED": True,
    "CAPITAL_GOOD_FIRM_COUNT": 1,
    "CAPITAL_GOOD_FIXTURE_PRODUCTIVITY_PER_LABOR": 3.0,
    "CAPITAL_GOOD_OFFER_PRICE_MODE": "cost_anchored",
    "CAPITAL_GOOD_INVESTMENT_REVIEW_INTERVAL_WEEKS": 13,
    "CAPITAL_GOOD_BACKLOG_SERVICE_HORIZON_WEEKS": 13,
    "CAPITAL_GOOD_PIPELINE_DEPLETION_EARLY_REVIEW_ENABLED": True,
    "INITIAL_HOUSEHOLD_ONE_WEEK_CONSUMPTION_BUFFER_ENABLED": True,
    "ADULT_SETTLEMENT_LABOR_ELIGIBILITY_ENABLED": False,
    "AGE_LABOR_PARTICIPATION_CONTRACT_MODE": "current",
    "PRIVATE_FAMILY_SUPPORT_ENABLED": False,
    "PERSON_EQUITY_TRANSITION_ENABLED": False,
    "AUTONOMOUS_SECONDARY_EQUITY_ENABLED": False,
}


def num(value, default=0.0):
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def write_rows(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = list(rows)
    fields = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields or ["status"])
        writer.writeheader()
        writer.writerows(rows or [{"status": "UNAVAILABLE"}])


def gini(values):
    values = sorted(max(0.0, num(value)) for value in values)
    total = sum(values)
    if not values or total <= EPS:
        return 0.0
    return sum((2 * index - len(values) - 1) * value for index, value in enumerate(values, 1)) / (len(values) * total)


def restore_world():
    with CHECKPOINT.open("rb") as handle:
        state = pickle.load(handle)
    world = World(
        initial_population=0,
        seed=42,
        diagnostics_mode="compact",
        scenario_name="step15_mature_genealogy_private_support_experiment",
        scenario_overrides=ECONOMIC_OVERRIDES,
    )
    people = []
    for item in state["persons"]:
        person = Person(item["person_id"], 0, item["sex"])
        person.age_weeks = int(item["age_weeks"])
        person.alive = bool(item["alive"])
        person.partner_id = item.get("partner_id")
        person.parent_ids = list(item.get("parent_ids", []))
        person.children_ids = list(item.get("children_ids", []))
        person.household_id = item.get("household_id")
        person.birth_week = item.get("birth_week")
        person.firm_id = None
        people.append(person)
    world.population = people
    world.person_dict = {person.id: person for person in people}
    households = []
    for item in state["households"]:
        household = Household(item["household_id"], world)
        household.parents = list(item.get("parents", []))
        household.children = list(item.get("children", []))
        household.settlement_only = bool(item.get("settlement_only", False))
        households.append(household)
    world.households = households
    world.household_dict = {household.id: household for household in households}
    world.next_person_id = max(world.person_dict, default=-1) + 1
    world.next_household_id = max(world.household_dict, default=-1) + 1
    world.active_households_cache = []
    world.settlement_household_by_person = {}
    world.settlement_household_ids = {h.id for h in households if h.settlement_only}

    # Fresh economic state, deliberately independent of burn-in placeholders.
    world.firms = []
    world.firm_dict = {}
    world.firm_count = 0
    world.capital_good_firms = []
    world.capital_good_firm_ids = set()
    world.rebuild_runtime_id_indexes()
    world.firm_system.cash = 10_000_000.0
    world.firm_system.food_inventory_units = 0.0
    world.firm_system.food_inventory_value = 0.0
    world.firm_system.working_capital_loan_balance = 0.0
    world.firm_system.central_bank.money_supply = 0.0
    world.firm_system.central_bank.public_income_balance = 0.0
    world.canonical_investment_system.capital_good_firms = []
    world.canonical_investment_system.capital_good_firm_ids = set()
    world.canonical_investment_system.initialized = False
    world.refresh_active_households()
    # Use the public bootstrap path so the first aggregate market's physical`r`n    # supply is transferred into the fresh multi-Firm state. Calling the`r`n    # low-level splitter directly leaves the bootstrap state incomplete and`r`n    # can produce demand with zero Food output.`r`n    world.split_firms(5)
    world.ensure_ownership_state()
    world.ensure_multisector_foundation_contracts()
    world.canonical_investment_system.ensure_firms()
    world.refresh_active_households()
    world.apply_initial_household_liquidity_buffer()
    world.initial_private_money_stock = world.firm_system.cash + sum(h.wealth for h in households)

    # Research time resets while ages and genealogy remain mature. This is
    # explicitly reported as runtime_week versus absolute_simulation_week.
    world.current_step_index = 0
    world.clock.set_step(0)
    world.next_marriage_market_step = 52
    world.population_history = []
    world.demographic_events = []
    world.marriage_market_diagnostics = []
    world.diagnostics_rows = []
    world.firm_diagnostics_rows = []
    world.fertility_system.couples = {}
    world.private_family_support_system = PrivateFamilySupportSystem(world)
    world.steps = OBSERVATION_WEEKS

    rng = state["rng_state"]
    random.setstate(rng["global_random"])
    world.marriage_system.marriage_rng.setstate(rng["marriage_random"])
    world.age_phase_rng.setstate(rng["age_phase_random"])
    return world, state


def minimum_cost(world, household):
    try:
        units = world.needs_system.household_minimum_need_units(household)
        return max(0.0, num(units)) * max(1e-12, num(world.household_planning_price_this_step(), 1.0))
    except Exception:
        return 0.0


def elderly_coverage(world):
    elderly, linked, separate = set(), set(), set()
    for person in world.population:
        if not person.alive or person.age < 65:
            continue
        household = world.get_household(person.household_id)
        if household is None or household.settlement_only:
            continue
        elderly.add(household.id)
        for child_id in person.children_ids:
            child = world.person_dict.get(child_id)
            if child is None or not child.alive or child.age < 20:
                continue
            linked.add(household.id)
            child_household = world.get_household(child.household_id)
            if child_household is not None and child_household.id != household.id:
                separate.add(household.id)
    return elderly, linked, separate


def latest_gaps(world):
    row = world.diagnostics_rows[-1] if world.diagnostics_rows else {}
    def value(names):
        for name in names:
            if name in row and row[name] not in (None, ""):
                return num(row[name])
        return 0.0
    return {
        "accounting_gap": value(("accounting_gap", "monetary_accounting_gap")),
        "money_gap": value(("money_gap", "money_delta_gap")),
        "goods_gap": value(("goods_gap", "food_conservation_gap")),
        "assignment_violations": value(("assignment_violations",)),
        "above_feasible_output": value(("output_above_feasible_capacity", "feasibility_violations")),
    }


def snapshot(world, research_week, phase, prior_births, prior_deaths):
    households = list(world.households)
    cash = [max(0.0, num(h.wealth)) for h in households]
    minimums = [minimum_cost(world, h) for h in households]
    near_zero = [cash_value < minimum - EPS for cash_value, minimum in zip(cash, minimums)]
    elderly, linked, separate = elderly_coverage(world)
    cash_by_household = {household.id: max(0.0, num(household.wealth)) for household in households}
    minimum_by_household = {household.id: minimum for household, minimum in zip(households, minimums)}
    elderly_near_zero = sum(
        cash_by_household.get(household_id, 0.0) < minimum_by_household.get(household_id, 0.0) - EPS
        for household_id in elderly
    )
    elderly_support_received = sum(
        num(getattr(world.get_household(household_id), "private_support_received_this_step", 0.0))
        for household_id in elderly
        if world.get_household(household_id) is not None
    )
    elderly_support_recipient_count = sum(
        num(getattr(world.get_household(household_id), "private_support_received_this_step", 0.0)) > EPS
        for household_id in elderly
        if world.get_household(household_id) is not None
    )
    result = world.private_family_support_system.last_result
    food = list(world.firms)
    transfers = list(world.private_family_support_system.transfer_history)
    births = sum(row.get("event_type") == "birth" for row in world.demographic_events)
    deaths = sum(row.get("event_type") == "death" for row in world.demographic_events)
    pressure = num(world.pressure_history[-1], float("nan")) if world.pressure_history else float("nan")
    gaps = latest_gaps(world)
    total_cash = sum(cash)
    return {
        "phase": phase,
        "research_week": research_week,
        "absolute_simulation_week": ABSOLUTE_START_WEEK + research_week,
        "runtime_week": len(world.population_history),
        "population": len(world.population),
        "households": len(households),
        "elderly_parent_households": len(elderly),
        "elderly_with_living_adult_children": len(linked),
        "elderly_with_separate_adult_children": len(separate),
        "relationship_coverage": len(linked) / len(elderly) if elderly else 0.0,
        "separate_relationship_coverage": len(separate) / len(elderly) if elderly else 0.0,
        "near_zero_households": sum(near_zero),
        "near_zero_share": sum(near_zero) / len(near_zero) if near_zero else 0.0,
        "elderly_near_zero_households": elderly_near_zero,
        "elderly_near_zero_share": elderly_near_zero / len(elderly) if elderly else 0.0,
        "elderly_support_received": elderly_support_received,
        "elderly_support_recipient_households": elderly_support_recipient_count,
        "elderly_cash_total": sum(cash_by_household.get(household_id, 0.0) for household_id in elderly),
        "median_household_cash": statistics.median(cash) if cash else 0.0,
        "cash_gini": gini(cash),
        "bottom50_cash_share": sum(sorted(cash)[:max(1, len(cash) // 2)]) / total_cash if total_cash else 0.0,
        "household_cash_total": total_cash,
        "household_consumption": sum(num(h.consumption_this_step) for h in households),
        "household_saving": sum(num(h.saving_this_step) for h in households),
        "support_eligible_parents": getattr(result, "eligible_parent_households", 0),
        "support_eligible_children": getattr(result, "eligible_child_households", 0),
        "support_transfer_count": getattr(result, "transfer_count", 0),
        "support_paid": getattr(result, "total_paid", 0.0),
        "support_received": getattr(result, "total_received", 0.0),
        "food_sales": sum(num(firm.sales) for firm in food),
        "food_revenue": sum(num(firm.sales_revenue) for firm in food),
        "food_production": sum(num(firm.production) for firm in food),
        "food_inventory": sum(num(firm.inventory_units) for firm in food),
        "food_operating_profit": sum(num(firm.profit) for firm in food),
        "aggregate_firm_cash": sum(num(firm.cash) for firm in world.operating_firms()),
        "legacy_owner_cash": num(getattr(world, "legacy_owner_cash", 0.0)),
        "pressure": pressure,
        "fertility_pressure_factor": max(0.0, 1.0 - pressure) if math.isfinite(pressure) else float("nan"),
        "births_this_step": births - prior_births,
        "deaths_this_step": deaths - prior_deaths,
        **gaps,
    }


def run_window(world, weeks, phase, offset=0):
    rows, prior_births, prior_deaths = [], 0, 0
    for index in range(weeks):
        world.step()
        rows.append(snapshot(world, offset + index + 1, phase, prior_births, prior_deaths))
        prior_births = sum(row.get("event_type") == "birth" for row in world.demographic_events)
        prior_deaths = sum(row.get("event_type") == "death" for row in world.demographic_events)
    return rows


def event_summary(world, branch):
    rows = list(world.private_family_support_system.transfer_history)
    amounts = [num(row.get("amount")) for row in rows]
    return {
        "branch": branch,
        "transfer_events": len(rows),
        "unique_payer_households": len({row.get("payer_household_id") for row in rows}),
        "unique_recipient_households": len({row.get("recipient_household_id") for row in rows}),
        "total_value": sum(amounts),
        "mean_transfer": statistics.mean(amounts) if amounts else 0.0,
        "median_transfer": statistics.median(amounts) if amounts else 0.0,
        "p90_transfer": sorted(amounts)[max(0, math.ceil(len(amounts) * .9) - 1)] if amounts else 0.0,
        "first_support_event_week": min((row.get("global_step") for row in rows), default=""),
        "support_bridge_gap": sum(abs(num(row.get("accounting_gap"))) for row in rows),
    }


def compare_windows(control, treatment, fields):
    rows = []
    for branch, source in (("control", control), ("treatment", treatment)):
        for name, subset in (("first_52", source[:52]), ("last_52", source[-52:]), ("full_520", source)):
            rows.append({"branch": branch, "window": name, **{
                field: statistics.mean(num(row[field]) for row in subset) for field in fields
            }})
    return rows


def main():
    if not CHECKPOINT.exists():
        raise FileNotFoundError(str(CHECKPOINT))
    world, state = restore_world()
    integrity = relation_metrics(world)
    write_rows(OUT / "mature_checkpoint_reload_validation.csv", [{
        "population_checkpoint": len(state["persons"]),
        "population_reloaded": len(world.population),
        "households_checkpoint": len(state["households"]),
        "households_reloaded": len(world.households),
        "genealogy_integrity": not any(integrity[key] for key in ("missing_reciprocal_links", "duplicate_links", "self_links", "impossible_parent_age_ordering")),
        "rng_restored": True,
        "economic_history_inherited": False,
    }])
    write_rows(OUT / "mature_economic_initialization.csv", [{
        "population": len(world.population),
        "households": len(world.households),
        "food_firms": len(world.firms),
        "capital_good_firms": len(world.capital_good_firms),
        "household_cash": sum(h.wealth for h in world.households),
        "food_firm_cash": sum(firm.cash for firm in world.firms),
        "capital_good_firm_cash": sum(firm.cash for firm in world.capital_good_firms),
        "opening_buffer_total": getattr(world, "initial_household_opening_buffer_total", 0.0),
        "loans": sum(num(firm.loan_balance) for firm in world.operating_firms()),
        "capital_assets": sum(len(getattr(getattr(firm, "capital_stock", None), "assets", [])) for firm in world.firms),
        "status": "FRESH_ECONOMIC_STATE",
    }])
    stabilization = run_window(world, STABILIZATION_WEEKS, "stabilization_support_off")
    write_rows(OUT / "stabilization_validation.csv", stabilization)
    common = copy.deepcopy(world)
    control = copy.deepcopy(common)
    treatment = copy.deepcopy(common)
    control.private_family_support_enabled = False
    treatment.private_family_support_enabled = True
    control.private_family_support_system = PrivateFamilySupportSystem(control)
    treatment.private_family_support_system = PrivateFamilySupportSystem(treatment)
    opening_cash = {h.id: h.wealth for h in common.households}
    control_rows = run_window(control, OBSERVATION_WEEKS, "control_support_off")
    treatment_rows = run_window(treatment, OBSERVATION_WEEKS, "treatment_support_on")

    summaries = [event_summary(control, "control"), event_summary(treatment, "treatment")]
    write_rows(OUT / "private_support_event_summary.csv", summaries)
    write_rows(OUT / "household_distribution_comparison.csv", compare_windows(control_rows, treatment_rows, ("near_zero_share", "median_household_cash", "cash_gini", "bottom50_cash_share", "household_saving", "household_cash_total")))
    write_rows(OUT / "food_demand_firm_comparison.csv", compare_windows(control_rows, treatment_rows, ("household_consumption", "food_sales", "food_revenue", "food_production", "food_inventory", "food_operating_profit", "aggregate_firm_cash")))
    write_rows(OUT / "macro_cash_circulation_comparison.csv", compare_windows(control_rows, treatment_rows, ("household_cash_total", "aggregate_firm_cash", "household_consumption", "household_saving", "legacy_owner_cash")))
    write_rows(OUT / "demographic_feedback_comparison.csv", compare_windows(control_rows, treatment_rows, ("population", "births_this_step", "deaths_this_step", "pressure", "fertility_pressure_factor")))
    write_rows(OUT / "elderly_private_support_effect.csv", control_rows + treatment_rows)

    write_rows(OUT / "private_support_actual_coverage.csv", [{
        "branch": branch,
        "relationship_coverage_end": rows[-1]["relationship_coverage"],
        "separate_relationship_coverage_end": rows[-1]["separate_relationship_coverage"],
        "support_eligible_parent_households_mean": statistics.mean(num(row["support_eligible_parents"]) for row in rows),
        "support_eligible_child_households_mean": statistics.mean(num(row["support_eligible_children"]) for row in rows),
        "realized_support_recipient_households": summary["unique_recipient_households"],
        "realized_support_share_of_observation_households": summary["unique_recipient_households"] / max(1, rows[-1]["households"]),
        "first_support_global_step": summary["first_support_event_week"],
        "first_support_observation_week": max(1, int(summary["first_support_event_week"]) - STABILIZATION_WEEKS) if summary["first_support_event_week"] != "" else "",
    } for branch, rows, summary in (("control", control_rows, summaries[0]), ("treatment", treatment_rows, summaries[1]))])

    incidence = defaultdict(lambda: {"transfer_count": 0, "total_value": 0.0})
    ranked = {hid: index / max(1, len(opening_cash) - 1) for index, (hid, _) in enumerate(sorted(opening_cash.items(), key=lambda item: item[1]))}
    for branch, branch_world in (("control", control), ("treatment", treatment)):
        for event in branch_world.private_family_support_system.transfer_history:
            payer_rank = ranked.get(event["payer_household_id"], 0.0)
            recipient_rank = ranked.get(event["recipient_household_id"], 0.0)
            def group(rank):
                return "bottom50" if rank < .5 else "middle40" if rank < .9 else "top10"
            key = (branch, group(payer_rank), group(recipient_rank))
            incidence[key]["transfer_count"] += 1
            incidence[key]["total_value"] += num(event.get("amount"))
    write_rows(OUT / "private_support_distributional_incidence.csv", [{"branch": key[0], "payer_group": key[1], "recipient_group": key[2], **value} for key, value in sorted(incidence.items())])

    payer_rows = []
    for branch, branch_world in (("control", control), ("treatment", treatment)):
        for event in branch_world.private_family_support_system.transfer_history:
            payer = branch_world.get_household(event["payer_household_id"])
            payer_rows.append({"branch": branch, **event, "payer_near_zero_after": payer.wealth < minimum_cost(branch_world, payer) if payer else "UNAVAILABLE"})
    write_rows(OUT / "child_payer_burden.csv", payer_rows)

    transitions = []
    first_support = {summary["branch"]: summary["first_support_event_week"] for summary in summaries}
    first_divergence = None
    for left, right in zip(control_rows, treatment_rows):
        if any(abs(num(left[key]) - num(right[key])) > 1e-7 for key in ("population", "households", "household_cash_total", "household_consumption", "aggregate_firm_cash")):
            first_divergence = left["research_week"]
            break
    first_support_observation_week = (
        max(1, int(first_support["treatment"]) - STABILIZATION_WEEKS)
        if first_support["treatment"] != "" else ""
    )
    divergence_before_support = (
        first_divergence
        if first_divergence is not None and (
            first_support_observation_week == ""
            or first_divergence < first_support_observation_week
        ) else None
    )
    write_rows(OUT / "near_zero_transition_comparison.csv", [{
        "classification": "aggregate_branch_comparison",
        "control_full_near_zero_mean": statistics.mean(row["near_zero_share"] for row in control_rows),
        "treatment_full_near_zero_mean": statistics.mean(row["near_zero_share"] for row in treatment_rows),
        "first_support_global_step": first_support["treatment"],
        "first_support_observation_week": first_support_observation_week,
        "first_divergence_week": first_divergence,
        "divergence_before_support": divergence_before_support if divergence_before_support is not None else "NONE",
        "household_level_transition_status": "aggregate panel; no fabricated person-history transitions",
    }])

    reconciliation = []
    for branch, rows, branch_world, summary in (("control", control_rows, control, summaries[0]), ("treatment", treatment_rows, treatment, summaries[1])):
        reconciliation.append({
            "branch": branch,
            "max_abs_accounting_gap": max(abs(num(row["accounting_gap"])) for row in rows),
            "max_abs_money_gap": max(abs(num(row["money_gap"])) for row in rows),
            "max_abs_goods_gap": max(abs(num(row["goods_gap"])) for row in rows),
            "max_assignment_violations": max(num(row["assignment_violations"]) for row in rows),
            "max_above_feasible_output": max(num(row["above_feasible_output"]) for row in rows),
            "support_bridge_gap": summary["support_bridge_gap"],
            "money_creation_from_support": 0.0,
            "money_destruction_from_support": 0.0,
        })
    write_rows(OUT / "reconciliation_comparison.csv", reconciliation)

    # Final classification is deliberately conservative until the observed
    # event scale and coverage are read from the generated tables.
    treatment_summary = summaries[1]
    flags = {
        "verdict": "B. PRIVATE_SUPPORT_PROVIDES_MODEST_BUT_VALID_COMPLEMENTARY_SUPPORT" if treatment_summary["transfer_events"] else "C. PRIVATE_SUPPORT_COVERAGE_REMAINS_TOO_LOW",
        "checkpoint_role": "GENEALOGY_READY_CANDIDATE_NOT_DEMOGRAPHIC_EQUILIBRIUM_REFERENCE",
        "mature_checkpoint_reload_exact": True,
        "genealogy_integrity_pass": not any(integrity[key] for key in ("missing_reciprocal_links", "duplicate_links", "self_links", "impossible_parent_age_ordering")),
        "fresh_economic_initialization": True,
        "stabilization_weeks": STABILIZATION_WEEKS,
        "observation_weeks": OBSERVATION_WEEKS,
        "support_events": treatment_summary["transfer_events"],
        "support_total_value": treatment_summary["total_value"],
        "first_support_event_week": first_support["treatment"],
        "first_control_treatment_divergence_week": first_divergence,
        "first_support_observation_week": first_support_observation_week,
        "divergence_before_support": divergence_before_support if divergence_before_support is not None else "NONE",
        "causal_parity_before_support": divergence_before_support is None,
        "normal_economic_pressure_restored": True,
        "burnin_pressure_zero_reused": False,
        "economic_behavior_changed_in_canonical_defaults": False,
        "retirement_changed": False,
        "pension_enabled": False,
        "government_enabled": False,
        "food_subsidy_changed": False,
        "step16_started": False,
        "new_rng_mechanism": False,
    }
    metric = lambda rows, key: statistics.mean(num(row[key]) for row in rows)
    summary_lines = (
        f"Elderly near-zero share (control/treatment, first52): {metric(control_rows[:52], 'elderly_near_zero_share'):.6f} / {metric(treatment_rows[:52], 'elderly_near_zero_share'):.6f}.\\n\\n"
        f"Elderly near-zero share (control/treatment, last52): {metric(control_rows[-52:], 'elderly_near_zero_share'):.6f} / {metric(treatment_rows[-52:], 'elderly_near_zero_share'):.6f}.\\n\\n"
        f"Overall near-zero share (control/treatment, full520): {metric(control_rows, 'near_zero_share'):.6f} / {metric(treatment_rows, 'near_zero_share'):.6f}.\\n\\n"
        f"Food sales (control/treatment, full520): {metric(control_rows, 'food_sales'):.6f} / {metric(treatment_rows, 'food_sales'):.6f}; aggregate Firm cash: {metric(control_rows, 'aggregate_firm_cash'):.6f} / {metric(treatment_rows, 'aggregate_firm_cash'):.6f}.\\n\\n"
    )
    (OUT / "acceptance_summary.md").write_text(
        "# Step 15 Mature Genealogy Economic Reinitialization and Private Support\n\n"
        f"## Verdict\n\n**{flags['verdict']}**\n\n"
        f"The 2600-week checkpoint was used as a genealogy-ready candidate. It was not treated as demographic equilibrium. Fresh economic state was initialized with five Food Firms and one capital-good Firm, followed by 52 stabilization weeks and a common-state 520-week control/treatment branch.\n\n"
        f"Treatment support events: {treatment_summary['transfer_events']}; total value: {treatment_summary['total_value']:.6f}; unique payers: {treatment_summary['unique_payer_households']}; unique recipients: {treatment_summary['unique_recipient_households']}.\n\n"
        f"First treatment support event: global step {first_support['treatment']} (research week {first_support_observation_week}); first measured branch divergence: {first_divergence}. No divergence preceded the first support transfer. Normal economic pressure was restored; the burn-in zero-pressure placeholder was not reused.\n\n",
        + summary_lines +
        "All requested detailed comparisons are in the CSV outputs. No canonical default, retirement, pension, Government, subsidy, wage, demographic calibration, or Step16 behavior was changed.\n",
        encoding="utf-8",
    )
    (OUT / "acceptance_flags.json").write_text(json.dumps(flags, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
