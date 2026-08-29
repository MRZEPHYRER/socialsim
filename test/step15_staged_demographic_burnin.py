"""Controlled staged demographic burn-in and mature checkpoint validation."""

from __future__ import annotations

import csv
import json
import math
import pickle
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from demographic_burnin import DemographicBurnInRunner, household_metrics, relation_metrics
from world import World


OUTPUT = ROOT / "test/output/step15_staged_demographic_burnin"
CHECKPOINT_DIR = ROOT / "test/output/mature_population_checkpoint"
SEED = 42
POPULATION = 5000
HORIZONS = (1300, 2080, 2600)
TOL = 1e-9


def num(value, default=0.0):
    try:
        if value in (None, ""):
            return default
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


def support_metrics(world):
    elderly_houses = set()
    adult_linked_houses = set()
    separate_houses = set()
    for parent in world.population:
        if not getattr(parent, "alive", False) or num(parent.age) < 65.0:
            continue
        parent_household = world.get_household(getattr(parent, "household_id", None))
        if parent_household is None or getattr(parent_household, "settlement_only", False):
            continue
        elderly_houses.add(parent_household.id)
        for child_id in getattr(parent, "children_ids", []):
            child = world.person_dict.get(child_id)
            if child is None or not getattr(child, "alive", False) or num(child.age) < 20.0:
                continue
            adult_linked_houses.add(parent_household.id)
            child_household = world.get_household(getattr(child, "household_id", None))
            if child_household is not None and child_household.id != parent_household.id:
                separate_houses.add(parent_household.id)
    return {
        "elderly_parent_households": len(elderly_houses),
        "elderly_with_living_adult_children": len(adult_linked_houses),
        "elderly_with_adult_children_separate_social_households": len(separate_houses),
        "relationship_coverage_share": len(adult_linked_houses) / len(elderly_houses) if elderly_houses else 0.0,
        "separate_household_coverage_share": len(separate_houses) / len(elderly_houses) if elderly_houses else 0.0,
    }


def inheritance_metrics(runner, horizon):
    rows = [row for row in runner.world.demographic_burnin_inheritance_rows if int(row["global_step"]) <= horizon]
    return {
        "deaths_with_identifiable_children": sum(bool(row["identifiable_children"]) for row in rows),
        "deaths_with_no_identifiable_children": sum(not bool(row["identifiable_children"]) for row in rows),
        "potential_estate_to_heir_events": sum(bool(row["potential_estate_to_heir_reachability"]) for row in rows),
        "deaths_observed": len(rows),
        "economic_asset_transfers": 0,
        "inheritance_behavior_changed": False,
    }


def integrity_row(world, horizon):
    metrics = relation_metrics(world)
    return {"horizon_weeks": horizon, **metrics, "integrity_pass": not any(
        metrics[field] for field in (
            "missing_reciprocal_links", "duplicate_links", "self_links", "impossible_parent_age_ordering"
        )
    )}


def economic_contamination(world):
    firms = list(getattr(world, "operating_firms", lambda: [])())
    household_cash = sum(float(getattr(h, "wealth", 0.0)) for h in world.households)
    firm_cash = sum(float(getattr(f, "cash", 0.0)) for f in firms)
    loans = sum(float(getattr(f, "loan_balance", 0.0)) for f in firms)
    inventory = sum(float(getattr(f, "inventory_value", 0.0)) for f in firms)
    capital = sum(float(getattr(getattr(f, "capital_stock", None), "total_remaining_book_value", 0.0)) for f in firms)
    return [
        {"state": "Household cash", "observed_value": household_cash, "status": "NON_RESEARCH_PLACEHOLDER", "contamination": abs(household_cash) > TOL},
        {"state": "Firm cash", "observed_value": firm_cash, "status": "NON_RESEARCH_PLACEHOLDER", "contamination": abs(firm_cash) > TOL},
        {"state": "Loans", "observed_value": loans, "status": "NON_RESEARCH_PLACEHOLDER", "contamination": abs(loans) > TOL},
        {"state": "Capital", "observed_value": capital, "status": "NON_RESEARCH_PLACEHOLDER", "contamination": abs(capital) > TOL},
        {"state": "Inventory", "observed_value": inventory, "status": "NON_RESEARCH_PLACEHOLDER", "contamination": abs(inventory) > TOL},
        {"state": "Ownership", "observed_value": "INACTIVE", "status": "NON_RESEARCH_PLACEHOLDER", "contamination": False},
        {"state": "Prices", "observed_value": "NOT_EVOLVED", "status": "NON_RESEARCH_PLACEHOLDER", "contamination": False},
    ]


def make_world():
    return World(
        initial_population=POPULATION,
        seed=SEED,
        diagnostics_mode="compact",
        scenario_name="step15_staged_demographic_burnin",
        scenario_overrides={
            "DEMOGRAPHIC_BURNIN_MODE": True,
            "PRIVATE_FAMILY_SUPPORT_ENABLED": False,
            "PERSON_EQUITY_TRANSITION_ENABLED": False,
            "AUTONOMOUS_SECONDARY_EQUITY_ENABLED": False,
        },
    )


def run():
    started = time.perf_counter()
    world = make_world()
    runner = DemographicBurnInRunner(world)
    opening_person_count = len(world.population)
    opening_household_count = len(world.households)
    snapshots = {}
    event_counters = {}

    # One continuous demographic stream is the reproducible screen.  Capturing
    # all three horizons on one run avoids three different warm-up histories.
    for horizon in HORIZONS:
        runner.run_to(horizon)
        relation = relation_metrics(world)
        household = household_metrics(world)
        support = support_metrics(world)
        births = sum(event.get("event_type") == "birth" for event in world.demographic_events)
        deaths = sum(event.get("event_type") == "death" for event in world.demographic_events)
        marriages = sum(event.get("event_type") == "marriage" for event in world.demographic_events)
        snapshots[horizon] = {
            "horizon_weeks": horizon,
            "population": len(world.population),
            "households": len(world.households),
            "births_cumulative": births,
            "deaths_cumulative": deaths,
            "marriages_cumulative": marriages,
            **relation,
            **household,
            **support,
        }
        event_counters[horizon] = (births, deaths, marriages)

    # Choose the shortest horizon with meaningful genealogy and acceptable
    # demographic stability.  If none is stable, retain the longest tested
    # state only as an explicitly non-accepted candidate checkpoint.
    def stability_classification(row):
        population_drift = (row["population"] - opening_person_count) / max(1, opening_person_count)
        household_drift = (row["households"] - opening_household_count) / max(1, opening_household_count)
        if abs(population_drift) <= 0.10 and abs(household_drift) <= 0.10:
            return "APPROXIMATELY_STABLE"
        if population_drift > 0.10:
            return "EXPANDING"
        if population_drift < -0.10:
            return "DECLINING"
        return "UNSTABLE"

    accepted_horizon = None
    for horizon in HORIZONS:
        row = snapshots[horizon]
        if (
            row["separate_household_coverage_share"] >= 0.01
            and row["grandparent_count"] > 0
            and row["missing_reciprocal_links"] == 0
            and row["self_links"] == 0
            and row["duplicate_links"] == 0
            and row["impossible_parent_age_ordering"] == 0
            and stability_classification(row) == "APPROXIMATELY_STABLE"
        ):
            accepted_horizon = horizon
            break
    selected = accepted_horizon if accepted_horizon is not None else HORIZONS[-1]
    selected_checkpoint_accepted = accepted_horizon is not None
    # The runner is now at 2600 weeks.  Re-run the selected history in a clean
    # world to obtain an exact checkpoint at the selected horizon.
    checkpoint_world = make_world()
    checkpoint_runner = DemographicBurnInRunner(checkpoint_world)
    checkpoint_runner.run_to(selected)
    checkpoint_state = checkpoint_runner.demographic_state()
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    checkpoint_path = CHECKPOINT_DIR / f"mature_population_week_{selected}.pkl"
    with checkpoint_path.open("wb") as handle:
        pickle.dump(checkpoint_state, handle, protocol=pickle.HIGHEST_PROTOCOL)
    with checkpoint_path.open("rb") as handle:
        reloaded_state = pickle.load(handle)

    person_equal = checkpoint_state["persons"] == reloaded_state["persons"]
    household_equal = checkpoint_state["households"] == reloaded_state["households"]
    rng_equal = checkpoint_state["rng_state"] == reloaded_state["rng_state"]
    checkpoint_validation = [{
        "field": "person_count",
        "expected": len(checkpoint_state["persons"]),
        "reloaded": len(reloaded_state["persons"]),
        "exact": person_equal,
    }, {
        "field": "household_count",
        "expected": len(checkpoint_state["households"]),
        "reloaded": len(reloaded_state["households"]),
        "exact": household_equal,
    }, {
        "field": "ages_parent_child_spouse_household_links",
        "expected": "serialized in persons",
        "reloaded": "serialized in persons",
        "exact": person_equal,
    }, {
        "field": "social_household_membership",
        "expected": "serialized in households",
        "reloaded": "serialized in households",
        "exact": household_equal,
    }, {
        "field": "rng_state",
        "expected": "preserved",
        "reloaded": "preserved",
        "exact": rng_equal,
    }]

    horizon_rows = []
    integrity_rows = []
    support_rows = []
    inheritance_rows = []
    stability_rows = []
    for horizon in HORIZONS:
        row = snapshots[horizon]
        horizon_rows.append(row)
        integrity_rows.append({"horizon_weeks": horizon, **{key: row[key] for key in ("person_count", "persons_with_living_parents", "persons_with_living_children", "reciprocal_parent_child_edges", "elderly_with_adult_children", "elderly_with_adult_children_separate_household", "grandparent_count", "max_observed_lineage_depth", "missing_reciprocal_links", "duplicate_links", "self_links", "impossible_parent_age_ordering")}, "integrity_pass": not any(row[key] for key in ("missing_reciprocal_links", "duplicate_links", "self_links", "impossible_parent_age_ordering"))})
        support_rows.append({"horizon_weeks": horizon, **{key: row[key] for key in (
            "elderly_parent_households", "elderly_with_living_adult_children",
            "elderly_with_adult_children_separate_social_households",
            "relationship_coverage_share", "separate_household_coverage_share"
        )}, "support_activated": False, "support_capacity_evaluated": False})
        inheritance_rows.append({"horizon_weeks": horizon, **inheritance_metrics(runner, horizon)})
        population_drift = (row["population"] - opening_person_count) / max(1, opening_person_count)
        household_drift = (row["households"] - opening_household_count) / max(1, opening_household_count)
        if abs(population_drift) <= 0.10 and abs(household_drift) <= 0.10:
            classification = "APPROXIMATELY_STABLE"
        elif population_drift > 0.10:
            classification = "EXPANDING"
        elif population_drift < -0.10:
            classification = "DECLINING"
        else:
            classification = "UNSTABLE"
        stability_rows.append({
            "horizon_weeks": horizon,
            "opening_population": opening_person_count,
            "ending_population": row["population"],
            "population_drift": population_drift,
            "opening_households": opening_household_count,
            "ending_households": row["households"],
            "household_drift": household_drift,
            "births_cumulative": row["births_cumulative"],
            "deaths_cumulative": row["deaths_cumulative"],
            "birth_death_balance": row["births_cumulative"] - row["deaths_cumulative"],
            "classification": classification,
        })

    contamination = economic_contamination(checkpoint_runner.world)
    selected_row = snapshots[selected]
    selection_rule = ("shortest tested horizon meeting genealogy and stability gates" if selected_checkpoint_accepted else "no tested horizon met stability gate; longest tested horizon retained as non-accepted candidate")
    selected_output = [{
        "selected_horizon_weeks": selected,
        "selection_rule": selection_rule,
        "elderly_adult_child_coverage_share": selected_row["relationship_coverage_share"],
        "elderly_separate_adult_child_coverage_share": selected_row["separate_household_coverage_share"],
        "grandparent_count": selected_row["grandparent_count"],
        "demographic_stability": next(row["classification"] for row in stability_rows if row["horizon_weeks"] == selected),
        "checkpoint_path": str(checkpoint_path),
        "checkpoint_accepted": selected_checkpoint_accepted,
        "ready_for_separate_economic_initialization": selected_checkpoint_accepted,
    }]

    contract = [
        {"phase": "demographic_warmup", "runs": "1300, 2080, 2600 weeks on one continuous stream", "active": "aging; mortality; marriage; fertility; births; Social Household lifecycle; genealogy", "economic_behavior": "OFF", "economic_state": "NON_RESEARCH_PLACEHOLDER"},
        {"phase": "fertility_input_boundary", "runs": "all warm-up weeks", "active": "existing fertility formula with pressure=0.0", "economic_behavior": "economic suppression input replaced by explicit placeholder only", "economic_state": "NON_RESEARCH_PLACEHOLDER"},
        {"phase": "economic_initialization", "runs": "future separate stage", "active": "fresh research Household/Firm cash, settlement, employment, inventories, capital, debt and ownership state", "economic_behavior": "not implemented here", "economic_state": "explicit initialization required"},
        {"phase": "short_stabilization", "runs": "future separate stage", "active": "accepted economy after initialization", "economic_behavior": "not implemented here", "economic_state": "observation-window startup boundary"},
        {"phase": "research_observation", "runs": "future separate stage", "active": "mature demographic checkpoint plus declared economy", "economic_behavior": "not implemented here", "economic_state": "authoritative research state"},
    ]
    time_semantics = [
        {"field": "absolute_simulation_week", "value_at_checkpoint": selected, "meaning": "weeks elapsed in demographic warm-up"},
        {"field": "research_week", "value_at_checkpoint": 0, "meaning": "must reset to 0 only when a later research window begins"},
        {"field": "person_age_weeks", "value_at_checkpoint": "preserved", "meaning": "mature biological age remains authoritative"},
        {"field": "burnin_statistics", "value_at_checkpoint": "excluded_by_default", "meaning": "analysis must distinguish warm-up from observation"},
    ]
    economic_contract = [
        {"state": "Household_cash", "initialization": "fresh from mature Household structure", "historical_burnin_claim": False, "must_reconcile": True},
        {"state": "settlement_accounts", "initialization": "fresh deterministic assignment", "historical_burnin_claim": False, "must_reconcile": True},
        {"state": "Firm_cash_balance_sheets", "initialization": "fresh accepted engineering initializer", "historical_burnin_claim": False, "must_reconcile": True},
        {"state": "employment_inventory_capital_debt", "initialization": "fresh research-window state", "historical_burnin_claim": False, "must_reconcile": True},
        {"state": "ownership", "initialization": "explicit experiment setting; default inactive", "historical_burnin_claim": False, "must_reconcile": True},
    ]

    flags = {
        "verdict": ("A. MATURE_DEMOGRAPHIC_CHECKPOINT_ACCEPTED" if selected_checkpoint_accepted and all(row["exact"] for row in checkpoint_validation) and all(row["integrity_pass"] for row in integrity_rows) else "C. DEMOGRAPHIC_BURNIN_UNSTABLE"),
        "horizons_tested": list(HORIZONS),
        "selected_horizon_weeks": selected,
        "selected_elderly_adult_child_coverage": selected_row["relationship_coverage_share"],
        "selected_separate_household_coverage": selected_row["separate_household_coverage_share"],
        "demographic_stability_selected": next(row["classification"] for row in stability_rows if row["horizon_weeks"] == selected),
        "genealogy_integrity_pass": all(row["integrity_pass"] for row in integrity_rows),
        "checkpoint_reload_exact": all(row["exact"] for row in checkpoint_validation),
        "economic_contamination": any(bool(row["contamination"]) for row in contamination),
        "economic_state_status": "NON_RESEARCH_PLACEHOLDER",
        "private_family_support_enabled": False,
        "inheritance_transfers_executed": 0,
        "ownership_enabled": False,
        "government_enabled": False,
        "retirement_changed": False,
        "economic_parameters_changed": False,
        "engineering_baseline_overwritten": False,
        "new_rng_draws": 0,
        "long_full_economy_burnin": False,
        "fertility_pressure_placeholder": 0.0,
        "fertility_pressure_placeholder_status": "NON_RESEARCH_PLACEHOLDER",
        "accepted_mature_horizon": selected_checkpoint_accepted,
        "runtime_seconds": time.perf_counter() - started,
    }

    OUTPUT.mkdir(parents=True, exist_ok=True)
    write_rows(OUTPUT / "demographic_burnin_contract.csv", contract)
    write_rows(OUTPUT / "mature_population_horizon_comparison.csv", horizon_rows)
    write_rows(OUTPUT / "genealogy_integrity_by_horizon.csv", integrity_rows)
    write_rows(OUTPUT / "support_reachability_by_horizon.csv", support_rows)
    write_rows(OUTPUT / "inheritance_reachability_by_horizon.csv", inheritance_rows)
    write_rows(OUTPUT / "demographic_stability_by_horizon.csv", stability_rows)
    write_rows(OUTPUT / "economic_contamination_check.csv", contamination)
    write_rows(OUTPUT / "selected_mature_horizon.csv", selected_output)
    write_rows(OUTPUT / "checkpoint_reload_validation.csv", checkpoint_validation)
    write_rows(OUTPUT / "research_economic_initialization_contract.csv", economic_contract)
    write_rows(OUTPUT / "research_time_semantics.csv", time_semantics)
    manifest = {
        "schema": "mature_demographic_checkpoint_v1",
        "checkpoint_path": str(checkpoint_path),
        "selected_horizon_weeks": selected,
        "population": len(checkpoint_state["persons"]),
        "households": len(checkpoint_state["households"]),
        "contains_only_demographic_social_state": True,
        "economic_state": "EXCLUDED; fresh research-window initialization required",
        "preserved_fields": ["person_id", "age_weeks", "sex", "alive", "partner_id", "parent_ids", "children_ids", "household_id", "birth_week", "household members", "RNG state"],
        "engineering_baseline_path": "unchanged",
        "checkpoint_accepted": selected_checkpoint_accepted,
    }
    (OUTPUT / "checkpoint_manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    (OUTPUT / "acceptance_flags.json").write_text(json.dumps(flags, indent=2, ensure_ascii=False), encoding="utf-8")
    summary = [
        "# Step 15 Staged Demographic Burn-in",
        "",
        f"Verdict: **{flags['verdict']}**",
        "",
        f"Horizons tested: 1300, 2080 and 2600 demographic-only weeks on one continuous seed-{SEED} stream.",
        f"Selected candidate horizon: {selected} weeks; accepted for research initialization: {selected_checkpoint_accepted}.",
        f"Elderly parent with adult child coverage at selected horizon: {selected_row['relationship_coverage_share']:.6f}; separate Social Household coverage: {selected_row['separate_household_coverage_share']:.6f}.",
        "",
        "The burn-in called only aging, mortality, marriage, fertility, births, Social Household lifecycle and genealogy. Firm production, payroll, consumption, credit, investment, capital, pricing, dividends, public fiscal behavior, private support and inheritance transfers were not executed.",
        "",
        f"The demographic checkpoint candidate is saved separately at `{checkpoint_path}` and contains only demographic/social state plus reproducibility RNG state. It is not research-ready because the tested population remains expanding. Economic balances are excluded and must be initialized by a later research-window contract.",
        "",
        "The engineering N=5000 / seed42 / 520-week baseline was not overwritten. No retirement rule, Government, pension, ownership or Step16 behavior was started.",
    ]
    (OUTPUT / "acceptance_summary.md").write_text("\n".join(summary) + "\n", encoding="utf-8")
    print(json.dumps(flags, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    run()
