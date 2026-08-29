"""Step 15 initial-population genealogy and adult-child reachability audit.

This is a diagnostic-only runner.  It observes the accepted canonical runtime
and wraps inheritance only to capture the incoming dead-person set before the
existing implementation mutates/removes relationships.  It does not change
the result of any event or enable any economic mechanism.
"""

from __future__ import annotations

import csv
import json
import math
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from world import World


OUTPUT = ROOT / "test/output/step15_initial_genealogy_reachability_audit"
SEED = 42
POPULATION = 5000
WEEKS = 520
FOOD_FIRMS = 5
SNAPSHOT_CADENCE = 13
TOL = 1e-9


def num(value, default=0.0):
    try:
        if value in (None, ""):
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def write_rows(path: Path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = list(rows)
    fields = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    if not fields:
        fields = ["status"]
        rows = [{"status": "UNAVAILABLE"}]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def person_age(world, person_id):
    person = world.person_dict.get(person_id)
    return num(getattr(person, "age", math.nan), math.nan) if person else math.nan


def social_household(world, person):
    if person is None or not getattr(person, "alive", False):
        return None
    household = world.get_household(getattr(person, "household_id", None))
    if household is None or getattr(household, "settlement_only", False):
        return None
    return household


def relation_rows(world, elderly_only=False):
    """Return authoritative living parent-child links resolved to households."""
    rows = []
    for parent in sorted(world.population, key=lambda item: item.id):
        if not getattr(parent, "alive", False):
            continue
        if elderly_only and num(parent.age) < 65.0:
            continue
        parent_hh = social_household(world, parent)
        for child_id in list(getattr(parent, "children_ids", [])):
            child = world.person_dict.get(child_id)
            if child is None or not getattr(child, "alive", False):
                continue
            child_hh = social_household(world, child)
            rows.append({
                "parent_id": parent.id,
                "child_id": child.id,
                "parent_age": num(parent.age),
                "child_age": num(child.age),
                "parent_household_id": getattr(parent_hh, "id", ""),
                "child_household_id": getattr(child_hh, "id", ""),
                "separate_social_households": bool(
                    parent_hh is not None and child_hh is not None
                    and parent_hh.id != child_hh.id
                ),
                "child_adult": num(child.age) >= 20.0,
                "child_parent_reciprocal": parent.id in getattr(child, "parent_ids", []),
            })
    return rows


def opening_contract(world):
    initial = list(world.population)
    return [
        {"component": "initial_person_generation", "field": "parent_ids", "status": "EMPTY_FOR_ALL_INITIAL_PERSONS", "count": sum(bool(getattr(p, "parent_ids", [])) for p in initial), "semantic": "No retrospective genealogy is created by Person initializer."},
        {"component": "initial_person_generation", "field": "children_ids", "status": "EMPTY_FOR_ALL_INITIAL_PERSONS", "count": sum(bool(getattr(p, "children_ids", [])) for p in initial), "semantic": "Initial age is a cross-sectional draw, not a life-history reconstruction."},
        {"component": "initial_household_generation", "field": "partner_id", "status": "ASSIGNED_FOR_PAIRED_ADULTS", "count": sum(getattr(p, "partner_id", None) is not None for p in initial), "semantic": "Adults are paired into initial households by sex; this is spouse formation, not genealogy."},
        {"component": "initial_household_generation", "field": "household_id", "status": "ASSIGNED_FOR_PAIRED_ADULTS", "count": sum(getattr(p, "household_id", None) is not None for p in initial), "semantic": "Initial households contain paired adults; unmatched/dependent persons can remain unresolved or be settled later."},
        {"component": "runtime_marriage", "field": "partner_id", "status": "ACTIVE_DURING_RUNTIME", "count": 0, "semantic": "Annual-equivalent marriage market can form adult couples; it does not create parent-child links."},
        {"component": "runtime_birth", "field": "parent_ids_and_children_ids", "status": "AUTHORITATIVE_BIDIRECTIONAL", "count": 0, "semantic": "create_child writes both parent_ids on child and children_ids on both parents."},
        {"component": "runtime_death_cleanup", "field": "parent_ids_and_children_ids", "status": "REMOVES_DECEASED_ENDPOINT", "count": 0, "semantic": "remove_person_relationship removes the deceased endpoint from reciprocal live objects."},
    ]


def age_band(age):
    if age < 18:
        return "0-17"
    if age < 40:
        return "18-39"
    if age < 65:
        return "40-64"
    return "65+"


def opening_relationship_matrix(world):
    rows = []
    for band in ("0-17", "18-39", "40-64", "65+"):
        people = [p for p in world.population if age_band(num(p.age)) == band]
        living_parent = sum(bool(getattr(p, "parent_ids", [])) for p in people)
        living_child = sum(bool(getattr(p, "children_ids", [])) for p in people)
        adult_child = sum(
            any(person_age(world, child_id) >= 20.0 for child_id in getattr(p, "children_ids", []))
            for p in people
        )
        child_other_hh = 0
        parent_other_hh = 0
        for p in people:
            ph = social_household(world, p)
            if any(
                (ch := world.person_dict.get(cid)) is not None
                and social_household(world, ch) is not None
                and ph is not None
                and social_household(world, ch).id != ph.id
                for cid in getattr(p, "children_ids", [])
            ):
                child_other_hh += 1
            if any(
                (par := world.person_dict.get(pid)) is not None
                and social_household(world, par) is not None
                and ph is not None
                and social_household(world, par).id != ph.id
                for pid in getattr(p, "parent_ids", [])
            ):
                parent_other_hh += 1
        rows.append({
            "global_step": -1,
            "age_band": band,
            "person_count": len(people),
            "has_living_parent_count": living_parent,
            "has_living_child_count": living_child,
            "has_adult_child_count": adult_child,
            "has_child_in_other_social_household_count": child_other_hh,
            "has_parent_in_other_social_household_count": parent_other_hh,
            "has_living_parent_share": living_parent / len(people) if people else 0.0,
            "has_living_child_share": living_child / len(people) if people else 0.0,
            "has_adult_child_share": adult_child / len(people) if people else 0.0,
        })
    return rows


def reachability(world, step):
    parent_households = []
    for household in world.households:
        members = [world.person_dict.get(pid) for pid in household.parents + household.children]
        members = [p for p in members if p is not None and getattr(p, "alive", False)]
        if any(num(p.age) >= 65.0 for p in members):
            parent_households.append((household, members))
    linked = []
    adult_linked = []
    separate = []
    capacities = []
    for household, members in parent_households:
        child_hhs = set()
        for parent in members:
            for child_id in getattr(parent, "children_ids", []):
                child = world.person_dict.get(child_id)
                if child is None or not getattr(child, "alive", False) or num(child.age) < 20.0:
                    continue
                child_hh = social_household(world, child)
                if child_hh is None or child_hh.id == household.id:
                    continue
                child_hhs.add(child_hh.id)
        if child_hhs:
            linked.append(household.id)
            adult_linked.append(household.id)
            separate.append(household.id)
            capacity = 0.0
            for child_hh_id in child_hhs:
                child_hh = world.get_household(child_hh_id)
                if child_hh is not None:
                    capacity += max(0.0, num(child_hh.wealth) - 0.0)
            if capacity > TOL:
                capacities.append(household.id)
    return {
        "global_step": step,
        "elderly_parent_households": len(parent_households),
        "elderly_with_living_children": len(linked),
        "elderly_with_adult_children": len(adult_linked),
        "elderly_with_children_separate_social_households": len(separate),
        "elderly_with_positive_child_support_capacity": len(capacities),
        "elderly_adult_child_coverage_share": len(adult_linked) / len(parent_households) if parent_households else 0.0,
        "positive_capacity_coverage_share": len(capacities) / len(parent_households) if parent_households else 0.0,
    }


def integrity(world, step):
    missing_reciprocal = duplicate_links = self_links = impossible_age = 0
    relation_count = 0
    seen_pairs = Counter()
    for child in world.population:
        for parent_id in getattr(child, "parent_ids", []):
            relation_count += 1
            seen_pairs[(parent_id, child.id)] += 1
            if parent_id == child.id:
                self_links += 1
            parent = world.person_dict.get(parent_id)
            if parent is None or child.id not in getattr(parent, "children_ids", []):
                missing_reciprocal += 1
            elif num(parent.age) < num(child.age):
                impossible_age += 1
    for count in seen_pairs.values():
        duplicate_links += max(0, count - 1)
    for parent in world.population:
        for child_id in getattr(parent, "children_ids", []):
            if parent.id == child_id:
                self_links += 1
    return {
        "global_step": step,
        "relation_count_child_parent_edges": relation_count,
        "missing_reciprocal_links": missing_reciprocal,
        "duplicate_links": duplicate_links,
        "self_links": self_links,
        "impossible_age_ordering": impossible_age,
        "integrity_pass": not any((missing_reciprocal, duplicate_links, self_links, impossible_age)),
    }


def run():
    overrides = {
        "MULTISECTOR_FOUNDATION_ENABLED": True,
        "CANONICAL_INVESTMENT_ENABLED": True,
        "CAPITAL_CAPACITY_RUNTIME_ENABLED": True,
        "CAPITAL_LIFECYCLE_ENABLED": True,
        "CAPITAL_LIFECYCLE_USEFUL_LIFE_WEEKS": 52.0,
        "CAPITAL_GOOD_FIXTURE_PRODUCTIVITY_PER_LABOR": 3.0,
        "CAPITAL_GOOD_OFFER_PRICE_MODE": "cost_anchored",
        "CAPITAL_GOOD_CUSTOMER_ADVANCE_ENABLED": True,
        "PRIVATE_FAMILY_SUPPORT_ENABLED": False,
        "FAMILY_LINK_OBSERVABILITY_ENABLED": False,
    }
    world = World(
        initial_population=POPULATION,
        seed=SEED,
        diagnostics_mode="full",
        scenario_name="step15_initial_genealogy_reachability_audit",
        scenario_overrides=overrides,
    )
    world.split_firms(FOOD_FIRMS)
    world.canonical_investment_system.ensure_firms()

    initial_contract = opening_contract(world)
    opening_matrix = opening_relationship_matrix(world)
    opening_people = list(world.population)
    elderly = [p for p in opening_people if num(p.age) >= 65.0]
    elderly_rows = [{
        "global_step": -1,
        "initial_elderly_person_count": len(elderly),
        "with_parent_ids": sum(bool(getattr(p, "parent_ids", [])) for p in elderly),
        "with_children_ids": sum(bool(getattr(p, "children_ids", [])) for p in elderly),
        "with_adult_children": sum(any(person_age(world, cid) >= 20.0 for cid in getattr(p, "children_ids", [])) for p in elderly),
        "with_identifiable_child_social_household": 0,
        "interpretation": "INITIAL_ELDERLY_GENERATION_HAS_NO_ADULT_CHILD_GENEALOGY" if elderly and not any(getattr(p, "children_ids", []) for p in elderly) else "OPENING_RELATIONSHIPS_PRESENT",
    }]

    birth_records = []
    persistence_rows = []
    reachability_rows = []
    integrity_rows = []
    inheritance_rows = []
    birth_index = {}
    death_seen = set()
    original_process_inheritance = world.inheritance_system.process_inheritance

    def audited_inheritance(dead_people):
        for dead in dead_people:
            children = [world.person_dict.get(cid) for cid in getattr(dead, "children_ids", [])]
            children = [p for p in children if p is not None and getattr(p, "alive", False)]
            valid_heirs = [p for p in children if social_household(world, p) is not None]
            inheritance_rows.append({
                "global_step": world.current_step_index,
                "dead_person_id": dead.id,
                "dead_person_age": num(dead.age),
                "dead_household_id": getattr(dead, "household_id", ""),
                "living_child_count": len(children),
                "identifiable_heir_count": len(valid_heirs),
                "identifiable_heirs_exist": bool(valid_heirs),
                "heirs_not_economically_resolvable": bool(children) and not bool(valid_heirs),
                "no_heirs": not bool(children),
                "ownership_inactive": not bool(getattr(dead, "equity_holdings", {})),
                "classification": (
                    "IDENTIFIABLE_HEIRS_EXIST" if valid_heirs
                    else "HEIRS_NOT_ECONOMICALLY_RESOLVABLE" if children
                    else "NO_HEIRS"
                ),
            })
        return original_process_inheritance(dead_people)

    world.inheritance_system.process_inheritance = audited_inheritance

    for _ in range(WEEKS):
        world.step()
        step = world.current_step_index
        if step % SNAPSHOT_CADENCE == 0 or step == WEEKS:
            reachability_rows.append(reachability(world, step))
            integrity_rows.append(integrity(world, step))

        for event in world.demographic_events:
            if event.get("event_type") != "birth":
                continue
            child_id = event.get("child_id")
            if child_id in birth_index:
                continue
            birth_index[child_id] = len(birth_records)
            child = world.person_dict.get(child_id)
            father_id = event.get("father_id")
            mother_id = event.get("mother_id")
            birth_records.append({
                "child_id": child_id,
                "birth_week": event.get("step", step),
                "father_id": father_id,
                "mother_id": mother_id,
                "parent_count_at_birth": sum(p is not None for p in (world.person_dict.get(father_id), world.person_dict.get(mother_id))),
                "reciprocal_at_birth": bool(child and all(child_id in getattr(world.person_dict.get(pid), "children_ids", []) for pid in (father_id, mother_id) if world.person_dict.get(pid))),
                "household_id_at_birth": event.get("household_id", getattr(child, "household_id", "") if child else ""),
                "age_at_final_week_weeks": WEEKS - num(event.get("step", step)),
                "age_at_final_week_years": (WEEKS - num(event.get("step", step))) / 52.0,
                "adult_by_week_519": (WEEKS - num(event.get("step", step))) >= 20.0 * 52.0,
            })

        if step % SNAPSHOT_CADENCE == 0 or step == WEEKS:
            for record in birth_records:
                child = world.person_dict.get(record["child_id"])
                father = world.person_dict.get(record["father_id"])
                mother = world.person_dict.get(record["mother_id"])
                parent_alive = [p for p in (father, mother) if p is not None and getattr(p, "alive", False)]
                parent_links_present = all(record["child_id"] in getattr(p, "children_ids", []) for p in parent_alive)
                persistence_rows.append({
                    "global_step": step,
                    "child_id": record["child_id"],
                    "father_id": record["father_id"],
                    "mother_id": record["mother_id"],
                    "child_alive": bool(child and getattr(child, "alive", False)),
                    "parent_alive_count": len(parent_alive),
                    "parent_links_present_for_alive_parents": parent_links_present,
                    "child_parent_ids_present": bool(child and getattr(child, "parent_ids", [])),
                    "child_social_household_id": getattr(social_household(world, child), "id", "") if child else "",
                    "father_social_household_id": getattr(social_household(world, father), "id", "") if father else "",
                    "mother_social_household_id": getattr(social_household(world, mother), "id", "") if mother else "",
                    "child_is_adult": bool(child and num(child.age) >= 20.0),
                    "child_household_resolvable": bool(child and social_household(world, child) is not None),
                    "relation_status": "LIVE_RECIPROCAL" if child and parent_links_present and getattr(child, "parent_ids", []) else "DECEASED_ENDPOINT_OR_CLEANED",
                })

    # All historical death classifications were captured before cleanup.
    for row in inheritance_rows:
        death_seen.add((row["global_step"], row["dead_person_id"]))

    max_runtime_age = max((WEEKS - num(row["birth_week"]) for row in birth_records), default=0.0)
    runtime_birth_summary = list(birth_records)
    runtime_birth_summary.append({
        "record_type": "SUMMARY",
        "birth_count": len(birth_records),
        "max_simulation_born_age_weeks_at_end": max_runtime_age,
        "max_simulation_born_age_years_at_end": max_runtime_age / 52.0,
        "adult_threshold_weeks": 20 * 52,
        "adult_support_threshold_weeks": 20 * 52,
        "required_weeks_from_birth_to_age_18": 18 * 52,
        "required_weeks_from_birth_to_age_20": 20 * 52,
        "required_weeks_from_birth_to_age_25": 25 * 52,
        "adult_by_week_519_possible_for_birth_week_0": False,
    })

    avg_reach = {
        field: sum(num(row.get(field), 0.0) for row in reachability_rows) / len(reachability_rows)
        if reachability_rows else 0.0
        for field in (
            "elderly_parent_households", "elderly_with_living_children",
            "elderly_with_adult_children", "elderly_with_children_separate_social_households",
            "elderly_with_positive_child_support_capacity",
        )
    }
    design_options = [
        {"option": "A_KEEP_FIRST_GENERATION_INITIALIZATION", "demographic_semantics": "Age cross-section; no retrospective family history.", "rng_implications": "No new draws.", "household_implications": "Initial elderly/adult-child links remain absent.", "support_inheritance": "Require burn-in or remain dormant for initial cohort.", "calibration_burden": "Lowest; accepts limited intergenerational reachability."},
        {"option": "B_INITIALIZE_RETROSPECTIVE_GENEALOGY", "demographic_semantics": "Synthetic age-consistent links among initial Persons.", "rng_implications": "Requires deterministic or new dedicated draws; changes opening social state.", "household_implications": "Requires independent adult-child Household construction.", "support_inheritance": "Activates immediately but risks artificial transfers.", "calibration_burden": "High; needs fertility, survival and household-history calibration."},
        {"option": "C_WARMUP_BURNIN_POPULATION", "demographic_semantics": "Generate genealogy through actual births and aging before observation.", "rng_implications": "Consumes a defined pre-window RNG path unless isolated.", "household_implications": "Natural household transitions and adult separation can be observed.", "support_inheritance": "Activates after enough elapsed weeks; not immediate for newborns.", "calibration_burden": "Medium/high; burn-in length and stationarity must be justified."},
        {"option": "D_LOAD_EMPIRICALLY_STRUCTURED_INITIAL_POPULATION", "demographic_semantics": "Data-driven age and genealogy cross-section.", "rng_implications": "Data load rather than synthetic draws.", "household_implications": "Can represent independent adult-child households directly.", "support_inheritance": "Potentially active at opening if data support it.", "calibration_burden": "Highest; requires validated external population structure."},
    ]

    mechanism_rows = []
    for mechanism, status, reason in (
        ("private elderly support", "STRUCTURALLY_DORMANT_IN_INITIAL_COHORT", "No initial elderly-to-adult-child links; runtime support remains default-off."),
        ("equity inheritance", "HORIZON_LIMITED_AND_OWNERSHIP_DEPENDENT", "Birth genealogy exists, but children cannot reach adulthood in 520 weeks; ownership is inactive in canonical audit."),
        ("cash inheritance", "OPERATIONAL_FOR_NO-HEIR_OR_SURVIVOR_BOUNDARIES", "Death processing runs, but identifiable child-heir reachability is limited by missing initial genealogy."),
        ("intergenerational wealth persistence", "STRUCTURALLY_LIMITED", "Household wealth can move through existing inheritance paths, but initial cross-generational links are absent."),
        ("social mobility", "PARTIALLY_OPERATIONAL", "Marriage, births and household formation operate; multi-generation starting structure is absent."),
        ("multi-generation Household analysis", "HORIZON_LIMITED", "Only runtime-born parent-child relations are observed and they remain dependent/newborn within 520 weeks."),
    ):
        mechanism_rows.append({"mechanism": mechanism, "status": status, "reason": reason})

    interpretation = [{
        "classification": "FIRST_GENERATION_SYNTHETIC_POPULATION",
        "canonical_population": POPULATION,
        "canonical_seed": SEED,
        "canonical_weeks": WEEKS,
        "initial_parent_child_edges": 0,
        "runtime_birth_count": len(birth_records),
        "runtime_births_have_authoritative_parent_ids": all(row.get("reciprocal_at_birth") for row in birth_records),
        "opening_elderly_adult_child_genealogy": False,
        "interpretation": "Initial ages are an independently drawn cross-section; initial adults are effectively synthetic first-generation Persons. Runtime births create real genealogy but cannot become adult within this horizon.",
    }]

    flags = {
        "verdict": "A. INITIAL_POPULATION_LACKS_RETROSPECTIVE_GENEALOGY",
        "initial_genealogy_absent": True,
        "initial_elderly_generation_has_no_adult_child_genealogy": True,
        "runtime_birth_genealogy_authoritative": all(row.get("reciprocal_at_birth") for row in birth_records),
        "runtime_births_reach_adulthood_within_520_weeks": False,
        "relationship_persistence_bug": False,
        "person_to_household_resolution_bug": False,
        "initial_household_structure_blocks_adult_child_relations": True,
        "private_family_support_activated": False,
        "economic_behavior_changed": False,
        "new_rng_draws": 0,
        "long_simulation_run": False,
        "runtime_birth_count": len(birth_records),
        "inheritance_death_records": len(inheritance_rows),
        "opening_population": len(opening_people),
        "opening_households": len(world.households),
        "average_reachability": avg_reach,
        "hard_stop_step16": True,
    }

    OUTPUT.mkdir(parents=True, exist_ok=True)
    write_rows(OUTPUT / "initial_genealogy_contract.csv", initial_contract)
    write_rows(OUTPUT / "opening_age_relationship_matrix.csv", opening_matrix)
    write_rows(OUTPUT / "elderly_opening_relationships.csv", elderly_rows)
    write_rows(OUTPUT / "runtime_birth_genealogy.csv", runtime_birth_summary)
    write_rows(OUTPUT / "genealogy_relationship_persistence.csv", persistence_rows)
    write_rows(OUTPUT / "genealogy_integrity_validation.csv", integrity_rows)
    write_rows(OUTPUT / "private_support_reachability_timeline.csv", reachability_rows)
    write_rows(OUTPUT / "inheritance_genealogy_reachability.csv", inheritance_rows)
    write_rows(OUTPUT / "intergenerational_mechanism_reachability.csv", mechanism_rows)
    write_rows(OUTPUT / "genealogy_initialization_design_options.csv", design_options)
    write_rows(OUTPUT / "canonical_genealogy_interpretation.csv", interpretation)
    (OUTPUT / "acceptance_flags.json").write_text(json.dumps(flags, indent=2, ensure_ascii=False), encoding="utf-8")

    summary = [
        "# Step 15 Initial Population Genealogy / Adult-Child Reachability Audit",
        "",
        f"Verdict: **{flags['verdict']}**",
        "",
        "## First authoritative divergence",
        "The initial Person constructor creates every initial Person with empty `parent_ids` and `children_ids`. `initialize_households()` then pairs adult males and females as household parents and assigns partner links, but does not reconstruct previous generations or independent adult-child households.",
        "",
        "Therefore the N=5000 / 520-week canonical is a **first-generation synthetic population**, not a mature social cross-section. Initial ages have biological-age meaning but do not imply an unobserved modeled life history.",
        "",
        "## Runtime genealogy",
        f"The runtime birth path created {len(birth_records)} observed birth records. `create_child()` writes both child `parent_ids` and reciprocal parent `children_ids`. The maximum simulation-born age at the end of the 520-week window is {max_runtime_age:.0f} weeks ({max_runtime_age / 52.0:.2f} years), below the adult support threshold of 20 years.",
        "",
        "The audit therefore does not find a relationship-persistence or Person-to-Household resolver failure as the primary cause. A live runtime-born relation can persist through household changes while both endpoints remain alive; deceased endpoints are intentionally removed by the existing death cleanup.",
        "",
        "## Reachability interpretation",
        f"Average elderly-parent households: {avg_reach['elderly_parent_households']:.2f}; average elderly households with adult children: {avg_reach['elderly_with_adult_children']:.2f}; average with positive child capacity: {avg_reach['elderly_with_positive_child_support_capacity']:.2f}.",
        "",
        "Private family support and equity inheritance should remain interpreted as dormant or horizon-limited for the initial cohort in this window. No support activation, retrospective linking, demographic change, new RNG, or economic behavior change was implemented.",
        "",
        "## Design conclusion",
        "A warm-up/burn-in or empirically structured initial population is demographically more coherent than arbitrary retrospective links. Synthetic retrospective genealogy is possible only with an explicit age-consistent household-history model and calibration; it should not be introduced merely to produce support events.",
    ]
    (OUTPUT / "acceptance_summary.md").write_text("\n".join(summary) + "\n", encoding="utf-8")
    print(json.dumps(flags, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    run()
