"""Demographic-only warm-up runner for a mature-population checkpoint.

The runner deliberately lives outside World.step. The ordinary economic
simulation therefore cannot silently enter demographic burn-in mode. A World
is used only as the existing lifecycle container; all economic balances are
zeroed before the first warm-up week and no Firm/economic step is called.
"""

from __future__ import annotations

import random


class DemographicBurnInRunner:
    mode = "DEMOGRAPHIC_BURNIN_MODE"
    economic_placeholder_status = "NON_RESEARCH_PLACEHOLDER"

    def __init__(self, world):
        self.world = world
        world.demographic_burnin_mode = True
        world.demographic_burnin_schema = "demographic_only_burnin_v1"
        world.demographic_burnin_absolute_week = 0
        world.demographic_burnin_fertility_pressure = 0.0
        world.demographic_burnin_birth_week_by_person = {}
        world.demographic_burnin_death_week_by_person = {}
        world.demographic_burnin_inheritance_rows = []
        self._zero_economic_placeholders()
        for person in world.population:
            person.birth_week = None
            world.demographic_burnin_birth_week_by_person[person.id] = None

    def _zero_economic_placeholders(self):
        world = self.world
        for household in getattr(world, "households", []):
            household.wealth = 0.0
            household.income_this_step = 0.0
            household.consumption_this_step = 0.0
            household.saving_this_step = 0.0
        firm_system = getattr(world, "firm_system", None)
        if firm_system is not None:
            firm_system.cash = 0.0
            if hasattr(firm_system, "inventory"):
                firm_system.inventory = 0.0
            if hasattr(firm_system, "loan_balance"):
                firm_system.loan_balance = 0.0
            central_bank = getattr(firm_system, "central_bank", None)
            if central_bank is not None:
                for field in ("money_supply", "public_income_balance", "cash"):
                    if hasattr(central_bank, field):
                        setattr(central_bank, field, 0.0)
        for firm in getattr(world, "operating_firms", lambda: [])():
            for field in ("cash", "inventory_value", "loan_balance", "interest_arrears"):
                if hasattr(firm, field):
                    setattr(firm, field, 0.0)
            if hasattr(firm, "employee_ids"):
                firm.employee_ids = []
        world.public_wealth = 0.0
        world.pending_household_formation_wealth = 0.0
        world.legacy_owner_cash = 0.0
        world.initial_private_money_stock = 0.0
        world.demographic_burnin_economic_state = {
            "status": self.economic_placeholder_status,
            "firm_cash": 0.0,
            "household_cash": 0.0,
            "loans": 0.0,
            "capital": 0.0,
            "inventory": 0.0,
            "ownership": "INACTIVE",
            "prices": "NON_RESEARCH_PLACEHOLDER",
        }

    @staticmethod
    def _social_household(world, person):
        household = world.get_household(getattr(person, "household_id", None))
        if household is None or getattr(household, "settlement_only", False):
            return None
        return household

    def _marriage_phase(self, step):
        world = self.world
        world.marriage_market_executed = False
        world.marriage_market_last_result = 0
        world.marriage_market_last_eligible_males = 0
        world.marriage_market_last_eligible_females = 0
        world.marriage_market_last_unmatched_males = 0
        world.marriage_market_last_unmatched_females = 0
        if step < world.next_marriage_market_step:
            return 0
        eligible_males, eligible_females = world.marriage_system.get_marriage_pool()
        matches = world.marriage_system.process_marriage()
        world.marriage_market_executed = True
        world.marriage_market_execution_count += 1
        world.marriage_market_last_result = matches
        world.marriage_market_last_eligible_males = len(eligible_males)
        world.marriage_market_last_eligible_females = len(eligible_females)
        world.marriage_market_last_unmatched_males = max(0, len(eligible_males) - matches)
        world.marriage_market_last_unmatched_females = max(0, len(eligible_females) - matches)
        world.next_marriage_market_step += 52
        world.record_marriage_market_diagnostic(
            len(eligible_males), len(eligible_females), matches
        )
        return matches

    def _demographic_step(self, step):
        world = self.world
        world.current_step_index = step
        world.clock.set_step(step)
        for person in list(world.population):
            person.grow()
        world.household_manager.step()
        marriages = self._marriage_phase(step)
        world.refresh_active_households()

        newborns, births = world.family_birth(
            world.demographic_burnin_fertility_pressure
        )
        for child in newborns:
            child.birth_week = step
            world.demographic_burnin_birth_week_by_person[child.id] = step

        survivors = []
        dead_people = []
        deaths = 0
        for person in list(world.population):
            person.check_death()
            if person.alive:
                survivors.append(person)
                continue
            deaths += 1
            dead_people.append(person)
            world.demographic_burnin_death_week_by_person[person.id] = step
            children = [
                world.person_dict.get(child_id)
                for child_id in getattr(person, "children_ids", [])
            ]
            children = [
                child for child in children
                if child is not None and getattr(child, "alive", False)
            ]
            resolvable_heirs = [
                child for child in children
                if self._social_household(world, child) is not None
            ]
            world.demographic_burnin_inheritance_rows.append({
                "global_step": step,
                "dead_person_id": person.id,
                "living_child_count": len(children),
                "resolvable_heir_count": len(resolvable_heirs),
                "identifiable_children": bool(children),
                "potential_estate_to_heir_reachability": bool(resolvable_heirs),
            })
            world.record_demographic_event(
                "death",
                person_id=person.id,
                sex=person.sex,
                household_id=person.household_id,
                age_years=person.age_years,
                age_weeks=person.age_weeks,
                demographic_burnin=True,
            )

        # No economic assets exist in this mode, so inheritance execution is
        # intentionally excluded from the burn-in.
        for person in dead_people:
            world.remove_person_relationship(person)
            household = world.get_household(getattr(person, "household_id", None))
            if household is not None:
                if person.id in household.parents:
                    household.parents.remove(person.id)
                if person.id in household.children:
                    household.children.remove(person.id)
            world.person_dict.pop(person.id, None)

        world.fertility_system.update_completed_couples()
        world.population = survivors + newborns
        world.household_manager.remove_dead_members()
        world.household_manager.remove_empty_households()
        world.refresh_active_households()
        world.current_step_index = step + 1
        world.demographic_burnin_absolute_week = step + 1
        return {
            "global_step": step + 1,
            "births": births,
            "deaths": deaths,
            "marriages": marriages,
        }

    def run_to(self, horizon, capture=None):
        capture = set(capture or ())
        rows = []
        while self.world.demographic_burnin_absolute_week < horizon:
            event = self._demographic_step(
                self.world.demographic_burnin_absolute_week
            )
            if event["global_step"] in capture:
                rows.append(event)
        return rows

    def demographic_state(self):
        world = self.world
        return {
            "absolute_simulation_week": world.demographic_burnin_absolute_week,
            "persons": sorted([
                {
                    "person_id": person.id,
                    "age_weeks": person.age_weeks,
                    "sex": person.sex,
                    "alive": bool(person.alive),
                    "partner_id": person.partner_id,
                    "parent_ids": sorted(set(person.parent_ids)),
                    "children_ids": sorted(set(person.children_ids)),
                    "household_id": person.household_id,
                    "birth_week": getattr(person, "birth_week", None),
                }
                for person in world.population
            ], key=lambda row: row["person_id"]),
            "households": sorted([
                {
                    "household_id": household.id,
                    "parents": sorted(set(household.parents)),
                    "children": sorted(set(household.children)),
                    "settlement_only": bool(
                        getattr(household, "settlement_only", False)
                    ),
                }
                for household in world.households
            ], key=lambda row: row["household_id"]),
            "rng_state": {
                "global_random": random.getstate(),
                "marriage_random": world.marriage_system.marriage_rng.getstate(),
                "age_phase_random": world.age_phase_rng.getstate(),
            },
        }


def relation_metrics(world):
    persons = list(world.population)
    person_dict = world.person_dict
    living_parent_count = 0
    living_child_count = 0
    reciprocal_edges = 0
    missing_reciprocal = 0
    duplicate_links = 0
    self_links = 0
    impossible_age = 0
    grandparents = set()
    elderly_parent_ids = set()
    elderly_separate_parent_ids = set()
    lineage_depth = {}
    for person in persons:
        if person.parent_ids:
            living_parent_count += 1
        if person.children_ids:
            living_child_count += 1
        duplicate_links += (
            max(0, len(person.parent_ids) - len(set(person.parent_ids)))
            + max(0, len(person.children_ids) - len(set(person.children_ids)))
        )
        for parent_id in person.parent_ids:
            reciprocal_edges += 1
            parent = person_dict.get(parent_id)
            if parent_id == person.id:
                self_links += 1
            if parent is None or person.id not in parent.children_ids:
                missing_reciprocal += 1
            elif parent.age < person.age:
                impossible_age += 1
        if any(
            person_dict.get(child_id) is not None
            and person_dict[child_id].children_ids
            for child_id in person.children_ids
        ):
            grandparents.add(person.id)
        if person.age >= 65.0 and any(
            person_dict.get(child_id) is not None
            and person_dict[child_id].age >= 20.0
            for child_id in person.children_ids
        ):
            elderly_parent_ids.add(person.id)
            relation_household = world.get_household(person.household_id)
            for child_id in person.children_ids:
                child = person_dict.get(child_id)
                child_hh = (
                    world.get_household(getattr(child, "household_id", None))
                    if child else None
                )
                if (
                    child_hh is not None
                    and relation_household is not None
                    and child_hh.id != relation_household.id
                ):
                    elderly_separate_parent_ids.add(person.id)
        lineage_depth[person.id] = 1 if person.parent_ids else 0
    return {
        "person_count": len(persons),
        "persons_with_living_parents": living_parent_count,
        "persons_with_living_children": living_child_count,
        "reciprocal_parent_child_edges": reciprocal_edges,
        "elderly_with_adult_children": len(elderly_parent_ids),
        "elderly_with_adult_children_separate_household": len(
            elderly_separate_parent_ids
        ),
        "grandparent_count": len(grandparents),
        "max_observed_lineage_depth": max(lineage_depth.values(), default=0),
        "missing_reciprocal_links": missing_reciprocal,
        "duplicate_links": duplicate_links,
        "self_links": self_links,
        "impossible_parent_age_ordering": impossible_age,
    }


def household_metrics(world):
    households = list(world.households)
    adults = [
        person for person in world.population
        if person.age >= 20.0 and person.alive
    ]
    married_adults = sum(person.partner_id is not None for person in adults)
    sizes = [household.size() for household in households]
    return {
        "social_household_count": len(households),
        "household_size_mean": (
            sum(sizes) / len(sizes) if sizes else 0.0
        ),
        "household_size_median": (
            sorted(sizes)[len(sizes) // 2] if sizes else 0
        ),
        "married_adult_count": married_adults,
        "unmarried_adult_count": len(adults) - married_adults,
    }


def horizon_stability(horizon_rows, current):
    prior = horizon_rows[-1] if horizon_rows else None
    if prior is None:
        population_drift = 0.0
        household_drift = 0.0
    else:
        population_drift = (
            current["person_count"] - prior["person_count"]
        ) / max(1.0, prior["person_count"])
        household_drift = (
            current["social_household_count"]
            - prior["social_household_count"]
        ) / max(1.0, prior["social_household_count"])
    balance = current["births"] - current["deaths"]
    if abs(population_drift) <= 0.10 and abs(household_drift) <= 0.10:
        classification = "APPROXIMATELY_STABLE"
    elif population_drift > 0.10:
        classification = "EXPANDING"
    elif population_drift < -0.10:
        classification = "DECLINING"
    else:
        classification = "UNSTABLE"
    return {
        "population_drift_from_prior_horizon": population_drift,
        "household_drift_from_prior_horizon": household_drift,
        "birth_death_balance_current_window": balance,
        "stability_classification": classification,
    }