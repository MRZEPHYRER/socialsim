import math
import random

from economy import config as economy_config
from fertility import config
from person import Person
from time_system import annual_probability_to_step_probability


class FertilitySystem:

    def __init__(self, world):
        self.world = world
        self.couples = {}

        raw_values = [
            self.raw_age_fertility(age)
            for age in range(
                config.BASE_REPRODUCTIVE_AGE_MIN,
                config.BASE_REPRODUCTIVE_AGE_MAX + 1
            )
        ]

        self.age_fertility_normalizer = (
            sum(raw_values)
            /
            len(raw_values)
        )
        self.age_fertility_multipliers = {
            age: self.raw_age_fertility(age) / self.age_fertility_normalizer
            for age in range(
                config.BASE_REPRODUCTIVE_AGE_MIN,
                config.BASE_REPRODUCTIVE_AGE_MAX + 1
            )
        }
        self.fertility_child_cost_value = self.fertility_child_cost()
        self.parity_fertility_factors = {
            child_count: self.parity_fertility_factor(child_count)
            for child_count in range(16)
        }
        self.birth_spacing_skipped_draws = 0
        self.observed_birth_intervals = []
        self.fertility_draws = 0
        self.fertility_probability_records = []

    def __setstate__(self, state):
        state = dict(state)
        state.setdefault("birth_spacing_skipped_draws", 0)
        state.setdefault("observed_birth_intervals", [])
        state.setdefault("fertility_draws", 0)
        state.setdefault("fertility_probability_records", [])
        self.__dict__.update(state)

    def couple_key(self, father_id, mother_id):
        return (
            father_id,
            mother_id
        )

    def record_couple(self, household, father, mother):
        key = self.couple_key(
            father.id,
            mother.id
        )

        if key not in self.couples:
            self.couples[key] = {
                "father_id": father.id,
                "mother_id": mother.id,
                "first_household_id": household.id,
                "births": 0,
                "completed": False,
                "ever_mother_reproductive": False,
            }

        if mother.can_reproduce():
            self.couples[key]["ever_mother_reproductive"] = True

        return self.couples[key]

    def update_completed_couples(self):
        for data in self.couples.values():
            if data["completed"]:
                continue

            father_alive = (
                self.world.get_person_by_id(data["father_id"])
                is not None
            )
            mother_alive = (
                self.world.get_person_by_id(data["mother_id"])
                is not None
            )

            if not father_alive and not mother_alive:
                data["completed"] = True

    def completed_couples(self):
        return [
            data
            for data in self.couples.values()
            if data["completed"]
        ]

    def fertility_exposed_completed_couples(self):
        return [
            data
            for data in self.completed_couples()
            if data["ever_mother_reproductive"]
        ]

    def incomplete_couples(self):
        return [
            data
            for data in self.couples.values()
            if not data["completed"]
        ]

    def clamp(self, value, low, high):
        return max(
            low,
            min(high, value)
        )

    def smoothstep(self, value):
        value = self.clamp(
            value,
            0.0,
            1.0
        )

        return value * value * (3 - 2 * value)

    def base_birth_probability(self, pressure):
        probability = (
            config.BASE_BIRTH_RATE
            *
            self.world.scenario_window_multiplier(
                "BIRTH_RATE_SHOCK_START",
                "BIRTH_RATE_SHOCK_END",
                "BIRTH_RATE_SHOCK_MULTIPLIER",
            )
            *
            (1 - pressure)
        )

        return max(
            0.0,
            probability
        )

    def raw_age_fertility(self, age):
        return (
            config.FERTILITY_FLOOR
            +
            math.exp(
                -((age - config.FERTILITY_PEAK_AGE) ** 2)
                /
                (2 * config.FERTILITY_SPREAD ** 2)
            )
        )

    def age_fertility_multiplier(self, age):
        # Preserve the pre-weekly model's integer calendar-age bins.
        return self.age_fertility_multipliers.get(int(age), 0.0)

    def base_child_cost(self):
        return (
            economy_config.CHILD_EXTRA_COST
            +
            economy_config.NEW_BASIC_CONSUMPTION_PER_EQUIVALENT_MEMBER
            *
            economy_config.NEW_CHILD_WEIGHT
        )

    def fertility_child_cost(self):
        return (
            self.base_child_cost()
            *
            config.FERTILITY_CHILD_COST_MULTIPLIER
        )

    def household_free_budget(self, household):
        necessary = getattr(
            household,
            "necessary_consumption_this_step",
            0.0
        )
        affordable = getattr(
            household,
            "affordable_consumption_this_step",
            0.0
        )

        return affordable - necessary

    def household_security_ratio(self, household):
        target_wealth = getattr(
            household,
            "target_wealth_this_step",
            0.0
        )

        if target_wealth <= 0:
            return 1.0

        return max(
            0.0,
            household.wealth / target_wealth
        )

    def economic_factor_from_security_ratio(self, security_ratio):
        exponent = -config.ECONOMIC_SECURITY_SLOPE * (
            security_ratio
            -
            config.ECONOMIC_SECURITY_MIDPOINT
        )
        exponent = self.clamp(
            exponent,
            -60.0,
            60.0
        )

        factor = (
            config.ECONOMIC_FACTOR_MIN
            +
            (
                config.ECONOMIC_FACTOR_MAX
                -
                config.ECONOMIC_FACTOR_MIN
            )
            /
            (1.0 + math.exp(exponent))
        )

        return self.clamp(
            factor,
            config.ECONOMIC_FACTOR_MIN,
            config.ECONOMIC_FACTOR_MAX
        )

    def economic_fertility_factor(self, household):
        return self.economic_factor_from_security_ratio(
            self.household_security_ratio(household)
        )

    def parity_fertility_factor(self, child_count):
        if hasattr(self, "parity_fertility_factors"):
            cached_factor = self.parity_fertility_factors.get(child_count)

            if cached_factor is not None:
                return cached_factor

        return (
            config.PARITY_FLOOR
            +
            config.PARITY_AMPLITUDE
            /
            (
                1
                +
                math.exp(
                    config.PARITY_SLOPE
                    *
                    (
                        child_count
                        -
                        config.PARITY_MIDPOINT
                    )
                )
            )
        )

    def household_birth_probability(
        self,
        household,
        mother,
        pressure
    ):
        annual_probability = (
            self.base_birth_probability(pressure)
            *
            self.age_fertility_multiplier(mother.age)
            *
            self.economic_fertility_factor(household)
            *
            self.parity_fertility_factor(
                len(household.children)
            )
        )

        annual_probability = self.clamp(
            annual_probability,
            0.0,
            1.0
        )
        return annual_probability_to_step_probability(annual_probability)

    def annual_composite_birth_probability(
        self,
        household,
        mother,
        pressure,
    ):
        """Return the uncoupled annual composite probability before conversion."""
        annual_probability = (
            self.base_birth_probability(pressure)
            * self.age_fertility_multiplier(mother.age)
            * self.economic_fertility_factor(household)
            * self.parity_fertility_factor(len(household.children))
        )
        return self.clamp(annual_probability, 0.0, 1.0)

    def fertility_diagnostics(self, household, mother, pressure):
        annual_probability = self.annual_composite_birth_probability(
            household,
            mother,
            pressure,
        )
        return {
            "age_years": mother.age_years,
            "age_weeks": mother.age_weeks,
            "base_annual_birth_rate": config.BASE_BIRTH_RATE,
            "fertility_age_factor": self.age_fertility_multiplier(mother.age),
            "fertility_economic_factor": self.economic_fertility_factor(household),
            "parity_factor": self.parity_fertility_factor(len(household.children)),
            "annual_composite_fertility_probability": annual_probability,
            "weekly_fertility_probability": annual_probability_to_step_probability(
                annual_probability
            ),
        }

    def create_child(self, household, father, mother):
        child = Person(
            self.world.next_person_id,
            0,
            random.choice(["M", "F"])
        )

        self.world.next_person_id += 1
        self.world.person_dict[child.id] = child

        child.household_id = household.id
        child.father_id = father.id
        child.mother_id = mother.id
        child.parent_ids = [
            father.id,
            mother.id
        ]

        father.add_child(child.id)
        mother.add_child(child.id)
        household.add_child(child.id)

        return child

    def family_birth(self, pressure):
        births = 0
        newborns = []
        person_dict = self.world.person_dict
        couples = self.couples
        base_probability = self.base_birth_probability(pressure)
        age_fertility_multipliers = self.age_fertility_multipliers
        parity_fertility_factors = self.parity_fertility_factors
        clamp = self.clamp
        random_value = random.random
        create_child = self.create_child
        economic_min = config.ECONOMIC_FACTOR_MIN
        economic_max = config.ECONOMIC_FACTOR_MAX
        security_midpoint = config.ECONOMIC_SECURITY_MIDPOINT
        security_slope = config.ECONOMIC_SECURITY_SLOPE

        for household in self.world.households:
            if len(household.parents) != 2:
                continue

            father = person_dict.get(household.parents[0])
            mother = person_dict.get(household.parents[1])

            if father is None or mother is None:
                continue

            key = (
                father.id,
                mother.id
            )
            couple = couples.get(key)

            if couple is None:
                couple = {
                    "father_id": father.id,
                    "mother_id": mother.id,
                    "first_household_id": household.id,
                    "births": 0,
                    "completed": False,
                    "ever_mother_reproductive": False,
                }
                couples[key] = couple

            mother_age = mother.age_years
            mother_age_bin = int(mother_age)

            if mother.sex != "F" or mother_age < 20 or mother_age > 40:
                continue

            current_step = self.world.current_step_index
            if not mother.birth_spacing_eligible(
                current_step,
                config.MIN_BIRTH_INTERVAL_WEEKS,
            ):
                self.birth_spacing_skipped_draws += 1
                continue

            couple["ever_mother_reproductive"] = True

            target_wealth = getattr(
                household,
                "target_wealth_this_step",
                0.0,
            )

            if target_wealth <= 0:
                economic_factor = 1.0
            else:
                security_ratio = max(
                    0.0,
                    household.wealth / target_wealth
                )
                exponent = -security_slope * (
                    security_ratio
                    -
                    security_midpoint
                )
                exponent = clamp(exponent, -60.0, 60.0)
                economic_factor = (
                    economic_min
                    +
                    (economic_max - economic_min)
                    /
                    (1.0 + math.exp(exponent))
                )

            economic_factor = clamp(
                economic_factor,
                economic_min,
                economic_max,
            )
            child_count = len(household.children)
            parity_factor = parity_fertility_factors.get(child_count)

            if parity_factor is None:
                parity_factor = self.parity_fertility_factor(child_count)

            annual_birth_probability = (
                base_probability
                *
                age_fertility_multipliers.get(mother_age_bin, 0.0)
                *
                economic_factor
                *
                parity_factor
            )
            annual_birth_probability = clamp(
                annual_birth_probability,
                0.0,
                1.0,
            )
            birth_probability = annual_probability_to_step_probability(
                annual_birth_probability
            )

            self.fertility_draws += 1
            self.fertility_probability_records.append({
                "global_step": current_step,
                "annual": annual_birth_probability,
                "weekly": birth_probability,
            })

            if random_value() >= birth_probability:
                continue

            previous_birth_step = mother.last_birth_step
            mother.last_birth_step = current_step
            if previous_birth_step is not None:
                self.observed_birth_intervals.append(
                    current_step - previous_birth_step
                )

            child = create_child(household, father, mother)
            self.world.record_demographic_event(
                "birth",
                child_id=child.id,
                mother_id=mother.id,
                father_id=father.id,
                household_id=household.id,
                mother_age_weeks=mother.age_weeks,
                mother_age_years=mother.age_years,
                parity_before_birth=child_count,
                last_birth_step_before=previous_birth_step,
                inter_birth_interval_weeks=(
                    current_step - previous_birth_step
                    if previous_birth_step is not None else None
                ),
                annual_composite_fertility_probability=annual_birth_probability,
                weekly_fertility_probability=birth_probability,
                fertility_age_factor=age_fertility_multipliers.get(mother_age_bin, 0.0),
                fertility_economic_factor=economic_factor,
                parity_factor=parity_factor,
            )
            newborns.append(child)
            couple["births"] += 1
            births += 1

        return newborns, births

    def birth_spacing_diagnostics(self):
        current_step = self.world.current_step_index
        intervals = self.observed_birth_intervals
        return {
            "birth_spacing_skipped_draws": self.birth_spacing_skipped_draws,
            "minimum_observed_inter_birth_interval": (
                min(intervals) if intervals else None
            ),
            "minimum_birth_interval_weeks": config.MIN_BIRTH_INTERVAL_WEEKS,
        }
