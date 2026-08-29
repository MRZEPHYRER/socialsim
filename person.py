import random
import math
from functools import lru_cache
from time_system import annual_hazard_to_step_probability


@lru_cache(maxsize=None)
def base_mortality_hazard(age):
    infant = (
        0.006
        *
        math.exp(-0.15 * age)
    )

    background = 0.0008

    aging = (
        0.00001
        *
        math.exp(0.105 * age)
    )

    return (
        infant
        +
        background
        +
        aging
    )


@lru_cache(maxsize=None)
def base_death_probability(age):
    probability = (
        1
        -
        math.exp(-base_mortality_hazard(age))
    )

    return min(
        probability,
        0.95
    )

class Person:

    def __init__(self, id, age, sex):

        # Canonical age storage is integer simulation weeks.  The public
        # ``age`` property remains calendar years for existing model code.
        self.age_weeks = int(round(float(age) * 52))
        self.sex = sex
        self.id = id
        self.household_id = None
        self.alive = True
        self.retired = False
        self.partner_id = None
        self.father_id = None
        self.mother_id = None
        self.marriage_wait = 0
        self.is_seeking_partner = False
        self.independent=False
        self.children_ids = []
        self.parent_ids = []
        self.last_birth_step = None
        # Passive Step 15E.2 runtime holdings.  Empty in the canonical model;
        # only the controlled issuance fixture populates this mapping.
        self.equity_holdings = {}
        # Cost basis is carried across passive ownership-lifecycle transfers.
        self.equity_cost_basis = {}
        # Step17.S tax-year state; inert while personal tax is disabled.
        self.tax_year_index = 0
        self.cumulative_taxable_wage = 0.0
        self.cumulative_tax_withheld = 0.0
        self.estimated_or_accrued_annual_liability = 0.0
        self.last_personal_tax_this_step = 0.0

    @property
    def age(self):
        return self.age_weeks / 52.0

    @age.setter
    def age(self, value):
        self.age_weeks = int(round(float(value) * 52))

    @property
    def age_years(self):
        return self.age

    def __setstate__(self, state):
        """Migrate annual-style pickles without changing old checkpoints."""
        state = dict(state)
        if "age_weeks" not in state:
            if "age" not in state:
                raise ValueError("Person checkpoint state has no age field")
            state["age_weeks"] = int(round(float(state.pop("age")) * 52))
        else:
            state.pop("age", None)
        # Legacy checkpoints contain no reliable maternal birth history.
        state.setdefault("last_birth_step", None)
        state.setdefault("retired", False)
        state.setdefault("equity_holdings", {})
        state.setdefault("equity_cost_basis", {})
        state.setdefault("tax_year_index", 0)
        state.setdefault("cumulative_taxable_wage", 0.0)
        state.setdefault("cumulative_tax_withheld", 0.0)
        state.setdefault("estimated_or_accrued_annual_liability", 0.0)
        state.setdefault("last_personal_tax_this_step", 0.0)
        self.__dict__.update(state)

    def weeks_since_last_birth(self, current_global_step):
        if self.last_birth_step is None:
            return None
        return max(0, int(current_global_step) - int(self.last_birth_step))

    def birth_spacing_eligible(self, current_global_step, minimum_weeks=52):
        weeks = self.weeks_since_last_birth(current_global_step)
        return weeks is None or weeks >= minimum_weeks


    # =========================
    # Age grows
    # =========================

    def grow(self):

        self.age_weeks += 1


    # =========================
    # Death probability
    # =========================

    def death_probability(self):

        shock = getattr(
            self,
            "mortality_multiplier",
            1.0
        )


        age = self.age_years
        annual_hazard = base_mortality_hazard(age)
        adjusted_annual_hazard = annual_hazard * shock

        # Mortality is modeled as an annual hazard, while each simulation
        # opportunity is one week.  Apply shocks before weekly conversion.
        return annual_hazard_to_step_probability(
            adjusted_annual_hazard
        )

    def mortality_diagnostics(self):
        """Return mortality units without changing the death decision."""
        age_years = self.age_years
        shock = getattr(self, "mortality_multiplier", 1.0)
        annual_hazard = base_mortality_hazard(age_years)
        adjusted_hazard = annual_hazard * shock
        return {
            "age_years": age_years,
            "age_weeks": self.age_weeks,
            "annual_mortality_hazard": annual_hazard,
            "weekly_mortality_probability": (
                annual_hazard_to_step_probability(adjusted_hazard)
            ),
            "mortality_shock_multiplier": shock,
            "adjusted_annual_mortality_hazard": adjusted_hazard,
        }



    # =========================
    # Death event
    # =========================

    def check_death(self):

        if random.random() < self.death_probability():

            self.alive = False



    # =========================
    # Reproduction
    # =========================

    def can_reproduce(self):

        return (

            self.sex == "F"

            and

            20 <= self.age <= 40

        )

    # =========================
    # Family relation
    # =========================

    def add_child(self, child_id):

        if child_id not in self.children_ids:

            self.children_ids.append(child_id)



    def remove_child(self, child_id):

        if child_id in self.children_ids:

            self.children_ids.remove(child_id)



    def add_parent(self, parent_id):

        if parent_id not in self.parent_ids:

            self.parent_ids.append(parent_id)
