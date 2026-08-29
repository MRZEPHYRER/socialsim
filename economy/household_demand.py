"""Passive household demand allocation contracts.

This module deliberately stops at currency-budget allocation.  It does not
choose a supplier, execute a purchase, or mutate a Household.
"""

from dataclasses import asdict, dataclass
import math


TOLERANCE = 1e-12


def _nonnegative(value):
    try:
        value = float(value)
    except (TypeError, ValueError):
        return 0.0
    if not math.isfinite(value):
        return 0.0
    return max(0.0, value)


@dataclass(frozen=True)
class HouseholdDemandAllocation:
    """Serializable shadow allocation for one Household and one market step."""

    household_id: object
    total_consumption_budget: float
    necessary_food_budget: float
    discretionary_budget: float
    food_budget: float
    service_budget: float
    unallocated_budget: float
    budget_constraint_status: str

    def to_dict(self):
        return asdict(self)

    @property
    def budget_identity_gap(self):
        return (
            self.total_consumption_budget
            - self.necessary_food_budget
            - self.discretionary_budget
        )

    @property
    def conservation_gap(self):
        return (
            self.food_budget
            + self.service_budget
            + self.unallocated_budget
            - self.total_consumption_budget
        )


class HouseholdDemandSystem:
    """Read-only boundary between a Food-only budget and future goods."""

    VERSION = 1

    def __init__(self, world):
        self.world = world

    @staticmethod
    def _status(household, total_budget, necessary_food_budget):
        if bool(getattr(household, "budget_constrained_this_step", False)):
            return "budget_constrained"
        if total_budget <= TOLERANCE:
            return "zero_budget"
        if necessary_food_budget < total_budget - TOLERANCE:
            return "discretionary_budget_available"
        return "necessary_food_only"

    def household_food_requirement(self, household):
        """Return (units, price, currency requirement) from canonical fields.

        The currency field is authoritative when the existing Food executor has
        already populated it.  The fallback is only for passive compatibility
        with old/incomplete household objects.
        """
        units = _nonnegative(
            getattr(household, "food_need_units_this_step", 0.0)
        )
        price = _nonnegative(
            getattr(household, "household_planning_price_used", 0.0)
        )
        currency_requirement = getattr(
            household,
            "necessary_consumption_this_step",
            None,
        )
        if currency_requirement is None:
            currency_requirement = units * price
        currency_requirement = _nonnegative(currency_requirement)
        return units, price, currency_requirement

    def allocate(
        self,
        household,
        total_consumption_budget=None,
        service_share=0.0,
    ):
        """Build one deterministic shadow allocation without mutating state.

        ``total_consumption_budget`` should be the canonical executed Food
        expenditure for the current step.  In the current one-good economy
        that is also the observed consumption budget available for this shadow
        decomposition.
        """
        if total_consumption_budget is None:
            total_consumption_budget = getattr(
                household, "consumption_this_step", 0.0
            )
        total_budget = _nonnegative(total_consumption_budget)
        _, _, food_requirement = self.household_food_requirement(household)
        necessary_food_budget = min(total_budget, food_requirement)
        discretionary_budget = max(
            total_budget - necessary_food_budget,
            0.0,
        )
        service_share = min(1.0, max(0.0, float(service_share)))
        service_budget = discretionary_budget * service_share
        food_budget = max(
            0.0,
            necessary_food_budget + discretionary_budget - service_budget,
        )
        # The construction above is exact in intent.  Assign the residual to
        # the non-service bucket so serialized rows close even after a long
        # floating-point run.
        unallocated_budget = max(
            0.0,
            total_budget - food_budget - service_budget,
        )
        if abs(unallocated_budget) <= TOLERANCE:
            unallocated_budget = 0.0
        return HouseholdDemandAllocation(
            household_id=getattr(household, "id", None),
            total_consumption_budget=total_budget,
            necessary_food_budget=necessary_food_budget,
            discretionary_budget=discretionary_budget,
            food_budget=food_budget,
            service_budget=service_budget,
            unallocated_budget=unallocated_budget,
            budget_constraint_status=self._status(
                household,
                total_budget,
                necessary_food_budget,
            ),
        )

    def reference_allocations(self, household, total_consumption_budget=None):
        """Return diagnostic-only R25/R50/R75 allocations."""
        return {
            "R25": self.allocate(
                household,
                total_consumption_budget,
                service_share=0.25,
            ),
            "R50": self.allocate(
                household,
                total_consumption_budget,
                service_share=0.50,
            ),
            "R75": self.allocate(
                household,
                total_consumption_budget,
                service_share=0.75,
            ),
        }

