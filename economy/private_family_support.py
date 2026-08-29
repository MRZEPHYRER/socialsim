"""Default-off private child-Household to parent-Household support.

This module is deliberately separate from the legacy ``IncomeSystem``.  It
uses existing Household cash accounts and the World Ledger, so support is a
real private transfer rather than an unfunded income allocation.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class PrivateFamilySupportResult:
    step: int
    total_paid: float = 0.0
    total_received: float = 0.0
    parent_need_total: float = 0.0
    child_capacity_total: float = 0.0
    eligible_parent_households: int = 0
    eligible_child_households: int = 0
    transfer_count: int = 0
    transfers: list[dict[str, Any]] = field(default_factory=list)
    execution_status: str = "INACTIVE"


class PrivateFamilySupportSystem:
    """Deterministic, cash-conserving experimental support system."""

    EPSILON = 1e-12

    def __init__(self, world):
        self.world = world
        self.last_result = PrivateFamilySupportResult(step=0)
        self.transfer_history: list[dict[str, Any]] = []

    def _social_household(self, person):
        household = self.world.get_household(getattr(person, "household_id", None))
        if household is None or getattr(household, "settlement_only", False):
            return None
        return household

    def _price(self):
        try:
            return max(1e-12, float(self.world.household_planning_price_this_step()))
        except (AttributeError, TypeError, ValueError):
            return 1.0

    def _minimum_cost(self, household, price):
        try:
            units = self.world.needs_system.household_minimum_need_units(household)
        except (AttributeError, TypeError, ValueError):
            units = 0.0
        return max(0.0, float(units)) * price

    @staticmethod
    def _reset_household_fields(world):
        for household in getattr(world, "households", []):
            household.private_support_received_this_step = 0.0
            household.private_support_paid_this_step = 0.0
            household.net_interhousehold_transfer_this_step = 0.0
            household.private_support_need_this_step = 0.0
            household.private_support_capacity_this_step = 0.0
            household.private_support_adjusted_opening_wealth = getattr(
                household, "wealth_before_income_this_step", household.wealth
            )

    def _empty(self, step, status="INACTIVE"):
        self._reset_household_fields(self.world)
        self.last_result = PrivateFamilySupportResult(
            step=step,
            execution_status=status,
        )
        return self.last_result

    def _parent_needs(self, price):
        parents = {}
        for person in getattr(self.world, "population", []):
            if not getattr(person, "alive", False):
                continue
            parent_household = self._social_household(person)
            if parent_household is None:
                continue
            parent_cost = self._minimum_cost(parent_household, price)
            need = max(0.0, parent_cost - float(getattr(parent_household, "wealth", 0.0)))
            # The first controlled institution targets parental/elderly need,
            # while still allowing a clearly measured low-income parent gap.
            if need <= self.EPSILON and float(getattr(person, "age", 0.0)) < 65.0:
                continue
            parents[parent_household.id] = {
                "household": parent_household,
                "need": need,
                "member_ids": set(),
            }
            parents[parent_household.id]["member_ids"].add(person.id)
        return parents

    def _payer_capacities(self, price, parent_edges):
        capacities = {}
        for payer_id in sorted(parent_edges):
            household = self.world.get_household(payer_id)
            if household is None or getattr(household, "settlement_only", False):
                continue
            minimum_cost = self._minimum_cost(household, price)
            capacity = max(0.0, float(getattr(household, "wealth", 0.0)) - minimum_cost)
            household.private_support_capacity_this_step = capacity
            if capacity > self.EPSILON:
                capacities[payer_id] = capacity
        return capacities

    def _build_edges(self, parents):
        edges = {}
        for child in sorted(getattr(self.world, "population", []), key=lambda item: item.id):
            if not getattr(child, "alive", False) or float(getattr(child, "age", 0.0)) < 20.0:
                continue
            payer = self._social_household(child)
            if payer is None:
                continue
            for parent_id in sorted(set(getattr(child, "parent_ids", []))):
                parent = self.world.get_person_by_id(parent_id)
                if parent is None or not getattr(parent, "alive", False):
                    continue
                recipient = self._social_household(parent)
                if recipient is None or recipient.id == payer.id or recipient.id not in parents:
                    continue
                edges.setdefault((payer.id, recipient.id), {
                    "payer": payer,
                    "recipient": recipient,
                    "child_ids": set(),
                    "parent_ids": set(),
                })
                edges[(payer.id, recipient.id)]["child_ids"].add(child.id)
                edges[(payer.id, recipient.id)]["parent_ids"].add(parent.id)
        return edges

    def _allocate(self, parents, capacities, edges):
        """Allocate capacity proportionally without first-child exhaustion."""
        remaining_capacity = dict(capacities)
        remaining_need = {
            parent_id: max(0.0, data["need"])
            for parent_id, data in parents.items()
        }
        allocation = {key: 0.0 for key in edges}

        for _ in range(max(1, len(edges) + len(parents))):
            proposals = {key: 0.0 for key in edges}
            for payer_id in sorted(remaining_capacity):
                available = remaining_capacity[payer_id]
                if available <= self.EPSILON:
                    continue
                connected = [
                    (key, remaining_need[key[1]])
                    for key in edges
                    if key[0] == payer_id and remaining_need.get(key[1], 0.0) > self.EPSILON
                ]
                total_need = sum(need for _, need in connected)
                if total_need <= self.EPSILON:
                    continue
                for key, need in connected:
                    proposals[key] = available * need / total_need

            if not any(value > self.EPSILON for value in proposals.values()):
                break

            actual = {key: value for key, value in proposals.items()}
            for parent_id in sorted(remaining_need):
                incoming = [key for key in edges if key[1] == parent_id]
                total = sum(proposals[key] for key in incoming)
                if total > remaining_need[parent_id] + self.EPSILON:
                    scale = remaining_need[parent_id] / total if total else 0.0
                    for key in incoming:
                        actual[key] *= scale

            progress = 0.0
            for key, amount in actual.items():
                amount = max(0.0, amount)
                if amount <= self.EPSILON:
                    continue
                amount = min(amount, remaining_capacity.get(key[0], 0.0))
                amount = min(amount, remaining_need.get(key[1], 0.0))
                allocation[key] += amount
                remaining_capacity[key[0]] -= amount
                remaining_need[key[1]] -= amount
                progress += amount
            if progress <= self.EPSILON:
                break
        return allocation

    def execute(self, step):
        if not getattr(self.world, "private_family_support_enabled", False):
            return self._empty(step)

        self._reset_household_fields(self.world)
        price = self._price()
        parents = self._parent_needs(price)
        edges = self._build_edges(parents)
        parent_edges = {}
        for payer_id, recipient_id in edges:
            parent_edges.setdefault(payer_id, set()).add(recipient_id)
        capacities = self._payer_capacities(price, parent_edges)
        allocation = self._allocate(parents, capacities, edges)
        result = PrivateFamilySupportResult(
            step=step,
            parent_need_total=sum(data["need"] for data in parents.values()),
            child_capacity_total=sum(capacities.values()),
            eligible_parent_households=len(parents),
            eligible_child_households=len(capacities),
            execution_status="ACTIVE",
        )

        for key in sorted(allocation):
            amount = allocation[key]
            if amount <= self.EPSILON:
                continue
            edge = edges[key]
            payer = edge["payer"]
            recipient = edge["recipient"]
            amount = min(amount, max(0.0, float(getattr(payer, "wealth", 0.0))))
            if amount <= self.EPSILON:
                continue
            payer_before = float(payer.wealth)
            recipient_before = float(recipient.wealth)
            self.world.ledger.transfer_attrs(
                payer_obj=payer,
                payer_attr="wealth",
                payer_name=f"household.{payer.id}.wealth",
                receiver_obj=recipient,
                receiver_attr="wealth",
                receiver_name=f"household.{recipient.id}.wealth",
                amount=amount,
                reason="private_family_support",
            )
            payer.private_support_paid_this_step += amount
            payer.net_interhousehold_transfer_this_step -= amount
            recipient.private_support_received_this_step += amount
            recipient.net_interhousehold_transfer_this_step += amount
            record = {
                "global_step": step,
                "payer_household_id": payer.id,
                "recipient_household_id": recipient.id,
                "child_person_ids": ";".join(map(str, sorted(edge["child_ids"]))),
                "parent_person_ids": ";".join(map(str, sorted(edge["parent_ids"]))),
                "amount": amount,
                "payer_cash_before": payer_before,
                "payer_cash_after": payer.wealth,
                "recipient_cash_before": recipient_before,
                "recipient_cash_after": recipient.wealth,
                "parent_need_gap": parents[recipient.id]["need"],
                "payer_available_support_capacity": capacities.get(payer.id, 0.0),
                "accounting_gap": (payer_before + recipient_before - payer.wealth - recipient.wealth),
                "execution_point": "after_payroll_and_dividend_stage_before_food_market",
            }
            result.transfers.append(record)
            self.transfer_history.append(record)
            result.total_paid += amount
            result.total_received += amount

        result.transfer_count = len(result.transfers)
        for parent_id, data in parents.items():
            data["household"].private_support_need_this_step = data["need"]
        self.last_result = result
        return result

    def finalize_after_consumption(self):
        """Put payer outflow into Household saving after final food refunds."""
        result = self.last_result
        for record in result.transfers:
            payer = self.world.get_household(record["payer_household_id"])
            if payer is None:
                continue
            payer.private_support_adjusted_opening_wealth = max(
                0.0,
                float(getattr(payer, "wealth_before_income_this_step", payer.wealth))
                - float(getattr(payer, "private_support_paid_this_step", 0.0)),
            )
        for household in getattr(self.world, "households", []):
            paid = float(getattr(household, "private_support_paid_this_step", 0.0))
            if paid > 0.0:
                household.saving_this_step -= paid


__all__ = ["PrivateFamilySupportSystem", "PrivateFamilySupportResult"]
