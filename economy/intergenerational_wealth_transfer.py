"""Default-off wealth-based adult-child to parent household transfers."""
from __future__ import annotations
from dataclasses import dataclass, field
import math

EPSILON = 1e-12

@dataclass
class WealthTransferResult:
    step: int
    execution_status: str = "INACTIVE"
    eligible_donor_households: int = 0
    donor_households_used: int = 0
    recipient_households: int = 0
    transfer_event_count: int = 0
    total_transfer: float = 0.0
    mean_transfer: float = 0.0
    median_transfer: float = 0.0
    donor_reserve_violations: int = 0
    recipient_target_overshoot_violations: int = 0
    family_edge_count_used: int = 0
    transfers: list[dict] = field(default_factory=list)

def _number(value, default=0.0):
    try:
        value = float(value)
        return value if math.isfinite(value) else default
    except (TypeError, ValueError):
        return default

class IntergenerationalWealthTransferSystem:
    """Deterministic wealth allocation, separate from needs-based support."""

    def __init__(self, world):
        self.world = world
        self.last_result = WealthTransferResult(step=0)
        self.history = []

    def _social_household(self, person):
        household = self.world.get_household(getattr(person, "household_id", None))
        if household is None or getattr(household, "settlement_only", False):
            return None
        return household

    def _minimum_cost(self, household):
        units = self.world.needs_system.household_minimum_need_units(household)
        try:
            price = max(1e-12, float(self.world.household_planning_price_this_step()))
        except (AttributeError, TypeError, ValueError):
            price = 1.0
        return max(0.0, _number(units)) * price

    def _edges(self):
        edges = set()
        for person in sorted(getattr(self.world, "population", []), key=lambda item: item.id):
            if not getattr(person, "alive", False) or _number(getattr(person, "age", 0.0)) < 20.0:
                continue
            donor = self._social_household(person)
            if donor is None:
                continue
            for parent_id in sorted(set(getattr(person, "parent_ids", []) or [])):
                parent = self.world.get_person_by_id(parent_id)
                if parent is None or not getattr(parent, "alive", False):
                    continue
                recipient = self._social_household(parent)
                if recipient is None or recipient.id == donor.id:
                    continue
                edges.add((donor.id, recipient.id))
        return sorted(edges)

    def execute(self, step):
        for household in getattr(self.world, "households", []):
            household.wealth_transfer_paid_this_step = 0.0
            household.wealth_transfer_received_this_step = 0.0
            household.intergenerational_transfer_income_this_step = 0.0
        result = WealthTransferResult(step=step)
        if not getattr(self.world, "intergenerational_wealth_transfer_enabled", False):
            self.last_result = result
            self.history.append(self.weekly_row(result))
            return result

        reserve_weeks = max(0.0, _number(getattr(self.world, "intergenerational_reserve_weeks", 13.0)))
        donor_share = max(0.0, min(1.0, _number(getattr(self.world, "intergenerational_donor_surplus_share", 0.25))))
        target_weeks = max(0.0, _number(getattr(self.world, "intergenerational_recipient_target_weeks", 1.0)))
        edges = self._edges()
        result.family_edge_count_used = len(edges)
        donors = {}
        recipients = {}
        donor_reserves = {}
        recipient_targets = {}
        for donor_id, recipient_id in edges:
            donor = self.world.get_household(donor_id)
            recipient = self.world.get_household(recipient_id)
            if donor is None or recipient is None:
                continue
            donor_minimum = self._minimum_cost(donor)
            recipient_minimum = self._minimum_cost(recipient)
            donor_reserves[donor_id] = reserve_weeks * donor_minimum
            recipient_targets[recipient_id] = target_weeks * recipient_minimum
            donors[donor_id] = max(0.0, (_number(donor.wealth) - donor_reserves[donor_id]) * donor_share)
            recipients[recipient_id] = max(0.0, recipient_targets[recipient_id] - _number(recipient.wealth))
        result.eligible_donor_households = sum(value > EPSILON for value in donors.values())
        result.recipient_households = sum(value > EPSILON for value in recipients.values())

        remaining_donors = dict(donors)
        remaining_recipients = dict(recipients)
        for donor_id, recipient_id in edges:
            amount = min(remaining_donors.get(donor_id, 0.0), remaining_recipients.get(recipient_id, 0.0))
            if amount <= EPSILON:
                continue
            donor = self.world.get_household(donor_id)
            recipient = self.world.get_household(recipient_id)
            amount = min(amount, max(0.0, _number(donor.wealth)))
            if amount <= EPSILON:
                continue
            donor_before = _number(donor.wealth)
            recipient_before = _number(recipient.wealth)
            self.world.ledger.transfer_attrs(
                payer_obj=donor, payer_attr="wealth",
                payer_name=f"household.{donor.id}.wealth",
                receiver_obj=recipient, receiver_attr="wealth",
                receiver_name=f"household.{recipient.id}.wealth",
                amount=amount, reason="intergenerational_wealth_transfer",
            )
            donor.wealth_transfer_paid_this_step += amount
            recipient.wealth_transfer_received_this_step += amount
            recipient.intergenerational_transfer_income_this_step += amount
            recipient.income_this_step += amount
            remaining_donors[donor_id] -= amount
            remaining_recipients[recipient_id] -= amount
            result.transfers.append({
                "global_step": step,
                "donor_household_id": donor.id,
                "recipient_household_id": recipient.id,
                "amount": amount,
                "donor_cash_before": donor_before,
                "donor_cash_after": donor.wealth,
                "recipient_cash_before": recipient_before,
                "recipient_cash_after": recipient.wealth,
                "donor_reserve_cash": donor_reserves[donor.id],
                "recipient_target_cash": recipient_targets[recipient.id],
                "execution_point": "after_payg_before_food_market",
            })

        amounts = [row["amount"] for row in result.transfers]
        result.transfer_event_count = len(amounts)
        result.donor_households_used = len({row["donor_household_id"] for row in result.transfers})
        result.total_transfer = sum(amounts)
        result.mean_transfer = result.total_transfer / len(amounts) if amounts else 0.0
        result.median_transfer = sorted(amounts)[len(amounts) // 2] if amounts else 0.0
        for row in result.transfers:
            # Validate the transfer contract at settlement time, before later
            # household consumption changes the donor or recipient cash stock.
            if row["donor_cash_after"] < row["donor_reserve_cash"] - 1e-9:
                result.donor_reserve_violations += 1
            if row["recipient_cash_after"] > row["recipient_target_cash"] + 1e-9:
                result.recipient_target_overshoot_violations += 1
        result.execution_status = "ACTIVE"
        self.last_result = result
        self.history.append(self.weekly_row(result))
        return result

    def weekly_row(self, result):
        return {
            "global_step": result.step,
            "execution_status": result.execution_status,
            "eligible_donor_households": result.eligible_donor_households,
            "donor_households_used": result.donor_households_used,
            "recipient_households": result.recipient_households,
            "transfer_event_count": result.transfer_event_count,
            "total_transfer": result.total_transfer,
            "mean_transfer": result.mean_transfer,
            "median_transfer": result.median_transfer,
            "donor_reserve_violations": result.donor_reserve_violations,
            "recipient_target_overshoot_violations": result.recipient_target_overshoot_violations,
            "family_edge_count_used": result.family_edge_count_used,
        }

