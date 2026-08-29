"""Conservative deterministic primary-equity transition adapter."""

from dataclasses import dataclass, field
import math

from economy.ledger import object_account


TRANSITION_SCHEMA = "gradual_person_primary_equity_transition_v1"


@dataclass
class EquityTransitionResult:
    step: int
    reviewed: bool = False
    issuance_price: float = 0.0
    fills: list = field(default_factory=list)
    total_issued: float = 0.0
    total_shares_issued: float = 0.0
    total_household_cash_used: float = 0.0
    total_firm_cash_raised: float = 0.0
    money_created: float = 0.0
    money_destroyed: float = 0.0
    reconciliation_gap: float = 0.0


class GradualPersonEquityTransition:
    """Issue small primary-share subscriptions without RNG or leverage."""

    def __init__(
        self,
        world,
        issuance_price=10.0,
        review_interval_weeks=52,
        max_household_available_cash_fraction=0.01,
        max_firm_opening_base_fraction=0.0001,
        max_buyers_per_firm=3,
    ):
        if issuance_price <= 0:
            raise ValueError("issuance_price must be positive")
        self.world = world
        self.issuance_price = float(issuance_price)
        self.review_interval_weeks = max(1, int(review_interval_weeks))
        self.max_household_available_cash_fraction = max(
            0.0, float(max_household_available_cash_fraction)
        )
        self.max_firm_opening_base_fraction = max(
            0.0, float(max_firm_opening_base_fraction)
        )
        self.max_buyers_per_firm = max(1, int(max_buyers_per_firm))
        self.last_review_step = None

    def _firms_and_payers(self):
        firms = list(getattr(self.world, "firms", []))
        if len(firms) <= 1:
            return [(firms[0] if firms else self.world.firm_system, self.world.firm_system)]
        return [(firm, firm) for firm in firms]

    def _target_reserve(self, household):
        target = max(
            0.0,
            float(getattr(household, "target_wealth_this_step", 0.0)),
        )
        if target <= 0.0:
            target = max(
                0.0,
                float(getattr(household, "necessary_consumption_this_step", 0.0))
                * 26.0,
            )
        return target

    def _eligible_buyers(self, firm_id, remaining_capacity):
        owners = []
        used_households = set()
        for person in sorted(
            getattr(self.world, "population", []),
            key=lambda item: (
                getattr(item, "age_weeks", 0),
                getattr(item, "id", 0),
            ),
        ):
            if not getattr(person, "alive", False):
                continue
            age_weeks = float(getattr(person, "age_weeks", 0.0))
            if age_weeks < 18.0 * 52.0 or age_weeks >= 65.0 * 52.0:
                continue
            household = getattr(self.world, "household_dict", {}).get(
                getattr(person, "household_id", None)
            )
            if household is None or household.id in used_households:
                continue
            # Keep the short transition screen from selecting a one-person
            # household whose equity asset could be removed by the existing
            # empty-household cleanup after the owner dies.  This is a
            # conservative eligibility filter, not a new ownership rule.
            if household.size() < 2:
                continue
            available = max(
                0.0,
                float(getattr(household, "wealth", 0.0))
                - self._target_reserve(household),
            )
            payment_cap = available * self.max_household_available_cash_fraction
            if payment_cap <= 1e-12:
                continue
            owners.append((person, household, min(payment_cap, remaining_capacity)))
            used_households.add(household.id)
            if len(owners) >= self.max_buyers_per_firm:
                break
            if sum(item[2] for item in owners) >= remaining_capacity - 1e-12:
                break
        return owners

    def review(self, step):
        result = EquityTransitionResult(
            step=int(step),
            issuance_price=self.issuance_price,
        )
        for household in getattr(self.world, "households", []):
            household.equity_purchase_cash_outflow_this_step = 0.0
        for target in [
            getattr(self.world, "firm_system", None),
            *getattr(self.world, "firms", []),
        ]:
            if target is None:
                continue
            target.equity_issuance_cash_this_step = 0.0
            target.equity_issuance_shares_this_step = 0.0
            target.equity_issuance_buyer_count = 0
        if step % self.review_interval_weeks != 0:
            self.world.last_equity_transition_result = result
            return result
        result.reviewed = True
        self.last_review_step = int(step)
        if hasattr(self.world, "ensure_ownership_state"):
            self.world.ensure_ownership_state()

        for firm, payer in self._firms_and_payers():
            table = getattr(firm, "cap_table", None)
            if table is None:
                continue
            opening_cash = max(0.0, float(getattr(payer, "cash", 0.0)))
            paid_in = max(0.0, float(getattr(firm, "paid_in_equity", 0.0)))
            opening_base = opening_cash + paid_in
            remaining = opening_base * self.max_firm_opening_base_fraction
            if remaining <= 1e-12:
                continue
            firm_issued = 0.0
            buyers = self._eligible_buyers(
                getattr(firm, "firm_id", 0),
                remaining,
            )
            for fill_index, (person, household, payment_cap) in enumerate(buyers):
                payment = min(payment_cap, remaining)
                if payment <= 1e-12:
                    continue
                shares = payment / self.issuance_price
                before_cash = float(getattr(household, "wealth", 0.0))
                before_equity = float(
                    getattr(household, "equity_asset_value", 0.0)
                )
                before_firm_cash = float(getattr(payer, "cash", 0.0))
                self.world.ledger.transfer_attrs(
                    payer_obj=household,
                    payer_attr="wealth",
                    payer_name=f"household.{household.id}.wealth",
                    receiver_obj=payer,
                    receiver_attr="cash",
                    receiver_name=f"firm[{getattr(firm, 'firm_id', 0)}].cash",
                    amount=payment,
                    reason="primary_equity_issuance",
                )
                household.equity_asset_value = before_equity + payment
                person.equity_cost_basis = dict(
                    getattr(person, "equity_cost_basis", {})
                )
                person.equity_cost_basis[getattr(firm, "firm_id", 0)] = (
                    person.equity_cost_basis.get(getattr(firm, "firm_id", 0), 0.0)
                    + payment
                )
                household.equity_purchase_cash_outflow_this_step = (
                    getattr(
                        household,
                        "equity_purchase_cash_outflow_this_step",
                        0.0,
                    )
                    + payment
                )
                person.equity_holdings = dict(
                    getattr(person, "equity_holdings", {})
                )
                person.equity_holdings[getattr(firm, "firm_id", 0)] = (
                    person.equity_holdings.get(getattr(firm, "firm_id", 0), 0.0)
                    + shares
                )
                firm.cap_table = table = table.with_primary_issuance(
                    person.id,
                    shares,
                    "person",
                )
                if hasattr(self.world, "equity_ownership_system"):
                    self.world.equity_ownership_system.cap_tables[
                        getattr(firm, "firm_id", 0)
                    ] = table
                firm_issued += payment
                new_paid_in = paid_in + firm_issued
                for equity_target in (firm, payer):
                    equity_target.paid_in_equity = new_paid_in
                remaining -= payment
                result.total_issued += payment
                result.total_shares_issued += shares
                result.total_household_cash_used += payment
                result.total_firm_cash_raised += (
                    float(getattr(payer, "cash", 0.0)) - before_firm_cash
                )
                fill = {
                    "step": int(step),
                    "firm_id": getattr(firm, "firm_id", 0),
                    "person_id": person.id,
                    "household_id": household.id,
                    "ownership_fraction_after": table.ownership_fraction("person"),
                    "issuance_price": self.issuance_price,
                    "payment": payment,
                    "shares_issued": shares,
                    "household_cash_before": before_cash,
                    "household_cash_after": household.wealth,
                    "household_equity_before": before_equity,
                    "household_equity_after": household.equity_asset_value,
                    "firm_cash_before": before_firm_cash,
                    "firm_cash_after": payer.cash,
                }
                result.fills.append(fill)
                remaining -= 0.0
                if remaining <= 1e-12:
                    break

            for target in (firm, payer):
                target.equity_issuance_cash_this_step = sum(
                    fill["payment"]
                    for fill in result.fills
                    if fill["firm_id"] == getattr(firm, "firm_id", 0)
                )
                target.equity_issuance_shares_this_step = sum(
                    fill["shares_issued"]
                    for fill in result.fills
                    if fill["firm_id"] == getattr(firm, "firm_id", 0)
                )
                target.equity_issuance_buyer_count = sum(
                    fill["firm_id"] == getattr(firm, "firm_id", 0)
                    for fill in result.fills
                )
                target.cash_end = getattr(target, "cash", 0.0)

        result.reconciliation_gap = (
            result.total_firm_cash_raised - result.total_household_cash_used
        )
        self.world.last_equity_transition_result = result
        return result


__all__ = ["EquityTransitionResult", "GradualPersonEquityTransition", "TRANSITION_SCHEMA"]
