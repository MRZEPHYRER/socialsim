"""Controlled primary-equity issuance fixture.

The simulation never calls this module.  It exists for Step 15E.2 to test a
single, deterministic Firm financing event against the runtime ownership and
household accounting state without enabling portfolio behavior.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Tuple

from economy.ownership_accounting import CapTable, GenericFinancialEvent


@dataclass(frozen=True)
class EquityIssuanceFill:
    issuing_firm_id: Any
    buyer_person_id: Any
    buyer_household_id: Any
    issuance_price: float
    shares_issued: float
    payment: float


@dataclass
class ControlledPrimaryIssuanceResult:
    issuing_firm_id: Any
    issuance_price: float
    fills: List[EquityIssuanceFill] = field(default_factory=list)
    events: List[GenericFinancialEvent] = field(default_factory=list)
    cap_table_before: CapTable = None
    cap_table_after: CapTable = None
    firm_cash_before: float = 0.0
    firm_cash_after: float = 0.0
    paid_in_equity_before: float = 0.0
    paid_in_equity_after: float = 0.0
    household_cash_before: Dict[Any, float] = field(default_factory=dict)
    household_cash_after: Dict[Any, float] = field(default_factory=dict)
    household_equity_before: Dict[Any, float] = field(default_factory=dict)
    household_equity_after: Dict[Any, float] = field(default_factory=dict)

    @property
    def total_payment(self):
        return sum(fill.payment for fill in self.fills)

    @property
    def total_shares_issued(self):
        return sum(fill.shares_issued for fill in self.fills)

    @property
    def money_created(self):
        return 0.0

    @property
    def money_destroyed(self):
        return 0.0


class ControlledPrimaryEquityIssuer:
    """Apply one small deterministic primary issuance to supplied objects.

    ``buyers`` is an ordered iterable of ``(Person, Household)`` pairs.  The
    caller chooses the buyers; this class never samples RNG and never selects
    buyers from the population.
    """

    def __init__(
        self,
        issuance_price=10.0,
        max_household_payment_fraction=0.01,
        max_firm_proceeds_fraction=0.0025,
    ):
        if issuance_price <= 0:
            raise ValueError("issuance_price must be positive")
        self.issuance_price = float(issuance_price)
        self.max_household_payment_fraction = float(
            max_household_payment_fraction
        )
        self.max_firm_proceeds_fraction = float(max_firm_proceeds_fraction)

    def execute(self, firm, buyers: Iterable[Tuple[Any, Any]]):
        buyers = list(buyers)
        if not buyers:
            raise ValueError("controlled issuance requires at least one buyer")

        person_ids = [person.id for person, _ in buyers]
        household_ids = [household.id for _, household in buyers]
        if len(set(person_ids)) != len(person_ids):
            raise ValueError("buyer persons must be distinct")
        if len(set(household_ids)) != len(household_ids):
            raise ValueError("buyers must come from distinct households")

        firm_id = getattr(firm, "firm_id", 0)
        table_before = getattr(firm, "cap_table", None)
        if table_before is None:
            table_before = CapTable.initial_legacy_owned(firm_id)
            firm.cap_table = table_before
        if table_before.firm_id != firm_id:
            raise ValueError("firm cap table owner does not match issuing firm")

        firm_cash_before = float(getattr(firm, "cash", 0.0))
        paid_in_before = float(getattr(firm, "paid_in_equity", 0.0))
        household_cash_before = {
            household.id: float(getattr(household, "wealth", 0.0))
            for _, household in buyers
        }
        household_equity_before = {
            household.id: float(getattr(household, "equity_asset_value", 0.0))
            for _, household in buyers
        }

        per_household_cap = min(
            cash * self.max_household_payment_fraction
            for cash in household_cash_before.values()
        )
        firm_proceeds_cap = (
            max(0.0, firm_cash_before)
            * self.max_firm_proceeds_fraction
        )
        total_proceeds = min(
            per_household_cap * len(buyers),
            firm_proceeds_cap,
        )
        if total_proceeds <= 0.0:
            raise ValueError("issuance constraints leave no feasible proceeds")
        payment = total_proceeds / len(buyers)

        result = ControlledPrimaryIssuanceResult(
            issuing_firm_id=firm_id,
            issuance_price=self.issuance_price,
            cap_table_before=table_before,
            firm_cash_before=firm_cash_before,
            paid_in_equity_before=paid_in_before,
            household_cash_before=dict(household_cash_before),
            household_equity_before=dict(household_equity_before),
        )

        table = table_before
        for fill_index, (person, household) in enumerate(buyers):
            shares = payment / self.issuance_price
            table = table.with_primary_issuance(person.id, shares, "person")

            household.wealth -= payment
            household.equity_asset_value = (
                household_equity_before[household.id] + payment
            )
            firm.cash += payment
            firm.paid_in_equity = paid_in_before + result.total_payment + payment
            holdings = getattr(person, "equity_holdings", None)
            if holdings is None:
                holdings = {}
                person.equity_holdings = holdings
            holdings[firm_id] = holdings.get(firm_id, 0.0) + shares

            result.fills.append(
                EquityIssuanceFill(
                    issuing_firm_id=firm_id,
                    buyer_person_id=person.id,
                    buyer_household_id=household.id,
                    issuance_price=self.issuance_price,
                    shares_issued=shares,
                    payment=payment,
                )
            )
            result.events.extend([
                GenericFinancialEvent(
                    event_id=f"step15e2:{firm_id}:{fill_index}:buyer",
                    actor=f"household:{household.id}",
                    counterparty=f"firm:{firm_id}",
                    transaction_type="primary_equity_purchase",
                    cash_change=-payment,
                    asset_change={"cash": -payment, "equity_asset": payment},
                    equity_change=0.0,
                    income_expense_classification="none",
                    cash_flow_classification="CFI",
                    reference_event_id=f"step15e2:{firm_id}:{fill_index}",
                    metadata={"person_id": person.id, "shares": shares},
                ),
                GenericFinancialEvent(
                    event_id=f"step15e2:{firm_id}:{fill_index}:firm",
                    actor=f"firm:{firm_id}",
                    counterparty=f"household:{household.id}",
                    transaction_type="primary_equity_issuance",
                    cash_change=payment,
                    asset_change={"cash": payment},
                    equity_change=payment,
                    income_expense_classification="none",
                    cash_flow_classification="CFF",
                    reference_event_id=f"step15e2:{firm_id}:{fill_index}",
                    metadata={"person_id": person.id, "shares": shares},
                ),
            ])

        firm.cap_table = table
        result.cap_table_after = table
        result.firm_cash_after = float(firm.cash)
        result.paid_in_equity_after = float(firm.paid_in_equity)
        result.household_cash_after = {
            household.id: float(getattr(household, "wealth", 0.0))
            for _, household in buyers
        }
        result.household_equity_after = {
            household.id: float(getattr(household, "equity_asset_value", 0.0))
            for _, household in buyers
        }
        return result


__all__ = [
    "ControlledPrimaryEquityIssuer",
    "ControlledPrimaryIssuanceResult",
    "EquityIssuanceFill",
]
