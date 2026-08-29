"""Controlled secondary transfer of existing Legacy shares.

This module provides the deterministic settlement boundary used by controlled
fixtures and the optional Step 15G.1 autonomous screen. It does not create an
order book, discover prices, or generate autonomous demand by itself.
"""

from dataclasses import dataclass, field

from economy.dividend_routing import ensure_dividend_routing_state


SECONDARY_TRANSFER_SCHEMA = "controlled_secondary_legacy_share_transfer_v1"


def _person_shares(table, person_id):
    return sum(
        holding.shares
        for holding in table.holdings
        if holding.holder_type == "person" and holding.holder_id == person_id
    )


@dataclass
class SecondaryTransferResult:
    firm_id: object
    price_per_share: float
    fills: list = field(default_factory=list)
    legacy_shares_before: float = 0.0
    legacy_shares_after: float = 0.0
    total_shares_before: float = 0.0
    total_shares_after: float = 0.0
    firm_cash_before: float = 0.0
    firm_cash_after: float = 0.0
    paid_in_equity_before: float = 0.0
    paid_in_equity_after: float = 0.0
    legacy_owner_cash_before: float = 0.0
    legacy_owner_cash_after: float = 0.0
    total_payment: float = 0.0
    total_shares_sold: float = 0.0
    money_created: float = 0.0
    money_destroyed: float = 0.0
    reconciliation_gap: float = 0.0


class ControlledSecondaryLegacyTransfer:
    """Execute caller-specified Legacy share transfers in deterministic order."""

    def __init__(self, world, price_per_share=10.0):
        if float(price_per_share) <= 0.0:
            raise ValueError("price_per_share must be positive")
        self.world = world
        self.price_per_share = float(price_per_share)

    def _payer(self, firm):
        if len(getattr(self.world, "firms", [])) <= 1:
            return self.world.firm_system
        return firm

    def execute(self, firm, buyers):
        ensure_dividend_routing_state(self.world)
        if not hasattr(self.world, "secondary_transfer_history"):
            self.world.secondary_transfer_history = []
        buyers = list(buyers)
        person_ids = [person.id for person, _, _ in buyers]
        household_ids = [household.id for _, household, _ in buyers]
        if len(set(person_ids)) != len(person_ids):
            raise ValueError("secondary buyers must be distinct Persons")
        if len(set(household_ids)) != len(household_ids):
            raise ValueError("secondary buyers must use distinct Households")

        table = getattr(firm, "cap_table", None)
        if table is None:
            raise ValueError("Firm has no CapTable")
        payer = self._payer(firm)
        result = SecondaryTransferResult(
            firm_id=getattr(firm, "firm_id", 0),
            price_per_share=self.price_per_share,
            legacy_shares_before=table.legacy_shares,
            total_shares_before=table.total_shares,
            firm_cash_before=float(getattr(payer, "cash", 0.0)),
            paid_in_equity_before=float(getattr(firm, "paid_in_equity", 0.0)),
            legacy_owner_cash_before=float(
                getattr(self.world, "legacy_owner_cash", 0.0)
            ),
        )

        for person, household, shares_requested in buyers:
            requested = float(shares_requested)
            if requested < 0.0:
                raise ValueError("shares_requested must be non-negative")
            if getattr(person, "household_id", household.id) != household.id:
                raise ValueError("buyer Person and Household relationship mismatch")

            person_before = _person_shares(table, person.id)
            household_cash_before = float(getattr(household, "wealth", 0.0))
            household_equity_before = float(
                getattr(household, "equity_asset_value", 0.0)
            )
            legacy_before = table.legacy_shares
            cash_limited = max(0.0, household_cash_before) / self.price_per_share
            filled = min(requested, legacy_before, cash_limited)
            payment = filled * self.price_per_share
            if filled <= 1e-12:
                result.fills.append({
                    "firm_id": result.firm_id,
                    "buyer_person_id": person.id,
                    "buyer_household_id": household.id,
                    "shares_requested": requested,
                    "price_per_share": self.price_per_share,
                    "payment": 0.0,
                    "shares_filled": 0.0,
                    "household_cash_before": household_cash_before,
                    "household_cash_after": household_cash_before,
                    "legacy_shares_before": legacy_before,
                    "legacy_shares_after": legacy_before,
                    "person_shares_before": person_before,
                    "person_shares_after": person_before,
                })
                continue

            self.world.ledger.transfer_attrs(
                payer_obj=household,
                payer_attr="wealth",
                payer_name=f"household.{household.id}.wealth",
                receiver_obj=self.world,
                receiver_attr="legacy_owner_cash",
                receiver_name="LegacyOwner.cash",
                amount=payment,
                reason="secondary_legacy_share_sale",
            )
            household.equity_asset_value = household_equity_before + payment
            person.equity_cost_basis = dict(
                getattr(person, "equity_cost_basis", {})
            )
            person.equity_cost_basis[result.firm_id] = (
                person.equity_cost_basis.get(result.firm_id, 0.0) + payment
            )
            household.equity_purchase_cash_outflow_this_step = (
                getattr(
                    household,
                    "equity_purchase_cash_outflow_this_step",
                    0.0,
                )
                + payment
            )
            table = table.with_secondary_legacy_transfer(
                person.id,
                filled,
                "person",
            )
            person.equity_holdings = dict(
                getattr(person, "equity_holdings", {})
            )
            person.equity_holdings[result.firm_id] = (
                person.equity_holdings.get(result.firm_id, 0.0) + filled
            )
            household_cash_after = float(getattr(household, "wealth", 0.0))
            person_after = _person_shares(table, person.id)
            result.fills.append({
                "firm_id": result.firm_id,
                "buyer_person_id": person.id,
                "buyer_household_id": household.id,
                "shares_requested": requested,
                "price_per_share": self.price_per_share,
                "payment": payment,
                "shares_filled": filled,
                "household_cash_before": household_cash_before,
                "household_cash_after": household_cash_after,
                "household_equity_before": household_equity_before,
                "household_equity_after": household.equity_asset_value,
                "legacy_shares_before": legacy_before,
                "legacy_shares_after": table.legacy_shares,
                "person_shares_before": person_before,
                "person_shares_after": person_after,
            })
            result.total_payment += payment
            result.total_shares_sold += filled

        firm.cap_table = table
        if hasattr(self.world, "equity_ownership_system"):
            self.world.equity_ownership_system.cap_tables[result.firm_id] = table

        result.legacy_shares_after = table.legacy_shares
        result.total_shares_after = table.total_shares
        result.firm_cash_after = float(getattr(payer, "cash", 0.0))
        result.paid_in_equity_after = float(
            getattr(firm, "paid_in_equity", 0.0)
        )
        result.legacy_owner_cash_after = float(
            getattr(self.world, "legacy_owner_cash", 0.0)
        )
        result.reconciliation_gap = (
            result.total_payment
            - (result.legacy_owner_cash_after - result.legacy_owner_cash_before)
        )
        self.world.secondary_transfer_history.append({
            "schema": SECONDARY_TRANSFER_SCHEMA,
            "firm_id": result.firm_id,
            "price_per_share": result.price_per_share,
            "total_payment": result.total_payment,
            "total_shares_sold": result.total_shares_sold,
            "legacy_shares_before": result.legacy_shares_before,
            "legacy_shares_after": result.legacy_shares_after,
            "total_shares_before": result.total_shares_before,
            "total_shares_after": result.total_shares_after,
            "fills": list(result.fills),
            "reconciliation_gap": result.reconciliation_gap,
        })
        self.world.last_secondary_transfer_result = result
        return result


__all__ = [
    "SECONDARY_TRANSFER_SCHEMA",
    "ControlledSecondaryLegacyTransfer",
    "SecondaryTransferResult",
]
