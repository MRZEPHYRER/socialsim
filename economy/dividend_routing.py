"""Firm dividend settlement through the runtime ownership table.

This module deliberately owns only the recipient-routing layer.  The caller
continues to decide whether a dividend is declared and how large it is.
"""

from dataclasses import dataclass, field
import math

from economy.ledger import object_account


DIVIDEND_ROUTING_SCHEMA = "person_dividend_routing_v1"
LEGACY_OWNER_ACCOUNT = "LegacyOwnershipPool"


@dataclass
class DividendRoutingResult:
    firm_id: object
    declared_dividend: float
    person_paid: float = 0.0
    legacy_entitlement: float = 0.0
    estate_paid: float = 0.0
    person_receipts: list = field(default_factory=list)
    estate_receipts: list = field(default_factory=list)
    reconciliation_gap: float = 0.0
    firm_cash_before: float = 0.0
    firm_cash_after: float = 0.0
    legacy_cash_before: float = 0.0
    legacy_cash_after: float = 0.0

    @property
    def household_cash_inflow(self):
        return self.person_paid


def ensure_dividend_routing_state(world):
    """Create neutral legacy-owner state for current and old checkpoints."""
    if not hasattr(world, "legacy_owner_cash"):
        world.legacy_owner_cash = 0.0
    if not hasattr(world, "legacy_dividend_history"):
        world.legacy_dividend_history = []
    if not hasattr(world, "dividend_routing_events"):
        world.dividend_routing_events = []
    return world


def _person_lookup(world, person_id):
    person = getattr(world, "person_dict", {}).get(person_id)
    if person is not None:
        return person
    for candidate in getattr(world, "population", []):
        if getattr(candidate, "id", None) == person_id:
            return candidate
    return None


def _firm_table(world, firm):
    table = getattr(firm, "cap_table", None)
    if table is None and hasattr(world, "ensure_ownership_state"):
        world.ensure_ownership_state()
        table = getattr(firm, "cap_table", None)
    if table is None:
        raise ValueError(f"Firm {getattr(firm, 'firm_id', None)} has no CapTable")
    if table.firm_id != getattr(firm, "firm_id", table.firm_id):
        raise ValueError("Dividend payer Firm and CapTable firm_id do not match")
    return table


def route_declared_dividend(world, firm, declared_dividend, payer=None):
    """Settle one declared dividend according to current CapTable holdings.

    Legacy ownership uses the explicit retained-account treatment.  It is a
    real cash transfer from the Firm to ``world.legacy_owner_cash`` and is
    therefore neither Household income nor money creation.
    """
    ensure_dividend_routing_state(world)
    if hasattr(world, "ensure_shareholder_estate_state"):
        world.ensure_shareholder_estate_state()
    amount = max(0.0, float(declared_dividend))
    firm_id = getattr(firm, "firm_id", 0)
    result = DividendRoutingResult(firm_id=firm_id, declared_dividend=amount)
    if amount <= 0.0:
        return result

    table = _firm_table(world, firm)
    total_shares = float(table.total_shares)
    if total_shares <= 0.0:
        raise ValueError(f"Firm {firm_id} CapTable has no issued shares")

    payer = firm if payer is None else payer
    payer_cash = float(getattr(payer, "cash", 0.0))
    legacy_cash_before = float(getattr(world, "legacy_owner_cash", 0.0))
    result.firm_cash_before = payer_cash
    result.legacy_cash_before = legacy_cash_before
    if payer_cash + 1e-9 < amount:
        raise ValueError(
            f"Firm {firm_id} cannot settle declared dividend {amount}; "
            f"payer cash is {payer_cash}"
        )

    ledger = getattr(world, "ledger", None)
    if ledger is None:
        raise ValueError("Dividend routing requires the accepted World ledger")

    payer_name = f"firm[{firm_id}].cash"
    person_total = 0.0
    for holding in table.holdings:
        entitlement = amount * float(holding.shares) / total_shares
        if entitlement <= 0.0:
            continue

        if holding.holder_type == "legacy":
            ledger.transfer(
                payer=object_account(payer, "cash", payer_name),
                receiver=object_account(
                    world,
                    "legacy_owner_cash",
                    "LegacyOwnershipPool.cash",
                ),
                amount=entitlement,
                reason="dividend_payment_legacy_retained",
            )
            result.legacy_entitlement += entitlement
            continue

        if holding.holder_type == "estate":
            estate = getattr(world, "estate_accounts", {}).get(
                holding.holder_id
            )
            if estate is None or getattr(estate, "status", "open") != "open":
                raise ValueError(
                    f"Estate shareholder {holding.holder_id} has no open EstateAccount"
                )
            ledger.transfer(
                payer=object_account(payer, "cash", payer_name),
                receiver=object_account(
                    estate,
                    "cash",
                    f"{holding.holder_id}.cash",
                ),
                amount=entitlement,
                reason="dividend_payment_estate",
            )
            estate.estate_dividend_income = (
                getattr(estate, "estate_dividend_income", 0.0)
                + entitlement
            )
            result.estate_paid += entitlement
            result.estate_receipts.append({
                "firm_id": firm_id,
                "estate_id": holding.holder_id,
                "deceased_person_id": getattr(
                    estate, "deceased_person_id", None
                ),
                "ownership_fraction": float(holding.shares) / total_shares,
                "dividend_received": entitlement,
            })
            continue

        if holding.holder_type != "person":
            # Future institutional holders need their own settlement account;
            # silently routing them to households would be an ownership bug.
            raise ValueError(
                f"Unsupported active dividend holder type: {holding.holder_type}"
            )

        person = _person_lookup(world, holding.holder_id)
        household = (
            getattr(world, "household_dict", {}).get(
                getattr(person, "household_id", None)
            )
            if person is not None
            else None
        )
        if person is None or household is None:
            raise ValueError(
                f"Person shareholder {holding.holder_id} has no settlement Household"
            )

        ledger.transfer(
            payer=object_account(payer, "cash", payer_name),
            receiver=object_account(
                household,
                "wealth",
                f"household.{household.id}.wealth",
            ),
            amount=entitlement,
            reason="dividend_payment_person",
        )
        household.income_this_step = (
            getattr(household, "income_this_step", 0.0) + entitlement
        )
        household.dividend_income_this_step = (
            getattr(household, "dividend_income_this_step", 0.0) + entitlement
        )
        person.dividend_entitlement_this_step = (
            getattr(person, "dividend_entitlement_this_step", 0.0)
            + entitlement
        )
        person.dividend_received_this_step = (
            getattr(person, "dividend_received_this_step", 0.0)
            + entitlement
        )
        person_total += entitlement
        result.person_receipts.append({
            "firm_id": firm_id,
            "person_id": holding.holder_id,
            "household_id": household.id,
            "ownership_fraction": float(holding.shares) / total_shares,
            "dividend_received": entitlement,
            "household_cash_before": household.wealth - entitlement,
            "household_cash_after": household.wealth,
        })

    result.person_paid = person_total
    result.reconciliation_gap = (
        result.person_paid
        + result.legacy_entitlement
        + result.estate_paid
        - amount
    )
    result.firm_cash_after = float(getattr(payer, "cash", 0.0))
    result.legacy_cash_after = float(getattr(world, "legacy_owner_cash", 0.0))
    world.legacy_dividend_history.append({
        "firm_id": firm_id,
        "declared_dividend": amount,
        "person_paid": result.person_paid,
        "legacy_entitlement": result.legacy_entitlement,
        "routing_gap": result.reconciliation_gap,
    })
    world.dividend_routing_events.append({
        "global_step": int(getattr(world, "current_step_index", 0)),
        "firm_id": firm_id,
        "declared_dividend": amount,
        "legacy_ownership_fraction": table.ownership_fraction("legacy"),
        "person_ownership_fraction": table.ownership_fraction("person"),
        "total_shares": table.total_shares,
        "legacy_shares": sum(
            holding.shares
            for holding in table.holdings
            if holding.holder_type == "legacy"
        ),
        "person_shares": sum(
            holding.shares
            for holding in table.holdings
            if holding.holder_type == "person"
        ),
        "person_paid": result.person_paid,
        "legacy_entitlement": result.legacy_entitlement,
        "estate_paid": result.estate_paid,
        "household_dividend_income": result.person_paid,
        "legacy_owner_cash_inflow": result.legacy_entitlement,
        "estate_cash_inflow": result.estate_paid,
        "firm_cash_before": result.firm_cash_before,
        "firm_cash_after": result.firm_cash_after,
        "legacy_cash_before": result.legacy_cash_before,
        "legacy_cash_after": result.legacy_cash_after,
        "household_receipts": list(result.person_receipts),
        "estate_receipts": list(result.estate_receipts),
        "routing_gap": result.reconciliation_gap,
    })

    for target in (firm, payer):
        target.person_dividend_paid = result.person_paid
        target.legacy_dividend_entitlement = result.legacy_entitlement
        target.estate_dividend_paid = result.estate_paid
        target.dividend_routing_gap = result.reconciliation_gap
        target.dividend_recipient_count = len(result.person_receipts)
        target.dividend_payment = amount
    return result


__all__ = [
    "DIVIDEND_ROUTING_SCHEMA",
    "DividendRoutingResult",
    "ensure_dividend_routing_state",
    "route_declared_dividend",
]
