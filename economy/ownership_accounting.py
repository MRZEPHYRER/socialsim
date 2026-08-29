"""Passive ownership and detailed financial-accounting architecture.

This module is an audit/design layer for Step 15E.0.  It is not imported by
the simulation runtime, does not replace ``Ledger``, and does not issue,
purchase, transfer, or pay dividends on shares.
"""

from dataclasses import asdict, dataclass, field
from typing import Any, Dict, Optional


ARCHITECTURE_VERSION = "15E0.1"
CF_CLASSIFICATIONS = {"CFO", "CFI", "CFF", "NONE"}
INCOME_EXPENSE_CLASSIFICATIONS = {"income", "expense", "none"}
HOLDER_TYPES = {
    "legacy",
    "person",
    "estate",
    "government",
    "fund",
    "pension_fund",
}


def _number(value, name):
    return float(value)


def _nonnegative(value, name):
    number = _number(value, name)
    if number < 0.0:
        raise ValueError(f"{name} must be non-negative")
    return number


@dataclass(frozen=True)
class GenericFinancialEvent:
    """Common event boundary; one event is still only a passive journal view."""

    event_id: Any
    actor: Any
    counterparty: Any
    transaction_type: str
    cash_change: float = 0.0
    asset_change: Dict[str, float] = field(default_factory=dict)
    liability_change: Dict[str, float] = field(default_factory=dict)
    equity_change: float = 0.0
    income_expense_classification: str = "none"
    income_expense_amount: float = 0.0
    cash_flow_classification: str = "NONE"
    reference_event_id: Optional[Any] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        if self.income_expense_classification not in INCOME_EXPENSE_CLASSIFICATIONS:
            raise ValueError("invalid income/expense classification")
        if self.cash_flow_classification not in CF_CLASSIFICATIONS:
            raise ValueError("invalid CFO/CFI/CFF classification")
        _number(self.cash_change, "cash_change")
        _number(self.equity_change, "equity_change")
        _nonnegative(self.income_expense_amount, "income_expense_amount")

    def to_dict(self):
        return asdict(self)


@dataclass(frozen=True)
class ShareHolding:
    holder_id: Any
    firm_id: Any
    shares: float
    holder_type: str = "person"

    def __post_init__(self):
        _nonnegative(self.shares, "shares")
        if self.holder_type not in HOLDER_TYPES:
            raise ValueError(f"unsupported holder type: {self.holder_type}")

    @property
    def is_person(self):
        return self.holder_type == "person"

    def to_dict(self):
        return asdict(self)


@dataclass(frozen=True)
class CapTable:
    """Firm-level ownership table with legacy ownership as the initial holder."""

    firm_id: Any
    issued_shares: float = 100.0
    holdings: tuple = field(default_factory=tuple)
    legacy_pool_id: str = "LegacyOwnershipPool"

    def __post_init__(self):
        _nonnegative(self.issued_shares, "issued_shares")
        total = sum(float(holding.shares) for holding in self.holdings)
        if abs(total - float(self.issued_shares)) > 1e-12:
            raise ValueError("CapTable holdings must sum to issued_shares")
        for holding in self.holdings:
            if holding.firm_id != self.firm_id:
                raise ValueError("ShareHolding firm_id must match CapTable firm_id")

    @classmethod
    def initial_legacy_owned(cls, firm_id, issued_shares=100.0):
        return cls(
            firm_id=firm_id,
            issued_shares=float(issued_shares),
            holdings=(
                ShareHolding(
                    holder_id="LegacyOwnershipPool",
                    firm_id=firm_id,
                    shares=float(issued_shares),
                    holder_type="legacy",
                ),
            ),
        )

    @classmethod
    def initial_person_owned(cls, firm_id, person_id, issued_shares=100.0):
        """Create an explicit single-Person initial ownership table."""
        shares = _nonnegative(issued_shares, "issued_shares")
        if shares <= 0.0:
            raise ValueError("issued_shares must be positive")
        if person_id is None or person_id == "LegacyOwnershipPool":
            raise ValueError("initial Person ownership requires a valid person_id")
        return cls(
            firm_id=firm_id,
            issued_shares=shares,
            holdings=(ShareHolding(
                holder_id=person_id,
                firm_id=firm_id,
                shares=shares,
                holder_type="person",
            ),),
        )

    @property
    def legacy_shares(self):
        return sum(
            holding.shares
            for holding in self.holdings
            if holding.holder_type == "legacy"
        )

    @property
    def person_shares(self):
        return sum(
            holding.shares
            for holding in self.holdings
            if holding.holder_type == "person"
        )

    @property
    def estate_shares(self):
        return sum(
            holding.shares
            for holding in self.holdings
            if holding.holder_type == "estate"
        )

    @property
    def total_shares(self):
        return self.issued_shares

    @property
    def holder_count(self):
        return len(self.holdings)

    @property
    def ownership_fractions(self):
        return {
            holder_type: self.ownership_fraction(holder_type)
            for holder_type in sorted(HOLDER_TYPES)
        }

    def read_only_view(self):
        return {
            "firm_id": self.firm_id,
            "total_shares": self.total_shares,
            "legacy_shares": self.legacy_shares,
            "person_shares": self.person_shares,
            "estate_shares": self.estate_shares,
            "ownership_fractions": dict(self.ownership_fractions),
            "holder_count": self.holder_count,
            "person_shareholder_count": sum(
                1
                for holding in self.holdings
                if holding.holder_type == "person" and holding.shares > 0
            ),
        }

    def ownership_fraction(self, holder_type):
        return sum(
            holding.shares
            for holding in self.holdings
            if holding.holder_type == holder_type
        ) / self.issued_shares if self.issued_shares > 0 else 0.0

    def validate(self):
        return {
            "holdings_sum_gap": sum(h.shares for h in self.holdings) - self.issued_shares,
            "legacy_fraction": self.ownership_fraction("legacy"),
            "person_fraction": self.ownership_fraction("person"),
            "passed": abs(sum(h.shares for h in self.holdings) - self.issued_shares) <= 1e-12,
        }

    def with_primary_issuance(self, holder_id, shares, holder_type="person"):
        """Return a new table after a controlled primary issuance.

        This is deliberately an explicit state transition helper.  It is not
        called by the simulation and does not imply a secondary market.
        """
        new_shares = _nonnegative(shares, "shares")
        if new_shares <= 0.0:
            raise ValueError("primary issuance must issue positive shares")
        if holder_type not in HOLDER_TYPES or holder_type == "legacy":
            raise ValueError("primary issuance holder must be non-legacy")

        holdings = list(self.holdings)
        for index, holding in enumerate(holdings):
            if (
                holding.holder_id == holder_id
                and holding.holder_type == holder_type
            ):
                holdings[index] = ShareHolding(
                    holder_id=holder_id,
                    firm_id=self.firm_id,
                    shares=holding.shares + new_shares,
                    holder_type=holder_type,
                )
                break
        else:
            holdings.append(
                ShareHolding(
                    holder_id=holder_id,
                    firm_id=self.firm_id,
                    shares=new_shares,
                    holder_type=holder_type,
                )
            )

        return CapTable(
            firm_id=self.firm_id,
            issued_shares=self.issued_shares + new_shares,
            holdings=tuple(holdings),
            legacy_pool_id=self.legacy_pool_id,
        )

    def with_secondary_legacy_transfer(self, holder_id, shares, holder_type="person"):
        """Return a table after transferring existing Legacy shares.

        This changes ownership only.  Issued shares and the Legacy pool's
        share count are updated in equal and opposite amounts; Firm cash and
        paid-in equity are intentionally outside this pure CapTable change.
        The controlled settlement layer is responsible for cash payment.
        """
        transfer = _nonnegative(shares, "shares")
        if transfer <= 0.0:
            raise ValueError("secondary transfer must transfer positive shares")
        if holder_type != "person":
            raise ValueError("controlled secondary transfer buyer must be a person")
        if holder_id == self.legacy_pool_id:
            raise ValueError("LegacyOwnershipPool cannot buy its own shares")
        if transfer > self.legacy_shares + 1e-12:
            raise ValueError("secondary transfer exceeds available Legacy shares")

        actual_transfer = min(transfer, self.legacy_shares)
        holdings = []
        legacy_remaining = actual_transfer
        buyer_index = None
        for holding in self.holdings:
            if holding.holder_type == "legacy":
                moved = min(holding.shares, legacy_remaining)
                kept = holding.shares - moved
                legacy_remaining -= moved
                if kept > 1e-12:
                    holdings.append(
                        ShareHolding(
                            holder_id=holding.holder_id,
                            firm_id=self.firm_id,
                            shares=kept,
                            holder_type=holding.holder_type,
                        )
                    )
                continue
            if holding.holder_id == holder_id and holding.holder_type == holder_type:
                buyer_index = len(holdings)
            holdings.append(holding)

        if legacy_remaining > 1e-10:
            raise ValueError("secondary transfer could not locate Legacy shares")

        if buyer_index is None:
            holdings.append(
                ShareHolding(
                    holder_id=holder_id,
                    firm_id=self.firm_id,
                    shares=actual_transfer,
                    holder_type=holder_type,
                )
            )
        else:
            buyer = holdings[buyer_index]
            holdings[buyer_index] = ShareHolding(
                holder_id=buyer.holder_id,
                firm_id=self.firm_id,
                shares=buyer.shares + actual_transfer,
                holder_type=buyer.holder_type,
            )

        return CapTable(
            firm_id=self.firm_id,
            issued_shares=self.issued_shares,
            holdings=tuple(holdings),
            legacy_pool_id=self.legacy_pool_id,
        )

    def _transfer_holder_shares(
        self,
        source_id,
        source_type,
        destination_id,
        destination_type,
        shares,
    ):
        """Return a table after an ownership-only holder transfer.

        The helper deliberately changes neither issued shares nor any Firm
        balance-sheet field.  It is used for death-to-estate and estate-to-
        heir transitions, where the cash/accounting settlement lives outside
        the CapTable.
        """
        amount = _nonnegative(shares, "shares")
        if amount <= 0.0:
            raise ValueError("holder transfer must transfer positive shares")
        if source_type not in HOLDER_TYPES or destination_type not in HOLDER_TYPES:
            raise ValueError("unsupported holder transfer type")
        if source_type == "legacy":
            raise ValueError("Legacy shares require the secondary-transfer path")
        if source_type == destination_type and source_id == destination_id:
            raise ValueError("holder transfer source and destination are identical")

        holdings = list(self.holdings)
        source_index = None
        source_available = 0.0
        destination_index = None
        for index, holding in enumerate(holdings):
            if (
                holding.holder_id == source_id
                and holding.holder_type == source_type
            ):
                source_index = index
                source_available = holding.shares
            if (
                holding.holder_id == destination_id
                and holding.holder_type == destination_type
            ):
                destination_index = index

        if source_index is None or amount > source_available + 1e-12:
            raise ValueError("holder transfer exceeds source shares")

        source_remaining = source_available - amount
        if source_remaining <= 1e-12:
            holdings.pop(source_index)
            if destination_index is not None and destination_index > source_index:
                destination_index -= 1
        else:
            holdings[source_index] = ShareHolding(
                holder_id=source_id,
                firm_id=self.firm_id,
                shares=source_remaining,
                holder_type=source_type,
            )

        if destination_index is None:
            holdings.append(
                ShareHolding(
                    holder_id=destination_id,
                    firm_id=self.firm_id,
                    shares=amount,
                    holder_type=destination_type,
                )
            )
        else:
            destination = holdings[destination_index]
            holdings[destination_index] = ShareHolding(
                holder_id=destination_id,
                firm_id=self.firm_id,
                shares=destination.shares + amount,
                holder_type=destination_type,
            )

        return CapTable(
            firm_id=self.firm_id,
            issued_shares=self.issued_shares,
            holdings=tuple(holdings),
            legacy_pool_id=self.legacy_pool_id,
        )

    def with_person_to_estate(self, person_id, estate_id, shares):
        return self._transfer_holder_shares(
            person_id, "person", estate_id, "estate", shares
        )

    def with_estate_to_person(self, estate_id, person_id, shares):
        return self._transfer_holder_shares(
            estate_id, "estate", person_id, "person", shares
        )

    def to_dict(self):
        payload = asdict(self)
        payload["holdings"] = [holding.to_dict() for holding in self.holdings]
        return payload


@dataclass
class EquityOwnershipSystem:
    """Passive registry prepared for Person and future institutional holders."""

    cap_tables: Dict[Any, CapTable] = field(default_factory=dict)
    supported_holder_types: tuple = tuple(sorted(HOLDER_TYPES))
    active_share_transactions: int = 0

    def initial_table(self, firm_id):
        table = CapTable.initial_legacy_owned(firm_id)
        self.cap_tables[firm_id] = table
        return table

    def ownership_summary(self, firm_id):
        table = self.cap_tables[firm_id]
        return {
            "firm_id": firm_id,
            "legacy_fraction": table.ownership_fraction("legacy"),
            "person_fraction": table.ownership_fraction("person"),
            "government_fraction": table.ownership_fraction("government"),
            "fund_fraction": table.ownership_fraction("fund")
            + table.ownership_fraction("pension_fund"),
            "share_transactions": self.active_share_transactions,
        }

    def to_dict(self):
        return {
            "cap_tables": {str(key): value.to_dict() for key, value in self.cap_tables.items()},
            "supported_holder_types": list(self.supported_holder_types),
            "active_share_transactions": self.active_share_transactions,
        }


@dataclass(frozen=True)
class HouseholdBalanceSheetView:
    """Future household statement; current model supplies only cash wealth."""

    household_id: Any
    cash_wealth: float = 0.0
    equity_assets: float = 0.0
    other_financial_assets: float = 0.0
    liabilities: float = 0.0

    @property
    def net_worth(self):
        return (
            self.cash_wealth
            + self.equity_assets
            + self.other_financial_assets
            - self.liabilities
        )

    @classmethod
    def from_current_cash_wealth(cls, household_id, wealth, equity_assets=0.0):
        return cls(
            household_id=household_id,
            cash_wealth=float(wealth),
            equity_assets=float(equity_assets),
        )

    def to_dict(self):
        payload = asdict(self)
        payload["net_worth"] = self.net_worth
        return payload


@dataclass(frozen=True)
class FirmBalanceSheetView:
    """Future firm statement using the accepted Assets = Liabilities + Equity identity."""

    firm_id: Any
    cash: float = 0.0
    inventory_book_value: float = 0.0
    capital_asset_book_value: float = 0.0
    other_assets: float = 0.0
    liabilities: float = 0.0
    equity: Optional[float] = None

    @property
    def total_assets(self):
        return self.cash + self.inventory_book_value + self.capital_asset_book_value + self.other_assets

    @property
    def implied_equity(self):
        return self.total_assets - self.liabilities

    @property
    def balance_sheet_gap(self):
        equity = self.implied_equity if self.equity is None else self.equity
        return self.total_assets - self.liabilities - equity

    def to_dict(self):
        payload = asdict(self)
        payload.update({
            "total_assets": self.total_assets,
            "implied_equity": self.implied_equity,
            "balance_sheet_gap": self.balance_sheet_gap,
        })
        return payload


__all__ = [
    "ARCHITECTURE_VERSION",
    "CF_CLASSIFICATIONS",
    "CapTable",
    "EquityOwnershipSystem",
    "FirmBalanceSheetView",
    "GenericFinancialEvent",
    "HouseholdBalanceSheetView",
    "ShareHolding",
]
