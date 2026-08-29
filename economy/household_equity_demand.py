"""Passive Household/Person demand contract for secondary equity.

The adapter is deliberately non-mutating.  It computes a portfolio-demand
intent from explicit liquidity and concentration inputs, but never settles a
share purchase or calls the secondary transfer runtime.
"""

from dataclasses import asdict, dataclass, field
from typing import Any, Iterable, Optional


SHADOW_EQUITY_DEMAND_SCHEMA = "shadow_household_secondary_equity_demand_v1"


def _nonnegative(value, name):
    number = float(value)
    if number < 0.0:
        raise ValueError(f"{name} must be non-negative")
    return number


@dataclass(frozen=True)
class HouseholdEquityDemandDecision:
    household_id: Any
    buyer_person_id: Any
    firm_id: Any
    household_cash: float
    liquidity_reserve: float
    necessary_consumption_buffer: float
    available_financial_cash: float
    current_equity_assets: float
    current_financial_net_worth: float
    current_equity_share_of_assets: float
    desired_equity_budget: float
    desired_shares: float
    offered_transfer_price: float
    expected_dividend_signal: float = 0.0
    expected_profit_signal: float = 0.0
    expected_cash_flow_signal: float = 0.0
    debt_arrears_signal: float = 0.0
    current_firm_equity_assets: float = 0.0
    current_person_firm_shares: float = 0.0
    available_legacy_shares: float = 0.0
    shadow_fillable_shares: float = 0.0
    unmet_shadow_shares: float = 0.0
    max_household_equity_share_of_assets: Optional[float] = None
    max_firm_exposure_value: Optional[float] = None
    max_person_ownership_fraction: Optional[float] = None
    concentration_binding: bool = False
    risk_metadata: dict = field(default_factory=dict)
    decision_reason: str = ""
    financing_source: str = "household_cash_only"
    secondary_settlement_account: str = "LegacyOwner.cash"

    def to_dict(self):
        return asdict(self)


class ShadowHouseholdEquityDemandSystem:
    """Build non-mutating secondary-equity demand decisions."""

    @staticmethod
    def select_buyer_person(household, persons: Iterable[Any]):
        """Select the lowest-id alive Person belonging to a Household."""
        candidates = [
            person
            for person in persons
            if getattr(person, "alive", False)
            and getattr(person, "household_id", None) == household.id
        ]
        if not candidates:
            return None
        return sorted(candidates, key=lambda person: getattr(person, "id", 0))[0]

    def decide(
        self,
        *,
        household_id,
        buyer_person_id,
        firm_id,
        household_cash,
        liquidity_reserve,
        necessary_consumption_buffer,
        offered_transfer_price,
        current_equity_assets=0.0,
        current_financial_net_worth=None,
        other_financial_assets=0.0,
        liabilities=0.0,
        desired_equity_budget=None,
        expected_dividend_signal=0.0,
        expected_profit_signal=0.0,
        expected_cash_flow_signal=0.0,
        debt_arrears_signal=0.0,
        current_firm_equity_assets=0.0,
        current_person_firm_shares=0.0,
        firm_total_shares=None,
        available_legacy_shares=0.0,
        max_household_equity_share_of_assets=None,
        max_firm_exposure_value=None,
        max_person_ownership_fraction=None,
    ):
        cash = _nonnegative(household_cash, "household_cash")
        reserve = _nonnegative(liquidity_reserve, "liquidity_reserve")
        necessary = _nonnegative(
            necessary_consumption_buffer,
            "necessary_consumption_buffer",
        )
        price = float(offered_transfer_price)
        if price <= 0.0:
            raise ValueError("offered_transfer_price must be positive")
        equity_assets = _nonnegative(current_equity_assets, "current_equity_assets")
        other_assets = _nonnegative(other_financial_assets, "other_financial_assets")
        debt = _nonnegative(liabilities, "liabilities")
        current_firm_assets = _nonnegative(
            current_firm_equity_assets,
            "current_firm_equity_assets",
        )
        current_person_shares = _nonnegative(
            current_person_firm_shares,
            "current_person_firm_shares",
        )
        total_firm_shares = (
            None
            if firm_total_shares is None
            else _nonnegative(firm_total_shares, "firm_total_shares")
        )
        legacy_shares = _nonnegative(
            available_legacy_shares,
            "available_legacy_shares",
        )
        available_cash = max(cash - reserve - necessary, 0.0)
        gross_assets = cash + equity_assets + other_assets
        net_worth = (
            float(current_financial_net_worth)
            if current_financial_net_worth is not None
            else gross_assets - debt
        )
        equity_share = equity_assets / gross_assets if gross_assets > 0.0 else 0.0

        requested = (
            available_cash
            if desired_equity_budget is None
            else max(0.0, float(desired_equity_budget))
        )
        budget = min(requested, available_cash)
        binding = False
        reasons = []
        if available_cash <= 1e-12:
            budget = 0.0
            reasons.append("no_excess_liquidity")
        elif requested > available_cash + 1e-12:
            reasons.append("cash_cap")

        if max_household_equity_share_of_assets is not None:
            max_share = float(max_household_equity_share_of_assets)
            if not 0.0 <= max_share <= 1.0:
                raise ValueError("max_household_equity_share_of_assets must be in [0, 1]")
            exposure_room = max(0.0, max_share * gross_assets - equity_assets)
            if budget > exposure_room:
                budget = exposure_room
                binding = True
                reasons.append("household_equity_concentration_cap")

        if max_firm_exposure_value is not None:
            firm_room = max(
                0.0,
                float(max_firm_exposure_value) - current_firm_assets,
            )
            if budget > firm_room:
                budget = firm_room
                binding = True
                reasons.append("firm_exposure_cap")

        if max_person_ownership_fraction is not None:
            max_person_fraction = float(max_person_ownership_fraction)
            if not 0.0 <= max_person_fraction <= 1.0:
                raise ValueError("max_person_ownership_fraction must be in [0, 1]")
            if total_firm_shares is None:
                # Keep the boundary visible without silently inventing a
                # total-share denominator.
                reasons.append("person_ownership_cap_metadata_only")
            else:
                person_room_shares = max(
                    0.0,
                    max_person_fraction * total_firm_shares
                    - current_person_shares,
                )
                person_room_value = person_room_shares * price
                if budget > person_room_value:
                    budget = person_room_value
                    binding = True
                    reasons.append("person_ownership_cap")

        if budget > 0.0 and not reasons:
            reasons.append("positive_excess_liquidity")
        if float(expected_dividend_signal) == 0.0 and float(expected_profit_signal) == 0.0:
            reasons.append("no_return_signal")

        desired_shares = budget / price
        fillable = min(desired_shares, legacy_shares)
        unmet = max(0.0, desired_shares - fillable)
        if unmet > 1e-12:
            reasons.append("legacy_supply_shortfall")
        if budget <= 1e-12 and not reasons:
            reasons.append("zero_shadow_demand")

        return HouseholdEquityDemandDecision(
            household_id=household_id,
            buyer_person_id=buyer_person_id,
            firm_id=firm_id,
            household_cash=cash,
            liquidity_reserve=reserve,
            necessary_consumption_buffer=necessary,
            available_financial_cash=available_cash,
            current_equity_assets=equity_assets,
            current_financial_net_worth=net_worth,
            current_equity_share_of_assets=equity_share,
            desired_equity_budget=budget,
            desired_shares=desired_shares,
            offered_transfer_price=price,
            expected_dividend_signal=float(expected_dividend_signal),
            expected_profit_signal=float(expected_profit_signal),
            expected_cash_flow_signal=float(expected_cash_flow_signal),
            debt_arrears_signal=float(debt_arrears_signal),
            current_firm_equity_assets=current_firm_assets,
            current_person_firm_shares=current_person_shares,
            available_legacy_shares=legacy_shares,
            shadow_fillable_shares=fillable,
            unmet_shadow_shares=unmet,
            max_household_equity_share_of_assets=max_household_equity_share_of_assets,
            max_firm_exposure_value=max_firm_exposure_value,
            max_person_ownership_fraction=max_person_ownership_fraction,
            concentration_binding=binding,
            risk_metadata={
                "signal_semantics": "diagnostic_inputs_only",
                "debt_financed": False,
                "stock_market_price": False,
                "secondary_settlement_account": "LegacyOwner.cash",
            },
            decision_reason=";".join(dict.fromkeys(reasons)),
        )


__all__ = [
    "SHADOW_EQUITY_DEMAND_SCHEMA",
    "HouseholdEquityDemandDecision",
    "ShadowHouseholdEquityDemandSystem",
]
