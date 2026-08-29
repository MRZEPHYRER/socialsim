"""Passive/shadow Firm investment planning for Step 15C.

The planner consumes capacity views supplied by a future technology layer.  It
does not select a production function, mutate a Firm, settle a transaction,
request credit, draw RNG, or create an asset.
"""

from dataclasses import asdict, dataclass, field
from typing import Any, Dict, Optional


def _nonnegative(value, name):
    number = float(value)
    if number < 0.0:
        raise ValueError(f"{name} must be non-negative")
    return number


@dataclass(frozen=True)
class FirmInvestmentDecision:
    """A shadow investment intent with no settlement semantics."""

    firm_id: Any
    expected_demand: float
    desired_output: float
    current_effective_capacity: float
    desired_capacity: float
    capacity_gap: float
    current_capital_capacity: float
    desired_capital_capacity: float
    replacement_investment_need: float
    expansion_investment_need: float
    desired_investment_expenditure: float
    financing_gap: float
    decision_reason: str
    committed_expansion_capacity: float = 0.0
    committed_replacement_capacity: float = 0.0
    effective_future_capital_capacity: float = 0.0
    residual_uncommitted_capacity_gap: float = 0.0
    internal_finance_capacity: float = 0.0
    replacement_investment_expenditure: float = 0.0
    expansion_investment_expenditure: float = 0.0
    finance_signal_metadata: Dict[str, Any] = field(default_factory=dict)
    capacity_metadata: Dict[str, Any] = field(default_factory=dict)
    transaction_created: bool = False
    capital_asset_created: bool = False
    money_created: float = 0.0
    debt_created: float = 0.0

    def __post_init__(self):
        for name in (
            "expected_demand",
            "desired_output",
            "current_effective_capacity",
            "desired_capacity",
            "capacity_gap",
            "current_capital_capacity",
            "desired_capital_capacity",
            "replacement_investment_need",
            "expansion_investment_need",
            "desired_investment_expenditure",
            "financing_gap",
            "committed_expansion_capacity",
            "committed_replacement_capacity",
            "effective_future_capital_capacity",
            "residual_uncommitted_capacity_gap",
            "internal_finance_capacity",
            "replacement_investment_expenditure",
            "expansion_investment_expenditure",
            "money_created",
            "debt_created",
        ):
            _nonnegative(getattr(self, name), name)
        if self.transaction_created or self.capital_asset_created:
            raise ValueError("shadow decision cannot create a transaction or asset")
        if abs(self.money_created) > 1e-12 or abs(self.debt_created) > 1e-12:
            raise ValueError("shadow decision cannot create money or debt")

    def to_dict(self):
        return asdict(self)


class ShadowInvestmentPlanner:
    """Build a decision from explicit capacity and finance shadow inputs."""

    def decide(
        self,
        *,
        firm_id,
        expected_demand,
        desired_output,
        current_effective_capacity,
        desired_capacity,
        non_capital_capacity,
        current_capital_capacity,
        committed_expansion_capacity=0.0,
        committed_replacement_capacity=0.0,
        replacement_capacity_loss=0.0,
        capital_technology_available=True,
        replacement_cost_per_capacity=0.0,
        expansion_cost_per_capacity=0.0,
        cash=0.0,
        operating_liquidity_floor=0.0,
        retained_cash_available: Optional[float] = None,
        finance_signal_metadata=None,
        capacity_metadata=None,
    ):
        expected = _nonnegative(expected_demand, "expected_demand")
        output = _nonnegative(desired_output, "desired_output")
        effective = _nonnegative(current_effective_capacity, "current_effective_capacity")
        desired = _nonnegative(desired_capacity, "desired_capacity")
        noncapital = _nonnegative(non_capital_capacity, "non_capital_capacity")
        current_capital = _nonnegative(current_capital_capacity, "current_capital_capacity")
        committed_expansion = _nonnegative(
            committed_expansion_capacity, "committed_expansion_capacity"
        )
        committed_replacement = _nonnegative(
            committed_replacement_capacity, "committed_replacement_capacity"
        )
        loss = _nonnegative(replacement_capacity_loss, "replacement_capacity_loss")
        replacement_cost = _nonnegative(
            replacement_cost_per_capacity, "replacement_cost_per_capacity"
        )
        expansion_cost = _nonnegative(
            expansion_cost_per_capacity, "expansion_cost_per_capacity"
        )

        capacity_gap = max(desired - effective, 0.0)
        if not capital_technology_available:
            desired_capital = 0.0
            replacement = 0.0
            expansion = 0.0
            reason = "no_capital_technology"
        else:
            desired_capital = max(desired - noncapital, 0.0)
            # Replacement commitment can restore only an observed retired
            # service loss.  It is not silently reclassified as expansion.
            recognized_committed_replacement = min(committed_replacement, loss)
            replacement = min(
                max(loss - recognized_committed_replacement, 0.0),
                desired_capital,
            )
            effective_future_capital = (
                current_capital
                + committed_expansion
                + recognized_committed_replacement
            )
            expansion = max(
                desired_capital
                - effective_future_capital
                - replacement,
                0.0,
            )
            if replacement > 0.0 and expansion > 0.0:
                reason = "replacement_and_expansion"
            elif replacement > 0.0:
                reason = "replacement_capacity_loss"
            elif expansion > 0.0:
                reason = "expansion_capacity_gap"
            elif capacity_gap <= 0.0:
                reason = "excess_capacity"
            else:
                reason = "capacity_gap_without_capital_gap"

        replacement_expenditure = replacement * replacement_cost
        expansion_expenditure = expansion * expansion_cost
        expenditure = replacement_expenditure + expansion_expenditure
        residual_uncommitted_gap = replacement + expansion
        cash_value = _nonnegative(cash, "cash")
        floor = _nonnegative(operating_liquidity_floor, "operating_liquidity_floor")
        internal = (
            _nonnegative(retained_cash_available, "retained_cash_available")
            if retained_cash_available is not None
            else max(cash_value - floor, 0.0)
        )
        financing_gap = max(expenditure - internal, 0.0)

        return FirmInvestmentDecision(
            firm_id=firm_id,
            expected_demand=expected,
            desired_output=output,
            current_effective_capacity=effective,
            desired_capacity=desired,
            capacity_gap=capacity_gap,
            current_capital_capacity=current_capital,
            desired_capital_capacity=desired_capital,
            replacement_investment_need=replacement,
            expansion_investment_need=expansion,
            desired_investment_expenditure=expenditure,
            financing_gap=financing_gap,
            decision_reason=reason,
            committed_expansion_capacity=committed_expansion,
            committed_replacement_capacity=recognized_committed_replacement,
            effective_future_capital_capacity=effective_future_capital if capital_technology_available else current_capital,
            residual_uncommitted_capacity_gap=residual_uncommitted_gap,
            internal_finance_capacity=internal,
            replacement_investment_expenditure=replacement_expenditure,
            expansion_investment_expenditure=expansion_expenditure,
            finance_signal_metadata={
                **dict(finance_signal_metadata or {}),
                "cash_observed": cash_value,
                "operating_liquidity_floor": floor,
                "working_capital_credit_investment_funding": False,
            },
            capacity_metadata={
                **dict(capacity_metadata or {}),
                "capacity_mapping": "shadow_input",
                "production_function_selected": False,
                "capital_productivity_selected": False,
                "non_capital_capacity": noncapital,
            },
        )


__all__ = ["FirmInvestmentDecision", "ShadowInvestmentPlanner"]
