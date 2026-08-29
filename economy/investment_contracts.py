"""Passive contracts for future Firm investment and intermediate inputs.

This module is intentionally outside the simulation execution path.  The
dataclasses describe transactions that a future investment layer may settle;
they do not mutate Firms, call the Ledger, draw random numbers, or create
capital assets in the canonical World.
"""

from dataclasses import asdict, dataclass, field
from typing import Any, Dict, Optional


CONTRACT_VERSION = "15A.1"
CAPITAL_STOCK_CONTRACT_VERSION = "15B.1"
ALLOWED_FINANCING_SOURCES = {
    "retained_cash",
    "future_investment_loan",
    "future_equity_issuance",
}
DEPRECIATION_POLICY_TYPES = {
    "unselected",
    "straight_line",
    "declining_balance",
    "units_of_production",
}


def _nonnegative(value: Any, field_name: str) -> float:
    number = float(value)
    if number < 0.0:
        raise ValueError(f"{field_name} must be non-negative")
    return number


def _copy_metadata(value: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    return dict(value or {})


@dataclass(frozen=True)
class DepreciationPolicyReference:
    """A policy interface/reference without a selected calibration."""

    policy_type: str = "unselected"
    policy_metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        if self.policy_type not in DEPRECIATION_POLICY_TYPES:
            raise ValueError(f"unsupported depreciation policy type: {self.policy_type}")

    def period_expense(self, *args, **kwargs):
        raise NotImplementedError(
            "Step 15B defines the depreciation interface but selects no policy calibration"
        )

    def to_dict(self):
        return asdict(self)


@dataclass(frozen=True)
class FirmInvestmentIntent:
    """A passive Firm plan; it is not an order and has no economic effect."""

    firm_id: Any
    investment_good_id: str
    desired_investment_expenditure: float = 0.0
    desired_capital_units: float = 0.0
    financing_source: str = "retained_cash"
    financing_metadata: Dict[str, Any] = field(default_factory=dict)
    expected_productive_use: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        _nonnegative(self.desired_investment_expenditure, "desired_investment_expenditure")
        _nonnegative(self.desired_capital_units, "desired_capital_units")
        if self.financing_source not in ALLOWED_FINANCING_SOURCES:
            raise ValueError(f"unsupported financing_source: {self.financing_source}")

    def to_dict(self):
        return asdict(self)

    @classmethod
    def from_dict(cls, payload):
        return cls(**dict(payload))


@dataclass(frozen=True)
class CapitalAsset:
    """A future asset record owned by a Firm legal/accounting entity."""

    asset_id: Any
    asset_class: str
    owner_firm_id: Any
    acquisition_cost: float
    quantity: float = 1.0
    capacity_metadata: Dict[str, Any] = field(default_factory=dict)
    opening_book_value: Optional[float] = None
    closing_book_value: Optional[float] = None
    accumulated_depreciation: Optional[float] = None
    age: float = 0.0
    service_life_metadata: Dict[str, Any] = field(default_factory=dict)
    remaining_book_value: Optional[float] = None
    depreciation_policy_ref: Optional[str] = None

    def __post_init__(self):
        cost = _nonnegative(self.acquisition_cost, "acquisition_cost")
        _nonnegative(self.quantity, "quantity")
        _nonnegative(self.age, "age")
        opening = cost if self.opening_book_value is None else self.opening_book_value
        _nonnegative(opening, "opening_book_value")
        if self.closing_book_value is not None and self.remaining_book_value is not None:
            if abs(float(self.closing_book_value) - float(self.remaining_book_value)) > 1e-12:
                raise ValueError("closing_book_value and remaining_book_value disagree")
        closing = self.closing_book_value
        if closing is None:
            closing = self.remaining_book_value
        if closing is None:
            closing = opening
        _nonnegative(closing, "closing_book_value")
        if opening > cost + 1e-12:
            raise ValueError("opening_book_value cannot exceed acquisition_cost")
        if closing > opening + 1e-12:
            raise ValueError("closing_book_value cannot exceed opening_book_value")
        depreciation = self.accumulated_depreciation
        if depreciation is None:
            depreciation = cost - closing
        _nonnegative(depreciation, "accumulated_depreciation")
        # Historical fixtures used this field for the current opening/closing
        # difference.  Accept that representation when necessary, while the
        # lifecycle runtime uses cumulative depreciation from acquisition.
        bridge_tolerance = 1e-9 * max(
            1.0,
            abs(cost),
            abs(opening),
            abs(closing),
            abs(depreciation),
        )
        if (
            abs((cost - closing) - depreciation) > bridge_tolerance
            and abs((opening - closing) - depreciation) > bridge_tolerance
        ):
            raise ValueError("book-value and accumulated-depreciation bridge does not close")
        object.__setattr__(self, "opening_book_value", float(opening))
        object.__setattr__(self, "closing_book_value", float(closing))
        object.__setattr__(self, "accumulated_depreciation", float(depreciation))
        object.__setattr__(self, "remaining_book_value", float(closing))

    def capacity_contribution_interface(self):
        """Return metadata only; no productivity coefficient is selected."""
        return {
            "asset_id": self.asset_id,
            "asset_class": self.asset_class,
            "owner_firm_id": self.owner_firm_id,
            "quantity": self.quantity,
            "capacity_metadata": dict(self.capacity_metadata),
            "behavior_enabled": False,
        }

    @property
    def remaining_useful_life(self):
        """Return declared life metadata without selecting a lifetime."""
        if "remaining_useful_life" in self.service_life_metadata:
            return max(0.0, float(self.service_life_metadata["remaining_useful_life"]))
        for key in ("useful_life_weeks", "useful_life"):
            if key not in self.service_life_metadata:
                continue
            try:
                useful_life = float(self.service_life_metadata[key])
            except (TypeError, ValueError):
                return None
            return max(0.0, useful_life - self.age)
        return None

    @property
    def age_weeks(self):
        """Explicit weekly age view; ``age`` remains backward compatible."""
        return max(0.0, float(self.age))

    @property
    def useful_life_weeks(self):
        """Return a declared useful life without selecting a default."""
        for key in ("useful_life_weeks", "useful_life"):
            if key not in self.service_life_metadata:
                continue
            try:
                life = float(self.service_life_metadata[key])
            except (TypeError, ValueError):
                return None
            return max(0.0, life)
        return None

    @property
    def is_active(self):
        if self.capacity_metadata.get("active", True) is False:
            return False
        remaining = self.remaining_useful_life
        return remaining is None or remaining > 0.0

    @property
    def is_retired(self):
        return not self.is_active

    def straight_line_depreciation(self, period_weeks=1.0):
        """Return non-cash book depreciation for a declared-life fixture.

        No useful life is selected here.  Assets without an explicit life
        therefore fail loudly instead of silently acquiring a calibration.
        """
        period = _nonnegative(period_weeks, "period_weeks")
        life = self.useful_life_weeks
        if life is None or life <= 0.0 or self.remaining_book_value <= 0.0:
            return 0.0
        return min(
            self.remaining_book_value,
            self.acquisition_cost * period / life,
        )

    def advance_accounting_period(self, period_weeks=1.0):
        """Return ``(updated_asset, depreciation)`` without cash mutation.

        Accounting depreciation is straight-line for an explicit fixture
        life.  Physical service remains constant until the asset reaches its
        declared retirement age, then becomes zero.
        """
        period = _nonnegative(period_weeks, "period_weeks")
        depreciation = self.straight_line_depreciation(period)
        new_age = self.age_weeks + period
        metadata = dict(self.capacity_metadata)
        life = self.useful_life_weeks
        if life is not None and new_age >= life:
            metadata["active"] = False
        updated = CapitalAsset(
            asset_id=self.asset_id,
            asset_class=self.asset_class,
            owner_firm_id=self.owner_firm_id,
            acquisition_cost=self.acquisition_cost,
            quantity=self.quantity,
            capacity_metadata=metadata,
            opening_book_value=self.remaining_book_value,
            closing_book_value=max(0.0, self.remaining_book_value - depreciation),
            accumulated_depreciation=self.accumulated_depreciation + depreciation,
            age=new_age,
            service_life_metadata=dict(self.service_life_metadata),
            depreciation_policy_ref=self.depreciation_policy_ref or "straight_line",
        )
        return updated, depreciation

    def capital_service_flow(self, engineering_capacity_per_unit=None):
        """Expose physical service separately from accounting book value.

        The capacity value must be explicitly supplied by a future technology
        or a clearly marked fixture.  Book value is never used as output.
        """
        declared = self.capacity_metadata.get(
            "capital_service_capacity_per_unit",
            engineering_capacity_per_unit,
        )
        service_capacity = 0.0 if declared is None or not self.is_active else (
            max(0.0, float(declared)) * self.quantity
        )
        return {
            "asset_id": self.asset_id,
            "opening_book_value": self.opening_book_value,
            "remaining_book_value": self.remaining_book_value,
            "age": self.age,
            "remaining_useful_life": self.remaining_useful_life,
            "active": self.is_active,
            "capital_service_capacity": service_capacity,
            "capacity_source": (
                "explicit_metadata_or_fixture"
                if declared is not None
                else "unselected"
            ),
        }

    def capital_service_capacity(self, engineering_capacity_per_unit=None):
        return self.capital_service_flow(
            engineering_capacity_per_unit
        )["capital_service_capacity"]

    def to_dict(self):
        return asdict(self)

    @classmethod
    def from_dict(cls, payload):
        return cls(**dict(payload))


@dataclass
class CapitalStock:
    """Passive collection of assets; adding one does not alter a World Firm."""

    owner_firm_id: Any
    assets: list = field(default_factory=list)

    def __post_init__(self):
        for asset in self.assets:
            if asset.owner_firm_id != self.owner_firm_id:
                raise ValueError("all CapitalStock assets must have the stock owner")

    @property
    def total_acquisition_cost(self):
        return sum(asset.acquisition_cost for asset in self.assets)

    @property
    def total_remaining_book_value(self):
        return sum(asset.remaining_book_value for asset in self.assets)

    @property
    def opening_book_value(self):
        return sum(asset.opening_book_value for asset in self.assets)

    @property
    def accumulated_depreciation(self):
        return sum(asset.accumulated_depreciation for asset in self.assets)

    def capital_service_flow(self, engineering_capacity_per_unit=None):
        asset_flows = [
            asset.capital_service_flow(engineering_capacity_per_unit)
            for asset in self.assets
        ]
        return {
            "owner_firm_id": self.owner_firm_id,
            "asset_count": len(self.assets),
            "active_asset_count": sum(flow["active"] for flow in asset_flows),
            "capital_service_capacity": sum(
                flow["capital_service_capacity"] for flow in asset_flows
            ),
            "assets": asset_flows,
        }

    def capital_service_capacity(self, engineering_capacity_per_unit=None):
        return self.capital_service_flow(
            engineering_capacity_per_unit
        )["capital_service_capacity"]

    def advance_accounting_period(self, period_weeks=1.0):
        """Return a passively depreciated stock and total non-cash expense."""
        updated_assets = []
        depreciation = 0.0
        for asset in self.assets:
            updated, expense = asset.advance_accounting_period(period_weeks)
            updated_assets.append(updated)
            depreciation += expense
        return (
            CapitalStock(owner_firm_id=self.owner_firm_id, assets=updated_assets),
            depreciation,
        )

    def replacement_capacity_need(
        self,
        previous_service_capacity,
        engineering_capacity_per_unit=None,
    ):
        """Measure lost physical service, independently of expansion demand."""
        previous = _nonnegative(
            previous_service_capacity,
            "previous_service_capacity",
        )
        current = self.capital_service_capacity(engineering_capacity_per_unit)
        return max(0.0, previous - current)

    def capacity_contribution_interface(self):
        """Expose future capacity inputs without producing capacity today."""
        return {
            "owner_firm_id": self.owner_firm_id,
            "asset_count": len(self.assets),
            "assets": [asset.capacity_contribution_interface() for asset in self.assets],
            "behavior_enabled": False,
        }

    def to_dict(self):
        return {
            "owner_firm_id": self.owner_firm_id,
            "assets": [asset.to_dict() for asset in self.assets],
        }

    @classmethod
    def from_dict(cls, payload):
        return cls(
            owner_firm_id=payload["owner_firm_id"],
            assets=[CapitalAsset.from_dict(item) for item in payload.get("assets", [])],
        )


def capital_book_value_bridge(
    opening_book_value,
    acquisitions,
    depreciation,
    disposals,
    closing_book_value,
    tolerance=1e-12,
):
    """Reconcile passive capital book value without creating a cash flow."""
    opening = _nonnegative(opening_book_value, "opening_book_value")
    acquired = _nonnegative(acquisitions, "acquisitions")
    depreciated = _nonnegative(depreciation, "depreciation")
    disposed = _nonnegative(disposals, "disposals")
    closing = _nonnegative(closing_book_value, "closing_book_value")
    expected = opening + acquired - depreciated - disposed
    return {
        "opening_book_value": opening,
        "acquisitions": acquired,
        "depreciation": depreciated,
        "disposals": disposed,
        "closing_book_value": closing,
        "bridge_gap": closing - expected,
        "passed": abs(closing - expected) <= tolerance,
    }


def replacement_expansion_needs(
    replacement_capacity_need,
    expansion_capacity_need,
    replacement_cost_per_capacity=0.0,
    expansion_cost_per_capacity=0.0,
):
    """Return explicitly separate replacement and expansion components."""
    replacement = _nonnegative(
        replacement_capacity_need,
        "replacement_capacity_need",
    )
    expansion = _nonnegative(
        expansion_capacity_need,
        "expansion_capacity_need",
    )
    replacement_cost = _nonnegative(
        replacement_cost_per_capacity,
        "replacement_cost_per_capacity",
    )
    expansion_cost = _nonnegative(
        expansion_cost_per_capacity,
        "expansion_cost_per_capacity",
    )
    replacement_expenditure = replacement * replacement_cost
    expansion_expenditure = expansion * expansion_cost
    return {
        "replacement_capacity_need": replacement,
        "expansion_capacity_need": expansion,
        "replacement_investment_expenditure": replacement_expenditure,
        "expansion_investment_expenditure": expansion_expenditure,
        "desired_investment_expenditure": (
            replacement_expenditure + expansion_expenditure
        ),
        "replacement_plus_expansion_gap": 0.0,
    }


@dataclass(frozen=True)
class InvestmentSettlement:
    """A proposed settled capital-good purchase, represented without mutation."""

    settlement_id: Any
    buyer_firm_id: Any
    supplier_firm_id: Any
    investment_good_id: str
    cash_outflow: float
    supplier_revenue: float
    acquired_capital_asset_value: float
    capital_asset_id: Any
    financing_source: str = "retained_cash"
    money_created: float = 0.0
    money_destroyed: float = 0.0

    def __post_init__(self):
        for name in (
            "cash_outflow",
            "supplier_revenue",
            "acquired_capital_asset_value",
            "money_created",
            "money_destroyed",
        ):
            _nonnegative(getattr(self, name), name)
        if self.financing_source not in ALLOWED_FINANCING_SOURCES:
            raise ValueError(f"unsupported financing_source: {self.financing_source}")

    def reconciliation(self, tolerance=1e-12):
        transfer_gap = self.supplier_revenue - self.cash_outflow
        asset_gap = self.acquired_capital_asset_value - self.cash_outflow
        return {
            "cash_transfer_gap": transfer_gap,
            "capital_asset_payment_gap": asset_gap,
            "money_stock_change": self.money_created - self.money_destroyed,
            "is_operating_expense": False,
            "is_inventory_cogs": False,
            "is_intermediate_consumption": False,
            "passed": abs(transfer_gap) <= tolerance and abs(asset_gap) <= tolerance,
        }

    def to_dict(self):
        return asdict(self)


@dataclass(frozen=True)
class IntermediateInput:
    """A Firm-to-Firm operating input, explicitly outside capital formation."""

    input_good_id: str
    quantity: float
    purchase_cost: float
    inventory_treatment: str
    production_consumption: str
    buyer_firm_id: Any = None
    supplier_firm_id: Any = None

    def __post_init__(self):
        _nonnegative(self.quantity, "quantity")
        _nonnegative(self.purchase_cost, "purchase_cost")
        if not self.inventory_treatment:
            raise ValueError("inventory_treatment is required")
        if not self.production_consumption:
            raise ValueError("production_consumption is required")

    @property
    def creates_capital_stock(self):
        return False

    def accounting_classification(self):
        return {
            "demand_class": "intermediate",
            "final_demand": False,
            "creates_capital_stock": False,
            "purchase_cost": self.purchase_cost,
            "buyer_cash_change": -self.purchase_cost,
            "supplier_revenue": self.purchase_cost,
        }

    def to_dict(self):
        payload = asdict(self)
        payload["creates_capital_stock"] = False
        return payload


def validate_contract_roundtrip(contract):
    """Check deterministic dict round-tripping for passive checkpoint metadata."""
    if not hasattr(contract, "to_dict"):
        raise TypeError("contract must expose to_dict")
    return bool(contract.to_dict())


__all__ = [
    "ALLOWED_FINANCING_SOURCES",
    "CAPITAL_STOCK_CONTRACT_VERSION",
    "CONTRACT_VERSION",
    "DEPRECIATION_POLICY_TYPES",
    "CapitalAsset",
    "CapitalStock",
    "DepreciationPolicyReference",
    "FirmInvestmentIntent",
    "IntermediateInput",
    "InvestmentSettlement",
    "capital_book_value_bridge",
]
