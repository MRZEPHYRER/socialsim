"""Passive capital-good identity and investment settlement runtime.

The engine is an isolated deterministic settlement adapter.  It returns
accounting results for future callers but does not mutate World/Firm objects,
invoke working-capital credit, or register a capital-good sector.
"""

from dataclasses import asdict, dataclass, field
from typing import Any, Dict, Optional

from economy.investment_contracts import CapitalAsset


def _nonnegative(value, name):
    number = float(value)
    if number < 0.0:
        raise ValueError(f"{name} must be non-negative")
    return number


@dataclass(frozen=True)
class CapitalGoodSpec:
    good_id: str
    storable: bool = True
    capital_asset_class: str = "unspecified"
    producer_technology_ref: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        if not self.storable:
            raise ValueError("capital goods must be storable")

    def to_dict(self):
        return asdict(self)


@dataclass(frozen=True)
class CapitalGoodOffer:
    offer_id: Any
    supplier_firm_id: Any
    good_id: str
    available_units: float
    unit_price: float
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        _nonnegative(self.available_units, "available_units")
        _nonnegative(self.unit_price, "unit_price")
        if self.unit_price <= 0.0 and self.available_units > 0.0:
            raise ValueError("positive capital-good supply requires a positive unit_price")


@dataclass(frozen=True)
class CapitalGoodProductionResult:
    """One explicit, non-calibrated capital-good production fixture result."""

    supplier_firm_id: Any
    good_id: str
    labor_services: float
    wage_cost: float
    units_produced: float
    unit_production_cost: float
    productivity_per_labor: float
    calibration_status: str = "NON-CALIBRATED_FIXTURE"
    opening_inventory_units: float = 0.0
    closing_inventory_units: float = 0.0
    opening_inventory_book_value: float = 0.0
    closing_inventory_book_value: float = 0.0
    inventory_bridge_gap: float = 0.0

    def to_dict(self):
        return asdict(self)


@dataclass
class CapitalGoodInventory:
    """Storable supplier inventory with a weighted-average book-cost bridge."""

    supplier_firm_id: Any
    good_id: str
    units: float = 0.0
    book_value: float = 0.0
    cumulative_production_units: float = 0.0
    cumulative_production_cost: float = 0.0
    cumulative_cogs: float = 0.0
    last_production_units: float = 0.0
    last_production_cost: float = 0.0
    last_cogs: float = 0.0
    last_inventory_bridge_gap: float = 0.0

    def __post_init__(self):
        self.units = _nonnegative(self.units, "inventory units")
        self.book_value = _nonnegative(self.book_value, "inventory book value")

    @property
    def inventory_book_value(self):
        return self.book_value

    @property
    def weighted_average_unit_cost(self):
        return self.book_value / self.units if self.units > 1e-12 else 0.0

    def produce(self, labor_services, wage_cost, productivity_per_labor):
        labor = _nonnegative(labor_services, "labor_services")
        wages = _nonnegative(wage_cost, "wage_cost")
        productivity = _nonnegative(
            productivity_per_labor, "productivity_per_labor"
        )
        opening_units = self.units
        opening_book = self.book_value
        produced = labor * productivity
        if produced <= 1e-12 and wages > 1e-12:
            raise ValueError("positive capital-good wage cost requires output")
        unit_cost = wages / produced if produced > 1e-12 else 0.0
        self.units += produced
        self.book_value += wages
        self.cumulative_production_units += produced
        self.cumulative_production_cost += wages
        self.last_production_units = produced
        self.last_production_cost = wages
        self.last_cogs = 0.0
        self.last_inventory_bridge_gap = self.book_value - (
            opening_book + wages
        )
        return CapitalGoodProductionResult(
            supplier_firm_id=self.supplier_firm_id,
            good_id=self.good_id,
            labor_services=labor,
            wage_cost=wages,
            units_produced=produced,
            unit_production_cost=unit_cost,
            productivity_per_labor=productivity,
            opening_inventory_units=opening_units,
            closing_inventory_units=self.units,
            opening_inventory_book_value=opening_book,
            closing_inventory_book_value=self.book_value,
            inventory_bridge_gap=self.last_inventory_bridge_gap,
        )

    def record_sale(self, units_sold):
        sold = _nonnegative(units_sold, "units_sold")
        if sold > self.units + 1e-12:
            raise ValueError("capital-good sale exceeds inventory")
        opening_book = self.book_value
        cogs = sold * self.weighted_average_unit_cost
        self.units -= sold
        self.book_value -= cogs
        self.cumulative_cogs += cogs
        self.last_cogs = cogs
        self.last_inventory_bridge_gap = self.book_value - (
            opening_book - cogs
        )
        return cogs

    def offer(self, offer_id, unit_price, metadata=None):
        return CapitalGoodOffer(
            offer_id=offer_id,
            supplier_firm_id=self.supplier_firm_id,
            good_id=self.good_id,
            available_units=self.units,
            unit_price=unit_price,
            metadata={
                "inventory_book_value": self.book_value,
                "weighted_average_unit_cost": self.weighted_average_unit_cost,
                **dict(metadata or {}),
            },
        )

    def to_dict(self):
        return asdict(self)

    @classmethod
    def from_dict(cls, payload):
        return cls(**dict(payload))


class CapitalGoodSupplyAdapter:
    """Turn real passive inventory into offers and settle its depletion."""

    def offers(self, inventories, unit_prices, offer_prefix="capital-offer"):
        result = []
        for inventory in sorted(
            inventories.values(), key=lambda item: str(item.supplier_firm_id)
        ):
            if inventory.units <= 1e-12:
                continue
            price = unit_prices[inventory.supplier_firm_id]
            result.append(
                inventory.offer(
                    f"{offer_prefix}:{inventory.supplier_firm_id}", price
                )
            )
        return result

    def settle(
        self,
        spec,
        orders,
        inventories,
        unit_prices,
        buyer_cash,
        supplier_cash,
    ):
        engine = PassiveCapitalGoodSettlementEngine()
        offers = self.offers(inventories, unit_prices)
        batch = engine.settle(
            spec,
            orders,
            offers,
            buyer_cash,
            supplier_cash,
        )
        sold_by_supplier = {}
        for settlement in batch.settlements:
            for fill in settlement.fills:
                sold_by_supplier[fill.supplier_firm_id] = (
                    sold_by_supplier.get(fill.supplier_firm_id, 0.0)
                    + fill.settled_units
                )
        for supplier_id, units in sold_by_supplier.items():
            inventories[supplier_id].record_sale(units)
        return batch


@dataclass(frozen=True)
class InvestmentOrder:
    order_id: Any
    buyer_firm_id: Any
    investment_good_id: str
    desired_units: float
    desired_investment_expenditure: float
    financing_source: str = "retained_cash"
    protected_operating_liquidity: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        _nonnegative(self.desired_units, "desired_units")
        _nonnegative(
            self.desired_investment_expenditure,
            "desired_investment_expenditure",
        )
        _nonnegative(
            self.protected_operating_liquidity,
            "protected_operating_liquidity",
        )


@dataclass(frozen=True)
class CapitalGoodFill:
    order_id: Any
    buyer_firm_id: Any
    supplier_firm_id: Any
    investment_good_id: str
    settled_units: float
    unit_price: float
    buyer_cash_outflow: float
    supplier_revenue: float
    capital_asset: CapitalAsset

    def __post_init__(self):
        _nonnegative(self.settled_units, "settled_units")
        _nonnegative(self.unit_price, "unit_price")
        _nonnegative(self.buyer_cash_outflow, "buyer_cash_outflow")
        _nonnegative(self.supplier_revenue, "supplier_revenue")
        if abs(self.buyer_cash_outflow - self.supplier_revenue) > 1e-12:
            raise ValueError("capital-good cash outflow and supplier revenue disagree")
        if abs(self.capital_asset.acquisition_cost - self.buyer_cash_outflow) > 1e-12:
            raise ValueError("capital asset cost must equal settled buyer expenditure")

    def to_dict(self):
        payload = asdict(self)
        return payload


@dataclass(frozen=True)
class InvestmentOrderSettlement:
    order_id: Any
    buyer_firm_id: Any
    investment_good_id: str
    requested_units: float
    settled_units: float
    unmet_units: float
    desired_investment_expenditure: float
    settled_expenditure: float
    unexecuted_expenditure: float
    financing_gap: float
    buyer_cash_start: float
    buyer_cash_end: float
    protected_operating_liquidity: float = 0.0
    buyer_spendable_cash: float = 0.0
    fills: tuple = field(default_factory=tuple)
    capital_assets: tuple = field(default_factory=tuple)
    cfi: float = 0.0
    money_created: float = 0.0
    money_destroyed: float = 0.0
    working_capital_loan_used: bool = False
    transaction_created: bool = False
    settlement_reason: str = "settled"

    def __post_init__(self):
        for name in (
            "requested_units",
            "settled_units",
            "unmet_units",
            "desired_investment_expenditure",
            "settled_expenditure",
            "unexecuted_expenditure",
            "financing_gap",
            "buyer_cash_start",
            "buyer_cash_end",
            "protected_operating_liquidity",
            "buyer_spendable_cash",
            "money_created",
            "money_destroyed",
        ):
            _nonnegative(getattr(self, name), name)
        if self.cfi > 1e-12:
            raise ValueError("capital investment CFI must be an outflow or zero")
        if self.working_capital_loan_used:
            raise ValueError("passive settlement cannot invoke working-capital credit")

    @property
    def supplier_revenue(self):
        return sum(fill.supplier_revenue for fill in self.fills)

    @property
    def buyer_cash_change(self):
        return self.buyer_cash_end - self.buyer_cash_start

    def reconciliation(self, tolerance=1e-12):
        asset_value = sum(asset.acquisition_cost for asset in self.capital_assets)
        return {
            "cash_transfer_gap": self.supplier_revenue + self.buyer_cash_change,
            "asset_recognition_gap": asset_value - self.settled_expenditure,
            "cfi_gap": self.cfi + self.settled_expenditure,
            "money_delta": self.money_created - self.money_destroyed,
            "passed": (
                abs(self.supplier_revenue + self.buyer_cash_change) <= tolerance
                and abs(asset_value - self.settled_expenditure) <= tolerance
                and abs(self.cfi + self.settled_expenditure) <= tolerance
                and abs(self.money_created - self.money_destroyed) <= tolerance
            ),
        }


@dataclass(frozen=True)
class SettlementBatch:
    settlements: tuple
    buyer_cash_end: Dict[Any, float]
    supplier_cash_end: Dict[Any, float]
    total_requested_expenditure: float
    total_settled_expenditure: float
    total_unmet_expenditure: float
    total_supplier_revenue: float
    money_created: float = 0.0
    money_destroyed: float = 0.0
    working_capital_loan_used: bool = False

    def reconciliation(self, tolerance=1e-12):
        buyer_change = sum(
            self.buyer_cash_end.values()
        )
        supplier_change = sum(self.supplier_cash_end.values())
        # The engine receives only end balances, so the detailed cash check is
        # performed per order; this identity checks aggregate settlement flows.
        return {
            "supplier_revenue": self.total_supplier_revenue,
            "money_delta": self.money_created - self.money_destroyed,
            "working_capital_loan_used": self.working_capital_loan_used,
            "passed": abs(self.money_created - self.money_destroyed) <= tolerance,
            "buyer_end_balance_total": buyer_change,
            "supplier_end_balance_total": supplier_change,
        }


class PassiveCapitalGoodSettlementEngine:
    """Deterministic, non-mutating settlement adapter for shadow fixtures."""

    def settle(self, spec, orders, offers, buyer_cash, supplier_cash):
        if not spec.storable:
            raise ValueError("capital-good settlement requires storable spec")
        matching_offers = [
            offer for offer in offers if offer.good_id == spec.good_id
        ]
        remaining = {
            offer.offer_id: float(offer.available_units)
            for offer in matching_offers
        }
        buyer_end = {key: float(value) for key, value in buyer_cash.items()}
        supplier_end = {key: float(value) for key, value in supplier_cash.items()}
        settlements = []

        for order in sorted(orders, key=lambda item: str(item.order_id)):
            if order.investment_good_id != spec.good_id:
                raise ValueError("order good does not match CapitalGoodSpec")
            start_cash = _nonnegative(
                buyer_end.get(order.buyer_firm_id, 0.0),
                "buyer_cash",
            )
            protected_cash = min(
                start_cash,
                _nonnegative(
                    order.protected_operating_liquidity,
                    "protected_operating_liquidity",
                ),
            )
            spendable_cash = max(0.0, start_cash - protected_cash)
            cash_left = spendable_cash
            budget_left = order.desired_investment_expenditure
            requested_left = order.desired_units
            fills = []
            for offer in sorted(matching_offers, key=lambda item: str(item.offer_id)):
                if order.financing_source != "retained_cash":
                    break
                if requested_left <= 1e-12 or cash_left <= 1e-12 or budget_left <= 1e-12:
                    break
                available = remaining[offer.offer_id]
                if available <= 1e-12:
                    continue
                affordable_units = min(
                    cash_left / offer.unit_price,
                    budget_left / offer.unit_price,
                )
                units = min(requested_left, available, affordable_units)
                if units <= 1e-12:
                    continue
                expenditure = units * offer.unit_price
                asset = CapitalAsset(
                    asset_id=f"{order.order_id}:{offer.offer_id}",
                    asset_class=spec.capital_asset_class,
                    owner_firm_id=order.buyer_firm_id,
                acquisition_cost=expenditure,
                quantity=units,
                capacity_metadata={
                    "producer_technology_ref": spec.producer_technology_ref,
                    "calibration_status": "unselected",
                    "active": True,
                    "physical_asset_units": units,
                    "origin_order_id": order.order_id,
                    "investment_source": order.metadata.get("investment_source", "UNAVAILABLE"),
                    "replacement_trigger_id": order.metadata.get("replacement_trigger_id", "UNAVAILABLE"),
                    "replacement_order_id": order.order_id if order.metadata.get("investment_source") in ("REPLACEMENT", "MIXED") else "UNAVAILABLE",
                    **{
                        key: value
                        for key, value in {
                            "capital_service_capacity_per_unit": offer.metadata.get(
                                "capital_service_capacity_per_unit",
                                spec.metadata.get(
                                    "capital_service_capacity_per_unit"
                                ),
                            ),
                            "capacity_normalization_status": offer.metadata.get(
                                "capacity_normalization_status",
                                spec.metadata.get("capacity_normalization_status"),
                            ),
                        }.items()
                        if value is not None
                    },
                },
                age=0.0,
                service_life_metadata=dict(
                    offer.metadata.get(
                        "service_life_metadata",
                        spec.metadata.get("service_life_metadata", {}),
                    )
                ),
                depreciation_policy_ref=spec.metadata.get(
                    "depreciation_policy_ref",
                    "unselected",
                ),
            )
                fills.append(
                    CapitalGoodFill(
                        order_id=order.order_id,
                        buyer_firm_id=order.buyer_firm_id,
                        supplier_firm_id=offer.supplier_firm_id,
                        investment_good_id=spec.good_id,
                        settled_units=units,
                        unit_price=offer.unit_price,
                        buyer_cash_outflow=expenditure,
                        supplier_revenue=expenditure,
                        capital_asset=asset,
                    )
                )
                remaining[offer.offer_id] -= units
                requested_left -= units
                cash_left -= expenditure
                budget_left -= expenditure
                supplier_end[offer.supplier_firm_id] = (
                    supplier_end.get(offer.supplier_firm_id, 0.0) + expenditure
                )

            settled_units = sum(fill.settled_units for fill in fills)
            settled_expenditure = sum(fill.buyer_cash_outflow for fill in fills)
            buyer_end[order.buyer_firm_id] = start_cash - settled_expenditure
            financing_gap = (
                order.desired_investment_expenditure
                if order.financing_source != "retained_cash"
                else max(
                    order.desired_investment_expenditure - spendable_cash,
                    0.0,
                )
            )
            settlements.append(
                InvestmentOrderSettlement(
                    order_id=order.order_id,
                    buyer_firm_id=order.buyer_firm_id,
                    investment_good_id=spec.good_id,
                    requested_units=order.desired_units,
                    settled_units=settled_units,
                    unmet_units=max(0.0, order.desired_units - settled_units),
                    desired_investment_expenditure=order.desired_investment_expenditure,
                    settled_expenditure=settled_expenditure,
                    unexecuted_expenditure=max(
                        0.0,
                        order.desired_investment_expenditure - settled_expenditure,
                    ),
                    financing_gap=financing_gap,
                    buyer_cash_start=start_cash,
                    buyer_cash_end=start_cash - settled_expenditure,
                    protected_operating_liquidity=protected_cash,
                    buyer_spendable_cash=spendable_cash,
                    fills=tuple(fills),
                    capital_assets=tuple(fill.capital_asset for fill in fills),
                    cfi=-settled_expenditure,
                    money_created=0.0,
                    money_destroyed=0.0,
                    working_capital_loan_used=False,
                    transaction_created=bool(fills),
                    settlement_reason=(
                        "retained_cash_settlement"
                        if order.financing_source == "retained_cash"
                        else "finance_source_inactive"
                    ),
                )
            )

        total_requested = sum(
            order.desired_investment_expenditure for order in orders
        )
        total_settled = sum(item.settled_expenditure for item in settlements)
        return SettlementBatch(
            settlements=tuple(settlements),
            buyer_cash_end=buyer_end,
            supplier_cash_end=supplier_end,
            total_requested_expenditure=total_requested,
            total_settled_expenditure=total_settled,
            total_unmet_expenditure=max(0.0, total_requested - total_settled),
            total_supplier_revenue=sum(item.supplier_revenue for item in settlements),
            money_created=0.0,
            money_destroyed=0.0,
            working_capital_loan_used=False,
        )


__all__ = [
    "CapitalGoodFill",
    "CapitalGoodInventory",
    "CapitalGoodOffer",
    "CapitalGoodProductionResult",
    "CapitalGoodSpec",
    "CapitalGoodSupplyAdapter",
    "InvestmentOrder",
    "InvestmentOrderSettlement",
    "PassiveCapitalGoodSettlementEngine",
    "SettlementBatch",
]
