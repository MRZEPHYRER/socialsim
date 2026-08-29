"""Passive multi-sector contracts for Step 14A.

The Food executor remains in ``World`` and ``FirmSystem``.  This module only
provides serializable identities, deterministic registries, and read-only
views over already-settled Food state.  It intentionally contains no random
number generation and no economic write-back.
"""

from dataclasses import dataclass, field
import math

from productivity import age_productivity
from economy.service_accounting import PassiveServiceAccountingAdapter


FOOD_SECTOR_ID = "food"
GENERIC_SERVICE_SECTOR_ID = "generic_services"
FOOD_TECHNOLOGY_ID = "labor_only_food_v1"
SERVICE_TECHNOLOGY_ID = "passive_service_v1"
FOOD_INVENTORY_POLICY_ID = "food_inventory_v1"
NO_INVENTORY_POLICY_ID = "no_inventory_v1"
GENERIC_SERVICE_GOOD_ID = "generic_service_good"
CAPITAL_GOODS_SECTOR_ID = "capital_goods"
CAPITAL_GOODS_TECHNOLOGY_ID = "labor_only_capital_goods_v1"
CAPITAL_GOOD_INVENTORY_POLICY_ID = "capital_good_inventory_v1"
CAPITAL_MACHINE_GOOD_ID = "capital_machine_good"


@dataclass(frozen=True)
class SectorSpec:
    sector_id: str
    name: str
    default_technology_id: str
    output_good_ids: tuple = field(default_factory=tuple)
    default_inventory_policy_id: str = NO_INVENTORY_POLICY_ID
    operating_metadata: dict = field(default_factory=dict)


@dataclass(frozen=True)
class TechnologySpec:
    technology_id: str
    name: str
    input_semantics: str
    output_semantics: str
    active: bool = False


@dataclass(frozen=True)
class InventoryPolicySpec:
    inventory_policy_id: str
    name: str
    carryover: bool
    desired_inventory_semantics: str
    active: bool = False


@dataclass(frozen=True)
class GenericFinanceView:
    firm_id: int
    scheduled_payroll: float
    target_cash: float
    requested_credit: float
    executed_credit: float
    payroll_funding_ratio: float
    principal: float
    arrears: float


class DeterministicRegistry:
    """Small insertion-ordered registry with deterministic keyed access."""

    def __init__(self, values=()):
        self._values = {}
        for value in values:
            key = self._key(value)
            if key in self._values:
                raise ValueError(f"Duplicate registry key: {key}")
            self._values[key] = value

    @staticmethod
    def _key(value):
        for name in ("sector_id", "technology_id", "inventory_policy_id"):
            if hasattr(value, name):
                return getattr(value, name)
        raise TypeError("Registry values must expose a stable identifier")

    def register(self, value):
        key = self._key(value)
        if key in self._values:
            raise ValueError(f"Duplicate registry key: {key}")
        self._values[key] = value
        return value

    def get(self, key):
        return self._values[key]

    def has(self, key):
        return key in self._values

    def ids(self):
        return list(self._values.keys())

    def all(self):
        return list(self._values.values())


# Named aliases keep the architecture explicit while sharing one deterministic
# implementation.  They are metadata registries, not economic executors.
SectorRegistry = DeterministicRegistry
SectorCatalog = DeterministicRegistry
TechnologyRegistry = DeterministicRegistry
InventoryPolicyRegistry = DeterministicRegistry


class MarketRegistry:
    """Passive good-id -> active supplier Firm-id registry."""

    def __init__(self):
        self._suppliers = {}

    def set_suppliers(self, good_id, firm_ids):
        self._suppliers[str(good_id)] = sorted({int(firm_id) for firm_id in firm_ids})

    def suppliers(self, good_id, active_only=True):
        # ``active_only`` is reserved for a future lifecycle implementation;
        # all current firms are active and therefore share this view.
        return list(self._suppliers.get(str(good_id), ()))

    def active_suppliers(self, good_id):
        return self.suppliers(good_id, active_only=True)

    def goods(self):
        return list(self._suppliers.keys())

    def as_dict(self):
        return {good_id: list(ids) for good_id, ids in self._suppliers.items()}


CAPACITY_COMBINATION_SEMANTICS = {
    "labor_only",
    "leontief_min",
    "capital_augmented_cap",
}


@dataclass(frozen=True)
class CapacityDecomposition:
    """Explicit labor/capital capacity decomposition for future technologies."""

    labor_capacity: float
    capital_capacity: float
    base_capacity: float
    feasible_capacity: float
    combination: str

    def to_dict(self):
        return {
            "labor_capacity": self.labor_capacity,
            "capital_capacity": self.capital_capacity,
            "base_capacity": self.base_capacity,
            "feasible_capacity": self.feasible_capacity,
            "combination": self.combination,
        }


class ProductionTechnology:
    """Minimal generic production contract with no economic side effects."""

    technology_id = None

    def technical_capacity(self, labor_services, **kwargs):
        raise NotImplementedError

    def capital_capacity_contribution(self, capital_stock=None, **kwargs):
        """Future hook; no capital contribution is selected in Step 15B."""
        return 0.0

    def labor_capacity(self, labor_services, **kwargs):
        return max(0.0, float(self.technical_capacity(labor_services, **kwargs)))

    def capital_capacity(self, capital_stock=None, **kwargs):
        return max(
            0.0,
            float(
                self.capital_capacity_contribution(
                    capital_stock=capital_stock,
                    **kwargs,
                )
            ),
        )

    @staticmethod
    def feasible_capacity(
        labor_capacity,
        capital_capacity,
        base_capacity=0.0,
        combination="labor_only",
    ):
        if combination not in CAPACITY_COMBINATION_SEMANTICS:
            raise ValueError(f"unsupported capacity combination: {combination}")
        labor = max(0.0, float(labor_capacity))
        capital = max(0.0, float(capital_capacity))
        base = max(0.0, float(base_capacity))
        if combination == "labor_only":
            return labor
        if combination == "leontief_min":
            return min(labor, capital)
        return min(labor, base + capital)

    def capacity_decomposition(
        self,
        labor_services,
        capital_stock=None,
        combination="labor_only",
        base_capacity=0.0,
        **kwargs,
    ):
        labor = self.labor_capacity(labor_services, **kwargs)
        capital = self.capital_capacity(capital_stock, **kwargs)
        feasible = self.feasible_capacity(
            labor,
            capital,
            base_capacity=base_capacity,
            combination=combination,
        )
        return CapacityDecomposition(
            labor_capacity=labor,
            capital_capacity=capital,
            base_capacity=max(0.0, float(base_capacity)),
            feasible_capacity=feasible,
            combination=combination,
        )

    def technical_capacity_with_capital(
        self,
        labor_services,
        capital_stock=None,
        capacity_combination="labor_only",
        base_capacity=0.0,
        **kwargs,
    ):
        """Return feasible capacity without changing current labor-only paths."""
        return self.capacity_decomposition(
            labor_services,
            capital_stock=capital_stock,
            combination=capacity_combination,
            base_capacity=base_capacity,
            **kwargs,
        ).feasible_capacity

    @staticmethod
    def feasible_output(desired_output, technical_capacity):
        return min(max(0.0, float(desired_output)), max(0.0, float(technical_capacity)))

    @staticmethod
    def funded_output(feasible_output, funded_capacity):
        return min(max(0.0, float(feasible_output)), max(0.0, float(funded_capacity)))

    @staticmethod
    def realized_output(funded_output, current_demand):
        return min(max(0.0, float(funded_output)), max(0.0, float(current_demand)))


@dataclass(frozen=True)
class ServiceFlowResult:
    desired_output: float
    technical_capacity: float
    feasible_output: float
    funded_capacity: float
    funded_output: float
    current_demand: float
    realized_output: float
    unused_capacity: float
    unmet_demand: float
    ending_inventory: float = 0.0
    spoilage_units: float = 0.0
    spoilage_book_loss: float = 0.0


class LaborOnlyServiceTechnology(ProductionTechnology):
    """Passive labor-only Service technology.

    A productivity coefficient is deliberately optional.  Tests may inject a
    transparent TEST-ONLY value; no permanent Service calibration exists in
    model configuration.
    """

    technology_id = SERVICE_TECHNOLOGY_ID

    def __init__(self, productivity_per_labor=None):
        self.productivity_per_labor = (
            None
            if productivity_per_labor is None
            else float(productivity_per_labor)
        )
        self.calibration_status = (
            "TEST-ONLY"
            if self.productivity_per_labor is not None
            else "UNSELECTED"
        )

    def technical_capacity(self, labor_services, productivity_per_labor=None):
        coefficient = (
            self.productivity_per_labor
            if productivity_per_labor is None
            else float(productivity_per_labor)
        )
        if coefficient is None:
            raise ValueError(
                "Service productivity is uncalibrated; provide a TEST-ONLY "
                "coefficient for passive unit tests."
            )
        return max(0.0, float(labor_services)) * max(0.0, coefficient)

    def capital_capacity_contribution(self, capital_stock=None, **kwargs):
        # Service remains labor-only; future capital-enabled technologies may
        # override the generic interface without changing this path.
        return 0.0

    capacity = technical_capacity

    def flow(
        self,
        labor_services,
        desired_output,
        funded_ratio,
        current_demand,
        productivity_per_labor=None,
    ):
        technical = self.technical_capacity(
            labor_services,
            productivity_per_labor=productivity_per_labor,
        )
        feasible = self.feasible_output(desired_output, technical)
        funded_capacity = min(
            technical,
            technical * max(0.0, float(funded_ratio)),
        )
        funded = self.funded_output(feasible, funded_capacity)
        realized = self.realized_output(funded, current_demand)
        demand = max(0.0, float(current_demand))
        return ServiceFlowResult(
            desired_output=max(0.0, float(desired_output)),
            technical_capacity=technical,
            feasible_output=feasible,
            funded_capacity=funded_capacity,
            funded_output=funded,
            current_demand=demand,
            realized_output=realized,
            unused_capacity=max(0.0, funded - realized),
            unmet_demand=max(0.0, demand - realized),
        )


class LaborOnlyCapitalGoodsTechnology(ProductionTechnology):
    """Labor-only capital-good technology for explicit passive fixtures.

    A productivity coefficient is intentionally not configured in the model.
    Tests must provide a clearly marked engineering coefficient, so adding the
    capital-good sector cannot silently calibrate future production behavior.
    """

    technology_id = CAPITAL_GOODS_TECHNOLOGY_ID

    def __init__(self, productivity_per_labor=None):
        self.productivity_per_labor = (
            None
            if productivity_per_labor is None
            else float(productivity_per_labor)
        )
        self.calibration_status = (
            "NON-CALIBRATED_FIXTURE"
            if self.productivity_per_labor is not None
            else "UNSELECTED"
        )

    def technical_capacity(self, labor_services, productivity_per_labor=None):
        coefficient = (
            self.productivity_per_labor
            if productivity_per_labor is None
            else float(productivity_per_labor)
        )
        if coefficient is None:
            raise ValueError(
                "Capital-good productivity is unselected; provide a "
                "NON-CALIBRATED_FIXTURE coefficient for passive tests."
            )
        if coefficient < 0.0:
            raise ValueError("capital-good fixture productivity must be non-negative")
        return max(0.0, float(labor_services)) * coefficient

    capacity = technical_capacity


class NoInventoryPolicy:
    """Generic no-storage policy for non-storable Service goods."""

    inventory_policy_id = NO_INVENTORY_POLICY_ID

    is_inventory_applicable = False

    def desired_inventory(self, *args, **kwargs):
        return 0.0

    def opening_inventory(self, *args, **kwargs):
        return 0.0

    def ending_inventory(self, *args, **kwargs):
        return 0.0

    def carryover_inventory(self, *args, **kwargs):
        return 0.0

    def inventory_adjustment(self, *args, **kwargs):
        return 0.0

    def spoilage_units(self, *args, **kwargs):
        return 0.0

    def spoilage_book_loss(self, *args, **kwargs):
        return 0.0

    def inventory_book_value(self, *args, **kwargs):
        return 0.0

    def carryover(self, *args, **kwargs):
        return 0.0

    def spoilage(self, *args, **kwargs):
        return 0.0


INVENTORY_SOURCE_MODE = "INVENTORY"
CURRENT_PERIOD_CAPACITY_SOURCE_MODE = "CURRENT_PERIOD_CAPACITY"


@dataclass(frozen=True)
class SupplyAvailabilityView:
    good_id: str
    firm_id: int
    available_units: float
    source_mode: str


class FoodSupplyAvailabilityAdapter:
    """Read-only view of the current Food inventory supplier quantity."""

    source_mode = INVENTORY_SOURCE_MODE

    def view(self, firm, good_id):
        return SupplyAvailabilityView(
            good_id=str(good_id),
            firm_id=int(firm.firm_id),
            available_units=max(0.0, float(getattr(firm, "inventory_units", 0.0))),
            source_mode=self.source_mode,
        )


class ServiceSupplyAvailabilityAdapter:
    """Test-only passive view of current funded Service capacity."""

    source_mode = CURRENT_PERIOD_CAPACITY_SOURCE_MODE

    def view(self, firm_id, good_id, funded_service_capacity):
        return SupplyAvailabilityView(
            good_id=str(good_id),
            firm_id=int(firm_id),
            available_units=max(0.0, float(funded_service_capacity)),
            source_mode=self.source_mode,
        )


@dataclass(frozen=True)
class ServiceSettlementLine:
    firm_id: int
    demand_units: float
    available_units: float
    sold_units: float
    price: float
    revenue: float
    unmet_demand: float
    source_mode: str = CURRENT_PERIOD_CAPACITY_SOURCE_MODE


@dataclass(frozen=True)
class ServiceSettlementResult:
    lines: tuple
    total_demand: float
    total_sold: float
    total_revenue: float
    total_unmet_demand: float


class ServiceMarketSettlementAdapter:
    """Deterministic current-capacity settlement for isolated Service tests."""

    def settle(self, demand_by_firm, capacity_by_firm, price_by_firm):
        firm_ids = sorted(
            set(demand_by_firm) | set(capacity_by_firm) | set(price_by_firm)
        )
        lines = []
        for firm_id in firm_ids:
            demand = max(0.0, float(demand_by_firm.get(firm_id, 0.0)))
            capacity = max(0.0, float(capacity_by_firm.get(firm_id, 0.0)))
            price = max(0.0, float(price_by_firm.get(firm_id, 0.0)))
            sold = min(demand, capacity)
            lines.append(
                ServiceSettlementLine(
                    firm_id=int(firm_id),
                    demand_units=demand,
                    available_units=capacity,
                    sold_units=sold,
                    price=price,
                    revenue=sold * price,
                    unmet_demand=max(0.0, demand - sold),
                )
            )
        return ServiceSettlementResult(
            lines=tuple(lines),
            total_demand=math.fsum(line.demand_units for line in lines),
            total_sold=math.fsum(line.sold_units for line in lines),
            total_revenue=math.fsum(line.revenue for line in lines),
            total_unmet_demand=math.fsum(line.unmet_demand for line in lines),
        )


@dataclass(frozen=True)
class SectorCapacityShare:
    firm_id: int
    sector_id: str
    capacity: float
    sector_capacity_total: float
    share: float


class SectorLocalCapacityShareAdapter:
    """Dimensionless capacity shares within comparable suppliers only."""

    def shares(self, firms, capacities, sector_id):
        pairs = [
            (firm, max(0.0, float(capacities.get(firm.firm_id, 0.0))))
            for firm in firms
            if getattr(firm, "sector_id", FOOD_SECTOR_ID) == sector_id
        ]
        total = math.fsum(capacity for _, capacity in pairs)
        return tuple(
            SectorCapacityShare(
                firm_id=int(firm.firm_id),
                sector_id=str(sector_id),
                capacity=capacity,
                sector_capacity_total=total,
                share=(capacity / total if total > 0 else 0.0),
            )
            for firm, capacity in pairs
        )


class PassiveServiceTechnology:
    """Schema-only service technology; it is never executed in Step 14A."""

    technology_id = SERVICE_TECHNOLOGY_ID


@dataclass
class FirmOperatingView:
    firm_id: int
    sector_id: str
    technology_id: str
    inventory_policy_id: str
    operating_state_by_good: dict


@dataclass
class SectorAggregateView:
    sector_id: str
    output_good_ids: tuple
    employee_count: int
    labor_services: float
    production_value: float
    sales_value: float
    cash: float
    principal: float
    arrears: float


class MultiSectorFoundation:
    """Passive contracts and reconstructed current-state views."""

    VERSION = 1

    def __init__(self, goods_catalog, firms=()):
        self.goods_catalog = goods_catalog
        food_good_id = goods_catalog.ids()[0]
        self.sectors = DeterministicRegistry(
            [
                SectorSpec(
                    sector_id=FOOD_SECTOR_ID,
                    name="Food",
                    default_technology_id=FOOD_TECHNOLOGY_ID,
                    output_good_ids=(food_good_id,),
                    default_inventory_policy_id=FOOD_INVENTORY_POLICY_ID,
                    operating_metadata={"runtime": "canonical_food_executor"},
                ),
                SectorSpec(
                    sector_id=GENERIC_SERVICE_SECTOR_ID,
                    name="Generic Non-Food Services",
                    default_technology_id=SERVICE_TECHNOLOGY_ID,
                    output_good_ids=(GENERIC_SERVICE_GOOD_ID,),
                    default_inventory_policy_id=NO_INVENTORY_POLICY_ID,
                    operating_metadata={"active": False, "schema_only": True},
                ),
                SectorSpec(
                    sector_id=CAPITAL_GOODS_SECTOR_ID,
                    name="Capital Goods",
                    default_technology_id=CAPITAL_GOODS_TECHNOLOGY_ID,
                    output_good_ids=(CAPITAL_MACHINE_GOOD_ID,),
                    default_inventory_policy_id=CAPITAL_GOOD_INVENTORY_POLICY_ID,
                    operating_metadata={
                        "active": False,
                        "schema_only": True,
                        "canonical_firm_count": 0,
                    },
                ),
            ]
        )
        self.technologies = DeterministicRegistry(
            [
                TechnologySpec(
                    technology_id=FOOD_TECHNOLOGY_ID,
                    name="Labor-only Food Technology",
                    input_semantics="labor_services",
                    output_semantics="food_units",
                ),
                LaborOnlyServiceTechnology(),
                LaborOnlyCapitalGoodsTechnology(),
            ]
        )
        self.inventory_policies = DeterministicRegistry(
            [
                InventoryPolicySpec(
                    inventory_policy_id=FOOD_INVENTORY_POLICY_ID,
                    name="Canonical Food Inventory",
                    carryover=True,
                    desired_inventory_semantics="canonical_food_inventory",
                ),
                NoInventoryPolicy(),
                InventoryPolicySpec(
                    inventory_policy_id=CAPITAL_GOOD_INVENTORY_POLICY_ID,
                    name="Capital Good Inventory",
                    carryover=True,
                    desired_inventory_semantics="passive_capital_good_inventory",
                    active=False,
                ),
            ]
        )
        self.no_inventory_policy = NoInventoryPolicy()
        self.food_supply_adapter = FoodSupplyAvailabilityAdapter()
        self.service_supply_adapter = ServiceSupplyAvailabilityAdapter()
        self.service_accounting_adapter = PassiveServiceAccountingAdapter()
        self.service_good_template = None
        self.capital_good_template = None
        self.capital_good_firms = tuple()
        self.capital_good_orders = tuple()
        from economy.capital_goods import CapitalGoodSupplyAdapter

        self.capital_good_supply_adapter = CapitalGoodSupplyAdapter()
        self.market_registry = MarketRegistry()
        self.firm_operating_views = {}
        self.sector_views = {}
        self.finance_views = {}
        self.latest_reconciliation = {}
        self.validation_history = []
        self.collect_validation = False
        self.refresh_firm_registry(firms)
        self.activate_service_schema_template()
        self.activate_capital_good_schema_template()

    @property
    def food_sector(self):
        return self.sectors.get(FOOD_SECTOR_ID)

    @property
    def capital_goods_sector(self):
        return self.sectors.get(CAPITAL_GOODS_SECTOR_ID)

    def activate_service_schema_template(self):
        """Create an inactive GoodSpec template without adding it to markets."""
        from economy.goods import GoodSpec

        self.service_good_template = GoodSpec(
            id=GENERIC_SERVICE_GOOD_ID,
            name="Generic Non-Food Service",
            is_essential=False,
            is_storable=False,
            perish_rate=None,
            unit="service",
            physical=False,
            service=True,
            inventory_policy_compatibility=NO_INVENTORY_POLICY_ID,
        )
        return self.service_good_template

    def activate_capital_good_schema_template(self):
        """Create an inactive storable capital-good identity.

        The template is not added to household markets and does not create a
        supplier, inventory, production, order, cash flow, or asset.
        """
        from economy.goods import GoodSpec

        from economy.capital_goods import CapitalGoodSpec

        self.capital_good_template = GoodSpec(
            id=CAPITAL_MACHINE_GOOD_ID,
            name="Capital Machine Good",
            is_essential=False,
            is_storable=True,
            perish_rate=0.0,
            unit="capital_unit",
            physical=True,
            service=False,
            inventory_policy_compatibility=CAPITAL_GOOD_INVENTORY_POLICY_ID,
        )
        self.capital_good_contract = CapitalGoodSpec(
            good_id=CAPITAL_MACHINE_GOOD_ID,
            storable=True,
            capital_asset_class="future_machine",
            producer_technology_ref=CAPITAL_GOODS_TECHNOLOGY_ID,
            metadata={"productivity_calibration": "unselected"},
        )
        return self.capital_good_template

    def refresh_firm_registry(self, firms):
        firm_list = sorted(list(firms or ()), key=lambda firm: int(firm.firm_id))
        food_good_id = self.food_sector.output_good_ids[0]
        self.market_registry.set_suppliers(
            food_good_id,
            [
                firm.firm_id
                for firm in firm_list
                if self.firm_sector_id(firm) == FOOD_SECTOR_ID
            ],
        )
        self.market_registry.set_suppliers(
            GENERIC_SERVICE_GOOD_ID,
            [
                firm.firm_id
                for firm in firm_list
                if self.firm_sector_id(firm) == GENERIC_SERVICE_SECTOR_ID
            ],
        )
        self.capital_good_firms = tuple(
            firm
            for firm in firm_list
            if self.firm_sector_id(firm) == CAPITAL_GOODS_SECTOR_ID
        )
        self.market_registry.set_suppliers(
            CAPITAL_MACHINE_GOOD_ID,
            [firm.firm_id for firm in self.capital_good_firms],
        )

    @staticmethod
    def firm_sector_id(firm):
        return getattr(firm, "sector_id", FOOD_SECTOR_ID) or FOOD_SECTOR_ID

    @staticmethod
    def firm_technology_id(firm):
        return getattr(firm, "technology_id", FOOD_TECHNOLOGY_ID) or FOOD_TECHNOLOGY_ID

    @staticmethod
    def firm_inventory_policy_id(firm):
        return getattr(firm, "inventory_policy_id", FOOD_INVENTORY_POLICY_ID) or FOOD_INVENTORY_POLICY_ID

    def operating_state_by_good(self, firm):
        service = self.firm_sector_id(firm) == GENERIC_SERVICE_SECTOR_ID
        capital_goods = self.firm_sector_id(firm) == CAPITAL_GOODS_SECTOR_ID
        good_id = (
            GENERIC_SERVICE_GOOD_ID
            if service
            else CAPITAL_MACHINE_GOOD_ID
            if capital_goods
            else self.food_sector.output_good_ids[0]
        )
        return {
            good_id: {
                "production": float(
                    getattr(
                        firm,
                        "realized_service_output"
                        if service
                        else "capital_good_production_units"
                        if capital_goods
                        else "actual_production",
                        0.0,
                    )
                ),
                "sales": float(
                    getattr(
                        firm,
                        "service_sales_units"
                        if service
                        else "capital_good_sales_units"
                        if capital_goods
                        else "sales_units",
                        0.0,
                    )
                ),
                "inventory": (
                    0.0
                    if service
                    else float(
                        getattr(
                            getattr(firm, "capital_good_inventory", None),
                            "units",
                            0.0,
                        )
                        if capital_goods
                        else getattr(firm, "inventory_units", 0.0)
                    )
                ),
                "unmet_demand": float(getattr(firm, "unmet_demand", 0.0)),
                "revenue": float(getattr(firm, "sales_revenue", 0.0)),
            }
        }

    def food_supply_availability(self, firms):
        """Reconstruct current Food inventory availability without execution."""
        good_id = self.food_sector.output_good_ids[0]
        return [
            self.food_supply_adapter.view(firm, good_id)
            for firm in sorted(firms or (), key=lambda item: int(item.firm_id))
        ]

    def passive_service_supply_availability(
        self,
        firm_id,
        funded_service_capacity,
    ):
        """Build a test-only capacity supply view; never registers a supplier."""
        return self.service_supply_adapter.view(
            firm_id,
            GENERIC_SERVICE_GOOD_ID,
            funded_service_capacity,
        )

    def finance_view(self, firm):
        return GenericFinanceView(
            firm_id=int(firm.firm_id),
            scheduled_payroll=float(getattr(firm, "scheduled_wage_bill", 0.0)),
            target_cash=float(getattr(firm, "target_cash", 0.0)),
            requested_credit=float(getattr(firm, "requested_credit", 0.0)),
            executed_credit=float(getattr(firm, "executed_credit", 0.0)),
            payroll_funding_ratio=float(getattr(firm, "payroll_funding_ratio", 1.0)),
            principal=float(getattr(firm, "loan_balance", 0.0)),
            arrears=float(getattr(firm, "interest_arrears", 0.0)),
        )

    def _sector_labor(self, world, firm_ids):
        firm_id_set = set(firm_ids)
        employee_count = 0
        labor_services = 0.0
        for person in getattr(world, "population", ()):
            if not getattr(person, "alive", False):
                continue
            if getattr(person, "firm_id", None) not in firm_id_set:
                continue
            if getattr(person, "household_id", None) not in getattr(world, "household_dict", {}):
                continue
            employee_count += 1
            labor_services += max(0.0, float(age_productivity(person.age)))
        return employee_count, labor_services

    def refresh(self, world, step=None):
        firms = sorted(
            list(
                world.operating_firms()
                if hasattr(world, "operating_firms")
                else getattr(world, "firms", ())
            ),
            key=lambda firm: int(firm.firm_id),
        )
        self.refresh_firm_registry(firms)
        self.firm_operating_views = {
            int(firm.firm_id): FirmOperatingView(
                firm_id=int(firm.firm_id),
                sector_id=self.firm_sector_id(firm),
                technology_id=self.firm_technology_id(firm),
                inventory_policy_id=self.firm_inventory_policy_id(firm),
                operating_state_by_good=self.operating_state_by_good(firm),
            )
            for firm in firms
        }
        self.finance_views = {
            int(firm.firm_id): self.finance_view(firm)
            for firm in firms
        }
        self.sector_views = {}
        for sector in self.sectors.all():
            sector_firms = [
                firm
                for firm in firms
                if self.firm_sector_id(firm) == sector.sector_id
            ]
            firm_ids = [int(firm.firm_id) for firm in sector_firms]
            employee_count, labor_services = self._sector_labor(world, firm_ids)
            production_value = math.fsum(
                float(
                    getattr(
                        firm,
                        "realized_service_output"
                        if sector.sector_id == GENERIC_SERVICE_SECTOR_ID
                        else "capital_good_production_units"
                        if sector.sector_id == CAPITAL_GOODS_SECTOR_ID
                        else "actual_production",
                        0.0,
                    )
                )
                * float(getattr(firm, "transaction_price", getattr(firm, "price", 0.0)))
                for firm in sector_firms
            )
            sales_value = math.fsum(
                float(getattr(firm, "sales_revenue", 0.0))
                for firm in sector_firms
            )
            self.sector_views[sector.sector_id] = SectorAggregateView(
                sector_id=sector.sector_id,
                output_good_ids=tuple(sector.output_good_ids),
                employee_count=employee_count,
                labor_services=labor_services,
                production_value=production_value,
                sales_value=sales_value,
                cash=math.fsum(float(getattr(firm, "cash", 0.0)) for firm in sector_firms),
                principal=math.fsum(float(getattr(firm, "loan_balance", 0.0)) for firm in sector_firms),
                arrears=math.fsum(float(getattr(firm, "interest_arrears", 0.0)) for firm in sector_firms),
            )

        food_good_id = self.food_sector.output_good_ids[0]
        runtime_supplier_ids = [
            int(firm.firm_id)
            for firm in firms
            if self.firm_sector_id(firm) == FOOD_SECTOR_ID
        ]
        registered_supplier_ids = self.market_registry.suppliers(food_good_id)
        food_firms = [
            firm
            for firm in firms
            if self.firm_sector_id(firm) == FOOD_SECTOR_ID
        ]
        food_view = self.sector_views[FOOD_SECTOR_ID]
        employee_check = sum(
            1
            for person in getattr(world, "population", ())
            if getattr(person, "alive", False)
            and getattr(person, "firm_id", None) in set(runtime_supplier_ids)
            and getattr(person, "household_id", None)
            in getattr(world, "household_dict", {})
        )
        food_production = math.fsum(
            float(getattr(firm, "actual_production", 0.0))
            for firm in food_firms
        )
        food_sales = math.fsum(
            float(getattr(firm, "sales_units", 0.0)) for firm in food_firms
        )
        food_inventory = math.fsum(
            float(getattr(firm, "inventory_units", 0.0)) for firm in food_firms
        )
        good_spec = self.goods_catalog.get(food_good_id)
        good_spec_gap = float(
            not (
                good_spec.good_id == food_good_id
                and good_spec.storable
                and good_spec.physical
                and not good_spec.service
                and good_spec.perishability == good_spec.perish_rate
            )
        )
        employee_labor_from_firm_slices = math.fsum(
            max(0.0, float(age_productivity(person.age)))
            for firm in food_firms
            for person_id in getattr(firm, "employee_ids", ())
            for person in [getattr(world, "person_dict", {}).get(person_id)]
            if person is not None
            and getattr(person, "alive", False)
            and getattr(person, "firm_id", None) == firm.firm_id
            and getattr(person, "household_id", None)
            in getattr(world, "household_dict", {})
        )
        self.latest_reconciliation = {
            "food_goodspec_view_gap": good_spec_gap,
            "food_sector_mapping_gap": 0.0,
            "market_registry_supplier_mismatch": 0.0 if registered_supplier_ids == runtime_supplier_ids else 1.0,
            "firm_sector_identity_mismatch": float(
                sum(self.firm_sector_id(firm) != FOOD_SECTOR_ID for firm in firms)
            ),
            "firm_technology_identity_mismatch": float(
                sum(self.firm_technology_id(firm) != FOOD_TECHNOLOGY_ID for firm in firms)
            ),
            "food_production_by_good_gap": food_production - math.fsum(
                view.operating_state_by_good[food_good_id]["production"]
                for view in self.firm_operating_views.values()
                if food_good_id in view.operating_state_by_good
            ),
            "food_sales_by_good_gap": food_sales - math.fsum(
                view.operating_state_by_good[food_good_id]["sales"]
                for view in self.firm_operating_views.values()
                if food_good_id in view.operating_state_by_good
            ),
            "food_inventory_by_good_gap": food_inventory - math.fsum(
                view.operating_state_by_good[food_good_id]["inventory"]
                for view in self.firm_operating_views.values()
                if food_good_id in view.operating_state_by_good
            ),
            "sector_employment_aggregation_gap": float(
                food_view.employee_count - employee_check
            ),
            "sector_labor_service_aggregation_gap": (
                food_view.labor_services - employee_labor_from_firm_slices
            ),
            "sector_cash_aggregation_gap": food_view.cash - math.fsum(
                float(getattr(firm, "cash", 0.0)) for firm in food_firms
            ),
            "sector_principal_aggregation_gap": food_view.principal - math.fsum(
                float(getattr(firm, "loan_balance", 0.0)) for firm in food_firms
            ),
            "sector_arrears_aggregation_gap": food_view.arrears - math.fsum(
                float(getattr(firm, "interest_arrears", 0.0)) for firm in food_firms
            ),
        }
        if self.collect_validation:
            row = {
                "global_step": int(
                    step if step is not None else getattr(world, "current_step_index", 0)
                )
            }
            row.update(self.latest_reconciliation)
            self.validation_history.append(row)
        return self.latest_reconciliation


def default_firm_identity():
    return {
        "sector_id": FOOD_SECTOR_ID,
        "technology_id": FOOD_TECHNOLOGY_ID,
        "inventory_policy_id": FOOD_INVENTORY_POLICY_ID,
        "output_good_id": "basic_consumption_good",
        "active": True,
    }
