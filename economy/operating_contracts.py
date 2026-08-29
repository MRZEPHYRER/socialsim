"""Passive operating contracts for the current Food benchmark.

The existing FirmSystem and World methods remain the economic executors.  The
objects in this module are deterministic, reconstructable views of outcomes
that have already been planned or settled; they never draw RNG or write back
to model agents.
"""

from dataclasses import asdict, dataclass, field

from economy.multisector import (
    FOOD_SECTOR_ID,
    FOOD_TECHNOLOGY_ID,
    GENERIC_SERVICE_GOOD_ID,
    GENERIC_SERVICE_SECTOR_ID,
    LaborOnlyServiceTechnology,
    ServiceFlowResult,
)


BENCHMARK_SECTOR_ID = FOOD_SECTOR_ID
BENCHMARK_TECHNOLOGY_ID = FOOD_TECHNOLOGY_ID


@dataclass
class FirmOperatingSnapshot:
    firm_id: int
    storage_index: int
    sector_id: str
    technology_id: str
    employee_ids: tuple
    employee_count: int
    labor_service_units: float
    opening_inventory_by_good: dict
    expected_demand_by_good: dict
    production_plan_by_good: dict
    technical_capacity_by_good: dict
    scheduled_capacity_by_good: dict
    scheduled_payroll: float
    payroll_funding_ratio: float
    price_by_good: dict
    cash: float
    principal: float
    arrears: float
    credit_limit: float
    credit_headroom: float
    distress_state: str
    active_contract_default: bool


@dataclass
class PassiveServiceOperatingSnapshot:
    """Service-shaped operating snapshot with no Food inventory semantics."""

    firm_id: int
    sector_id: str = GENERIC_SERVICE_SECTOR_ID
    technology_id: str = "passive_service_v1"
    output_good_id: str = GENERIC_SERVICE_GOOD_ID
    employee_count: int = 0
    labor_services: float = 0.0
    desired_output: float = 0.0
    technical_capacity: float = 0.0
    funded_capacity: float = 0.0
    realized_output: float = 0.0
    price: object = None
    cash: float = 0.0
    scheduled_payroll: float = 0.0
    requested_credit: float = 0.0
    executed_credit: float = 0.0
    credit_limit: object = None
    credit_headroom: object = None
    payroll_funding_ratio: float = 1.0
    principal: float = 0.0
    interest_arrears: float = 0.0
    inventory_by_good: dict = field(default_factory=dict)
    inventory_book_value: object = None


@dataclass
class PassiveServiceOperatingContract:
    snapshot: PassiveServiceOperatingSnapshot
    flow: ServiceFlowResult
    accounting: object = None

    def to_dict(self):
        return asdict(self)


def build_passive_service_contract(
    firm_id,
    labor_services,
    desired_output,
    funded_ratio,
    current_demand,
    productivity_per_labor=None,
    **finance,
):
    """Build a deterministic Service contract for tests and schema checks."""
    technology = LaborOnlyServiceTechnology(productivity_per_labor)
    flow = technology.flow(
        labor_services=labor_services,
        desired_output=desired_output,
        funded_ratio=funded_ratio,
        current_demand=current_demand,
    )
    snapshot = PassiveServiceOperatingSnapshot(
        firm_id=int(firm_id),
        employee_count=int(finance.get("employee_count", 0)),
        labor_services=float(labor_services),
        desired_output=flow.desired_output,
        technical_capacity=flow.technical_capacity,
        funded_capacity=flow.funded_capacity,
        realized_output=flow.realized_output,
        price=finance.get("price"),
        cash=float(finance.get("cash", 0.0)),
        scheduled_payroll=float(finance.get("scheduled_payroll", 0.0)),
        requested_credit=float(finance.get("requested_credit", 0.0)),
        executed_credit=float(finance.get("executed_credit", 0.0)),
        credit_limit=finance.get("credit_limit"),
        credit_headroom=finance.get("credit_headroom"),
        payroll_funding_ratio=float(finance.get("payroll_funding_ratio", 1.0)),
        principal=float(finance.get("principal", 0.0)),
        interest_arrears=float(finance.get("interest_arrears", 0.0)),
    )
    return PassiveServiceOperatingContract(snapshot=snapshot, flow=flow)


@dataclass
class OperatingIntent:
    expected_demand_by_good: dict
    target_inventory_by_good: dict
    desired_inventory_adjustment_by_good: dict
    inventory_adjusted_desired_output_by_good: dict
    desired_output_by_good: dict
    desired_labor_services: float
    desired_worker_count_approx: object
    desired_capacity_by_good: dict


@dataclass
class OperatingConstraints:
    assigned_labor_services: float
    technical_capacity_by_good: dict
    funded_capacity_by_good: dict
    funding_ratio: float
    capital_services: object = None
    intermediate_inputs: dict = field(default_factory=dict)


@dataclass
class OperatingResult:
    feasible_output_by_good: dict
    funded_output_by_good: dict
    realized_output_by_good: dict
    actual_payroll: float
    actual_production_by_good: dict
    sales_by_good: dict
    unmet_demand_by_good: dict
    inventory_change_by_good: dict


@dataclass
class FundingDecisionView:
    requested_credit: float
    executed_credit: float
    denied_credit: float
    credit_headroom: float
    scheduled_payroll: float
    payroll_funding_ratio: float
    liquidity_shortfall: float
    current_interest_due: float
    ending_principal: float
    ending_arrears: float


@dataclass
class FirmRecoveryDiagnostics:
    cfo_before_interest: float
    current_interest_due: float
    current_interest_paid: float
    unpaid_current_interest: float
    cash_to_target_gap: float
    principal_change: float
    arrears_change: float
    denied_credit: float
    payroll_shortfall: float
    inventory_book_investment: float
    inventory_loss: float


@dataclass
class StageAOperatingContractBundle:
    snapshot: FirmOperatingSnapshot
    intent: OperatingIntent
    constraints: OperatingConstraints
    result: OperatingResult
    funding: FundingDecisionView
    recovery: FirmRecoveryDiagnostics
    reconciliation: dict

    def to_dict(self):
        return asdict(self)

    def diagnostics(self):
        """Return scalar fields suitable for the existing Firm CSV."""
        good_id = next(iter(self.result.realized_output_by_good))
        result = self.result
        intent = self.intent
        snapshot = self.snapshot
        constraints = self.constraints
        funding = self.funding
        recovery = self.recovery
        row = {
            "stageA_sector_id": snapshot.sector_id,
            "stageA_technology_id": snapshot.technology_id,
            "stageA_good_id": good_id,
            "stageA_storage_index": snapshot.storage_index,
            "stageA_current_labor_services": snapshot.labor_service_units,
            "stageA_desired_labor_services_shadow": intent.desired_labor_services,
            "stageA_labor_service_gap": (
                intent.desired_labor_services - snapshot.labor_service_units
            ),
            "stageA_desired_worker_count_shadow_approx": (
                intent.desired_worker_count_approx
                if intent.desired_worker_count_approx is not None else ""
            ),
            "stageA_desired_output_shadow": intent.desired_output_by_good[good_id],
            "stageA_inventory_adjusted_desired_output": (
                intent.inventory_adjusted_desired_output_by_good[good_id]
            ),
            "stageA_feasible_output_shadow": result.feasible_output_by_good[good_id],
            "stageA_funded_output_shadow": result.funded_output_by_good[good_id],
            "stageA_realized_output_shadow": result.realized_output_by_good[good_id],
            "stageA_technical_capacity_shadow": (
                constraints.technical_capacity_by_good[good_id]
            ),
            "stageA_funded_capacity_view": (
                constraints.funded_capacity_by_good[good_id]
            ),
            "stageA_opening_inventory_units": (
                snapshot.opening_inventory_by_good[good_id]
            ),
            "stageA_inventory_change_units": result.inventory_change_by_good[good_id],
            "stageA_requested_credit_view": funding.requested_credit,
            "stageA_executed_credit_view": funding.executed_credit,
            "stageA_denied_credit_view": funding.denied_credit,
            "stageA_scheduled_payroll_view": funding.scheduled_payroll,
            "stageA_payroll_funding_ratio_view": funding.payroll_funding_ratio,
            "stageA_cfo_before_interest": recovery.cfo_before_interest,
            "stageA_cash_to_target_gap": recovery.cash_to_target_gap,
            "stageA_principal_change": recovery.principal_change,
            "stageA_arrears_change": recovery.arrears_change,
            "stageA_inventory_book_investment": recovery.inventory_book_investment,
            "stageA_inventory_loss": recovery.inventory_loss,
        }
        row.update({f"stageA_{key}": value for key, value in self.reconciliation.items()})
        return row


class BenchmarkDemandForecaster:
    """Exact passive view of the accepted runtime expected-demand state."""

    def __init__(self, good_id):
        self.good_id = good_id

    def forecast(self, firm):
        return {self.good_id: max(0.0, float(firm.expected_demand))}


class BenchmarkFoodInventoryPolicy:
    """Read-only adapter over the benchmark target and GoodSpec decay rule."""

    def __init__(self, good_spec):
        self.good_spec = good_spec

    def target_inventory(self, firm):
        return max(0.0, float(firm.target_inventory_units))

    def inventory_gap(self, firm):
        return float(firm.inventory_gap_units)

    def expected_decay(self, pre_decay_units):
        return max(0.0, float(pre_decay_units)) * self.good_spec.perishability


class LaborOnlyFoodTechnology:
    """Passive representation of the existing labor-only capacity relation."""

    technology_id = BENCHMARK_TECHNOLOGY_ID

    def __init__(self, productivity_per_labor):
        self.productivity_per_labor = float(productivity_per_labor)

    def capacity(self, labor_service_units, production_scale):
        return (
            max(0.0, float(labor_service_units))
            * self.productivity_per_labor
            * max(0.0, float(production_scale))
        )


class StageAOperatingContractAdapter:
    """Current-week shadow contracts; no histories and no behavioral writes."""

    VERSION = 1

    def __init__(self, good_spec, productivity_per_labor):
        self.good_spec = good_spec
        self.good_id = good_spec.good_id
        self.demand_forecaster = BenchmarkDemandForecaster(self.good_id)
        self.inventory_policy = BenchmarkFoodInventoryPolicy(good_spec)
        self.technology = LaborOnlyFoodTechnology(productivity_per_labor)
        self.step = None
        self.credit_context = {}
        self.plan_context = {}
        self.current_bundles = {}

    def begin_step(self, step):
        self.step = int(step)
        self.credit_context = {}
        self.plan_context = {}
        self.current_bundles = {}

    def capture_credit_context(
        self,
        firm,
        storage_index,
        labor_service_units,
        production_scale,
    ):
        self.credit_context[firm.firm_id] = {
            "storage_index": int(storage_index),
            "labor_service_units": float(labor_service_units),
            "production_scale": float(production_scale),
            "scheduled_capacity": float(firm.scheduled_productive_capacity),
            "funded_capacity": float(firm.funded_productive_capacity),
        }

    def capture_plan_context(
        self,
        firm,
        opening_inventory,
        planning_expected_demand,
        raw_desired_output,
        desired_output_shadow,
        current_plan,
    ):
        self.plan_context[firm.firm_id] = {
            "opening_inventory": float(opening_inventory),
            "planning_expected_demand": float(planning_expected_demand),
            "raw_desired_output": float(raw_desired_output),
            "desired_output_shadow": float(desired_output_shadow),
            "current_plan": float(current_plan),
        }

    @staticmethod
    def _accounting_row(accounting, firm_id):
        state = getattr(accounting, "firms", {}).get(firm_id)
        rows = getattr(state, "rows", []) if state is not None else []
        return rows[-1] if rows else {}

    def build_for_firm(self, world, firm, storage_index):
        good_id = self.good_id
        credit = self.credit_context.get(firm.firm_id, {})
        plan = self.plan_context.get(firm.firm_id, {})
        accounting = self._accounting_row(world.accounting, firm.firm_id)

        labor_services = float(credit.get("labor_service_units", 0.0))
        production_scale = float(
            credit.get("production_scale", world.firm_system.production_scale)
        )
        capacity_shadow = self.technology.capacity(
            labor_services,
            production_scale,
        )
        scheduled_capacity = float(
            credit.get(
                "scheduled_capacity",
                getattr(firm, "scheduled_productive_capacity", 0.0),
            )
        )
        funded_capacity = float(
            credit.get(
                "funded_capacity",
                getattr(firm, "funded_productive_capacity", 0.0),
            )
        )
        opening_inventory = float(
            plan.get("opening_inventory", getattr(firm, "inventory_units", 0.0))
        )
        planning_forecast = float(
            plan.get("planning_expected_demand", getattr(firm, "expected_demand", 0.0))
        )
        raw_desired = float(
            plan.get("raw_desired_output", getattr(firm, "desired_production", 0.0))
        )
        current_plan = float(
            plan.get("current_plan", getattr(firm, "production_plan", 0.0))
        )
        desired_output = float(
            plan.get("desired_output_shadow", current_plan)
        )
        forecast_view = self.demand_forecaster.forecast(firm)
        target_inventory = self.inventory_policy.target_inventory(firm)
        inventory_gap = self.inventory_policy.inventory_gap(firm)
        feasible_output = min(max(0.0, desired_output), max(0.0, scheduled_capacity))
        funded_output = min(feasible_output, max(0.0, funded_capacity))
        realized_output = max(0.0, float(firm.actual_production))

        output_per_labor = self.technology.capacity(1.0, production_scale)
        desired_labor = (
            desired_output / output_per_labor if output_per_labor > 0.0 else 0.0
        )
        desired_worker_count = (
            len(firm.employee_ids) * desired_labor / labor_services
            if labor_services > 0.0 else None
        )
        ending_inventory = max(0.0, float(firm.inventory_units))
        actual_decay = max(0.0, float(getattr(firm, "spoilage_units", 0.0)))
        predicted_decay = self.inventory_policy.expected_decay(
            ending_inventory + actual_decay
        )
        current_expected_demand = forecast_view[good_id]

        funding = FundingDecisionView(
            requested_credit=float(getattr(firm, "requested_credit", 0.0)),
            executed_credit=float(getattr(firm, "executed_credit", 0.0)),
            denied_credit=float(getattr(firm, "denied_credit", 0.0)),
            credit_headroom=float(getattr(firm, "credit_headroom", 0.0)),
            scheduled_payroll=float(getattr(firm, "scheduled_wage_bill", 0.0)),
            payroll_funding_ratio=float(getattr(firm, "payroll_funding_ratio", 1.0)),
            liquidity_shortfall=float(
                getattr(firm, "target_liquidity_shortfall", 0.0)
            ),
            current_interest_due=float(getattr(firm, "current_interest_due", 0.0)),
            ending_principal=float(getattr(firm, "loan_balance", 0.0)),
            ending_arrears=float(getattr(firm, "interest_arrears", 0.0)),
        )
        inventory_book_investment = (
            float(accounting.get("capitalized_production_cost", 0.0))
            - float(accounting.get("inventory_cost_or_cogs", 0.0))
            - float(accounting.get("spoilage_or_inventory_loss", 0.0))
        )
        recovery = FirmRecoveryDiagnostics(
            cfo_before_interest=(
                float(accounting.get("cfo", 0.0))
                + float(accounting.get("interest_paid", 0.0))
            ),
            current_interest_due=float(getattr(firm, "current_interest_due", 0.0)),
            current_interest_paid=float(
                getattr(firm, "interest_paid_to_current_due", 0.0)
            ),
            unpaid_current_interest=float(
                getattr(firm, "current_interest_unpaid", 0.0)
            ),
            cash_to_target_gap=float(firm.cash) - float(firm.target_cash),
            principal_change=(
                float(getattr(firm, "loan_balance", 0.0))
                - float(getattr(firm, "opening_principal", 0.0))
            ),
            arrears_change=(
                float(getattr(firm, "interest_arrears", 0.0))
                - float(getattr(firm, "opening_interest_arrears", 0.0))
            ),
            denied_credit=float(getattr(firm, "denied_credit", 0.0)),
            payroll_shortfall=float(getattr(firm, "payroll_cash_shortfall", 0.0)),
            inventory_book_investment=inventory_book_investment,
            inventory_loss=float(
                accounting.get("spoilage_or_inventory_loss", 0.0)
            ),
        )
        snapshot = FirmOperatingSnapshot(
            firm_id=firm.firm_id,
            storage_index=int(storage_index),
            sector_id=getattr(firm, "sector_id", BENCHMARK_SECTOR_ID),
            technology_id=getattr(
                firm,
                "technology_id",
                self.technology.technology_id,
            ),
            employee_ids=tuple(firm.employee_ids),
            employee_count=len(firm.employee_ids),
            labor_service_units=labor_services,
            opening_inventory_by_good={good_id: opening_inventory},
            expected_demand_by_good={good_id: current_expected_demand},
            production_plan_by_good={good_id: current_plan},
            technical_capacity_by_good={good_id: capacity_shadow},
            scheduled_capacity_by_good={good_id: scheduled_capacity},
            scheduled_payroll=funding.scheduled_payroll,
            payroll_funding_ratio=funding.payroll_funding_ratio,
            price_by_good={good_id: float(firm.price)},
            cash=float(firm.cash),
            principal=float(getattr(firm, "loan_balance", 0.0)),
            arrears=float(getattr(firm, "interest_arrears", 0.0)),
            credit_limit=float(getattr(firm, "credit_limit", 0.0)),
            credit_headroom=float(getattr(firm, "credit_headroom", 0.0)),
            distress_state=str(getattr(firm, "default_distress_state", "D0")),
            active_contract_default=bool(
                getattr(firm, "active_contract_default", False)
            ),
        )
        intent = OperatingIntent(
            expected_demand_by_good={good_id: planning_forecast},
            target_inventory_by_good={good_id: target_inventory},
            desired_inventory_adjustment_by_good={good_id: inventory_gap},
            inventory_adjusted_desired_output_by_good={good_id: raw_desired},
            desired_output_by_good={good_id: desired_output},
            desired_labor_services=desired_labor,
            desired_worker_count_approx=desired_worker_count,
            desired_capacity_by_good={good_id: desired_output},
        )
        constraints = OperatingConstraints(
            assigned_labor_services=labor_services,
            technical_capacity_by_good={good_id: capacity_shadow},
            funded_capacity_by_good={good_id: funded_capacity},
            funding_ratio=funding.payroll_funding_ratio,
        )
        result = OperatingResult(
            feasible_output_by_good={good_id: feasible_output},
            funded_output_by_good={good_id: funded_output},
            realized_output_by_good={good_id: realized_output},
            actual_payroll=float(getattr(firm, "wage_payment", 0.0)),
            actual_production_by_good={good_id: realized_output},
            sales_by_good={good_id: float(getattr(firm, "sales_units", 0.0))},
            unmet_demand_by_good={
                good_id: float(getattr(firm, "unmet_demand", 0.0))
            },
            inventory_change_by_good={good_id: ending_inventory - opening_inventory},
        )
        funding_view_gap = max(
            abs(funding.requested_credit - float(getattr(firm, "requested_credit", 0.0))),
            abs(funding.executed_credit - float(getattr(firm, "executed_credit", 0.0))),
            abs(funding.denied_credit - float(getattr(firm, "denied_credit", 0.0))),
            abs(funding.scheduled_payroll - float(getattr(firm, "scheduled_wage_bill", 0.0))),
            abs(funding.payroll_funding_ratio - float(getattr(firm, "payroll_funding_ratio", 1.0))),
        )
        reconciliation = {
            "shadow_expected_demand_gap": (
                current_expected_demand - float(firm.expected_demand)
            ),
            "shadow_capacity_gap": capacity_shadow - scheduled_capacity,
            "desired_current_plan_semantic_gap": raw_desired - current_plan,
            "operating_intent_runtime_plan_gap": desired_output - current_plan,
            "feasible_output_gap": (
                feasible_output
                - min(max(0.0, desired_output), max(0.0, scheduled_capacity))
            ),
            "funded_output_gap": (
                funded_output - min(feasible_output, funded_capacity)
            ),
            "realized_output_gap": realized_output - float(firm.actual_production),
            "scheduled_payroll_view_gap": (
                funding.scheduled_payroll
                - float(getattr(firm, "scheduled_wage_bill", 0.0))
            ),
            "funding_decision_view_gap": funding_view_gap,
            "inventory_target_view_gap": (
                target_inventory - float(firm.target_inventory_units)
            ),
            "inventory_decay_view_gap": predicted_decay - actual_decay,
            "realized_over_funded_violation": max(
                0.0, realized_output - funded_output
            ),
            "funded_over_feasible_violation": max(
                0.0, funded_output - feasible_output
            ),
        }
        return StageAOperatingContractBundle(
            snapshot=snapshot,
            intent=intent,
            constraints=constraints,
            result=result,
            funding=funding,
            recovery=recovery,
            reconciliation=reconciliation,
        )

    def build_current_bundles(self, world):
        self.current_bundles = {}
        for storage_index, firm in enumerate(world.firms):
            keyed_firm = world.get_firm_by_id(firm.firm_id)
            if keyed_firm is not firm:
                raise ValueError(
                    "Keyed Firm lookup does not match current storage identity"
                )
            self.current_bundles[firm.firm_id] = self.build_for_firm(
                world,
                firm,
                storage_index,
            )
        return self.current_bundles
