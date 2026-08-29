"""Isolated active-capable runtime for a generic non-inventory Firm.

This is a composed runtime fixture, not a ServiceFirm subclass and not part of
the canonical World execution path. It reuses the accepted Step13 interest and
payroll-credit semantics while keeping all inputs explicit for tests.
"""

from dataclasses import dataclass

from central_bank import config as central_bank_config
from economy.interest import settle_interest
from economy.multisector import (
    GENERIC_SERVICE_GOOD_ID,
    GENERIC_SERVICE_SECTOR_ID,
    NO_INVENTORY_POLICY_ID,
    LaborOnlyServiceTechnology,
)
from economy.service_accounting import (
    PassiveServiceAccountingAdapter,
    resolve_service_accounting_adapter,
)


def _positive(value):
    try:
        return max(0.0, float(value))
    except (TypeError, ValueError):
        return 0.0


@dataclass(frozen=True)
class ServiceRuntimePeriod:
    firm_id: int
    sector_id: str
    output_good_id: str
    price: float
    cash_start: float
    cash_end: float
    scheduled_payroll: float
    executed_payroll: float
    requested_credit: float
    executed_credit: float
    denied_credit: float
    credit_limit: float
    credit_headroom: float
    payroll_funding_ratio: float
    labor_services: float
    technical_capacity: float
    feasible_capacity: float
    funded_capacity: float
    desired_output: float
    current_demand: float
    realized_output: float
    unused_capacity: float
    unmet_demand: float
    sales_revenue: float
    operating_profit: float
    interest_due: float
    interest_paid: float
    interest_arrears: float
    principal_start: float
    principal_repaid: float
    principal_end: float
    cash_flow_gap: float
    balance_sheet_gap: float
    inventory_units: float
    inventory_book_value: float
    spoilage_units: float


class GenericNonInventoryFirmRuntime:
    """Run one explicit current-period Service operating/finance fixture."""

    VERSION = 1

    def __init__(
        self,
        firm,
        productivity_per_labor,
        accounting_adapter=None,
        price=1.0,
    ):
        self.firm = firm
        self.technology = LaborOnlyServiceTechnology(productivity_per_labor)
        self.accounting_adapter = (
            accounting_adapter
            or resolve_service_accounting_adapter(
                getattr(firm, "sector_id", None),
                getattr(firm, "inventory_policy_id", None),
            )
            or PassiveServiceAccountingAdapter()
        )
        self.price = float(price)
        self.last_flow = None
        self.last_accounting = None

        if getattr(firm, "sector_id", None) != GENERIC_SERVICE_SECTOR_ID:
            raise ValueError("fixture Firm must use the generic Service sector")
        if getattr(firm, "inventory_policy_id", None) != NO_INVENTORY_POLICY_ID:
            raise ValueError("Service runtime requires the no-inventory policy")
        if getattr(firm, "output_good_id", None) != GENERIC_SERVICE_GOOD_ID:
            raise ValueError("fixture Firm must output the generic Service good")

    def run_period(
        self,
        labor_services,
        scheduled_payroll,
        desired_output,
        current_demand,
        target_cash,
        credit_limit=float("inf"),
        annual_interest_rate=0.0,
        operating_liquidity_floor=0.0,
        principal_repayment_rate=None,
        dividends=0.0,
    ):
        firm = self.firm
        cash_start = _positive(getattr(firm, "cash", 0.0))
        principal_start = _positive(getattr(firm, "loan_balance", 0.0))
        arrears_start = _positive(getattr(firm, "interest_arrears", 0.0))
        scheduled_payroll = _positive(scheduled_payroll)
        target_cash = _positive(target_cash)
        credit_limit = max(0.0, float(credit_limit))
        opening_exposure = principal_start + arrears_start
        credit_headroom = max(0.0, credit_limit - opening_exposure)
        requested_credit = max(0.0, target_cash - cash_start)
        executed_credit = min(requested_credit, credit_headroom)
        denied_credit = max(0.0, requested_credit - executed_credit)
        cash_after_credit = cash_start + executed_credit
        payroll_ratio = (
            1.0
            if scheduled_payroll <= 1e-12
            else min(1.0, max(0.0, cash_after_credit / scheduled_payroll))
        )
        executed_payroll = scheduled_payroll * payroll_ratio
        technical_capacity = self.technology.technical_capacity(labor_services)
        feasible_capacity = min(max(0.0, float(desired_output)), technical_capacity)
        funded_capacity = feasible_capacity * payroll_ratio
        demand = _positive(current_demand)
        realized_output = min(funded_capacity, demand)
        unused_capacity = max(0.0, funded_capacity - realized_output)
        unmet_demand = max(0.0, demand - realized_output)
        sales_revenue = realized_output * self.price

        cash_before_interest = cash_after_credit + sales_revenue - executed_payroll
        interest = settle_interest(
            opening_principal=principal_start,
            opening_arrears=arrears_start,
            cash_before_interest=cash_before_interest,
            operating_liquidity_floor=operating_liquidity_floor,
            annual_rate=annual_interest_rate,
        )
        interest_paid = interest["interest_paid"]
        cash_after_interest = interest["cash_after_interest"]
        repayment_rate = (
            central_bank_config.CENTRAL_BANK_FIRM_LOAN_REPAYMENT_RATE
            if principal_repayment_rate is None
            else max(0.0, float(principal_repayment_rate))
        )
        principal_after_issuance = principal_start + executed_credit
        principal_repaid = min(
            principal_after_issuance,
            principal_after_issuance * repayment_rate,
            max(0.0, cash_after_interest - operating_liquidity_floor),
        )
        principal_end = principal_after_issuance - principal_repaid
        cash_end = cash_after_interest - principal_repaid - _positive(dividends)

        firm.cash = cash_end
        firm.loan_balance = principal_end
        firm.interest_arrears = interest["closing_interest_arrears"]
        firm.sector_id = GENERIC_SERVICE_SECTOR_ID
        firm.output_good_id = GENERIC_SERVICE_GOOD_ID
        firm.technology_id = self.technology.technology_id
        firm.inventory_policy_id = NO_INVENTORY_POLICY_ID
        firm.price = self.price
        firm.labor_services = max(0.0, float(labor_services))
        firm.desired_output = max(0.0, float(desired_output))
        firm.technical_capacity = technical_capacity
        firm.feasible_service_capacity = feasible_capacity
        firm.funded_service_capacity = funded_capacity
        firm.current_service_demand = demand
        firm.realized_service_output = realized_output
        firm.service_sales_units = realized_output
        firm.unused_service_capacity = unused_capacity
        firm.unmet_demand = unmet_demand
        firm.scheduled_wage_bill = scheduled_payroll
        firm.executed_wage_bill = executed_payroll
        firm.sales_revenue = sales_revenue
        firm.operating_profit = 0.0
        firm.service_spoilage_units = 0.0
        firm.inventory_units = 0.0
        firm.inventory_book_value = 0.0

        self.last_accounting = self.accounting_adapter.record(
            service_revenue=sales_revenue,
            executed_labor_expense=executed_payroll,
            interest_paid=interest_paid,
            cash_start=cash_start,
            loan_issued=executed_credit,
            principal_repaid=principal_repaid,
            dividends=dividends,
            principal=principal_end,
            interest_arrears=interest["closing_interest_arrears"],
            cash_end=cash_end,
        )
        firm.operating_profit = self.last_accounting.operating_profit
        firm.net_income = self.last_accounting.net_income
        self.last_flow = ServiceRuntimePeriod(
            firm_id=int(getattr(firm, "firm_id", 0)),
            sector_id=GENERIC_SERVICE_SECTOR_ID,
            output_good_id=GENERIC_SERVICE_GOOD_ID,
            price=self.price,
            cash_start=cash_start,
            cash_end=cash_end,
            scheduled_payroll=scheduled_payroll,
            executed_payroll=executed_payroll,
            requested_credit=requested_credit,
            executed_credit=executed_credit,
            denied_credit=denied_credit,
            credit_limit=credit_limit,
            credit_headroom=credit_headroom,
            payroll_funding_ratio=payroll_ratio,
            labor_services=max(0.0, float(labor_services)),
            technical_capacity=technical_capacity,
            feasible_capacity=feasible_capacity,
            funded_capacity=funded_capacity,
            desired_output=max(0.0, float(desired_output)),
            current_demand=demand,
            realized_output=realized_output,
            unused_capacity=unused_capacity,
            unmet_demand=unmet_demand,
            sales_revenue=sales_revenue,
            operating_profit=self.last_accounting.operating_profit,
            interest_due=interest["current_interest_due"],
            interest_paid=interest_paid,
            interest_arrears=interest["closing_interest_arrears"],
            principal_start=principal_start,
            principal_repaid=principal_repaid,
            principal_end=principal_end,
            cash_flow_gap=self.last_accounting.cash_flow_gap,
            balance_sheet_gap=self.last_accounting.balance_sheet_gap,
            inventory_units=0.0,
            inventory_book_value=0.0,
            spoilage_units=0.0,
        )
        return self.last_flow
