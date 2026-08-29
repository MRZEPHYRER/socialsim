"""Passive accounting semantics for a future non-storable Service Firm."""

from dataclasses import dataclass


@dataclass(frozen=True)
class ServiceAccountingView:
    service_revenue: float
    current_period_labor_expense: float
    other_operating_expense: float
    operating_profit: float
    interest_expense: float
    net_income: float
    sales_collections: float
    wage_payments: float
    interest_paid: float
    other_operating_cashflows: float
    cfo: float
    cfi: float
    cff: float
    loan_issued: float
    principal_repaid: float
    dividends: float
    cash_start: float
    cash_end: float
    cash_flow_gap: float
    cash: float
    principal: float
    interest_arrears: float
    total_assets: float
    total_liabilities: float
    equity: float
    balance_sheet_gap: float
    inventory_book_value: object = None
    inventory_cost_or_cogs: object = None
    inventory_loss: object = None


class PassiveServiceAccountingAdapter:
    """Construct accounting views without touching the live AccountingLayer."""

    VERSION = 1

    def record(
        self,
        service_revenue,
        executed_labor_expense,
        interest_paid=0.0,
        cash_start=0.0,
        sales_collections=None,
        wage_payments=None,
        other_operating_expense=0.0,
        other_operating_cashflows=0.0,
        loan_issued=0.0,
        principal_repaid=0.0,
        dividends=0.0,
        principal=0.0,
        interest_arrears=0.0,
        cash_end=None,
    ):
        revenue = float(service_revenue)
        labor_expense = float(executed_labor_expense)
        interest = float(interest_paid)
        other_expense = float(other_operating_expense)
        collections = revenue if sales_collections is None else float(sales_collections)
        wages = labor_expense if wage_payments is None else float(wage_payments)
        loan_issued = float(loan_issued)
        principal_repaid = float(principal_repaid)
        dividends = float(dividends)
        other_cashflows = float(other_operating_cashflows)
        cfo = collections - wages + other_cashflows - interest
        cfi = 0.0
        cff = loan_issued - principal_repaid - dividends
        opening_cash = float(cash_start)
        computed_cash_end = opening_cash + cfo + cfi + cff
        ending_cash = computed_cash_end if cash_end is None else float(cash_end)
        cash_flow_gap = ending_cash - opening_cash - cfo - cfi - cff
        operating_profit = revenue - labor_expense - other_expense
        net_income = operating_profit - interest
        principal = float(principal)
        interest_arrears = float(interest_arrears)
        total_assets = ending_cash
        total_liabilities = principal + interest_arrears
        equity = total_assets - total_liabilities
        balance_sheet_gap = total_assets - total_liabilities - equity
        return ServiceAccountingView(
            service_revenue=revenue,
            current_period_labor_expense=labor_expense,
            other_operating_expense=other_expense,
            operating_profit=operating_profit,
            interest_expense=interest,
            net_income=net_income,
            sales_collections=collections,
            wage_payments=wages,
            interest_paid=interest,
            other_operating_cashflows=other_cashflows,
            cfo=cfo,
            cfi=cfi,
            cff=cff,
            loan_issued=loan_issued,
            principal_repaid=principal_repaid,
            dividends=dividends,
            cash_start=opening_cash,
            cash_end=ending_cash,
            cash_flow_gap=cash_flow_gap,
            cash=ending_cash,
            principal=principal,
            interest_arrears=interest_arrears,
            total_assets=total_assets,
            total_liabilities=total_liabilities,
            equity=equity,
            balance_sheet_gap=balance_sheet_gap,
        )


def resolve_service_accounting_adapter(
    sector_id=None,
    inventory_policy_id=None,
):
    """Resolve the non-inventory adapter without touching live accounting."""
    if (
        sector_id == "generic_services"
        or inventory_policy_id == "no_inventory_v1"
    ):
        return PassiveServiceAccountingAdapter()
    return None
