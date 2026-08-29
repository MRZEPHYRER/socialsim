"""Default-off Government/PublicBudget cash account and explicit tax settlement.

The account is deliberately separate from the SocialInsuranceFund. Tax
collection is cash constrained and never creates a tax debt in Step17.S.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import math

EPSILON = 1e-12


def _number(value, default=0.0):
    try:
        value = float(value)
        return value if math.isfinite(value) else default
    except (TypeError, ValueError):
        return default


@dataclass
class GovernmentPublicBudget:
    account_id: str = "government_public_budget"
    cash: float = 0.0
    tax_revenue_inflow_this_step: float = 0.0
    other_fiscal_inflow_this_step: float = 0.0
    public_expenditure_outflow_this_step: float = 0.0
    scheduled_tax_this_step: float = 0.0
    actual_tax_this_step: float = 0.0
    tax_shortfall_this_step: float = 0.0
    personal_tax_scheduled_this_step: float = 0.0
    personal_tax_actual_this_step: float = 0.0
    personal_tax_shortfall_this_step: float = 0.0
    corporate_tax_scheduled_this_step: float = 0.0
    corporate_tax_actual_this_step: float = 0.0
    corporate_tax_shortfall_this_step: float = 0.0
    scheduled_public_pension_transfer_this_step: float = 0.0
    actual_public_pension_transfer_this_step: float = 0.0
    public_pension_transfer_shortfall_this_step: float = 0.0
    tax_records: list[dict] = field(default_factory=list)
    personal_tax_records: list[dict] = field(default_factory=list)
    weekly_history: list[dict] = field(default_factory=list)

    def reset_step(self):
        self.tax_revenue_inflow_this_step = 0.0
        self.other_fiscal_inflow_this_step = 0.0
        self.public_expenditure_outflow_this_step = 0.0
        self.scheduled_tax_this_step = 0.0
        self.actual_tax_this_step = 0.0
        self.tax_shortfall_this_step = 0.0
        self.personal_tax_scheduled_this_step = 0.0
        self.personal_tax_actual_this_step = 0.0
        self.personal_tax_shortfall_this_step = 0.0
        self.corporate_tax_scheduled_this_step = 0.0
        self.corporate_tax_actual_this_step = 0.0
        self.corporate_tax_shortfall_this_step = 0.0
        self.scheduled_public_pension_transfer_this_step = 0.0
        self.actual_public_pension_transfer_this_step = 0.0
        self.public_pension_transfer_shortfall_this_step = 0.0
        self.tax_records = []
        self.personal_tax_records = []


def taxable_base(firm, tax_base):
    base = str(tax_base or "").strip().upper()
    if base == "FIRM_SALES_TAX":
        return max(0.0, _number(getattr(firm, "sales_revenue", getattr(firm, "sales", 0.0))))
    if base == "FIRM_POSITIVE_OPERATING_PROFIT_TAX":
        pre_tax = getattr(firm, "pre_tax_operating_profit", None)
        if pre_tax is None:
            pre_tax = getattr(firm, "operating_profit", getattr(firm, "profit", 0.0))
        return max(0.0, _number(pre_tax))
    if base == "FIRM_POSITIVE_TAXABLE_PROFIT":
        pre_tax = getattr(firm, "pre_tax_operating_profit", None)
        if pre_tax is None:
            pre_tax = getattr(firm, "operating_profit", getattr(firm, "profit", 0.0))
        employer = getattr(firm, "actual_employer_contribution_this_step", 0.0)
        return max(0.0, _number(pre_tax) - _number(employer))
    if base == "FIRM_PAYROLL_TAX":
        return max(0.0, _number(getattr(firm, "executed_wage_bill", getattr(firm, "wage_payment", 0.0))))
    return 0.0


def _append_budget_history(budget, step, base_name, rate):
    budget.weekly_history.append({
        "global_step": step,
        "tax_base": base_name,
        "tax_rate": rate,
        "personal_tax": budget.personal_tax_actual_this_step,
        "corporate_profit_tax": budget.corporate_tax_actual_this_step,
        "opening_government_cash": max(0.0, _number(budget.cash)) - budget.actual_tax_this_step,
        "scheduled_tax": budget.scheduled_tax_this_step,
        "actual_tax": budget.actual_tax_this_step,
        "tax_shortfall": budget.tax_shortfall_this_step,
        "other_fiscal_inflow": budget.other_fiscal_inflow_this_step,
        "public_expenditure": budget.public_expenditure_outflow_this_step,
        "closing_government_cash": max(0.0, _number(budget.cash)),
        "government_stock_flow_gap": 0.0,
        "pension_public_transfer": 0.0,
    })


def collect_firm_tax(world, step, tax_base, tax_rate, reset=True):
    """Collect one explicit Firm tax after operating results exist."""
    budget = world.public_budget
    if reset:
        budget.reset_step()
    base_name = str(tax_base or "").strip().upper()
    rate = max(0.0, min(1.0, _number(tax_rate)))
    enabled = bool(getattr(world, "public_revenue_tax_enabled", False))
    firms = list(getattr(world, "operating_firms", lambda: [])())
    firm_scheduled = firm_actual = 0.0
    for firm in firms:
        if hasattr(firm, "profit_before_dividend"):
            firm.pre_tax_operating_profit = _number(getattr(firm, "profit_before_dividend", 0.0))
        base = taxable_base(firm, base_name) if enabled else 0.0
        scheduled = rate * base if enabled else 0.0
        opening_cash = max(0.0, _number(getattr(firm, "cash", 0.0)))
        actual = min(scheduled, opening_cash)
        if actual > EPSILON:
            world.ledger.transfer_attrs(
                payer_obj=firm, payer_attr="cash",
                payer_name=f"firm.{getattr(firm, 'firm_id', '')}.cash",
                receiver_obj=budget, receiver_attr="cash",
                receiver_name=budget.account_id, amount=actual,
                reason="government_public_tax",
            )
        firm.public_tax_base_this_step = base
        firm.scheduled_public_tax_this_step = scheduled
        firm.actual_public_tax_this_step = actual
        firm.public_tax_shortfall_this_step = max(0.0, scheduled - actual)
        firm.corporate_profit_tax_this_step = actual if base_name == "FIRM_POSITIVE_TAXABLE_PROFIT" else 0.0
        firm.cash_end = _number(getattr(firm, "cash", 0.0))
        firm.other_cash_outflow = max(0.0, _number(getattr(firm, "other_cash_outflow", 0.0))) + actual
        if base_name == "FIRM_POSITIVE_TAXABLE_PROFIT":
            firm_scheduled += scheduled
            firm_actual += actual
        budget.scheduled_tax_this_step += scheduled
        budget.actual_tax_this_step += actual
        budget.tax_records.append({
            "global_step": step, "firm_id": getattr(firm, "firm_id", ""),
            "sector_id": getattr(firm, "sector_id", "unknown"),
            "tax_base": base_name, "taxable_base": base, "tax_rate": rate,
            "scheduled_tax": scheduled, "actual_tax": actual,
            "tax_shortfall": max(0.0, scheduled - actual),
            "opening_firm_cash": opening_cash,
            "closing_firm_cash": max(0.0, _number(getattr(firm, "cash", 0.0))),
        })
    budget.corporate_tax_scheduled_this_step = firm_scheduled
    budget.corporate_tax_actual_this_step = firm_actual
    budget.corporate_tax_shortfall_this_step = max(0.0, firm_scheduled - firm_actual)
    budget.tax_revenue_inflow_this_step = budget.actual_tax_this_step
    budget.tax_shortfall_this_step = max(0.0, budget.scheduled_tax_this_step - budget.actual_tax_this_step)
    _append_budget_history(budget, step, base_name, rate)
    return budget