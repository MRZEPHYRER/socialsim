"""Passive financial accounting for already executed model events.

This module deliberately has no economic decisions.  It consumes the state
and event totals produced by the simulation and keeps separate book values
from the market-value diagnostics used by the model.
"""

import csv
import math
import os
from dataclasses import dataclass, field
from time_system import SimulationClock


def _positive(value):
    return max(0.0, float(value or 0.0))


@dataclass
class FirmAccounts:
    inventory_book_value: float = 0.0
    previous_equity: float = None
    previous_inventory_book_value: float = None
    previous_capital_book_value: float = None
    previous_customer_advance_liability: float = None
    previous_prepaid_investment_asset: float = None
    rows: list = field(default_factory=list)


class AccountingLayer:
    VERSION = 1

    def __init__(self, world):
        self.world = world
        self.clock = SimulationClock(0)
        self.firms = {}
        self.rows = []
        self.household_rows = []
        self.public_rows = []
        self.central_bank_rows = []
        self.reconciliation_rows = []
        self.household_lifecycle_rows = []
        self.previous_credit_money_liabilities = None

    def _firm_state(self, firm_id, inventory_units, price):
        state = self.firms.get(firm_id)
        if state is None:
            state = FirmAccounts(
                inventory_book_value=_positive(inventory_units) * _positive(price)
            )
            self.firms[firm_id] = state
        return state

    def record_firms(self, step, firms, aggregate=None):
        time_metadata = self.world.time_metadata(step)
        total_loan = 0.0
        total_cash = 0.0

        for firm in firms:
            firm_id = getattr(firm, "firm_id", 0)
            inventory_units = _positive(getattr(firm, "inventory_units", 0.0))
            price = _positive(getattr(firm, "price", 0.0))
            production = _positive(getattr(firm, "production", 0.0))
            sales_units = _positive(getattr(firm, "sales_units", 0.0))
            spoilage_units = _positive(getattr(firm, "spoilage_units", 0.0))
            public_purchase_units = _positive(
                getattr(firm, "public_inventory_purchase_units", 0.0)
            )
            wage = _positive(getattr(firm, "wage_payment", getattr(firm, "wage_bill", 0.0)))
            sales_revenue = _positive(getattr(firm, "sales_revenue", 0.0))
            public_inflow = _positive(getattr(firm, "public_sector_cash_inflow", 0.0))
            recorded_other_inflow = _positive(getattr(firm, "other_cash_inflow", 0.0))
            recorded_other_outflow = _positive(getattr(firm, "other_cash_outflow", 0.0))
            loan_issued = _positive(getattr(firm, "loan_issued", 0.0))
            loan_repaid = _positive(getattr(firm, "loan_repaid", 0.0))
            post_termout_arrears = _positive(
                getattr(firm, "interest_arrears", 0.0)
            )
            legacy_term_claim = _positive(
                getattr(firm, "legacy_arrears_term_claim", 0.0)
            )
            dividends = _positive(getattr(firm, "dividend_payment", 0.0))
            equity_issuance = _positive(
                getattr(firm, "equity_issuance_cash_this_step", 0.0)
            )
            startup_inflow = _positive(
                getattr(firm, "startup_capitalization_inflow_this_step", 0.0)
            )
            startup_outflow = _positive(
                getattr(firm, "startup_capitalization_outflow_this_step", 0.0)
            )
            interest = _positive(getattr(firm, "loan_interest_paid", 0.0))
            investment_expenditure = _positive(
                getattr(firm, "investment_expenditure_this_step", 0.0)
            )
            customer_advance_inflow = _positive(
                getattr(firm, "customer_advance_received_this_step", 0.0)
            )
            customer_advance_delivered = _positive(
                getattr(firm, "customer_advance_delivered_this_step", 0.0)
            )
            prepaid_investment_paid = _positive(
                getattr(firm, "prepaid_capital_investment_paid_this_step", 0.0)
            )
            prepaid_investment_capitalized = _positive(
                getattr(
                    firm,
                    "prepaid_capital_investment_capitalized_this_step",
                    0.0,
                )
            )
            prepaid_investment_asset = _positive(
                getattr(firm, "prepaid_capital_investment_asset", 0.0)
            )
            sales_collections = _positive(
                getattr(firm, "sales_collections_this_step", sales_revenue)
                if (
                    getattr(firm, "sector_id", "") == "capital_goods"
                    and getattr(self.world, "capital_good_customer_advance_enabled", False)
                )
                else sales_revenue
            )
            state = self._firm_state(firm_id, inventory_units, price)
            capital_stock = getattr(firm, "capital_stock", None)
            capital_book_value = _positive(
                getattr(capital_stock, "total_remaining_book_value", 0.0)
            )
            capital_depreciation = _positive(
                getattr(firm, "capital_depreciation_expense_this_step", 0.0)
            )
            capital_opening_book = _positive(
                getattr(
                    firm,
                    "capital_book_value_opening_this_step",
                    state.previous_capital_book_value
                    if state.previous_capital_book_value is not None
                    else 0.0,
                )
            )
            capital_acquisitions = _positive(
                getattr(
                    firm,
                    "capital_asset_acquisitions_this_step",
                    getattr(firm, "investment_assets_acquired_this_step", 0.0),
                )
            )
            capital_disposals = _positive(
                getattr(firm, "capital_asset_disposals_this_step", 0.0)
            )
            capital_inventory = getattr(firm, "capital_good_inventory", None)
            is_capital_good_firm = (
                getattr(firm, "sector_id", "") == "capital_goods"
                and capital_inventory is not None
            )
            # Equity subscriptions are financing cash flow, even when the
            # legacy cash bridge recorded them in its residual other-inflow.
            other_inflow = max(
                0.0,
                recorded_other_inflow
                - public_inflow
                - equity_issuance
                - customer_advance_inflow,
            )
            other_outflow = max(
                0.0,
                recorded_other_outflow
                - dividends
                - interest
                - prepaid_investment_paid,
            )
            cash_start = float(
                getattr(
                    firm,
                    "accounting_cash_start_this_step",
                    getattr(firm, "cash_start", getattr(firm, "cash", 0.0)),
                )
            )
            cash_end = float(getattr(firm, "cash_end", getattr(firm, "cash", 0.0)))

            first_record = state.previous_inventory_book_value is None
            unit_production_cost = wage / production if production > 1e-12 else 0.0
            production_cost = production * unit_production_cost
            if is_capital_good_firm:
                cogs = _positive(getattr(capital_inventory, "last_cogs", 0.0))
                spoilage_loss = 0.0
                ending_book = _positive(getattr(capital_inventory, "book_value", 0.0))
                opening_book = (
                    state.inventory_book_value
                    if not first_record
                    else max(0.0, ending_book - production_cost + cogs)
                )
                if first_record:
                    state.inventory_book_value = opening_book
                available_book = opening_book + production_cost
            else:
                opening_book = (
                    state.inventory_book_value
                    if not first_record
                    else inventory_units * price
                )
                available_book = opening_book + production_cost
                available_units = (
                    inventory_units + sales_units + public_purchase_units + spoilage_units
                )
                average_cost = available_book / available_units if available_units > 1e-12 else 0.0
                sold_units = sales_units + public_purchase_units
                cogs = min(available_book, sold_units * average_cost)
                spoilage_loss = min(max(0.0, available_book - cogs), spoilage_units * average_cost)
                ending_book = max(0.0, available_book - cogs - spoilage_loss)
            state.inventory_book_value = ending_book

            # Customer advances are balance-sheet movements, not revenue.
            # Revenue is recognized only when inventory is delivered.
            accounting_revenue = sales_revenue + public_inflow + other_inflow
            # Wages attached to actual production are capitalized into
            # inventory cost.  Any wage paid while no output was produced is
            # a genuine period expense; leaving it in neither category breaks
            # the equity bridge when a runtime capacity constraint binds.
            unallocated_wage_expense = max(0.0, wage - production_cost)
            public_tax_payment = _positive(
                getattr(firm, "actual_public_tax_this_step", 0.0)
            )
            pre_tax_operating_profit = (
                accounting_revenue
                - cogs
                - spoilage_loss
                - unallocated_wage_expense
                - capital_depreciation
            )
            operating_profit = pre_tax_operating_profit - public_tax_payment
            net_income = operating_profit - interest
            cfo = (
                sales_collections
                + public_inflow
                + other_inflow
                + customer_advance_inflow
                - wage
                - other_outflow
                - interest
            )
            cfi = (
                -prepaid_investment_paid
                if getattr(self.world, "capital_good_customer_advance_enabled", False)
                else -investment_expenditure
            )
            # Startup capitalization is a financing transfer from existing
            # Food-Firm cash into the new capital-good Firm.  It is not
            # operating revenue and must not inflate operating profit.
            cff = (
                loan_issued
                - loan_repaid
                - dividends
                + equity_issuance
                + startup_inflow
                - startup_outflow
            )
            cash_flow_gap = cash_end - cash_start - cfo - cfi - cff
            assets = cash_end + ending_book + capital_book_value + prepaid_investment_asset
            loan_principal = _positive(getattr(firm, "loan_balance", 0.0))
            customer_advance_liability = _positive(
                getattr(firm, "customer_advance_liability", 0.0)
            )
            lender_liabilities = (
                loan_principal + post_termout_arrears + legacy_term_claim
            )
            liabilities = lender_liabilities + customer_advance_liability
            equity = assets - liabilities
            balance_sheet_gap = assets - liabilities - equity
            inventory_bridge_gap = (
                0.0
                if first_record
                else ending_book
                - state.previous_inventory_book_value
                - production_cost
                + cogs
                + spoilage_loss
            )
            equity_bridge_gap = (
                0.0
                if state.previous_equity is None
                else equity
                - state.previous_equity
                - net_income
                - equity_issuance
                - startup_inflow
                + startup_outflow
                + dividends
            )
            capital_book_value_bridge_gap = (
                capital_book_value
                - capital_opening_book
                - capital_acquisitions
                + capital_depreciation
                + capital_disposals
            )
            advance_liability = _positive(
                getattr(firm, "customer_advance_liability", 0.0)
            )
            advance_liability_opening = (
                state.previous_customer_advance_liability
                if state.previous_customer_advance_liability is not None
                else max(0.0, advance_liability - customer_advance_inflow + customer_advance_delivered)
            )
            customer_advance_bridge_gap = (
                advance_liability
                - advance_liability_opening
                - customer_advance_inflow
                + customer_advance_delivered
            )
            prepaid_opening = (
                state.previous_prepaid_investment_asset
                if state.previous_prepaid_investment_asset is not None
                else max(0.0, prepaid_investment_asset - prepaid_investment_paid + customer_advance_delivered)
            )
            prepaid_investment_bridge_gap = (
                prepaid_investment_asset
                - prepaid_opening
                - prepaid_investment_paid
                + prepaid_investment_capitalized
            )

            row = {
                "step": step, "firm_id": firm_id,
                "sales_revenue": sales_revenue,
                "other_existing_operating_revenue": public_inflow + other_inflow,
                "accounting_revenue": accounting_revenue,
                "production_wage_cost": wage,
                "manufacturing_cost": production_cost,
                "wage_expense": unallocated_wage_expense,
                "inventory_cost_or_cogs": cogs,
                "spoilage_or_inventory_loss": spoilage_loss,
                "accounting_operating_profit": operating_profit,
                "pre_tax_operating_profit": pre_tax_operating_profit,
                "public_tax_expense": public_tax_payment,
                "accounting_net_income": net_income,
                "legacy_profit_before_dividend": getattr(firm, "profit", 0.0),
                "sales_collections": sales_collections,
                "wage_payments": wage,
                "customer_advance_cash_inflow": customer_advance_inflow,
                "prepaid_investment_cash_outflow": prepaid_investment_paid,
                "other_operating_cashflows": (
                    public_inflow
                    + other_inflow
                    + customer_advance_inflow
                    - other_outflow
                    - interest
                ),
                "cfo": cfo, "cfi": cfi, "cff": cff,
                "fixed_investment_expenditure": investment_expenditure,
                "customer_advance_received": customer_advance_inflow,
                "customer_advance_delivered": customer_advance_delivered,
                "customer_advance_liability_bridge_gap": customer_advance_bridge_gap,
                "prepaid_capital_investment_asset": prepaid_investment_asset,
                "prepaid_capital_investment_paid": prepaid_investment_paid,
                "prepaid_investment_capitalized": prepaid_investment_capitalized,
                "prepaid_investment_bridge_gap": prepaid_investment_bridge_gap,
                "capital_asset_book_value": capital_book_value,
                "capital_book_value_opening": capital_opening_book,
                "capital_book_value_closing": capital_book_value,
                "capital_depreciation_expense": capital_depreciation,
                "capital_asset_acquisitions": capital_acquisitions,
                "capital_asset_disposals": capital_disposals,
                "capital_book_value_bridge_gap": capital_book_value_bridge_gap,
                "loan_issued": loan_issued, "principal_repaid": loan_repaid,
                "dividends": dividends, "interest_paid": interest,
                "equity_issuance_cash": equity_issuance,
                "startup_capitalization_inflow": startup_inflow,
                "startup_capitalization_outflow": startup_outflow,
                "startup_capitalization_financing_cashflow": (
                    startup_inflow - startup_outflow
                ),
                "cash_start": cash_start, "cash_end": cash_end,
                "inventory_units": inventory_units,
                "public_inventory_purchase_units": public_purchase_units,
                "inventory_market_value": inventory_units * price,
                "inventory_book_value": ending_book,
                "unit_production_cost": unit_production_cost,
                "cash": cash_end, "total_assets": assets,
                # `loan_balance` remains a compatibility field, but now
                # means only the Firm's true principal.  Customer advances
                # and interest claims stay separately named components.
                "loan_principal": loan_principal,
                "loan_balance": loan_principal,
                "customer_advance_liability": customer_advance_liability,
                "lender_liabilities": lender_liabilities,
                "total_liabilities": liabilities,
                "equity": equity,
                "revolving_principal_claim": loan_principal,
                "post_termout_interest_arrears_claim": post_termout_arrears,
                "legacy_arrears_term_claim": legacy_term_claim,
                "total_lender_claim": lender_liabilities,
                "claim_reclassification_gap": getattr(
                    firm, "termout_claim_reclassification_gap", 0.0
                ),
                "legacy_term_claim_payment": getattr(
                    firm, "legacy_term_claim_payment", 0.0
                ),
                "post_termout_arrears_payment": getattr(
                    firm, "post_termout_arrears_payment", 0.0
                ),
                "cash_flow_gap": cash_flow_gap,
                "balance_sheet_gap": balance_sheet_gap,
                "capitalized_production_cost": production_cost,
                "capital_asset_acquisition_cost": _positive(
                    getattr(firm, "investment_assets_acquired_this_step", 0.0)
                ),
                "inventory_bridge_gap": inventory_bridge_gap,
                "equity_bridge_gap": equity_bridge_gap,
            }
            row.update(time_metadata)
            state.rows.append(row)
            self.rows.append(row)
            state.previous_inventory_book_value = ending_book
            state.previous_equity = equity
            state.previous_capital_book_value = capital_book_value
            state.previous_customer_advance_liability = advance_liability
            state.previous_prepaid_investment_asset = prepaid_investment_asset
            total_loan += loan_principal
            total_cash += cash_end

        central_bank = getattr(getattr(self.world, "firm_system", None), "central_bank", None)
        cb_loan_assets = _positive(getattr(central_bank, "firm_loan_balance", 0.0))
        cb_interest_receivable = _positive(
            getattr(central_bank, "interest_receivable", 0.0)
        )
        money_supply = _positive(getattr(central_bank, "money_supply", 0.0))
        household_cash = sum(
            _positive(getattr(h, "wealth", 0.0))
            for h in getattr(self.world, "households", [])
        )
        public_cash = float(getattr(self.world, "public_wealth", 0.0)) + float(
            getattr(central_bank, "public_income_balance", 0.0)
        )
        government_budget = getattr(self.world, "public_budget", None)
        government_cash = float(getattr(government_budget, "cash", 0.0))
        legacy_owner_cash = _positive(
            getattr(self.world, "legacy_owner_cash", 0.0)
        )
        estate_cash = math.fsum(
            _positive(getattr(account, "cash", 0.0))
            for account in getattr(self.world, "estate_accounts", {}).values()
        )
        pending_formation_cash = float(
            getattr(self.world, "pending_household_formation_wealth", 0.0)
        )
        total_non_household_owner_cash = legacy_owner_cash + estate_cash
        located_money = (
            total_cash
            + household_cash
            + public_cash
            + government_cash
            + total_non_household_owner_cash
            + pending_formation_cash
        )
        # The old implementation inferred the pre-existing monetary base
        # from the observed cash holders, making the location gap tautological
        # even when a holder was omitted.  Use the runtime opening stock plus
        # net central-bank money supply as the authoritative liability scope.
        opening_money_stock = float(
            getattr(self.world, "initial_private_money_stock", 0.0)
        )
        authoritative_money_stock = opening_money_stock + money_supply
        preexisting_money_base = opening_money_stock
        unallocated_money = max(0.0, authoritative_money_stock - located_money)
        total_money_liabilities = authoritative_money_stock
        reconciliation_row = {
            "step": step,
            "firm_loan_liabilities": total_loan,
            "central_bank_loan_assets": cb_loan_assets,
            "loan_reconciliation_gap": total_loan - cb_loan_assets,
            "firm_total_lender_claim": math.fsum(
                _positive(getattr(firm, "loan_balance", 0.0))
                + _positive(getattr(firm, "interest_arrears", 0.0))
                + _positive(getattr(firm, "legacy_arrears_term_claim", 0.0))
                for firm in firms
            ),
            "central_bank_interest_receivable": cb_interest_receivable,
            "central_bank_total_lender_claim": cb_loan_assets + cb_interest_receivable,
            "total_lender_claim_gap": (
                math.fsum(
                    _positive(getattr(firm, "loan_balance", 0.0))
                    + _positive(getattr(firm, "interest_arrears", 0.0))
                    + _positive(getattr(firm, "legacy_arrears_term_claim", 0.0))
                    for firm in firms
                )
                - cb_loan_assets
                - cb_interest_receivable
            ),
            "central_bank_money_supply": money_supply,
            "authoritative_money_stock": authoritative_money_stock,
            "firm_cash_stock": total_cash,
            "household_cash_stock": household_cash,
            "public_cash_stock": public_cash,
            "government_budget_cash": government_cash,
            "legacy_owner_cash": legacy_owner_cash,
            "estate_cash": estate_cash,
            "pending_household_formation_wealth": pending_formation_cash,
            "total_non_household_owner_cash": total_non_household_owner_cash,
            "located_money_total": located_money,
            "central_bank_unallocated_money": unallocated_money,
            "preexisting_non_central_bank_money": preexisting_money_base,
            "total_money_liabilities": total_money_liabilities,
            "money_location_total": located_money,
            "money_location_gap": total_money_liabilities - located_money,
            "full_money_location_gap": total_money_liabilities - located_money,
        }
        reconciliation_row.update(time_metadata)
        self.reconciliation_rows.append(reconciliation_row)

    def record_households(self, step):
        time_metadata = self.world.time_metadata(step)
        wages = dividends = transfers = gross_paid = gross_received = consumption = saving = wealth = 0.0
        for household in getattr(self.world, "households", []):
            if not getattr(household, "active", True):
                continue
            income = _positive(getattr(household, "income_this_step", 0.0))
            consumption_value = _positive(getattr(household, "consumption_this_step", 0.0))
            household_dividend = _positive(getattr(household, "dividend_income_this_step", 0.0))
            household_wage = _positive(
                getattr(household, "wage_income_this_step", income - household_dividend)
            )
            paid_transfer = _positive(getattr(household, "private_support_paid_this_step", 0.0))
            received_transfer = _positive(getattr(household, "private_support_received_this_step", 0.0))
            household_saving = (
                income
                - consumption_value
                - paid_transfer
                - _positive(getattr(household, "personal_income_tax_this_step", 0.0))
            )
            wages += household_wage
            dividends += household_dividend
            gross_paid += paid_transfer
            gross_received += received_transfer
            transfers += received_transfer - paid_transfer
            consumption += consumption_value
            saving += household_saving
            wealth += float(getattr(household, "wealth", 0.0))
        audit = getattr(self.world, "current_public_wealth_audit", {})
        estate_no_heir_outflow = float(
            audit.get("public_wealth_from_no_heir", 0.0)
            + audit.get("public_wealth_from_household_dissolution", 0.0)
        )
        previous = getattr(self, "previous_household_wealth", None)
        equity_purchase_cash_outflow = sum(
            _positive(
                getattr(
                    household,
                    "equity_purchase_cash_outflow_this_step",
                    0.0,
                )
            )
            for household in getattr(self.world, "households", [])
            if getattr(household, "active", True)
        )
        wealth_bridge_gap = (
            0.0
            if previous is None
            else (
                wealth
                - previous
                - saving
                + estate_no_heir_outflow
                + equity_purchase_cash_outflow
            )
        )
        row = {
            "step": step, "wages": wages, "dividends": dividends,
            "net_interhousehold_transfer": transfers,
            "gross_interhousehold_transfer_paid": gross_paid,
            "gross_interhousehold_transfer_received": gross_received,
            "inheritance_public_transfers": 0.0,
            "consumption_expenditure": consumption, "saving": saving,
            "personal_income_tax": sum(_positive(getattr(household, "personal_income_tax_this_step", 0.0)) for household in getattr(self.world, "households", []) if getattr(household, "active", True)),
            "cash_wealth": wealth,
            "equity_purchase_cash_outflow": equity_purchase_cash_outflow,
            "estate_no_heir_outflow": estate_no_heir_outflow,
            "household_wealth_bridge_gap": wealth_bridge_gap,
        }
        row.update(time_metadata)
        self.household_rows.append(row)
        self.previous_household_wealth = wealth

    def record_public_and_central_bank(self, step):
        time_metadata = self.world.time_metadata(step)
        cb = getattr(getattr(self.world, "firm_system", None), "central_bank", None)
        public_wealth = float(getattr(self.world, "public_wealth", 0.0))
        public_balance = float(getattr(cb, "public_income_balance", 0.0))
        audit = getattr(self.world, "current_public_wealth_audit", {})
        estate_no_heir_income = float(
            audit.get("public_wealth_from_no_heir", 0.0)
        )
        inventory_release_income = float(
            getattr(cb, "market_release_public_income_this_step", 0.0)
        )
        estate_other_income = float(
            audit.get("public_wealth_from_household_dissolution", 0.0)
            + audit.get("public_wealth_from_other", 0.0)
        )
        loan_interest_income = float(
            audit.get("central_bank_public_income_from_loan_interest", 0.0)
        )
        other_actual_cash_revenue = estate_other_income + loan_interest_income
        public_revenue = (
            estate_no_heir_income
            + inventory_release_income
            + other_actual_cash_revenue
        )
        public_expenditure = float(getattr(cb, "public_income_used_this_step", 0.0))
        government_budget = getattr(self.world, "public_budget", None)
        government_cash = float(getattr(government_budget, "cash", 0.0))
        government_tax = float(getattr(government_budget, "actual_tax_this_step", 0.0))
        government_shortfall = float(getattr(government_budget, "tax_shortfall_this_step", 0.0))
        public_cash_end = public_wealth + public_balance
        previous_public_cash = getattr(self, "previous_public_cash", None)
        public_cash_flow_gap = (
            0.0
            if previous_public_cash is None
            else public_cash_end - previous_public_cash - public_revenue + public_expenditure
        )
        public_row = {
            "step": step,
            "estate_no_heir_income": estate_no_heir_income,
            "inventory_release_sale_income": inventory_release_income,
            "other_actual_cash_revenue": other_actual_cash_revenue,
            "public_cash_revenue": public_revenue,
            "inventory_purchase_cash_expenditure": public_expenditure,
            "public_cash_expenditure": public_expenditure,
            "in_kind_public_support_value": float(getattr(cb, "poverty_subsidy_value_this_step", 0.0)),
            "public_cash_start": previous_public_cash if previous_public_cash is not None else public_cash_end,
            "public_cash_end": public_cash_end,
            "public_cash_flow_gap": public_cash_flow_gap,
            "government_budget_cash": government_cash,
            "government_tax_revenue": government_tax,
            "government_tax_shortfall": government_shortfall,
            "personal_income_tax": float(getattr(government_budget, "personal_tax_actual_this_step", 0.0)),
            "corporate_profit_tax": float(getattr(government_budget, "corporate_tax_actual_this_step", 0.0)),
            "government_stock_flow_gap": 0.0,
            "balance": public_cash_end,
        }
        public_row.update(time_metadata)
        self.public_rows.append(public_row)
        self.previous_public_cash = public_cash_end
        credit_liabilities = float(getattr(cb, "firm_loan_balance", 0.0))
        gross_credit_created = float(getattr(cb, "loan_issued_this_step", 0.0))
        credit_destroyed = float(getattr(cb, "loan_principal_repaid_this_step", 0.0))
        credit_stock_flow_gap = (
            0.0
            if self.previous_credit_money_liabilities is None
            else credit_liabilities
            - self.previous_credit_money_liabilities
            - gross_credit_created
            + credit_destroyed
        )
        central_bank_row = {
            "step": step,
            "loan_assets": float(getattr(cb, "firm_loan_balance", 0.0)),
            "credit_money_liabilities": credit_liabilities,
            "gross_money_created": gross_credit_created,
            "money_destroyed": credit_destroyed,
            "net_money_change": gross_credit_created - credit_destroyed,
            "credit_money_stock_flow_gap": credit_stock_flow_gap,
            "inventory_money_created": max(
                0.0,
                float(getattr(cb, "money_issued_this_step", 0.0))
                - gross_credit_created
                + credit_destroyed,
            ),
            "loan_issued": float(getattr(cb, "loan_issued_this_step", 0.0)),
            "principal_repaid": float(getattr(cb, "loan_principal_repaid_this_step", 0.0)),
        }
        central_bank_row.update(time_metadata)
        self.central_bank_rows.append(central_bank_row)
        self.previous_credit_money_liabilities = credit_liabilities

    def record_step(self, step, firms, aggregate=None):
        self.record_firms(step, firms, aggregate)
        self.record_households(step)
        self.record_public_and_central_bank(step)

    def export_csv(self, directory):
        os.makedirs(directory, exist_ok=True)
        tables = {
            "firm_accounting.csv": self.rows,
            "household_accounting.csv": self.household_rows,
            "public_accounting.csv": self.public_rows,
            "central_bank_accounting.csv": self.central_bank_rows,
            "accounting_reconciliation.csv": self.reconciliation_rows,
            "household_lifecycle.csv": getattr(
                self.world, "household_lifecycle_events", []
            ),
        }
        paths = []
        for filename, rows in tables.items():
            path = os.path.join(directory, filename)
            fields = []
            for row in rows:
                for key in row:
                    if key not in fields:
                        fields.append(key)
            with open(path, "w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(handle, fieldnames=fields)
                if fields:
                    writer.writeheader()
                    writer.writerows(rows)
            paths.append(path)
        return paths
