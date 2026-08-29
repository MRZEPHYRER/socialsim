import json
import math

from economy import config
from economy.ledger import object_account
import central_bank.config as central_bank_config
from central_bank.central_bank import CentralBank
from economy.investment_contracts import CapitalStock
from economy.ownership_accounting import CapTable
from economy.dividend_routing import (
    ensure_dividend_routing_state,
    route_declared_dividend,
)
from productivity import age_productivity
from time_system import months_to_steps


def safe_ratio(numerator, denominator):
    if denominator == 0:
        return 0.0

    return numerator / denominator


class FirmSystem:

    def __init__(self, world):
        self.world = world
        self.product_id = config.BASIC_CONSUMPTION_GOOD_ID
        self.product = self.world.goods_catalog.get(self.product_id)
        self.cash = config.FIRM_INITIAL_CASH
        # Passive Step 15B container.  It is empty and intentionally ignored
        # by the current labor-only Food executor.
        self.capital_stock = CapitalStock(owner_firm_id=0)
        self.cap_table = CapTable.initial_legacy_owned(0)
        self.paid_in_equity = 0.0
        self.price = config.FOOD_INITIAL_PRICE
        self.expected_sales = None
        self.last_profit_before_dividend = 0.0
        self.cash_history = []

        self.food_productivity = config.FOOD_PRODUCTIVITY_PER_LABOR
        self.target_inventory_days = config.FOOD_TARGET_INVENTORY_DAYS
        self.production_scale = 1.0
        self.food_inventory_units = 0.0
        self.food_output_units = 0.0
        self.food_demand_units = 0.0
        self.food_sales_units = 0.0
        self.unmet_food_demand_units = 0.0
        self.inventory_demand_ratio = 0.0
        self.available_food_supply_units = 0.0
        self.demand_pressure = 0.0

        # Step 15I.1 passive capital-good supply state.  The default Firm is
        # not a capital-good producer; these fields remain empty/inactive
        # until a future generic Firm is explicitly assigned that sector.
        self.capital_good_inventory = None
        self.capital_good_production_units = 0.0
        self.capital_good_sales_units = 0.0
        self.capital_good_unit_production_cost = 0.0
        self.capital_good_labor_services = 0.0

        self.real_output_value = 0.0
        self.nominal_output_value = 0.0
        self.food_inventory_value = 0.0
        self.wage_bill = 0.0
        self.scheduled_wage_bill = 0.0
        self.executed_wage_bill = 0.0
        self.scheduled_productive_capacity = 0.0
        self.funded_productive_capacity = 0.0
        self.unit_labor_cost = 0.0
        self.previous_unit_labor_cost = None
        self.unit_labor_cost_growth = 0.0
        self.price_inventory_gap = 0.0
        self.price_cost_growth = 0.0
        self.price_inflation_signal = 0.0
        self.price_log_adjustment = 0.0
        self.dividend_paid = 0.0
        self.household_dividend_paid_this_step = 0.0
        self.person_dividend_paid = 0.0
        self.legacy_dividend_entitlement = 0.0
        self.estate_dividend_paid = 0.0
        self.dividend_routing_gap = 0.0
        self.dividend_recipient_count = 0
        self.equity_issuance_cash_this_step = 0.0
        self.equity_issuance_shares_this_step = 0.0
        self.equity_issuance_buyer_count = 0
        self.sales_revenue = 0.0
        self.profit_before_dividend = 0.0
        self.inventory_depreciation = 0.0
        self.sales_income_ratio = 0.0
        self.cash_slope_signal = 0.0

        self.money_issued_this_step = 0.0
        self.inventory_monetized_units_this_step = 0.0
        self.cumulative_money_issued = 0.0
        self.cumulative_inventory_monetized_units = 0.0
        self.working_capital_loan_balance = 0.0
        self.working_capital_loan_issued_this_step = 0.0
        self.working_capital_loan_repaid_this_step = 0.0
        self.working_capital_interest_paid_this_step = 0.0
        self.working_capital_target_cash = 0.0
        self.working_capital_funding_gap = 0.0
        self.executed_wage_bill = 0.0

        self.central_bank = CentralBank()
        self.central_bank_inventory_purchase = 0.0
        self.central_bank_market_release_revenue = 0.0
        self.central_bank_market_release_money_destroyed = 0.0
        self.central_bank_market_release_public_income = 0.0
        self.central_bank_poverty_subsidy_value = 0.0
        self.central_bank_public_income = 0.0
        self.central_bank_public_income_used = 0.0
        self.central_bank_public_income_balance = 0.0
        self.central_bank_food_inventory_units = 0.0
        self.central_bank_food_purchase_units = 0.0
        self.central_bank_food_release_units = 0.0
        self.central_bank_food_subsidy_units = 0.0

    def net_worth(self):
        return self.cash + self.food_inventory_value

    def cash_account(self):
        return object_account(
            self,
            "cash",
            "firm.cash",
        )

    def household_wealth_account(self, household):
        return object_account(
            household,
            "wealth",
            f"household.{household.id}.wealth",
        )

    def active_households(self):
        return self.world.active_households()

    def total_labor(self):
        household_dict = self.world.household_dict
        capital_firm_ids = set(
            getattr(self.world, "capital_good_firm_ids", set())
        )
        total = 0.0

        for person in self.world.population:
            if (
                person.alive
                and person.household_id in household_dict
                and self.world.labor_participates(person)
                and getattr(person, "firm_id", None) not in capital_firm_ids
            ):
                total += age_productivity(person.age)

        return total

    def update_expected_sales(self, sales):
        if self.expected_sales is None:
            self.expected_sales = sales
            return

        self.expected_sales = (
            config.FIRM_EXPECTED_SALES_ALPHA
            *
            sales
            +
            (1 - config.FIRM_EXPECTED_SALES_ALPHA)
            *
            self.expected_sales
        )

    def recent_cash_slope(self):
        if len(self.cash_history) < 2:
            return 0.0

        recent = self.cash_history[
            -config.FOOD_PRICE_CASH_WINDOW:
        ]
        n = len(recent)

        if n < 2:
            return 0.0

        mean_x = (n - 1) / 2
        mean_y = sum(recent) / n
        numerator = 0.0
        denominator = 0.0

        for index, value in enumerate(recent):
            dx = index - mean_x
            numerator += dx * (value - mean_y)
            denominator += dx * dx

        if denominator == 0:
            return 0.0

        return numerator / denominator

    def distribute_wages(self, wage_bill, total_labor):
        for household in self.world.households:
            household.wealth_before_income_this_step = household.wealth
            household.income_this_step = 0.0
            household.wage_income_this_step = 0.0
            household.dividend_income_this_step = 0.0

        # Capital-good wages are settled by the separately gated investment
        # layer before the Food executor runs.  Add them after the canonical
        # reset so they enter the same household budget this week without
        # being paid a second time by the Food FirmSystem.
        pending_capital_wages = getattr(
            self.world,
            "pending_capital_wages",
            {},
        )
        for household_id, amount in pending_capital_wages.items():
            household = self.world.household_dict.get(household_id)
            if household is None:
                continue
            household.wealth += amount
            household.income_this_step += amount
            household.wage_income_this_step += amount

        payroll_provenance = {}
        for person_id, item in getattr(
            self.world, "pending_capital_person_wages", {}
        ).items():
            household_id = item.get("household_id")
            if household_id is None:
                continue
            record = payroll_provenance.setdefault(household_id, {
                "paid_person_ids": set(),
                "firm_components": {},
                "person_components": [],
            })
            firm_id = item.get("firm_id")
            component = record["firm_components"].setdefault(firm_id, {
                "firm_id": firm_id,
                "sector_id": item.get("sector_id", "capital_goods"),
                "scheduled_wage": 0.0,
                "paid_wage": 0.0,
                "scheduled_person_ids": set(),
                "paid_person_ids": set(),
            })
            component["scheduled_wage"] += float(item.get("scheduled_wage", 0.0))
            component["paid_wage"] += float(item.get("paid_wage", 0.0))
            if float(item.get("scheduled_wage", 0.0)) > 1e-12:
                component["scheduled_person_ids"].add(person_id)
            if float(item.get("paid_wage", 0.0)) > 1e-12:
                component["paid_person_ids"].add(person_id)
                record["paid_person_ids"].add(person_id)
            record["person_components"].append({
                "person_id": person_id,
                "firm_id": firm_id,
                "sector_id": item.get("sector_id", "capital_goods"),
                "age": float(item.get("age", 0.0)),
                "productivity": float(item.get("productivity", 0.0)),
                "scheduled_wage": float(item.get("scheduled_wage", 0.0)),
                "paid_wage": float(item.get("paid_wage", 0.0)),
            })

        if total_labor <= 0:
            return

        household_wages = {}
        household_firm_wages = {}
        employer_instrumentation = bool(
            getattr(
                self.world,
                "household_employer_exposure_instrumentation_enabled",
                False,
            )
            or getattr(self.world, "payg_pension_enabled", False)
        )
        household_exposure = {}
        multi_firm = len(getattr(self.world, "firms", [])) > 1
        funding_ratios = getattr(
            self.world,
            "multi_firm_payroll_funding_ratios",
            {},
        )
        paid_by_firm = {}

        get_household = self.world.household_dict.get
        productivity = age_productivity
        household_wages_get = household_wages.get

        for person in self.world.population:
            if not person.alive or not self.world.labor_participates(person):
                continue

            if getattr(person, "firm_id", None) in set(
                getattr(self.world, "capital_good_firm_ids", set())
            ):
                continue

            household = self.world.settlement_household_for_person(person)

            if household is None:
                continue

            scheduled_wage = (
                wage_bill
                *
                productivity(person.age)
                /
                total_labor
            )
            ratio = (
                max(
                    0.0,
                    min(
                        1.0,
                        funding_ratios.get(getattr(person, "firm_id", None), 1.0),
                    ),
                )
                if multi_firm
                else 1.0
            )
            wage = scheduled_wage * ratio
            settlement_household_id = household.id
            payroll_record = payroll_provenance.setdefault(settlement_household_id, {
                "paid_person_ids": set(),
                "firm_components": {},
                "person_components": [],
            })
            firm_id = getattr(person, "firm_id", None)
            firm_for_sector = self.world.firm_dict.get(firm_id, self) if multi_firm else self
            component = payroll_record["firm_components"].setdefault(firm_id, {
                "firm_id": firm_id,
                "sector_id": getattr(firm_for_sector, "sector_id", "food"),
                "scheduled_wage": 0.0,
                "paid_wage": 0.0,
                "scheduled_person_ids": set(),
                "paid_person_ids": set(),
            })
            component["scheduled_wage"] += scheduled_wage
            component["paid_wage"] += wage
            if scheduled_wage > 1e-12:
                component["scheduled_person_ids"].add(person.id)
            if wage > 1e-12:
                component["paid_person_ids"].add(person.id)
                payroll_record["paid_person_ids"].add(person.id)
            payroll_record["person_components"].append({
                "person_id": person.id,
                "firm_id": firm_id,
                "sector_id": getattr(firm_for_sector, "sector_id", "food"),
                "age": float(person.age),
                "productivity": productivity(person.age),
                "scheduled_wage": scheduled_wage,
                "paid_wage": wage,
            })
            if employer_instrumentation:
                firm_id = getattr(person, "firm_id", None)
                firm = self.world.firm_dict.get(firm_id, self) if multi_firm else self
                firm_requested = max(0.0, float(getattr(firm, "requested_credit", 0.0)))
                firm_issued = max(
                    0.0,
                    float(getattr(firm, "loan_issued", 0.0)),
                    float(getattr(firm, "executed_credit", 0.0)),
                )
                firm_binding = bool(getattr(firm, "credit_limit_binding", False))
                firm_underfunded = float(getattr(firm, "payroll_funding_ratio", 1.0)) < 1.0 - 1e-9
                record = household_exposure.setdefault(household.id, {
                    "household_id": household.id,
                    "employed_worker_count": 0,
                    "employer_ids": set(),
                    "scheduled_wage_income": 0.0,
                    "executed_wage_income": 0.0,
                    "workers_at_borrowing_firms": 0,
                    "workers_at_credit_binding_firms": 0,
                    "workers_at_payroll_underfunded_firms": 0,
                    "firm_states": {},
                })
                record["employed_worker_count"] += 1
                if firm_id is not None:
                    record["employer_ids"].add(firm_id)
                record["scheduled_wage_income"] += scheduled_wage
                record["executed_wage_income"] += wage
                record["workers_at_borrowing_firms"] += int(firm_issued > 1e-12)
                record["workers_at_credit_binding_firms"] += int(firm_binding)
                record["workers_at_payroll_underfunded_firms"] += int(firm_underfunded)
                if firm_id is not None:
                    record["firm_states"][str(firm_id)] = {
                        "loan_principal": float(getattr(firm, "loan_balance", 0.0)),
                        "credit_requested": firm_requested,
                        "credit_executed": firm_issued,
                        "credit_denied": float(getattr(firm, "denied_credit", 0.0)),
                        "credit_binding": firm_binding,
                        "payroll_funding_ratio": float(getattr(firm, "payroll_funding_ratio", 1.0)),
                        "scheduled_wage_bill": float(getattr(firm, "scheduled_wage_bill", 0.0)),
                        "executed_wage_bill": float(getattr(firm, "executed_wage_bill", 0.0)),
                        "financial_state": int(getattr(firm, "financing_state", 0)),
                    }
            if multi_firm:
                firm_id = getattr(person, "firm_id", None)
                paid_by_firm[firm_id] = (
                    paid_by_firm.get(firm_id, 0.0) + wage
                )
            household_wages[household] = (
                household_wages_get(household, 0.0)
                +
                wage
            )
            if multi_firm and wage > 0:
                firm_wages = household_firm_wages.setdefault(household, {})
                firm_id = getattr(person, "firm_id", None)
                firm_wages[firm_id] = (
                    firm_wages.get(firm_id, 0.0) + wage
                )

        for household, wage in household_wages.items():
            household.income_this_step += wage
            household.wage_income_this_step += wage
            payments = (
                household_firm_wages.get(household, {})
                if multi_firm
                else {None: wage}
            )
            for firm_id, amount in payments.items():
                payer_obj = self
                payer_name = "firm.cash"
                if multi_firm:
                    payer_obj = self.world.firm_dict.get(firm_id, self)
                    payer_name = f"firm[{firm_id}].cash"
                if self.world.ledger.record_details:
                    self.world.ledger.transfer_attrs(
                        payer_obj=payer_obj,
                        payer_attr="cash",
                        payer_name=payer_name,
                        receiver_obj=household,
                        receiver_attr="wealth",
                        receiver_name=f"household.{household.id}.wealth",
                        amount=amount,
                        reason="wage_payment",
                    )
                else:
                    household.wealth += amount

        executed_wage_bill = sum(paid_by_firm.values()) if multi_firm else wage_bill
        if not self.world.ledger.record_details:
            if multi_firm:
                for firm_id, amount in paid_by_firm.items():
                    firm = self.world.firm_dict.get(firm_id)
                    if firm is not None:
                        firm.cash -= amount
            else:
                self.cash -= wage_bill
            self.world.ledger.record_summary(executed_wage_bill)

        self.executed_wage_bill = executed_wage_bill
        self.wage_payment = executed_wage_bill
        if employer_instrumentation:
            for household_id, record in payroll_provenance.items():
                components = []
                for component in record["firm_components"].values():
                    components.append({
                        "firm_id": component["firm_id"],
                        "sector_id": component["sector_id"],
                        "scheduled_wage": component["scheduled_wage"],
                        "paid_wage": component["paid_wage"],
                        "scheduled_person_count": len(component["scheduled_person_ids"]),
                        "paid_person_count": len(component["paid_person_ids"]),
                    })
                record["paid_person_count"] = len(record["paid_person_ids"])
                record["firm_components"] = components
                record.pop("paid_person_ids", None)
            self.world._household_payroll_provenance_weekly_state = payroll_provenance
        elif hasattr(self.world, "_household_payroll_provenance_weekly_state"):
            self.world._household_payroll_provenance_weekly_state = {}
        if employer_instrumentation:
            for household_id, record in household_exposure.items():
                worker_count = record["employed_worker_count"]
                record["employer_ids"] = ";".join(
                    str(value) for value in sorted(record["employer_ids"])
                )
                record["wage_shortfall"] = max(
                    0.0,
                    record["scheduled_wage_income"] - record["executed_wage_income"],
                )
                record["household_wage_execution_ratio"] = (
                    record["executed_wage_income"] / record["scheduled_wage_income"]
                    if record["scheduled_wage_income"] > 1e-12 else 1.0
                )
                record["share_workers_at_borrowing_firms"] = (
                    record["workers_at_borrowing_firms"] / worker_count
                    if worker_count else 0.0
                )
                record["share_workers_at_credit_binding_firms"] = (
                    record["workers_at_credit_binding_firms"] / worker_count
                    if worker_count else 0.0
                )
                record["share_workers_at_payroll_underfunded_firms"] = (
                    record["workers_at_payroll_underfunded_firms"] / worker_count
                    if worker_count else 0.0
                )
                record["firm_states"] = json.dumps(
                    record["firm_states"], sort_keys=True, separators=(",", ":")
                )
            self.world._household_employer_weekly_state = household_exposure
        elif hasattr(self.world, "_household_employer_weekly_state"):
            self.world._household_employer_weekly_state = {}
        return executed_wage_bill

    def distribute_dividends(self, dividend):
        dividend = max(0.0, float(dividend))
        ensure_dividend_routing_state(self.world)
        self.household_dividend_paid_this_step = 0.0
        self.person_dividend_paid = 0.0
        self.legacy_dividend_entitlement = 0.0
        self.estate_dividend_paid = 0.0
        self.dividend_routing_gap = 0.0
        self.dividend_recipient_count = 0

        if dividend <= 0:
            return 0.0

        # Multi-Firm settlement is finalized by World after each FirmSlice's
        # own dividend is known.  The aggregate executor must not create a
        # temporary Household payment or legacy payment here.
        if len(getattr(self.world, "firms", [])) > 1:
            self.dividend_payment = dividend
            return dividend

        # In the one-Firm path the aggregate FirmSystem is the cash payer,
        # while the compatibility FirmSlice carries the authoritative table.
        table_firm = self.world.get_firm_by_id(0, self)
        result = route_declared_dividend(
            self.world,
            table_firm,
            dividend,
            payer=self,
        )
        self.household_dividend_paid_this_step = result.person_paid
        self.person_dividend_paid = result.person_paid
        self.legacy_dividend_entitlement = result.legacy_entitlement
        self.estate_dividend_paid = result.estate_paid
        self.dividend_routing_gap = result.reconciliation_gap
        self.dividend_recipient_count = len(result.person_receipts)

        return dividend

    def wage_bill_for_sales_constrained_rule(self, nominal_output_value):
        output_wage = nominal_output_value * config.FIRM_WAGE_SHARE

        if self.expected_sales is None:
            return output_wage

        sales_target_wage = (
            self.expected_sales
            *
            config.FIRM_TARGET_WAGE_TO_SALES
        )
        blended_wage = (
            (1 - config.FIRM_SALES_CONSTRAINT_WEIGHT)
            *
            output_wage
            +
            config.FIRM_SALES_CONSTRAINT_WEIGHT
            *
            sales_target_wage
        )
        wage_floor = output_wage * config.FIRM_WAGE_FLOOR_SHARE

        return min(output_wage, max(wage_floor, blended_wage))

    def base_wage_bill(self, total_labor):
        base_wage = total_labor * config.FIRM_WAGE_PER_LABOR
        sales_adjustment = max(
            0.80,
            min(
                1.15,
                1.0
                +
                config.FIRM_WAGE_SALES_FEEDBACK
                *
                (
                    self.sales_income_ratio
                    -
                    config.FOOD_TARGET_SALES_INCOME_RATIO
                ),
            ),
        )
        cash_adjustment = max(
            0.80,
            min(
                1.10,
                1.0
                -
                config.FIRM_WAGE_CASH_FEEDBACK
                *
                (
                    -self.cash_slope_signal
                    /
                    max(1.0, abs(self.cash))
                ),
            ),
        )

        wage = (
            base_wage
            *
            config.WAGE_MULTIPLIER
            *
            sales_adjustment
            *
            cash_adjustment
        )

        return wage * self.world.scenario_multiplier(
            "WAGE_MULTIPLIER_STEP",
            "WAGE_MULTIPLIER",
        )

    def household_food_need_units(self, household):
        return self.world.needs_system.household_minimum_need_units(
            household
        )

    def household_food_purchase_units(self, household, need_units):
        household.food_subsidy_units_this_step = 0.0
        household.food_subsidy_value_this_step = 0.0

        price = self.world.household_planning_price_this_step()
        income_this_step = household.income_this_step
        necessary_consumption = need_units * price
        household.household_planning_price_used = price
        household.necessary_consumption_this_step = necessary_consumption
        starting_wealth = household.wealth_before_income_this_step
        private_support_paid = max(
            0.0,
            float(getattr(household, "private_support_paid_this_step", 0.0)),
        )
        starting_wealth = max(0.0, starting_wealth - private_support_paid)
        household.private_support_adjusted_opening_wealth = starting_wealth
        reserve_weeks = months_to_steps(
            config.TARGET_WEALTH_RESERVE_MONTHS
        )
        target_wealth = reserve_weeks * necessary_consumption
        household.target_wealth_reserve_months = config.TARGET_WEALTH_RESERVE_MONTHS
        household.target_wealth_reserve_weeks = reserve_weeks
        wealth_gap = target_wealth - starting_wealth
        positive_gap_ratio = safe_ratio(
            max(0.0, wealth_gap),
            max(1.0, target_wealth),
        )
        optional_consumption_multiplier = max(
            config.MIN_OPTIONAL_CONSUMPTION_MULTIPLIER,
            1.0
            -
            config.TARGET_WEALTH_GAP_SENSITIVITY
            *
            positive_gap_ratio,
        )
        optional_income_base = max(
            0.0,
            income_this_step
            -
            necessary_consumption,
        )
        optional_income_consumption = (
            optional_income_base
            *
            config.DISCRETIONARY_INCOME_RATE
            *
            optional_consumption_multiplier
        )
        excess_wealth_consumption = (
            max(0.0, -wealth_gap)
            *
            config.EXCESS_WEALTH_CONSUMPTION_RATE
        )
        optional_consumption = (
            optional_income_consumption
            +
            excess_wealth_consumption
        )
        optional_consumption *= self.world.scenario_multiplier(
            "DEMAND_SHOCK_STEP",
            "DEMAND_SHOCK_MULTIPLIER",
        )
        desired_money = (
            necessary_consumption
            +
            optional_consumption
        )
        affordable_money = (
            income_this_step
            +
            starting_wealth
            *
            config.WEALTH_DRAWDOWN_RATE
        )
        # Keep behavioral affordability separate from the cash available at
        # settlement: mandatory deductions may have changed current cash.
        budget_feasible_money = max(
            0.0,
            min(desired_money, affordable_money),
        )
        current_available_cash = max(0.0, float(getattr(household, "wealth", 0.0)))
        cash_settleable_money = min(
            budget_feasible_money,
            current_available_cash,
        )
        actual_money = max(0.0, cash_settleable_money)
        cash_constraint_binding = (
            budget_feasible_money - actual_money > 1e-9
        )
        planned_minus_realized = max(0.0, desired_money - actual_money)
        cash_constraint_suppressed = max(0.0, budget_feasible_money - actual_money)
        actual_units = actual_money / max(0.01, price)
        food_need_gap_units = max(0.0, need_units - actual_units)

        household.desired_consumption_this_step = desired_money
        household.optional_consumption_this_step = optional_consumption
        household.optional_income_consumption_this_step = (
            optional_income_consumption
        )
        household.excess_wealth_consumption_this_step = (
            excess_wealth_consumption
        )
        household.target_wealth_this_step = target_wealth
        household.wealth_gap_this_step = wealth_gap
        household.optional_consumption_multiplier_this_step = (
            optional_consumption_multiplier
        )
        household.affordable_consumption_this_step = affordable_money
        household.budget_feasible_consumption_this_step = budget_feasible_money
        household.current_available_cash_this_step = current_available_cash
        household.cash_settleable_consumption_this_step = cash_settleable_money
        household.consumption_cash_constraint_binding = cash_constraint_binding
        household.planned_minus_realized_consumption_this_step = planned_minus_realized
        household.cash_constraint_suppressed_consumption_this_step = cash_constraint_suppressed
        household.consumption_this_step = actual_money
        household.saving_this_step = (
            income_this_step
            - actual_money
            - private_support_paid
            - max(0.0, float(getattr(household, "payg_contribution_this_step", 0.0)))
            - max(0.0, float(getattr(household, "wealth_transfer_paid_this_step", 0.0)))
            - max(0.0, float(getattr(household, "personal_income_tax_this_step", 0.0)))
        )
        if self.world.ledger.record_details:
            self.world.ledger.transfer_attrs(
                payer_obj=household,
                payer_attr="wealth",
                payer_name=f"household.{household.id}.wealth",
                receiver_obj=self,
                receiver_attr="cash",
                receiver_name="firm.cash",
                amount=actual_money,
                reason="household_market_purchase",
            )
        else:
            household.wealth -= actual_money
        if -1e-8 < household.wealth < 0.0:
            household.wealth = 0.0

        household.economic_pressure_this_step = (
            desired_money / affordable_money
            if affordable_money > 0
            else float("inf")
        )
        household.budget_constrained_this_step = desired_money > affordable_money
        household.consumption_capped_this_step = (
            household.budget_constrained_this_step
        )
        household.max_consumption_this_step = affordable_money
        household.food_desired_units_this_step = actual_units
        household.food_need_units_this_step = need_units
        household.food_need_gap_units_this_step = food_need_gap_units
        household.real_food_consumption_units_this_step = actual_units
        household.food_constrained_after_subsidy_this_step = (
            food_need_gap_units > 0
        )
        if hasattr(self.world, "record_payg_cash_safety_consumption_plan"):
            self.world.record_payg_cash_safety_consumption_plan(household, household.wealth + actual_money)

        return actual_units, actual_money

    def run_household_food_market(self):
        demand_units = 0.0
        paid_money = 0.0
        constrained = 0
        consumed = 0
        households = self.world.households
        if hasattr(self.world, "record_payg_cash_safety_phase"):
            self.world.record_payg_cash_safety_phase("before_consumption")
        purchase_units = self.household_food_purchase_units
        need_units_for = self.household_food_need_units

        for household in households:
            need_units = need_units_for(household)

            if household.parents or household.children:
                consumed += 1

            units, money = purchase_units(
                household,
                need_units,
            )
            demand_units += units
            paid_money += money

            if getattr(household, "budget_constrained_this_step", False):
                constrained += 1

        consumption = self.world.consumption_system
        consumption.total_consumption = paid_money
        consumption.constrained_household_count = constrained
        consumption.consumed_household_count = consumed
        consumption.constrained_household_ratio = safe_ratio(
            constrained,
            consumed,
        )
        consumption.capped_household_count = constrained
        consumption.capped_household_ratio = consumption.constrained_household_ratio

        if not self.world.ledger.record_details:
            self.cash += paid_money
            self.world.ledger.record_summary(paid_money)

        return demand_units, paid_money

    def refund_unfilled_food_orders(self, fulfillment_ratio):
        if fulfillment_ratio >= 1.0:
            return

        fulfillment_ratio = max(0.0, fulfillment_ratio)

        for household in self.world.households:
            original_consumption = getattr(
                household,
                "consumption_this_step",
                0.0,
            )
            original_units = getattr(
                household,
                "food_desired_units_this_step",
                0.0,
            )
            adjusted_consumption = original_consumption * fulfillment_ratio
            adjusted_units = original_units * fulfillment_ratio
            refund = original_consumption - adjusted_consumption

            if refund <= 0:
                continue

            household.consumption_this_step = adjusted_consumption
            household.saving_this_step += refund
            if self.world.ledger.record_details:
                self.world.ledger.transfer_attrs(
                    payer_obj=self,
                    payer_attr="cash",
                    payer_name="firm.cash",
                    receiver_obj=household,
                    receiver_attr="wealth",
                    receiver_name=f"household.{household.id}.wealth",
                    amount=refund,
                    reason="unfilled_market_order_refund",
                )
            else:
                self.cash -= refund
                household.wealth += refund
                self.world.ledger.record_summary(refund)
            household.food_desired_units_this_step = adjusted_units
            household.real_food_consumption_units_this_step = (
                adjusted_units
                +
                getattr(household, "food_subsidy_units_this_step", 0.0)
            )
            household.food_need_gap_units_this_step = max(
                0.0,
                getattr(household, "food_need_units_this_step", 0.0)
                -
                household.real_food_consumption_units_this_step,
            )
            household.food_constrained_after_subsidy_this_step = (
                household.food_need_gap_units_this_step > 0
            )

    def update_unit_labor_cost(self):
        if self.food_output_units > config.FOOD_PRICE_EPSILON:
            unit_labor_cost = self.executed_wage_bill / self.food_output_units
        elif self.previous_unit_labor_cost is not None:
            unit_labor_cost = self.previous_unit_labor_cost
        else:
            unit_labor_cost = 0.0

        if (
            self.previous_unit_labor_cost is None
            or
            self.previous_unit_labor_cost <= config.FOOD_PRICE_EPSILON
        ):
            cost_growth = 0.0
        else:
            cost_growth = (
                unit_labor_cost
                -
                self.previous_unit_labor_cost
            ) / self.previous_unit_labor_cost

        self.unit_labor_cost = unit_labor_cost
        self.unit_labor_cost_growth = cost_growth
        self.price_cost_growth = cost_growth
        self.previous_unit_labor_cost = unit_labor_cost

    def update_demand_pressure(self, available_supply_units):
        self.available_food_supply_units = available_supply_units
        self.demand_pressure = (
            self.food_demand_units
            -
            available_supply_units
        ) / (
            available_supply_units
            +
            config.FOOD_PRICE_EPSILON
        )

    def update_price(self):
        demand_base = max(1.0, self.food_demand_units)
        self.inventory_demand_ratio = self.food_inventory_units / demand_base
        target_inventory = self.target_inventory_days * demand_base
        inventory_gap = safe_ratio(
            target_inventory - self.food_inventory_units,
            target_inventory + config.FOOD_PRICE_EPSILON,
        )
        inflation_signal = (
            config.FOOD_PRICE_INVENTORY_GAP_WEIGHT
            *
            inventory_gap
            +
            config.FOOD_PRICE_COST_GROWTH_WEIGHT
            *
            self.unit_labor_cost_growth
        )
        log_adjustment = max(
            -config.FOOD_PRICE_MAX_LOG_CHANGE,
            min(config.FOOD_PRICE_MAX_LOG_CHANGE, inflation_signal)
        )
        old_price = self.price

        self.price *= math.exp(log_adjustment)
        self.price = max(config.FOOD_MIN_PRICE, min(config.FOOD_MAX_PRICE, self.price))
        self.price_inventory_gap = inventory_gap
        self.price_inflation_signal = inflation_signal
        self.price_log_adjustment = math.log(self.price / old_price)

    def update_production_scale(self):
        target_inventory = (
            self.target_inventory_days
            *
            max(1.0, self.food_demand_units)
        )
        inventory_gap = safe_ratio(
            target_inventory - self.food_inventory_units,
            max(1.0, target_inventory),
        )
        adjustment = config.FOOD_PRODUCTION_ADJUSTMENT_RATE * inventory_gap
        self.production_scale *= max(0.92, min(1.08, 1.0 + adjustment))
        self.production_scale = max(
            config.FOOD_MIN_PRODUCTION_SCALE,
            min(config.FOOD_MAX_PRODUCTION_SCALE, self.production_scale),
        )

    def monetize_excess_inventory(self):
        self.money_issued_this_step = 0.0
        self.inventory_monetized_units_this_step = 0.0

        if not config.TEMP_INVENTORY_BACKED_MONEY_ISSUANCE_ENABLED:
            return

        demand_base = max(1.0, self.food_demand_units)
        target_inventory_units = self.target_inventory_days * demand_base
        mint_threshold_units = (
            target_inventory_units
            +
            config.TEMP_MINT_MIN_EXCESS_INVENTORY_DAYS
            *
            demand_base
        )
        excess_units = max(0.0, self.food_inventory_units - mint_threshold_units)

        if excess_units <= 0:
            return

        monetized_units = excess_units * config.TEMP_MINT_INVENTORY_RATE
        issued_money = (
            monetized_units
            *
            self.price
            *
            config.TEMP_MINT_COLLATERAL_RATIO
        )

        self.food_inventory_units = max(
            0.0,
            self.food_inventory_units - monetized_units,
        )
        self.food_inventory_value = self.food_inventory_units * self.price
        self.world.ledger.create_money_attr(
            receiver_obj=self,
            receiver_attr="cash",
            receiver_name="firm.cash",
            amount=issued_money,
            reason="temporary_inventory_backed_money_issue",
        )

        self.money_issued_this_step = issued_money
        self.inventory_monetized_units_this_step = monetized_units
        self.cumulative_money_issued += issued_money
        self.cumulative_inventory_monetized_units += monetized_units

    def step(self):
        self.central_bank.reset_step()
        self.central_bank.collect_public_wealth(self.world)
        self.working_capital_loan_issued_this_step = 0.0
        self.working_capital_loan_repaid_this_step = 0.0
        self.working_capital_interest_paid_this_step = 0.0
        self.working_capital_target_cash = 0.0
        self.working_capital_funding_gap = 0.0
        opening_food_inventory_units = self.food_inventory_units
        opening_central_bank_food_inventory_units = (
            self.central_bank.food_inventory_units
        )
        # Newborns and other newly active workers can enter between two
        # weekly firm settlements.  Assign them before payroll planning so
        # scheduled payroll, credit capacity, and actual wage transfers use
        # exactly the same worker set.
        if len(getattr(self.world, "firms", [])) > 1:
            self.world.assign_unassigned_workers_to_firms()
        total_labor = self.total_labor()
        self.wage_bill = self.base_wage_bill(total_labor)
        self.scheduled_wage_bill = self.wage_bill
        pre_loan_cash = self.cash

        if len(getattr(self.world, "firms", [])) > 1:
            self.world.prepare_multi_firm_credit(
                scheduled_wage_bill=self.wage_bill,
                total_labor=total_labor,
            )

        affordable_dividend = min(
            max(0.0, pre_loan_cash - self.wage_bill),
            max(0.0, self.last_profit_before_dividend)
            *
            config.FIRM_DIVIDEND_SHARE,
        )
        self.working_capital_target_cash = (
            central_bank_config.CENTRAL_BANK_FIRM_CREDIT_CASH_BUFFER
            +
            self.wage_bill
            *
            central_bank_config.CENTRAL_BANK_FIRM_CREDIT_WAGE_BUFFER
        )
        self.working_capital_funding_gap = max(
            0.0,
            self.working_capital_target_cash - pre_loan_cash,
        )
        if len(getattr(self.world, "firms", [])) > 1:
            # Multi-firm credit was already settled per FirmSlice above.
            self.working_capital_loan_issued_this_step = 0.0
        else:
            self.working_capital_loan_issued_this_step = (
                self.central_bank.issue_firm_working_capital_loan(
                    self,
                    self.working_capital_target_cash,
                )
            )

        self.distribute_wages(self.wage_bill, total_labor)
        if hasattr(self.world, "record_payg_cash_safety_phase"):
            self.world.record_payg_cash_safety_phase("after_payroll")

        self.dividend_paid = self.distribute_dividends(affordable_dividend)
        if hasattr(self.world, "record_payg_cash_safety_phase"):
            self.world.record_payg_cash_safety_phase("after_dividends")

        private_support_result = (
            self.world.private_family_support_system.execute(
                self.world.current_step_index
            )
            if hasattr(self.world, "private_family_support_system")
            else None
        )
        self.private_support_result = private_support_result
        if hasattr(self.world, "record_payg_cash_safety_phase"):
            self.world.record_payg_cash_safety_phase("after_legacy_private_support")
        active_social_result = self.world.apply_active_social_policy(
            self.world.current_step_index
        )

        self.scheduled_productive_capacity = (
            total_labor * self.food_productivity * self.production_scale
        )
        self.funded_productive_capacity = (
            getattr(self.world, "multi_firm_funded_capacity_total", None)
            if len(getattr(self.world, "firms", [])) > 1
            else self.scheduled_productive_capacity
        )
        if self.funded_productive_capacity is None:
            self.funded_productive_capacity = self.scheduled_productive_capacity
        self.food_output_units = (
            self.funded_productive_capacity * self.production_scale
        )
        # Preserve the physical supply used by this period's legacy market so
        # a fresh multi-firm bootstrap can hand it off without recreating it.
        pre_market_inventory_units = max(0.0, self.food_inventory_units)
        pre_market_production_units = max(0.0, self.food_output_units)
        self.pre_market_inventory_units = pre_market_inventory_units
        self.pre_market_production_units = pre_market_production_units
        self.pre_market_supply_units = (
            pre_market_inventory_units + pre_market_production_units
        )
        self.real_output_value = self.food_output_units
        self.nominal_output_value = self.food_output_units * self.price
        self.update_unit_labor_cost()

        demand_units, paid_money = self.run_household_food_market()
        self.food_demand_units = demand_units
        self.inventory_demand_ratio = safe_ratio(
            self.food_inventory_units,
            max(1.0, self.food_demand_units),
        )

        households = self.active_households()
        self.central_bank.distribute_poverty_food_subsidy(
            households,
            self.world.household_planning_price_this_step(),
        )
        for household in households:
            household.real_food_consumption_units_this_step = (
                getattr(household, "food_desired_units_this_step", 0.0)
                +
                getattr(household, "food_subsidy_units_this_step", 0.0)
            )
            household.food_constrained_after_subsidy_this_step = (
                household.real_food_consumption_units_this_step
                <
                getattr(household, "food_need_units_this_step", 0.0)
            )

        firm_supply_before_release = self.food_output_units + self.food_inventory_units
        released_units = self.central_bank.release_to_market(self)
        available_food_units = self.food_output_units + self.food_inventory_units
        self.update_demand_pressure(available_food_units)
        self.food_sales_units = min(demand_units, available_food_units)
        self.unmet_food_demand_units = max(
            0.0,
            demand_units - self.food_sales_units,
        )
        fulfillment_ratio = safe_ratio(
            self.food_sales_units,
            demand_units,
        )
        self.refund_unfilled_food_orders(fulfillment_ratio)
        if hasattr(self.world, "record_payg_cash_safety_phase"):
            self.world.record_payg_cash_safety_phase("after_consumption")
        if hasattr(self.world, "finalize_active_social_recipient_observation"):
            self.world.finalize_active_social_recipient_observation(
                self.world.current_step_index
            )
        if (
            getattr(self.world, "private_family_support_system", None) is not None
        ):
            self.world.private_family_support_system.finalize_after_consumption()
        total_market_revenue = self.food_sales_units * self.price
        release_revenue = (
            total_market_revenue
            *
            safe_ratio(
                released_units,
                firm_supply_before_release + released_units,
            )
        )
        self.central_bank.record_market_release_revenue(
            release_revenue,
            ledger=self.world.ledger,
            payer_obj=self,
            payer_attr="cash",
            payer_name="firm.cash",
        )
        self.sales_revenue = total_market_revenue - release_revenue
        self.world.consumption_system.total_consumption = total_market_revenue

        spoilage_units = (
            self.food_inventory_units
            *
            self.product.perish_rate
        )
        self.food_inventory_units = max(
            0.0,
            self.food_inventory_units
            +
            self.food_output_units
            -
            self.food_sales_units
            -
            spoilage_units,
        )
        self.food_inventory_value = self.food_inventory_units * self.price
        self.inventory_depreciation = spoilage_units * self.price
        unsold_output_value = max(
            0.0,
            self.food_output_units - self.food_sales_units,
        ) * self.price
        self.profit_before_dividend = (
            self.sales_revenue
            +
            unsold_output_value
            -
            self.executed_wage_bill
            -
            self.inventory_depreciation
        )
        self.last_profit_before_dividend = self.profit_before_dividend
        self.update_expected_sales(self.sales_revenue)

        household_income = (
            self.executed_wage_bill
            + self.household_dividend_paid_this_step
        )
        self.sales_income_ratio = safe_ratio(
            self.sales_revenue,
            household_income,
        )

        self.update_price()
        self.food_inventory_value = self.food_inventory_units * self.price
        self.update_production_scale()
        if len(getattr(self.world, "firms", [])) <= 1:
            self.central_bank.repay_firm_working_capital_loan(self)
        central_bank_purchase = self.central_bank.purchase_excess_food_inventory(self)
        self.profit_before_dividend += central_bank_purchase
        self.last_profit_before_dividend = self.profit_before_dividend
        self.sales_income_ratio = safe_ratio(
            self.sales_revenue + central_bank_purchase,
            household_income,
        )
        self.cash_history.append(self.cash)
        self.cash_slope_signal = self.recent_cash_slope()
        self.inventory_demand_ratio = safe_ratio(
            self.food_inventory_units,
            max(1.0, self.food_demand_units),
        )
        self.money_issued_this_step = self.central_bank.money_issued_this_step
        self.working_capital_loan_balance = self.central_bank.firm_loan_balance
        self.working_capital_loan_repaid_this_step = (
            self.central_bank.loan_principal_repaid_this_step
        )
        self.working_capital_interest_paid_this_step = (
            self.central_bank.loan_interest_paid_this_step
        )
        self.inventory_monetized_units_this_step = (
            self.central_bank.food_purchase_units_this_step
        )
        self.cumulative_money_issued = self.central_bank.money_supply
        self.cumulative_inventory_monetized_units += (
            self.central_bank.food_purchase_units_this_step
        )
        self.central_bank_inventory_purchase = (
            self.central_bank.inventory_purchase_value_this_step
        )
        self.central_bank_market_release_revenue = (
            self.central_bank.market_release_revenue_this_step
        )
        self.central_bank_market_release_money_destroyed = (
            self.central_bank.market_release_money_destroyed_this_step
        )
        self.central_bank_market_release_public_income = (
            self.central_bank.market_release_public_income_this_step
        )
        self.central_bank_poverty_subsidy_value = (
            self.central_bank.poverty_subsidy_value_this_step
        )
        self.central_bank_public_income = (
            self.central_bank.public_income_this_step
        )
        self.central_bank_public_income_used = (
            self.central_bank.public_income_used_this_step
        )
        self.central_bank_public_income_balance = (
            self.central_bank.public_income_balance
        )
        self.central_bank_food_inventory_units = (
            self.central_bank.food_inventory_units
        )
        self.central_bank_food_purchase_units = (
            self.central_bank.food_purchase_units_this_step
        )
        self.central_bank_food_release_units = (
            self.central_bank.food_release_units_this_step
        )
        self.central_bank_food_subsidy_units = (
            self.central_bank.food_subsidy_units_this_step
        )

        return {
            "good_id": self.product_id,
            "good_name": self.product.name,
            "total_labor": total_labor,
            "real_output_value": self.real_output_value,
            "nominal_output_value": self.nominal_output_value,
            "wage_bill": self.wage_bill,
            "scheduled_wage_bill": self.wage_bill,
            "executed_wage_bill": self.executed_wage_bill,
            "wage_payment": self.executed_wage_bill,
            "dividend": self.dividend_paid,
            "household_dividend": self.household_dividend_paid_this_step,
            "person_dividend_paid": self.person_dividend_paid,
            "private_support_paid": (
                getattr(self.private_support_result, "total_paid", 0.0)
                if self.private_support_result is not None else 0.0
            ),
            "private_support_received": (
                getattr(self.private_support_result, "total_received", 0.0)
                if self.private_support_result is not None else 0.0
            ),
            "private_support_transfer_count": (
                getattr(self.private_support_result, "transfer_count", 0)
                if self.private_support_result is not None else 0
            ),
            "payg_contribution": active_social_result.get("payg_contribution", 0.0),
            "pension_paid": active_social_result.get("pension_paid", 0.0),
            "wealth_transfer_paid": active_social_result.get("wealth_transfer_paid", 0.0),
            "wealth_transfer_received": active_social_result.get("wealth_transfer_received", 0.0),
            "legacy_dividend_entitlement": self.legacy_dividend_entitlement,
            "dividend_routing_gap": self.dividend_routing_gap,
            "sales": total_market_revenue,
            "firm_sales_revenue": self.sales_revenue,
            "firm_cash": self.cash,
            "firm_inventory": self.food_inventory_value,
            "firm_net_worth": self.net_worth(),
            "profit_before_dividend": self.profit_before_dividend,
            "food_price": self.price,
            "unit_labor_cost": self.unit_labor_cost,
            "unit_labor_cost_growth": self.unit_labor_cost_growth,
            "price_inventory_gap": self.price_inventory_gap,
            "price_cost_growth": self.price_cost_growth,
            "price_inflation_signal": self.price_inflation_signal,
            "price_log_adjustment": self.price_log_adjustment,
            "demand_pressure": self.demand_pressure,
            "available_food_supply_units": self.available_food_supply_units,
            "sales_income_ratio": self.sales_income_ratio,
            "cash_slope_signal": self.cash_slope_signal,
            "food_output_units": self.food_output_units,
            "scheduled_productive_capacity": self.scheduled_productive_capacity,
            "funded_productive_capacity": self.funded_productive_capacity,
            "food_demand_units": self.food_demand_units,
            "food_sales_units": self.food_sales_units,
            "food_inventory_units": self.food_inventory_units,
            "opening_food_inventory_units": opening_food_inventory_units,
            "pre_market_inventory_units": pre_market_inventory_units,
            "pre_market_production_units": pre_market_production_units,
            "pre_market_supply_units": self.pre_market_supply_units,
            "opening_central_bank_food_inventory_units": (
                opening_central_bank_food_inventory_units
            ),
            "food_spoilage_units": spoilage_units,
            "inventory_demand_ratio": self.inventory_demand_ratio,
            "unmet_food_demand_units": self.unmet_food_demand_units,
            "money_issued": self.money_issued_this_step,
            "working_capital_loan_issued": (
                self.working_capital_loan_issued_this_step
            ),
            "working_capital_target_cash": self.working_capital_target_cash,
            "working_capital_funding_gap": self.working_capital_funding_gap,
            "target_cash": self.working_capital_target_cash,
            "funding_gap": self.working_capital_funding_gap,
            "loan_issued": self.working_capital_loan_issued_this_step,
            "working_capital_loan_repaid": (
                self.working_capital_loan_repaid_this_step
            ),
            "working_capital_interest_paid": (
                self.working_capital_interest_paid_this_step
            ),
            "working_capital_loan_balance": (
                self.working_capital_loan_balance
            ),
            "loan_balance": self.working_capital_loan_balance,
            "inventory_monetized_units": self.inventory_monetized_units_this_step,
            "cumulative_money_issued": self.cumulative_money_issued,
            "cumulative_inventory_monetized_units": (
                self.cumulative_inventory_monetized_units
            ),
            "central_bank_inventory_purchase": self.central_bank_inventory_purchase,
            "central_bank_market_release_revenue": (
                self.central_bank_market_release_revenue
            ),
            "central_bank_market_release_money_destroyed": (
                self.central_bank_market_release_money_destroyed
            ),
            "central_bank_market_release_public_income": (
                self.central_bank_market_release_public_income
            ),
            "central_bank_poverty_subsidy_value": (
                self.central_bank_poverty_subsidy_value
            ),
            "central_bank_public_income": self.central_bank_public_income,
            "central_bank_public_income_used": (
                self.central_bank_public_income_used
            ),
            "central_bank_public_income_balance": (
                self.central_bank_public_income_balance
            ),
            "central_bank_net_money_issued": (
                self.money_issued_this_step
            ),
            "central_bank_food_inventory_units": (
                self.central_bank_food_inventory_units
            ),
            "central_bank_food_purchase_units": self.central_bank_food_purchase_units,
            "central_bank_food_release_units": self.central_bank_food_release_units,
            "central_bank_food_subsidy_units": self.central_bank_food_subsidy_units,
        }



