import math
import random
from dataclasses import dataclass

import numpy as np

from central_bank import CentralBank
from economy import config as economy_config
from productivity import age_productivity

import config


def safe_ratio(numerator, denominator):
    if denominator == 0:
        return 0.0

    return numerator / denominator


def deadband(value, threshold):
    if abs(value) <= threshold:
        return 0.0

    return value - np.sign(value) * threshold


@dataclass
class FoodFirm:
    id: int
    cash: float
    price: float
    productivity: float
    production_scale: float
    inventory_units: float
    target_inventory_days: float
    labor_share: float
    expected_market_share: float
    brand_preference: float
    initial_strategy_bias: float
    normal_unit_cost: float
    strategy_bias: float = 0.0
    expected_profit: float = 0.0
    output_units: float = 0.0
    demand_units: float = 0.0
    sales_units: float = 0.0
    sales_revenue: float = 0.0
    wage_bill: float = 0.0
    profit_before_dividend: float = 0.0
    inventory_depreciation: float = 0.0
    cb_purchase_value: float = 0.0
    cb_purchase_units: float = 0.0
    cb_release_units: float = 0.0
    cash_flow: float = 0.0

    @property
    def inventory_value(self):
        return self.inventory_units * self.price

    @property
    def net_worth(self):
        return self.cash + self.inventory_value

    @property
    def inventory_demand_ratio(self):
        return safe_ratio(self.inventory_units, max(1.0, self.demand_units))


class CompetingFoodFirmMarket:

    def __init__(self, world, source_firm, firm_count=config.FIRM_COUNT):
        self.world = world
        self.firm_count = firm_count
        self.price_sensitivity = config.PRICE_SENSITIVITY
        self.loyalty_weight = config.LOYALTY_WEIGHT
        self.price_adjustment_rate = config.FIRM_PRICE_ADJUSTMENT_RATE
        self.production_adjustment_rate = config.FIRM_PRODUCTION_ADJUSTMENT_RATE
        self.strategy_exploration_noise = config.STRATEGY_EXPLORATION_NOISE
        self.labor_reallocation_rate = config.LABOR_REALLOCATION_RATE
        self.profit_labor_weight = config.PROFIT_LABOR_WEIGHT
        self.strategy_adaptation_rate = config.STRATEGY_ADAPTATION_RATE
        self.central_bank = source_firm.central_bank
        self.price = source_firm.price
        self.target_inventory_days = source_firm.target_inventory_days
        self.sales_income_ratio = source_firm.sales_income_ratio
        self.last_profit_before_dividend = source_firm.last_profit_before_dividend
        self.cash_history = list(source_firm.cash_history)
        self.cash_slope_signal = source_firm.cash_slope_signal
        self.expected_sales = source_firm.expected_sales
        self.cumulative_inventory_monetized_units = (
            source_firm.cumulative_inventory_monetized_units
        )

        cash_per_firm = source_firm.cash / firm_count
        inventory_per_firm = source_firm.food_inventory_units / firm_count
        labor_share = 1.0 / firm_count
        normal_unit_cost = (
            economy_config.FIRM_WAGE_PER_LABOR
            /
            max(
                1.0,
                source_firm.food_productivity * config.FIRM_COST_ANCHOR_SCALE,
            )
        )

        self.firms = []
        if firm_count > 1:
            strategy_biases = np.linspace(
                -config.STRATEGY_BIAS_SPREAD,
                config.STRATEGY_BIAS_SPREAD,
                firm_count,
            )
        else:
            strategy_biases = [0.0]

        for firm_id in range(firm_count):
            price_noise = 1.0 + random.uniform(
                -config.INITIAL_PRICE_NOISE,
                config.INITIAL_PRICE_NOISE,
            )
            productivity_noise = 1.0 + random.uniform(
                -config.INITIAL_PRODUCTIVITY_NOISE,
                config.INITIAL_PRODUCTIVITY_NOISE,
            )
            self.firms.append(
                FoodFirm(
                    id=firm_id,
                    cash=cash_per_firm,
                    price=max(0.01, source_firm.price * price_noise),
                    productivity=source_firm.food_productivity * productivity_noise,
                    production_scale=source_firm.production_scale,
                    inventory_units=inventory_per_firm,
                    target_inventory_days=source_firm.target_inventory_days,
                    labor_share=labor_share,
                    expected_market_share=labor_share,
                    brand_preference=labor_share,
                    initial_strategy_bias=float(strategy_biases[firm_id]),
                    normal_unit_cost=normal_unit_cost,
                    strategy_bias=float(strategy_biases[firm_id]),
                )
            )

        self.reset_aggregate_state()

    def reset_aggregate_state(self):
        self.food_productivity = np.mean([firm.productivity for firm in self.firms])
        self.production_scale = np.mean([firm.production_scale for firm in self.firms])
        self.food_inventory_units = sum(firm.inventory_units for firm in self.firms)
        self.food_output_units = 0.0
        self.food_demand_units = 0.0
        self.food_sales_units = 0.0
        self.unmet_food_demand_units = 0.0
        self.inventory_demand_ratio = 0.0
        self.real_output_value = 0.0
        self.nominal_output_value = 0.0
        self.food_inventory_value = sum(firm.inventory_value for firm in self.firms)
        self.wage_bill = 0.0
        self.dividend_paid = 0.0
        self.sales_revenue = 0.0
        self.profit_before_dividend = 0.0
        self.inventory_depreciation = 0.0
        self.money_issued_this_step = 0.0
        self.inventory_monetized_units_this_step = 0.0
        self.cumulative_money_issued = self.central_bank.money_supply
        self.central_bank_inventory_purchase = 0.0
        self.central_bank_market_release_revenue = 0.0
        self.central_bank_poverty_subsidy_value = 0.0
        self.central_bank_public_income = 0.0
        self.central_bank_public_income_used = 0.0
        self.central_bank_public_income_balance = 0.0
        self.central_bank_food_inventory_units = self.central_bank.food_inventory_units
        self.central_bank_food_purchase_units = 0.0
        self.central_bank_food_release_units = 0.0
        self.central_bank_food_subsidy_units = 0.0

    def active_households(self):
        return [
            household
            for household in self.world.households
            if household.size() > 0
        ]

    def total_labor(self):
        return sum(
            age_productivity(person.age)
            for person in self.world.population
            if person.alive
            and
            self.world.get_household(person.household_id) is not None
        )

    def distribute_wages(self, wage_bill, total_labor):
        for household in self.world.households:
            household.income_this_step = 0.0

        if total_labor <= 0:
            return

        for person in self.world.population:
            if not person.alive:
                continue

            household = self.world.get_household(person.household_id)

            if household is None:
                continue

            household.income_this_step += (
                wage_bill
                *
                age_productivity(person.age)
                /
                total_labor
            )

    def distribute_dividends(self, dividend):
        households = self.active_households()

        if dividend <= 0 or not households:
            return 0.0

        dividend_per_household = dividend / len(households)

        for household in households:
            household.income_this_step += dividend_per_household

        return dividend

    def base_wage_bill(self, total_labor):
        base_wage = total_labor * economy_config.FIRM_WAGE_PER_LABOR
        sales_adjustment = max(
            0.80,
            min(
                1.15,
                1.0
                +
                economy_config.FIRM_WAGE_SALES_FEEDBACK
                *
                (
                    self.sales_income_ratio
                    -
                    economy_config.FOOD_TARGET_SALES_INCOME_RATIO
                ),
            ),
        )
        cash_adjustment = max(
            0.80,
            min(
                1.10,
                1.0
                -
                economy_config.FIRM_WAGE_CASH_FEEDBACK
                *
                (
                    -self.cash_slope_signal
                    /
                    max(1.0, abs(self.cash()))
                ),
            ),
        )

        return base_wage * sales_adjustment * cash_adjustment

    def cash(self):
        return sum(firm.cash for firm in self.firms)

    def net_worth(self):
        return sum(firm.net_worth for firm in self.firms)

    def representative_price(self):
        total_sales = sum(firm.sales_units for firm in self.firms)

        if total_sales > 0:
            return sum(firm.price * firm.sales_units for firm in self.firms) / total_sales

        return float(np.mean([firm.price for firm in self.firms]))

    def household_food_need_units(self, household):
        needs_system = getattr(self.world, "needs_system", None)

        if needs_system is not None:
            return needs_system.household_minimum_need_units(household)

        return self.world.consumption_system.household_necessary_consumption(
            household
        )

    def expected_price_for_household(self):
        weights = self.firm_choice_weights()
        return sum(firm.price * weight for firm, weight in zip(self.firms, weights))

    def household_food_purchase_units(self, household, need_units, expected_price):
        household.food_subsidy_units_this_step = 0.0
        household.food_subsidy_value_this_step = 0.0

        household.necessary_consumption_this_step = need_units * expected_price
        discretionary_money = (
            max(
                0.0,
                household.income_this_step
                -
                household.necessary_consumption_this_step,
            )
            *
            economy_config.DISCRETIONARY_INCOME_RATE
        )
        wealth_food_money = (
            household.wealth
            *
            economy_config.NEW_WEALTH_CONSUMPTION_RATE
        )
        desired_money = (
            household.necessary_consumption_this_step
            +
            discretionary_money
            +
            wealth_food_money
        )
        affordable_money = (
            household.income_this_step
            +
            household.wealth
            *
            economy_config.WEALTH_DRAWDOWN_RATE
        )
        actual_money = max(0.0, min(desired_money, affordable_money))
        actual_units = actual_money / max(0.01, expected_price)

        household.desired_consumption_this_step = desired_money
        household.affordable_consumption_this_step = affordable_money
        household.consumption_this_step = actual_money
        household.saving_this_step = household.income_this_step - actual_money
        household.wealth += household.saving_this_step

        if household.wealth < 0:
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
        household.real_food_consumption_units_this_step = actual_units
        household.food_need_gap_units_this_step = max(0.0, need_units - actual_units)
        household.food_constrained_after_subsidy_this_step = (
            household.food_need_gap_units_this_step > 0
        )

        return actual_units, actual_money

    def firm_choice_weights(self):
        prices = np.array([firm.price for firm in self.firms], dtype=float)
        inventory = np.array(
            [
                max(0.0, firm.output_units + firm.inventory_units)
                for firm in self.firms
            ],
            dtype=float,
        )
        price_score = np.exp(
            -self.price_sensitivity
            *
            (prices / max(0.01, np.mean(prices)) - 1.0)
        )
        mean_inventory = max(1.0, float(np.mean(inventory)))
        availability_score = np.clip(inventory / mean_inventory, 0.50, 1.10) ** 0.20
        loyalty_score = np.array(
            [
                max(0.001, firm.brand_preference)
                for firm in self.firms
            ],
            dtype=float,
        )
        scores = (
            price_score
            *
            availability_score
            *
            (
                (1.0 - self.loyalty_weight)
                +
                self.loyalty_weight
                *
                loyalty_score
                *
                self.firm_count
            )
        )
        total = float(np.sum(scores))

        if total <= 0:
            return [1.0 / self.firm_count] * self.firm_count

        return list(scores / total)

    def run_household_food_market(self):
        expected_price = self.expected_price_for_household()
        demand_units = 0.0
        paid_money = 0.0
        constrained = 0
        consumed = 0

        for household in self.world.households:
            need_units = self.household_food_need_units(household)

            if household.size() > 0:
                consumed += 1

            units, money = self.household_food_purchase_units(
                household,
                need_units,
                expected_price,
            )
            demand_units += units
            paid_money += money

            if getattr(household, "budget_constrained_this_step", False):
                constrained += 1

        consumption = self.world.consumption_system
        consumption.total_consumption = paid_money
        consumption.constrained_household_count = constrained
        consumption.consumed_household_count = consumed
        consumption.constrained_household_ratio = safe_ratio(constrained, consumed)
        consumption.capped_household_count = constrained
        consumption.capped_household_ratio = consumption.constrained_household_ratio

        return demand_units, paid_money

    def allocate_sales(self, demand_units):
        remaining_demand = demand_units
        weights = self.firm_choice_weights()
        desired_by_firm = [
            demand_units * weight
            for weight in weights
        ]
        supplies = [
            firm.output_units + firm.inventory_units
            for firm in self.firms
        ]
        sales_units = [
            min(desired, supply)
            for desired, supply in zip(desired_by_firm, supplies)
        ]
        remaining_demand -= sum(sales_units)

        # If preferred firms sell out, unmet customers spill over to firms that
        # still have inventory. This is still price-weighted, not cheapest-only.
        for _ in range(self.firm_count):
            if remaining_demand <= 1e-9:
                break

            residual_capacity = [
                max(0.0, supply - sales)
                for supply, sales in zip(supplies, sales_units)
            ]
            total_capacity = sum(residual_capacity)

            if total_capacity <= 0:
                break

            residual_scores = [
                weights[index] * residual_capacity[index]
                for index in range(self.firm_count)
            ]
            score_total = sum(residual_scores)

            if score_total <= 0:
                residual_scores = residual_capacity
                score_total = total_capacity

            added_total = 0.0

            for index, score in enumerate(residual_scores):
                add_units = min(
                    residual_capacity[index],
                    remaining_demand * score / score_total,
                )
                sales_units[index] += add_units
                added_total += add_units

            remaining_demand -= added_total

        return sales_units, max(0.0, demand_units - sum(sales_units)), desired_by_firm

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
            household.wealth += refund
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

    def settle_household_payments_to_revenue(self, target_revenue):
        current_payment = sum(
            getattr(household, "consumption_this_step", 0.0)
            for household in self.world.households
        )
        delta = target_revenue - current_payment

        if abs(delta) <= 1e-9:
            return current_payment

        payers = [
            household
            for household in self.world.households
            if getattr(household, "consumption_this_step", 0.0) > 0
        ]

        if not payers:
            return current_payment

        if delta < 0:
            refund_total = -delta

            for household in payers:
                share = (
                    household.consumption_this_step
                    /
                    max(1.0, current_payment)
                )
                refund = refund_total * share
                household.consumption_this_step -= refund
                household.saving_this_step += refund
                household.wealth += refund

            return target_revenue

        remaining_delta = delta
        base_payment = max(1.0, current_payment)

        for household in payers:
            extra_charge = delta * household.consumption_this_step / base_payment
            actual_charge = min(extra_charge, household.wealth)

            if actual_charge <= 0:
                continue

            household.consumption_this_step += actual_charge
            household.saving_this_step -= actual_charge
            household.wealth -= actual_charge
            remaining_delta -= actual_charge

        return target_revenue - max(0.0, remaining_delta)

    def distribute_poverty_food_subsidy(self):
        households = self.active_households()
        available_units = (
            self.central_bank.food_inventory_units
            *
            config.CB_POVERTY_SUBSIDY_RATE
        )
        needy_households = [
            household
            for household in households
            if getattr(household, "food_need_gap_units_this_step", 0.0) > 0
        ]
        total_gap = sum(
            getattr(household, "food_need_gap_units_this_step", 0.0)
            for household in needy_households
        )

        if available_units <= 0 or total_gap <= 0:
            return 0.0

        fill = min(1.0, available_units / total_gap)
        used_units = 0.0
        price = self.representative_price()

        for household in needy_households:
            subsidy_units = (
                getattr(household, "food_need_gap_units_this_step", 0.0)
                *
                fill
            )
            household.food_subsidy_units_this_step = subsidy_units
            household.food_subsidy_value_this_step = subsidy_units * price
            used_units += subsidy_units

        self.central_bank.food_inventory_units -= used_units
        self.central_bank.food_subsidy_units_this_step = used_units
        self.central_bank.poverty_subsidy_value_this_step = used_units * price

        return self.central_bank.poverty_subsidy_value_this_step

    def release_central_bank_inventory(self):
        firm_supply = sum(
            firm.output_units + firm.inventory_units
            for firm in self.firms
        )
        current_shortage = max(0.0, self.food_demand_units - firm_supply)
        demand_base = max(1.0, self.food_demand_units)
        inventory_pressure = safe_ratio(
            max(0.0, self.target_inventory_days - self.inventory_demand_ratio),
            max(1.0, self.target_inventory_days),
        )
        price_pressure = safe_ratio(
            max(0.0, self.representative_price() - config.CB_PRICE_ANCHOR),
            max(0.01, config.CB_PRICE_ANCHOR),
        )
        release_units = min(
            self.central_bank.food_inventory_units,
            current_shortage
            +
            demand_base
            *
            config.CB_RELEASE_RATE
            *
            max(inventory_pressure, price_pressure),
        )

        if release_units <= 0:
            return 0.0

        self.central_bank.food_inventory_units -= release_units
        self.central_bank.food_release_units_this_step = release_units

        weights = self.firm_choice_weights()
        for firm, weight in zip(self.firms, weights):
            firm.cb_release_units = release_units * weight
            firm.inventory_units += firm.cb_release_units

        return release_units

    def purchase_excess_inventory(self):
        planned_purchases = []
        planned_purchase_value = 0.0

        for firm in self.firms:
            target_cash = (
                economy_config.FIRM_INITIAL_CASH
                *
                config.CB_TARGET_FIRM_CASH_SHARE
                /
                self.firm_count
            )
            cash_buffer = max(
                1.0,
                economy_config.FIRM_INITIAL_CASH
                *
                config.CB_FIRM_CASH_BUFFER_SHARE
                /
                self.firm_count,
            )
            cash_gate = safe_ratio(
                target_cash + cash_buffer - firm.cash,
                cash_buffer,
            )
            cash_gate = max(0.0, min(1.0, cash_gate))

            if cash_gate <= 0:
                continue

            demand_base = max(1.0, firm.demand_units)
            threshold_units = (
                firm.target_inventory_days
                +
                config.CB_EXCESS_INVENTORY_DAYS
            ) * demand_base
            excess_units = max(0.0, firm.inventory_units - threshold_units)
            purchase_units = min(
                firm.inventory_units,
                excess_units * config.CB_PURCHASE_RATE * cash_gate,
            )

            if purchase_units <= 0:
                continue

            purchase_value = (
                purchase_units
                *
                firm.price
                *
                config.CB_PURCHASE_PRICE_HAIRCUT
            )
            planned_purchases.append((firm, purchase_units, purchase_value))
            planned_purchase_value += purchase_value

        if planned_purchase_value <= 0:
            return 0.0

        available_purchase_budget = (
            self.central_bank.public_income_balance
            +
            self.central_bank.market_release_revenue_this_step
            +
            config.CB_MAX_NEW_MONEY_ISSUE_PER_STEP
        )
        purchase_scale = min(
            1.0,
            safe_ratio(available_purchase_budget, planned_purchase_value),
        )

        total_purchase_value = 0.0
        total_purchase_units = 0.0

        for firm, planned_units, planned_value in planned_purchases:
            purchase_units = planned_units * purchase_scale
            purchase_value = planned_value * purchase_scale

            if purchase_units <= 0 or purchase_value <= 0:
                continue

            firm.inventory_units -= purchase_units
            firm.cb_purchase_units = purchase_units
            firm.cb_purchase_value = purchase_value
            firm.cash += purchase_value
            self.central_bank.food_inventory_units += purchase_units
            total_purchase_value += purchase_value
            total_purchase_units += purchase_units

        if total_purchase_value <= 0:
            return 0.0

        public_income_used = min(
            self.central_bank.public_income_balance,
            total_purchase_value,
        )
        remaining_purchase_value = total_purchase_value - public_income_used
        recycled_release_revenue = min(
            self.central_bank.market_release_revenue_this_step,
            remaining_purchase_value,
        )
        newly_issued_money = remaining_purchase_value - recycled_release_revenue
        newly_issued_money = min(
            newly_issued_money,
            config.CB_MAX_NEW_MONEY_ISSUE_PER_STEP,
        )
        self.central_bank.public_income_balance -= public_income_used
        self.central_bank.public_income_used_this_step = public_income_used
        self.central_bank.money_issued_this_step = newly_issued_money
        self.central_bank.money_supply += recycled_release_revenue + newly_issued_money
        self.central_bank.inventory_purchase_value_this_step = total_purchase_value
        self.central_bank.food_purchase_units_this_step = total_purchase_units

        return total_purchase_value

    def recent_cash_slope(self):
        recent = self.cash_history[-economy_config.FOOD_PRICE_CASH_WINDOW:]
        n = len(recent)

        if n < 2:
            return 0.0

        x = np.arange(n, dtype=float)
        return float(np.polyfit(x, np.asarray(recent, dtype=float), 1)[0])

    def update_firm_expectations_and_strategy(self):
        total_sales = max(1.0, sum(firm.sales_units for firm in self.firms))
        total_positive_profit = sum(
            max(0.0, firm.profit_before_dividend)
            for firm in self.firms
        )
        attractiveness = []

        for firm in self.firms:
            market_share = firm.sales_units / total_sales
            firm.expected_market_share = (
                config.FIRM_MARKET_SHARE_ALPHA
                *
                market_share
                +
                (1 - config.FIRM_MARKET_SHARE_ALPHA)
                *
                firm.expected_market_share
            )
            firm.expected_profit = (
                config.FIRM_PROFIT_ALPHA
                *
                firm.profit_before_dividend
                +
                (1 - config.FIRM_PROFIT_ALPHA)
                *
                firm.expected_profit
            )
            profit_rate = safe_ratio(
                firm.expected_profit,
                max(1.0, firm.wage_bill),
            )
            target_cash = (
                economy_config.FIRM_INITIAL_CASH
                *
                config.CB_TARGET_FIRM_CASH_SHARE
                /
                self.firm_count
            )
            cash_pressure = safe_ratio(
                target_cash - firm.cash,
                max(1.0, target_cash),
            )
            adaptation_signal = (
                deadband(profit_rate, config.STRATEGY_PROFIT_RATE_DEADBAND)
                -
                config.STRATEGY_CASH_PRESSURE_WEIGHT
                *
                max(0.0, cash_pressure)
            )
            firm.strategy_bias += (
                self.strategy_adaptation_rate
                *
                adaptation_signal
            )
            firm.strategy_bias = max(
                config.STRATEGY_MIN_BIAS,
                min(config.STRATEGY_MAX_BIAS, firm.strategy_bias),
            )
            inventory_gap = safe_ratio(
                firm.target_inventory_days * max(1.0, firm.demand_units)
                -
                firm.inventory_units,
                max(1.0, firm.target_inventory_days * max(1.0, firm.demand_units)),
            )
            inventory_signal = deadband(
                float(np.tanh(inventory_gap)),
                config.FIRM_INVENTORY_GAP_DEADBAND,
            )
            sales_gap = deadband(
                market_share - (1.0 / self.firm_count),
                config.FIRM_MARKET_SHARE_DEADBAND,
            )
            unit_cost = firm.wage_bill / max(1.0, firm.output_units)
            unbiased_normal_target_price = (
                firm.normal_unit_cost
                *
                config.FIRM_TARGET_MARKUP
            )
            normal_target_price = (
                unbiased_normal_target_price
                *
                (1.0 + firm.strategy_bias)
            )
            current_target_price = (
                unit_cost
                *
                config.FIRM_TARGET_MARKUP
                *
                (1.0 + firm.strategy_bias)
            )
            target_price = (
                config.FIRM_PRICE_COST_ANCHOR_WEIGHT
                *
                normal_target_price
                +
                (1.0 - config.FIRM_PRICE_COST_ANCHOR_WEIGHT)
                *
                current_target_price
            )
            target_price = max(
                target_price,
                unbiased_normal_target_price
                *
                config.FIRM_NORMAL_PRICE_FLOOR_FACTOR,
            )
            margin_gap = deadband(
                safe_ratio(target_price - firm.price, max(0.01, target_price)),
                config.FIRM_MARGIN_GAP_DEADBAND,
            )
            inventory_price_signal = inventory_signal
            normal_price_floor = (
                unbiased_normal_target_price
                *
                config.FIRM_NORMAL_PRICE_FLOOR_FACTOR
            )
            if firm.price <= normal_price_floor and inventory_price_signal < 0.0:
                inventory_price_signal *= config.FIRM_BELOW_COST_INVENTORY_PRICE_WEIGHT

            price_adjustment = (
                self.price_adjustment_rate
                *
                (
                    config.FIRM_PRICE_INVENTORY_WEIGHT * inventory_price_signal
                    +
                    config.FIRM_PRICE_SHARE_WEIGHT * sales_gap
                    +
                    config.FIRM_PRICE_MARGIN_WEIGHT * margin_gap
                )
            )
            price_adjustment += random.uniform(
                -self.strategy_exploration_noise,
                self.strategy_exploration_noise,
            )
            firm.price *= max(
                1.0 - config.MAX_PRICE_CHANGE_PER_STEP,
                min(1.0 + config.MAX_PRICE_CHANGE_PER_STEP, 1.0 + price_adjustment),
            )
            firm.price = max(
                config.MIN_FIRM_PRICE,
                min(config.MAX_FIRM_PRICE, firm.price),
            )

            production_adjustment = (
                self.production_adjustment_rate
                *
                (
                    config.FIRM_PRODUCTION_INVENTORY_WEIGHT * inventory_signal
                    +
                    config.FIRM_PRODUCTION_PROFIT_WEIGHT
                    *
                    safe_ratio(firm.expected_profit, max(1.0, firm.cash))
                    -
                    0.8 * firm.strategy_bias
                )
            )
            production_adjustment += random.uniform(
                -self.strategy_exploration_noise,
                self.strategy_exploration_noise,
            )
            firm.production_scale *= max(
                1.0 - config.MAX_PRODUCTION_CHANGE_PER_STEP,
                min(1.0 + config.MAX_PRODUCTION_CHANGE_PER_STEP, 1.0 + production_adjustment),
            )
            firm.production_scale = max(
                config.MIN_PRODUCTION_SCALE,
                min(config.MAX_PRODUCTION_SCALE, firm.production_scale),
            )
            profit_score = safe_ratio(
                max(0.0, firm.profit_before_dividend),
                max(1.0, total_positive_profit),
            )
            attractiveness.append(
                max(
                    0.001,
                    (1 - self.profit_labor_weight) * firm.expected_market_share
                    +
                    self.profit_labor_weight * profit_score,
                )
            )

        if self.labor_reallocation_rate > 0:
            total_attractiveness = sum(attractiveness)
            target_shares = [
                score / total_attractiveness
                for score in attractiveness
            ]

            for firm, target_share in zip(self.firms, target_shares):
                firm.labor_share = (
                    (1 - self.labor_reallocation_rate)
                    *
                    firm.labor_share
                    +
                    self.labor_reallocation_rate
                    *
                    target_share
                )

        total_share = sum(firm.labor_share for firm in self.firms)
        for firm in self.firms:
            firm.labor_share /= total_share

    def step(self):
        self.central_bank.reset_step()
        self.central_bank.collect_public_wealth(self.world)
        total_labor = self.total_labor()
        self.wage_bill = self.base_wage_bill(total_labor)
        self.distribute_wages(self.wage_bill, total_labor)

        affordable_dividend = min(
            max(0.0, self.cash()),
            max(0.0, self.last_profit_before_dividend)
            *
            economy_config.FIRM_DIVIDEND_SHARE,
        )
        self.dividend_paid = self.distribute_dividends(affordable_dividend)

        if self.dividend_paid > 0:
            total_cash = max(1.0, self.cash())

            for firm in self.firms:
                firm.cash -= self.dividend_paid * safe_ratio(
                    max(0.0, firm.cash),
                    total_cash,
                )

        for firm in self.firms:
            firm.cash_flow = 0.0
            firm.cash_before_step = firm.cash
            firm.output_units = (
                total_labor
                *
                firm.labor_share
                *
                firm.productivity
                *
                firm.production_scale
            )
            firm.wage_bill = self.wage_bill * firm.labor_share
            firm.demand_units = 0.0
            firm.sales_units = 0.0
            firm.sales_revenue = 0.0
            firm.cb_purchase_value = 0.0
            firm.cb_purchase_units = 0.0
            firm.cb_release_units = 0.0

        self.food_output_units = sum(firm.output_units for firm in self.firms)
        self.food_inventory_units = sum(firm.inventory_units for firm in self.firms)
        self.food_demand_units, paid_money = self.run_household_food_market()
        self.inventory_demand_ratio = safe_ratio(
            self.food_inventory_units,
            max(1.0, self.food_demand_units),
        )
        self.release_central_bank_inventory()

        sales_units, unmet_units, desired_by_firm = self.allocate_sales(self.food_demand_units)
        self.food_sales_units = sum(sales_units)
        self.unmet_food_demand_units = unmet_units
        fulfillment_ratio = safe_ratio(
            self.food_sales_units,
            self.food_demand_units,
        )
        self.refund_unfilled_food_orders(fulfillment_ratio)

        total_revenue = 0.0
        release_revenue = 0.0
        for firm, sales_units_for_firm, desired_units_for_firm in zip(
            self.firms,
            sales_units,
            desired_by_firm,
        ):
            firm.demand_units = desired_units_for_firm
            firm.sales_units = sales_units_for_firm
            firm.sales_revenue = sales_units_for_firm * firm.price
            total_supply = firm.output_units + firm.inventory_units
            release_revenue += (
                firm.sales_revenue
                *
                safe_ratio(firm.cb_release_units, total_supply)
            )
            firm.inventory_units = max(
                0.0,
                firm.inventory_units
                +
                firm.output_units
                -
                firm.sales_units,
            )
            spoilage_units = (
                firm.inventory_units
                *
                economy_config.FOOD_INVENTORY_SPOILAGE_RATE
            )
            firm.inventory_units = max(0.0, firm.inventory_units - spoilage_units)
            firm.inventory_depreciation = spoilage_units * firm.price
            firm.cash += firm.sales_revenue - firm.wage_bill
            firm.profit_before_dividend = (
                firm.sales_revenue
                +
                max(0.0, firm.output_units - firm.sales_units) * firm.price
                -
                firm.wage_bill
                -
                firm.inventory_depreciation
            )
            total_revenue += firm.sales_revenue

        gross_revenue = total_revenue
        collected_revenue = self.settle_household_payments_to_revenue(gross_revenue)

        if collected_revenue < gross_revenue:
            collection_ratio = safe_ratio(collected_revenue, gross_revenue)

            for firm in self.firms:
                uncollected = firm.sales_revenue * (1.0 - collection_ratio)
                firm.sales_revenue *= collection_ratio
                firm.cash -= uncollected

            release_revenue *= collection_ratio
            total_revenue = collected_revenue

        if release_revenue > 0:
            self.central_bank.market_release_revenue_this_step += release_revenue
            self.central_bank.money_supply -= release_revenue
            total_revenue -= release_revenue
            # Remove central-bank release revenue proportionally from firms.
            gross_firm_revenue = sum(max(0.0, firm.sales_revenue) for firm in self.firms)
            for firm in self.firms:
                firm.cash -= release_revenue * safe_ratio(
                    max(0.0, firm.sales_revenue),
                    gross_firm_revenue,
                )

        self.sales_revenue = total_revenue
        self.world.consumption_system.total_consumption = sum(
            getattr(household, "consumption_this_step", 0.0)
            for household in self.world.households
        )

        self.distribute_poverty_food_subsidy()
        central_bank_purchase = self.purchase_excess_inventory()

        self.profit_before_dividend = sum(
            firm.profit_before_dividend
            for firm in self.firms
        ) + central_bank_purchase
        self.last_profit_before_dividend = self.profit_before_dividend
        household_income = self.wage_bill + self.dividend_paid
        self.sales_income_ratio = safe_ratio(
            self.sales_revenue + central_bank_purchase,
            household_income,
        )
        for firm in self.firms:
            firm.cash_flow = firm.cash - getattr(firm, "cash_before_step", firm.cash)

        self.cash_history.append(self.cash())
        self.cash_slope_signal = self.recent_cash_slope()
        self.update_firm_expectations_and_strategy()

        self.price = self.representative_price()
        self.food_inventory_units = sum(firm.inventory_units for firm in self.firms)
        self.food_inventory_value = sum(firm.inventory_value for firm in self.firms)
        self.food_productivity = np.mean([firm.productivity for firm in self.firms])
        self.production_scale = np.mean([firm.production_scale for firm in self.firms])
        self.real_output_value = self.food_output_units
        self.nominal_output_value = sum(
            firm.output_units * firm.price
            for firm in self.firms
        )
        self.inventory_depreciation = sum(
            firm.inventory_depreciation
            for firm in self.firms
        )
        self.inventory_demand_ratio = safe_ratio(
            self.food_inventory_units,
            max(1.0, self.food_demand_units),
        )
        self.money_issued_this_step = self.central_bank.money_issued_this_step
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
            "total_labor": total_labor,
            "real_output_value": self.real_output_value,
            "nominal_output_value": self.nominal_output_value,
            "wage_bill": self.wage_bill,
            "dividend": self.dividend_paid,
            "sales": self.world.consumption_system.total_consumption,
            "firm_sales_revenue": self.sales_revenue,
            "firm_cash": self.cash(),
            "firm_inventory": self.food_inventory_value,
            "firm_net_worth": self.net_worth(),
            "profit_before_dividend": self.profit_before_dividend,
            "food_price": self.price,
            "sales_income_ratio": self.sales_income_ratio,
            "cash_slope_signal": self.cash_slope_signal,
            "food_output_units": self.food_output_units,
            "food_demand_units": self.food_demand_units,
            "food_sales_units": self.food_sales_units,
            "food_inventory_units": self.food_inventory_units,
            "inventory_demand_ratio": self.inventory_demand_ratio,
            "unmet_food_demand_units": self.unmet_food_demand_units,
            "money_issued": self.money_issued_this_step,
            "inventory_monetized_units": self.inventory_monetized_units_this_step,
            "cumulative_money_issued": self.cumulative_money_issued,
            "cumulative_inventory_monetized_units": (
                self.cumulative_inventory_monetized_units
            ),
            "central_bank_inventory_purchase": self.central_bank_inventory_purchase,
            "central_bank_market_release_revenue": (
                self.central_bank_market_release_revenue
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
            "central_bank_net_money_issued": self.money_issued_this_step,
            "central_bank_food_inventory_units": (
                self.central_bank_food_inventory_units
            ),
            "central_bank_food_purchase_units": self.central_bank_food_purchase_units,
            "central_bank_food_release_units": self.central_bank_food_release_units,
            "central_bank_food_subsidy_units": self.central_bank_food_subsidy_units,
        }
