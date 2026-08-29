from economy import config
from time_system import months_to_steps


class ConsumptionSystem:

    def __init__(self, world):
        self.world = world
        self.total_consumption = 0.0
        self.constrained_household_count = 0
        self.consumed_household_count = 0
        self.constrained_household_ratio = 0.0
        self.capped_household_count = 0
        self.capped_household_ratio = 0.0

    def get_members(self, household):
        return self.world.needs_system.get_members(household)

    def household_necessary_consumption(self, household):
        return self.world.needs_system.household_minimum_need_units(household)

    def household_consumption(self, household, price_multiplier=1.0):
        household.necessary_consumption_this_step = 0.0
        household.desired_consumption_this_step = 0.0
        household.affordable_consumption_this_step = 0.0
        household.economic_pressure_this_step = 0.0
        household.budget_constrained_this_step = False

        base_necessary_consumption = self.household_necessary_consumption(
            household
        )
        necessary_consumption = (
            base_necessary_consumption
            *
            price_multiplier
        )

        if necessary_consumption == 0:
            return 0.0

        income = household.income_this_step
        wealth = household.wealth

        reserve_weeks = months_to_steps(
            config.TARGET_WEALTH_RESERVE_MONTHS
        )
        target_wealth = reserve_weeks * necessary_consumption
        household.target_wealth_reserve_months = config.TARGET_WEALTH_RESERVE_MONTHS
        household.target_wealth_reserve_weeks = reserve_weeks
        wealth_gap = target_wealth - wealth
        positive_gap_ratio = (
            max(0.0, wealth_gap)
            /
            max(1.0, target_wealth)
        )
        optional_consumption_multiplier = max(
            config.MIN_OPTIONAL_CONSUMPTION_MULTIPLIER,
            1.0
            -
            config.TARGET_WEALTH_GAP_SENSITIVITY
            *
            positive_gap_ratio,
        )
        optional_income_consumption = (
            max(0.0, income - necessary_consumption)
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

        desired_consumption = (
            necessary_consumption
            +
            optional_consumption
        )

        affordable_consumption = (
            income
            +
            wealth
            *
            config.WEALTH_DRAWDOWN_RATE
        )

        consumption = min(
            desired_consumption,
            affordable_consumption
        )
        consumption = max(
            consumption,
            0.0
        )

        household.necessary_consumption_this_step = necessary_consumption
        household.desired_consumption_this_step = desired_consumption
        household.optional_consumption_this_step = optional_consumption
        household.optional_income_consumption_this_step = optional_income_consumption
        household.excess_wealth_consumption_this_step = excess_wealth_consumption
        household.target_wealth_this_step = target_wealth
        household.wealth_gap_this_step = wealth_gap
        household.optional_consumption_multiplier_this_step = (
            optional_consumption_multiplier
        )
        household.affordable_consumption_this_step = affordable_consumption
        household.economic_pressure_this_step = (
            desired_consumption / affordable_consumption
            if affordable_consumption > 0
            else float("inf")
        )
        household.budget_constrained_this_step = (
            desired_consumption > affordable_consumption
        )

        # Backward-compatible diagnostic names used by older test scripts.
        household.max_consumption_this_step = affordable_consumption
        household.consumption_capped_this_step = (
            household.budget_constrained_this_step
        )

        return consumption

    def step(self, price_multiplier=1.0):
        total_consumption = 0.0
        constrained_household_count = 0
        consumed_household_count = 0

        for household in self.world.households:
            consumption = self.household_consumption(
                household,
                price_multiplier=price_multiplier,
            )

            if household.size() > 0:
                consumed_household_count += 1

                if household.budget_constrained_this_step:
                    constrained_household_count += 1

            household.consumption_this_step = consumption
            household.saving_this_step = (
                household.income_this_step
                -
                consumption
            )
            household.wealth += household.saving_this_step

            if household.wealth < 0:
                household.wealth = 0

            total_consumption += consumption

        self.total_consumption = total_consumption
        self.constrained_household_count = constrained_household_count
        self.consumed_household_count = consumed_household_count

        if consumed_household_count > 0:
            self.constrained_household_ratio = (
                constrained_household_count
                /
                consumed_household_count
            )
        else:
            self.constrained_household_ratio = 0.0

        # Backward-compatible diagnostic names used by older test scripts.
        self.capped_household_count = constrained_household_count
        self.capped_household_ratio = self.constrained_household_ratio
