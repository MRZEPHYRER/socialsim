import central_bank.config as config
from economy.ledger import object_account


def safe_ratio(numerator, denominator):
    if denominator == 0:
        return 0.0

    return numerator / denominator


class CentralBank:

    def __init__(self):
        self.money_supply = config.CENTRAL_BANK_INITIAL_MONEY_SUPPLY
        self.food_inventory_units = 0.0
        self.money_issued_food_inventory_units = 0.0
        self.public_income_balance = 0.0
        self.firm_loan_balance = 0.0
        self.interest_receivable = 0.0
        self.cumulative_interest_income = 0.0

        self.inventory_purchase_value_this_step = 0.0
        self.money_issued_this_step = 0.0
        self.money_destroyed_this_step = 0.0
        self.loan_issued_this_step = 0.0
        self.loan_principal_repaid_this_step = 0.0
        self.loan_interest_paid_this_step = 0.0
        self.public_income_this_step = 0.0
        self.public_income_used_this_step = 0.0
        self.food_purchase_units_this_step = 0.0
        self.food_purchase_requested_units_this_step = 0.0
        self.food_purchase_unfilled_units_this_step = 0.0
        self.market_release_revenue_this_step = 0.0
        self.market_release_money_destroyed_this_step = 0.0
        self.market_release_public_income_this_step = 0.0
        self.food_release_units_this_step = 0.0
        self.money_issued_food_release_units_this_step = 0.0
        self.poverty_subsidy_value_this_step = 0.0
        self.food_subsidy_units_this_step = 0.0

    def money_supply_account(self):
        return object_account(
            self,
            "money_supply",
            "central_bank.money_supply",
        )

    def public_income_account(self):
        return object_account(
            self,
            "public_income_balance",
            "central_bank.public_income_balance",
        )

    def reset_step(self):
        self.inventory_purchase_value_this_step = 0.0
        self.money_issued_this_step = 0.0
        self.money_destroyed_this_step = 0.0
        self.loan_issued_this_step = 0.0
        self.loan_principal_repaid_this_step = 0.0
        self.loan_interest_paid_this_step = 0.0
        self.public_income_this_step = 0.0
        self.public_income_used_this_step = 0.0
        self.food_purchase_units_this_step = 0.0
        self.food_purchase_requested_units_this_step = 0.0
        self.food_purchase_unfilled_units_this_step = 0.0
        self.market_release_revenue_this_step = 0.0
        self.market_release_money_destroyed_this_step = 0.0
        self.market_release_public_income_this_step = 0.0
        self.food_release_units_this_step = 0.0
        self.money_issued_food_release_units_this_step = 0.0
        self.poverty_subsidy_value_this_step = 0.0
        self.food_subsidy_units_this_step = 0.0

    def issue_firm_working_capital_loan(self, firm, required_cash):
        if (
            not config.CENTRAL_BANK_ENABLED
            or
            not config.CENTRAL_BANK_FIRM_CREDIT_ENABLED
        ):
            return 0.0

        loan_amount = max(0.0, required_cash - firm.cash)

        if loan_amount <= 0:
            return 0.0

        firm.world.ledger.create_loan_attr(
            bank_name="central_bank.firm_credit",
            borrower_obj=firm,
            borrower_attr="cash",
            borrower_name="firm.cash",
            amount=loan_amount,
            reason="firm_working_capital_loan",
            money_supply_obj=self,
            money_supply_attr="money_supply",
            loan_obj=self,
            loan_attr="firm_loan_balance",
        )
        self.money_issued_this_step += loan_amount
        self.loan_issued_this_step += loan_amount

        return loan_amount

    def repay_firm_working_capital_loan(self, firm):
        if (
            not config.CENTRAL_BANK_ENABLED
            or
            not config.CENTRAL_BANK_FIRM_CREDIT_ENABLED
            or
            self.firm_loan_balance <= 0
        ):
            return 0.0

        cash_buffer = max(
            0.0,
            config.CENTRAL_BANK_FIRM_LOAN_REPAYMENT_CASH_BUFFER,
        )
        repayable_cash = max(0.0, firm.cash - cash_buffer)
        scheduled_principal = (
            self.firm_loan_balance
            *
            config.CENTRAL_BANK_FIRM_LOAN_REPAYMENT_RATE
        )
        principal = min(
            self.firm_loan_balance,
            scheduled_principal,
            repayable_cash,
        )

        if principal <= 0:
            return 0.0

        interest = min(
            max(0.0, firm.cash - principal - cash_buffer),
            self.firm_loan_balance
            *
            config.CENTRAL_BANK_FIRM_LOAN_INTEREST_RATE,
        )
        firm.world.ledger.repay_loan_attrs(
            borrower_obj=firm,
            borrower_attr="cash",
            borrower_name="firm.cash",
            principal=principal,
            reason="firm_working_capital_loan_repayment",
            money_supply_obj=self,
            money_supply_attr="money_supply",
            loan_obj=self,
            loan_attr="firm_loan_balance",
            bank_obj=self,
            bank_attr="public_income_balance",
            bank_name="central_bank.public_income_balance",
            interest=interest,
        )
        self.money_issued_this_step -= principal
        self.money_destroyed_this_step += principal
        self.loan_principal_repaid_this_step += principal
        self.loan_interest_paid_this_step += interest

        if interest > 0:
            self.public_income_this_step += interest
            firm.world.record_central_bank_public_income_source(
                "loan_interest",
                interest,
            )

        return principal + interest

    def collect_public_wealth(self, world):
        public_income = max(0.0, getattr(world, "public_wealth", 0.0))

        if public_income <= 0:
            return 0.0

        world.ledger.transfer(
            payer=object_account(
                world,
                "public_wealth",
                "world.public_wealth",
            ),
            receiver=self.public_income_account(),
            amount=public_income,
            reason="collect_public_wealth",
        )
        world.record_public_wealth_collection(public_income)
        self.public_income_this_step += public_income

        return public_income

    def purchase_excess_food_inventory(self, firm):
        if (
            not config.CENTRAL_BANK_ENABLED
            or
            not config.CENTRAL_BANK_INVENTORY_PURCHASE_ENABLED
        ):
            return 0.0

        purchase_gate = 1.0

        if config.CENTRAL_BANK_FIRM_CASH_PURCHASE_GATE_ENABLED:
            cash_buffer = max(
                1.0,
                config.CENTRAL_BANK_FIRM_CASH_BUFFER,
            )
            purchase_gate = safe_ratio(
                config.CENTRAL_BANK_TARGET_FIRM_CASH
                +
                cash_buffer
                -
                firm.cash,
                cash_buffer,
            )
            purchase_gate = max(
                0.0,
                min(1.0, purchase_gate)
            )

        if purchase_gate <= 0:
            return 0.0

        demand_base = max(1.0, firm.food_demand_units)
        threshold_units = (
            firm.target_inventory_days
            +
            config.CENTRAL_BANK_FOOD_PURCHASE_EXCESS_INVENTORY_DAYS
        ) * demand_base
        excess_units = max(0.0, firm.food_inventory_units - threshold_units)
        purchase_units = min(
            firm.food_inventory_units,
            excess_units
            *
            config.CENTRAL_BANK_FOOD_PURCHASE_RATE
            *
            purchase_gate,
        )

        if purchase_units <= 0:
            return 0.0

        self.food_purchase_requested_units_this_step = purchase_units

        purchase_value = (
            purchase_units
            *
            firm.price
            *
            config.CENTRAL_BANK_FOOD_PURCHASE_PRICE_HAIRCUT
        )
        if not config.CENTRAL_BANK_INVENTORY_PURCHASE_MONEY_ISSUE_ENABLED:
            purchase_value = min(
                purchase_value,
                self.public_income_balance,
            )

            if purchase_value <= 0:
                return 0.0

            purchase_units = purchase_value / (
                firm.price
                *
                config.CENTRAL_BANK_FOOD_PURCHASE_PRICE_HAIRCUT
            )

        firm.food_inventory_units = max(
            0.0,
            firm.food_inventory_units - purchase_units,
        )
        firm.food_inventory_value = firm.food_inventory_units * firm.price
        firm.inventory_demand_ratio = safe_ratio(
            firm.food_inventory_units,
            demand_base,
        )

        self.food_inventory_units += purchase_units
        self.inventory_purchase_value_this_step = purchase_value
        self.food_purchase_units_this_step = purchase_units
        self.food_purchase_unfilled_units_this_step = 0.0

        public_income_used = min(self.public_income_balance, purchase_value)
        if config.CENTRAL_BANK_INVENTORY_PURCHASE_MONEY_ISSUE_ENABLED:
            newly_issued_money = purchase_value - public_income_used
        else:
            newly_issued_money = 0.0
        self.public_income_used_this_step = public_income_used
        self.money_issued_this_step += newly_issued_money

        if public_income_used > 0:
            firm.world.ledger.transfer_attrs(
                payer_obj=self,
                payer_attr="public_income_balance",
                payer_name="central_bank.public_income_balance",
                receiver_obj=firm,
                receiver_attr="cash",
                receiver_name="firm.cash",
                amount=public_income_used,
                reason="central_bank_inventory_purchase_public_income",
            )

        if newly_issued_money > 0:
            firm.world.ledger.create_money_attr(
                receiver_obj=firm,
                receiver_attr="cash",
                receiver_name="firm.cash",
                amount=newly_issued_money,
                reason="central_bank_inventory_purchase_money_issue",
                money_supply_obj=self,
                money_supply_attr="money_supply",
            )
            self.money_issued_food_inventory_units += (
                purchase_units
                *
                safe_ratio(newly_issued_money, purchase_value)
            )

        return purchase_value

    def release_units_for_market(self, firm):
        if (
            not config.CENTRAL_BANK_ENABLED
            or
            not config.CENTRAL_BANK_MARKET_RELEASE_ENABLED
            or
            self.food_inventory_units <= 0
        ):
            return 0.0

        demand_base = max(1.0, firm.food_demand_units)
        current_shortage_units = 0.0

        if config.CENTRAL_BANK_SHORTAGE_FIRST_RELEASE_ENABLED:
            current_firm_supply = (
                firm.food_output_units
                +
                firm.food_inventory_units
            )
            current_shortage_units = max(
                0.0,
                firm.food_demand_units - current_firm_supply,
            )

        shortage_ratio = safe_ratio(
            firm.unmet_food_demand_units,
            demand_base,
        )
        inventory_pressure = safe_ratio(
            max(0.0, firm.target_inventory_days - firm.inventory_demand_ratio),
            max(1.0, firm.target_inventory_days),
        )
        price_pressure = safe_ratio(
            max(0.0, firm.price - config.CENTRAL_BANK_FOOD_PRICE_ANCHOR),
            max(0.01, config.CENTRAL_BANK_FOOD_PRICE_ANCHOR),
        )
        if config.CENTRAL_BANK_SHORTAGE_FIRST_RELEASE_ENABLED:
            pressure = max(inventory_pressure, price_pressure)
        else:
            pressure = max(shortage_ratio, inventory_pressure, price_pressure)
        desired_units = (
            current_shortage_units
            +
            demand_base
            *
            config.CENTRAL_BANK_FOOD_RELEASE_RATE
            *
            pressure
        )

        return min(self.food_inventory_units, desired_units)

    def release_to_market(self, firm):
        release_units = self.release_units_for_market(firm)

        if release_units <= 0:
            return 0.0

        money_issued_inventory_share = safe_ratio(
            self.money_issued_food_inventory_units,
            self.food_inventory_units,
        )
        money_issued_release_units = min(
            self.money_issued_food_inventory_units,
            release_units * money_issued_inventory_share,
        )
        self.food_inventory_units -= release_units
        self.money_issued_food_inventory_units -= money_issued_release_units
        self.food_release_units_this_step = release_units
        self.money_issued_food_release_units_this_step = money_issued_release_units
        firm.food_inventory_units += release_units

        return release_units

    def record_market_release_revenue(
        self,
        revenue,
        payer=None,
        ledger=None,
        payer_obj=None,
        payer_attr=None,
        payer_name=None,
    ):
        if revenue <= 0:
            return

        self.market_release_revenue_this_step += revenue
        money_destroyed = (
            revenue
            *
            safe_ratio(
                self.money_issued_food_release_units_this_step,
                self.food_release_units_this_step,
            )
        )
        public_income = revenue - money_destroyed

        if money_destroyed > 0 and payer_obj is not None and payer_attr is not None and ledger is not None:
            ledger.destroy_money_attr(
                payer_obj=payer_obj,
                payer_attr=payer_attr,
                payer_name=payer_name,
                amount=money_destroyed,
                reason="central_bank_market_release_revenue",
                money_supply_obj=self,
                money_supply_attr="money_supply",
            )
        elif money_destroyed > 0 and payer is not None and ledger is not None:
            ledger.destroy_money(
                payer=payer,
                amount=money_destroyed,
                reason="central_bank_market_release_revenue",
                money_supply=self.money_supply_account(),
            )
        else:
            self.money_supply -= money_destroyed

        if public_income > 0:
            if payer_obj is not None and payer_attr is not None and ledger is not None:
                ledger.transfer_attrs(
                    payer_obj=payer_obj,
                    payer_attr=payer_attr,
                    payer_name=payer_name,
                    receiver_obj=self,
                    receiver_attr="public_income_balance",
                    receiver_name="central_bank.public_income_balance",
                    amount=public_income,
                    reason="central_bank_market_release_public_income",
                )
            else:
                self.public_income_balance += public_income

        self.money_issued_this_step -= money_destroyed
        self.money_destroyed_this_step += money_destroyed
        self.market_release_money_destroyed_this_step += money_destroyed
        self.market_release_public_income_this_step += public_income

        if public_income > 0 and payer_obj is not None:
            payer_obj.world.record_central_bank_public_income_source(
                "public_inventory_income",
                public_income,
            )

    def distribute_poverty_food_subsidy(self, households, food_price):
        if (
            not config.CENTRAL_BANK_ENABLED
            or
            not config.CENTRAL_BANK_POVERTY_SUBSIDY_ENABLED
            or
            self.food_inventory_units <= 0
        ):
            return 0.0

        available_units = (
            self.food_inventory_units
            *
            config.CENTRAL_BANK_FOOD_POVERTY_SUBSIDY_RATE
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

        for household in needy_households:
            subsidy_units = (
                getattr(household, "food_need_gap_units_this_step", 0.0)
                *
                fill
            )
            household.food_subsidy_units_this_step = subsidy_units
            household.food_subsidy_value_this_step = subsidy_units * food_price
            used_units += subsidy_units

        self.food_inventory_units -= used_units
        self.food_subsidy_units_this_step = used_units
        self.poverty_subsidy_value_this_step = used_units * food_price

        return self.poverty_subsidy_value_this_step
