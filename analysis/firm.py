import numpy as np
import matplotlib.pyplot as plt

from analysis.common import CommonAnalysis


class FirmAnalysis(CommonAnalysis):

    def has_firm_history(self):
        return (
            hasattr(self.world, "firm_cash_history")
            and len(self.world.firm_cash_history) > 0
        )

    def firm_cash(self):
        return self.world.firm_cash_history

    def firm_inventory(self):
        return self.world.firm_inventory_history

    def firm_net_worth(self):
        return self.world.firm_net_worth_history

    def firm_profit(self):
        return self.world.firm_profit_history

    def food_price(self):
        """Legacy compatibility price; prefer planning/realized indices."""
        return self.world.food_price_history

    def household_planning_price(self):
        return [
            float(row.get("household_planning_price_index", 0.0))
            for row in getattr(self.world, "diagnostics_rows", [])
        ]

    def realized_transaction_price(self):
        return [
            float(row.get("realized_transaction_price_index", 0.0))
            for row in getattr(self.world, "diagnostics_rows", [])
        ]

    def unit_labor_cost(self):
        return self.optional_history("unit_labor_cost_history")

    def unit_labor_cost_growth(self):
        return self.optional_history("unit_labor_cost_growth_history")

    def price_inventory_gap(self):
        return self.optional_history("price_inventory_gap_history")

    def price_cost_growth(self):
        return self.optional_history("price_cost_growth_history")

    def price_inflation_signal(self):
        return self.optional_history("price_inflation_signal_history")

    def price_log_adjustment(self):
        return self.optional_history("price_log_adjustment_history")

    def demand_pressure(self):
        return self.optional_history("demand_pressure_history")

    def available_food_supply_units(self):
        return self.optional_history("available_food_supply_units_history")

    def firm_sales_income_ratio(self):
        return self.world.firm_sales_income_ratio_history

    def optional_history(self, name):
        return getattr(self.world, name, [])

    def food_output_units(self):
        return self.optional_history("food_output_units_history")

    def food_demand_units(self):
        return self.optional_history("food_demand_units_history")

    def food_sales_units(self):
        return self.optional_history("food_sales_units_history")

    def food_inventory_units(self):
        return self.optional_history("food_inventory_units_history")

    def food_inventory_demand_ratio(self):
        return self.optional_history("food_inventory_demand_ratio_history")

    def unmet_food_demand_units(self):
        return self.optional_history("unmet_food_demand_units_history")

    def money_issued(self):
        return self.optional_history("money_issued_history")

    def cumulative_money_issued(self):
        return self.optional_history("cumulative_money_issued_history")

    def working_capital_loan_issued(self):
        return self.optional_history("working_capital_loan_issued_history")

    def working_capital_target_cash(self):
        return self.optional_history("working_capital_target_cash_history")

    def working_capital_funding_gap(self):
        return self.optional_history("working_capital_funding_gap_history")

    def working_capital_loan_repaid(self):
        return self.optional_history("working_capital_loan_repaid_history")

    def working_capital_loan_balance(self):
        return self.optional_history("working_capital_loan_balance_history")

    def inventory_monetized_units(self):
        return self.optional_history("inventory_monetized_units_history")

    def household_wealth(self):
        return self.optional_history("wealth_history")

    def consumption(self):
        return self.optional_history("consumption_history")

    def central_bank_inventory_purchase(self):
        return self.optional_history("central_bank_inventory_purchase_history")

    def central_bank_market_release_revenue(self):
        return self.optional_history("central_bank_market_release_revenue_history")

    def central_bank_poverty_subsidy_value(self):
        return self.optional_history("central_bank_poverty_subsidy_value_history")

    def central_bank_public_income(self):
        return self.optional_history("central_bank_public_income_history")

    def central_bank_public_income_used(self):
        return self.optional_history("central_bank_public_income_used_history")

    def central_bank_public_income_balance(self):
        return self.optional_history("central_bank_public_income_balance_history")

    def central_bank_net_money_issued(self):
        history = self.optional_history("central_bank_net_money_issued_history")

        if history:
            return history

        return self.money_issued()

    def central_bank_food_inventory_units(self):
        return self.optional_history("central_bank_food_inventory_units_history")

    def central_bank_food_purchase_units(self):
        return self.optional_history("central_bank_food_purchase_units_history")

    def central_bank_food_release_units(self):
        return self.optional_history("central_bank_food_release_units_history")

    def central_bank_food_subsidy_units(self):
        return self.optional_history("central_bank_food_subsidy_units_history")

    def private_cash_stock(self):
        return self.optional_history("private_cash_stock_history")

    def public_wealth(self):
        return self.optional_history("public_wealth_history")

    def expected_money_stock(self):
        return self.optional_history("expected_money_stock_history")

    def located_money_stock(self):
        return self.optional_history("located_money_stock_history")

    def monetary_accounting_gap(self):
        return self.optional_history("monetary_accounting_gap_history")

    def series_slope(self, values):
        values = np.asarray(values, dtype=float)

        if len(values) <= 1:
            return 0.0

        x = np.arange(len(values), dtype=float)
        return float(np.polyfit(x, values, 1)[0])

    def firm_diagnostics_rows(self):
        return getattr(self.world, "firm_diagnostics_rows", [])

    def has_multi_firm_diagnostics(self):
        rows = self.firm_diagnostics_rows()

        if not rows:
            return False

        firm_ids = {
            row.get("firm_id")
            for row in rows
        }

        return len(firm_ids) > 1

    def firm_series_by_id(self, field):
        series = {}

        for row in self.firm_diagnostics_rows():
            firm_id = row.get("firm_id")

            if firm_id is None:
                continue

            series.setdefault(firm_id, []).append(row.get(field, 0.0))

        return series

    def firm_weeks_by_id(self):
        series = {}
        for row in self.firm_diagnostics_rows():
            firm_id = row.get("firm_id")
            if firm_id is None:
                continue
            series.setdefault(firm_id, []).append(
                row.get("simulation_week", row.get("global_step", row.get("step", 0)))
            )
        return series

    def multi_firm_report(self, window=200):
        print("======================")
        print("Multi-Firm Report")
        print("======================")

        rows = self.firm_diagnostics_rows()

        if not rows:
            print("No per-firm diagnostics recorded.")
            return

        firm_ids = sorted({
            row["firm_id"]
            for row in rows
        })

        if len(firm_ids) <= 1:
            print("Only one firm is active; multi-firm diagnostics are trivial.")
            return

        latest_step = max(row["step"] for row in rows)
        latest_rows = [
            row
            for row in rows
            if row["step"] == latest_step
        ]
        latest_rows = sorted(
            latest_rows,
            key=lambda row: row["firm_id"],
        )
        window_steps = sorted({
            row["step"]
            for row in rows
        })[-window:]
        window_rows = [
            row
            for row in rows
            if row["step"] in window_steps
        ]

        print("\nFinal Multi-Firm State")
        print("----------------------")
        print(f"Firm Count: {len(firm_ids)}")
        print(f"Latest Firm Diagnostic Week: {latest_step}")

        for row in latest_rows:
            print(
                "Firm "
                f"{row['firm_id']}: "
                f"price={row.get('price', 0.0):.4f}, "
                f"choice_prob={row.get('choice_probability', 0.0):.4f}, "
                f"actual_share={row.get('actual_market_share', 0.0):.4f}, "
                f"sales={row.get('sales', 0.0):.2f}, "
                f"cash={row.get('cash', 0.0):.2f}, "
                f"inventory={row.get('inventory', 0.0):.2f}, "
                f"loan={row.get('loan_balance', 0.0):.2f}"
            )

        print("\nChoice Mechanism")
        print("----------------")
        sorted_by_price = sorted(
            latest_rows,
            key=lambda row: row.get("price", 0.0),
        )
        if sorted_by_price:
            cheapest = sorted_by_price[0]
            most_expensive = sorted_by_price[-1]
            print(
                "Cheapest Firm: "
                f"{cheapest['firm_id']} "
                f"(price={cheapest.get('price', 0.0):.4f}, "
                f"choice_prob={cheapest.get('choice_probability', 0.0):.4f}, "
                f"actual_share={cheapest.get('actual_market_share', 0.0):.4f})"
            )
            print(
                "Most Expensive Firm: "
                f"{most_expensive['firm_id']} "
                f"(price={most_expensive.get('price', 0.0):.4f}, "
                f"choice_prob={most_expensive.get('choice_probability', 0.0):.4f}, "
                f"actual_share={most_expensive.get('actual_market_share', 0.0):.4f})"
            )

        shares = [
            row.get("actual_market_share", 0.0)
            for row in latest_rows
        ]
        probabilities = [
            row.get("choice_probability", 0.0)
            for row in latest_rows
        ]
        print(f"Sum Choice Probability: {sum(probabilities):.6f}")
        print(f"Sum Actual Market Share: {sum(shares):.6f}")
        print(
            "Market Share Range: "
            f"{min(shares):.4f} - {max(shares):.4f}"
        )

        print(f"\nLast {len(window_steps)} Weeks Averages")
        print("------------------------")
        review_counts = {}
        decision_counts = {}

        for firm_id in firm_ids:
            firm_window = [
                row
                for row in window_rows
                if row["firm_id"] == firm_id
            ]

            if not firm_window:
                continue

            review_counts[firm_id] = sum(
                1
                for row in firm_window
                if row.get("price_reviewed", False)
            )

            for row in firm_window:
                decision = row.get("price_decision", "none")
                decision_counts[decision] = decision_counts.get(decision, 0) + 1

            print(
                "Firm "
                f"{firm_id}: "
                f"avg_share={np.mean([r.get('actual_market_share', 0.0) for r in firm_window]):.4f}, "
                f"avg_choice_prob={np.mean([r.get('choice_probability', 0.0) for r in firm_window]):.4f}, "
                f"avg_sales={np.mean([r.get('sales', 0.0) for r in firm_window]):.2f}, "
                f"avg_unmet_demand={np.mean([r.get('unmet_demand', 0.0) for r in firm_window]):.4f}, "
                f"avg_margin={np.mean([r.get('margin', 0.0) for r in firm_window]):.4f}, "
                f"avg_loan_issued/wage={np.mean([r.get('loan_issued_to_wage_bill', 0.0) for r in firm_window]):.4f}, "
                f"avg_loan_repaid/wage={np.mean([r.get('loan_repaid_to_wage_bill', 0.0) for r in firm_window]):.4f}, "
                f"avg_observed_demand={np.mean([r.get('observed_demand', 0.0) for r in firm_window]):.2f}, "
                f"avg_unit_share={np.mean([r.get('unit_market_share', r.get('actual_market_share', 0.0)) for r in firm_window]):.4f}, "
                f"avg_revenue_share={np.mean([r.get('revenue_market_share', 0.0) for r in firm_window]):.4f}, "
                f"avg_dividend={np.mean([r.get('dividend_paid', r.get('dividend_payment', 0.0)) for r in firm_window]):.2f}, "
                f"avg_eligible_profit={np.mean([r.get('dividend_eligible_profit', 0.0) for r in firm_window]):.2f}, "
                f"avg_production_plan={np.mean([r.get('production_plan', 0.0) for r in firm_window]):.2f}, "
                f"avg_actual_production={np.mean([r.get('actual_production', r.get('production', 0.0)) for r in firm_window]):.2f}, "
                f"avg_capacity_utilization={np.mean([r.get('capacity_utilization', 0.0) for r in firm_window]):.4f}, "
                f"avg_inventory_gap_units={np.mean([r.get('inventory_gap_units', 0.0) for r in firm_window]):.2f}, "
                f"avg_gradient={np.mean([r.get('estimated_gradient', 0.0) for r in firm_window]):.2f}, "
                f"reviews={review_counts[firm_id]}, "
                f"cash_slope={self.series_slope([r.get('cash', 0.0) for r in firm_window]):.2f}"
            )

        print("\nCash Bridge Diagnostics")
        print("-----------------------")
        max_bridge_gap = max(
            abs(row.get("cash_bridge_gap", 0.0))
            for row in window_rows
        )
        credit_modes = sorted({
            row.get("credit_allocation_mode", "not_recorded")
            for row in window_rows
        })
        depletion_counts = {}

        for row in window_rows:
            stage = row.get("cash_depletion_stage", "not_recorded")
            depletion_counts[stage] = depletion_counts.get(stage, 0) + 1

        print(f"Max |Cash Bridge Gap|: {max_bridge_gap:.6g}")
        print(f"Credit Allocation Modes: {', '.join(credit_modes)}")

        for stage, count in sorted(depletion_counts.items()):
            print(f"{stage}: {count}")

        for firm_id in firm_ids:
            firm_window = [
                row
                for row in window_rows
                if row["firm_id"] == firm_id
            ]

            if not firm_window:
                continue

            zero_cash_steps = sum(
                1
                for row in firm_window
                if row.get("cash_end", row.get("cash", 0.0)) <= 1e-9
            )
            avg_other_inflow = np.mean([
                row.get("other_cash_inflow", 0.0)
                for row in firm_window
            ])
            avg_other_outflow = np.mean([
                row.get("other_cash_outflow", 0.0)
                for row in firm_window
            ])
            avg_funding_gap = np.mean([
                row.get("funding_gap", 0.0)
                for row in firm_window
            ])
            avg_loan_issued = np.mean([
                row.get("loan_issued", 0.0)
                for row in firm_window
            ])

            print(
                "Firm "
                f"{firm_id}: "
                f"zero_cash_steps={zero_cash_steps}/{len(firm_window)}, "
                f"avg_funding_gap={avg_funding_gap:.2f}, "
                f"avg_loan_issued={avg_loan_issued:.2f}, "
                f"avg_other_in={avg_other_inflow:.2f}, "
                f"avg_other_out={avg_other_outflow:.2f}"
            )

        if decision_counts:
            print("\nPrice Review Decisions")
            print("----------------------")
            for decision, count in sorted(decision_counts.items()):
                print(f"{decision}: {count}")

        first_step = min(row["step"] for row in rows)
        first_rows = sorted(
            [
                row
                for row in rows
                if row["step"] == first_step
            ],
            key=lambda row: row["price"],
        )
        final_price_order = [
            row["firm_id"]
            for row in sorted_by_price
        ]
        first_price_order = [
            row["firm_id"]
            for row in first_rows
        ]
        print(
            "Price Order Changed: "
            f"{first_price_order != final_price_order}"
        )

        print("\nAggregation Checks")
        print("------------------")
        total_share_gap = abs(sum(shares) - 1.0)
        total_probability_gap = abs(sum(probabilities) - 1.0)
        print(f"Actual Share Sum Gap: {total_share_gap:.6g}")
        print(f"Choice Probability Sum Gap: {total_probability_gap:.6g}")

        if self.diagnostics_rows_available():
            macro_latest = self.world.diagnostics_rows[-1]
            checks = [
                ("cash", "firm_cash"),
                ("inventory_units", "food_inventory_units"),
                ("loan_balance", "loan_balance"),
                ("sales", "firm_sales_revenue"),
            ]

            for firm_key, macro_key in checks:
                firm_total = sum(row.get(firm_key, 0.0) for row in latest_rows)
                macro_total = macro_latest.get(macro_key, 0.0)
                print(
                    f"Aggregate {firm_key} Gap: "
                    f"{firm_total - macro_total:.6g}"
                )

            firm_scheduled = sum(
                row.get("scheduled_wage_bill", 0.0) for row in latest_rows
            )
            aggregate_scheduled = macro_latest.get(
                "scheduled_aggregate_wage_bill",
                0.0,
            )
            firm_executed = sum(
                row.get("executed_wage_bill", row.get("wage_payment", 0.0))
                for row in latest_rows
            )
            aggregate_executed = macro_latest.get(
                "executed_aggregate_wage_bill",
                macro_latest.get("wage_payment", 0.0),
            )
            print(
                "Aggregate Scheduled Wage Bill Gap: "
                f"{firm_scheduled - aggregate_scheduled:.6g}"
            )
            print(
                "Aggregate Executed Payroll Gap: "
                f"{firm_executed - aggregate_executed:.6g}"
            )
            print(
                "Aggregate Payroll Funding Shortfall: "
                f"{aggregate_scheduled - aggregate_executed:.6g}"
            )

    def diagnostics_rows_available(self):
        return bool(getattr(self.world, "diagnostics_rows", []))

    def accounting_rows(self):
        accounting = getattr(self.world, "accounting", None)
        return getattr(accounting, "rows", []) if accounting is not None else []

    def accounting_reconciliation_rows(self):
        accounting = getattr(self.world, "accounting", None)
        return (
            getattr(accounting, "reconciliation_rows", [])
            if accounting is not None
            else []
        )

    def accounting_report(self, window=200):
        print("======================")
        print("Accounting Report")
        print("======================")

        rows = self.accounting_rows()
        reconciliation = self.accounting_reconciliation_rows()
        if not rows:
            print("No accounting records recorded.")
            return

        grouped = {}
        for row in rows:
            grouped.setdefault(row.get("step"), []).append(row)
        step_keys = list(grouped.keys())[-window:]
        window_rows = [row for step in step_keys for row in grouped[step]]
        latest_step_rows = grouped[step_keys[-1]]
        sum_fields = {
            "accounting_revenue", "inventory_cost_or_cogs",
            "accounting_operating_profit", "accounting_net_income",
            "legacy_profit_before_dividend", "cfo", "cfi", "cff",
            "loan_issued", "principal_repaid", "dividends", "cash",
            "inventory_book_value", "inventory_market_value", "total_assets",
            "loan_balance", "total_liabilities", "equity",
        }

        def aggregate_step(step_rows):
            result = {}
            for field in sum_fields:
                result[field] = sum(float(row.get(field, 0.0)) for row in step_rows)
            return result

        aggregate_rows = [aggregate_step(grouped[step]) for step in step_keys]
        latest = aggregate_rows[-1]

        def mean(field):
            return float(np.mean([float(row.get(field, 0.0)) for row in window_rows]))

        def maximum_abs(field):
            return max(abs(float(row.get(field, 0.0))) for row in window_rows)

        print("\nFirm Income Statement")
        print("---------------------")
        print(f"Latest Aggregate Accounting Revenue: {float(latest.get('accounting_revenue', 0.0)):.2f}")
        print(f"Latest COGS: {float(latest.get('inventory_cost_or_cogs', 0.0)):.2f}")
        print(f"Latest Aggregate Operating Profit: {float(latest.get('accounting_operating_profit', 0.0)):.2f}")
        print(f"Latest Aggregate Net Income: {float(latest.get('accounting_net_income', 0.0)):.2f}")
        print(f"Last {len(aggregate_rows)} Weeks Avg Aggregate Operating Profit: {np.mean([r['accounting_operating_profit'] for r in aggregate_rows]):.2f}")
        print(f"Last {len(aggregate_rows)} Weeks Avg Aggregate Legacy Profit: {np.mean([r['legacy_profit_before_dividend'] for r in aggregate_rows]):.2f}")

        print("\nFirm Cash Flow Statement")
        print("------------------------")
        print(f"Last {len(aggregate_rows)} Weeks Avg Aggregate CFO: {np.mean([r['cfo'] for r in aggregate_rows]):.2f}")
        print(f"Last {len(aggregate_rows)} Weeks Avg Aggregate CFF: {np.mean([r['cff'] for r in aggregate_rows]):.2f}")
        print(f"Last {len(aggregate_rows)} Weeks Avg Aggregate Loan Issued: {np.mean([r['loan_issued'] for r in aggregate_rows]):.2f}")
        print(f"Last {len(aggregate_rows)} Weeks Avg Aggregate Principal Repaid: {np.mean([r['principal_repaid'] for r in aggregate_rows]):.2f}")
        print(f"Last {len(aggregate_rows)} Weeks Avg Aggregate Dividends: {np.mean([r['dividends'] for r in aggregate_rows]):.2f}")
        print(f"Max |Cash Flow Gap|: {maximum_abs('cash_flow_gap'):.6g}")

        print("\nFirm Balance Sheet")
        print("------------------")
        print(f"Latest Aggregate Cash: {float(latest.get('cash', 0.0)):.2f}")
        print(f"Latest Inventory Book Value: {float(latest.get('inventory_book_value', 0.0)):.2f}")
        print(f"Latest Inventory Market Value: {float(latest.get('inventory_market_value', 0.0)):.2f}")
        print(f"Latest Loan Balance: {float(latest.get('loan_balance', 0.0)):.2f}")
        print(f"Latest Equity: {float(latest.get('equity', 0.0)):.2f}")
        print(f"Max |Balance Sheet Gap|: {maximum_abs('balance_sheet_gap'):.6g}")

        if reconciliation:
            rec_window = reconciliation[-min(window, len(reconciliation)):]
            latest_rec = reconciliation[-1]
            max_loan_gap = max(
                abs(float(row.get('loan_reconciliation_gap', 0.0)))
                for row in rec_window
            )
            max_money_gap = max(
                abs(float(row.get('money_location_gap', 0.0)))
                for row in rec_window
            )
            print("\nSystem Accounting Reconciliation")
            print("-------------------------------")
            print(f"Latest Firm Loan Liabilities: {float(latest_rec.get('firm_loan_liabilities', 0.0)):.2f}")
            print(f"Latest Central Bank Loan Assets: {float(latest_rec.get('central_bank_loan_assets', 0.0)):.2f}")
            print(f"Max |Firm Loan - CB Asset Gap|: {max_loan_gap:.6g}")
            print(f"Max |Money Location Gap|: {max_money_gap:.6g}")
            print(f"Latest Pre-existing Money Base: {float(latest_rec.get('preexisting_non_central_bank_money', 0.0)):.2f}")

        household_rows = getattr(getattr(self.world, "accounting", None), "household_rows", [])
        public_rows = getattr(getattr(self.world, "accounting", None), "public_rows", [])
        central_rows = getattr(getattr(self.world, "accounting", None), "central_bank_rows", [])
        if household_rows:
            print("\nHousehold Accounting")
            print("--------------------")
            print(f"Latest Household Saving: {float(household_rows[-1].get('saving', 0.0)):.2f}")
            print(f"Latest Estate No-Heir Outflow: {float(household_rows[-1].get('estate_no_heir_outflow', 0.0)):.2f}")
            print(
                "Max |Household Wealth Bridge Gap|: "
                f"{max(abs(float(row.get('household_wealth_bridge_gap', 0.0))) for row in household_rows):.6g}"
            )
        if public_rows:
            print("\nPublic Accounting")
            print("-----------------")
            print(f"Latest Public Cash Revenue: {float(public_rows[-1].get('public_cash_revenue', 0.0)):.2f}")
            print(f"Latest Public Cash Expenditure: {float(public_rows[-1].get('public_cash_expenditure', 0.0)):.2f}")
            print(
                "Max |Public Cash Flow Gap|: "
                f"{max(abs(float(row.get('public_cash_flow_gap', 0.0))) for row in public_rows):.6g}"
            )
        if central_rows:
            print("\nCentral Bank Credit Accounting")
            print("------------------------------")
            print(f"Latest Credit Money Liabilities: {float(central_rows[-1].get('credit_money_liabilities', 0.0)):.2f}")
            print(
                "Max |Credit Money Stock-Flow Gap|: "
                f"{max(abs(float(row.get('credit_money_stock_flow_gap', 0.0))) for row in central_rows):.6g}"
            )

    def plot_accounting_diagnostics(self):
        rows = self.accounting_rows()
        if not rows:
            print("No accounting records recorded.")
            return

        steps = [row.get("simulation_week", row.get("step", index)) for index, row in enumerate(rows)]
        fig, ax = plt.subplots(2, 3, figsize=(16, 9))
        series = [
            ("accounting_operating_profit", "Accounting Operating Profit"),
            ("cfo", "Operating Cash Flow"),
            ("cff", "Financing Cash Flow"),
            ("inventory_book_value", "Inventory Book Value"),
            ("loan_balance", "Loan Balance"),
            ("equity", "Firm Equity"),
        ]
        for axis, (field, title) in zip(ax.ravel(), series):
            axis.plot(steps, [float(row.get(field, 0.0)) for row in rows])
            axis.axhline(0, linestyle="--", linewidth=1)
            axis.set_title(title)
            axis.set_xlabel("Simulation week")
        plt.tight_layout()
        plt.show()

    def firm_report(self, window=200):
        print("======================")
        print("Firm Report")
        print("======================")

        if not self.has_firm_history():
            print("No firm history recorded.")
            return

        window = min(window, len(self.firm_cash()))
        cash_window = self.firm_cash()[-window:]
        inventory_window = self.firm_inventory()[-window:]
        net_worth_window = self.firm_net_worth()[-window:]
        profit_window = self.firm_profit()[-window:]
        price_window = self.food_price()[-window:]
        unit_labor_cost_window = self.unit_labor_cost()[-window:]
        unit_labor_cost_growth_window = self.unit_labor_cost_growth()[-window:]
        price_inventory_gap_window = self.price_inventory_gap()[-window:]
        price_cost_growth_window = self.price_cost_growth()[-window:]
        price_signal_window = self.price_inflation_signal()[-window:]
        price_log_adjustment_window = self.price_log_adjustment()[-window:]
        demand_pressure_window = self.demand_pressure()[-window:]
        ratio_window = self.firm_sales_income_ratio()[-window:]
        money_issued_window = self.money_issued()[-window:]
        loan_issued_window = self.working_capital_loan_issued()[-window:]
        target_cash_window = self.working_capital_target_cash()[-window:]
        funding_gap_window = self.working_capital_funding_gap()[-window:]
        loan_repaid_window = self.working_capital_loan_repaid()[-window:]
        loan_balance_window = self.working_capital_loan_balance()[-window:]
        inventory_units_window = self.food_inventory_units()[-window:]
        inventory_ratio_window = self.food_inventory_demand_ratio()[-window:]
        cb_purchase_window = self.central_bank_inventory_purchase()[-window:]
        cb_release_window = self.central_bank_market_release_revenue()[-window:]
        cb_subsidy_window = self.central_bank_poverty_subsidy_value()[-window:]
        cb_public_income_window = self.central_bank_public_income()[-window:]
        cb_public_used_window = self.central_bank_public_income_used()[-window:]
        cb_net_issued_window = self.central_bank_net_money_issued()[-window:]
        cb_inventory_window = self.central_bank_food_inventory_units()[-window:]
        cb_public_balance_window = (
            self.central_bank_public_income_balance()[-window:]
        )
        accounting_gap_window = self.monetary_accounting_gap()[-window:]

        cash_change = self.firm_cash()[-1] - self.firm_cash()[0]
        price_change = self.food_price()[-1] - self.food_price()[0]
        negative_cash_steps = sum(value < 0 for value in self.firm_cash())
        negative_cash_share = negative_cash_steps / len(self.firm_cash())

        print("\nFinal Firm State")
        print("----------------")
        print(f"Firm Cash: {self.firm_cash()[-1]:.2f}")
        print(f"Firm Inventory Value: {self.firm_inventory()[-1]:.2f}")
        print(f"Firm Net Worth: {self.firm_net_worth()[-1]:.2f}")
        if self.household_planning_price():
            print(f"Household Planning Price: {self.household_planning_price()[-1]:.3f}")
            print(f"Realized Transaction Price: {self.realized_transaction_price()[-1]:.3f}")
        print(f"Legacy Food Price (Compatibility): {self.food_price()[-1]:.3f}")
        print(f"Sales / Wage+Dividend Income: {self.firm_sales_income_ratio()[-1]:.3f}")

        if self.food_inventory_units():
            print(f"Food Inventory Units: {self.food_inventory_units()[-1]:.2f}")
            print(
                "Food Inventory / Demand: "
                f"{self.food_inventory_demand_ratio()[-1]:.3f}"
            )

        print("\nFirm Stability")
        print("--------------")
        print(f"Full-Window Cash Change: {cash_change:.2f}")
        print(f"Full-Window Cash Slope / Week: {self.series_slope(self.firm_cash()):.2f}")
        print(f"Last {window} Weeks Cash Slope / Week: {self.series_slope(cash_window):.2f}")
        print(f"Last {window} Weeks Inventory Slope / Week: {self.series_slope(inventory_window):.2f}")
        print(f"Last {window} Weeks Net Worth Slope / Week: {self.series_slope(net_worth_window):.2f}")
        print(f"Negative Firm-Cash Steps: {negative_cash_steps} ({negative_cash_share:.2%})")

        if self.food_inventory_units():
            print(
                f"Last {window} Weeks Food Inventory Units Slope / Week: "
                f"{self.series_slope(inventory_units_window):.2f}"
            )
            print(
                f"Last {window} Weeks Inventory / Demand Slope / Week: "
                f"{self.series_slope(inventory_ratio_window):.6f}"
            )

        print("\nFirm Flow Diagnostics")
        print("---------------------")
        print(f"Average Profit Before Dividend: {np.mean(self.firm_profit()):.2f}")
        print(f"Last {window} Weeks Average Profit Before Dividend: {np.mean(profit_window):.2f}")
        print(f"Average Sales / Income Ratio: {np.mean(self.firm_sales_income_ratio()):.3f}")
        print(f"Last {window} Weeks Sales / Income Ratio: {np.mean(ratio_window):.3f}")
        print(f"Legacy Food Price Change: {price_change:.3f}")
        print(f"Last {window} Weeks Legacy Food Price Slope / Week: {self.series_slope(price_window):.6f}")

        if self.unit_labor_cost():
            print("\nPrice Mechanism")
            print("---------------")
            print(f"Final Unit Labor Cost: {self.unit_labor_cost()[-1]:.6f}")
            print(
                f"Last {window} Weeks Average Week-over-Week Unit Labor Cost Growth: "
                f"{np.mean(unit_labor_cost_growth_window):.6f}"
            )
            print(
                f"Last {window} Weeks Average Inventory Gap: "
                f"{np.mean(price_inventory_gap_window):.6f}"
            )
            print(
                f"Last {window} Weeks Average Weekly Internal Cost-Growth Signal: "
                f"{np.mean(price_cost_growth_window):.6f}"
            )
            print(
                f"Last {window} Weeks Average Weekly Internal Price-Adjustment Signal: "
                f"{np.mean(price_signal_window):.6f}"
            )
            print(
                f"Last {window} Weeks Average Price Log Adjustment: "
                f"{np.mean(price_log_adjustment_window):.6f}"
            )
            print(
                f"Last {window} Weeks Average Demand Pressure: "
                f"{np.mean(demand_pressure_window):.6f}"
            )

        if self.money_issued():
            money_stock_window = self.cumulative_money_issued()[-window:]
            consumption_window = self.consumption()[-window:]
            household_wealth_window = self.household_wealth()[-window:]
            firm_cash_window = self.firm_cash()[-window:]
            private_money = [
                firm_cash + household_wealth
                for firm_cash, household_wealth in zip(
                    self.firm_cash(),
                    self.household_wealth(),
                )
            ]
            private_money_window = private_money[-window:]
            final_money_to_consumption = (
                self.cumulative_money_issued()[-1]
                /
                max(1.0, np.mean(consumption_window))
                if consumption_window
                else 0.0
            )
            print("\nCredit-Created Money and Total Money")
            print("-----------------------------------")
            print(
                "Credit-Created Money Change: "
                f"{self.cumulative_money_issued()[-1]:.2f}"
            )
            print(
                "Last "
                f"{window} Weeks Total Money Stock Slope / Week: "
                f"{self.series_slope(money_stock_window):.2f}"
            )
            print(f"Average Net Money Issued / Week: {np.mean(self.central_bank_net_money_issued()):.2f}")
            print(
                f"Last {window} Weeks Net Money Issued / Week: "
                f"{np.mean(cb_net_issued_window):.2f}"
            )
            print(
                "Last "
                f"{window} Weeks Net Money Issued Slope / Week: "
                f"{self.series_slope(cb_net_issued_window):.2f}"
            )
            print(
                "Credit-Created Money / Last "
                f"{window} Weeks Avg Consumption: "
                f"{final_money_to_consumption:.3f}"
            )
            if self.working_capital_loan_balance():
                print(
                    "Working-Capital Semantics: payroll-anchored base buffer "
                    "plus current Firm wage component; credit capacity is scenario-configured"
                )
                print(
                    "Principal Repayment: 35% weekly target/cap, cash constrained"
                )
                print(
                    "Final Working Capital Loan Balance: "
                    f"{self.working_capital_loan_balance()[-1]:.2f}"
                )
                if self.working_capital_target_cash():
                    print(
                        "Final Working Capital Target Cash: "
                        f"{self.working_capital_target_cash()[-1]:.2f}"
                    )
                    print(
                        "Final Working Capital Funding Gap: "
                        f"{self.working_capital_funding_gap()[-1]:.2f}"
                    )
                print(
                    "Last "
                    f"{window} Weeks Loan Issued / Week: "
                    f"{np.mean(loan_issued_window):.2f}"
                )
                if self.working_capital_target_cash():
                    print(
                        "Last "
                        f"{window} Weeks Avg Target Cash: "
                        f"{np.mean(target_cash_window):.2f}"
                    )
                    print(
                        "Last "
                        f"{window} Weeks Avg Funding Gap: "
                        f"{np.mean(funding_gap_window):.2f}"
                    )
                print(
                    "Last "
                    f"{window} Weeks Loan Repaid / Week: "
                    f"{np.mean(loan_repaid_window):.2f}"
                )
                print(
                    "Last "
                    f"{window} Weeks Loan Balance Slope / Week: "
                    f"{self.series_slope(loan_balance_window):.2f}"
                )
            print(
                "Last "
                f"{window} Weeks Gross CB Inventory Purchase / Week: "
                f"{np.mean(cb_purchase_window):.2f}"
            )
            print(
                "Last "
                f"{window} Weeks Public Wealth Income / Week: "
                f"{np.mean(cb_public_income_window):.2f}"
            )
            print(
                "Last "
                f"{window} Weeks Public Wealth Used / Week: "
                f"{np.mean(cb_public_used_window):.2f}"
            )

            if self.household_wealth():
                print("\nMoney Distribution")
                print("------------------")
                print(
                    "Final Household Cash Wealth: "
                    f"{self.household_wealth()[-1]:.2f}"
                )
                print(
                    "Household Cash Wealth Change: "
                    f"{self.household_wealth()[-1] - self.household_wealth()[0]:.2f}"
                )
                print(
                    "Last "
                    f"{window} Weeks Household Cash Wealth Slope / Week: "
                    f"{self.series_slope(household_wealth_window):.2f}"
                )
                print(
                    "Last "
                    f"{window} Weeks Firm Cash Slope / Week: "
                    f"{self.series_slope(firm_cash_window):.2f}"
                )
                print(
                    "Final Private Cash Stock: "
                    f"{private_money[-1]:.2f}"
                )
                print(
                    "Last "
                    f"{window} Weeks Private Cash Stock Slope / Week: "
                    f"{self.series_slope(private_money_window):.2f}"
                )
                print(
                    "Household Cash Wealth / Total Money Stock: "
                    f"{self.household_wealth()[-1] / max(1.0, self.expected_money_stock()[-1] if self.expected_money_stock() else 1.0):.3f}"
                )

            if self.private_cash_stock() and self.expected_money_stock():
                print("\nMonetary Accounting")
                print("-------------------")
                print(
                    "Initial Private Cash Stock: "
                    f"{getattr(self.world, 'initial_private_money_stock', 0.0):.2f}"
                )
                print(
                    "Final Expected Money Stock: "
                    f"{self.expected_money_stock()[-1]:.2f}"
                )
                print(
                    "Final Private Cash Stock: "
                    f"{self.private_cash_stock()[-1]:.2f}"
                )
                if self.located_money_stock():
                    print(
                        "Final Located Money Stock: "
                        f"{self.located_money_stock()[-1]:.2f}"
                    )
                print(
                    "Final Central Bank Public Income Balance: "
                    f"{self.central_bank_public_income_balance()[-1]:.2f}"
                )
                print(
                    "Final Uncollected Public Wealth: "
                    f"{getattr(self.world, 'public_wealth', 0.0):.2f}"
                )
                print(
                    "Final Monetary Accounting Gap: "
                    f"{self.monetary_accounting_gap()[-1]:.2f}"
                )
                print(
                    "Last "
                    f"{window} Weeks CB Public Balance Slope / Week: "
                    f"{self.series_slope(cb_public_balance_window):.2f}"
                )
                print(
                    "Last "
                    f"{window} Weeks Accounting Gap Slope / Week: "
                    f"{self.series_slope(accounting_gap_window):.2f}"
                )

        if self.central_bank_food_inventory_units():
            print("\nCentral Bank Buffer Stock")
            print("-------------------------")
            print(
                "Final Central Bank Food Inventory Units: "
                f"{self.central_bank_food_inventory_units()[-1]:.2f}"
            )
            print(
                "Last "
                f"{window} Weeks CB Food Inventory Slope / Week: "
                f"{self.series_slope(cb_inventory_window):.2f}"
            )
            print(
                "Last "
                f"{window} Weeks Inventory Purchase / Week: "
                f"{np.mean(cb_purchase_window):.2f}"
            )
            print(
                "Last "
                f"{window} Weeks Market Release Revenue / Week: "
                f"{np.mean(cb_release_window):.2f}"
            )
            print(
                "Last "
                f"{window} Weeks Poverty Subsidy Value / Week: "
                f"{np.mean(cb_subsidy_window):.2f}"
            )

    def plot_firm_diagnostics(self):
        if not self.has_firm_history():
            print("No firm history recorded.")
            return

        has_food_units = bool(self.food_inventory_units())
        has_money_issuer = bool(self.money_issued())
        has_central_bank = bool(self.central_bank_food_inventory_units())
        rows = 3 if has_food_units or has_money_issuer else 2
        fig, ax = plt.subplots(rows, 3, figsize=(15, 4 * rows))

        ax[0, 0].plot(self.firm_cash(), label="Cash")
        ax[0, 0].axhline(0, linestyle="--", linewidth=1)
        ax[0, 0].set_title("Firm Cash")

        ax[0, 1].plot(self.firm_inventory(), label="Inventory")
        ax[0, 1].set_title("Firm Inventory Value")

        ax[0, 2].plot(self.firm_net_worth(), label="Net Worth")
        ax[0, 2].set_title("Firm Net Worth")

        ax[1, 0].plot(self.firm_profit(), label="Profit Before Dividend")
        ax[1, 0].axhline(0, linestyle="--", linewidth=1)
        ax[1, 0].set_title("Profit Before Dividend")

        if self.household_planning_price():
            ax[1, 1].plot(self.household_planning_price(), label="Planning price")
            ax[1, 1].plot(self.realized_transaction_price(), label="Realized price")
        ax[1, 1].plot(self.food_price(), label="Legacy price", alpha=0.55)
        ax[1, 1].set_title("Market Price Concepts")

        ax[1, 2].plot(self.firm_sales_income_ratio(), label="Sales / Income")
        ax[1, 2].axhline(1, linestyle="--", linewidth=1)
        ax[1, 2].set_title("Sales / Wage+Dividend Income")

        if has_food_units:
            ax[2, 0].plot(self.food_output_units(), label="Output Units")
            ax[2, 0].plot(self.food_demand_units(), label="Demand Units")
            ax[2, 0].plot(self.food_sales_units(), label="Sales Units")
            ax[2, 0].set_title("Real Food Flow")

            ax[2, 1].plot(self.food_inventory_units(), label="Inventory Units")
            ax[2, 1].set_title("Real Food Inventory")

            ax[2, 2].plot(
                self.food_inventory_demand_ratio(),
                label="Inventory / Demand",
            )
            ax[2, 2].set_title("Inventory Coverage (Weeks)")

        if has_money_issuer:
            target_axis = ax[2, 2] if has_food_units else ax[1, 2]
            target_axis.plot(self.central_bank_net_money_issued(), label="Net Money Issued")
            target_axis.set_title("Inventory / Demand + Issuance")

        if has_central_bank:
            ax[2, 1].plot(
                self.central_bank_food_inventory_units(),
                label="CB Food Inventory",
            )
            ax[2, 2].plot(
                self.central_bank_inventory_purchase(),
                label="CB Purchase Gross",
            )
            ax[2, 2].plot(
                self.central_bank_public_income(),
                label="Public Wealth Income",
            )
            ax[2, 2].plot(
                self.central_bank_public_income_used(),
                label="Public Wealth Used",
            )
            ax[2, 2].plot(
                self.central_bank_net_money_issued(),
                label="Net Money Issued",
            )
            ax[2, 2].plot(
                self.central_bank_market_release_revenue(),
                label="CB Release Revenue",
            )
            ax[2, 2].plot(
                self.central_bank_poverty_subsidy_value(),
                label="CB Poverty Subsidy",
            )
            ax[2, 2].set_title("Central Bank Flows")

        for axis in ax.ravel():
            axis.legend()

        self.apply_week_axes(ax)

        plt.tight_layout()
        plt.show()

    def plot_multi_firm_diagnostics(self):
        if not self.has_multi_firm_diagnostics():
            print("No multi-firm diagnostics recorded.")
            return

        field_titles = [
            ("price", "Price"),
            ("price_change", "Price Change"),
            ("choice_probability", "Choice Probability"),
            ("actual_market_share", "Actual Market Share"),
            ("sales", "Sales Revenue"),
            ("inventory", "Inventory Value"),
            ("loan_balance", "Loan Balance"),
            ("cash", "Cash"),
            ("cash_to_wage_bill", "Cash / Wage Bill"),
            ("observed_demand", "Observed Demand"),
            ("expected_demand", "Expected Demand"),
            ("production_plan", "Production Plan"),
            ("actual_production", "Actual Production"),
            ("productive_capacity", "Productive Capacity"),
            ("capacity_utilization", "Capacity Utilization"),
        ]
        series_by_field = {
            field: self.firm_series_by_id(field)
            for field, _ in field_titles
        }
        weeks_by_firm = self.firm_weeks_by_id()

        fig, ax = plt.subplots(5, 3, figsize=(16, 20))

        for axis, (field, title) in zip(ax.ravel(), field_titles):
            for firm_id, values in sorted(series_by_field[field].items()):
                axis.plot(weeks_by_firm.get(firm_id, self.simulation_weeks(len(values))), values, label=f"Firm {firm_id}")

            axis.set_title(title)

            if field in {
                "choice_probability",
                "actual_market_share",
            }:
                axis.axhline(
                    1.0 / max(1, len(series_by_field[field])),
                    linestyle="--",
                    linewidth=1,
                )

            axis.legend()

        self.apply_week_axes(ax)

        plt.tight_layout()
        plt.show()

    def plot_multi_firm_market_shares(self):
        if not self.has_multi_firm_diagnostics():
            print("No multi-firm diagnostics recorded.")
            return

        actual = self.firm_series_by_id("actual_market_share")
        expected = self.firm_series_by_id("choice_probability")
        weeks_by_firm = self.firm_weeks_by_id()

        fig, ax = plt.subplots(1, 2, figsize=(14, 5))

        for firm_id, values in sorted(expected.items()):
            ax[0].plot(weeks_by_firm.get(firm_id, self.simulation_weeks(len(values))), values, label=f"Firm {firm_id}")

        ax[0].set_title("Expected Choice Share")

        for firm_id, values in sorted(actual.items()):
            ax[1].plot(weeks_by_firm.get(firm_id, self.simulation_weeks(len(values))), values, label=f"Firm {firm_id}")

        ax[1].set_title("Actual Market Share")

        for axis in ax:
            axis.legend()

        self.apply_week_axes(ax)

        plt.tight_layout()
        plt.show()

    def plot_firm_balance_sheet(self):
        if not self.has_firm_history():
            print("No firm history recorded.")
            return

        plt.figure(figsize=(10, 5))
        plt.plot(self.firm_cash(), label="Cash")
        plt.plot(self.firm_inventory(), label="Inventory")
        plt.plot(self.firm_net_worth(), label="Net Worth")
        plt.axhline(0, linestyle="--", linewidth=1)
        plt.legend()
        plt.title("Firm Balance Sheet")
        plt.xlabel("Simulation week")
        plt.show()

    def plot_monetary_system(self):
        if not self.has_firm_history():
            print("No firm history recorded.")
            return

        if not self.cumulative_money_issued():
            print("No money supply history recorded.")
            return

        fig, ax = plt.subplots(2, 3, figsize=(16, 9))

        ax[0, 0].plot(
            self.cumulative_money_issued(),
            label="Credit-Created Money",
        )
        ax[0, 0].set_title("Credit-Created Money Stock")

        ax[0, 1].plot(
            self.central_bank_inventory_purchase(),
            label="CB Purchase Gross",
        )
        ax[0, 1].plot(
            self.central_bank_public_income(),
            label="Public Wealth Income",
        )
        ax[0, 1].plot(
            self.central_bank_public_income_used(),
            label="Public Wealth Used",
        )
        ax[0, 1].plot(
            self.central_bank_net_money_issued(),
            label="Net Money Issued",
        )
        ax[0, 1].plot(
            self.central_bank_market_release_revenue(),
            label="Market Release / Money Recovered",
        )
        ax[0, 1].plot(
            self.central_bank_poverty_subsidy_value(),
            label="In-Kind Subsidy Value",
        )
        ax[0, 1].set_title("Central Bank Flows")

        ax[0, 2].plot(self.household_planning_price(), label="Planning price")
        ax[0, 2].plot(self.realized_transaction_price(), label="Realized price")
        ax[0, 2].plot(self.food_price(), label="Legacy price", alpha=0.55)
        ax[0, 2].set_title("Market Price Concepts")

        ax[1, 0].plot(
            self.firm_cash(),
            label="Firm Cash",
        )
        if self.household_wealth():
            ax[1, 0].plot(
                self.household_wealth(),
                label="Household Wealth",
            )
        ax[1, 0].set_title("Private Monetary Positions")

        if self.consumption():
            money_to_consumption = [
                money / max(1.0, consumption)
                for money, consumption in zip(
                    self.cumulative_money_issued(),
                    self.consumption(),
                )
            ]
            ax[1, 1].plot(
                money_to_consumption,
                label="Money Supply / Consumption",
            )
        ax[1, 1].set_title("Money Relative to Consumption")

        ax[1, 2].plot(
            self.central_bank_food_inventory_units(),
            label="CB Food Inventory Units",
        )
        ax[1, 2].set_title("Central Bank Buffer Stock")

        for axis in ax.ravel():
            axis.legend()

        self.apply_week_axes(ax)

        plt.tight_layout()
        plt.show()

        self.plot_money_accounting()

    def plot_household_money_stock(self):
        if not self.has_firm_history():
            print("No firm history recorded.")
            return

        if not self.household_wealth():
            print("No household wealth history recorded.")
            return

        fig, ax = plt.subplots(2, 2, figsize=(13, 8))

        private_money = [
            firm_cash + household_wealth
            for firm_cash, household_wealth in zip(
                self.firm_cash(),
                self.household_wealth(),
            )
        ]

        ax[0, 0].plot(
            self.household_wealth(),
            label="Household Cash Wealth",
        )
        ax[0, 0].set_title("Household Cash Wealth Stock")

        ax[0, 1].plot(
            self.cumulative_money_issued(),
            label="Total Money Stock",
        )
        ax[0, 1].plot(
            self.household_wealth(),
            label="Household Cash Wealth",
        )
        ax[0, 1].set_title("Total Money Stock vs Household Cash")

        ax[1, 0].plot(
            self.firm_cash(),
            label="Firm Cash",
        )
        ax[1, 0].plot(
            self.household_wealth(),
            label="Household Cash Wealth",
        )
        ax[1, 0].set_title("Firm Cash vs Household Cash")

        ax[1, 1].plot(
            private_money,
            label="Firm Cash + Household Cash",
        )
        ax[1, 1].plot(
            self.cumulative_money_issued(),
            label="Total Money Stock",
        )
        ax[1, 1].set_title("Private Cash Stock vs Total Money Stock")

        for axis in ax.ravel():
            axis.legend()

        self.apply_week_axes(ax)

        plt.tight_layout()
        plt.show()

    def plot_money_accounting(self):
        if not self.expected_money_stock():
            print("No monetary accounting history recorded.")
            return

        fig, ax = plt.subplots(2, 2, figsize=(13, 8))

        ax[0, 0].plot(
            self.expected_money_stock(),
            label="Expected Money Stock",
        )
        if self.private_cash_stock():
            ax[0, 0].plot(
                self.private_cash_stock(),
                label="Private Cash Stock",
            )
        if self.located_money_stock():
            ax[0, 0].plot(
                self.located_money_stock(),
                label="Located Money Stock",
            )
        ax[0, 0].set_title("Expected vs Located Cash")

        ax[0, 1].plot(
            self.firm_cash(),
            label="Firm Cash",
        )
        if self.household_wealth():
            ax[0, 1].plot(
                self.household_wealth(),
                label="Household Cash",
            )
        ax[0, 1].plot(
            self.central_bank_public_income_balance(),
            label="CB Public Income Balance",
        )
        if self.public_wealth():
            ax[0, 1].plot(
                self.public_wealth(),
                label="Uncollected Public Wealth",
            )
        ax[0, 1].set_title("Cash Holders")

        ax[1, 0].plot(
            self.central_bank_inventory_purchase(),
            label="CB Purchase Gross",
        )
        ax[1, 0].plot(
            self.central_bank_public_income_used(),
            label="Public Wealth Used",
        )
        ax[1, 0].plot(
            self.central_bank_net_money_issued(),
            label="Net Money Issued",
        )
        ax[1, 0].set_title("Issuance Offset")

        ax[1, 1].plot(
            self.monetary_accounting_gap(),
            label="Accounting Gap",
        )
        ax[1, 1].axhline(0, linestyle="--", linewidth=1)
        ax[1, 1].set_title("Unlocated / Miscounted Money")

        for axis in ax.ravel():
            axis.legend()

        self.apply_week_axes(ax)

        plt.tight_layout()
        plt.show()

    def plot_central_bank_food_reserve(self):
        if not self.has_firm_history():
            print("No firm history recorded.")
            return

        reserve = self.central_bank_food_inventory_units()

        if not reserve:
            print("No central bank food reserve history recorded.")
            return

        plt.figure(figsize=(10, 5))
        plt.plot(
            reserve,
            label="Central Bank Food Reserve Units",
        )
        plt.axhline(0, linestyle="--", linewidth=1)
        plt.legend()
        plt.title("Central Bank Food Reserve Stock")
        plt.xlabel("Simulation week")
        plt.ylabel("Food Units")
        plt.tight_layout()
        plt.show()

