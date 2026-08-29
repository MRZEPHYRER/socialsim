import numpy as np
import matplotlib.pyplot as plt

from analysis.common import CommonAnalysis


class EconomyAnalysis(CommonAnalysis):

    def gdp(self):
        """Legacy alias for weekly nominal output value."""
        return self.world.gdp_history

    def average_gdp(self):
        return np.mean(self.world.gdp_history)

    def consumption(self):
        return self.world.consumption_history

    def average_consumption(self):
        return np.mean(self.world.consumption_history)

    def saving(self):
        return self.world.saving_history

    def average_saving(self):
        return np.mean(self.world.saving_history)

    def consumption_rate(self):
        return self.world.consumption_rate_history

    def saving_rate(self):
        return self.world.saving_rate_history

    def gdp_per_capita(self):
        result = []

        for gdp, pop in zip(
            self.world.gdp_history,
            self.world.population_history,
        ):
            if pop > 0:
                result.append(gdp / pop)
            else:
                result.append(0)

        return result

    def living_person(self, person_id):
        person = self.world.get_person_by_id(person_id)

        if person is None or not person.alive:
            return None

        return person

    def household_record(self, household):
        members = [
            self.living_person(person_id)
            for person_id in household.parents + household.children
        ]
        members = [
            person
            for person in members
            if person is not None
        ]
        size = max(1, len(members))
        children = sum(1 for person in members if person.age < 20)
        workers = sum(1 for person in members if 20 <= person.age <= 60)
        elderly = sum(1 for person in members if person.age > 60)
        wealth = household.wealth
        income = getattr(household, "income_this_step", 0.0)
        consumption = getattr(household, "consumption_this_step", 0.0)
        saving = getattr(household, "saving_this_step", 0.0)
        subsidy = getattr(household, "food_subsidy_value_this_step", 0.0)
        pressure = getattr(household, "economic_pressure_this_step", 0.0)

        return {
            "id": household.id,
            "type": self.household_type(household),
            "size": size,
            "children": children,
            "workers": workers,
            "elderly": elderly,
            "wealth": wealth,
            "income": income,
            "consumption": consumption,
            "saving": saving,
            "subsidy": subsidy,
            "real_consumption": consumption + subsidy,
            "necessary": getattr(household, "necessary_consumption_this_step", 0.0),
            "desired": getattr(household, "desired_consumption_this_step", 0.0),
            "affordable": getattr(household, "affordable_consumption_this_step", 0.0),
            "pressure": pressure if np.isfinite(pressure) else np.nan,
            "wealth_pc": wealth / size,
            "income_pc": income / size,
            "consumption_pc": consumption / size,
            "saving_pc": saving / size,
            "budget_constrained": getattr(
                household,
                "budget_constrained_this_step",
                False,
            ),
            "food_constrained_after_subsidy": getattr(
                household,
                "food_constrained_after_subsidy_this_step",
                False,
            ),
            "negative_saving": saving < 0,
            "zero_wealth": wealth <= 0,
        }

    def household_records(self):
        return [
            self.household_record(household)
            for household in self.active_households()
        ]

    def record_column(self, records, name):
        return np.array(
            [record[name] for record in records],
            dtype=float,
        )

    def safe_mean(self, values):
        values = np.asarray(values, dtype=float)
        values = values[np.isfinite(values)]

        if len(values) == 0:
            return 0.0

        return float(np.mean(values))

    def safe_median(self, values):
        values = np.asarray(values, dtype=float)
        values = values[np.isfinite(values)]

        if len(values) == 0:
            return 0.0

        return float(np.median(values))

    def safe_percentile(self, values, pct):
        values = np.asarray(values, dtype=float)
        values = values[np.isfinite(values)]

        if len(values) == 0:
            return 0.0

        return float(np.percentile(values, pct))

    def add_quantile_groups(self, records, key, group_key, groups):
        if not records:
            return

        ordered = sorted(
            records,
            key=lambda record: (record[key], record["id"]),
        )

        for rank, record in enumerate(ordered):
            group = min(groups, int(rank * groups / len(ordered)) + 1)
            record[group_key] = f"Q{group}"

    def grouped_records(self, records, key):
        grouped = {}

        for record in records:
            grouped.setdefault(record[key], []).append(record)

        return grouped

    def group_summary(self, records):
        return {
            "n": len(records),
            "size": self.safe_mean(self.record_column(records, "size")),
            "wealth": self.safe_mean(self.record_column(records, "wealth")),
            "wealth_pc": self.safe_mean(self.record_column(records, "wealth_pc")),
            "income": self.safe_mean(self.record_column(records, "income")),
            "income_pc": self.safe_mean(self.record_column(records, "income_pc")),
            "consumption": self.safe_mean(
                self.record_column(records, "consumption")
            ),
            "saving": self.safe_mean(self.record_column(records, "saving")),
            "subsidy": self.safe_mean(self.record_column(records, "subsidy")),
            "pressure": self.safe_median(self.record_column(records, "pressure")),
            "pressure_p90": self.safe_percentile(
                self.record_column(records, "pressure"),
                90,
            ),
            "negative_saving": self.safe_mean(
                self.record_column(records, "negative_saving")
            ),
            "zero_wealth": self.safe_mean(
                self.record_column(records, "zero_wealth")
            ),
            "budget_constrained": self.safe_mean(
                self.record_column(records, "budget_constrained")
            ),
            "food_constrained_after_subsidy": self.safe_mean(
                self.record_column(records, "food_constrained_after_subsidy")
            ),
        }

    def print_group_table(self, title, groups):
        print(f"\n{title}")
        print("-" * len(title))
        print(
            f"{'group':<24} {'n':>6} {'size':>5} {'wealth':>9} "
            f"{'w pc':>8} {'income':>8} {'i pc':>7} {'cons':>8} "
            f"{'save':>8} {'subsidy':>8} {'medp':>7} {'p90p':>7} "
            f"{'zero':>7} {'negsv':>7} {'budget':>7} {'foodc':>7}"
        )

        for name, records in groups:
            summary = self.group_summary(records)
            print(
                f"{str(name):<24} "
                f"{summary['n']:>6} "
                f"{summary['size']:>5.2f} "
                f"{summary['wealth']:>9.2f} "
                f"{summary['wealth_pc']:>8.2f} "
                f"{summary['income']:>8.2f} "
                f"{summary['income_pc']:>7.2f} "
                f"{summary['consumption']:>8.2f} "
                f"{summary['saving']:>8.2f} "
                f"{summary['subsidy']:>8.2f} "
                f"{summary['pressure']:>7.2f} "
                f"{summary['pressure_p90']:>7.2f} "
                f"{summary['zero_wealth']:>7.2%} "
                f"{summary['negative_saving']:>7.2%} "
                f"{summary['budget_constrained']:>7.2%} "
                f"{summary['food_constrained_after_subsidy']:>7.2%}"
            )

    def top_share(self, values, share):
        values = np.asarray(values, dtype=float)
        values = values[np.isfinite(values)]
        values = values[values >= 0]

        if len(values) == 0:
            return 0.0

        total = np.sum(values)

        if total <= 0:
            return 0.0

        count = max(1, int(np.ceil(len(values) * share)))
        values.sort()

        return float(np.sum(values[-count:]) / total)

    def print_top_bottom_households(self, records, key, n=5):
        ordered = sorted(records, key=lambda record: record[key])

        for label, group in [
            ("Bottom", ordered[:n]),
            ("Top", ordered[-n:][::-1]),
        ]:
            print(f"\n{label} {n} Households by {key}")
            print("-" * (len(label) + len(key) + 18))
            print(
                f"{'id':>6} {'type':<28} {'size':>4} {'wealth':>9} "
                f"{'income':>8} {'cons':>8} {'save':>8} {'press':>7}"
            )

            for record in group:
                print(
                    f"{record['id']:>6} "
                    f"{record['type']:<28} "
                    f"{record['size']:>4} "
                    f"{record['wealth']:>9.2f} "
                    f"{record['income']:>8.2f} "
                    f"{record['consumption']:>8.2f} "
                    f"{record['saving']:>8.2f} "
                    f"{record['pressure']:>7.2f}"
                )

    def quantile_sort_key(self, item):
        name = str(item[0])

        if name.startswith("Q"):
            try:
                return int(name[1:])
            except ValueError:
                return name

        return name

    def household_inequality_report(self):
        records = self.household_records()

        if not records:
            print("\nHousehold Inequality Diagnostics")
            print("--------------------------------")
            print("No active households.")
            return

        self.add_quantile_groups(records, "wealth", "wealth_decile", 10)
        self.add_quantile_groups(records, "income", "income_decile", 10)

        wealth = self.record_column(records, "wealth")
        income = self.record_column(records, "income")
        consumption = self.record_column(records, "consumption")
        saving = self.record_column(records, "saving")

        print("\nHousehold Inequality Diagnostics")
        print("--------------------------------")
        print(f"Active Households: {len(records)}")
        print(f"Wealth Gini: {self.gini_coefficient(wealth):.3f}")
        print(f"Income Gini: {self.gini_coefficient(income):.3f}")
        print(f"Consumption Gini: {self.gini_coefficient(consumption):.3f}")
        print(f"Saving Gini: {self.gini_coefficient(saving):.3f}")
        print(f"Top 10% Wealth Share: {self.top_share(wealth, 0.10):.2%}")
        print(f"Top 20% Wealth Share: {self.top_share(wealth, 0.20):.2%}")
        print(f"Top 10% Income Share: {self.top_share(income, 0.10):.2%}")
        print(f"Top 20% Income Share: {self.top_share(income, 0.20):.2%}")

        self.print_group_table(
            "By Wealth Decile",
            sorted(
                self.grouped_records(records, "wealth_decile").items(),
                key=self.quantile_sort_key,
            ),
        )
        self.print_group_table(
            "By Income Decile",
            sorted(
                self.grouped_records(records, "income_decile").items(),
                key=self.quantile_sort_key,
            ),
        )
        self.print_group_table(
            "By Household Type",
            sorted(
                self.grouped_records(records, "type").items(),
                key=lambda item: len(item[1]),
                reverse=True,
            ),
        )

        self.print_top_bottom_households(records, "wealth")
        self.print_top_bottom_households(records, "income")

    def lorenz_curve(self, values):
        values = np.asarray(values, dtype=float)
        values = values[np.isfinite(values)]
        values = values[values >= 0]

        if len(values) == 0 or np.sum(values) <= 0:
            return np.array([0.0, 1.0]), np.array([0.0, 1.0])

        values.sort()
        cumulative = np.cumsum(values)
        cumulative = np.insert(cumulative, 0, 0.0)
        cumulative = cumulative / cumulative[-1]
        population = np.linspace(0.0, 1.0, len(cumulative))

        return population, cumulative

    def economy_report(self):
        print("======================")
        print("Economy Report")
        print("======================")

        window = min(200, len(self.world.income_history))
        active_households = self.active_households()

        wealth = self.household_values("wealth")
        income = self.household_values("income_this_step")
        consumption = self.household_values("consumption_this_step")
        saving = self.household_values("saving_this_step")
        necessary = self.household_values("necessary_consumption_this_step")
        desired = self.household_values("desired_consumption_this_step")
        affordable = self.household_values("affordable_consumption_this_step")
        pressure = [
            value
            for value in self.household_values("economic_pressure_this_step")
            if np.isfinite(value)
        ]

        negative_saving = [value for value in saving if value < 0]
        zero_wealth = [value for value in wealth if value == 0]
        constrained = [
            household
            for household in active_households
            if getattr(household, "budget_constrained_this_step", False)
        ]

        print("\nMacro Economy")
        print("-------------")
        print(f"Average Weekly Nominal Output Value: {self.average_gdp():.2f}")
        print(f"Average Weekly Consumption: {self.average_consumption():.2f}")
        print(f"Average Weekly Saving: {self.average_saving():.2f}")
        print(f"Average Consumption Rate: {np.mean(self.consumption_rate()):.3f}")
        print(f"Average Saving Rate: {np.mean(self.saving_rate()):.3f}")
        print(f"Average Weekly Nominal Output Per Capita: {np.mean(self.gdp_per_capita()):.2f}")

        if window > 0:
            print(f"Last {window} Weeks Average Nominal Output: {np.mean(self.world.gdp_history[-window:]):.2f}")
            print(f"Last {window} Weeks Average Consumption: {np.mean(self.world.consumption_history[-window:]):.2f}")
            print(f"Last {window} Weeks Average Saving: {np.mean(self.world.saving_history[-window:]):.2f}")

        print(f"Final Household Wealth: {self.world.wealth_history[-1]:.2f}")
        print(
            "Average Aggregate Household Wealth Over Time: "
            f"{np.mean(self.world.wealth_history):.2f}"
        )
        print(f"Final Wealth Per Capita: {self.world.wealth_history[-1]/self.world.population_history[-1]:.2f}")

        print("\nFinal Household Economy")
        print("-----------------------")
        print(f"Active Households: {len(active_households)}")
        print(f"Negative Saving Households: {len(negative_saving)} ({len(negative_saving)/len(active_households):.2%})")
        print(f"Zero Wealth Households: {len(zero_wealth)} ({len(zero_wealth)/len(active_households):.2%})")
        print(f"Budget Constrained Households: {len(constrained)} ({len(constrained)/len(active_households):.2%})")

        if len(pressure) > 0:
            print(f"Median Economic Pressure: {np.median(pressure):.3f}")
            print(f"P90 Economic Pressure: {np.percentile(pressure, 90):.3f}")

        self.print_distribution("Household Wealth Distribution", wealth)
        self.print_distribution("Household Income Distribution", income)
        self.print_distribution("Household Consumption Distribution", consumption)
        self.print_distribution("Household Saving Distribution", saving)
        self.print_distribution("Household Necessary Consumption Distribution", necessary)
        self.print_distribution("Household Desired Consumption Distribution", desired)
        self.print_distribution("Household Affordable Consumption Distribution", affordable)

        self.household_inequality_report()

        print("\nHousehold Type Distribution")
        print("---------------------------")

        for name, count in sorted(
            self.household_type_counts().items(),
            key=lambda item: item[1],
            reverse=True,
        ):
            print(f"{name}: {count} ({count/len(active_households):.2%})")

    def diagnostic_series(self, name):
        return [
            float(row.get(name, 0.0))
            for row in getattr(self.world, "diagnostics_rows", [])
        ]

    def normalized_window_slope(self, values, window=500):
        values = np.asarray(values, dtype=float)
        values = values[np.isfinite(values)]

        if len(values) < 2:
            return 0.0

        window = min(window, len(values))
        recent = values[-window:]
        mean = float(np.mean(recent))

        if abs(mean) < 1e-9:
            return 0.0

        x = np.arange(window, dtype=float)
        slope = float(np.polyfit(x, recent, 1)[0])

        return slope * window / mean

    def steady_state_report(self, window=500, stable_threshold=0.01):
        rows = getattr(self.world, "diagnostics_rows", [])

        if not rows:
            print("\nLegacy Heuristic Steady-State Diagnostic")
            print("----------------------------------------")
            print("This is a legacy heuristic trend screen, not the Analysis V2 mature-window acceptance verdict.")
            print("No diagnostics rows recorded.")
            return

        window = min(window, len(rows))
        last = rows[-1]

        def value(name):
            return float(last.get(name, 0.0))

        def mean(name):
            series = self.diagnostic_series(name)[-window:]
            return float(np.mean(series)) if series else 0.0

        def norm(name):
            return self.normalized_window_slope(
                self.diagnostic_series(name),
                window=window,
            )

        tracked = [
            ("Household Wealth", "total_household_wealth"),
            ("Median Security Ratio", "median_security_ratio"),
            ("Firm Cash", "firm_cash"),
            ("Total Money Stock", "total_money_stock"),
            ("Nominal Consumption", "total_consumption"),
            ("Simplified Money Velocity", "simplified_money_velocity"),
            ("Net Household Saving Rate", "net_household_saving_rate"),
            ("Population", "population"),
        ]

        print("\nLegacy Heuristic Steady-State Diagnostic")
        print("----------------------------------------")
        print("This is a legacy heuristic trend screen, not the Analysis V2 mature-window acceptance verdict.")
        print(f"Window: last {window} weeks")
        print(
            "Stable threshold: "
            f"|normalized slope| < {stable_threshold:.3f}"
        )
        print(
            "Normalized slope means projected window change / window mean."
        )

        print("\nWindow Trend")
        print("------------")
        print(f"{'Metric':32s} {'Mean':>14s} {'NormSlope':>12s} {'Stable':>8s}")

        for label, name in tracked:
            normalized = norm(name)
            stable = abs(normalized) < stable_threshold
            print(
                f"{label:32s} "
                f"{mean(name):14.4f} "
                f"{normalized:12.4f} "
                f"{str(stable):>8s}"
            )

        print("\nMoney Stock And Circulation")
        print("---------------------------")
        print(f"Total Money Stock: {value('total_money_stock'):.2f}")
        print(f"Net New Money: {value('net_new_money'):.2f}")
        print(
            "Credit Money Outstanding: "
            f"{value('credit_money_outstanding'):.2f}"
        )
        print(f"Money / Income: {value('money_income_ratio'):.3f}")
        print(f"Money / Consumption: {value('money_consumption_ratio'):.3f}")
        print(
            "Weekly Simplified Velocity (Consumption / Money): "
            f"{value('simplified_money_velocity'):.6f}"
        )

        print("\nMoney Location")
        print("--------------")
        print(f"Household Money Share: {value('household_money_share'):.3%}")
        print(f"Firm Money Share: {value('firm_money_share'):.3%}")
        print(f"Public Money Share: {value('public_money_share'):.3%}")
        print(
            "Share Sum: "
            f"{value('household_money_share') + value('firm_money_share') + value('public_money_share'):.3%}"
        )

        print("\nHousehold Security")
        print("------------------")
        print("Target Reserve Horizon: 6 calendar months = 26 weeks")
        print(
            "Household Wealth / Target Wealth: "
            f"{value('household_wealth_to_target_wealth'):.3f}"
        )
        print(f"Median Security Ratio: {value('median_security_ratio'):.3f}")
        print(f"P10 Security Ratio: {value('p10_security_ratio'):.3f}")
        print(f"P90 Security Ratio: {value('p90_security_ratio'):.3f}")
        print(
            "Households Below Target: "
            f"{value('share_households_below_target'):.2%}"
        )

        print("\nLiquidity And Saving")
        print("--------------------")
        print(f"Firm Cash / Wage Bill: {value('firm_cash_to_wage_bill'):.3f}")
        print(f"Net Household Saving: {value('net_household_saving'):.2f}")
        print(
            "Net Household Saving Rate: "
            f"{value('net_household_saving_rate'):.3%}"
        )

    def plot_macro_economy(self):
        plt.figure(figsize=(10, 5))
        weeks = self.simulation_weeks(len(self.world.gdp_history))
        plt.plot(weeks, self.world.gdp_history, label="Weekly nominal output")
        plt.plot(weeks, self.world.consumption_history, label="Weekly consumption")
        plt.plot(weeks, self.world.saving_history, label="Weekly saving")

        plt.legend()
        plt.title("Weekly Macro Flows")
        plt.xlabel("Simulation week")
        plt.show()

    def plot_economy_ratio(self):
        plt.figure(figsize=(10, 4))
        weeks = self.simulation_weeks(len(self.consumption_rate()))
        plt.plot(weeks, self.consumption_rate(), label="Weekly consumption rate")
        plt.plot(weeks, self.saving_rate(), label="Weekly saving rate")
        plt.legend()
        plt.title("Consumption / Saving Rate")
        plt.xlabel("Simulation week")
        plt.show()

    def plot_household_economy_distribution(self):
        data = [
            (self.household_values("wealth"), "Household Wealth"),
            (self.household_values("income_this_step"), "Weekly Household Income"),
            (self.household_values("consumption_this_step"), "Weekly Household Consumption"),
            (self.household_values("saving_this_step"), "Weekly Household Saving"),
        ]
        fig, ax = plt.subplots(2, 2, figsize=(12, 8))

        for axis, (values, title) in zip(ax.ravel(), data):
            axis.hist(values, bins=50, alpha=0.82)
            axis.axvline(np.median(values), linestyle="--", label="Median")
            axis.set_title(title)
            axis.set_ylabel("Households")
            axis.legend()

        plt.tight_layout()
        plt.show()

    def plot_household_economy_diagnostics(self):
        households = self.active_households()
        type_counts = self.household_type_counts()
        type_names = list(type_counts.keys())
        type_values = [type_counts[name] for name in type_names]

        pressure = [
            value
            for value in self.household_values("economic_pressure_this_step")
            if np.isfinite(value)
        ]

        constrained_by_type = []
        negative_saving_by_type = []

        for name in type_names:
            group = [
                household
                for household in households
                if self.household_type(household) == name
            ]

            constrained_by_type.append(
                sum(getattr(h, "budget_constrained_this_step", False) for h in group)
                /
                len(group)
            )
            negative_saving_by_type.append(
                sum(h.saving_this_step < 0 for h in group)
                /
                len(group)
            )

        fig, ax = plt.subplots(2, 2, figsize=(13, 9))
        weeks = self.simulation_weeks(len(self.world.wealth_history))
        ax[0, 0].plot(weeks, self.world.wealth_history, label="Household wealth stock")
        ax[0, 0].plot(weeks, self.world.gdp_history, label="Weekly nominal output")
        ax[0, 0].plot(weeks, self.world.consumption_history, label="Weekly consumption")
        ax[0, 0].set_title("Macro Stock and Weekly Flows")
        ax[0, 0].set_xlabel("Simulation week")
        ax[0, 0].legend()

        ax[0, 1].bar(type_names, type_values)
        ax[0, 1].set_title("Household Types")
        ax[0, 1].tick_params(axis="x", rotation=45)

        ax[1, 0].bar(type_names, negative_saving_by_type, label="Negative Saving")
        ax[1, 0].bar(type_names, constrained_by_type, alpha=0.65, label="Budget Constrained")
        ax[1, 0].set_title("Pressure by Household Type")
        ax[1, 0].tick_params(axis="x", rotation=45)
        ax[1, 0].legend()

        if len(pressure) > 0:
            ax[1, 1].hist(pressure, bins=50, alpha=0.82)

        ax[1, 1].set_title("Economic Pressure")
        ax[1, 1].set_xlabel("Desired / Affordable Consumption")
        ax[1, 1].set_ylabel("Households")

        plt.tight_layout()
        plt.show()

    def plot_household_inequality_diagnostics(self):
        records = self.household_records()

        if not records:
            print("No active households for inequality diagnostics.")
            return

        self.add_quantile_groups(records, "wealth", "wealth_decile", 10)
        wealth = self.record_column(records, "wealth")
        income = self.record_column(records, "income")
        consumption = self.record_column(records, "consumption")
        saving = self.record_column(records, "saving")
        pressure = self.record_column(records, "pressure")
        size = self.record_column(records, "size")

        fig, ax = plt.subplots(3, 3, figsize=(17, 13))

        for values, label in [
            (wealth, "Wealth"),
            (income, "Income"),
            (consumption, "Consumption"),
        ]:
            x, y = self.lorenz_curve(values)
            ax[0, 0].plot(
                x,
                y,
                label=f"{label} Gini={self.gini_coefficient(values):.3f}",
            )
        ax[0, 0].plot([0, 1], [0, 1], linestyle="--", linewidth=1)
        ax[0, 0].set_title("Lorenz Curves")
        ax[0, 0].legend()

        ax[0, 1].hist(wealth, bins=60, alpha=0.82)
        ax[0, 1].axvline(np.median(wealth), linestyle="--", linewidth=1)
        ax[0, 1].set_title("Wealth Distribution")

        ax[0, 2].hist(income, bins=60, alpha=0.82)
        ax[0, 2].axvline(np.median(income), linestyle="--", linewidth=1)
        ax[0, 2].set_title("Income Distribution")

        ax[1, 0].hist(consumption, bins=60, alpha=0.82)
        ax[1, 0].axvline(np.median(consumption), linestyle="--", linewidth=1)
        ax[1, 0].set_title("Consumption Distribution")

        ax[1, 1].hist(saving, bins=60, alpha=0.82)
        ax[1, 1].axvline(0, linestyle="--", linewidth=1)
        ax[1, 1].set_title("Saving Distribution")

        finite_pressure = pressure[np.isfinite(pressure)]
        if len(finite_pressure) > 0:
            ax[1, 2].hist(finite_pressure, bins=60, alpha=0.82)
        ax[1, 2].set_title("Economic Pressure")

        limit = max(
            1.0,
            float(np.max(income)) if len(income) else 0.0,
            float(np.max(consumption)) if len(consumption) else 0.0,
        )
        scatter = ax[2, 0].scatter(
            income,
            consumption,
            c=size,
            cmap="viridis",
            alpha=0.70,
        )
        ax[2, 0].plot([0, limit], [0, limit], linestyle="--", linewidth=1)
        ax[2, 0].set_xlabel("Income")
        ax[2, 0].set_ylabel("Consumption")
        ax[2, 0].set_title("Income vs Consumption")
        fig.colorbar(scatter, ax=ax[2, 0], label="Household Size")

        wealth_groups = sorted(
            self.grouped_records(records, "wealth_decile").items()
        )
        labels = [name for name, _ in wealth_groups]
        x = np.arange(len(labels))
        width = 0.28
        ax[2, 1].bar(
            x - width,
            [self.group_summary(group)["income"] for _, group in wealth_groups],
            width,
            label="Income",
        )
        ax[2, 1].bar(
            x,
            [self.group_summary(group)["consumption"] for _, group in wealth_groups],
            width,
            label="Consumption",
        )
        ax[2, 1].bar(
            x + width,
            [self.group_summary(group)["saving"] for _, group in wealth_groups],
            width,
            label="Saving",
        )
        ax[2, 1].axhline(0, linestyle="--", linewidth=1)
        ax[2, 1].set_xticks(x)
        ax[2, 1].set_xticklabels(labels)
        ax[2, 1].set_title("Flows by Wealth Decile")
        ax[2, 1].legend()

        type_groups = sorted(
            self.grouped_records(records, "type").items(),
            key=lambda item: len(item[1]),
            reverse=True,
        )[:8]
        labels = [name for name, _ in type_groups]
        x = np.arange(len(labels))
        ax[2, 2].bar(
            x - width / 2,
            [
                self.group_summary(group)["negative_saving"]
                for _, group in type_groups
            ],
            width,
            label="Negative Saving",
        )
        ax[2, 2].bar(
            x + width / 2,
            [
                self.group_summary(group)["budget_constrained"]
                for _, group in type_groups
            ],
            width,
            label="Budget Constrained",
        )
        ax[2, 2].set_xticks(x)
        ax[2, 2].set_xticklabels(labels, rotation=45, ha="right")
        ax[2, 2].set_title("Pressure by Household Type")
        ax[2, 2].legend()

        plt.tight_layout()
        plt.show()
