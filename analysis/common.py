import numpy as np

from time_system import STEPS_PER_YEAR, steps_to_years


class CommonAnalysis:

    def simulation_weeks(self, length=None):
        if length is None:
            length = len(getattr(self.world, "population_history", []))
        return np.arange(length, dtype=float)

    def model_years(self, length=None):
        return np.asarray([
            steps_to_years(week)
            for week in self.simulation_weeks(length)
        ])

    def elapsed_weeks(self, rows):
        if not rows:
            return np.array([], dtype=float)
        start = float(rows[0].get("global_step", rows[0].get("step", 0)))
        return np.asarray([
            float(row.get("global_step", row.get("step", 0))) - start
            for row in rows
        ])

    def apply_week_axis(self, axis):
        axis.set_xlabel("Simulation week")

    def apply_week_axes(self, axes):
        for axis in np.asarray(axes).ravel():
            self.apply_week_axis(axis)

    def rolling_year_window(self):
        return STEPS_PER_YEAR

    def active_households(self):
        return [
            household
            for household in self.world.households
            if household.size() > 0
        ]

    def household_type(self, household):
        parents = len(household.parents)
        children = len(household.children)

        if parents == 2:
            if children == 0:
                return "couple_no_children"
            if children == 1:
                return "couple_one_child"
            if children == 2:
                return "couple_two_children"
            return "couple_three_plus_children"

        if parents == 1:
            if children == 0:
                return "single_adult"
            if children == 1:
                return "single_parent_one_child"
            if children == 2:
                return "single_parent_two_children"
            return "single_parent_three_plus_children"

        if parents == 0 and children > 0:
            return "children_only"

        return "empty"

    def household_values(self, attr):
        return [
            getattr(household, attr, 0.0)
            for household in self.active_households()
        ]

    def distribution_summary(self, values):
        values = np.array(values, dtype=float)

        if len(values) == 0:
            return {
                "min": 0.0,
                "p05": 0.0,
                "p10": 0.0,
                "median": 0.0,
                "mean": 0.0,
                "p90": 0.0,
                "p95": 0.0,
                "max": 0.0,
            }

        return {
            "min": np.min(values),
            "p05": np.percentile(values, 5),
            "p10": np.percentile(values, 10),
            "median": np.median(values),
            "mean": np.mean(values),
            "p90": np.percentile(values, 90),
            "p95": np.percentile(values, 95),
            "max": np.max(values),
        }

    def gini_coefficient(self, values):
        values = np.array(values, dtype=float)
        values = values[values >= 0]

        if len(values) == 0:
            return 0.0

        total = np.sum(values)

        if total == 0:
            return 0.0

        values.sort()
        index = np.arange(1, len(values) + 1)

        return (
            np.sum((2 * index - len(values) - 1) * values)
            /
            (len(values) * total)
        )

    def print_distribution(self, name, values):
        summary = self.distribution_summary(values)

        print(f"\n{name}")
        print("-" * len(name))
        print(f"Mean: {summary['mean']:.2f}")
        print(f"Median: {summary['median']:.2f}")
        print(f"P05 / P10: {summary['p05']:.2f} / {summary['p10']:.2f}")
        print(f"P90 / P95: {summary['p90']:.2f} / {summary['p95']:.2f}")
        print(f"Min / Max: {summary['min']:.2f} / {summary['max']:.2f}")
        print(f"Gini: {self.gini_coefficient(values):.3f}")

    def household_type_counts(self):
        counts = {}

        for household in self.active_households():
            name = self.household_type(household)
            counts[name] = counts.get(name, 0) + 1

        return counts
