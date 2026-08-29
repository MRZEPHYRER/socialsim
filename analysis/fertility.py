from collections import Counter

import matplotlib.pyplot as plt
import numpy as np

from analysis.common import CommonAnalysis


class FertilityAnalysis(CommonAnalysis):

    def couple_birth_bucket(self, births):
        if births >= 4:
            return "4+"

        return str(births)

    def couple_birth_distribution(self, couples):
        counter = Counter(
            self.couple_birth_bucket(data["births"])
            for data in couples
        )

        return {
            "0": counter["0"],
            "1": counter["1"],
            "2": counter["2"],
            "3": counter["3"],
            "4+": counter["4+"],
        }

    def print_couple_birth_distribution(self, title, couples):
        total = len(couples)
        distribution = self.couple_birth_distribution(couples)
        births = [data["births"] for data in couples]

        print(f"\n{title}")
        print("-" * len(title))
        print(f"Couples: {total}")

        if total == 0:
            return

        print(f"Mean Lifetime Births: {np.mean(births):.3f}")
        print(f"Median Lifetime Births: {np.median(births):.3f}")
        print(f"P90 Lifetime Births: {np.percentile(births, 90):.3f}")
        print(f"{'Births':<8} {'Couples':>10} {'Share':>10}")

        for name in ["0", "1", "2", "3", "4+"]:
            count = distribution[name]
            print(f"{name:<8} {count:>10} {count / total:>9.2%}")

    def completed_fertility_report(self):
        fertility_system = self.world.fertility_system
        completed = fertility_system.completed_couples()
        exposed = fertility_system.fertility_exposed_completed_couples()
        incomplete = fertility_system.incomplete_couples()

        print("======================")
        print("Completed Fertility Report")
        print("======================")
        print(f"Tracked Couples: {len(fertility_system.couples)}")
        print(f"Completed Couples: {len(completed)}")
        print(f"Fertility-Exposed Completed Couples: {len(exposed)}")
        print(f"Incomplete Couples: {len(incomplete)}")

        self.print_couple_birth_distribution(
            "All Completed Couples",
            completed,
        )
        self.print_couple_birth_distribution(
            "Fertility-Exposed Completed Couples",
            exposed,
        )
        self.print_couple_birth_distribution(
            "Incomplete Couples",
            incomplete,
        )

    def plot_completed_fertility_distribution(self):
        fertility_system = self.world.fertility_system
        groups = [
            ("All Completed", fertility_system.completed_couples()),
            ("Exposed Completed", fertility_system.fertility_exposed_completed_couples()),
            ("Incomplete", fertility_system.incomplete_couples()),
        ]
        labels = ["0", "1", "2", "3", "4+"]

        fig, axes = plt.subplots(1, 3, figsize=(14, 4), sharey=True)

        for axis, (title, couples) in zip(axes, groups):
            distribution = self.couple_birth_distribution(couples)
            values = [distribution[label] for label in labels]
            axis.bar(labels, values, alpha=0.82)
            axis.set_title(title)
            axis.set_xlabel("Lifetime Births")
            axis.grid(alpha=0.25, axis="y")

        axes[0].set_ylabel("Couples")
        fig.tight_layout()
        plt.show()
