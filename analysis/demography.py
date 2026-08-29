import numpy as np
import matplotlib.pyplot as plt

from analysis.common import CommonAnalysis


class DemographyAnalysis(CommonAnalysis):

    def population(self):
        return self.world.population_history

    def average_population(self):
        return np.mean(self.world.population_history)

    def dependency_ratio(self):
        result = []

        for group in self.world.age_group_history:
            children = group["children"]
            workers = group["workers"]
            elderly = group["elderly"]

            if workers > 0:
                result.append((children + elderly) / workers)
            else:
                result.append(0)

        return result

    def labor_ratio(self):
        result = []

        for i, group in enumerate(self.world.age_group_history):
            population = self.world.population_history[i]

            if population > 0:
                result.append(group["workers"] / population)
            else:
                result.append(0)

        return result

    def aging_index(self):
        result = []

        for group in self.world.age_group_history:
            children = group["children"]
            elderly = group["elderly"]

            if children > 0:
                result.append(elderly / children)
            else:
                result.append(0)

        return result

    def natural_growth_rate(self):
        return [
            b - d
            for b, d in zip(
                self.world.birth_rate_history,
                self.world.death_rate_history,
            )
        ]

    def stability_index(self):
        mean = np.mean(self.world.population_history)
        std = np.std(self.world.population_history)

        if std == 0:
            return np.inf

        return mean / std

    def equilibrium_report(self):
        window_weeks = min(200, len(self.world.population_history))
        population = np.mean(self.world.population_history[-window_weeks:])
        groups = self.world.age_group_history[-window_weeks:]

        children = np.mean([x["children"] for x in groups])
        workers = np.mean([x["workers"] for x in groups])
        elderly = np.mean([x["elderly"] for x in groups])

        print("======================")
        print("Equilibrium Report")
        print("======================")
        print(f"Window: last {window_weeks} weeks")
        print(f"Average Population: {population:.2f}")
        print(f"Children: {children:.2f} ({children/population:.2%})")
        print(f"Workers: {workers:.2f} ({workers/population:.2%})")
        print(f"Elderly: {elderly:.2f} ({elderly/population:.2%})")
        print(f"Dependency Ratio: {(children+elderly)/workers:.3f}")
        print(f"Aging Index: {elderly/children:.3f}")
        print(f"Weekly Realized Birth Event Rate: {np.mean(self.world.birth_rate_history[-window_weeks:]):.4f}")
        print(f"Weekly Realized Death Event Rate: {np.mean(self.world.death_rate_history[-window_weeks:]):.4f}")
        print("Marriage Market Cadence: every 52 weeks")
        print("Minimum Birth Spacing: 52 weeks")
        print(f"Stability Index: {self.stability_index():.2f}")

    def plot_age_heatmap(self):
        data = np.array(self.world.age_distribution_history).T

        plt.figure(figsize=(12, 6))
        plt.imshow(data, aspect="auto", origin="lower")
        plt.colorbar(label="Population")
        plt.xlabel("Simulation week")
        plt.ylabel("Age (calendar years)")
        plt.title("Age Structure Evolution")
        plt.show()

    def plot_population_pyramid(self):
        age_distribution = self.world.age_distribution_history[-1]
        ages = np.arange(101)

        plt.figure(figsize=(6, 8))
        plt.barh(ages, age_distribution)
        plt.xlabel("Population")
        plt.ylabel("Age (calendar years)")
        plt.title("Current Population Pyramid")
        plt.show()

    def plot_dependency_ratio(self):
        plt.figure(figsize=(10, 4))
        plt.plot(self.simulation_weeks(len(self.dependency_ratio())), self.dependency_ratio())
        plt.title("Dependency Ratio")
        plt.xlabel("Simulation week")
        plt.ylabel("Ratio")
        plt.show()
