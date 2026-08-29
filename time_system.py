"""Passive calendar conversions for the Step 12 weekly time contract.

This module contains no simulation decisions.  Behavioural systems are not
wired to these helpers until a later, explicitly approved migration step.
"""

import math


STEPS_PER_YEAR = 52
WEEKS_PER_STEP = 1
YEARS_PER_STEP = 1 / STEPS_PER_YEAR
MONTHS_PER_YEAR = 12
MARRIAGE_MARKET_INTERVAL_WEEKS = 52


def steps_to_weeks(steps):
    return float(steps) * WEEKS_PER_STEP


def steps_to_years(steps):
    return float(steps) * YEARS_PER_STEP


def weeks_to_steps(weeks):
    return float(weeks) / WEEKS_PER_STEP


def years_to_steps(years):
    return float(years) * STEPS_PER_YEAR


def months_to_steps(months):
    return float(months) * STEPS_PER_YEAR / MONTHS_PER_YEAR


def annual_probability_to_step_probability(probability):
    probability = min(max(float(probability), 0.0), 1.0)
    return 1.0 - (1.0 - probability) ** (1.0 / STEPS_PER_YEAR)


def annual_hazard_to_step_probability(hazard):
    return 1.0 - math.exp(-max(0.0, float(hazard)) / STEPS_PER_YEAR)


def annual_rate_to_step_rate(rate):
    return (1.0 + float(rate)) ** (1.0 / STEPS_PER_YEAR) - 1.0


class SimulationClock:
    """A passive view of the technical step under the weekly convention."""

    steps_per_year = STEPS_PER_YEAR
    weeks_per_step = WEEKS_PER_STEP
    years_per_step = YEARS_PER_STEP

    def __init__(self, global_step=0):
        self.global_step = int(global_step)

    def set_step(self, global_step):
        self.global_step = int(global_step)

    def metadata(self, global_step=None):
        step = self.global_step if global_step is None else int(global_step)
        return {
            "global_step": step,
            "simulation_week": steps_to_weeks(step),
            "simulation_year": steps_to_years(step),
        }
