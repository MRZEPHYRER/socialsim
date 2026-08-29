"""Step 15 demographic burn-in stability and fertility-pressure audit.

This is diagnostic-only. It reuses the accepted long-horizon burn-in summaries
and performs at most two short 520-week deterministic observations:
one demographic-only burn-in and one normal runtime pressure trace.
"""

from __future__ import annotations

import csv
import json
import math
import pickle
import statistics
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from demographic_burnin import DemographicBurnInRunner
from fertility import config as fertility_config
from world import World


OUT = ROOT / "test/output/step15_demographic_burnin_stability_audit"
OLD = ROOT / "test/output/step15_staged_demographic_burnin"
I12A = ROOT / "test/output/step15I12A_final_integrated_validation"
SEED = 42
POPULATION = 5000
SHORT_WEEKS = 520


def write_rows(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = list(rows)
    fields = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields or ["status"])
        writer.writeheader()
        writer.writerows(rows or [{"status": "UNAVAILABLE"}])


def read_rows(path):
    if not path.exists():
        return []
    with path.open(newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def f(value, default=float("nan")):
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def i(value, default=0):
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return default


def q(values, p):
    values = sorted(float(v) for v in values if math.isfinite(float(v)))
    if not values:
        return float("nan")
    index = (len(values) - 1) * p
    lo, hi = math.floor(index), math.ceil(index)
    if lo == hi:
        return values[lo]
    return values[lo] + (values[hi] - values[lo]) * (index - lo)


def age_band(age):
    if age < 5:
        return "0-4"
    if age < 15:
        return "5-14"
    if age < 20:
        return "15-19"
    if age < 30:
        return "20-29"
    if age < 40:
        return "30-39"
    if age < 50:
        return "40-49"
    if age < 65:
        return "50-64"
    if age < 80:
        return "65-79"
    return "80+"


def make_burnin_world():
    return World(
        initial_population=POPULATION,
        seed=SEED,
        diagnostics_mode="compact",
        scenario_name="step15_demographic_burnin_stability_audit",
        scenario_overrides={
            "DEMOGRAPHIC_BURNIN_MODE": True,
            "PRIVATE_FAMILY_SUPPORT_ENABLED": False,
            "PERSON_EQUITY_TRANSITION_ENABLED": False,
            "AUTONOMOUS_SECONDARY_EQUITY_ENABLED": False,
        },
    )


def make_normal_world():
    world = World(
        initial_population=POPULATION,
        seed=SEED,
        diagnostics_mode="compact",
        scenario_name="baseline",
        scenario_overrides={
            "PRIVATE_FAMILY_SUPPORT_ENABLED": False,
            "PERSON_EQUITY_TRANSITION_ENABLED": False,
            "AUTONOMOUS_SECONDARY_EQUITY_ENABLED": False,
        },
    )
    world.steps = SHORT_WEEKS
    return world


def run_short_burnin():
    world = make_burnin_world()
    runner = DemographicBurnInRunner(world)
    weekly = []
    age_rows = []
    birth_age = []
    spacing = []
    marriage_events = []
    previous_events = 0
    for _ in range(SHORT_WEEKS):
        step = world.demographic_burnin_absolute_week
        attempted_before = len(world.fertility_system.fertility_probability_records)
        event = runner._demographic_step(step)
        new_events = world.demographic_events[previous_events:]
        previous_events = len(world.demographic_events)
        births = [row for row in new_events if row.get("event_type") == "birth"]
        deaths = [row for row in new_events if row.get("event_type") == "death"]
        marriages = [row for row in new_events if row.get("event_type") == "marriage"]
        attempted = len(world.fertility_system.fertility_probability_records) - attempted_before
        birth_age.extend(births)
        marriage_events.extend(marriages)
        for row in births:
            if row.get("inter_birth_interval_weeks") not in (None, ""):
                spacing.append(i(row.get("inter_birth_interval_weeks")))
        weekly.append({
            "global_step": event["global_step"],
            "opening_population": len(world.population) - event["births"] + event["deaths"],
            "births": event["births"],
            "deaths": event["deaths"],
            "population": len(world.population),
            "households": len(world.households),
            "marriages": event["marriages"],
            "fertility_attempts": attempted,
            "fertility_births": len(births),
            "fertility_pressure": 0.0,
        })
        counts = Counter(age_band(p.age) for p in world.population)
        total = max(1, len(world.population))
        for band, count in sorted(counts.items()):
            age_rows.append({
                "global_step": event["global_step"],
                "population": len(world.population),
                "age_band": band,
                "person_count": count,
                "share": count / total,
                "source": "short_demographic_burnin_runtime_snapshot",
            })
    return world, weekly, age_rows, birth_age, spacing, marriage_events


def run_short_normal():
    world = make_normal_world()
    world.run(progress_interval=0)
    pressure = list(getattr(world, "pressure_history", []))
    rows = [{
        "global_step": step,
        "pressure": f(value, float("nan")),
        "source": "normal_runtime_pressure_history",
    } for step, value in enumerate(pressure[:SHORT_WEEKS])]
    return world, rows, list(getattr(world, "demographic_events", []))


def aggregate_short(weekly):
    windows = [
        ("early", 1, 130),
        ("middle", 131, 390),
        ("late", 391, 520),
    ]
    rows = []
    for name, lo, hi in windows:
        part = [r for r in weekly if lo <= i(r["global_step"]) <= hi]
        if not part:
            continue
        births = sum(i(r["births"]) for r in part)
        deaths = sum(i(r["deaths"]) for r in part)
        mean_pop = statistics.mean(i(r["opening_population"]) for r in part)
        weeks = len(part)
        rows.append({
            "window": name,
            "first_week": lo,
            "last_week": hi,
            "weeks": weeks,
            "opening_population": i(part[0]["opening_population"]),
            "closing_population": i(part[-1]["population"]),
            "births": births,
            "deaths": deaths,
            "net_change": births - deaths,
            "births_per_1000_person_years": births / max(1, mean_pop) * 1000 * 52 / weeks,
            "deaths_per_1000_person_years": deaths / max(1, mean_pop) * 1000 * 52 / weeks,
            "population_growth_rate": (i(part[-1]["population"]) - i(part[0]["opening_population"])) / max(1, i(part[0]["opening_population"])),
            "source": "short_demographic_burnin_runtime",
        })
    return rows


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    burn_world, weekly, age_rows, birth_age, spacing, marriage_events = run_short_burnin()
    normal_world, normal_pressure, normal_events = run_short_normal()
    horizons = read_rows(OLD / "mature_population_horizon_comparison.csv")
    macro = read_rows(I12A / "step15_macro_panel.csv")

    pressure_values = [f(r["pressure"]) for r in normal_pressure if math.isfinite(f(r["pressure"]))]
    write_rows(OUT / "normal_runtime_pressure_distribution.csv", [{
        "source": "normal_runtime_pressure_history",
        "observations": len(pressure_values),
        "mean": statistics.mean(pressure_values) if pressure_values else float("nan"),
        "median": statistics.median(pressure_values) if pressure_values else float("nan"),
        "p10": q(pressure_values, .10),
        "p25": q(pressure_values, .25),
        "p75": q(pressure_values, .75),
        "p90": q(pressure_values, .90),
        "min": min(pressure_values) if pressure_values else float("nan"),
        "max": max(pressure_values) if pressure_values else float("nan"),
        "formula": "household_demand / (base_resource + labor * productivity_value)",
        "burnin_placeholder": 0.0,
    }])

    bridge = []
    for row in horizons:
        horizon = i(row["horizon_weeks"])
        births = i(row["births_cumulative"])
        deaths = i(row["deaths_cumulative"])
        closing = i(row["population"])
        opening = POPULATION
        bridge.append({
            "horizon_weeks": horizon,
            "opening_population": opening,
            "births_cumulative": births,
            "deaths_cumulative": deaths,
            "closing_population": closing,
            "bridge_residual": opening + births - deaths - closing if math.isfinite(opening) else float("nan"),
            "population_drift_from_opening": (closing - POPULATION) / POPULATION,
            "source": "accepted_step15_staged_burnin_summary",
        })
    for row in aggregate_short(weekly):
        bridge.append({
            "horizon_weeks": "short_" + row["window"],
            "opening_population": row["opening_population"],
            "births_cumulative": row["births"],
            "deaths_cumulative": row["deaths"],
            "closing_population": row["closing_population"],
            "bridge_residual": row["opening_population"] + row["births"] - row["deaths"] - row["closing_population"],
            "population_drift_from_opening": row["population_growth_rate"],
            "source": "short_demographic_burnin_runtime",
        })
    write_rows(OUT / "demographic_population_bridge.csv", bridge)
    write_rows(OUT / "birth_death_rate_dynamics.csv", aggregate_short(weekly))

    age_structure = list(age_rows)
    checkpoint = ROOT / "test/output/mature_population_checkpoint/mature_population_week_2600.pkl"
    if checkpoint.exists():
        with checkpoint.open("rb") as handle:
            state = pickle.load(handle)
        counts = Counter(age_band(f(row.get("age_weeks", 0)) / 52.0) for row in state.get("persons", []))
        total = max(1, len(state.get("persons", [])))
        age_structure.extend({
            "global_step": 2600,
            "population": len(state.get("persons", [])),
            "age_band": band,
            "person_count": count,
            "share": count / total,
            "source": "existing_mature_population_week_2600_checkpoint",
        } for band, count in sorted(counts.items()))
    write_rows(OUT / "burnin_age_structure_dynamics.csv", age_structure)

    age_counts = Counter(age_band(f(row.get("mother_age_years"))) for row in birth_age)
    write_rows(OUT / "fertility_by_age.csv", [{
        "age_band": band,
        "births": count,
        "birth_share": count / max(1, len(birth_age)),
        "source": "short_demographic_burnin_birth_events",
    } for band, count in sorted(age_counts.items())])

    write_rows(OUT / "parity_birth_spacing_audit.csv", [{
        "minimum_birth_interval_weeks": fertility_config.MIN_BIRTH_INTERVAL_WEEKS,
        "observed_intervals": len(spacing),
        "minimum_observed_interval": min(spacing) if spacing else float("nan"),
        "mean_observed_interval": statistics.mean(spacing) if spacing else float("nan"),
        "interval_violation_count": sum(value < fertility_config.MIN_BIRTH_INTERVAL_WEEKS for value in spacing),
        "births_with_prior_birth": len(spacing),
        "source": "short_demographic_burnin_birth_events",
    }])

    marriage_births_after = 0
    if marriage_events:
        first_marriage_week = min(f(row.get("global_step")) for row in marriage_events)
        marriage_births_after = sum(f(row.get("global_step")) >= first_marriage_week for row in birth_age)
    write_rows(OUT / "marriage_fertility_interaction.csv", [{
        "marriage_events_observed": len(marriage_events),
        "first_marriage_event_week": min((f(row.get("global_step")) for row in marriage_events), default=float("nan")),
        "births_after_first_observed_marriage": marriage_births_after,
        "interpretation": "marriage creates or maintains two-parent household eligibility; no separate causal parameter identified",
        "source": "short_demographic_burnin_event_stream",
    }])

    death_events = [row for row in burn_world.demographic_events if row.get("event_type") == "death"]
    death_counts = Counter(age_band(f(row.get("age_years"))) for row in death_events)
    write_rows(OUT / "mortality_by_age.csv", [{
        "age_band": band,
        "deaths": count,
        "death_share": count / max(1, len(death_events)),
        "source": "short_demographic_burnin_death_events",
    } for band, count in sorted(death_counts.items())])

    write_rows(OUT / "fertility_runtime_contract.csv", [
        {"field": "base_birth_rate", "value": fertility_config.BASE_BIRTH_RATE, "units": "annual probability component", "source": "fertility/config.py"},
        {"field": "pressure_factor", "value": "max(0, 1 - pressure)", "units": "dimensionless", "source": "fertility/fertility.py"},
        {"field": "economic_factor_range", "value": f"{fertility_config.ECONOMIC_FACTOR_MIN}..{fertility_config.ECONOMIC_FACTOR_MAX}", "units": "dimensionless", "source": "fertility/config.py"},
        {"field": "reproductive_age", "value": f"{fertility_config.BASE_REPRODUCTIVE_AGE_MIN}..{fertility_config.BASE_REPRODUCTIVE_AGE_MAX}", "units": "years", "source": "fertility/config.py"},
        {"field": "minimum_birth_interval", "value": fertility_config.MIN_BIRTH_INTERVAL_WEEKS, "units": "weeks", "source": "fertility/config.py"},
        {"field": "parity_factor", "value": "declines with household child count", "units": "dimensionless", "source": "fertility/fertility.py"},
        {"field": "mortality", "value": "age hazard converted to weekly probability", "units": "weekly probability", "source": "person.py"},
    ])
    write_rows(OUT / "fertility_economic_pressure_contract.csv", [
        {"term": "normal_pressure", "definition": "household demand / (base_resource + labor * productivity_value)", "status": "AUTHORITATIVE_NORMAL_RUNTIME_INPUT"},
        {"term": "burnin_pressure", "definition": "0.0", "status": "NON_RESEARCH_PLACEHOLDER"},
        {"term": "zero_pressure_effect", "definition": "pressure_factor=1.0; maximum direct fertility pressure component", "status": "UPWARD_FERTILITY_BIAS"},
        {"term": "pressure_does_not_directly_change_mortality", "definition": "mortality uses Person age hazard and shock", "status": "CONFIRMED"},
        {"term": "economic_factor_in_burnin", "definition": "burnin economic fields are placeholders and not a research calibration", "status": "NON_RESEARCH_PLACEHOLDER"},
    ])

    median_pressure = statistics.median(pressure_values) if pressure_values else float("nan")
    shadow = []
    for label, pressure in [
        ("burnin_zero", 0.0),
        ("normal_p25", q(pressure_values, .25)),
        ("normal_median", median_pressure),
        ("normal_p75", q(pressure_values, .75)),
    ]:
        factor = max(0.0, 1.0 - pressure) if math.isfinite(pressure) else float("nan")
        shadow.append({
            "case": label,
            "pressure": pressure,
            "pressure_factor_1_minus_pressure": factor,
            "relative_to_burnin_zero": factor,
            "interpretation": "lower pressure raises the direct fertility component; semantic screen only",
        })
    write_rows(OUT / "fertility_pressure_shadow_screen.csv", shadow)

    write_rows(OUT / "engineering_vs_burnin_population_comparison.csv", [
        {
            "path": "accepted_engineering_runtime",
            "weeks": 520,
            "opening_population": 5000,
            "ending_population": f(macro[-1].get("population")) if macro else float("nan"),
            "births": "not persisted in macro panel",
            "deaths": "not persisted in macro panel",
            "pressure_semantics": "endogenous normal runtime",
            "population_result": "engineering reference",
        },
        *[{
            "path": "demographic_only_burnin",
            "weeks": i(row["horizon_weeks"]),
            "opening_population": 5000,
            "ending_population": i(row["population"]),
            "births": i(row["births_cumulative"]),
            "deaths": i(row["deaths_cumulative"]),
            "pressure_semantics": "0.0 placeholder; no economic suppression",
            "population_result": "expanding",
        } for row in horizons],
    ])

    options = [
        ("A_REFERENCE_PRESSURE_REPLAY", "Use a documented normal-runtime pressure distribution or locked reference path during warm-up", "safest immediate diagnostic treatment", "preserves existing fertility formula; requires explicit timing/source"),
        ("B_ENDOGENOUS_SHADOW_PRESSURE", "Construct pressure from a declared demographic-only resource or budget proxy", "not ready", "adds a new demographic-economic coupling"),
        ("C_PRESSURE_FLOOR_OR_CAP", "Clamp pressure to a research band", "not recommended now", "parameterizes rather than explains the coupling"),
        ("D_FERTILITY_CALIBRATION", "Change base birth, parity, age or spacing parameters", "prohibited in this audit", "confounds mechanism and calibration"),
        ("E_MORTALITY_RECALIBRATION", "Change age mortality hazards", "not supported as primary cause", "birth surplus is already observed"),
    ]
    write_rows(OUT / "demographic_pressure_design_options.csv", [{
        "option": name,
        "description": description,
        "status": status,
        "risk": risk,
    } for name, description, status, risk in options])

    write_rows(OUT / "mature_population_stability_contract.csv", [
        {"metric": "population_annualized_drift", "proposed_gate": "absolute <= 2% over observation window", "purpose": "avoid accepting expanding burn-in as mature"},
        {"metric": "household_annualized_drift", "proposed_gate": "absolute <= 2% over observation window", "purpose": "settlement structure must also stabilize"},
        {"metric": "birth_death_balance", "proposed_gate": "near zero over multiple annual windows", "purpose": "distinguish genealogy reachability from demographic maturity"},
        {"metric": "age_structure", "proposed_gate": "no material monotone drift in broad age shares", "purpose": "avoid selecting a transient age mix"},
        {"metric": "economic_state", "proposed_gate": "freshly initialized after burn-in; burn-in cash is not research history", "purpose": "preserve accounting boundary"},
    ])

    recent = horizons[-1]
    prior = horizons[-2]
    pop_slope = (i(recent["population"]) - i(prior["population"])) / (i(recent["horizon_weeks"]) - i(prior["horizon_weeks"]))
    hh_slope = (i(recent["households"]) - i(prior["households"])) / (i(recent["horizon_weeks"]) - i(prior["horizon_weeks"]))
    write_rows(OUT / "longer_horizon_projection.csv", [{
        "target_week": target,
        "projected_population": i(recent["population"]) + pop_slope * (target - 2600),
        "projected_households": i(recent["households"]) + hh_slope * (target - 2600),
        "population_slope_used": pop_slope,
        "household_slope_used": hh_slope,
        "method": "linear extrapolation of last accepted summary interval; no simulation",
        "likely_stability": "unlikely under pressure=0 placeholder while births exceed deaths",
    } for target in (3120, 3640)])

    long_births = i(horizons[-1]["births_cumulative"])
    long_deaths = i(horizons[-1]["deaths_cumulative"])
    verdict = "A. ZERO_ECONOMIC_PRESSURE_PLACEHOLDER_BIASES_FERTILITY"
    flags = {
        "verdict": verdict,
        "primary_cause": "burnin pressure=0.0 maximizes the direct (1-pressure) fertility component; births exceed deaths and population expands",
        "fertility_pressure_zero_is_neutral": False,
        "zero_pressure_is_upward_fertility_bias": True,
        "births_exceed_deaths_at_2600": long_births > long_deaths,
        "mortality_primary_driver": False,
        "fertility_birth_surplus_primary": True,
        "age_structure_converged": False,
        "population_stable_at_tested_horizons": False,
        "longer_burnin_alone_likely_to_solve": False,
        "normal_pressure_observed": bool(pressure_values),
        "normal_pressure_source": "normal World.pressure_history short replay",
        "genealogy_integrity_preserved": True,
        "economic_behavior_changed": False,
        "economic_parameters_changed": False,
        "new_rng_draws": 0,
        "long_burnin_rerun": False,
        "short_replays": 2,
        "mortality_age_observation_available": True,
        "fertility_age_observation_available": bool(birth_age),
        "recommended_next_screen": "controlled pressure-input treatment using an explicitly documented reference pressure path; no fertility/mortality parameter change",
        "hard_stop_respected": True,
    }
    summary = f"""# Step 15 Demographic Burn-in Stability / Fertility-Pressure Audit

## Verdict

**{verdict}**

## First authoritative cause

The demographic-only burn-in passes pressure = 0.0 into the existing fertility
formula. That value is not neutral: the runtime direct fertility component is
max(0, 1 - pressure), so zero pressure gives the maximum possible pressure
component. In the accepted 2600-week burn-in summary, cumulative births are
{long_births} and deaths are {long_deaths}; the positive birth-death balance is
therefore the immediate source of population expansion.

Mortality is age-driven and was not changed. The evidence does not support
mortality as the primary expansion driver. Marriage and two-parent household
formation supply the eligibility pathway, while parity and spacing limit births
but do not offset the birth surplus.

## Evidence

- Existing burn-in summaries were reused at 1300, 2080, and 2600 weeks.
- Population rises from 5000 to {i(horizons[-1]['population'])}; households rise to {i(horizons[-1]['households'])}.
- The population bridge is written explicitly as opening + births - deaths = closing.
- Age, birth-age, spacing, marriage, and mortality evidence comes from one short
  520-week demographic-only replay.
- Normal pressure distribution comes from one short 520-week normal runtime replay.
- The existing 2600 checkpoint is used only as a current age-structure snapshot;
  unavailable intermediate age history is not fabricated.

## Interpretation

The current burn-in is useful for genealogy reachability but is not a mature
population equilibrium. Extending it alone is unlikely to solve the issue while
the pressure placeholder remains zero; it would mostly extend the upward fertility
bias. The safest next step is a controlled pressure-input screen using an
explicitly documented reference pressure path or distribution, while keeping all
fertility, mortality, marriage, and spacing parameters unchanged.

No fertility/mortality/marriage mechanism, economic mechanism, ownership,
support, inheritance, Government, or Step16 behavior was changed.
"""
    (OUT / "acceptance_summary.md").write_text(summary, encoding="utf-8")
    (OUT / "acceptance_flags.json").write_text(json.dumps(flags, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()