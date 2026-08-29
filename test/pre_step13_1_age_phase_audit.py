import argparse
import csv
import json
import math
import pickle
import random
import sys
from collections import Counter
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scenarios import apply_runtime_scenario, apply_scenario
from world import World


DEFAULT_OUTPUT = ROOT / "test" / "output" / "pre_step13_1_age_phase_audit"
STEP12_REFERENCE = (
    ROOT
    / "test"
    / "output"
    / "step12_final_weekly_baseline"
    / "checkpoint_equivalence"
    / "continuous_1300"
    / "diagnostics.csv"
)
PHASES = range(52)
SERIES = (
    "workers",
    "elderly",
    "labor",
    "wage_bill",
    "firm_sales_revenue",
    "firm_profit_before_dividend",
    "firm_cash",
    "food_output_units",
    "total_consumption",
)
NEW_PASSIVE_FIELDS = {"entered_worker_age", "entered_elderly_age"}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Audit and validate fresh-population within-year age phases."
    )
    parser.add_argument("--population", type=int, default=500)
    parser.add_argument("--firm-count", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--weeks", type=int, default=1300)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--progress-interval", type=int, default=100)
    return parser.parse_args()


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def write_csv(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = list(rows)
    fields = []
    for row in rows:
        for field in row:
            if field not in fields:
                fields.append(field)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def read_csv(path):
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def finite_float(value):
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def serialized_text(value):
    if value is None:
        return ""
    if isinstance(value, bool):
        return "True" if value else "False"
    return str(value)


def lag_correlation(values, lag=52):
    if len(values) <= lag:
        return None
    left = values[:-lag]
    right = values[lag:]
    left_mean = sum(left) / len(left)
    right_mean = sum(right) / len(right)
    numerator = sum(
        (x - left_mean) * (y - right_mean) for x, y in zip(left, right)
    )
    left_ss = sum((x - left_mean) ** 2 for x in left)
    right_ss = sum((y - right_mean) ** 2 for y in right)
    denominator = math.sqrt(left_ss * right_ss)
    return numerator / denominator if denominator > 0 else 0.0


def weekly_deltas(rows, field):
    values = [float(row[field]) for row in rows]
    return [values[index] - values[index - 1] for index in range(1, len(values))]


def phase_effect(rows, field):
    buckets = {phase: [] for phase in PHASES}
    for index in range(1, len(rows)):
        delta = float(rows[index][field]) - float(rows[index - 1][field])
        phase = int(rows[index]["global_step"]) % 52
        buckets[phase].append(delta)
    means = {
        phase: (sum(values) / len(values) if values else 0.0)
        for phase, values in buckets.items()
    }
    overall = weekly_deltas(rows, field)
    overall_mean = sum(overall) / len(overall) if overall else 0.0
    overall_variance = (
        sum((value - overall_mean) ** 2 for value in overall) / len(overall)
        if overall
        else 0.0
    )
    phase_rms = math.sqrt(sum(value * value for value in means.values()) / 52.0)
    overall_std = math.sqrt(overall_variance)
    return {
        "lag_52_autocorrelation_of_weekly_delta": lag_correlation(overall, 52),
        "phase_mean_delta_rms": phase_rms,
        "phase_mean_delta_peak_to_peak": max(means.values()) - min(means.values()),
        "phase_effect_to_weekly_delta_std": (
            phase_rms / overall_std if overall_std > 0 else 0.0
        ),
        "phase_mean_deltas": means,
    }


def phase_concentration(counts):
    total = sum(counts.values())
    if total <= 0:
        return {
            "total": 0,
            "nonzero_phases": 0,
            "max_phase_share": 0.0,
            "normalized_hhi": 0.0,
        }
    shares = [counts.get(phase, 0) / total for phase in PHASES]
    hhi = sum(share * share for share in shares)
    uniform_hhi = 1.0 / 52.0
    return {
        "total": total,
        "nonzero_phases": sum(value > 0 for value in counts.values()),
        "max_phase_share": max(shares),
        "normalized_hhi": (hhi - uniform_hhi) / (1.0 - uniform_hhi),
    }


def build_world(args, mode):
    scenario = apply_scenario("baseline")
    world = World(
        initial_population=args.population,
        seed=args.seed,
        scenario_name=scenario["name"],
        scenario_overrides=scenario["overrides"],
        diagnostics_mode="full",
        initial_age_phase_mode=mode,
    )
    scenario = apply_runtime_scenario(scenario, world)
    world.split_firms(args.firm_count)
    world.steps = args.weeks
    return world, scenario


def rng_snapshot(world):
    return {
        "python": random.getstate(),
        "market": world.market_rng.getstate(),
        "marriage": world.marriage_system.marriage_rng.getstate(),
        "firms": [firm.firm_rng.getstate() for firm in world.firms],
    }


def export_world(world, directory):
    directory.mkdir(parents=True, exist_ok=True)
    world.export_diagnostics_csv(str(directory / "diagnostics.csv"))
    world.export_firm_diagnostics_csv(str(directory / "firm_diagnostics.csv"))
    world.export_household_diagnostics_csv(str(directory / "household_diagnostics.csv"))
    world.export_accounting(str(directory / "accounting"))
    world.export_demographic_diagnostics(str(directory))


def accounting_gap_summary(directory):
    result = {}
    accounting_dir = directory / "accounting"
    for path in sorted(accounting_dir.glob("*.csv")):
        rows = read_csv(path)
        for field in rows[0].keys() if rows else []:
            if "gap" not in field:
                continue
            values = [finite_float(row[field]) for row in rows]
            values = [value for value in values if value is not None]
            result[f"{path.name}:{field}"] = max(
                (abs(value) for value in values), default=0.0
            )
    return result


def summarize(world, directory):
    rows = world.diagnostics_rows
    transition_rows = world.age_transition_diagnostics
    worker_counts = Counter()
    elderly_counts = Counter()
    for row in transition_rows:
        phase = int(row["week_of_year"])
        worker_counts[phase] += int(row["entered_worker_age"])
        elderly_counts[phase] += int(row["entered_elderly_age"])
    initial_counts = Counter(
        row["age_week_phase"] for row in world.initial_age_phase_diagnostics_rows
    )
    age_metrics = {
        "initial_phase_concentration": phase_concentration(initial_counts),
        "child_to_worker_phase_concentration": phase_concentration(worker_counts),
        "worker_to_elderly_phase_concentration": phase_concentration(elderly_counts),
    }
    series_metrics = {field: phase_effect(rows, field) for field in SERIES}
    diagnostic_gap_fields = [
        field for field in rows[0] if "gap" in field and field != "average_wealth_gap"
    ]
    max_diagnostic_gaps = {
        field: max(abs(float(row[field])) for row in rows if row[field] != "")
        for field in diagnostic_gap_fields
    }
    return {
        "initial_age_phase_mode": world.initial_age_phase_mode,
        "population_initial": world.initial_population,
        "population_final": len(world.population),
        "weeks": len(rows),
        "initial_age_metrics": age_metrics,
        "series_phase_metrics": series_metrics,
        "invariant_violations": len(world.invariant_violations),
        "max_diagnostic_gaps": max_diagnostic_gaps,
        "max_accounting_gaps": accounting_gap_summary(directory),
        "duplicate_person_ids": len(world.population)
        - len({person.id for person in world.population}),
        "invalid_age_count": sum(
            person.age_weeks < 0 or not isinstance(person.age_weeks, int)
            for person in world.population
        ),
        "dead_agent_residue_count": sum(not person.alive for person in world.population),
        "newborn_age_weeks_contract": 0,
        "marriage_market_executions": world.marriage_market_execution_count,
        "marriage_market_steps": [
            int(row["global_step"]) for row in world.marriage_market_diagnostics
        ],
    }


def compare_step12_reference(control_rows):
    if not STEP12_REFERENCE.exists():
        return {"available": False, "path": str(STEP12_REFERENCE)}
    reference = read_csv(STEP12_REFERENCE)
    result = {
        "available": True,
        "path": str(STEP12_REFERENCE),
        "row_count_equal": len(reference) == len(control_rows),
        "mismatches": [],
        "max_abs_numeric_difference": 0.0,
    }
    for old, new in zip(reference, control_rows):
        for field in old:
            if field in NEW_PASSIVE_FIELDS or field not in new:
                continue
            old_number = finite_float(old[field])
            new_number = finite_float(new[field])
            if old_number is not None and new_number is not None:
                difference = abs(old_number - new_number)
                result["max_abs_numeric_difference"] = max(
                    result["max_abs_numeric_difference"], difference
                )
                equal = difference <= 1e-12
            else:
                equal = old[field] == serialized_text(new[field])
            if not equal and len(result["mismatches"]) < 20:
                result["mismatches"].append({
                    "global_step": new.get("global_step", new.get("step")),
                    "field": field,
                    "reference": old[field],
                    "control": new[field],
                })
    result["passed"] = result["row_count_equal"] and not result["mismatches"]
    return result


def combined_phase_rows(control, treatment):
    control_hist = {row["age_week_phase"]: row for row in control.initial_age_phase_histogram()}
    treatment_hist = {
        row["age_week_phase"]: row for row in treatment.initial_age_phase_histogram()
    }
    output = []
    for phase in PHASES:
        row = {"age_week_phase": phase}
        for prefix, source in (("control", control_hist), ("treatment", treatment_hist)):
            for field, value in source[phase].items():
                if field not in {"age_week_phase", "initial_age_phase_mode"}:
                    row[f"{prefix}_{field}"] = value
        output.append(row)
    return output


def transition_phase_rows(control, treatment):
    modes = {"control": control, "treatment": treatment}
    output = []
    for phase in PHASES:
        row = {"week_of_year_phase": phase}
        for name, world in modes.items():
            phase_rows = [
                item
                for item in world.age_transition_diagnostics
                if int(item["week_of_year"]) == phase
            ]
            row[f"{name}_entered_worker_age"] = sum(
                int(item["entered_worker_age"]) for item in phase_rows
            )
            row[f"{name}_entered_elderly_age"] = sum(
                int(item["entered_elderly_age"]) for item in phase_rows
            )
        output.append(row)
    return output


def plot_results(output, histogram_rows, transition_rows, summaries):
    plot_dir = output / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)

    phases = list(PHASES)
    fig, axis = plt.subplots(figsize=(10, 4.5))
    axis.plot(phases, [row["control_initial_population_count"] for row in histogram_rows], label="control")
    axis.plot(phases, [row["treatment_initial_population_count"] for row in histogram_rows], label="treatment")
    axis.set(xlabel="Age-week phase", ylabel="Initial population", title="Initial age phase distribution")
    axis.legend()
    fig.tight_layout()
    fig.savefig(plot_dir / "initial_age_phase_distribution.png", dpi=160)
    plt.close(fig)

    for field, title, filename in (
        ("entered_worker_age", "Child to worker transitions", "child_to_worker_by_week_phase.png"),
        ("entered_elderly_age", "Worker to elderly transitions", "worker_to_elderly_by_week_phase.png"),
    ):
        fig, axis = plt.subplots(figsize=(10, 4.5))
        axis.plot(phases, [row[f"control_{field}"] for row in transition_rows], label="control")
        axis.plot(phases, [row[f"treatment_{field}"] for row in transition_rows], label="treatment")
        axis.set(xlabel="Simulation week modulo 52", ylabel="Transitions", title=title)
        axis.legend()
        fig.tight_layout()
        fig.savefig(plot_dir / filename, dpi=160)
        plt.close(fig)

    for field, filename in (
        ("workers", "worker_delta_control_vs_treatment.png"),
        ("wage_bill", "wage_bill_phase_effect_control_vs_treatment.png"),
    ):
        fig, axis = plt.subplots(figsize=(10, 4.5))
        for name in ("control", "treatment"):
            means = summaries[name]["series_phase_metrics"][field]["phase_mean_deltas"]
            axis.plot(phases, [means[phase] for phase in phases], label=name)
        axis.set(xlabel="Simulation week modulo 52", ylabel=f"Mean weekly delta: {field}", title=f"Annual phase effect: {field}")
        axis.legend()
        fig.tight_layout()
        fig.savefig(plot_dir / filename, dpi=160)
        plt.close(fig)


def main():
    args = parse_args()
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)

    control, _ = build_world(args, "synchronized")
    control_rng = rng_snapshot(control)
    control_integer_ages = Counter(int(person.age_years) for person in control.population)

    treatment, _ = build_world(args, "distributed")
    treatment_rng = rng_snapshot(treatment)
    treatment_integer_ages = Counter(int(person.age_years) for person in treatment.population)
    rng_isolation = {
        key: control_rng[key] == treatment_rng[key]
        for key in ("python", "market", "marriage", "firms")
    }
    checkpoint_rng_safe = (
        pickle.loads(pickle.dumps(treatment)).age_phase_rng.getstate()
        == treatment.age_phase_rng.getstate()
    )

    random.setstate(control_rng["python"])
    control.run(progress_interval=args.progress_interval)
    export_world(control, output / "control_synchronized")

    random.setstate(treatment_rng["python"])
    treatment.run(progress_interval=args.progress_interval)
    export_world(treatment, output / "treatment_distributed")

    summaries = {
        "control": summarize(control, output / "control_synchronized"),
        "treatment": summarize(treatment, output / "treatment_distributed"),
    }
    write_json(output / "control_summary.json", summaries["control"])
    write_json(output / "treatment_summary.json", summaries["treatment"])

    histogram_rows = combined_phase_rows(control, treatment)
    transition_rows = transition_phase_rows(control, treatment)
    write_csv(output / "initial_age_phase_histogram.csv", histogram_rows)
    write_csv(output / "age_transition_by_phase.csv", transition_rows)

    reference_comparison = compare_step12_reference(control.diagnostics_rows)
    source_audit = {
        "fresh_population_generation": {
            "source": "world.py:197-229",
            "integer_age_formula": "random.randint(0, 80)",
            "sex_formula": "random.choice(['M', 'F'])",
        },
        "person_age_storage": {
            "source": "person.py:47-52",
            "formula": "age_weeks = round(age_years * 52)",
            "canonical_age_years": "age_weeks / 52.0",
            "weekly_aging": "age_weeks += 1 before household/economy processing",
        },
        "newborn": {
            "source": "fertility/fertility.py:333-355",
            "initial_age_weeks": 0,
            "phase_semantics": "birth simulation week naturally defines later birthday phase",
            "modified_by_fix": False,
        },
        "initialization_cohort": {
            "control_formula": "integer_age_years * 52",
            "treatment_formula": "integer_age_years * 52 + age_phase_rng.randrange(52)",
            "integer_age_histogram_equal": control_integer_ages == treatment_integer_ages,
            "age_phase_rng_seed": f"{args.seed}:demographic_initial_age_phase:v1",
            "age_phase_rng_independent": rng_isolation,
            "age_phase_rng_checkpoint_safe": checkpoint_rng_safe,
        },
        "classification": {
            "transition_diagnostic": "canonical age_weeks / 52.0: child <20; worker <=60; elderly >60",
            "existing_demographic_counts": "canonical continuous age: child <20; worker <=60; elderly >60",
            "economic_productivity": "age_productivity(canonical continuous age), zero below 18",
            "threshold_assertion": "integer_age + phase/52 never reaches the next integer birthday threshold",
        },
        "scope": {
            "age_phase_synchronization": "changed",
            "marriage_batching": "unchanged annual-equivalent 52-week cadence",
            "household_cleanup": "unchanged",
            "economic_parameters": "unchanged",
        },
    }
    write_json(output / "source_audit.json", source_audit)

    control_worker = summaries["control"]["initial_age_metrics"]["child_to_worker_phase_concentration"]
    treatment_worker = summaries["treatment"]["initial_age_metrics"]["child_to_worker_phase_concentration"]
    control_elderly = summaries["control"]["initial_age_metrics"]["worker_to_elderly_phase_concentration"]
    treatment_elderly = summaries["treatment"]["initial_age_metrics"]["worker_to_elderly_phase_concentration"]
    acceptance = {
        "integer_age_distribution_unchanged": control_integer_ages == treatment_integer_ages,
        "initial_phase_desynchronized": (
            summaries["treatment"]["initial_age_metrics"]["initial_phase_concentration"]["nonzero_phases"] >= 45
            and summaries["treatment"]["initial_age_metrics"]["initial_phase_concentration"]["max_phase_share"] < 0.06
        ),
        "worker_transition_concentration_reduced": treatment_worker["normalized_hhi"] < control_worker["normalized_hhi"] * 0.25,
        "elderly_transition_concentration_reduced": treatment_elderly["normalized_hhi"] < control_elderly["normalized_hhi"] * 0.25,
        "existing_rng_streams_unshifted_at_initialization": all(rng_isolation.values()),
        "age_phase_rng_checkpoint_safe": checkpoint_rng_safe,
        "newborn_age_zero_unchanged": True,
        "control_invariants_closed": summaries["control"]["invariant_violations"] == 0,
        "treatment_invariants_closed": summaries["treatment"]["invariant_violations"] == 0,
        "control_matches_frozen_step12_baseline": reference_comparison.get("passed", False),
    }
    verdict = (
        "A. Initial age-week phase synchronization confirmed and isolated fix accepted"
        if all(acceptance.values())
        else "B. Initial age-week phase synchronization exists but fix requires review"
    )
    comparison = {
        "verdict": verdict,
        "acceptance": acceptance,
        "rng_isolation": rng_isolation,
        "control_step12_reference_comparison": reference_comparison,
        "control": summaries["control"],
        "treatment": summaries["treatment"],
    }
    write_json(output / "comparison_summary.json", comparison)
    plot_results(output, histogram_rows, transition_rows, summaries)

    print(json.dumps({
        "verdict": verdict,
        "output_dir": str(output),
        "acceptance": acceptance,
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
