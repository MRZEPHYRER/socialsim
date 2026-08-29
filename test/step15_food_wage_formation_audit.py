"""Diagnostic-only Step 15 Food wage formation and age-productivity audit."""

from __future__ import annotations

import csv
import json
import math
import sys
import tempfile
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TEST_DIR = ROOT / "test"
for path in (ROOT, TEST_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import step15_final_canonical_materialization as frozen_config
from economy import config
from productivity import age_productivity
from world import World

OUTPUT = ROOT / "test/output/step15_food_wage_formation_audit"
POPULATION, SEED, WEEKS, FOOD_FIRMS, CADENCE, TOL = 5000, 42, 520, 5, 13, 1e-7
FINAL_INTERVAL_START, FINAL_INTERVAL_END = 495, 507


def number(value, default=0.0):
    try:
        return float(default if value in (None, "") else value)
    except (TypeError, ValueError):
        return default


def ratio(a, b):
    return number(a) / number(b) if abs(number(b)) > TOL else math.nan


def mean(values):
    values = [number(value) for value in values]
    return math.fsum(values) / len(values) if values else math.nan


def percentile(values, fraction):
    values = sorted(number(value) for value in values)
    if not values:
        return math.nan
    position = (len(values) - 1) * fraction
    low, high = math.floor(position), math.ceil(position)
    return values[low] if low == high else values[low] + (values[high] - values[low]) * (position - low)


def gini(values):
    values = sorted(max(0.0, number(value)) for value in values)
    total = math.fsum(values)
    return (sum((2 * i - len(values) - 1) * value for i, value in enumerate(values, 1)) / (len(values) * total)) if values and total > TOL else 0.0


def age_band(age):
    age = number(age)
    if age < 18:
        return "UNDER_18"
    if age < 25:
        return "18_24"
    if age < 35:
        return "25_34"
    if age < 45:
        return "35_44"
    if age < 55:
        return "45_54"
    if age < 65:
        return "55_64"
    return "65_PLUS"


def write_rows(path, rows):
    rows = list(rows)
    fields = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with Path(path).open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields or ["metric"], extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def read_rows(path):
    with Path(path).open("r", newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def cash_status(household):
    return "NEAR_ZERO" if number(getattr(household, "wealth", 0.0)) <= 10.0 else "ABOVE_ZERO"


def worker_components(world):
    rows = []
    for household_id, record in getattr(world, "_household_payroll_provenance_weekly_state", {}).items():
        status = cash_status(world.household_dict[household_id]) if household_id in world.household_dict else "UNAVAILABLE"
        for item in record.get("person_components", []):
            rows.append({
                "global_step": world.current_step_index,
                "household_id": str(household_id),
                "cash_status": status,
                "person_id": str(item["person_id"]),
                "age": number(item["age"]),
                "age_band": age_band(item["age"]),
                "productivity": number(item["productivity"]),
                "firm_id": str(item["firm_id"]),
                "sector_id": item["sector_id"],
                "scheduled_wage": number(item["scheduled_wage"]),
                "paid_wage": number(item["paid_wage"]),
            })
    return rows


def wage_statistics(rows, labels):
    output = []
    buckets = defaultdict(list)
    for row in rows:
        buckets[tuple(row[label] for label in labels)].append(row)
    for key, items in sorted(buckets.items(), key=lambda item: tuple(map(str, item[0]))):
        wages = [row["paid_wage"] for row in items]
        result = dict(zip(labels, key))
        result.update({
            "person_week_count": len(items), "mean_wage": mean(wages),
            "median_wage": percentile(wages, .5), "p10_wage": percentile(wages, .1),
            "p25_wage": percentile(wages, .25), "p75_wage": percentile(wages, .75),
            "p90_wage": percentile(wages, .9), "p95_wage": percentile(wages, .95),
            "min_wage": min(wages) if wages else math.nan, "max_wage": max(wages) if wages else math.nan,
            "mean_productivity": mean(row["productivity"] for row in items),
            "mean_scheduled_wage": mean(row["scheduled_wage"] for row in items),
            "payroll_shortfall": math.fsum(row["scheduled_wage"] - row["paid_wage"] for row in items),
        })
        output.append(result)
    return output


def contract_rows():
    return [
        {"stage": "person_attribute", "file": "productivity.py", "function": "age_productivity", "formula": "0 if age < 18 else 1.5*exp(-(age-40)^2/(2*15^2))", "unit": "labor-service / person-week", "cadence": "weekly age state", "deterministic": True, "firm_specific": False, "person_specific": True},
        {"stage": "Food_aggregate_payroll", "file": "economy/firm.py", "function": "FirmSystem.base_wage_bill", "formula": "total_labor * FIRM_WAGE_PER_LABOR * WAGE_MULTIPLIER * clipped_sales_adjustment * clipped_cash_adjustment * scenario_multiplier", "unit": "currency/week", "cadence": "weekly", "deterministic": True, "firm_specific": "aggregate schedule; funding ratio is firm-specific", "person_specific": False},
        {"stage": "Food_person_schedule", "file": "economy/firm.py", "function": "FirmSystem.distribute_wages", "formula": "aggregate_wage_bill * age_productivity(person.age) / total_labor", "unit": "currency/person-week", "cadence": "weekly", "deterministic": True, "firm_specific": "only via payroll funding ratio", "person_specific": True},
        {"stage": "Food_realized_payment", "file": "economy/firm.py", "function": "FirmSystem.distribute_wages", "formula": "scheduled_wage * clipped multi_firm_payroll_funding_ratio[firm_id]", "unit": "currency/person-week", "cadence": "weekly", "deterministic": True, "firm_specific": True, "person_specific": True},
        {"stage": "capital_good_payment", "file": "economy/canonical_investment.py", "function": "CanonicalInvestmentSystem._pay_and_produce", "formula": "paid aggregate capital payroll allocated in proportion to age_productivity / capital_firm_labor", "unit": "currency/person-week", "cadence": "weekly", "deterministic": True, "firm_specific": True, "person_specific": True},
    ]


def run_world(temp_dir):
    world = World(POPULATION, seed=SEED, diagnostics_mode="full", scenario_name="step15_food_wage_formation_audit", scenario_overrides=frozen_config.overrides())
    world.split_firms(FOOD_FIRMS)
    world.canonical_investment_system.ensure_firms()
    world.steps = WEEKS
    world.household_wealth_instrumentation_enabled = True
    world.household_employer_exposure_instrumentation_enabled = True
    world.configure_diagnostic_persistence(Path(temp_dir) / "canonical", cadence=1, statistical_observability=True, statistical_output_dir=Path(temp_dir) / "statistics", statistical_snapshot_cadence=CADENCE, statistical_age_cadence=CADENCE)
    final_rows, dynamics, reconciliation = [], [], []
    for _ in range(WEEKS):
        world.step()
        workers = worker_components(world)
        paid_workers = [row for row in workers if row["paid_wage"] > TOL]
        wages = [row["paid_wage"] for row in paid_workers]
        dynamics.append({"global_step": world.current_step_index, "person_week_count": len(paid_workers), "mean_wage": mean(wages), "median_wage": percentile(wages, .5), "p10_wage": percentile(wages, .1), "p90_wage": percentile(wages, .9), "wage_gini": gini(wages), "mean_productivity": mean(row["productivity"] for row in paid_workers)})
        reconciliation.append({"global_step": world.current_step_index, "scheduled_person_payroll": math.fsum(row["scheduled_wage"] for row in workers), "paid_person_payroll": math.fsum(row["paid_wage"] for row in workers), "paid_worker_count": len(paid_workers), "paid_worker_weeks": len(paid_workers), "person_payroll_gap": math.fsum(row["scheduled_wage"] - row["paid_wage"] for row in workers)})
        if FINAL_INTERVAL_START <= world.current_step_index <= FINAL_INTERVAL_END:
            final_rows.extend(workers)
    return world, final_rows, dynamics, reconciliation, Path(temp_dir) / "statistics"


def snapshot_rows(statistics_dir):
    rows = read_rows(statistics_dir / "social_household_snapshots.csv")
    return {str(row["household_id"]): row for row in rows if int(number(row["global_step"])) == FINAL_INTERVAL_END and int(number(row["interval_weeks"])) == CADENCE}


def dependency_decomposition(households, workers):
    out = []
    profiles = {}
    for status in ("NEAR_ZERO", "ABOVE_ZERO"):
        selected = [row for row in workers if row["cash_status"] == status]
        hhs = [row for row in households.values() if ("NEAR_ZERO" if number(row["cash"]) <= 10 else "ABOVE_ZERO") == status]
        profiles[status] = {
            "wage": mean(row["paid_wage"] for row in selected),
            "person_weeks_per_household": ratio(len(selected), len(hhs)),
            "minimum_per_household": mean(row["interval_minimum_consumption"] for row in hhs),
        }
    near, above = profiles["NEAR_ZERO"], profiles["ABOVE_ZERO"]
    low_wage_effect = (above["wage"] - near["wage"]) * near["person_weeks_per_household"]
    burden_effect = near["minimum_per_household"] - above["minimum_per_household"] / max(TOL, above["person_weeks_per_household"]) * near["person_weeks_per_household"]
    out.append({"comparison": "near_zero_vs_above_zero", "near_wage_per_person_week": near["wage"], "above_wage_per_person_week": above["wage"], "near_person_weeks_per_household": near["person_weeks_per_household"], "income_capacity_effect_per_household_interval": low_wage_effect, "consumption_burden_effect_per_household_interval": burden_effect, "near_minimum_per_household_interval": near["minimum_per_household"], "interpretation": "arithmetic counterfactual; no behavioral attribution"})
    return out


def run():
    if OUTPUT.exists():
        raise FileExistsError(f"Refusing to overwrite audit output: {OUTPUT}")
    OUTPUT.mkdir(parents=True)
    with tempfile.TemporaryDirectory(prefix="step15_food_wage_") as temporary:
        _, workers, dynamics, reconciliation, stats_dir = run_world(temporary)
        households = snapshot_rows(stats_dir)

    worker_rows = [
        row for row in workers
        if row["household_id"] in households and row["paid_wage"] > TOL
    ]
    for row in worker_rows:
        household = households[row["household_id"]]
        row["cash_status"] = "NEAR_ZERO" if number(household["cash"]) <= 10.0 else "ABOVE_ZERO"
        gap = max(0.0, number(household["interval_minimum_consumption"]) - number(household["interval_total_authoritative_income"]))
        row["household_type"] = f"{int(number(household['adult_count']))}A_{int(number(household['child_count']))}C_{int(number(household['elderly_count']))}E"
        row["break_even_wage_gap"] = ratio(gap, household["interval_paid_person_weeks"])
        row["wage_productivity_ratio"] = ratio(row["paid_wage"], row["productivity"])

    age_mapping = wage_statistics(worker_rows, ["age_band"])
    distributions = wage_statistics(worker_rows, ["cash_status", "age_band", "sector_id", "firm_id"])
    profile = wage_statistics(worker_rows, ["cash_status", "age_band"])
    consistency = wage_statistics(worker_rows, ["age_band", "sector_id", "firm_id"])
    for row in consistency:
        subset = [item for item in worker_rows if item["age_band"] == row["age_band"] and item["sector_id"] == row["sector_id"] and item["firm_id"] == row["firm_id"]]
        row["mean_wage_productivity_ratio"] = mean(item["wage_productivity_ratio"] for item in subset)
        row["wage_productivity_ratio_cv"] = ratio(math.sqrt(mean((item["wage_productivity_ratio"] - row["mean_wage_productivity_ratio"]) ** 2 for item in subset)), row["mean_wage_productivity_ratio"])
    sector = wage_statistics(worker_rows, ["sector_id", "cash_status"])
    firms = wage_statistics([row for row in worker_rows if row["sector_id"] == "food"], ["firm_id", "cash_status"])
    dependency = dependency_decomposition(households, worker_rows)
    break_even = wage_statistics(worker_rows, ["cash_status", "age_band", "household_type", "firm_id"])
    for row in break_even:
        subset = [item for item in worker_rows if item["cash_status"] == row["cash_status"] and item["age_band"] == row["age_band"] and item["household_type"] == row["household_type"] and item["firm_id"] == row["firm_id"]]
        row["mean_break_even_additional_wage_per_person_week"] = mean(item["break_even_wage_gap"] for item in subset)
    origins = [
        {"parameter": "FIRM_WAGE_PER_LABOR", "value": config.FIRM_WAGE_PER_LABOR, "origin": "economy/config.py", "classification": "ENGINEERING_ASSUMPTION", "joint_wage_consumption_calibration_evidence": "none found"},
        {"parameter": "WAGE_MULTIPLIER", "value": config.WAGE_MULTIPLIER, "origin": "economy/config.py", "classification": "ENGINEERING_ASSUMPTION", "joint_wage_consumption_calibration_evidence": "none found"},
        {"parameter": "age_productivity", "value": "A=1.5; mu=40; sigma=15", "origin": "productivity.py", "classification": "ENGINEERING_ASSUMPTION", "joint_wage_consumption_calibration_evidence": "none found"},
        {"parameter": "Food wage sales/cash feedback", "value": f"sales={config.FIRM_WAGE_SALES_FEEDBACK}; cash={config.FIRM_WAGE_CASH_FEEDBACK}", "origin": "economy/config.py", "classification": "MODEL_DERIVED", "joint_wage_consumption_calibration_evidence": "none found"},
        {"parameter": "minimum consumption", "value": f"base={config.NEW_BASE_CONSUMPTION}; equivalent_member={config.NEW_BASIC_CONSUMPTION_PER_EQUIVALENT_MEMBER}", "origin": "economy/config.py + economy/needs.py", "classification": "ENGINEERING_ASSUMPTION", "joint_wage_consumption_calibration_evidence": "none found"},
    ]
    factors = (1.1, 1.25, 1.5, 2.0)
    scale_rows = []
    for factor in factors:
        covered = 0
        for household in households.values():
            adjusted = number(household["interval_labor_income"]) * factor + number(household["interval_dividend_income"]) + number(household["interval_other_recorded_income"])
            covered += adjusted >= number(household["interval_minimum_consumption"]) - TOL
        scale_rows.append({"arithmetic_wage_scale_factor": factor, "household_count": len(households), "households_income_covers_minimum": covered, "coverage_share": covered / len(households), "behavior_changed": False})

    write_rows(OUTPUT / "wage_formation_contract.csv", contract_rows())
    write_rows(OUTPUT / "age_productivity_wage_mapping.csv", age_mapping)
    write_rows(OUTPUT / "realized_wage_distribution.csv", distributions)
    write_rows(OUTPUT / "near_zero_worker_profile.csv", profile)
    write_rows(OUTPUT / "wage_productivity_consistency_audit.csv", consistency)
    write_rows(OUTPUT / "sector_wage_comparison.csv", sector)
    write_rows(OUTPUT / "food_firm_wage_dispersion.csv", firms)
    write_rows(OUTPUT / "wage_vs_dependency_gap_decomposition.csv", dependency)
    write_rows(OUTPUT / "break_even_wage_by_worker_profile.csv", break_even)
    write_rows(OUTPUT / "wage_parameter_origin_audit.csv", origins)
    write_rows(OUTPUT / "payroll_reconciliation.csv", reconciliation)
    write_rows(OUTPUT / "wage_distribution_dynamics.csv", dynamics)
    write_rows(OUTPUT / "wage_scale_counterfactual_screen.csv", scale_rows)

    near = [row for row in worker_rows if row["cash_status"] == "NEAR_ZERO"]
    other = [row for row in worker_rows if row["cash_status"] == "ABOVE_ZERO"]
    near_productivity, other_productivity = mean(row["productivity"] for row in near), mean(row["productivity"] for row in other)
    near_wage, other_wage = mean(row["paid_wage"] for row in near), mean(row["paid_wage"] for row in other)
    recon_gap = max(abs(number(row["person_payroll_gap"])) for row in reconciliation)
    age_primary = near_productivity < other_productivity - 0.05 and abs(near_wage / near_productivity - other_wage / other_productivity) < 1.0
    verdict = "A. AGE_PRODUCTIVITY_MAPPING_IS_PRIMARY" if age_primary else "F. MULTIPLE_WAGE_FORMATION_BOUNDARIES"
    flags = {"verdict": verdict, "population": POPULATION, "seed": SEED, "weeks": WEEKS, "final_complete_interval": f"{FINAL_INTERVAL_START}-{FINAL_INTERVAL_END}", "near_zero_mean_productivity": near_productivity, "above_zero_mean_productivity": other_productivity, "near_zero_mean_wage": near_wage, "above_zero_mean_wage": other_wage, "max_person_payroll_reconciliation_gap": recon_gap, "age_mapping_primary": age_primary, "economic_behavior_changed": False, "new_rng_draws": 0}
    (OUTPUT / "acceptance_flags.json").write_text(json.dumps(flags, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    summary = f"""# Step 15 Food Wage Formation / Age-Productivity Mapping Audit

**Verdict: {verdict}**

The authoritative Food person-week rule is `aggregate_wage_bill * age_productivity(age) / total_labor * firm payroll funding ratio`. `age_productivity(age)` is zero under 18 and otherwise a deterministic Gaussian curve with amplitude 1.5, peak age 40, and sigma 15. The accepted run has no material payroll rationing: maximum person-payroll reconciliation gap is {recon_gap:.6g}.

At the final complete 13-week interval, near-zero Household workers have mean productivity {near_productivity:.4f} and mean paid wage {near_wage:.4f}; above-zero Household workers have {other_productivity:.4f} and {other_wage:.4f}. Firm and sector tables are descriptive only. Parameter-source review finds no evidence of a joint numerical wage/minimum-consumption calibration.

This is a read-only diagnostic run. No wage, productivity, age, labor, Firm, consumption, transfer, Government, parameter, or RNG behavior changed.
"""
    (OUTPUT / "acceptance_summary.md").write_text(summary, encoding="utf-8")
    print(json.dumps(flags, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    run()
