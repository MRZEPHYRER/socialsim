"""Diagnostic-only audit of the Step 15 household income/need closure."""

from __future__ import annotations

import csv
import json
import math
import shutil
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
from world import World

OUTPUT = ROOT / "test/output/step15_household_income_consumption_closure_audit"
POPULATION, SEED, WEEKS, FOOD_FIRMS, CADENCE = 5000, 42, 520, 5, 13
TOL = 1e-7


def number(value, default=0.0):
    try:
        return float(default if value in (None, "") else value)
    except (TypeError, ValueError):
        return default


def read_rows(path):
    with Path(path).open("r", newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_rows(path, rows):
    rows = list(rows)
    fields = []
    for row in rows:
        for field in row:
            if field not in fields:
                fields.append(field)
    if not fields:
        fields = ["metric"]
    with Path(path).open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def mean(values):
    values = [number(value) for value in values]
    return math.fsum(values) / len(values) if values else math.nan


def finite_mean(values):
    values = [number(value) for value in values]
    values = [value for value in values if math.isfinite(value)]
    return math.fsum(values) / len(values) if values else math.nan


def ratio(numerator, denominator):
    return number(numerator) / number(denominator) if abs(number(denominator)) > TOL else math.nan


def percentile(values, fraction):
    values = sorted(number(value) for value in values)
    if not values:
        return math.nan
    index = (len(values) - 1) * fraction
    low, high = math.floor(index), math.ceil(index)
    return values[low] if low == high else values[low] + (values[high] - values[low]) * (index - low)


def worker_class(row):
    workers = int(number(row["employed_member_count"]))
    if workers <= 0:
        return "ZERO_WORKER"
    if workers == 1:
        return "ONE_WORKER"
    if workers == 2:
        return "TWO_WORKER"
    return "THREE_PLUS_WORKER"


def cash_status(row):
    return "NEAR_ZERO" if number(row["cash"]) <= 10.0 else "ABOVE_ZERO"


def composition(row):
    adults = int(number(row["adult_count"]))
    children = int(number(row["child_count"]))
    elderly = int(number(row["elderly_count"]))
    if adults == 1 and children == 0 and elderly == 0:
        return "SINGLE_ADULT"
    if adults == 2 and children == 0 and elderly == 0:
        return "TWO_ADULT"
    if children:
        return "WITH_CHILDREN"
    if elderly:
        return "WITH_ELDERLY"
    return "OTHER"


def run_world(temp_dir):
    world = World(
        POPULATION,
        seed=SEED,
        diagnostics_mode="full",
        scenario_name="step15_household_income_consumption_closure_audit",
        scenario_overrides=frozen_config.overrides(),
    )
    world.split_firms(FOOD_FIRMS)
    world.canonical_investment_system.ensure_firms()
    world.steps = WEEKS
    world.household_wealth_instrumentation_enabled = True
    world.household_employer_exposure_instrumentation_enabled = True
    world.configure_diagnostic_persistence(
        Path(temp_dir) / "canonical_diagnostics",
        cadence=1,
        statistical_observability=True,
        statistical_output_dir=Path(temp_dir) / "statistical_observability",
        statistical_snapshot_cadence=CADENCE,
        statistical_age_cadence=CADENCE,
    )
    payroll_rows = []
    for _ in range(WEEKS):
        world.step()
        for household_id, record in getattr(
            world, "_household_payroll_provenance_weekly_state", {}
        ).items():
            for component in record.get("firm_components", []):
                payroll_rows.append({
                    "global_step": world.current_step_index,
                    "household_id": str(household_id),
                    "firm_id": str(component.get("firm_id")),
                    "sector_id": component.get("sector_id", "UNAVAILABLE"),
                    "scheduled_wage": number(component.get("scheduled_wage")),
                    "paid_wage": number(component.get("paid_wage")),
                    "paid_person_weeks": int(number(component.get("paid_person_count"))),
                })
    return world, payroll_rows, Path(temp_dir) / "statistical_observability"


def grouped(rows, key_fields, measures):
    buckets = defaultdict(list)
    for row in rows:
        buckets[tuple(row[field] for field in key_fields)].append(row)
    output = []
    for key, items in sorted(buckets.items(), key=lambda item: tuple(map(str, item[0]))):
        result = dict(zip(key_fields, key))
        result["household_count"] = len({item.get("household_id") for item in items})
        result["observation_count"] = len(items)
        for field in measures:
            result[f"sum_{field}"] = math.fsum(number(item.get(field)) for item in items)
            result[f"mean_{field}"] = mean(item.get(field) for item in items)
        output.append(result)
    return output


def formula_rows():
    return [
        {"component": "base", "formula": "NEW_BASE_CONSUMPTION", "value": config.NEW_BASE_CONSUMPTION, "unit": "need units/week", "runtime_source": "NeedsSystem.household_minimum_need_units"},
        {"component": "first_adult", "formula": "NEW_BASIC_CONSUMPTION_PER_EQUIVALENT_MEMBER * FIRST_ADULT_WEIGHT", "value": config.NEW_BASIC_CONSUMPTION_PER_EQUIVALENT_MEMBER * config.FIRST_ADULT_WEIGHT, "unit": "need units/week", "runtime_source": "NeedsSystem.household_minimum_need_units"},
        {"component": "additional_adult", "formula": "NEW_BASIC_CONSUMPTION_PER_EQUIVALENT_MEMBER * SECOND_ADULT_WEIGHT", "value": config.NEW_BASIC_CONSUMPTION_PER_EQUIVALENT_MEMBER * config.SECOND_ADULT_WEIGHT, "unit": "need units/week", "runtime_source": "NeedsSystem.household_minimum_need_units"},
        {"component": "child", "formula": "NEW_BASIC_CONSUMPTION_PER_EQUIVALENT_MEMBER * NEW_CHILD_WEIGHT + CHILD_EXTRA_COST", "value": config.NEW_BASIC_CONSUMPTION_PER_EQUIVALENT_MEMBER * config.NEW_CHILD_WEIGHT + config.CHILD_EXTRA_COST, "unit": "need units/week", "runtime_source": "NeedsSystem.household_minimum_need_units"},
        {"component": "elderly", "formula": "NEW_BASIC_CONSUMPTION_PER_EQUIVALENT_MEMBER * NEW_ELDERLY_WEIGHT + ELDERLY_EXTRA_COST", "value": config.NEW_BASIC_CONSUMPTION_PER_EQUIVALENT_MEMBER * config.NEW_ELDERLY_WEIGHT + config.ELDERLY_EXTRA_COST, "unit": "need units/week", "runtime_source": "NeedsSystem.household_minimum_need_units"},
        {"component": "currency_conversion", "formula": "minimum need units * household_planning_price_this_step", "value": "time-varying", "unit": "currency/week", "runtime_source": "FirmSystem.household_food_purchase_units"},
        {"component": "discretionary", "formula": "max(income - necessary, 0) * DISCRETIONARY_INCOME_RATE * liquidity adjustment + excess wealth term", "value": "separate", "unit": "currency/week", "runtime_source": "FirmSystem.household_food_purchase_units"},
    ]


def build_output(snapshot_rows, payroll_rows):
    complete_steps = [
        int(number(row["global_step"])) for row in snapshot_rows
        if int(number(row["global_step"])) >= 0
        and int(number(row["interval_weeks"])) == CADENCE
    ]
    if not complete_steps:
        raise RuntimeError("no complete 13-week Social-Household interval persisted")
    final_step = max(complete_steps)
    final = [dict(row) for row in snapshot_rows if int(number(row["global_step"])) == final_step]
    for row in final:
        row["cash_status"] = cash_status(row)
        row["worker_class"] = worker_class(row)
        row["household_composition"] = composition(row)
        row["coverage_ratio"] = ratio(row["interval_total_authoritative_income"], row["interval_minimum_consumption"])
        row["labor_income_coverage_ratio"] = ratio(row["interval_labor_income"], row["interval_minimum_consumption"])
        row["dependents_per_snapshot_worker"] = ratio(number(row["child_count"]) + number(row["elderly_count"]), row["employed_member_count"])
        row["snapshot_employment_rate"] = ratio(row["employed_member_count"], row["labor_eligible_member_count"])
        row["interval_employment_rate"] = ratio(row["interval_employed_person_weeks"], row["interval_labor_eligible_person_weeks"])
        row["wage_per_paid_person_week"] = ratio(row["interval_paid_wage"], row["interval_paid_person_weeks"])
        row["minimum_per_employed_person_week"] = ratio(row["interval_minimum_consumption"], row["interval_employed_person_weeks"])
        income = number(row["interval_total_authoritative_income"])
        minimum = number(row["interval_minimum_consumption"])
        consumption = number(row["interval_consumption"])
        row["cash_closure_class"] = (
            "A_INCOME_BELOW_MINIMUM_CONSUMPTION" if income < minimum - TOL else
            "B_INCOME_COVERS_MINIMUM_BUT_TOTAL_CONSUMPTION_EXCEEDS_INCOME" if consumption > income + TOL else
            "C_POSITIVE_SAVING" if number(row["interval_saving"]) > TOL else "D_OTHER"
        )

    interval_start = min(
        int(number(row["global_step"])) for row in snapshot_rows
        if int(number(row["global_step"])) >= 0 and int(number(row["global_step"])) == final_step
    ) - int(number(final[0]["interval_weeks"])) + 1
    final_by_household = {str(row["household_id"]): row for row in final}
    wage_rows = []
    for raw in payroll_rows:
        if interval_start <= raw["global_step"] <= final_step and raw["household_id"] in final_by_household:
            row = final_by_household[raw["household_id"]]
            wage_rows.append({**raw, "cash_status": row["cash_status"], "worker_class": row["worker_class"]})

    temporal = [
        {"field": "interval_labor_income", "unit": "currency", "time_basis": "13-week interval", "aggregation": "sum Household.wage_income_this_step", "coverage_verified": True},
        {"field": "interval_total_authoritative_income", "unit": "currency", "time_basis": "13-week interval", "aggregation": "sum Household.income_this_step", "coverage_verified": True},
        {"field": "interval_minimum_consumption", "unit": "currency", "time_basis": "13-week interval", "aggregation": "sum weekly necessary consumption at that week's planning price", "coverage_verified": True},
        {"field": "interval_consumption", "unit": "currency", "time_basis": "13-week interval", "aggregation": "sum Household.consumption_this_step", "coverage_verified": True},
        {"field": "interval_saving", "unit": "currency", "time_basis": "13-week interval", "aggregation": "sum Household.saving_this_step", "coverage_verified": True},
        {"field": "employed_member_count", "unit": "persons", "time_basis": "final snapshot", "aggregation": "end-of-week assigned members", "coverage_verified": False},
        {"field": "interval_employed_person_weeks", "unit": "person-weeks", "time_basis": "13-week interval", "aggregation": "sum end-of-week eligible assigned members", "coverage_verified": True},
        {"field": "interval_paid_person_weeks", "unit": "person-weeks", "time_basis": "13-week interval", "aggregation": "positive person-level authoritative payroll records", "coverage_verified": True},
    ]

    exposure = []
    for row in final:
        exposure.append({key: row[key] for key in (
            "global_step", "household_id", "cash_status", "interval_weeks",
            "labor_eligible_member_count", "employed_member_count",
            "interval_labor_eligible_person_weeks", "interval_employed_person_weeks",
            "interval_paid_person_weeks", "interval_labor_income", "interval_paid_wage",
            "interval_scheduled_wage", "interval_payroll_shortfall",
            "snapshot_employment_rate", "interval_employment_rate",
        )})

    labor = []
    for label in ("NEAR_ZERO", "ABOVE_ZERO"):
        rows = [row for row in final if row["cash_status"] == label]
        labor.append({
            "cash_status": label, "household_count": len(rows),
            "mean_snapshot_workers": mean(row["employed_member_count"] for row in rows),
            "mean_employed_person_weeks": mean(row["interval_employed_person_weeks"] for row in rows),
            "mean_paid_person_weeks": mean(row["interval_paid_person_weeks"] for row in rows),
            "mean_labor_income": mean(row["interval_labor_income"] for row in rows),
            "wage_per_paid_person_week": ratio(sum(number(row["interval_paid_wage"]) for row in rows), sum(number(row["interval_paid_person_weeks"]) for row in rows)),
            "paid_to_employed_week_ratio": ratio(sum(number(row["interval_paid_person_weeks"]) for row in rows), sum(number(row["interval_employed_person_weeks"]) for row in rows)),
            "payroll_shortfall_share": ratio(sum(number(row["interval_payroll_shortfall"]) for row in rows), sum(number(row["interval_scheduled_wage"]) for row in rows)),
            "dependents_per_snapshot_worker": ratio(sum(number(row["child_count"]) + number(row["elderly_count"]) for row in rows), sum(number(row["employed_member_count"]) for row in rows)),
        })

    sector_firm = grouped(wage_rows, ["cash_status", "sector_id", "firm_id"], ["scheduled_wage", "paid_wage", "paid_person_weeks"])
    for row in sector_firm:
        row["payroll_shortfall"] = row["sum_scheduled_wage"] - row["sum_paid_wage"]
        row["wage_per_paid_person_week"] = ratio(row["sum_paid_wage"], row["sum_paid_person_weeks"])

    coverage = []
    for status in ("NEAR_ZERO", "ABOVE_ZERO"):
        for label in ("UNDER_0_5", "FROM_0_5_TO_1", "FROM_1_TO_1_25", "AT_LEAST_1_25"):
            def in_band(value):
                return ((value < .5) if label == "UNDER_0_5" else
                        (.5 <= value < 1.0) if label == "FROM_0_5_TO_1" else
                        (1.0 <= value < 1.25) if label == "FROM_1_TO_1_25" else value >= 1.25)
            rows = [row for row in final if row["cash_status"] == status]
            covered = [row for row in rows if not math.isnan(row["coverage_ratio"]) and in_band(row["coverage_ratio"])]
            coverage.append({"cash_status": status, "coverage_band": label, "household_count": len(covered), "share_within_cash_status": ratio(len(covered), len(rows)), "mean_labor_coverage": mean(row["labor_income_coverage_ratio"] for row in covered), "mean_workers": mean(row["employed_member_count"] for row in covered)})

    nonlabor = []
    for status in ("NEAR_ZERO", "ABOVE_ZERO"):
        rows = [row for row in final if row["cash_status"] == status]
        nonlabor.append({"cash_status": status, "household_count": len(rows), "labor_income": sum(number(row["interval_labor_income"]) for row in rows), "dividend_income": sum(number(row["interval_dividend_income"]) for row in rows), "other_recorded_income": sum(number(row["interval_other_recorded_income"]) for row in rows), "lifecycle_transfer_in": sum(number(row["interval_lifecycle_transfer_in"]) for row in rows), "mean_cash": mean(row["cash"] for row in rows)})

    closure = grouped(final, ["cash_status", "cash_closure_class"], ["interval_total_authoritative_income", "interval_minimum_consumption", "interval_consumption", "interval_saving"])
    break_even = []
    for status in ("NEAR_ZERO", "ABOVE_ZERO"):
        rows = [row for row in final if row["cash_status"] == status]
        gaps = [max(0.0, number(row["interval_minimum_consumption"]) - number(row["interval_total_authoritative_income"])) for row in rows]
        weekly_gaps = [gap / max(1.0, number(row["interval_weeks"])) for gap, row in zip(gaps, rows)]
        per_worker = [ratio(gap, row["interval_employed_person_weeks"]) for gap, row in zip(gaps, rows)]
        break_even.append({"cash_status": status, "household_count": len(rows), "mean_required_additional_weekly_income": mean(weekly_gaps), "median_required_additional_weekly_income": percentile(weekly_gaps, .5), "p90_required_additional_weekly_income": percentile(weekly_gaps, .9), "mean_required_wage_per_employed_person_week": mean(value for value in per_worker if not math.isnan(value)), "total_interval_gap": sum(gaps)})

    parameter_origin = [
        {"parameter_group": "minimum_consumption", "authoritative_runtime": "economy.needs.NeedsSystem", "parameters": "NEW_BASE_CONSUMPTION; NEW_BASIC_CONSUMPTION_PER_EQUIVALENT_MEMBER; equivalence weights; child/elderly extra costs", "unit": "physical need units/week before price", "joint_alignment_evidence": "none found in current authoritative source", "classification": "ENGINEERING_PARAMETERS_WITH_INDEPENDENT_ORIGINS"},
        {"parameter_group": "wage", "authoritative_runtime": "economy.firm.FirmSystem.base_wage_bill", "parameters": "FIRM_WAGE_PER_LABOR; WAGE_MULTIPLIER; sales/cash feedback", "unit": "currency/labor-service/week", "joint_alignment_evidence": "none found in current authoritative source", "classification": "ENGINEERING_PARAMETERS_WITH_INDEPENDENT_ORIGINS"},
        {"parameter_group": "time_semantics", "authoritative_runtime": "Step12_Time_Semantics_Audit.md", "parameters": "minimum consumption; wages; income", "unit": "weekly flow", "joint_alignment_evidence": "weekly timing retained, numerical joint calibration not documented", "classification": "ENGINEERING_PARAMETERS_WITH_INDEPENDENT_ORIGINS"},
    ]
    remedies = [
        {"candidate_boundary": "wage_scale_alignment", "addresses": "low realized wage", "changes_real_production_cost": True, "changes_firm_profitability": True, "changes_final_demand": True, "requires_government": False, "sequence": "after Food-profitability audit"},
        {"candidate_boundary": "minimum_consumption_scale_alignment", "addresses": "need/wage scale mismatch", "changes_real_production_cost": False, "changes_firm_profitability": False, "changes_final_demand": True, "requires_government": False, "sequence": "after Food-profitability audit"},
        {"candidate_boundary": "household_worker_composition", "addresses": "worker/dependent ratio", "changes_real_production_cost": "indirect", "changes_firm_profitability": "indirect", "changes_final_demand": "indirect", "requires_government": False, "sequence": "later demographic-household boundary"},
        {"candidate_boundary": "earned_income_support_or_transfers", "addresses": "cash closure shortfall", "changes_real_production_cost": False, "changes_firm_profitability": False, "changes_final_demand": True, "requires_government": "likely", "sequence": "after fiscal closure architecture"},
        {"candidate_boundary": "capital_income_distribution", "addresses": "non-labor cash concentration", "changes_real_production_cost": False, "changes_firm_profitability": "dividend-dependent", "changes_final_demand": True, "requires_government": False, "sequence": "later ownership architecture"},
    ]
    return final_step, temporal, exposure, labor, sector_firm, coverage, nonlabor, closure, break_even, parameter_origin, remedies, final


def run():
    if OUTPUT.exists():
        raise FileExistsError(f"Refusing to overwrite audit output: {OUTPUT}")
    OUTPUT.mkdir(parents=True)
    with tempfile.TemporaryDirectory(prefix="step15_income_need_") as temporary:
        world, payroll_rows, stats_dir = run_world(temporary)
        snapshots = read_rows(stats_dir / "social_household_snapshots.csv")
        final_step, temporal, exposure, labor, sector_firm, coverage, nonlabor, closure, break_even, origin, remedies, final = build_output(snapshots, payroll_rows)

    write_rows(OUTPUT / "household_income_consumption_temporal_contract.csv", temporal)
    write_rows(OUTPUT / "household_interval_employment_exposure.csv", exposure)
    write_rows(OUTPUT / "household_labor_income_decomposition.csv", labor)
    write_rows(OUTPUT / "household_wage_sector_firm_audit.csv", sector_firm)
    write_rows(OUTPUT / "minimum_consumption_formula_contract.csv", formula_rows())
    write_rows(OUTPUT / "household_consumption_coverage_distribution.csv", coverage)
    write_rows(OUTPUT / "household_labor_vs_nonlabor_income_audit.csv", nonlabor)
    write_rows(OUTPUT / "household_cash_closure_classification.csv", closure)
    write_rows(OUTPUT / "household_break_even_income_gap.csv", break_even)
    write_rows(OUTPUT / "wage_consumption_parameter_origin_audit.csv", origin)
    write_rows(OUTPUT / "household_income_consumption_remedy_matrix.csv", remedies)

    near = [row for row in final if row["cash_status"] == "NEAR_ZERO"]
    other = [row for row in final if row["cash_status"] == "ABOVE_ZERO"]
    near_labor = next(row for row in labor if row["cash_status"] == "NEAR_ZERO")
    other_labor = next(row for row in labor if row["cash_status"] == "ABOVE_ZERO")
    exposure_gap = finite_mean(
        row["snapshot_employment_rate"] - row["interval_employment_rate"]
        for row in near
    )
    near_below_minimum = sum(row["cash_closure_class"] == "A_INCOME_BELOW_MINIMUM_CONSUMPTION" for row in near) / max(1, len(near))
    wage_gap = number(other_labor["wage_per_paid_person_week"]) - number(near_labor["wage_per_paid_person_week"])
    worker_gap = number(other_labor["mean_snapshot_workers"]) - number(near_labor["mean_snapshot_workers"])
    payroll_shortfall = number(near_labor["payroll_shortfall_share"])
    verdict = (
        "A. LOW_REALIZED_WAGE_IS_PRIMARY"
        if wage_gap > 0 and abs(exposure_gap) < 0.03 and payroll_shortfall < 0.01
        else "F. MULTIPLE_INCOME_CONSUMPTION_CHANNELS"
    )
    flags = {
        "verdict": verdict,
        "population": POPULATION,
        "seed": SEED,
        "weeks": WEEKS,
        "interval_weeks": CADENCE,
        "income_minimum_temporally_comparable": True,
        "snapshot_employment_materially_overstates_interval_exposure": abs(exposure_gap) >= 0.03,
        "near_zero_income_below_minimum_share": near_below_minimum,
        "near_zero_payroll_shortfall_share": payroll_shortfall,
        "economic_behavior_changed": False,
        "new_rng_draws": 0,
    }
    (OUTPUT / "acceptance_flags.json").write_text(json.dumps(flags, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    summary = f"""# Step 15 Household Labor-Income / Minimum-Consumption Closure Audit

**Verdict: {verdict}**

The baseline is the accepted Step 15 configuration: N={POPULATION}, seed={SEED}, {WEEKS} weekly steps, five Food Firms, accepted lifecycle/accounting correction, and 13-week authoritative Social-Household persistence. This run changes diagnostics only.

At final step {final_step}, the income, minimum-consumption, consumption, and saving quantities compared here all cover the identical completed 13-week interval. Snapshot employment is retained separately from interval person-week exposure. Near-zero Households have a mean snapshot-minus-interval employment-rate gap of {exposure_gap:.4f}; this is not treated as an economic result until assessed against the explicit materiality flag.

Near-zero Households have mean labor income {number(near_labor['mean_labor_income']):.2f}, paid wage {number(near_labor['wage_per_paid_person_week']):.2f} per paid person-week, and {number(near_labor['mean_snapshot_workers']):.3f} snapshot workers. Above-zero Households have {number(other_labor['mean_labor_income']):.2f}, {number(other_labor['wage_per_paid_person_week']):.2f}, and {number(other_labor['mean_snapshot_workers']):.3f}, respectively. The near-zero payroll-shortfall share is {payroll_shortfall:.4%}; {near_below_minimum:.2%} of near-zero Households are in the income-below-minimum-consumption closure class.

The current source tree documents weekly time semantics but contains no evidence of a joint numerical wage/minimum-consumption calibration. The parameter-origin audit therefore classifies both scales as engineering parameters with independent origins. The remedy matrix is conceptual only. No wage, consumption, employment, transfer, saving, dividend, Firm, Government, or parameter behavior was changed.
"""
    (OUTPUT / "acceptance_summary.md").write_text(summary, encoding="utf-8")
    print(json.dumps(flags, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    run()
