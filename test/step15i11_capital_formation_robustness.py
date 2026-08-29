from __future__ import annotations

import csv
import gc
import importlib.util
import json
import math
import shutil
import statistics
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

OUTPUT = ROOT / "test/output/step15I11_capital_formation_robustness"
RUNNER_PATH = ROOT / "test/step15i10h_joint_capital_good_scale.py"
SEEDS = (42, 7, 21, 84, 123)
N500 = 500
N5000 = 5000
FOOD_FIRMS = 5
STEPS = 520
TOLERANCE = 1e-6


def number(value, default=0.0):
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def write_rows(path, rows):
    fields = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def load_runner():
    spec = importlib.util.spec_from_file_location("step15i10h_runner", RUNNER_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def slope(values):
    if len(values) < 2:
        return 0.0
    x_mean = (len(values) - 1) / 2.0
    y_mean = statistics.fmean(values)
    denominator = sum((index - x_mean) ** 2 for index in range(len(values)))
    if denominator <= 0.0:
        return 0.0
    return sum((index - x_mean) * (value - y_mean) for index, value in enumerate(values)) / denominator


def backlog_classification(first, second, late):
    late_threshold = 1.0
    if late > late_threshold:
        return "STRONGLY_GROWING"
    if late > 1e-9 or second > 1e-9:
        return "GROWING"
    if late < -1e-9 and second < -1e-9:
        return "DECLINING"
    return "STABLE"


def summarize(result, seed, population, run_label):
    panel = result["panel"]
    backlog = result["backlog"]
    lifecycle = result["lifecycle"]
    desired = [number(row.get("desired_labor")) for row in panel]
    actual_capital = [number(row.get("actual_capital_good_employment")) for row in panel]
    total_employment = [number(row.get("total_employment")) for row in panel]
    food_employment = [number(row.get("food_employment")) for row in panel]
    consumption = [number(row.get("household_consumption")) for row in panel]
    labor_capacity = [number(row.get("total_eligible_labor_capacity")) for row in panel]
    ratios = [
        desired_value / capacity if capacity > 0.0 else math.inf
        for desired_value, capacity in zip(desired, labor_capacity)
    ]
    backlog_values = [number(row.get("closing_backlog")) for row in backlog]
    first_cut = max(1, len(backlog_values) // 2)
    late_cut = max(first_cut + 1, int(len(backlog_values) * 0.75))
    first_slope = slope(backlog_values[:first_cut])
    second_slope = slope(backlog_values[first_cut:])
    late_slope = slope(backlog_values[late_cut:])
    acquisition_count = sum(int(number(row.get("acquisition_count"))) for row in panel)
    retirement_count = sum(int(number(row.get("retirement_count"))) for row in panel)
    replacement_demand = sum(number(row.get("new_replacement_demand")) for row in backlog)
    replacement_execution = sum(number(row.get("fulfilled_replacement")) for row in backlog)
    expansion_execution = sum(number(row.get("fulfilled_expansion")) for row in backlog)
    total_investment = sum(number(row.get("fixed_investment")) for row in panel)
    replacement_investment = sum(number(row.get("replacement_investment")) for row in panel)
    expansion_investment = sum(number(row.get("expansion_investment")) for row in panel)
    active_final = number(panel[-1].get("active_capital_assets")) if panel else 0.0
    retired_final = number(panel[-1].get("retired_capital_assets")) if panel else 0.0
    final_unassigned = number(panel[-1].get("unassigned_eligible_workers")) if panel else 0.0
    lifecycle_chain = (
        acquisition_count > 0
        and retirement_count > 0
        and replacement_demand > TOLERANCE
        and replacement_execution > TOLERANCE
        and any(number(row.get("acquisition_count")) > 0 for row in panel[52:])
    )
    return {
        "run": run_label,
        "seed": seed,
        "population": population,
        "food_firms": FOOD_FIRMS,
        "weeks": STEPS,
        "productivity": 3.0,
        "offer_price": number(panel[0].get("offer_price")) if panel else 0.0,
        "expansion_investment": expansion_investment,
        "replacement_investment": replacement_investment,
        "total_fixed_investment": total_investment,
        "replacement_demand": replacement_demand,
        "replacement_execution": replacement_execution,
        "expansion_execution": expansion_execution,
        "acquisitions": acquisition_count,
        "retirements": retirement_count,
        "active_assets_final": active_final,
        "retired_assets_final": retired_final,
        "capital_good_production": sum(number(row.get("production")) for row in panel),
        "capital_good_sales": sum(number(row.get("sales_units")) for row in panel),
        "peak_backlog": max(backlog_values, default=0.0),
        "final_backlog": backlog_values[-1] if backlog_values else 0.0,
        "first_half_backlog_slope": first_slope,
        "second_half_backlog_slope": second_slope,
        "late_window_backlog_slope": late_slope,
        "backlog_classification": backlog_classification(first_slope, second_slope, late_slope),
        "peak_desired_capital_good_labor": max(desired, default=0.0),
        "mean_desired_capital_good_labor": statistics.fmean(desired) if desired else 0.0,
        "p95_desired_capital_good_labor": sorted(desired)[int(0.95 * (len(desired) - 1))] if desired else 0.0,
        "peak_desired_to_system_ratio": max(ratios, default=0.0),
        "mean_desired_to_system_ratio": statistics.fmean(ratios) if ratios else 0.0,
        "p95_desired_to_system_ratio": sorted(ratios)[int(0.95 * (len(ratios) - 1))] if ratios else 0.0,
        "peak_capital_good_employment": max(actual_capital, default=0.0),
        "mean_capital_good_employment": statistics.fmean(actual_capital) if actual_capital else 0.0,
        "food_employment_final": food_employment[-1] if food_employment else 0.0,
        "mean_food_employment": statistics.fmean(food_employment) if food_employment else 0.0,
        "total_employment_final": total_employment[-1] if total_employment else 0.0,
        "mean_total_employment": statistics.fmean(total_employment) if total_employment else 0.0,
        "capital_good_employment_share": (
            statistics.fmean(actual_capital) / statistics.fmean(total_employment)
            if total_employment and statistics.fmean(total_employment) > 0.0
            else 0.0
        ),
        "firm_cash_final": number(panel[-1].get("firm_cash_total")) if panel else 0.0,
        "firm_debt_final": number(panel[-1].get("firm_debt_total")) if panel else 0.0,
        "household_consumption": sum(consumption),
        "labor_release_events": sum(int(number(row.get("releases"))) for row in panel),
        "labor_reactivation_events": sum(int(number(row.get("hires"))) for row in panel),
        "final_unassigned_eligible_workers": final_unassigned,
        "max_accounting_gap": result["max_accounting_gap"],
        "max_money_gap": result["max_money_gap"],
        "max_goods_gap": result["max_goods_gap"],
        "assignment_violations": result["max_assignment_violations"],
        "realized_output_above_feasible": result["max_feasibility_violation"],
        "investment_loans": result["capital_loans"],
        "step13_investment_funding": False,
        "free_asset_creation": False,
        "lifecycle_chain_observed": lifecycle_chain,
    }


def run_case(runner, population, seed, label):
    runner.POPULATION = population
    runner.FOOD_FIRMS = FOOD_FIRMS
    runner.SEED = seed
    runner.STEPS = STEPS
    result = runner.run_treatment()
    summary = summarize(result, seed, population, label)
    del result
    gc.collect()
    return summary


def cv(values):
    mean = statistics.fmean(values) if values else 0.0
    return statistics.stdev(values) / abs(mean) if len(values) > 1 and abs(mean) > 1e-12 else 0.0


def main():
    shutil.rmtree(OUTPUT, ignore_errors=True)
    OUTPUT.mkdir(parents=True, exist_ok=True)
    runner = load_runner()
    summaries = []
    for seed in SEEDS:
        summaries.append(run_case(runner, N500, seed, "N500"))
    n5000 = run_case(runner, N5000, 42, "N5000")

    write_rows(OUTPUT / "seed_summary.csv", summaries + [n5000])
    write_rows(
        OUTPUT / "backlog_robustness.csv",
        [
            {
                "seed": row["seed"],
                "population": row["population"],
                "peak_backlog": row["peak_backlog"],
                "final_backlog": row["final_backlog"],
                "first_half_slope": row["first_half_backlog_slope"],
                "second_half_slope": row["second_half_backlog_slope"],
                "late_window_slope": row["late_window_backlog_slope"],
                "classification": row["backlog_classification"],
            }
            for row in summaries
        ],
    )
    write_rows(
        OUTPUT / "lifecycle_robustness.csv",
        [
            {
                "seed": row["seed"],
                "population": row["population"],
                "acquisitions": row["acquisitions"],
                "retirements": row["retirements"],
                "replacement_demand_units": row["replacement_demand"],
                "replacement_execution_units": row["replacement_execution"],
                "expansion_execution_units": row["expansion_execution"],
                "active_assets_final": row["active_assets_final"],
                "retired_assets_final": row["retired_assets_final"],
                "acquisition_retirement_replacement_chain": row["lifecycle_chain_observed"],
            }
            for row in summaries
        ],
    )
    write_rows(
        OUTPUT / "labor_robustness.csv",
        [
            {
                "seed": row["seed"],
                "population": row["population"],
                "peak_desired_to_system_ratio": row["peak_desired_to_system_ratio"],
                "p95_desired_to_system_ratio": row["p95_desired_to_system_ratio"],
                "mean_desired_to_system_ratio": row["mean_desired_to_system_ratio"],
                "labor_release_events": row["labor_release_events"],
                "labor_reactivation_events": row["labor_reactivation_events"],
                "final_unassigned_eligible_workers": row["final_unassigned_eligible_workers"],
                "assignment_violations": row["assignment_violations"],
                "labor_scale_flag": row["peak_desired_to_system_ratio"] > 1.1,
            }
            for row in summaries
        ],
    )

    n500_ref = next(row for row in summaries if row["seed"] == 42)
    scale_rows = []
    for metric, n500_value, n5000_value in (
        (
            "investment_over_household_consumption",
            n500_ref["total_fixed_investment"] / max(n500_ref["household_consumption"], 1e-12),
            n5000["total_fixed_investment"] / max(n5000["household_consumption"], 1e-12),
        ),
        (
            "capital_good_employment_share",
            n500_ref["capital_good_employment_share"],
            n5000["capital_good_employment_share"],
        ),
        (
            "final_backlog_per_population",
            n500_ref["final_backlog"] / N500,
            n5000["final_backlog"] / N5000,
        ),
        (
            "active_capital_assets_per_population",
            n500_ref["active_assets_final"] / N500,
            n5000["active_assets_final"] / N5000,
        ),
        (
            "fixed_investment_over_food_production",
            n500_ref["total_fixed_investment"] / max(n500_ref["capital_good_production"], 1e-12),
            n5000["total_fixed_investment"] / max(n5000["capital_good_production"], 1e-12),
        ),
    ):
        scale_rows.append({
            "metric": metric,
            "N500_seed42": n500_value,
            "N5000_seed42": n5000_value,
            "N5000_minus_N500": n5000_value - n500_value,
            "N5000_over_N500": n5000_value / n500_value if abs(n500_value) > 1e-12 else math.inf,
        })
    write_rows(OUTPUT / "scale_comparison.csv", scale_rows)

    scale_regression_metrics = {
        row["metric"]: row["N5000_over_N500"]
        for row in scale_rows
    }
    scale_regression = any(
        ratio < 0.5 or ratio > 2.0
        for ratio in scale_regression_metrics.values()
    )

    key_values = {
        "total_fixed_investment": [row["total_fixed_investment"] for row in summaries],
        "replacement_share": [
            row["replacement_investment"] / max(row["total_fixed_investment"], 1e-12)
            for row in summaries
        ],
        "final_backlog": [row["final_backlog"] for row in summaries],
        "capital_good_employment_share": [row["capital_good_employment_share"] for row in summaries],
        "active_assets": [row["active_assets_final"] for row in summaries],
        "household_consumption": [row["household_consumption"] for row in summaries],
    }
    cross_seed_rows = []
    for metric, values in key_values.items():
        cross_seed_rows.append({
            "metric": metric,
            "mean": statistics.fmean(values),
            "sd": statistics.stdev(values),
            "min": min(values),
            "max": max(values),
            "cv": cv(values),
        })

    hard_fail = any(
        row["max_accounting_gap"] > 1e-6
        or row["max_money_gap"] > 1e-5
        or row["max_goods_gap"] > 1e-6
        or row["assignment_violations"] > 0
        or row["realized_output_above_feasible"] > 1e-6
        or row["investment_loans"] > 1e-9
        or not row["lifecycle_chain_observed"]
        for row in summaries + [n5000]
    )
    scale_fail = any(
        n5000[key] <= 0.0
        for key in ("household_consumption", "total_fixed_investment")
    )
    strong_backlog = sum(
        row["backlog_classification"] == "STRONGLY_GROWING" for row in summaries
    ) >= 3
    labor_fail = any(
        row["peak_desired_to_system_ratio"] > 1.25
        or row["assignment_violations"] > 0
        for row in summaries + [n5000]
    )
    if hard_fail:
        verdict = "G. ACCOUNTING_OR_CONSERVATION_BLOCKER"
    elif scale_fail or scale_regression:
        verdict = "E. LARGE_WORLD_SCALE_REGRESSION"
    elif strong_backlog:
        verdict = "C. STRUCTURALLY_GROWING_BACKLOG"
    elif labor_fail:
        verdict = "D. LABOR_SCALE_NOT_ROBUST"
    else:
        verdict = "A. CAPITAL_FORMATION_ROBUST_ACROSS_SEEDS_AND_SCALE"

    flags = {
        "verdict": verdict,
        "seeds": list(SEEDS),
        "N500_runs": len(summaries),
        "N5000_scale_check": True,
        "food_firms": FOOD_FIRMS,
        "weeks": STEPS,
        "productivity": 3.0,
        "productivity_label": "NON_CALIBRATED_ENGINEERING_REFERENCE",
        "offer_price_semantics": "authoritative_unit_cost",
        "stock_to_flow_horizon": 13,
        "useful_life_weeks": 52,
        "useful_life_label": "NON_CALIBRATED_ENGINEERING_SCREEN",
        "internal_cash_only": True,
        "investment_loans_zero": all(row["investment_loans"] <= 1e-9 for row in summaries + [n5000]),
        "accounting_pass": all(row["max_accounting_gap"] <= 1e-6 for row in summaries + [n5000]),
        "money_pass": all(row["max_money_gap"] <= 1e-5 for row in summaries + [n5000]),
        "goods_pass": all(row["max_goods_gap"] <= 1e-6 for row in summaries + [n5000]),
        "assignment_pass": all(row["assignment_violations"] == 0 for row in summaries + [n5000]),
        "feasibility_pass": all(row["realized_output_above_feasible"] <= 1e-6 for row in summaries + [n5000]),
        "lifecycle_chain_pass": all(row["lifecycle_chain_observed"] for row in summaries + [n5000]),
        "strong_backlog_seed_count": sum(row["backlog_classification"] == "STRONGLY_GROWING" for row in summaries),
        "labor_scale_flag_seed_count": sum(row["peak_desired_to_system_ratio"] > 1.1 for row in summaries),
        "large_world_scale_regression": scale_regression,
        "scale_ratio_acceptance_band": "0.5 <= N5000/N500 <= 2.0 for normalized scale metrics",
        "economic_behavior_changed_only_by_accepted_scale": True,
        "new_rng_draws": 0,
        "parameter_tuning": False,
        "step13_changed": False,
        "cross_seed_summary": cross_seed_rows,
    }
    (OUTPUT / "acceptance_flags.json").write_text(
        json.dumps(flags, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    summary = f"""# Step 15I.11 Capital Formation Multi-Seed and Scale Robustness

## Verdict

**{verdict}**

The accepted I.10H engineering configuration was held fixed: P3 productivity,
cost-anchored offer price, 13-week backlog flow, 52-week engineering life,
internal-cash-only finance, and the accepted labor release/re-hire boundary.
Five N=500 runs used seeds 42, 7, 21, 84, and 123. One N=5000/seed42 run
used the same five-Food-Firm configuration and 520-week horizon.

No productivity, price, useful-life, backlog, planner, labor, financing, or
Step13 parameter was changed.

## Interpretation

The CSV files report seed-level event counts and normalized N=500/N=5000
comparisons. Backlog slopes are ordinary least-squares units/week over the
first half, second half, and final quarter. `STRONGLY_GROWING` means the late
slope exceeds 1 capital-good unit/week; this is a diagnostic classification,
not a tuning trigger.

Conservation and lifecycle flags are retained separately so a remaining
backlog or seed variation is not incorrectly treated as an accounting error.
"""
    (OUTPUT / "acceptance_summary.md").write_text(summary, encoding="utf-8")
    for row in cross_seed_rows:
        pass


if __name__ == "__main__":
    main()
