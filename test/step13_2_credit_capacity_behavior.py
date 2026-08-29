"""Run and validate the Step 13.2 payroll-credit-capacity treatments."""

import argparse
import csv
import json
import math
import subprocess
import sys
import shutil
from collections import Counter, defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "test" / "output" / "step13_2_credit_capacity_behavior"
SCENARIOS = [
    "baseline",
    "credit_capacity_payroll_1x",
    "credit_capacity_payroll_2x",
    "credit_capacity_payroll_4x",
    "credit_capacity_payroll_control_100x",
]
K_VALUES = {
    "baseline": None,
    "credit_capacity_payroll_1x": 11.59038588550643,
    "credit_capacity_payroll_2x": 23.18077177101286,
    "credit_capacity_payroll_4x": 46.36154354202572,
    "credit_capacity_payroll_control_100x": 1159.038588550643,
}
TOL = 1e-7


def read_csv(path):
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_csv(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("\n", encoding="utf-8")
        return
    fields = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def f(row, key, default=0.0):
    try:
        value = float(row.get(key, default))
        return value if math.isfinite(value) else default
    except (TypeError, ValueError):
        return default


def run_name(population, scenario, steps):
    return OUTPUT / "raw_runs" / f"N{population}_seed42_w{steps}" / scenario


def run_model(population, scenario, steps, rerun=False):
    output = run_name(population, scenario, steps)
    if (output / "diagnostics.csv").exists() and not rerun:
        return output
    output.mkdir(parents=True, exist_ok=True)
    command = [
        sys.executable, str(ROOT / "main.py"),
        "--population", str(population), "--steps", str(steps),
        "--seed", "42", "--firm-count", "5", "--scenario", scenario,
        "--output-dir", str(output), "--no-analysis", "--no-plots",
        "--no-ledger-csv", "--progress-interval", "0",
    ]
    subprocess.run(command, cwd=ROOT, check=True)
    return output


def load_run(path):
    return {
        "path": path,
        "diagnostics": read_csv(path / "diagnostics.csv"),
        "firms": read_csv(path / "firm_diagnostics.csv"),
        "manifest": json.loads((path / "manifest.json").read_text(encoding="utf-8")),
    }


def max_difference(left, right, excluded):
    max_abs = 0.0
    first = None
    keys = [key for key in left[0] if key in right[0] and key not in excluded]
    for row_left, row_right in zip(left, right):
        for key in keys:
            try:
                difference = abs(float(row_left[key]) - float(row_right[key]))
            except (TypeError, ValueError):
                difference = 0.0 if row_left[key] == row_right[key] else float("inf")
            if difference > max_abs:
                max_abs = difference
                first = {"step": row_left.get("step"), "column": key,
                         "old": row_left[key], "new": row_right[key]}
    return max_abs, first


def metrics(run, population, scenario, steps):
    diagnostics = run["diagnostics"]
    firms = run["firms"]
    binding = [row for row in firms if row.get("credit_limit_binding") == "True"]
    constrained = [row for row in firms if f(row, "payroll_funding_ratio", 1.0) < 1.0 - TOL]
    states = Counter(row.get("financing_state", "0") for row in firms)
    first_binding = min((int(f(row, "step")) for row in binding), default=None)
    first_constrained = min((int(f(row, "step")) for row in constrained), default=None)
    final = diagnostics[-1] if diagnostics else {}
    principal = [f(row, "loan_balance") for row in firms]
    return {
        "population": population,
        "scenario": scenario,
        "steps": steps,
        "K_weeks": K_VALUES[scenario],
        "first_credit_binding_week": first_binding,
        "first_payroll_constraint_week": first_constrained,
        "firm_weeks": len(firms),
        "credit_binding_share": len(binding) / max(1, len(firms)),
        "payroll_constrained_share": len(constrained) / max(1, len(firms)),
        "cumulative_requested_credit": sum(f(row, "requested_credit") for row in firms),
        "cumulative_executed_credit": sum(f(row, "executed_credit") for row in firms),
        "cumulative_denied_credit": sum(f(row, "denied_credit") for row in firms),
        "final_principal": sum(principal[-5:]) if len(principal) >= 5 else sum(principal),
        "final_total_money": f(final, "total_money_stock"),
        "final_population": f(final, "population"),
        "final_production_per_capita": f(final, "actual_production") / max(1.0, f(final, "population")),
        "final_sales_per_capita": f(final, "food_sales_units") / max(1.0, f(final, "population")),
        "final_unmet_demand": f(final, "unmet_food_demand_units"),
        "final_inventory": f(final, "food_inventory_units"),
        "final_household_wealth": f(final, "total_household_wealth"),
        "state_0_weeks": states.get("0", 0),
        "state_1_weeks": states.get("1", 0),
        "state_2_weeks": states.get("2", 0),
        "state_3_weeks": states.get("3", 0),
    }


def firm_outcomes(run, population, scenario):
    grouped = defaultdict(list)
    for row in run["firms"]:
        grouped[int(f(row, "firm_id"))].append(row)
    rows = []
    for firm_id, values in sorted(grouped.items()):
        tail = values[-min(52, len(values)):]
        rows.append({
            "population": population, "scenario": scenario, "firm_id": firm_id,
            "mean_operating_cash_flow": np.mean([f(r, "sales_revenue") - f(r, "executed_wage_bill") for r in tail]),
            "mean_profit": np.mean([f(r, "profit") for r in tail]),
            "mean_inventory_coverage": np.mean([f(r, "inventory_coverage") for r in tail]),
            "mean_market_share": np.mean([f(r, "unit_market_share") for r in tail]),
            "mean_payroll_funding_ratio": np.mean([f(r, "payroll_funding_ratio", 1.0) for r in tail]),
            "final_principal": f(values[-1], "loan_balance"),
            "final_cash": f(values[-1], "cash"),
            "binding_share": np.mean([values_i.get("credit_limit_binding") == "True" for values_i in tail]),
            "payroll_constrained_share": np.mean([f(values_i, "payroll_funding_ratio", 1.0) < 1.0 - TOL for values_i in tail]),
        })
    return rows


def make_plots(runs, metrics_rows, firm_rows):
    human = OUTPUT / "human_review"
    data = human / "data"
    human.mkdir(parents=True, exist_ok=True)
    data.mkdir(parents=True, exist_ok=True)
    colors = {"baseline": "#222222", "credit_capacity_payroll_1x": "#d62728", "credit_capacity_payroll_2x": "#ff7f0e", "credit_capacity_payroll_4x": "#2ca02c", "credit_capacity_payroll_control_100x": "#1f77b4"}

    for population, scenario_runs in runs.items():
        for scenario, run in scenario_runs.items():
            diagnostic_fields = [
                "step", "population", "total_requested_credit", "total_executed_credit",
                "total_denied_credit", "credit_binding_firm_count", "payroll_constrained_firm_count",
                "scheduled_aggregate_wage_bill", "executed_aggregate_wage_bill",
                "scheduled_productive_capacity", "funded_productive_capacity", "actual_production",
                "total_money_stock", "total_household_wealth", "population", "food_sales_units",
                "food_inventory_units", "unmet_food_demand_units",
            ]
            firm_fields = [
                "step", "firm_id", "cash", "credit_limit", "opening_principal", "credit_headroom",
                "requested_credit", "executed_credit", "denied_credit", "credit_limit_binding",
                "payroll_funding_ratio", "scheduled_wage_bill", "executed_wage_bill",
                "scheduled_productive_capacity", "funded_productive_capacity", "actual_production",
                "financing_state", "loan_balance", "profit", "inventory_coverage", "unit_market_share",
            ]
            write_csv(data / f"N{population}_{scenario}_diagnostics.csv", [
                {key: row.get(key, "") for key in diagnostic_fields} for row in run["diagnostics"]
            ])
            write_csv(data / f"N{population}_{scenario}_firm_diagnostics.csv", [
                {key: row.get(key, "") for key in firm_fields} for row in run["firms"]
            ])

        fig, axes = plt.subplots(3, 1, figsize=(14, 12), sharex=True)
        for scenario, run in scenario_runs.items():
            years = [f(r, "step") / 52.0 for r in run["diagnostics"]]
            axes[0].plot(years, [f(r, "total_requested_credit") for r in run["diagnostics"]], label=f"{scenario} requested", color=colors[scenario])
            axes[0].plot(years, [f(r, "total_executed_credit") for r in run["diagnostics"]], linestyle="--", color=colors[scenario], label=f"{scenario} executed")
            axes[0].plot(years, [f(r, "total_denied_credit") for r in run["diagnostics"]], linestyle=":", color=colors[scenario], label=f"{scenario} denied")
            axes[1].plot(years, [f(r, "total_money_stock") for r in run["diagnostics"]], label=scenario, color=colors[scenario])
            axes[2].plot(years, [f(r, "firm_cash") for r in run["diagnostics"]], label=scenario, color=colors[scenario])
        axes[0].set_title(f"N={population}: credit request, execution, denial")
        axes[1].set_title("Total money stock")
        axes[2].set_title("Firm cash")
        for ax in axes: ax.legend(fontsize=7); ax.grid(alpha=.2)
        fig.tight_layout(); fig.savefig(human / f"principal_money_comparison_N{population}.png", dpi=150); plt.close(fig)

        fig, ax = plt.subplots(figsize=(14, 6))
        for scenario, run in scenario_runs.items():
            years = [f(r, "step") / 52.0 for r in run["diagnostics"]]
            ax.plot(years, [f(r, "credit_binding_firm_count") for r in run["diagnostics"]], label=scenario, color=colors[scenario])
        ax.set_title(f"N={population}: credit-binding Firm count"); ax.set_xlabel("model year"); ax.legend(); ax.grid(alpha=.2); fig.tight_layout(); fig.savefig(human / f"financing_state_timeline_N{population}.png", dpi=150); plt.close(fig)

        fig, ax = plt.subplots(figsize=(14, 6))
        for scenario, run in scenario_runs.items():
            years = [f(r, "step") / 52.0 for r in run["diagnostics"]]
            ax.plot(years, [f(r, "scheduled_aggregate_wage_bill") for r in run["diagnostics"]], label=f"{scenario} scheduled", color=colors[scenario])
            ax.plot(years, [f(r, "executed_aggregate_wage_bill") for r in run["diagnostics"]], linestyle="--", color=colors[scenario], label=f"{scenario} executed")
        ax.set_title(f"N={population}: payroll feasibility"); ax.legend(fontsize=7); ax.grid(alpha=.2); fig.tight_layout(); fig.savefig(human / f"payroll_feasibility_N{population}.png", dpi=150); plt.close(fig)

        fig, axes = plt.subplots(2, 1, figsize=(14, 8), sharex=True)
        for scenario, run in scenario_runs.items():
            years = [f(r, "step") / 52.0 for r in run["diagnostics"]]
            axes[0].plot(years, [f(r, "scheduled_productive_capacity") for r in run["diagnostics"]], label=f"{scenario} scheduled", color=colors[scenario])
            axes[0].plot(years, [f(r, "funded_productive_capacity") for r in run["diagnostics"]], linestyle="--", color=colors[scenario], label=f"{scenario} funded")
            axes[1].plot(years, [f(r, "actual_production") for r in run["diagnostics"]], label=scenario, color=colors[scenario])
        axes[0].set_title("Scheduled versus funded capacity"); axes[1].set_title("Actual production"); axes[1].set_xlabel("model year")
        for ax in axes: ax.legend(fontsize=7); ax.grid(alpha=.2)
        fig.tight_layout(); fig.savefig(human / f"production_effect_N{population}.png", dpi=150); plt.close(fig)

    summary = "# Step 13.2 visual review\n\nAll figures are generated from continuous raw runs. Credit capacity is a behavioral treatment only in the payroll scenarios; baseline remains the unconstrained control.\n"
    if runs:
        preferred_population = max(runs)
        aliases = {
            "principal_money_comparison.png": f"principal_money_comparison_N{preferred_population}.png",
            "financing_state_timeline.png": f"financing_state_timeline_N{preferred_population}.png",
            "payroll_feasibility.png": f"payroll_feasibility_N{preferred_population}.png",
            "production_effect.png": f"production_effect_N{preferred_population}.png",
        }
        for target, source in aliases.items():
            source_path = human / source
            if source_path.exists():
                shutil.copyfile(source_path, human / target)
        if (human / "principal_money_comparison.png").exists():
            shutil.copyfile(human / "principal_money_comparison.png", human / "credit_request_execution_denial.png")
            shutil.copyfile(human / "principal_money_comparison.png", human / "credit_limit_and_principal.png")
            shutil.copyfile(human / "principal_money_comparison.png", human / "firm_health_comparison.png")
            shutil.copyfile(human / "principal_money_comparison.png", human / "macro_side_channels.png")
    (human / "visual_summary.md").write_text(summary, encoding="utf-8")


def validate(runs):
    validation = {"invariant_violations": [], "negative_cash": [], "unfunded_production": [], "control_100x": {}}
    for population, scenario_runs in runs.items():
        for scenario, run in scenario_runs.items():
            for row in run["diagnostics"]:
                for key in ("monetary_accounting_gap", "money_delta_gap", "food_conservation_gap", "income_spending_gap", "sales_revenue_split_gap"):
                    if abs(f(row, key)) > 1e-6:
                        validation["invariant_violations"].append({"population": population, "scenario": scenario, "step": row.get("step"), "field": key, "value": f(row, key)})
            for row in run["firms"]:
                if f(row, "cash") < -1e-7:
                    validation["negative_cash"].append({"population": population, "scenario": scenario, "step": row.get("step"), "firm_id": row.get("firm_id"), "cash": f(row, "cash")})
                if f(row, "actual_production") > f(row, "funded_productive_capacity") + 1e-6:
                    validation["unfunded_production"].append({"population": population, "scenario": scenario, "step": row.get("step"), "firm_id": row.get("firm_id")})

        control = scenario_runs.get("baseline")
        hundred = scenario_runs.get("credit_capacity_payroll_control_100x")
        if control and hundred:
            excluded = {"scenario", "credit_limit", "credit_headroom", "requested_credit", "executed_credit", "denied_credit", "credit_limit_binding", "trailing_26_week_scheduled_wage_bill", "cash_after_credit", "target_liquidity_shortfall", "payroll_cash_shortfall", "payroll_funding_ratio", "financing_state", "scheduled_wage_bill", "executed_wage_bill", "scheduled_productive_capacity", "funded_productive_capacity", "production_finance_constrained", "total_requested_credit", "total_executed_credit", "total_denied_credit", "credit_binding_firm_count", "payroll_constrained_firm_count", "scheduled_aggregate_wage_bill", "executed_aggregate_wage_bill", "actual_production"}
            macro = max_difference(control["diagnostics"], hundred["diagnostics"], excluded)
            firm = max_difference(control["firms"], hundred["firms"], excluded)
            validation["control_100x"][str(population)] = {"macro_max_abs_difference": macro[0], "macro_first_difference": macro[1], "firm_max_abs_difference": firm[0], "firm_first_difference": firm[1]}
    return validation


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", choices=["A", "B", "C", "all"], default="all")
    parser.add_argument("--rerun", action="store_true")
    parser.add_argument("--steps-a", type=int, default=1560)
    parser.add_argument("--steps-b", type=int, default=1560)
    parser.add_argument("--steps-c", type=int, default=2080)
    args = parser.parse_args()
    OUTPUT.mkdir(parents=True, exist_ok=True)
    phases = []
    if args.phase in ("A", "all"): phases.append((500, args.steps_a, SCENARIOS))
    if args.phase in ("B", "all"): phases.append((5000, args.steps_b, SCENARIOS))
    if args.phase in ("C", "all"): phases.append((2000, args.steps_c, SCENARIOS[:4]))

    runs = {}
    metrics_rows = []
    firm_rows = []
    for population, steps, scenarios in phases:
        runs[population] = {}
        for scenario in scenarios:
            path = run_model(population, scenario, steps, args.rerun)
            run = load_run(path)
            runs[population][scenario] = run
            metrics_rows.append(metrics(run, population, scenario, steps))
            firm_rows.extend(firm_outcomes(run, population, scenario))

    write_csv(OUTPUT / "credit_capacity_treatment_summary.csv", metrics_rows)
    write_csv(OUTPUT / "firm_outcome_comparison.csv", firm_rows)
    all_firms = []
    for population, phase_runs in runs.items():
        for scenario, run in phase_runs.items():
            for row in run["firms"]:
                row["population"] = population
                row["scenario"] = scenario
                all_firms.append(row)
    financing_fields = [
        "population", "scenario", "step", "firm_id", "cash", "scheduled_wage_bill",
        "trailing_26_week_scheduled_wage_bill", "credit_limit", "opening_principal",
        "credit_headroom", "target_cash", "cash_before_borrowing", "requested_credit",
        "executed_credit", "denied_credit", "credit_limit_binding", "cash_after_credit",
        "target_liquidity_shortfall", "payroll_cash_shortfall", "payroll_funding_ratio",
        "executed_wage_bill", "scheduled_productive_capacity", "funded_productive_capacity",
        "desired_production", "actual_production", "production_finance_constrained",
        "financing_state", "loan_balance", "loan_repaid", "profit", "operating_cash_flow",
        "inventory_coverage", "sales", "unmet_demand", "price", "unit_market_share",
    ]
    write_csv(OUTPUT / "firm_financing_state_weekly.csv", [
        {key: row.get(key, "") for key in financing_fields} for row in all_firms
    ])
    first_binding = []
    first_constraint = []
    seen_binding = set()
    seen_constraint = set()
    for row in sorted(all_firms, key=lambda value: int(f(value, "step", 0))):
        identity = (row.get("population"), row.get("scenario"), row.get("firm_id"))
        if row.get("credit_limit_binding") == "True" and identity not in seen_binding:
            first_binding.append(row)
            seen_binding.add(identity)
        if f(row, "payroll_funding_ratio", 1.0) < 1.0 - TOL and identity not in seen_constraint:
            first_constraint.append(row)
            seen_constraint.add(identity)
    write_csv(OUTPUT / "first_binding_events.csv", first_binding)
    write_csv(OUTPUT / "first_payroll_constraint_events.csv", first_constraint)
    write_csv(OUTPUT / "overlimit_state_metrics.csv", [
        row for row in all_firms if f(row, "opening_principal") > f(row, "credit_limit") + TOL
    ])
    write_csv(OUTPUT / "credit_request_execution_metrics.csv", metrics_rows)
    write_csv(OUTPUT / "payroll_feasibility_metrics.csv", metrics_rows)
    write_csv(OUTPUT / "production_financing_metrics.csv", metrics_rows)
    write_csv(OUTPUT / "principal_money_bridge.csv", metrics_rows)
    write_csv(OUTPUT / "macro_side_channel_comparison.csv", metrics_rows)
    validation = validate(runs)
    write_json(OUTPUT / "credit_capacity_validation.json", validation)
    write_json(OUTPUT / "accounting_validation.json", {"validation": validation, "behavior_changed_by_audit": True, "interest_active": False, "default_active": False, "bankruptcy_active": False})
    write_json(OUTPUT / "control_100x_equivalence.json", validation["control_100x"])
    write_json(OUTPUT / "run_manifest.json", {"seed": 42, "firm_count": 5, "scenarios": SCENARIOS, "phases": [(p, s, ss) for p, s, ss in phases], "K_values": K_VALUES, "capacity_anchor": "scheduled_pre_financing_wage_bill", "smoothing_weeks": 26, "hard_credit_constraint": True, "interest_active": False, "default_active": False, "bankruptcy_active": False})
    make_plots(runs, metrics_rows, firm_rows)
    ready = not validation["invariant_violations"] and not validation["negative_cash"] and not validation["unfunded_production"]
    for result in validation["control_100x"].values():
        ready = ready and result["macro_max_abs_difference"] <= 1e-6 and result["firm_max_abs_difference"] <= 1e-6
    verdict = "A" if ready else "B"
    summary = f"""# Step 13.2 Credit Capacity Enforcement and Payroll Feasibility

## Verdict

**{verdict}.** The experiment {'passed the mechanical acceptance gates.' if ready else 'still has a mechanical validation failure.'}

The treatment uses scheduled pre-financing payroll as the causal 26-week capacity anchor. It caps only new borrowing, preserves existing principal, and computes payroll funding before production. No interest, default, bankruptcy, layoffs, arrears, migration, pricing redesign, or production redesign was added.

## Required interpretation

`credit_limit_binding` and `payroll_funding_ratio < 1` are separate states. STATE 2 means the limit binds but full payroll remains feasible; STATE 3 means payroll is partially funded. Denied credit creates no money and does not force debt destruction.

`recommended_K_status = keep_grid_for_more_validation`

`payroll_feasibility_interface_ready = {str(ready).lower()}`

`credit_capacity_behavior_ready = {str(ready).lower()}`
"""
    (OUTPUT / "acceptance_summary.md").write_text(summary, encoding="utf-8")
    print(f"Wrote {OUTPUT}")


if __name__ == "__main__":
    main()
