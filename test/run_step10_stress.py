import argparse
import csv
import json
import os
import subprocess
import sys
import time

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from scenarios import git_version_identifier


DEFAULT_SCENARIOS = [
    "baseline",
    "liquidity_ratio_10",
    "liquidity_ratio_5",
    "liquidity_ratio_2",
    "liquidity_ratio_1",
    "liquidity_ratio_0_5",
    "wage_shock_1_2",
    "wage_shock_1_5",
    "demand_shock",
    "demand_shock_0_5",
    "population_boom",
]

SUMMARY_FIELDS = [
    "scenario",
    "status",
    "returncode",
    "elapsed_seconds",
    "output_dir",
    "diagnostics_csv",
    "manifest_json",
    "final_step",
    "final_population",
    "final_birth_rate",
    "final_food_price",
    "final_unit_labor_cost",
    "final_unit_labor_cost_growth",
    "final_price_inventory_gap",
    "final_price_inflation_signal",
    "final_price_log_adjustment",
    "final_demand_pressure",
    "final_inventory_demand_ratio",
    "final_firm_cash",
    "final_firm_cash_to_wage_bill",
    "final_working_capital_loan_balance",
    "final_working_capital_loan_issued",
    "final_working_capital_loan_repaid",
    "final_credit_money_outstanding",
    "final_total_money_stock",
    "final_net_new_money",
    "final_simplified_money_velocity",
    "final_median_security_ratio",
    "final_share_households_below_target",
    "invariant_violations",
]


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run Step 10 stress scenarios sequentially."
    )
    parser.add_argument(
        "--population",
        type=int,
        default=5000,
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=5000,
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
    )
    parser.add_argument(
        "--steady-window",
        type=int,
        default=500,
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help=(
            "Batch output directory. Default: "
            "test/output/step10_stress_batch/seed_<seed>_pop_<population>_steps_<steps>."
        ),
    )
    parser.add_argument(
        "--scenario",
        action="append",
        choices=DEFAULT_SCENARIOS,
        help="Scenario to run. Can be repeated. Default: all Step 10 scenarios.",
    )
    parser.add_argument(
        "--include-ledger",
        action="store_true",
        help="Write detailed ledger.csv for each scenario. This is slower.",
    )
    parser.add_argument(
        "--progress-interval",
        type=int,
        default=0,
    )
    parser.add_argument(
        "--show-main-output",
        action="store_true",
        help="Also print each main.py output to the console.",
    )

    return parser.parse_args()


def batch_output_dir(args):
    if args.output_dir:
        if os.path.isabs(args.output_dir):
            return args.output_dir

        return os.path.join(PROJECT_ROOT, args.output_dir)

    run_name = (
        f"seed_{args.seed}_"
        f"pop_{args.population}_"
        f"steps_{args.steps}"
    )

    return os.path.join(
        PROJECT_ROOT,
        "test",
        "output",
        "step10_stress_batch",
        run_name,
    )


def read_final_diagnostics(path):
    if not os.path.exists(path):
        return {}

    with open(path, "r", encoding="utf-8", newline="") as file:
        rows = list(csv.DictReader(file))

    if not rows:
        return {}

    return rows[-1]


def final_value(row, field):
    value = row.get(field)

    if value is None:
        return ""

    return value


def build_summary_row(run, final_diagnostics):
    row = {
        "scenario": run["scenario"],
        "status": run["status"],
        "returncode": run["returncode"],
        "elapsed_seconds": f"{run['elapsed_seconds']:.3f}",
        "output_dir": run["output_dir"],
        "diagnostics_csv": run["diagnostics_csv"],
        "manifest_json": run["manifest_json"],
        "final_step": final_value(final_diagnostics, "step"),
        "final_population": final_value(final_diagnostics, "population"),
        "final_birth_rate": final_value(final_diagnostics, "birth_rate"),
        "final_food_price": final_value(final_diagnostics, "food_price"),
        "final_unit_labor_cost": final_value(final_diagnostics, "unit_labor_cost"),
        "final_unit_labor_cost_growth": final_value(
            final_diagnostics,
            "unit_labor_cost_growth",
        ),
        "final_price_inventory_gap": final_value(
            final_diagnostics,
            "price_inventory_gap",
        ),
        "final_price_inflation_signal": final_value(
            final_diagnostics,
            "price_inflation_signal",
        ),
        "final_price_log_adjustment": final_value(
            final_diagnostics,
            "price_log_adjustment",
        ),
        "final_demand_pressure": final_value(final_diagnostics, "demand_pressure"),
        "final_inventory_demand_ratio": final_value(
            final_diagnostics,
            "inventory_demand_ratio",
        ),
        "final_firm_cash": final_value(final_diagnostics, "firm_cash"),
        "final_firm_cash_to_wage_bill": final_value(
            final_diagnostics,
            "firm_cash_to_wage_bill",
        ),
        "final_working_capital_loan_balance": final_value(
            final_diagnostics,
            "working_capital_loan_balance",
        ),
        "final_working_capital_loan_issued": final_value(
            final_diagnostics,
            "working_capital_loan_issued",
        ),
        "final_working_capital_loan_repaid": final_value(
            final_diagnostics,
            "working_capital_loan_repaid",
        ),
        "final_credit_money_outstanding": final_value(
            final_diagnostics,
            "credit_money_outstanding",
        ),
        "final_total_money_stock": final_value(final_diagnostics, "total_money_stock"),
        "final_net_new_money": final_value(final_diagnostics, "net_new_money"),
        "final_simplified_money_velocity": final_value(
            final_diagnostics,
            "simplified_money_velocity",
        ),
        "final_median_security_ratio": final_value(
            final_diagnostics,
            "median_security_ratio",
        ),
        "final_share_households_below_target": final_value(
            final_diagnostics,
            "share_households_below_target",
        ),
        "invariant_violations": final_value(
            final_diagnostics,
            "invariant_violations",
        ),
    }

    return row


def run_scenario(args, batch_dir, scenario):
    scenario_dir = os.path.join(batch_dir, scenario)
    os.makedirs(scenario_dir, exist_ok=True)

    diagnostics_csv = os.path.join(scenario_dir, "diagnostics.csv")
    manifest_json = os.path.join(scenario_dir, "manifest.json")
    run_log = os.path.join(scenario_dir, "run.log")

    command = [
        sys.executable,
        "main.py",
        "--scenario",
        scenario,
        "--population",
        str(args.population),
        "--steps",
        str(args.steps),
        "--seed",
        str(args.seed),
        "--steady-window",
        str(args.steady_window),
        "--progress-interval",
        str(args.progress_interval),
        "--no-plots",
        "--output-dir",
        scenario_dir,
    ]

    if args.include_ledger:
        command.append("--ledger-csv")

    start_time = time.perf_counter()
    result = subprocess.run(
        command,
        capture_output=True,
        text=True,
        cwd=PROJECT_ROOT,
    )
    elapsed_seconds = time.perf_counter() - start_time

    with open(run_log, "w", encoding="utf-8") as file:
        file.write(result.stdout)

        if result.stderr:
            file.write("\n\n--- STDERR ---\n")
            file.write(result.stderr)

    if args.show_main_output:
        print(result.stdout)

        if result.stderr:
            print(result.stderr, file=sys.stderr)

    status = "ok" if result.returncode == 0 else "failed"

    return {
        "scenario": scenario,
        "status": status,
        "returncode": result.returncode,
        "elapsed_seconds": elapsed_seconds,
        "output_dir": scenario_dir,
        "diagnostics_csv": diagnostics_csv,
        "manifest_json": manifest_json,
        "run_log": run_log,
        "command": command,
    }


def write_summary(path, rows):
    with open(path, "w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=SUMMARY_FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def write_batch_manifest(path, args, scenarios, runs):
    manifest = {
        "batch": "step10_stress",
        "seed": args.seed,
        "population": args.population,
        "steps": args.steps,
        "steady_window": args.steady_window,
        "include_ledger": args.include_ledger,
        "scenarios": scenarios,
        "runs": runs,
        "version_identifier": git_version_identifier(),
    }

    with open(path, "w", encoding="utf-8") as file:
        json.dump(
            manifest,
            file,
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
        file.write("\n")


def main():
    args = parse_args()
    scenarios = args.scenario or DEFAULT_SCENARIOS
    batch_dir = batch_output_dir(args)
    os.makedirs(batch_dir, exist_ok=True)

    print("Step 10 stress batch")
    print("Output directory:", batch_dir)
    print("Scenarios:", ", ".join(scenarios))

    runs = []
    summary_rows = []

    for index, scenario in enumerate(scenarios, start=1):
        print(f"[{index}/{len(scenarios)}] Running {scenario}...")
        run = run_scenario(args, batch_dir, scenario)
        final_diagnostics = read_final_diagnostics(run["diagnostics_csv"])
        summary_rows.append(
            build_summary_row(run, final_diagnostics)
        )
        runs.append(run)
        print(
            f"[{index}/{len(scenarios)}] {scenario}: "
            f"{run['status']} in {run['elapsed_seconds']:.2f}s"
        )

    summary_csv = os.path.join(batch_dir, "summary.csv")
    batch_manifest = os.path.join(batch_dir, "batch_manifest.json")
    write_summary(summary_csv, summary_rows)
    write_batch_manifest(batch_manifest, args, scenarios, runs)

    failed = [
        run["scenario"]
        for run in runs
        if run["status"] != "ok"
    ]

    print("Summary CSV:", summary_csv)
    print("Batch manifest:", batch_manifest)

    if failed:
        print("Failed scenarios:", ", ".join(failed))
        return 1

    print("All scenarios finished.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
