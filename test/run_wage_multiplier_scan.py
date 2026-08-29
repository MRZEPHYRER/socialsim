import argparse
import csv
import json
import os
import shutil
import subprocess
import sys
import time

import numpy as np

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from scenarios import git_version_identifier


WAGE_SCENARIOS = [
    (1.20, "wage_shock_1_2"),
    (1.25, "wage_shock_1_25"),
    (1.30, "wage_shock"),
    (1.35, "wage_shock_1_35"),
    (1.40, "wage_shock_1_4"),
    (1.45, "wage_shock_1_45"),
    (1.50, "wage_shock_1_5"),
]

FINE_WAGE_SCENARIOS = [
    (1.46, "wage_shock_1_46"),
    (1.47, "wage_shock_1_47"),
    (1.48, "wage_shock_1_48"),
    (1.49, "wage_shock_1_49"),
]

REUSABLE_RESULT_DIRS = {
    "wage_shock_1_5": os.path.join(
        PROJECT_ROOT,
        "test",
        "output",
        "step10_5A_public_wealth_audit",
        "baseline_vs_wage_1_5_v2",
        "wage_shock_1_5",
    ),
}

SUMMARY_FIELDS = [
    "wage_multiplier",
    "scenario",
    "status",
    "returncode",
    "elapsed_seconds",
    "reused_existing_result",
    "output_dir",
    "diagnostics_csv",
    "manifest_json",
    "first_loan_step",
    "final_loan_balance",
    "loan_slope_last_500",
    "final_firm_cash_to_wage_bill",
    "cumulative_no_heir_public_wealth",
    "final_public_money_share",
]


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run Step 10.5B wage multiplier scan."
    )
    parser.add_argument("--population", type=int, default=5000)
    parser.add_argument("--steps", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--shock-step", type=int, default=1000)
    parser.add_argument("--steady-window", type=int, default=500)
    parser.add_argument(
        "--output-dir",
        default=None,
        help=(
            "Batch output directory. Default: "
            "test/output/step10_5B_wage_scan/seed_<seed>_pop_<population>_steps_<steps>."
        ),
    )
    parser.add_argument(
        "--no-reuse",
        action="store_true",
        help="Do not reuse matching existing diagnostics from earlier runs.",
    )
    parser.add_argument(
        "--progress-interval",
        type=int,
        default=0,
    )
    parser.add_argument(
        "--full-diagnostics",
        action="store_true",
        help="Write full diagnostics and analysis output. Default is compact and faster.",
    )
    parser.add_argument(
        "--fine-grid",
        action="store_true",
        help="Run only wage multipliers 1.46, 1.47, 1.48, and 1.49.",
    )
    return parser.parse_args()


def batch_output_dir(args):
    if args.output_dir:
        if os.path.isabs(args.output_dir):
            return args.output_dir

        return os.path.join(PROJECT_ROOT, args.output_dir)

    return os.path.join(
        PROJECT_ROOT,
        "test",
        "output",
        "step10_5B_wage_scan",
        f"seed_{args.seed}_pop_{args.population}_steps_{args.steps}",
    )


def read_diagnostics(path):
    with open(path, "r", encoding="utf-8", newline="") as file:
        return list(csv.DictReader(file))


def float_value(row, field, default=0.0):
    try:
        return float(row.get(field) or default)
    except ValueError:
        return default


def series_slope(values):
    values = np.asarray(values, dtype=float)

    if len(values) <= 1:
        return 0.0

    x = np.arange(len(values), dtype=float)
    return float(np.polyfit(x, values, 1)[0])


def median(values):
    if not values:
        return 0.0

    return float(np.median(np.asarray(values, dtype=float)))


def analyze_diagnostics(rows, window):
    final = rows[-1]
    loan_balances = [
        float_value(row, "working_capital_loan_balance")
        for row in rows
    ]
    loan_issued = [
        float_value(row, "working_capital_loan_issued")
        for row in rows
    ]
    loan_repaid = [
        float_value(row, "working_capital_loan_repaid")
        for row in rows
    ]
    cash_to_wage = [
        float_value(row, "firm_cash_to_wage_bill")
        for row in rows
    ]
    security = [
        float_value(row, "median_security_ratio")
        for row in rows
    ]
    tail = min(window, len(rows))
    loan_tail = loan_balances[-tail:]

    first_loan_step = ""

    for row, balance in zip(rows, loan_balances):
        issued = float_value(row, "working_capital_loan_issued")

        if balance > 1e-9 or issued > 1e-9:
            first_loan_step = row.get("step", "")
            break

    return {
        "first_loan_step": first_loan_step,
        "final_loan_balance": loan_balances[-1],
        "peak_loan_balance": max(loan_balances) if loan_balances else 0.0,
        "loan_slope_last_500": series_slope(loan_tail),
        "avg_loan_issued_last_500": float(np.mean(loan_issued[-tail:])),
        "avg_loan_repaid_last_500": float(np.mean(loan_repaid[-tail:])),
        "final_firm_cash_to_wage_bill": float_value(
            final,
            "firm_cash_to_wage_bill",
        ),
        "median_firm_cash_to_wage_bill_last_500": median(cash_to_wage[-tail:]),
        "final_median_security_ratio": float_value(
            final,
            "median_security_ratio",
        ),
        "median_security_ratio_last_500": median(security[-tail:]),
        "final_population": float_value(final, "population"),
        "cumulative_no_heir_public_wealth": float_value(
            final,
            "cumulative_public_wealth_from_no_heir",
        ),
        "final_public_money_share": float_value(final, "public_money_share"),
        "final_public_money_stock": float_value(final, "public_money_stock"),
        "final_central_bank_public_income_balance": float_value(
            final,
            "central_bank_public_income_balance",
        ),
        "final_public_wealth": float_value(final, "public_wealth"),
        "final_total_money_stock": float_value(final, "total_money_stock"),
        "final_net_new_money": float_value(final, "net_new_money"),
        "final_food_price": float_value(final, "food_price"),
        "final_unit_labor_cost": float_value(final, "unit_labor_cost"),
        "final_invariant_failed": final.get("invariant_failed", ""),
    }


def copy_reusable_result(source_dir, target_dir):
    if not os.path.exists(os.path.join(source_dir, "diagnostics.csv")):
        return False

    if not os.path.exists(target_dir):
        shutil.copytree(source_dir, target_dir)
        return True

    target_diagnostics = os.path.join(target_dir, "diagnostics.csv")

    if not os.path.exists(target_diagnostics):
        for name in os.listdir(source_dir):
            source = os.path.join(source_dir, name)
            target = os.path.join(target_dir, name)

            if os.path.isdir(source):
                shutil.copytree(source, target, dirs_exist_ok=True)
            else:
                shutil.copy2(source, target)

        return True

    return False


def can_reuse_existing_result(args):
    return (
        args.population == 5000
        and
        args.steps == 5000
        and
        args.seed == 42
        and
        args.shock_step == 1000
    )


def run_scenario(args, batch_dir, multiplier, scenario):
    scenario_dir = os.path.join(
        batch_dir,
        f"wage_{multiplier:.2f}".replace(".", "_"),
    )
    os.makedirs(scenario_dir, exist_ok=True)
    diagnostics_csv = os.path.join(scenario_dir, "diagnostics.csv")
    manifest_json = os.path.join(scenario_dir, "manifest.json")
    run_log = os.path.join(scenario_dir, "run.log")

    reused = False

    if (
        not args.no_reuse
        and
        can_reuse_existing_result(args)
        and
        os.path.exists(diagnostics_csv)
    ):
        return {
            "scenario": scenario,
            "status": "ok",
            "returncode": 0,
            "elapsed_seconds": 0.0,
            "reused_existing_result": True,
            "output_dir": scenario_dir,
            "diagnostics_csv": diagnostics_csv,
            "manifest_json": manifest_json,
            "run_log": run_log,
        }

    source_dir = REUSABLE_RESULT_DIRS.get(scenario)

    if (
        not args.no_reuse
        and
        can_reuse_existing_result(args)
        and
        source_dir is not None
    ):
        reused = copy_reusable_result(source_dir, scenario_dir)

    if reused:
        return {
            "scenario": scenario,
            "status": "ok",
            "returncode": 0,
            "elapsed_seconds": 0.0,
            "reused_existing_result": True,
            "output_dir": scenario_dir,
            "diagnostics_csv": diagnostics_csv,
            "manifest_json": manifest_json,
            "run_log": run_log,
        }

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
    if args.full_diagnostics:
        command.append("--diagnostics-mode")
        command.append("full")
    else:
        command.append("--diagnostics-mode")
        command.append("compact")
        command.append("--no-analysis")
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

    return {
        "scenario": scenario,
        "status": "ok" if result.returncode == 0 else "failed",
        "returncode": result.returncode,
        "elapsed_seconds": elapsed_seconds,
        "reused_existing_result": False,
        "output_dir": scenario_dir,
        "diagnostics_csv": diagnostics_csv,
        "manifest_json": manifest_json,
        "run_log": run_log,
    }


def write_summary(path, rows):
    with open(path, "w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=SUMMARY_FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def write_manifest(path, args, runs, wage_scenarios):
    manifest = {
        "batch": "step10_5B_wage_multiplier_scan",
        "seed": args.seed,
        "population": args.population,
        "steps": args.steps,
        "shock_step": args.shock_step,
        "steady_window": args.steady_window,
        "wage_scenarios": [
            {
                "wage_multiplier": multiplier,
                "scenario": scenario,
            }
            for multiplier, scenario in wage_scenarios
        ],
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
    wage_scenarios = FINE_WAGE_SCENARIOS if args.fine_grid else WAGE_SCENARIOS

    if args.shock_step != 1000:
        raise ValueError(
            "Current scenario definitions use WAGE_MULTIPLIER_STEP=1000."
        )

    batch_dir = batch_output_dir(args)
    os.makedirs(batch_dir, exist_ok=True)
    print("Step 10.5B wage multiplier scan")
    print("Output directory:", batch_dir)

    runs = []
    summary_rows = []

    for index, (multiplier, scenario) in enumerate(wage_scenarios, start=1):
        print(
            f"[{index}/{len(wage_scenarios)}] "
            f"Running wage multiplier {multiplier:.2f} ({scenario})..."
        )
        run = run_scenario(args, batch_dir, multiplier, scenario)
        runs.append(run)

        if run["status"] == "ok":
            diagnostics = read_diagnostics(run["diagnostics_csv"])
            metrics = analyze_diagnostics(diagnostics, args.steady_window)
        else:
            metrics = {}

        row = {
            "wage_multiplier": f"{multiplier:.2f}",
            **run,
            **metrics,
        }
        summary_rows.append({
            field: row.get(field, "")
            for field in SUMMARY_FIELDS
        })
        print(
            f"[{index}/{len(wage_scenarios)}] "
            f"{scenario}: {run['status']} "
            f"reused={run['reused_existing_result']} "
            f"elapsed={run['elapsed_seconds']:.2f}s"
        )

    summary_csv = os.path.join(batch_dir, "wage_scan_summary.csv")
    manifest_json = os.path.join(batch_dir, "wage_scan_manifest.json")
    write_summary(summary_csv, summary_rows)
    write_manifest(manifest_json, args, runs, wage_scenarios)
    print("Summary CSV:", summary_csv)
    print("Manifest:", manifest_json)

    failed = [
        run["scenario"]
        for run in runs
        if run["status"] != "ok"
    ]

    if failed:
        print("Failed scenarios:", ", ".join(failed))
        return 1

    print("All wage multiplier scans finished.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
