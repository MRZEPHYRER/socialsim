import argparse
import csv
import os
import subprocess
import sys
from collections import defaultdict


DEFAULT_CHECKPOINT = (
    "test/output/step10_9_warm_checkpoint/"
    "wage_shock_1_47_seed_42_pop_5000_step_5000/"
    "world_step_5000.pkl"
)


def run_command(command):
    result = subprocess.run(
        command,
        capture_output=True,
        text=True,
    )

    if result.returncode != 0:
        print(result.stdout)
        print(result.stderr, file=sys.stderr)
        raise SystemExit(result.returncode)


def read_rows(path):
    with open(path, newline="", encoding="utf-8") as file:
        return list(csv.DictReader(file))


def compare_macro(control_csv, treatment_csv):
    control_rows = read_rows(control_csv)
    treatment_rows = read_rows(treatment_csv)

    if len(control_rows) != len(treatment_rows):
        raise SystemExit(
            "macro row count mismatch: "
            f"{len(control_rows)} vs {len(treatment_rows)}"
        )

    for index, (control, treatment) in enumerate(
        zip(control_rows, treatment_rows)
    ):
        if control.keys() != treatment.keys():
            raise SystemExit(f"macro column mismatch at row {index}")

        for key in control:
            if control[key] != treatment[key]:
                raise SystemExit(
                    f"macro mismatch row={index} key={key}: "
                    f"control={control[key]!r} treatment={treatment[key]!r}"
                )

    return treatment_rows


def check_firm_aggregation(macro_rows, firm_csv):
    firm_rows = read_rows(firm_csv)
    by_step = defaultdict(list)

    for row in firm_rows:
        by_step[int(row["step"])].append(row)

    macro_by_step = {
        int(row["step"]): row
        for row in macro_rows
    }
    checks = [
        ("firm_cash", "cash"),
        ("firm_inventory_value", "inventory"),
        ("loan_balance", "loan_balance"),
        ("wage_bill", "wage_bill"),
        ("firm_sales_revenue", "sales"),
    ]
    max_gap = 0.0

    for step, rows in by_step.items():
        macro = macro_by_step[step]

        for macro_key, firm_key in checks:
            firm_total = sum(float(row[firm_key]) for row in rows)
            gap = abs(firm_total - float(macro[macro_key]))
            max_gap = max(max_gap, gap)

    if max_gap > 1e-6:
        raise SystemExit(f"firm aggregation gap too large: {max_gap}")

    return len(firm_rows), len(by_step), max_gap


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default=DEFAULT_CHECKPOINT)
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument(
        "--output-dir",
        default="test/output/step11A_aggregate_equivalence",
    )
    parser.add_argument(
        "--diagnostics-mode",
        choices=["full", "compact"],
        default="compact",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    control_dir = os.path.join(args.output_dir, "control_1_firm")
    treatment_dir = os.path.join(args.output_dir, "treatment_5_firms")

    base = [
        sys.executable,
        "main.py",
        "--load-checkpoint",
        args.checkpoint,
        "--steps",
        str(args.steps),
        "--progress-interval",
        "0",
        "--steady-window",
        str(min(500, args.steps)),
        "--no-plots",
        "--no-analysis",
        "--diagnostics-mode",
        args.diagnostics_mode,
    ]
    run_command(
        base
        +
        [
            "--firm-count",
            "1",
            "--output-dir",
            control_dir,
        ]
    )
    run_command(
        base
        +
        [
            "--firm-count",
            "5",
            "--output-dir",
            treatment_dir,
        ]
    )

    treatment_rows = compare_macro(
        os.path.join(control_dir, "diagnostics.csv"),
        os.path.join(treatment_dir, "diagnostics.csv"),
    )
    firm_row_count, firm_step_count, max_gap = check_firm_aggregation(
        treatment_rows,
        os.path.join(treatment_dir, "firm_diagnostics.csv"),
    )
    print("Step 11A aggregate equivalence passed.")
    print(f"firm_rows={firm_row_count}")
    print(f"firm_steps={firm_step_count}")
    print(f"max_firm_aggregate_gap={max_gap}")


if __name__ == "__main__":
    main()
