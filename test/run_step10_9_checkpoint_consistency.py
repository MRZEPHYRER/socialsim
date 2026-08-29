import argparse
import csv
import os
import subprocess
import sys


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

    return result


def read_diagnostics(path):
    with open(path, newline="", encoding="utf-8") as file:
        return list(csv.DictReader(file))


def compare_rows(left_rows, right_rows):
    if len(left_rows) != len(right_rows):
        return (
            f"row count mismatch: continuous={len(left_rows)}, "
            f"checkpoint={len(right_rows)}"
        )

    for index, (left, right) in enumerate(zip(left_rows, right_rows)):
        if left.keys() != right.keys():
            return f"column mismatch at row {index}"

        for key in left:
            if left[key] != right[key]:
                return (
                    f"value mismatch at row {index}, column {key}: "
                    f"continuous={left[key]!r}, checkpoint={right[key]!r}"
                )

    return None


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--population", type=int, default=200)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--scenario", default="wage_shock_1_47")
    parser.add_argument("--checkpoint-step", type=int, default=100)
    parser.add_argument("--continue-steps", type=int, default=20)
    parser.add_argument(
        "--output-dir",
        default="test/output/step10_9_checkpoint_consistency",
    )
    parser.add_argument(
        "--diagnostics-mode",
        choices=["full", "compact"],
        default="compact",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    total_steps = args.checkpoint_step + args.continue_steps
    continuous_dir = os.path.join(args.output_dir, "continuous")
    checkpoint_dir = os.path.join(args.output_dir, "checkpoint")
    resumed_dir = os.path.join(args.output_dir, "resumed")
    checkpoint_path = os.path.join(checkpoint_dir, "world_step_checkpoint.pkl")
    continuous_csv = os.path.join(continuous_dir, "diagnostics.csv")
    resumed_csv = os.path.join(resumed_dir, "diagnostics.csv")

    base_command = [
        sys.executable,
        "main.py",
        "--scenario",
        args.scenario,
        "--population",
        str(args.population),
        "--seed",
        str(args.seed),
        "--progress-interval",
        "0",
        "--steady-window",
        str(min(500, total_steps)),
        "--no-plots",
        "--no-analysis",
        "--diagnostics-mode",
        args.diagnostics_mode,
    ]

    run_command(
        base_command
        +
        [
            "--steps",
            str(total_steps),
            "--output-dir",
            continuous_dir,
        ]
    )
    run_command(
        base_command
        +
        [
            "--steps",
            str(args.checkpoint_step),
            "--output-dir",
            checkpoint_dir,
            "--save-checkpoint",
            checkpoint_path,
        ]
    )
    run_command(
        [
            sys.executable,
            "main.py",
            "--load-checkpoint",
            checkpoint_path,
            "--steps",
            str(args.continue_steps),
            "--progress-interval",
            "0",
            "--steady-window",
            str(min(500, total_steps)),
            "--no-plots",
            "--no-analysis",
            "--diagnostics-mode",
            args.diagnostics_mode,
            "--output-dir",
            resumed_dir,
        ]
    )

    mismatch = compare_rows(
        read_diagnostics(continuous_csv),
        read_diagnostics(resumed_csv),
    )

    if mismatch:
        raise SystemExit(mismatch)

    print("Checkpoint consistency passed.")
    print(f"Continuous diagnostics: {continuous_csv}")
    print(f"Resumed diagnostics: {resumed_csv}")
    print(f"Checkpoint: {checkpoint_path}")


if __name__ == "__main__":
    main()
