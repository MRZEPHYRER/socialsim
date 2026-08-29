"""Step 12.R3 analysis audit and final weekly-baseline validation helpers."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def read_csv(path):
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def value(raw):
    if raw in ("", None):
        return raw
    try:
        result = float(raw)
        return result if math.isfinite(result) else raw
    except (TypeError, ValueError):
        return raw


def compare_csv(old_path, new_path, excluded=()):
    old_rows = read_csv(old_path)
    new_rows = read_csv(new_path)
    result = {
        "file": str(old_path.name),
        "old_rows": len(old_rows),
        "new_rows": len(new_rows),
        "max_abs_difference": 0.0,
        "first_difference": None,
        "exact": len(old_rows) == len(new_rows),
    }
    if len(old_rows) != len(new_rows):
        result["first_difference"] = "row_count"
        return result
    columns = sorted((set(old_rows[0]) & set(new_rows[0])) - set(excluded)) if old_rows else []
    for index, (old, new) in enumerate(zip(old_rows, new_rows)):
        for column in columns:
            left = value(old.get(column))
            right = value(new.get(column))
            if isinstance(left, float) and isinstance(right, float):
                difference = abs(left - right)
                result["max_abs_difference"] = max(result["max_abs_difference"], difference)
                if difference != 0 and result["first_difference"] is None:
                    result["first_difference"] = {"row": index, "column": column, "old": left, "new": right}
            elif left != right:
                result["exact"] = False
                if result["first_difference"] is None:
                    result["first_difference"] = {"row": index, "column": column, "old": left, "new": right}
    result["exact"] = result["exact"] and result["max_abs_difference"] == 0 and result["first_difference"] is None
    return result


def compare_runs(old_dir, new_dir):
    files = [
        "diagnostics.csv", "firm_diagnostics.csv", "household_diagnostics.csv",
        "demographic_events.csv", "marriage_market_diagnostics.csv",
        "demographic_annual_summary.csv",
        "accounting/accounting_reconciliation.csv",
        "accounting/central_bank_accounting.csv",
        "accounting/firm_accounting.csv",
        "accounting/household_accounting.csv",
        "accounting/household_lifecycle.csv",
        "accounting/public_accounting.csv",
    ]
    comparisons = []
    for relative in files:
        old_path = old_dir / relative
        new_path = new_dir / relative
        if old_path.exists() and new_path.exists():
            comparisons.append(compare_csv(old_path, new_path))
    return {
        "comparisons": comparisons,
        "max_behavioral_difference": max((item["max_abs_difference"] for item in comparisons), default=0.0),
        "all_exact": all(item["exact"] for item in comparisons),
    }


def maximum(rows, field):
    return max((abs(float(row.get(field, 0.0) or 0.0)) for row in rows), default=0.0)


def cv(values):
    if not values:
        return 0.0
    mean = sum(values) / len(values)
    if abs(mean) <= 1e-15:
        return 0.0
    variance = sum((item - mean) ** 2 for item in values) / len(values)
    return math.sqrt(variance) / abs(mean)


def truthy(raw):
    if isinstance(raw, bool):
        return raw
    return str(raw).strip().lower() in {"1", "true", "yes", "y"}


def validate_baseline(run_dir):
    diagnostics = read_csv(run_dir / "diagnostics.csv")
    firms = read_csv(run_dir / "firm_diagnostics.csv")
    accounting = run_dir / "accounting"
    gap_specs = {
        "income_spending_gap": (run_dir / "diagnostics.csv", "income_spending_gap"),
        "sales_revenue_split_gap": (run_dir / "diagnostics.csv", "sales_revenue_split_gap"),
        "monetary_accounting_gap": (run_dir / "diagnostics.csv", "monetary_accounting_gap"),
        "money_delta_gap": (run_dir / "diagnostics.csv", "money_delta_gap"),
        "money_location_gap": (accounting / "accounting_reconciliation.csv", "money_location_gap"),
        "loan_reconciliation_gap": (accounting / "accounting_reconciliation.csv", "loan_reconciliation_gap"),
        "goods_conservation_gap": (run_dir / "diagnostics.csv", "food_conservation_gap"),
        "firm_cash_flow_gap": (accounting / "firm_accounting.csv", "cash_flow_gap"),
        "firm_balance_sheet_gap": (accounting / "firm_accounting.csv", "balance_sheet_gap"),
        "inventory_bridge_gap": (accounting / "firm_accounting.csv", "inventory_bridge_gap"),
        "equity_bridge_gap": (accounting / "firm_accounting.csv", "equity_bridge_gap"),
        "household_wealth_bridge_gap": (accounting / "household_accounting.csv", "household_wealth_bridge_gap"),
        "public_cash_flow_gap": (accounting / "public_accounting.csv", "public_cash_flow_gap"),
        "lifecycle_cash_gap": (accounting / "household_lifecycle.csv", "lifecycle_cash_gap"),
    }
    gaps = {}
    for name, (path, field) in gap_specs.items():
        gaps[name] = maximum(read_csv(path), field) if path.exists() else None

    by_step = {}
    for row in firms:
        by_step.setdefault(int(float(row["global_step"])), []).append(row)
    planning = []
    weight_gaps = []
    realized = []
    heterogeneity = {field: [] for field in ("price", "unit_market_share", "sales_units", "actual_production", "inventory_units", "profit", "loan_balance")}
    diagnostic_by_step = {int(float(row["global_step"])): row for row in diagnostics}
    for step, step_rows in by_step.items():
        macro = diagnostic_by_step.get(step, {})
        planning.append(float(macro.get("household_planning_price_index", 0.0)) - sum(float(row.get("planning_price_component", 0.0)) for row in step_rows))
        weight_gaps.append(sum(float(row.get("planning_weight", 0.0)) for row in step_rows) - 1.0)
        units = float(macro.get("food_sales_units", 0.0))
        if units > 0:
            realized.append(float(macro.get("realized_transaction_price_index", 0.0)) - float(macro.get("market_sales_revenue", 0.0)) / units)
        for field in heterogeneity:
            heterogeneity[field].append(cv([float(row.get(field, 0.0)) for row in step_rows]))

    events = read_csv(run_dir / "demographic_events.csv")
    markets = read_csv(run_dir / "marriage_market_diagnostics.csv")
    birth_interval_violations = sum(
        1 for row in events
        if row.get("event_type") == "birth"
        and row.get("inter_birth_interval_weeks") not in ("", None)
        and float(row["inter_birth_interval_weeks"]) < 52
    )
    market_steps = [int(float(row["global_step"])) for row in markets]
    marriage_cadence_ok = all(b - a == 52 for a, b in zip(market_steps, market_steps[1:]))
    finite_firms = all(
        math.isfinite(float(row.get(field, 0.0)))
        for row in firms
        for field in ("price", "actual_production", "productive_capacity", "inventory_units", "cash", "loan_balance")
    )
    nonnegative_physical = all(float(row.get("inventory_units", 0.0)) >= -1e-12 for row in firms)

    return {
        "final": diagnostics[-1] if diagnostics else {},
        "invariant_violations": sum(truthy(row.get("invariant_failed", False)) for row in diagnostics),
        "maximum_reconciliation_gaps": gaps,
        "planning_price_max_residual": max((abs(item) for item in planning), default=0.0),
        "planning_weight_sum_max_residual": max((abs(item) for item in weight_gaps), default=0.0),
        "realized_price_max_residual": max((abs(item) for item in realized), default=0.0),
        "birth_spacing_violations_lt_52_weeks": birth_interval_violations,
        "marriage_market_steps": market_steps,
        "marriage_52_week_cadence": marriage_cadence_ok,
        "all_firm_values_finite": finite_firms,
        "all_physical_inventory_nonnegative": nonnegative_physical,
        "bootstrap_execution_counts": sorted({int(float(row.get("multi_firm_bootstrap_execution_count", 0))) for row in diagnostics}),
        "heterogeneity": {
            field: {"mean_cv": sum(values) / len(values) if values else 0.0, "max_cv": max(values) if values else 0.0}
            for field, values in heterogeneity.items()
        },
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--compare", nargs=2, type=Path)
    parser.add_argument("--baseline", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    result = {}
    if args.compare:
        result["behavioral_regression"] = compare_runs(*args.compare)
    if args.baseline:
        result["baseline_validation"] = validate_baseline(args.baseline)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
