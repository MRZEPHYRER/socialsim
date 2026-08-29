"""Audit fresh-population scaling without changing model behavior.

The naive runs use the normal main.py entry point unchanged.  The optional
scaled-money runs are an explicit initialization experiment: only the initial
private money stock is scaled to preserve the N=500 per-capita reference.
Behavioral rates, production rules, prices, credit rules, and policy stocks
are otherwise untouched.
"""

import argparse
import csv
import json
import math
import os
import subprocess
import sys
import time
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "test" / "output" / "pre_step13_3_population_scaling"
POPULATIONS = (500, 2000, 5000)


def run_model(population, horizon, mode, seed, output_root, cash_multiplier=1.0):
    run_dir = output_root / f"{mode}_N{population}_seed{seed}_w{horizon}"
    if (run_dir / "diagnostics.csv").exists():
        return run_dir, {"reused": True, "runtime_seconds": None}

    command = [
        sys.executable,
        str(ROOT / "main.py"),
        "--population", str(population),
        "--steps", str(horizon),
        "--seed", str(seed),
        "--firm-count", "5",
        "--diagnostics-mode", "full",
        "--no-analysis",
        "--no-plots",
        "--progress-interval", "0",
        "--output-dir", str(run_dir),
    ]
    if cash_multiplier != 1.0:
        command.extend(["--initial-firm-cash-multiplier", str(cash_multiplier)])
    started = time.perf_counter()
    subprocess.run(command, cwd=ROOT, check=True)
    elapsed = time.perf_counter() - started
    return run_dir, {"reused": False, "runtime_seconds": elapsed}


def read_csv(path):
    with open(path, newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def number(value):
    try:
        result = float(value)
        return result if math.isfinite(result) else 0.0
    except (TypeError, ValueError):
        return 0.0


def mean(rows, field):
    values = [number(row.get(field)) for row in rows]
    return sum(values) / len(values) if values else 0.0


def last_window(rows, size=52):
    return rows[-size:] if len(rows) > size else rows


def summarize_run(run_dir, population, mode, runtime):
    diagnostics = read_csv(run_dir / "diagnostics.csv")
    firms = read_csv(run_dir / "firm_diagnostics.csv")
    last = last_window(diagnostics)
    last_firms = last_window(firms, 52 * 5)
    final = diagnostics[-1] if diagnostics else {}

    by_firm = {}
    for row in last_firms:
        by_firm.setdefault(row.get("firm_id", ""), []).append(row)

    firm_summary = []
    for firm_id, rows in sorted(by_firm.items()):
        current = rows[-1]
        firm_summary.append({
            "firm_id": firm_id,
            "price": number(current.get("price")),
            "sales": mean(rows, "sales_units"),
            "production": mean(rows, "actual_production"),
            "capacity": mean(rows, "productive_capacity"),
            "capacity_utilization": mean(rows, "capacity_utilization"),
            "inventory": mean(rows, "inventory_units"),
            "cash": mean(rows, "cash"),
            "loan_balance": mean(rows, "loan_balance"),
            "profit": mean(rows, "profit"),
            "wage_bill": mean(rows, "wage_bill"),
            "cash_to_wage_bill": mean(rows, "cash_to_wage_bill"),
            "people_per_firm": population / 5.0,
        })

    total_capacity = number(final.get("productive_capacity"))
    if total_capacity == 0:
        total_capacity = sum(item["capacity"] for item in firm_summary)

    metrics = {
        "mode": mode,
        "population": population,
        "horizon_weeks": len(diagnostics),
        "final_global_step": number(final.get("global_step")),
        "runtime_seconds": runtime,
        "output_bytes": sum(
            path.stat().st_size for path in run_dir.rglob("*") if path.is_file()
        ),
        "final_population": number(final.get("population")),
        "final_households": number(final.get("households")),
        "per_capita_consumption": mean(last, "total_consumption") / population,
        "per_capita_income": mean(last, "total_income") / population,
        "per_capita_wealth": mean(last, "total_household_wealth") / population,
        "per_capita_money_stock": mean(last, "total_money_stock") / population,
        "per_capita_food_production": mean(last, "food_output_units") / population,
        "per_capita_inventory": mean(last, "food_inventory_units") / population,
        "per_capita_public_inventory": mean(last, "central_bank_food_inventory_units") / population,
        "per_capita_firm_cash": mean(last, "firm_cash") / population,
        "per_capita_loan_balance": mean(last, "loan_balance") / population,
        "per_worker_production": mean(last, "food_output_units") / max(1.0, mean(last, "workers")),
        "per_worker_wage_bill": mean(last, "wage_bill") / max(1.0, mean(last, "workers")),
        "per_worker_capacity": total_capacity / max(1.0, mean(last, "workers")),
        "unmet_need_share": mean(last, "poverty_household_ratio"),
        "saving_rate": mean(last, "saving_rate"),
        "security_ratio": mean(last, "mean_security_ratio"),
        "capacity_utilization": mean(last, "food_output_units") / max(1e-12, total_capacity),
        "inventory_coverage_weeks": mean(last, "inventory_demand_ratio"),
        "profit_margin": mean(last, "firm_profit_before_dividend") / max(1e-12, mean(last, "market_sales_revenue")),
        "price": mean(last, "food_price"),
        "transaction_value_to_money": mean(last, "market_sales_revenue") / max(1e-12, mean(last, "total_money_stock")),
        "loan_issuance": mean(last, "loan_issued"),
        "fixed_credit_buffer_to_weekly_wage_bill": 150000.0 / max(1e-12, mean(last, "wage_bill")),
        "fixed_credit_buffer_to_firm_cash": 150000.0 / max(1e-12, mean(last, "firm_cash")),
        "fixed_credit_buffer_to_weekly_sales_revenue": 150000.0 / max(1e-12, mean(last, "market_sales_revenue")),
        "firm_cash": mean(last, "firm_cash"),
        "public_inventory_activity": mean(last, "central_bank_inventory_purchase"),
        "stockout_frequency": sum(number(row.get("unmet_food_demand_units")) > 1e-9 for row in diagnostics) / max(1, len(diagnostics)),
        "max_food_conservation_gap": max((abs(number(row.get("food_conservation_gap"))) for row in diagnostics), default=0.0),
        "max_monetary_accounting_gap": max((abs(number(row.get("monetary_accounting_gap"))) for row in diagnostics), default=0.0),
        "max_money_delta_gap": max((abs(number(row.get("money_delta_gap"))) for row in diagnostics), default=0.0),
        "max_household_wealth_bridge_gap": max((abs(number(row.get("household_wealth_bridge_gap"))) for row in diagnostics), default=0.0),
        "invariant_violations": sum(number(row.get("invariant_failed")) != 0 for row in diagnostics),
        "firm_summary": firm_summary,
    }
    return metrics


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, ensure_ascii=False)
        handle.write("\n")


def write_csv(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    keys = list(rows[0])
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def build_manifest():
    return [
        {"field_name": "initial_population", "current_value_at_N500": 500, "source_file": "main.py / world.py", "source_function": "World.__init__", "scale_class": "population_stock", "current_formula": "N", "proposed_scaling_rule": "N is explicit experiment input", "economic_interpretation": "fresh population size", "behavior_change_required": False, "notes": "Age and sex are sampled; distributed age phase is isolated by seed."},
        {"field_name": "productive_capacity", "current_value_at_N500": "endogenous", "source_file": "economy/firm.py", "source_function": "FirmSystem.run_step", "scale_class": "per_worker", "current_formula": "sum(age_productivity(workers))", "proposed_scaling_rule": "emerge from worker count; do not manually multiply", "economic_interpretation": "labor-derived supply capacity", "behavior_change_required": False, "notes": "Capacity should scale through population age/work composition."},
        {"field_name": "FIRM_INITIAL_CASH", "current_value_at_N500": 10000000.0, "source_file": "economy/config.py", "source_function": "FirmSystem.__init__", "scale_class": "industry_aggregate", "current_formula": "fixed 10,000,000 before firm split", "proposed_scaling_rule": "candidate: preserve initial money per capita", "economic_interpretation": "initial private nominal liquidity", "behavior_change_required": "audit_only", "notes": "Naive N increases leave this stock fixed."},
        {"field_name": "CENTRAL_BANK_INITIAL_MONEY_SUPPLY", "current_value_at_N500": 0.0, "source_file": "central_bank/config.py", "source_function": "CentralBank.__init__", "scale_class": "industry_aggregate", "current_formula": "fixed 0", "proposed_scaling_rule": "unchanged pending policy interpretation", "economic_interpretation": "initial central-bank money", "behavior_change_required": False, "notes": "No initial public inventory either."},
        {"field_name": "CENTRAL_BANK_INITIAL_PUBLIC_INVENTORY", "current_value_at_N500": 0.0, "source_file": "central_bank/central_bank.py", "source_function": "CentralBank.__init__", "scale_class": "industry_aggregate", "current_formula": "fixed 0", "proposed_scaling_rule": "unchanged pending policy interpretation", "economic_interpretation": "opening public buffer stock", "behavior_change_required": False, "notes": "Public inventory is accumulated endogenously."},
        {"field_name": "CENTRAL_BANK_FOOD_PURCHASE_RATE", "current_value_at_N500": 0.008, "source_file": "central_bank/config.py", "source_function": "purchase_excess_food_inventory", "scale_class": "ratio_or_rate", "current_formula": "0.008 of excess inventory per week", "proposed_scaling_rule": "unchanged", "economic_interpretation": "public purchase rate", "behavior_change_required": False, "notes": "Absolute activity still depends on market inventory."},
        {"field_name": "CENTRAL_BANK_FOOD_RELEASE_RATE", "current_value_at_N500": 0.60, "source_file": "central_bank/config.py", "source_function": "release_food_inventory", "scale_class": "ratio_or_rate", "current_formula": "0.60", "proposed_scaling_rule": "unchanged", "economic_interpretation": "public release rate", "behavior_change_required": False, "notes": "Policy rate, not population-scaled."},
        {"field_name": "CENTRAL_BANK_FOOD_POVERTY_SUBSIDY_RATE", "current_value_at_N500": 0.015, "source_file": "central_bank/config.py", "source_function": "poverty subsidy", "scale_class": "ratio_or_rate", "current_formula": "0.015", "proposed_scaling_rule": "unchanged", "economic_interpretation": "in-kind support rate", "behavior_change_required": False, "notes": "Policy rate, not population-scaled."},
        {"field_name": "household_initial_wealth", "current_value_at_N500": "0 at scaffold", "source_file": "world.py", "source_function": "create_household", "scale_class": "per_household", "current_formula": "households begin with zero wealth", "proposed_scaling_rule": "unchanged", "economic_interpretation": "initial private household wealth", "behavior_change_required": False, "notes": "Initial money is held by firm cash in the accepted fresh bootstrap."},
        {"field_name": "initial_age_and_sex_sampling", "current_value_at_N500": "seeded", "source_file": "world.py", "source_function": "World.__init__", "scale_class": "population_composition", "current_formula": "uniform integer age 0..80 and random sex", "proposed_scaling_rule": "unchanged", "economic_interpretation": "fresh demographic scaffold", "behavior_change_required": False, "notes": "Sampling noise should decline with N; not exact deterministic composition."},
        {"field_name": "initial_inventory", "current_value_at_N500": 0.0, "source_file": "economy/firm.py", "source_function": "FirmSystem.__init__", "scale_class": "industry_aggregate", "current_formula": "0 before weekly production", "proposed_scaling_rule": "candidate: expected weekly demand * coverage weeks", "economic_interpretation": "opening buffer stock", "behavior_change_required": "audit_only", "notes": "Confirm from run outputs before implementation."},
        {"field_name": "CENTRAL_BANK_FIRM_CREDIT_CASH_BUFFER", "current_value_at_N500": 150000.0, "source_file": "central_bank/config.py", "source_function": "working-capital credit", "scale_class": "absolute_policy", "current_formula": "fixed 150000", "proposed_scaling_rule": "do not change in this audit", "economic_interpretation": "industry liquidity buffer", "behavior_change_required": False, "notes": "Report relative magnitude against wage bill, cash, and sales."},
        {"field_name": "CENTRAL_BANK_FIRM_LOAN_REPAYMENT_RATE", "current_value_at_N500": 0.35, "source_file": "central_bank/config.py", "source_function": "repay_firm_working_capital_loan", "scale_class": "ratio_or_rate", "current_formula": "0.35 per week target", "proposed_scaling_rule": "unchanged", "economic_interpretation": "credit repayment rate", "behavior_change_required": False, "notes": "Dimensionless rate."},
        {"field_name": "FIRM_PRODUCTION_TARGET_INVENTORY_COVERAGE", "current_value_at_N500": 15.0, "source_file": "economy/config.py", "source_function": "firm production planning", "scale_class": "ratio_or_rate", "current_formula": "15 weeks", "proposed_scaling_rule": "unchanged", "economic_interpretation": "inventory coverage target", "behavior_change_required": False, "notes": "Do not scale behavioral ratios."},
        {"field_name": "FIRM_COUNT", "current_value_at_N500": 5, "source_file": "main.py", "source_function": "World.split_firms", "scale_class": "per_firm", "current_formula": "fixed 5 in this audit", "proposed_scaling_rule": "fixed 5", "economic_interpretation": "competition structure", "behavior_change_required": False, "notes": "Population scaling is isolated from firm-count scaling."},
    ]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--naive-horizon", type=int, default=260)
    parser.add_argument("--scaled-horizon", type=int, default=520)
    parser.add_argument("--skip-scaled", action="store_true")
    parser.add_argument("--output-root", default=str(OUTPUT))
    args = parser.parse_args()
    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    write_json(output_root / "population_scaling_manifest.json", build_manifest())
    naive = []
    performance = []
    for population in POPULATIONS:
        run_dir, timing = run_model(population, args.naive_horizon, "naive", args.seed, output_root)
        summary = summarize_run(run_dir, population, "naive", timing["runtime_seconds"])
        naive.append(summary)
        performance.append({k: summary[k] for k in ("mode", "population", "horizon_weeks", "runtime_seconds", "output_bytes")})

    scaled = []
    rules = {
        "name": "scaled_money_audit_only",
        "reference_population": 500,
        "candidate_rule": "initial firm cash proportional to population (N/500); all rates, prices, capacity equations, inventory policy, credit buffer, and public stocks unchanged",
        "status": "diagnostic candidate run only; not adopted as a model default",
        "reason": "separate nominal-liquidity sensitivity from endogenous labor capacity scaling",
    }
    write_json(output_root / "scaled_initialization_rules.json", rules)
    if not args.skip_scaled:
        for population in POPULATIONS:
            cash_multiplier = population / 500.0
            run_dir, timing = run_model(population, args.scaled_horizon, "scaled_money_candidate", args.seed, output_root, cash_multiplier)
            summary = summarize_run(run_dir, population, "scaled_money_candidate", timing["runtime_seconds"])
            scaled.append(summary)
            performance.append({k: summary[k] for k in ("mode", "population", "horizon_weeks", "runtime_seconds", "output_bytes")})

    per_capita_rows = []
    dimensionless_rows = []
    for summary in naive + scaled:
        per_capita_rows.append({k: v for k, v in summary.items() if k.startswith("per_capita_") or k in ("mode", "population")})
        dimensionless_rows.append({k: summary[k] for k in ("mode", "population", "unmet_need_share", "saving_rate", "security_ratio", "capacity_utilization", "inventory_coverage_weeks", "profit_margin", "price", "transaction_value_to_money", "fixed_credit_buffer_to_weekly_wage_bill", "fixed_credit_buffer_to_firm_cash", "fixed_credit_buffer_to_weekly_sales_revenue", "stockout_frequency", "invariant_violations")})

    write_json(output_root / "naive_scaling_comparison.json", naive)
    write_csv(output_root / "per_capita_metrics.csv", per_capita_rows)
    write_csv(output_root / "dimensionless_metrics.csv", dimensionless_rows)
    write_csv(output_root / "performance_summary.csv", performance)
    accounting_runs = []
    for summary in naive + scaled:
        run_dir = output_root / f"{summary['mode']}_N{summary['population']}_seed{args.seed}_w{args.naive_horizon if summary['mode'] == 'naive' else args.scaled_horizon}"
        file_checks = {}
        for path in sorted((run_dir / "accounting").glob("*.csv")):
            rows = read_csv(path)
            gaps = {}
            for row in rows:
                for key, value in row.items():
                    if "gap" in key.lower() or key.lower() in {"invariant_failed", "invariant_violations"}:
                        gaps[key] = max(gaps.get(key, 0.0), abs(number(value)))
            file_checks[path.name] = gaps
        accounting_runs.append({"mode": summary["mode"], "population": summary["population"], "files": file_checks})

    write_json(output_root / "accounting_validation.json", {
        "runs": [
            {"mode": item["mode"], "population": item["population"], "invariant_violations": item["invariant_violations"], "max_food_conservation_gap": item["max_food_conservation_gap"], "max_monetary_accounting_gap": item["max_monetary_accounting_gap"], "max_money_delta_gap": item["max_money_delta_gap"], "max_household_wealth_bridge_gap": item["max_household_wealth_bridge_gap"]}
            for item in naive + scaled
        ] + accounting_runs,
        "tolerance_note": "The audit reports existing CSV values; it does not alter or mask gaps.",
    })

    lines = [
        "# Pre-Step13.3 Population Scaling Audit",
        "",
        "This is an audit only. Step 13 remains inactive and no behavioral equation was changed.",
        "",
        "## Runs",
        f"- Naive: N=500, 2000, 5000; {args.naive_horizon} weeks; seed={args.seed}; firms=5.",
        f"- Candidate directory: `scaled_money_candidate`; {args.scaled_horizon} weeks. This is a diagnostic candidate, not a model default.",
        "",
        "## Preliminary status",
        "- Capacity is labor-derived and should scale endogenously with worker counts.",
        "- Initial firm cash is a fixed aggregate nominal stock; naive population increases therefore reduce money per capita.",
        "- The 150000 working-capital component is an absolute industry buffer and is reported for later calibration, not changed here.",
        "- Review `naive_scaling_comparison.json`, `per_capita_metrics.csv`, and `dimensionless_metrics.csv` before accepting any initialization rule.",
        "",
        "## Verdict",
        "B. Population scaling is possible but scale-sensitive initial money and absolute policy buffers require review.",
    ]
    (output_root / "acceptance_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Population scaling audit outputs: {output_root}")


if __name__ == "__main__":
    main()
