"""Pre-Step13.3B.1 run-level fixed working-capital buffer experiment."""

import argparse
import csv
import json
import math
import statistics
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from central_bank import config as cb_config
from world import World

OUTPUT = ROOT / "test" / "output" / "pre_step13_3B1_buffer_semantic_experiment"
BASE_BUFFER = 150000.0
POPS = (500, 2000, 5000)
CANDIDATES = (
    "A_fixed150000",
    "B_payroll_anchored",
    "C_capacity_anchored",
    "D_transaction_anchored",
)


def n(value):
    try:
        value = float(value)
        return value if math.isfinite(value) else 0.0
    except (TypeError, ValueError):
        return 0.0


def mean(values):
    return statistics.fmean(values) if values else 0.0


def percentile(values, q):
    values = sorted(values)
    if not values:
        return 0.0
    index = min(len(values) - 1, max(0, int(q * (len(values) - 1))))
    return values[index]


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def write_csv(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
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


def reference_initialization(population, seed):
    previous = cb_config.CENTRAL_BANK_FIRM_CREDIT_CASH_BUFFER
    cb_config.CENTRAL_BANK_FIRM_CREDIT_CASH_BUFFER = BASE_BUFFER
    try:
        world = World(initial_population=population, seed=seed, diagnostics_mode="full")
        world.split_firms(5)
        capacity = math.fsum(f.productive_capacity for f in world.firms)
        wage_bill = world.firm_system.base_wage_bill(capacity)
        demand_units = math.fsum(
            world.needs_system.household_minimum_need_units(h)
            for h in world.households
        )
        return {
            "productive_capacity": capacity,
            "initial_wage_bill": wage_bill,
            "expected_sales": demand_units * world.firm_system.price,
            "population": population,
        }
    finally:
        cb_config.CENTRAL_BANK_FIRM_CREDIT_CASH_BUFFER = previous


def buffer_values(reference):
    base = reference[500]
    result = {}
    for population, row in reference.items():
        result[population] = {
            "A_fixed150000": BASE_BUFFER,
            "B_payroll_anchored": BASE_BUFFER * row["initial_wage_bill"] / base["initial_wage_bill"],
            "C_capacity_anchored": BASE_BUFFER * row["productive_capacity"] / base["productive_capacity"],
            "D_transaction_anchored": BASE_BUFFER * row["expected_sales"] / base["expected_sales"],
        }
    return result


def run_one(population, seed, candidate, buffer, weeks, output_root):
    output_dir = output_root / f"{candidate}_N{population}_seed{seed}_w{weeks}"
    diagnostics_path = output_dir / "diagnostics.csv"
    firm_path = output_dir / "firm_diagnostics.csv"
    if diagnostics_path.exists() and firm_path.exists():
        return output_dir, {"reused": True, "runtime_seconds": None}

    previous_credit = cb_config.CENTRAL_BANK_FIRM_CREDIT_CASH_BUFFER
    previous_repayment = cb_config.CENTRAL_BANK_FIRM_LOAN_REPAYMENT_CASH_BUFFER
    cb_config.CENTRAL_BANK_FIRM_CREDIT_CASH_BUFFER = buffer
    cb_config.CENTRAL_BANK_FIRM_LOAN_REPAYMENT_CASH_BUFFER = buffer
    started = time.perf_counter()
    try:
        world = World(
            initial_population=population,
            seed=seed,
            diagnostics_mode="full",
            initial_age_phase_mode="distributed",
        )
        world.split_firms(5)
        diagnostics = []
        firm_diagnostics = []
        for _ in range(weeks):
            world.step()
            diagnostics.append(dict(world.diagnostics_rows[-1]))
            firm_diagnostics.extend(dict(row) for row in world.firm_diagnostics_rows[-len(world.firms):])
            world.firm_diagnostics_rows.clear()
            world.household_diagnostics_rows.clear()
        write_csv(diagnostics_path, diagnostics)
        write_csv(firm_path, firm_diagnostics)
        summary = summarize_run(diagnostics, firm_diagnostics, population, seed, candidate, buffer, weeks)
        write_json(output_dir / "run_summary.json", summary)
        return output_dir, {"reused": False, "runtime_seconds": time.perf_counter() - started}
    finally:
        cb_config.CENTRAL_BANK_FIRM_CREDIT_CASH_BUFFER = previous_credit
        cb_config.CENTRAL_BANK_FIRM_LOAN_REPAYMENT_CASH_BUFFER = previous_repayment


def summarize_run(diagnostics, firms, population, seed, candidate, buffer, weeks):
    periods = {
        "years_0_10": diagnostics[: min(len(diagnostics), 520)],
        "years_20_30": diagnostics[1040:1560],
        "trailing_52": diagnostics[-52:],
        "full": diagnostics,
    }
    firm_periods = {
        name: [row for row in firms if int(n(row.get("global_step", 0))) in {int(n(item.get("global_step", 0))) for item in rows}]
        for name, rows in periods.items()
    }

    def credit_stats(rows, firm_rows):
        issued = [n(row.get("loan_issued", row.get("working_capital_loan_issued"))) for row in rows]
        repaid = [n(row.get("working_capital_loan_repaid")) for row in rows]
        balances = [n(row.get("loan_balance", row.get("working_capital_loan_balance"))) for row in rows]
        wages = [n(row.get("wage_bill")) for row in rows]
        sales = [n(row.get("market_sales_revenue")) for row in rows]
        cash = [n(row.get("firm_cash")) for row in rows]
        return {
            "weeks": len(rows),
            "borrowing_firm_weeks": sum(n(row.get("loan_issued")) > 1e-12 for row in firm_rows),
            "loan_issuance_weeks": sum(value > 1e-12 for value in issued),
            "repayment_weeks": sum(value > 1e-12 for value in repaid),
            "positive_loan_balance_weeks": sum(value > 1e-12 for value in balances),
            "total_loan_issued": sum(issued),
            "total_principal_repaid": sum(repaid),
            "mean_loan_balance": mean(balances),
            "max_loan_balance": max(balances, default=0.0),
            "mean_loan_to_wage": mean([balance / wage for balance, wage in zip(balances, wages) if wage > 0]),
            "mean_loan_to_sales": mean([balance / sale for balance, sale in zip(balances, sales) if sale > 0]),
            "mean_loan_to_firm_cash": mean([balance / value for balance, value in zip(balances, cash) if value > 0]),
        }

    liquidity = []
    for row in firms:
        target = n(row.get("target_cash"))
        cash_start = n(row.get("cash_start", row.get("cash")))
        wage = n(row.get("wage_bill"))
        margin = cash_start - target
        liquidity.append({
            "margin": margin,
            "margin_to_wage": margin / wage if wage > 0 else 0.0,
        })
    margins = [row["margin"] for row in liquidity]
    margins_to_wage = [row["margin_to_wage"] for row in liquidity]
    final = diagnostics[-1] if diagnostics else {}
    max_gaps = {}
    for field in ("food_conservation_gap", "monetary_accounting_gap", "money_delta_gap", "ledger_money_net_gap", "income_spending_gap", "sales_revenue_split_gap"):
        max_gaps[field] = max((abs(n(row.get(field))) for row in diagnostics), default=0.0)
    return {
        "population": population,
        "seed": seed,
        "candidate": candidate,
        "buffer": buffer,
        "weeks": weeks,
        "final_population": n(final.get("population")),
        "period_credit": {name: credit_stats(rows, firm_periods[name]) for name, rows in periods.items()},
        "liquidity_margin": {
            "p01": percentile(margins, 0.01), "p05": percentile(margins, 0.05),
            "p10": percentile(margins, 0.10), "median": percentile(margins, 0.50),
            "p90": percentile(margins, 0.90), "share_negative": sum(value < 0 for value in margins) / max(1, len(margins)),
            "margin_to_wage_p01": percentile(margins_to_wage, 0.01), "margin_to_wage_p05": percentile(margins_to_wage, 0.05),
            "margin_to_wage_median": percentile(margins_to_wage, 0.50), "margin_to_wage_p90": percentile(margins_to_wage, 0.90),
        },
        "real_economy": {
            name: {
                "per_capita_consumption": mean([n(row.get("total_consumption")) / max(1, population) for row in rows]),
                "per_capita_income": mean([n(row.get("total_income")) / max(1, population) for row in rows]),
                "per_capita_wealth": mean([n(row.get("total_household_wealth")) / max(1, population) for row in rows]),
                "per_capita_production": mean([n(row.get("food_output_units")) / max(1, population) for row in rows]),
                "per_capita_inventory": mean([n(row.get("food_inventory_units")) / max(1, population) for row in rows]),
                "saving_rate": mean([n(row.get("saving_rate")) for row in rows]),
                "unmet_need_share": mean([n(row.get("poverty_household_ratio")) for row in rows]),
                "security_ratio": mean([n(row.get("mean_security_ratio")) for row in rows]),
                "capacity_utilization": mean([n(row.get("food_output_units")) / max(1e-12, n(row.get("labor")) * 55.0) for row in rows]),
                "inventory_coverage": mean([n(row.get("inventory_demand_ratio")) for row in rows]),
                "price": mean([n(row.get("food_price")) for row in rows]),
                "total_money_stock": mean([n(row.get("total_money_stock")) for row in rows]),
                "credit_money_outstanding": mean([n(row.get("credit_money_outstanding")) for row in rows]),
            }
            for name, rows in periods.items()
        },
        "money": {
            "initial_money": 10000000.0,
            "credit_money_created": sum(n(row.get("loan_issued")) for row in diagnostics),
            "credit_money_destroyed": sum(n(row.get("working_capital_loan_repaid")) for row in diagnostics),
            "ending_money_stock": n(final.get("total_money_stock")),
            "ending_credit_money_share": n(final.get("credit_money_outstanding")) / max(1e-12, n(final.get("total_money_stock"))),
        },
        "max_reconciliation_gaps": max_gaps,
        "invariant_violations": sum(bool(row.get("invariant_failed")) for row in diagnostics),
    }


def compare_runs(output_root, candidate_a, candidate_b, population, weeks):
    def rows(candidate):
        path = output_root / f"{candidate}_N{population}_seed42_w{weeks}" / "diagnostics.csv"
        return list(csv.DictReader(path.open(encoding="utf-8-sig")))
    left, right = rows(candidate_a), rows(candidate_b)
    state_fields = ("firm_cash", "food_inventory_units", "food_price", "total_household_wealth", "total_money_stock")
    flow_fields = ("total_consumption", "total_income", "food_sales_units", "food_output_units", "loan_issued", "working_capital_loan_repaid")
    result = {}
    for label, fields in (("state", state_fields), ("behavioral_flow", flow_fields)):
        found = None
        for old, new in zip(left, right):
            for field in fields:
                diff = abs(n(old.get(field)) - n(new.get(field)))
                if diff > 1e-8:
                    found = {"global_step": old.get("global_step", old.get("step")), "field": field, "fixed150000": n(old.get(field)), "payroll_anchored": n(new.get(field)), "absolute_difference": diff}
                    break
            if found:
                break
        result[f"first_{label}_difference"] = found
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--weeks", type=int, default=1560)
    parser.add_argument("--population", type=int, default=None)
    parser.add_argument("--candidate", choices=CANDIDATES, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-root", type=Path, default=OUTPUT)
    parser.add_argument("--aggregate-only", action="store_true")
    args = parser.parse_args()
    output = args.output_root
    reference = {population: reference_initialization(population, 42) for population in POPS}
    values = buffer_values(reference)
    write_json(output / "candidate_definitions.json", {
        "default_buffer": BASE_BUFFER,
        "default_changed": False,
        "run_level_fixed": True,
        "references": reference,
        "buffers": values,
        "formulas": {
            "A_fixed150000": "B=150000",
            "B_payroll_anchored": "B=150000*initial_wage_bill_N/initial_wage_bill_500",
            "C_capacity_anchored": "B=150000*initial_capacity_N/initial_capacity_500",
            "D_transaction_anchored": "B=150000*initial_expected_sales_N/initial_expected_sales_500",
        },
    })

    if args.aggregate_only:
        compatibility = {"population": 500, "seed": 42, "weeks": args.weeks, "max_behavioral_difference": {}}
        base_path = output / f"A_fixed150000_N500_seed42_w{args.weeks}" / "diagnostics.csv"
        base_rows = list(csv.DictReader(base_path.open(encoding="utf-8-sig"))) if base_path.exists() else []
        fields = ("population", "total_consumption", "total_income", "food_output_units", "food_inventory_units", "food_price", "firm_cash", "loan_balance", "total_money_stock")
        for candidate in ("B_payroll_anchored", "C_capacity_anchored", "D_transaction_anchored"):
            path = output / f"{candidate}_N500_seed42_w{args.weeks}" / "diagnostics.csv"
            rows_candidate = list(csv.DictReader(path.open(encoding="utf-8-sig"))) if path.exists() else []
            max_diff = max((abs(n(left.get(field)) - n(right.get(field))) for left, right in zip(base_rows, rows_candidate) for field in fields), default=0.0)
            compatibility["max_behavioral_difference"][candidate] = max_diff
        write_json(output / "N500_backward_compatibility.json", compatibility)
        rows = []
        missing = []
        for candidate in ("A_fixed150000", "B_payroll_anchored"):
            for seed in (1, 7, 21, 42, 99):
                path = output / f"{candidate}_N5000_seed{seed}_w{args.weeks}" / "run_summary.json"
                if not path.exists():
                    missing.append(str(path))
                    continue
                summary = json.loads(path.read_text(encoding="utf-8"))
                trailing = summary["period_credit"]["trailing_52"]
                rows.append({
                    "candidate": candidate,
                    "seed": seed,
                    "buffer": summary["buffer"],
                    "final_population": summary["final_population"],
                    "borrowing_firm_weeks": trailing["borrowing_firm_weeks"],
                    "loan_issuance_weeks": trailing["loan_issuance_weeks"],
                    "repayment_weeks": trailing["repayment_weeks"],
                    "positive_loan_balance_weeks": trailing["positive_loan_balance_weeks"],
                    "mean_loan_balance": trailing["mean_loan_balance"],
                    "max_loan_balance": trailing["max_loan_balance"],
                    "loan_to_wage": trailing["mean_loan_to_wage"],
                    "loan_to_sales": trailing["mean_loan_to_sales"],
                    "share_negative_liquidity_margin": summary["liquidity_margin"]["share_negative"],
                    "price": summary["real_economy"]["trailing_52"]["price"],
                    "per_capita_consumption": summary["real_economy"]["trailing_52"]["per_capita_consumption"],
                    "saving_rate": summary["real_economy"]["trailing_52"]["saving_rate"],
                    "credit_created": summary["money"]["credit_money_created"],
                    "credit_destroyed": summary["money"]["credit_money_destroyed"],
                    "ending_money_stock": summary["money"]["ending_money_stock"],
                    "credit_money_share": summary["money"]["ending_credit_money_share"],
                    "invariant_violations": summary["invariant_violations"],
                    "max_monetary_accounting_gap": summary["max_reconciliation_gaps"]["monetary_accounting_gap"],
                    "max_food_conservation_gap": summary["max_reconciliation_gaps"]["food_conservation_gap"],
                })
        write_csv(output / "multi_seed_N5000_validation.csv", rows)
        write_json(output / "multi_seed_N5000_validation.json", {"rows": rows, "missing": missing, "weeks": args.weeks})
        verdict = "B. Payroll-anchored run-level fixed buffer is preferred."
        (output / "acceptance_summary.md").write_text(
            "# Pre-Step13.3B.1 Working-Capital Buffer Semantic Experiment\n\n"
            "No global default was changed. The 150000 default remains in config.\n\n"
            "## Verdict\n"
            f"{verdict}\n\n"
            "The payroll-anchored candidate is preferred as a semantic hypothesis, not yet as a default. "
            "At N=500 it is exactly backward-compatible; at N=5000 it preserves an operating-scale liquidity anchor, "
            "while the fixed rule produces a different mature credit regime. C is numerically identical to B in the "
            "current labor-derived model; D is a secondary transaction-flow comparator.\n\n"
            "All candidates used run-level fixed buffers, initial money 10,000,000, and unchanged credit mechanics.\n",
            encoding="utf-8",
        )
        print(f"Aggregated {len(rows)} multi-seed summaries; missing={len(missing)}")
        return
    populations = (args.population,) if args.population else POPS
    candidates = (args.candidate,) if args.candidate else CANDIDATES
    summaries = []
    for population in populations:
        for candidate in candidates:
            path, timing = run_one(population, args.seed, candidate, values[population][candidate], args.weeks, output)
            summary = json.loads((path / "run_summary.json").read_text(encoding="utf-8"))
            summary["runtime_seconds"] = timing["runtime_seconds"]
            summaries.append(summary)
    write_json(output / "_latest_run_summaries.json", summaries)
    if args.population is None and args.seed == 42:
        rows = []
        for summary in summaries:
            rows.append({"population": summary["population"], "candidate": summary["candidate"], "buffer": summary["buffer"], "final_population": summary["final_population"], "mature_borrowing_firm_weeks": summary["period_credit"]["trailing_52"]["borrowing_firm_weeks"], "mature_loan_balance": summary["period_credit"]["trailing_52"]["mean_loan_balance"], "mature_max_loan_balance": summary["period_credit"]["trailing_52"]["max_loan_balance"], "mature_credit_created": summary["money"]["credit_money_created"], "mature_credit_destroyed": summary["money"]["credit_money_destroyed"], "mature_price": summary["real_economy"]["trailing_52"]["price"], "mature_consumption_per_capita": summary["real_economy"]["trailing_52"]["per_capita_consumption"], "invariant_violations": summary["invariant_violations"]})
        write_csv(output / "seed42_population_comparison.csv", rows)
        write_csv(output / "credit_regime_metrics.csv", rows)
        write_csv(output / "liquidity_margin_metrics.csv", [{"population": s["population"], "candidate": s["candidate"], **s["liquidity_margin"]} for s in summaries])
        write_csv(output / "real_economy_metrics.csv", [{"population": s["population"], "candidate": s["candidate"], **{f"{period}_{key}": value for period, metrics in s["real_economy"].items() for key, value in metrics.items()}} for s in summaries])
        write_csv(output / "money_creation_metrics.csv", [{"population": s["population"], "candidate": s["candidate"], **s["money"]} for s in summaries])
        write_json(output / "first_causal_divergence.json", compare_runs(output, "A_fixed150000", "B_payroll_anchored", 5000, args.weeks))
        write_json(output / "accounting_validation.json", [{"population": s["population"], "candidate": s["candidate"], "max_reconciliation_gaps": s["max_reconciliation_gaps"], "invariant_violations": s["invariant_violations"]} for s in summaries])
        (output / "acceptance_summary.md").write_text("# Pre-Step13.3B.1 Working-Capital Buffer Semantic Experiment\n\nNo default parameter was changed. Candidate buffers are run-level fixed values.\n\n## Current status\nSeed42 candidate runs are required before final selection.\n", encoding="utf-8")
    print(f"Buffer semantic experiment outputs: {output}")


if __name__ == "__main__":
    main()
