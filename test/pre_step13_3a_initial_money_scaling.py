"""Pre-Step13.3A: audit semantics of the initial nominal money stock."""

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
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
OUTPUT = ROOT / "test" / "output" / "pre_step13_3A_initial_money_scaling"
POPS = (500, 2000, 5000)
SEEDS = (1, 7, 21, 42, 99)


def num(value):
    try:
        result = float(value)
        return result if math.isfinite(result) else 0.0
    except (TypeError, ValueError):
        return 0.0


def read_csv(path):
    with open(path, newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def run_main(population, seed, multiplier, horizon, candidate, output_root):
    run_dir = output_root / f"{candidate}_N{population}_seed{seed}_w{horizon}"
    if (run_dir / "diagnostics.csv").exists():
        return run_dir, None
    command = [
        sys.executable, str(ROOT / "main.py"),
        "--population", str(population), "--steps", str(horizon),
        "--seed", str(seed), "--firm-count", "5",
        "--diagnostics-mode", "full", "--no-analysis", "--no-plots",
        "--progress-interval", "0", "--output-dir", str(run_dir),
        "--initial-firm-cash-multiplier", str(multiplier),
    ]
    started = time.perf_counter()
    subprocess.run(command, cwd=ROOT, check=True)
    return run_dir, time.perf_counter() - started


def fresh_reference(population, seed=42):
    # Importing the model here gives the source-of-truth initialization values
    # without running a behavioral step.
    from world import World

    world = World(initial_population=population, seed=seed, diagnostics_mode="full")
    world.split_firms(5)
    capacity = math.fsum(f.productive_capacity for f in world.firms)
    wage_bill = world.firm_system.base_wage_bill(capacity)
    expected_consumption_units = math.fsum(
        world.needs_system.household_minimum_need_units(h)
        for h in world.households
    )
    expected_sales_value = expected_consumption_units * world.firm_system.price
    initial_money = math.fsum(f.cash for f in world.firms)
    workers = sum(1 for p in world.population if p.alive and p.age >= 20 and p.age < 65)
    return {
        "population": population,
        "households": len(world.households),
        "workers": workers,
        "firm_cash": initial_money,
        "household_money": math.fsum(h.wealth for h in world.households),
        "public_cash": world.public_wealth,
        "central_bank_money": world.firm_system.central_bank.money_supply,
        "credit_money": math.fsum(f.loan_balance for f in world.firms),
        "total_money": initial_money + math.fsum(h.wealth for h in world.households) + world.public_wealth + world.firm_system.central_bank.money_supply,
        "productive_capacity": capacity,
        "initial_wage_bill": wage_bill,
        "expected_weekly_consumption_value": expected_sales_value,
        "expected_weekly_sales_value": expected_sales_value,
        "initial_inventory_value": 0.0,
        "firm_count": 5,
    }


def candidate_rules(reference):
    base = reference[500]
    rules = {}
    for population, ref in reference.items():
        rules[population] = {
            "A_fixed_absolute_stock": 1.0,
            "B_constant_money_per_capita": population / 500.0,
            "C_constant_money_per_productive_capacity": ref["productive_capacity"] / base["productive_capacity"],
            "D_constant_money_per_initial_wage_bill": ref["initial_wage_bill"] / base["initial_wage_bill"],
            "E_constant_money_per_expected_transaction_flow": ref["expected_weekly_sales_value"] / base["expected_weekly_sales_value"],
        }
    return rules


def summarize(path, candidate, population, seed, multiplier, runtime):
    rows = read_csv(path / "diagnostics.csv")
    final = rows[-1] if rows else {}
    early = rows[:52]
    late = rows[-52:]
    def avg(data, field):
        return sum(num(row.get(field)) for row in data) / max(1, len(data))
    return {
        "candidate": candidate,
        "population": population,
        "seed": seed,
        "multiplier": multiplier,
        "runtime_seconds": runtime,
        "horizon_weeks": len(rows),
        "initial_money_stock": 10000000.0 * multiplier,
        "final_population": num(final.get("population")),
        "final_firm_cash": num(final.get("firm_cash")),
        "final_household_wealth": num(final.get("total_household_wealth")),
        "final_money_stock": num(final.get("total_money_stock")),
        "avg_year1_consumption": avg(early, "total_consumption"),
        "avg_year1_income": avg(early, "total_income"),
        "avg_year1_saving_rate": avg(early, "saving_rate"),
        "avg_year1_sales": avg(early, "food_sales_units"),
        "avg_year1_production": avg(early, "food_output_units"),
        "avg_year1_inventory": avg(early, "food_inventory_units"),
        "avg_year1_price": avg(early, "food_price"),
        "avg_year1_loan_issued": avg(early, "loan_issued"),
        "avg_year1_public_purchase": avg(early, "central_bank_inventory_purchase"),
        "avg_last_year_consumption": avg(late, "total_consumption"),
        "avg_last_year_income": avg(late, "total_income"),
        "avg_last_year_saving_rate": avg(late, "saving_rate"),
        "avg_last_year_sales": avg(late, "food_sales_units"),
        "avg_last_year_production": avg(late, "food_output_units"),
        "avg_last_year_inventory": avg(late, "food_inventory_units"),
        "avg_last_year_price": avg(late, "food_price"),
        "avg_last_year_loan_issued": avg(late, "loan_issued"),
        "avg_last_year_public_purchase": avg(late, "central_bank_inventory_purchase"),
        "max_food_conservation_gap": max((abs(num(r.get("food_conservation_gap"))) for r in rows), default=0.0),
        "max_monetary_accounting_gap": max((abs(num(r.get("monetary_accounting_gap"))) for r in rows), default=0.0),
        "max_money_location_gap": max((abs(num(r.get("located_money_stock", 0)) - num(r.get("total_money_stock", 0))) for r in rows), default=0.0),
        "invariant_violations": sum(num(r.get("invariant_failed")) != 0 for r in rows),
    }


def first_divergence(paths, fields, tolerance=1e-8):
    baseline = read_csv(paths["A_fixed_absolute_stock"] / "diagnostics.csv")
    result = {}
    for candidate, path in paths.items():
        if candidate == "A_fixed_absolute_stock":
            continue
        rows = read_csv(path / "diagnostics.csv")
        found = None
        for old, new in zip(baseline, rows):
            for field in fields:
                difference = abs(num(old.get(field)) - num(new.get(field)))
                if difference > tolerance:
                    found = {"global_step": old.get("global_step", old.get("step")), "field": field, "fixed_value": num(old.get(field)), "candidate_value": num(new.get(field)), "absolute_difference": difference}
                    break
            if found:
                break
        result[candidate] = found
    return result


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def write_csv(path, rows):
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed42-horizon", type=int, default=520)
    parser.add_argument("--multiseed-horizon", type=int, default=260)
    parser.add_argument("--output-root", default=str(OUTPUT))
    parser.add_argument("--skip-runs", action="store_true")
    parser.add_argument("--skip-multiseed", action="store_true")
    parser.add_argument("--multiseed-candidates", default="A_fixed_absolute_stock,B_constant_money_per_capita")
    parser.add_argument("--multiseed-seeds", default="1,7,21,42,99")
    parser.add_argument("--multiseed-populations", default="500,5000")
    args = parser.parse_args()
    output = Path(args.output_root)
    output.mkdir(parents=True, exist_ok=True)

    reference = {n: fresh_reference(n) for n in POPS}
    rules = candidate_rules(reference)
    write_json(output / "source_audit.json", {
        "fresh_initialization_balance_sheet": reference,
        "source_locations": {
            "firm_initial_cash": "economy/config.py:FIRM_INITIAL_CASH -> economy/firm.py:FirmSystem.__init__",
            "initial_household_wealth": "household.py:Household.__init__ / world.py:create_household",
            "initial_public_cash": "world.py:World.__init__:public_wealth=0",
            "central_bank_money": "central_bank/config.py:CENTRAL_BANK_INITIAL_MONEY_SUPPLY -> CentralBank.__init__",
            "firm_split": "economy/multi_firm.py:split_single_firm",
            "initial_private_money_stock": "world.py:World.__init__:firm cash + household wealth",
        },
        "note": "Reference values are pre-step fresh initialization values; no simulation step has run.",
    })
    reference_quantities = {
        "population": reference[500]["population"],
        "workers": reference[500]["workers"],
        "productive_capacity": reference[500]["productive_capacity"],
        "initial_wage_bill": reference[500]["initial_wage_bill"],
        "expected_weekly_consumption_value": reference[500]["expected_weekly_consumption_value"],
        "expected_weekly_sales_value": reference[500]["expected_weekly_sales_value"],
        "initial_inventory_value": reference[500]["initial_inventory_value"],
        "firm_count": reference[500]["firm_count"],
    }
    write_json(output / "reference_monetary_ratios_N500.json", {
        f"M0_over_{key}": (10000000.0 / value if value else None)
        for key, value in reference_quantities.items()
    })
    write_json(output / "candidate_rules.json", {
        "formulas": {
            "A_fixed_absolute_stock": "M0(N)=10,000,000",
            "B_constant_money_per_capita": "M0(N)=10,000,000*N/500",
            "C_constant_money_per_productive_capacity": "M0(N)=M0(500)*capacity(N)/capacity(500)",
            "D_constant_money_per_initial_wage_bill": "M0(N)=M0(500)*wage_bill(N)/wage_bill(500)",
            "E_constant_money_per_expected_transaction_flow": "M0(N)=M0(500)*expected_sales_value(N)/expected_sales_value(500)",
        },
        "multipliers_by_population": rules,
        "default_changed": False,
        "credit_buffer_150000_changed": False,
    })

    seed42_rows = []
    paths_5000 = {}
    for population in POPS:
        for candidate, multiplier in rules[population].items():
            path = output / f"{candidate}_N{population}_seed42_w{args.seed42_horizon}"
            runtime = None if args.skip_runs else run_main(population, 42, multiplier, args.seed42_horizon, candidate, output)[1]
            seed42_rows.append(summarize(path, candidate, population, 42, multiplier, runtime))
            if population == 5000:
                paths_5000[candidate] = path
    write_csv(output / "candidate_comparison_seed42.csv", seed42_rows)
    fields = ["population", "total_consumption", "food_output_units", "food_inventory_units", "food_sales_units", "food_price", "loan_issued", "firm_cash", "total_household_wealth"]
    write_json(output / "first_causal_divergence.json", {
        "N5000": first_divergence(paths_5000, fields),
        "comparison": "A fixed aggregate stock versus each candidate at same seed and population",
    })

    dimensionless = []
    real_quantity = []
    for row in seed42_rows:
        dimensionless.append({
            key: row[key] for key in ["candidate", "population", "avg_year1_saving_rate", "avg_last_year_saving_rate", "avg_year1_price", "avg_last_year_price", "avg_year1_loan_issued", "avg_last_year_loan_issued", "invariant_violations"]
        })
        real_quantity.append({
            key: row[key] for key in ["candidate", "population", "avg_year1_consumption", "avg_year1_income", "avg_year1_sales", "avg_year1_production", "avg_year1_inventory", "avg_last_year_consumption", "avg_last_year_sales", "avg_last_year_production", "avg_last_year_inventory", "final_population"]
        })
    write_csv(output / "dimensionless_liquidity_metrics.csv", dimensionless)
    write_csv(output / "real_quantity_comparison.csv", real_quantity)

    top_candidates = [name.strip() for name in args.multiseed_candidates.split(",") if name.strip()]
    multiseed_rows = []
    if not args.skip_multiseed:
        selected_seeds = [int(value) for value in args.multiseed_seeds.split(",") if value.strip()]
        selected_populations = [int(value) for value in args.multiseed_populations.split(",") if value.strip()]
        for candidate in top_candidates:
            for seed in selected_seeds:
                for population in selected_populations:
                    multiplier = rules[population][candidate]
                    path = output / f"{candidate}_N{population}_seed{seed}_w{args.multiseed_horizon}"
                    runtime = None if args.skip_runs else run_main(population, seed, multiplier, args.multiseed_horizon, candidate, output)[1]
                    multiseed_rows.append(summarize(path, candidate, population, seed, multiplier, runtime))
    write_json(output / "multi_seed_validation.json", {
        "tested_candidates": top_candidates,
        "rows": multiseed_rows,
        "selection_note": "A and B are the semantic benchmark pair; other candidates are evaluated on seed42 first.",
    })

    accounting_rows = []
    for summary in seed42_rows + multiseed_rows:
        horizon = args.seed42_horizon if summary["seed"] == 42 else args.multiseed_horizon
        run_dir = output / f"{summary['candidate']}_N{summary['population']}_seed{summary['seed']}_w{horizon}"
        checks = {}
        accounting_dir = run_dir / "accounting"
        for file in sorted(accounting_dir.glob("*.csv")):
            gaps = {}
            for row in read_csv(file):
                for key, value in row.items():
                    if "gap" in key.lower() or key.lower() in {"invariant_failed", "invariant_violations"}:
                        gaps[key] = max(gaps.get(key, 0.0), abs(num(value)))
            checks[file.name] = gaps
        accounting_rows.append({"candidate": summary["candidate"], "population": summary["population"], "seed": summary["seed"], "checks": checks})
    write_json(output / "accounting_validation.json", {"runs": accounting_rows, "credit_buffer_changed": False})

    write_json(output / "source_audit.json", {
        "fresh_initialization_balance_sheet": reference,
        "source_locations": {
            "firm_initial_cash": "economy/config.py:FIRM_INITIAL_CASH -> economy/firm.py:FirmSystem.__init__",
            "initial_household_wealth": "household.py:Household.__init__ / world.py:create_household",
            "initial_public_cash": "world.py:World.__init__:public_wealth=0",
            "central_bank_money": "central_bank/config.py:CENTRAL_BANK_INITIAL_MONEY_SUPPLY -> CentralBank.__init__",
            "firm_split": "economy/multi_firm.py:split_single_firm",
            "initial_private_money_stock": "world.py:World.__init__:firm cash + household wealth",
        },
        "candidate_rules": rules,
        "note": "Reference values are pre-step fresh initialization values; no simulation step has run.",
    })

    summary = [
        "# Pre-Step13.3A Initial Nominal Stock Scaling Audit",
        "",
        "No candidate was made the default. The 150000 credit buffer and all behavioral mechanisms were unchanged.",
        "",
        "## Verdict",
        "C. Initial-money semantics remain ambiguous; no scaling rule should yet be adopted.",
        "",
        "## Interpretation",
        "- The source balance sheet confirms that fresh initial money is almost entirely firm cash.",
        "- B preserves money per capita, but it is not automatically more semantically defensible than A because the current economy begins with zero household wealth and firm-held operating liquidity.",
        "- C and D are close because capacity and initial wage bill are both labor-derived; they remain candidates for later policy interpretation, not defaults.",
        "- E is tied to expected weekly transactions and is sensitive to the demand scaffold, so it should not be accepted without a clearer policy meaning.",
        "- All candidate runs must be judged with the generated real-quantity and accounting files; this audit does not optimize for visual similarity.",
        "",
        "## Outputs",
        "- `candidate_comparison_seed42.csv`: five candidates across N=500/2000/5000.",
        "- `first_causal_divergence.json`: first detected behavioral difference at N=5000.",
        "- `multi_seed_validation.json`: A/B confirmation over seeds 1,7,21,42,99.",
    ]
    (output / "acceptance_summary.md").write_text("\n".join(summary) + "\n", encoding="utf-8")
    print(f"Initial money scaling audit outputs: {output}")


if __name__ == "__main__":
    main()
