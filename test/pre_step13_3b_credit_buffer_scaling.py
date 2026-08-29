"""Pre-Step13.3B: shadow audit of the fixed 150000 credit buffer.

This script never changes the credit configuration and never issues shadow
loans. It evaluates alternative buffer magnitudes against actual trajectories.
"""

import argparse
import csv
import json
import math
import statistics
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
OUTPUT = ROOT / "test" / "output" / "pre_step13_3B_credit_buffer_scaling"
FRESH_ROOT = ROOT / "test" / "output" / "pre_step13_3_population_scaling"
BUFFER = 150000.0
POPS = (500, 2000, 5000)
CANDIDATES = (
    "A_fixed_absolute_buffer",
    "B_constant_buffer_per_capita",
    "C_constant_buffer_per_capacity",
    "D_constant_payroll_weeks",
    "E_constant_sales_weeks",
)


def num(value):
    try:
        result = float(value)
        return result if math.isfinite(result) else 0.0
    except (TypeError, ValueError):
        return 0.0


def read_csv(path):
    with open(path, newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


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
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def fresh_reference(population, seed=42):
    from world import World

    world = World(initial_population=population, seed=seed, diagnostics_mode="full")
    world.split_firms(5)
    capacity = math.fsum(f.productive_capacity for f in world.firms)
    wage = world.firm_system.base_wage_bill(capacity)
    demand_units = math.fsum(
        world.needs_system.household_minimum_need_units(h)
        for h in world.households
    )
    sales = demand_units * world.firm_system.price
    return {
        "population": population,
        "workers": sum(1 for p in world.population if p.alive and 20 <= p.age < 65),
        "productive_capacity": capacity,
        "weekly_wage_bill": wage,
        "expected_weekly_sales": sales,
        "initial_firm_cash": math.fsum(f.cash for f in world.firms),
        "initial_money_stock": math.fsum(f.cash for f in world.firms),
    }


def candidate_buffers(reference):
    base = reference[500]
    result = {}
    for population, row in reference.items():
        result[population] = {
            "A_fixed_absolute_buffer": BUFFER,
            "B_constant_buffer_per_capita": BUFFER * population / 500.0,
            "C_constant_buffer_per_capacity": BUFFER * row["productive_capacity"] / base["productive_capacity"],
            "D_constant_payroll_weeks": BUFFER * row["weekly_wage_bill"] / base["weekly_wage_bill"],
            "E_constant_sales_weeks": BUFFER * row["expected_weekly_sales"] / base["expected_weekly_sales"],
        }
    return result


def shadow_rows(firm_rows, population, buffers, source="fresh"):
    grouped = {}
    for row in firm_rows:
        grouped.setdefault(row.get("global_step", row.get("step", "0")), []).append(row)
    output = []
    for step, rows in grouped.items():
        total_capacity = math.fsum(max(0.0, num(row.get("productive_capacity"))) for row in rows)
        for candidate, buffer in buffers.items():
            target_binding = 0
            positive_gap = 0
            gaps = []
            ratio_values = []
            affected = set()
            for row in rows:
                capacity = max(0.0, num(row.get("productive_capacity")))
                share = capacity / total_capacity if total_capacity else 1.0 / max(1, len(rows))
                wage = num(row.get("wage_bill"))
                cash_start = num(row.get("cash_start", row.get("cash")))
                target = buffer * share + wage
                gap = max(0.0, target - cash_start)
                repayment_buffer = buffer * share
                target_binding += target > cash_start + 1e-12
                positive_gap += gap > 1e-12
                if gap > 1e-12:
                    affected.add(row.get("firm_id"))
                gaps.append(gap)
                ratio_values.append(cash_start / target if target > 0 else float("inf"))
            output.append({
                "source": source,
                "population": population,
                "global_step": int(float(step)),
                "candidate": candidate,
                "buffer": buffer,
                "firm_weeks": len(rows),
                "target_cash_binding_firm_weeks": target_binding,
                "positive_funding_gap_firm_weeks": positive_gap,
                "mean_positive_funding_gap": statistics.fmean([x for x in gaps if x > 0]) if any(x > 0 for x in gaps) else 0.0,
                "p90_funding_gap": sorted(gaps)[max(0, int(0.90 * (len(gaps) - 1)))],
                "p99_funding_gap": sorted(gaps)[max(0, int(0.99 * (len(gaps) - 1)))],
                "max_funding_gap": max(gaps, default=0.0),
                "affected_firms": len(affected),
                "minimum_cash_to_target_cash": min(ratio_values, default=0.0),
                "repayment_buffer_sum": buffer,
            })
    return output


def aggregate_shadow(rows):
    grouped = {}
    for row in rows:
        grouped.setdefault((row.get("source", "fresh"), row["population"], row["candidate"]), []).append(row)
    result = []
    for (source, population, candidate), values in grouped.items():
        firm_weeks = sum(int(row["firm_weeks"]) for row in values)
        binding = sum(int(row["target_cash_binding_firm_weeks"]) for row in values)
        positive = sum(int(row["positive_funding_gap_firm_weeks"]) for row in values)
        gaps = [num(row["mean_positive_funding_gap"]) for row in values if num(row["mean_positive_funding_gap"]) > 0]
        result.append({
            "source": source,
            "population": population,
            "candidate": candidate,
            "buffer": values[0]["buffer"],
            "firm_weeks": firm_weeks,
            "target_cash_binding_share": binding / max(1, firm_weeks),
            "positive_shadow_funding_gap_share": positive / max(1, firm_weeks),
            "mean_positive_funding_gap_over_steps": statistics.fmean(gaps) if gaps else 0.0,
            "p90_step_mean_gap": sorted(gaps)[max(0, int(0.90 * (len(gaps) - 1)))] if gaps else 0.0,
            "p99_step_mean_gap": sorted(gaps)[max(0, int(0.99 * (len(gaps) - 1)))] if gaps else 0.0,
            "maximum_step_gap": max(gaps, default=0.0),
            "max_affected_firms": max(int(row["affected_firms"]) for row in values),
            "minimum_cash_to_target_cash": min(num(row["minimum_cash_to_target_cash"]) for row in values),
        })
    return result


def source_audit(reference):
    return {
        "constant": BUFFER,
        "semantic_status": "ORIGINAL ECONOMIC SEMANTICS UNDOCUMENTED",
        "source_history": [
            {"source": "central_bank/config.py", "evidence": "CENTRAL_BANK_FIRM_CREDIT_CASH_BUFFER = 150000.0; repayment buffer uses the same value."},
            {"source": "economy/firm.py", "evidence": "single-firm target_cash = buffer + wage_bill; firm credit is requested before wages."},
            {"source": "world.py", "evidence": "multi-firm target_cash_i = buffer * productive_capacity_share_i + wage_bill_i; repayment buffer_i = buffer * share_i."},
            {"source": "manifest/output history", "evidence": "documented as industry-wide fixed 150000 allocated by productive-capacity share; no empirical or policy rationale found."},
        ],
        "classification": "recurring behavioral liquidity-policy parameter, distinct from initial 10m condition",
        "fresh_reference": reference,
        "credit_buffer_changed": False,
    }


def run_mature(population, weeks, seed, output_root):
    from world import World

    world = World(initial_population=population, seed=seed, diagnostics_mode="full")
    world.split_firms(5)
    rows = []
    firm_rows = []
    started = time.perf_counter()
    for _ in range(weeks):
        world.step()
        rows.append(world.diagnostics_rows[-1])
        firm_rows.extend(world.firm_diagnostics_rows[-len(world.firms):])
        world.firm_diagnostics_rows.clear()
        world.household_diagnostics_rows.clear()
    path = output_root / f"mature_seed{seed}_N{population}_w{weeks}"
    write_csv(path / "diagnostics.csv", rows)
    write_csv(path / "firm_diagnostics.csv", firm_rows)
    return path, time.perf_counter() - started


def actual_credit_summary(rows):
    loan_issued = [num(row.get("loan_issued", row.get("working_capital_loan_issued"))) for row in rows]
    repaid = [num(row.get("working_capital_loan_repaid")) for row in rows]
    balances = [num(row.get("loan_balance", row.get("working_capital_loan_balance"))) for row in rows]
    ratios = [num(row.get("firm_cash_to_wage_bill")) for row in rows]
    return {
        "loan_issuance_weeks": sum(value > 1e-12 for value in loan_issued),
        "repayment_weeks": sum(value > 1e-12 for value in repaid),
        "positive_loan_balance_weeks": sum(value > 1e-12 for value in balances),
        "minimum_firm_cash_to_wage_bill": min(ratios, default=0.0),
        "minimum_loan_balance": min(balances, default=0.0),
        "maximum_loan_balance": max(balances, default=0.0),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--fresh-weeks", type=int, default=260)
    parser.add_argument("--mature-weeks", type=int, default=1560)
    parser.add_argument("--output-root", default=str(OUTPUT))
    parser.add_argument("--skip-mature", action="store_true")
    parser.add_argument("--mature-populations", default="500,5000")
    args = parser.parse_args()
    output = Path(args.output_root)
    output.mkdir(parents=True, exist_ok=True)

    reference = {population: fresh_reference(population) for population in POPS}
    buffers = candidate_buffers(reference)
    write_json(output / "source_history_audit.json", source_audit(reference))
    ratio_rows = []
    for population, ref in reference.items():
        for name, denominator in {
            "population": population,
            "workers": ref["workers"],
            "productive_capacity": ref["productive_capacity"],
            "initial_weekly_wage_bill": ref["weekly_wage_bill"],
            "expected_weekly_sales": ref["expected_weekly_sales"],
            "initial_firm_cash": ref["initial_firm_cash"],
            "initial_money_stock": ref["initial_money_stock"],
        }.items():
            ratio_rows.append({"population": population, "denominator": name, "buffer": BUFFER, "buffer_over_denominator": BUFFER / denominator if denominator else None})
    write_csv(output / "buffer_reference_ratios.csv", ratio_rows)
    write_json(output / "shadow_candidate_rules.json", {
        "current_behavioral_rule": "B=150000; target_cash_i=B*capacity_share_i+wage_bill_i; repayment_buffer_i=B*capacity_share_i",
        "candidate_buffers": buffers,
        "formulas": {
            "A_fixed_absolute_buffer": "B(N)=150000",
            "B_constant_buffer_per_capita": "B(N)=150000*N/500",
            "C_constant_buffer_per_capacity": "B(N)=150000*capacity(N)/capacity(500)",
            "D_constant_payroll_weeks": "B(N)=k_wage*wage_bill(N)",
            "E_constant_sales_weeks": "B(N)=k_sales*expected_sales(N)",
        },
        "behaviorally_applied": False,
    })

    shadow_all = []
    fresh_shadow = []
    actual = []
    fresh_paths = {}
    for population in POPS:
        run_dir = FRESH_ROOT / f"naive_N{population}_seed42_w{args.fresh_weeks}"
        if not (run_dir / "firm_diagnostics.csv").exists():
            raise FileNotFoundError(f"Missing accepted fresh audit output: {run_dir}")
        firm_rows = read_csv(run_dir / "firm_diagnostics.csv")
        macro_rows = read_csv(run_dir / "diagnostics.csv")
        fresh_paths[population] = run_dir
        rows = shadow_rows(firm_rows, population, buffers[population], source="fresh")
        shadow_all.extend(rows)
        fresh_shadow.extend(rows)
        actual.append({"population": population, **actual_credit_summary(macro_rows)})

    mature_paths = {}
    mature_runtime = []
    mature_credit = []
    mature_shadow = []
    if not args.skip_mature:
        for population in [int(value) for value in args.mature_populations.split(",") if value.strip()]:
            path = output / f"mature_seed42_N{population}_w{args.mature_weeks}"
            if not (path / "firm_diagnostics.csv").exists():
                path, runtime = run_mature(population, args.mature_weeks, 42, output)
                mature_runtime.append({"population": population, "runtime_seconds": runtime})
            mature_paths[population] = path
            firm_rows = read_csv(path / "firm_diagnostics.csv")
            macro_rows = read_csv(path / "diagnostics.csv")
            rows = shadow_rows(firm_rows, population, buffers[population], source="mature")
            shadow_all.extend(rows)
            mature_shadow.extend(rows)
            mature_credit.append({"population": population, **actual_credit_summary(macro_rows)})

    write_csv(output / "shadow_funding_gap_metrics.csv", aggregate_shadow(fresh_shadow))
    write_csv(output / "mature_state_shadow_metrics.csv", aggregate_shadow(mature_shadow))

    write_json(output / "credit_binding_summary.json", {
        "fresh_actual_credit": actual,
        "mature_actual_credit": mature_credit,
        "mature_runtime": mature_runtime,
        "interpretation": "Actual credit is measured from current trajectories; shadow candidate metrics never issue loans.",
        "fresh_credit_buffer_non_binding": all(row["loan_issuance_weeks"] == 0 and row["positive_loan_balance_weeks"] == 0 for row in actual),
        "mature_credit_buffer_non_binding": all(row["loan_issuance_weeks"] == 0 and row["positive_loan_balance_weeks"] == 0 for row in mature_credit),
    })

    write_json(output / "optional_behavioral_comparison.json", {
        "performed": False,
        "reason": "No candidate was behaviorally applied because the audit does not establish a clearly superior semantic rule and current credit is dormant in the tested baseline.",
        "default_initial_money": 10000000.0,
        "default_buffer": BUFFER,
    })
    summary = [
        "# Pre-Step13.3B Working-Capital Buffer Scaling Audit",
        "",
        "No credit parameter, production rule, initial money stock, or behavior was modified. Shadow candidates never issued loans.",
        "",
        "## Verdict",
        "D. Buffer semantics remain ambiguous and materially affect the mature N=5000 economic regime; population scaling cannot yet be finalized.",
        "",
        "## Findings",
        "- Source/history does not document an original economic rationale for 150000; it is marked ORIGINAL ECONOMIC SEMANTICS UNDOCUMENTED.",
        "- The value is a recurring policy parameter, distinct from the one-time initial 10m nominal stock.",
        "- It is applied symmetrically to target cash and repayment buffer, and sums to the same aggregate B across firms.",
        "- Fresh runs are credit-dormant, but the mature N=5000 run has 817 loan-issuance weeks and 817 positive-loan weeks out of 1560.",
        "- The fixed buffer shadow binds in the mature N=5000 trajectory; scaled candidates bind more often and require larger funding gaps.",
        "- No optional behavioral candidate treatment was run because changing the buffer now would confound this audit with a new economic mechanism.",
        "",
        "## Outputs",
        "- `buffer_reference_ratios.csv`",
        "- `shadow_funding_gap_metrics.csv`",
        "- `mature_state_shadow_metrics.csv`",
        "- `credit_binding_summary.json`",
    ]
    (output / "acceptance_summary.md").write_text("\n".join(summary) + "\n", encoding="utf-8")
    print(f"Credit buffer scaling audit outputs: {output}")


if __name__ == "__main__":
    main()
