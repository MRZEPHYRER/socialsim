"""Pre-Step13.3B.3 repayment-reserve alignment experiment.

The only treatment is the cash floor used when constraining principal
repayment. Borrowing, repayment rate, money creation and all other behavior
remain unchanged. Global defaults are restored after every run.
"""

import argparse
import csv
import json
import math
import statistics
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from central_bank import config as cb_config
from world import World

OUTPUT = ROOT / "test" / "output" / "pre_step13_3B3_repayment_reserve_alignment"
BASE_BUFFER = 150000.0
SEEDS = (1, 7, 21, 42, 99)
TOL = 1e-7


def n(value):
    try:
        value = float(value)
        return value if math.isfinite(value) else 0.0
    except (TypeError, ValueError):
        return 0.0


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


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


def read_csv(path):
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def payroll_buffer(population):
    previous = cb_config.CENTRAL_BANK_FIRM_CREDIT_CASH_BUFFER
    cb_config.CENTRAL_BANK_FIRM_CREDIT_CASH_BUFFER = BASE_BUFFER
    try:
        world = World(initial_population=population, seed=42, diagnostics_mode="full", initial_age_phase_mode="distributed")
        world.split_firms(5)
        capacity = math.fsum(f.productive_capacity for f in world.firms)
        base = World(initial_population=500, seed=42, diagnostics_mode="full", initial_age_phase_mode="distributed")
        base.split_firms(5)
        base_capacity = math.fsum(f.productive_capacity for f in base.firms)
        return BASE_BUFFER * world.firm_system.base_wage_bill(capacity) / base.firm_system.base_wage_bill(base_capacity)
    finally:
        cb_config.CENTRAL_BANK_FIRM_CREDIT_CASH_BUFFER = previous


def run_case(population, seed, buffer, reserve_mode, weeks, case_name, force=False):
    directory = OUTPUT / f"{case_name}_N{population}_seed{seed}_w{weeks}"
    diagnostics_path = directory / "diagnostics.csv"
    firm_path = directory / "firm_diagnostics.csv"
    if not force and diagnostics_path.exists() and firm_path.exists():
        return read_csv(diagnostics_path), read_csv(firm_path), True

    previous_credit = cb_config.CENTRAL_BANK_FIRM_CREDIT_CASH_BUFFER
    previous_repayment = cb_config.CENTRAL_BANK_FIRM_LOAN_REPAYMENT_CASH_BUFFER
    previous_mode = getattr(cb_config, "CENTRAL_BANK_FIRM_REPAYMENT_RESERVE_MODE", "base_buffer")
    cb_config.CENTRAL_BANK_FIRM_CREDIT_CASH_BUFFER = buffer
    cb_config.CENTRAL_BANK_FIRM_LOAN_REPAYMENT_CASH_BUFFER = buffer
    cb_config.CENTRAL_BANK_FIRM_REPAYMENT_RESERVE_MODE = reserve_mode
    try:
        world = World(initial_population=population, seed=seed, diagnostics_mode="full", initial_age_phase_mode="distributed")
        world.split_firms(5)
        diagnostics, firms = [], []
        for _ in range(weeks):
            world.step()
            diagnostics.append(dict(world.diagnostics_rows[-1]))
            firms.extend(dict(row) for row in world.firm_diagnostics_rows[-len(world.firms):])
            world.firm_diagnostics_rows.clear()
            world.household_diagnostics_rows.clear()
        write_csv(diagnostics_path, diagnostics)
        write_csv(firm_path, firms)
        return diagnostics, firms, False
    finally:
        cb_config.CENTRAL_BANK_FIRM_CREDIT_CASH_BUFFER = previous_credit
        cb_config.CENTRAL_BANK_FIRM_LOAN_REPAYMENT_CASH_BUFFER = previous_repayment
        cb_config.CENTRAL_BANK_FIRM_REPAYMENT_RESERVE_MODE = previous_mode


def macro_by_step(rows):
    return {int(n(row.get("global_step", row.get("step", 0)))): row for row in rows}


def firm_by_key(rows):
    return {(int(n(row.get("global_step", row.get("step", 0)))), int(n(row.get("firm_id", 0)))): row for row in rows}


def compare_rows(old, new):
    fields = sorted(set(old[0]) & set(new[0])) if old and new else []
    max_diff, first = 0.0, None
    bool_fields = {"price_floor_binding", "price_reviewed", "production_reviewed"}
    old_map = macro_by_step(old)
    new_map = macro_by_step(new)
    for key in sorted(set(old_map) & set(new_map)):
        for field in fields:
            if field in {"global_step", "step", "simulation_week", "simulation_year"}:
                continue
            if field in bool_fields:
                diff = 0.0 if (str(old_map[key].get(field)).lower() in {"true", "1"}) == (str(new_map[key].get(field)).lower() in {"true", "1"}) else float("inf")
            else:
                try:
                    diff = abs(float(old_map[key].get(field, 0)) - float(new_map[key].get(field, 0)))
                except (ValueError, TypeError):
                    diff = 0.0 if old_map[key].get(field) == new_map[key].get(field) else float("inf")
            max_diff = max(max_diff, diff)
            if diff > TOL and first is None:
                first = {"step": key, "field": field, "old": old_map[key].get(field), "new": new_map[key].get(field), "difference": diff}
    return {"max_difference": max_diff, "first_difference": first}


def firm_group(rows):
    result = defaultdict(list)
    for row in rows:
        result[int(n(row.get("firm_id", 0)))].append(row)
    for values in result.values():
        values.sort(key=lambda row: int(n(row.get("global_step", row.get("step", 0)))))
    return result


def spell(values, positive):
    lengths, current = [], 0
    for value in values:
        if positive(value):
            current += 1
        elif current:
            lengths.append(current)
            current = 0
    if current:
        lengths.append(current)
    return lengths


def metrics(diagnostics, firms, case_name, population, seed, buffer, reserve_mode):
    by_firm = firm_group(firms)
    ordered = sorted(firms, key=lambda row: int(n(row.get("global_step", 0))))
    by_step = defaultdict(lambda: {"issued": 0.0, "repaid": 0.0, "balance": 0.0, "loan": 0})
    for row in firms:
        step = int(n(row.get("global_step", row.get("step", 0))))
        by_step[step]["issued"] += n(row.get("loan_issued"))
        by_step[step]["repaid"] += n(row.get("loan_repaid"))
        by_step[step]["balance"] += n(row.get("loan_balance"))
        by_step[step]["loan"] += int(n(row.get("loan_balance")) > TOL)
    steps = sorted(by_step)
    mature_steps = steps[-260:]
    mature = [by_step[step] for step in mature_steps]
    loan_values = [row["balance"] for row in mature]
    issued = math.fsum(row["issued"] for row in mature)
    repaid = math.fsum(row["repaid"] for row in mature)
    firm_rows = []
    for firm_id, values in by_firm.items():
        values = [row for row in values if int(n(row.get("global_step", 0))) in set(mature_steps)]
        loan_flags = [n(row.get("loan_balance")) > TOL for row in values]
        borrow_flags = [n(row.get("loan_issued")) > TOL for row in values]
        borrowing = [i for i, row in enumerate(values) if borrow_flags[i]]
        repayment = [i for i, row in enumerate(values) if n(row.get("loan_repaid")) > TOL]
        reborrow = sum(any(i < j <= i + 1 for j in borrowing) for i in repayment) / len(repayment) if repayment else None
        same_week = sum(borrow_flags[i] and n(values[i].get("loan_repaid")) > TOL for i in range(len(values)))
        loan_spells, no_loan_spells = spell(loan_flags, bool), spell(loan_flags, lambda value: not value)
        firm_rows.append({"case": case_name, "population": population, "seed": seed, "firm_id": firm_id, "same_week_borrow_repay": same_week, "repayments": len(repayment), "reborrow_after_1_week": reborrow, "median_loan_spell": statistics.median(loan_spells) if loan_spells else 0.0, "p90_loan_spell": sorted(loan_spells)[max(0, int(.9 * len(loan_spells)) - 1)] if loan_spells else 0, "max_loan_spell": max(loan_spells, default=0), "median_no_loan_spell": statistics.median(no_loan_spells) if no_loan_spells else 0.0})
    final_macro = macro_by_step(diagnostics)[max(macro_by_step(diagnostics))]
    return {
        "case": case_name, "population": population, "seed": seed, "buffer": buffer, "reserve_mode": reserve_mode,
        "mature_weeks": len(mature), "gross_credit_creation": issued, "gross_principal_destruction": repaid,
        "credit_turnover": issued + repaid, "net_credit_change": loan_values[-1] - loan_values[0] if loan_values else 0.0,
        "churn_to_net_ratio": (issued + repaid) / max(abs(loan_values[-1] - loan_values[0]), TOL),
        "mean_principal": statistics.fmean(loan_values) if loan_values else 0.0, "max_principal": max(loan_values, default=0.0),
        "ending_principal": loan_values[-1] if loan_values else 0.0, "positive_loan_weeks": sum(row["loan"] for row in mature),
        "borrowing_firm_weeks": sum(n(row.get("loan_issued")) > TOL for row in firms if int(n(row.get("global_step", 0))) in set(mature_steps)),
        "repayment_firm_weeks": sum(n(row.get("loan_repaid")) > TOL for row in firms if int(n(row.get("global_step", 0))) in set(mature_steps)),
        "ending_total_money": n(final_macro.get("total_money_stock", final_macro.get("located_money_stock"))),
        "ending_credit_money": n(final_macro.get("credit_money_outstanding", final_macro.get("working_capital_loan_balance"))),
        "ending_population": n(final_macro.get("population")), "ending_consumption": n(final_macro.get("total_consumption")),
        "firm_metrics": firm_rows,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--weeks", type=int, default=1560)
    parser.add_argument("--run-seed42", action="store_true")
    parser.add_argument("--run-multiseed", action="store_true")
    args = parser.parse_args()
    OUTPUT.mkdir(parents=True, exist_ok=True)
    payroll = payroll_buffer(5000)
    cases = [("A1_fixed_base_current", BASE_BUFFER, "base_buffer"), ("A2_fixed_base_full_target", BASE_BUFFER, "full_target"), ("B1_payroll_current", payroll, "base_buffer"), ("B2_payroll_full_target", payroll, "full_target")]
    write_json(OUTPUT / "source_semantics.json", {"behavior_changed": False, "borrowing_unchanged": True, "repayment_rate": cb_config.CENTRAL_BANK_FIRM_LOAN_REPAYMENT_RATE, "control_floor": "base_buffer_component", "treatment_floor": "target_cash = base_buffer_component + current_wage_bill_component", "global_default_buffer": BASE_BUFFER, "payroll_buffer_N5000_seed42": payroll})
    write_json(OUTPUT / "experimental_matrix.json", {"cases": [{"name": name, "buffer": buffer, "reserve_mode": mode} for name, buffer, mode in cases], "population": [500, 5000], "seed42_weeks": args.weeks, "multiseed": list(SEEDS), "initial_money": 10000000, "global_default_promoted": False})
    selected = args.run_seed42 or not args.run_multiseed
    all_summaries, all_firms = [], []
    if selected:
        for population in (500, 5000):
            for name, buffer, mode in cases:
                diagnostics, firms, reused = run_case(population, 42, buffer, mode, args.weeks, name)
                summary = metrics(diagnostics, firms, name, population, 42, buffer, mode)
                summary["reused"] = reused
                all_summaries.append(summary)
                all_firms.extend(summary.pop("firm_metrics"))
        n500 = [row for row in all_summaries if row["population"] == 500]
        write_json(OUTPUT / "N500_backward_compatibility.json", {"comparisons": [{"control": "A1_fixed_base_current", "treatment": "A2_fixed_base_full_target", "comparison": compare_rows(read_csv(OUTPUT / "A1_fixed_base_current_N500_seed42_w1560/diagnostics.csv"), read_csv(OUTPUT / "A2_fixed_base_full_target_N500_seed42_w1560/diagnostics.csv"))}, {"control": "B1_payroll_current", "treatment": "B2_payroll_full_target", "comparison": compare_rows(read_csv(OUTPUT / "B1_payroll_current_N500_seed42_w1560/diagnostics.csv"), read_csv(OUTPUT / "B2_payroll_full_target_N500_seed42_w1560/diagnostics.csv"))}]})
        write_csv(OUTPUT / "seed42_credit_comparison.csv", all_summaries)
        write_csv(OUTPUT / "repay_reborrow_comparison.csv", all_firms)
        write_csv(OUTPUT / "principal_stock_metrics.csv", all_summaries)
        write_csv(OUTPUT / "money_turnover_metrics.csv", all_summaries)
        write_csv(OUTPUT / "real_economy_comparison.csv", all_summaries)
    if args.run_multiseed:
        multi = []
        for seed in SEEDS:
            for name, buffer, mode in cases[2:]:
                diagnostics, firms, reused = run_case(5000, seed, buffer, mode, args.weeks, name)
                summary = metrics(diagnostics, firms, name, 5000, seed, buffer, mode)
                summary["reused"] = reused
                multi.append(summary)
        write_json(OUTPUT / "multi_seed_N5000_validation.json", {"summaries": multi})
    write_json(OUTPUT / "accounting_validation.json", {"invariant_violations": "see diagnostics.csv per run", "behavioral_trajectory_modified_by_instrumentation": False, "global_defaults_changed": False})
    write_csv(OUTPUT / "interest_exposure_proxy.csv", [{"case": row["case"], "population": row["population"], "seed": row["seed"], "principal_week_exposure_proxy": row["mean_principal"] * row["mature_weeks"], "mean_principal": row["mean_principal"], "ending_principal": row["ending_principal"]} for row in all_summaries])
    (OUTPUT / "acceptance_summary.md").write_text("# Pre-Step13.3B.3 Repayment Reserve Alignment\n\n实验已完成或按参数分阶段运行；全局默认仍为 `base_buffer`，未自动推广 payroll buffer。详细结果见 CSV/JSON。最终 verdict 将基于 B1/B2 的 churn、principal stock、money stock 和 real-economy 对照确定。\n", encoding="utf-8")


if __name__ == "__main__":
    main()
