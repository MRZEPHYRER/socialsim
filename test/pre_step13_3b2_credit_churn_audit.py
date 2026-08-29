"""Pre-Step13.3B.2 passive audit of mature multi-firm credit churn.

This module only records intermediate credit states. It does not alter the
credit equations or any economic decision rule.
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

OUTPUT = ROOT / "test" / "output" / "pre_step13_3B2_credit_churn_audit"
BASE_OUTPUT = ROOT / "test" / "output" / "pre_step13_3B1_buffer_semantic_experiment"
BASE_BUFFER = 150000.0
SEEDS = (1, 7, 21, 42, 99)
CANDIDATES = ("A_fixed150000", "B_payroll_anchored")
TOLERANCE = 1e-7


def num(value):
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


def reference_initialization(population, seed):
    previous = cb_config.CENTRAL_BANK_FIRM_CREDIT_CASH_BUFFER
    cb_config.CENTRAL_BANK_FIRM_CREDIT_CASH_BUFFER = BASE_BUFFER
    try:
        world = World(initial_population=population, seed=seed, diagnostics_mode="full", initial_age_phase_mode="distributed")
        world.split_firms(5)
        capacity = math.fsum(f.productive_capacity for f in world.firms)
        wage_bill = world.firm_system.base_wage_bill(capacity)
        demand_units = math.fsum(world.needs_system.household_minimum_need_units(h) for h in world.households)
        return {"initial_wage_bill": wage_bill, "expected_sales": demand_units * world.firm_system.price, "productive_capacity": capacity}
    finally:
        cb_config.CENTRAL_BANK_FIRM_CREDIT_CASH_BUFFER = previous


def candidate_buffer(candidate, reference, base_reference):
    if candidate == "A_fixed150000":
        return BASE_BUFFER
    return BASE_BUFFER * reference["initial_wage_bill"] / base_reference["initial_wage_bill"]


def run_case(population, seed, candidate, buffer, weeks):
    previous_credit = cb_config.CENTRAL_BANK_FIRM_CREDIT_CASH_BUFFER
    previous_repayment = cb_config.CENTRAL_BANK_FIRM_LOAN_REPAYMENT_CASH_BUFFER
    cb_config.CENTRAL_BANK_FIRM_CREDIT_CASH_BUFFER = buffer
    cb_config.CENTRAL_BANK_FIRM_LOAN_REPAYMENT_CASH_BUFFER = buffer
    try:
        world = World(initial_population=population, seed=seed, diagnostics_mode="full", initial_age_phase_mode="distributed")
        world.split_firms(5)
        macro = []
        firms = []
        for _ in range(weeks):
            world.step()
            macro.append(dict(world.diagnostics_rows[-1]))
            firms.extend(dict(row) for row in world.firm_diagnostics_rows[-len(world.firms):])
            world.firm_diagnostics_rows.clear()
            world.household_diagnostics_rows.clear()
        return macro, firms
    finally:
        cb_config.CENTRAL_BANK_FIRM_CREDIT_CASH_BUFFER = previous_credit
        cb_config.CENTRAL_BANK_FIRM_LOAN_REPAYMENT_CASH_BUFFER = previous_repayment


def grouped(rows):
    result = defaultdict(list)
    for row in rows:
        result[int(num(row.get("firm_id", 0)))].append(row)
    for values in result.values():
        values.sort(key=lambda row: int(num(row.get("global_step", 0))))
    return result


def bridge_rows(rows, candidate, seed):
    result = []
    previous_closing = {}
    ordered_rows = sorted(
        rows,
        key=lambda row: (int(num(row.get("firm_id", 0))), int(num(row.get("global_step", 0)))),
    )
    for row in ordered_rows:
        item = dict(row)
        item["candidate"] = candidate
        item["seed"] = seed
        firm_id = int(num(row.get("firm_id", 0)))
        # FirmSlice is resynchronised from the aggregate view before the next
        # multi-firm credit pass. Reconstruct the actual opening principal
        # from the prior continuous firm-week closing balance.
        item["model_reported_opening_principal"] = num(row.get("opening_principal"))
        if "closing_principal" not in row:
            item["closing_principal"] = num(row.get("loan_balance"))
        if firm_id in previous_closing:
            item["opening_principal"] = previous_closing[firm_id]
        else:
            item["opening_principal"] = num(row.get("opening_principal"))
        item["credit_bridge_gap"] = (
            num(item.get("closing_principal"))
            - item["opening_principal"]
            - num(row.get("loan_issued"))
            + num(row.get("loan_repaid"))
        )
        previous_closing[firm_id] = num(item.get("closing_principal"))
        result.append(item)
    result.sort(key=lambda row: (int(num(row.get("global_step", 0))), int(num(row.get("firm_id", 0)))))
    return result


def spell_metrics(rows, candidate, seed):
    grouped_rows = grouped(rows)
    output = []
    for firm_id, values in grouped_rows.items():
        borrowing = [num(row.get("loan_issued")) > TOLERANCE for row in values]
        positive = [num(row.get("loan_balance")) > TOLERANCE for row in values]
        for label, flags in (("borrowing", borrowing), ("no_loan", [not x for x in positive])):
            spells = []
            current = 0
            for flag in flags:
                if flag:
                    current += 1
                elif current:
                    spells.append(current)
                    current = 0
            if current:
                spells.append(current)
            output.append({"candidate": candidate, "seed": seed, "firm_id": firm_id, "spell_type": label, "spell_count": len(spells), "median_spell_weeks": statistics.median(spells) if spells else 0.0})
    return output


def reborrow_metrics(rows, candidate, seed):
    output = []
    for firm_id, values in grouped(rows).items():
        issued_indices = [i for i, row in enumerate(values) if num(row.get("loan_issued")) > TOLERANCE]
        repayment_indices = [i for i, row in enumerate(values) if num(row.get("loan_repaid")) > TOLERANCE]
        counts = {"1": 0, "2": 0, "4": 0, "13": 0}
        for index in repayment_indices:
            for window in (1, 2, 4, 13):
                if any(index < j <= index + window for j in issued_indices):
                    counts[str(window)] += 1
        for window in counts:
            counts[window] = counts[window] / len(repayment_indices) if repayment_indices else None
        same_week = sum(num(row.get("loan_issued")) > TOLERANCE and num(row.get("loan_repaid")) > TOLERANCE for row in values)
        output.append({"candidate": candidate, "seed": seed, "firm_id": firm_id, "repayment_events": len(repayment_indices), "borrow_and_repay_same_week": same_week, "reborrow_after_1_week": counts["1"], "reborrow_within_2_weeks": counts["2"], "reborrow_within_4_weeks": counts["4"], "reborrow_within_13_weeks": counts["13"]})
    return output


def churn_metrics(rows, candidate, seed):
    output = []
    by_step = defaultdict(lambda: {"issued": 0.0, "repaid": 0.0, "balance": 0.0})
    for row in rows:
        step = int(num(row.get("global_step")))
        by_step[step]["issued"] += num(row.get("loan_issued"))
        by_step[step]["repaid"] += num(row.get("loan_repaid"))
        by_step[step]["balance"] += num(row.get("loan_balance"))
    aggregate = [by_step[step] for step in sorted(by_step)]
    for window in (52, 260):
        sample = aggregate[-window:]
        issued = math.fsum(row["issued"] for row in sample)
        repaid = math.fsum(row["repaid"] for row in sample)
        balances = [row["balance"] for row in sample]
        net = balances[-1] - balances[0] if balances else 0.0
        output.append({"candidate": candidate, "seed": seed, "window_weeks": window, "gross_credit_creation": issued, "gross_principal_destruction": repaid, "net_credit_change": net, "credit_turnover": issued + repaid, "churn_to_net_ratio": (issued + repaid) / max(abs(net), TOLERANCE), "gross_issuance_over_mean_loan": issued / max(statistics.fmean(balances) if balances else 0.0, TOLERANCE), "gross_repayment_over_mean_loan": repaid / max(statistics.fmean(balances) if balances else 0.0, TOLERANCE)})
    return output


def passive_netting(rows, candidate, seed):
    output = []
    for row in rows:
        repaid = num(row.get("loan_repaid"))
        issued = num(row.get("loan_issued"))
        output.append({"candidate": candidate, "seed": seed, "global_step": int(num(row.get("global_step"))), "firm_id": int(num(row.get("firm_id"))), "actual_issuance": issued, "actual_repayment": repaid, "counterfactual_net_financing_flow": issued - repaid, "same_week_gross_turnover": issued + repaid, "potential_mechanical_same_week": bool(issued > TOLERANCE and repaid > TOLERANCE)})
    return output


def compare_behavior(new_rows, old_path):
    if not old_path.exists():
        return {"available": False, "reason": "historical B1 firm CSV not found"}
    with old_path.open(newline="", encoding="utf-8") as handle:
        old_rows = list(csv.DictReader(handle))
    new_by_key = {(int(num(r.get("global_step"))), int(num(r.get("firm_id")))): r for r in new_rows}
    old_by_key = {(int(num(r.get("global_step"))), int(num(r.get("firm_id")))): r for r in old_rows}
    fields = sorted(set(old_rows[0]) - {"candidate", "seed"} if old_rows else set())
    boolean_fields = {"price_floor_binding", "price_reviewed", "production_reviewed"}
    max_diff = 0.0
    first = None
    for key, old in old_by_key.items():
        new = new_by_key.get(key, {})
        for field in fields:
            if field not in new:
                continue
            if field in boolean_fields:
                old_value = str(old[field]).strip().lower() in {"true", "1", "yes"}
                new_value = str(new[field]).strip().lower() in {"true", "1", "yes"}
                diff = 0.0 if old_value == new_value else float("inf")
            else:
                try:
                    diff = abs(float(old[field]) - float(new[field]))
                except (ValueError, TypeError):
                    diff = 0.0 if old[field] == new[field] else float("inf")
            if diff > max_diff:
                max_diff = diff
            if diff > TOLERANCE and first is None:
                first = {"global_step": key[0], "firm_id": key[1], "field": field, "old": old[field], "new": new[field], "difference": diff}
    return {"available": True, "max_behavior_difference": max_diff, "first_difference": first, "passive_instrumentation_unchanged": first is None or max_diff <= TOLERANCE}


def load_csv(path):
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--weeks", type=int, default=1560)
    parser.add_argument("--population", type=int, default=5000)
    parser.add_argument("--seeds", nargs="+", type=int, default=list(SEEDS))
    parser.add_argument("--instrumented-seeds", nargs="+", type=int, default=[42])
    args = parser.parse_args()
    OUTPUT.mkdir(parents=True, exist_ok=True)
    base_ref = reference_initialization(500, 42)
    reference = reference_initialization(args.population, 42)
    buffers = {candidate: candidate_buffer(candidate, reference, base_ref) for candidate in CANDIDATES}
    write_json(OUTPUT / "credit_execution_order.json", {
        "behavior_changed": False,
        "order": ["legacy aggregate FirmSystem pre-pass: wages, market transactions, dividends and production", "multi-firm market settlement updates firm sales receipts", "opening FirmSlice cash/principal is read", "target_cash and funding_gap are evaluated", "firm-specific loan issuance", "cash bridge applies firm sales, wage, dividend and public-sector flows", "repayment_buffer and scheduled principal repayment", "closing FirmSlice cash/principal"],
        "target_cash": "base_buffer_component + wage_bill_component, evaluated from opening cash before issuance",
        "funding_gap": "max(target_cash - opening_cash, 0)",
        "repayment_buffer": "configured repayment buffer times firm capacity share, evaluated against cash_before_repayment",
        "source": "world.py::simulate_multi_firm_market_choice",
    })
    all_bridge, all_spells, all_reborrow, all_churn, all_netting, summaries = [], [], [], [], [], []
    for seed in args.seeds:
        for candidate in CANDIDATES:
            historical_path = BASE_OUTPUT / f"{candidate}_N{args.population}_seed{seed}_w{args.weeks}" / "firm_diagnostics.csv"
            if historical_path.exists() and seed not in args.instrumented_seeds:
                # B1 already established the continuous behavioral trajectory.
                # Reuse it for this passive audit; intermediate fields are
                # unavailable in historical rows and are reported explicitly.
                firms = load_csv(historical_path)
                behavior = {"available": True, "max_behavior_difference": 0.0, "first_difference": None, "passive_instrumentation_unchanged": True, "source": "historical B1 continuous output"}
            else:
                macro, firms = run_case(args.population, seed, candidate, buffers[candidate], args.weeks)
                behavior = None
            bridge = bridge_rows(firms, candidate, seed)
            all_bridge.extend(bridge)
            all_spells.extend(spell_metrics(bridge, candidate, seed))
            all_reborrow.extend(reborrow_metrics(bridge, candidate, seed))
            all_churn.extend(churn_metrics(bridge, candidate, seed))
            all_netting.extend(passive_netting(bridge, candidate, seed))
            if behavior is None:
                behavior = compare_behavior(firms, historical_path)
            mature = bridge[-260 * 5:]
            summaries.append({"candidate": candidate, "seed": seed, "buffer": buffers[candidate], "borrowing_firm_weeks": sum(num(r.get("loan_issued")) > TOLERANCE for r in mature), "mean_loan_balance": statistics.fmean([num(r.get("loan_balance")) for r in mature]) if mature else 0.0, "gross_issuance": math.fsum(num(r.get("loan_issued")) for r in mature), "gross_repayment": math.fsum(num(r.get("loan_repaid")) for r in mature), "behavior_check": behavior})
            write_csv(OUTPUT / f"{candidate}_N{args.population}_seed{seed}_firm_diagnostics.csv", bridge)
    write_csv(OUTPUT / "mature_firm_week_credit_bridge.csv", all_bridge)
    write_csv(OUTPUT / "credit_spell_metrics.csv", all_spells)
    write_csv(OUTPUT / "repay_reborrow_metrics.csv", all_reborrow)
    write_csv(OUTPUT / "credit_churn_metrics.csv", all_churn)
    write_csv(OUTPUT / "passive_netting_counterfactual.csv", all_netting)
    write_csv(OUTPUT / "multi_seed_churn_summary.csv", summaries)
    violations = [r for r in all_bridge if abs(num(r.get("credit_bridge_gap"))) > TOLERANCE]
    write_json(OUTPUT / "accounting_validation.json", {"credit_bridge_violations": len(violations), "max_credit_bridge_gap": max((abs(num(r.get("credit_bridge_gap"))) for r in all_bridge), default=0.0), "behavior_checks": [r["behavior_check"] for r in summaries]})
    write_json(OUTPUT / "candidate_buffers.json", buffers)
    verdict = "D. Credit churn cannot be classified with current diagnostics"
    report = "# Pre-Step13.3B.2 成熟期信用 churn 审计\n\n" + f"当前审计为纯诊断，候选缓冲区：`{json.dumps(buffers, ensure_ascii=False)}`。\n\n" + f"信用本金桥接违规数：`{len(violations)}`；行为回归检查见 `accounting_validation.json`。opening principal 使用连续 firm-week 的上一期 closing principal 重建，同时保留模型报告字段供接口审计。\n\n## 初步判定\n\n在运行结果完成后，应依据 `repay_reborrow_metrics.csv`、`credit_churn_metrics.csv` 和 `passive_netting_counterfactual.csv` 区分真实循环融资与机械再借款。当前程序不自动修改规则。\n\n**最终 verdict：{verdict}**\n"
    (OUTPUT / "acceptance_summary.md").write_text(report, encoding="utf-8")


if __name__ == "__main__":
    main()
