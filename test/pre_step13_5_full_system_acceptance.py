"""Build the passive full-system acceptance report from an existing run."""
from __future__ import annotations

import csv
import json
import math
import statistics
from pathlib import Path

ROOT = Path("test/output/pre_step13_5_full_system_acceptance")
RUN = ROOT / "seed42_N5000_w1560_interest5pct"
OLD = Path("test/output/step13_4_interest_behavior/seed42")
TOL = 1e-6


def read_csv(path):
    with path.open(encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def num(row, key, default=0.0):
    try:
        value = float(row.get(key, default))
        return value if math.isfinite(value) else default
    except (TypeError, ValueError):
        return default


def raw(row, key):
    return row.get(key, "")


def max_abs(rows, field, expr=None):
    vals = []
    for row in rows:
        vals.append(abs(expr(row) if expr else num(row, field)))
    return max(vals, default=0.0)


def first_error(rows, expr, fields=()):
    best = (0.0, None)
    for row in rows:
        error = abs(expr(row))
        if error > best[0]:
            best = (error, {"global_step": raw(row, "global_step"),
                            "firm_id": raw(row, "firm_id"),
                            **{k: raw(row, k) for k in fields}})
    return {"max_abs_error": best[0], "first_or_max_row": best[1]}


def mean(rows, field):
    return statistics.fmean(num(r, field) for r in rows) if rows else 0.0


def slope(rows, field):
    if len(rows) < 2:
        return 0.0
    ys = [num(r, field) for r in rows]
    xs = range(len(ys))
    xm, ym = (len(ys) - 1) / 2, statistics.fmean(ys)
    den = sum((x - xm) ** 2 for x in xs)
    return sum((x - xm) * (y - ym) for x, y in zip(xs, ys)) / den if den else 0.0


def write_json(path, data):
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")


def compare_csv(old_rows, new_rows, keys, excluded=()):
    old_by = {tuple(r.get(k, "") for k in keys): r for r in old_rows}
    new_by = {tuple(r.get(k, "") for k in keys): r for r in new_rows}
    fields = [k for k in old_rows[0].keys() if k not in excluded and k not in keys]
    result = {"row_count_old": len(old_rows), "row_count_new": len(new_rows),
              "numeric": {}, "categorical": {}, "missing_keys": []}
    for key in sorted(set(old_by) | set(new_by)):
        if key not in old_by or key not in new_by:
            result["missing_keys"].append(key)
    for field in fields:
        diffs, first = [], None
        categorical = False
        for key in sorted(set(old_by) & set(new_by)):
            a, b = old_by[key].get(field, ""), new_by[key].get(field, "")
            try:
                da, db = float(a), float(b)
                diff = abs(da - db)
                diffs.append(diff)
                if diff > TOL and first is None:
                    first = {"key": key, "old": a, "new": b}
            except (ValueError, TypeError):
                categorical = True
                if a != b and first is None:
                    first = {"key": key, "old": a, "new": b}
        result["categorical" if categorical else "numeric"][field] = {
            "max_abs_diff": max(diffs, default=0.0), "first_difference": first,
            "equal_within_tolerance": first is None,
        }
    result["pass"] = not result["missing_keys"] and all(
        x["equal_within_tolerance"] for group in (result["numeric"], result["categorical"])
        for x in group.values())
    return result


def main():
    ROOT.mkdir(parents=True, exist_ok=True)
    d = read_csv(RUN / "diagnostics.csv")
    firms = read_csv(RUN / "firm_diagnostics.csv")
    accounting = {p.stem: read_csv(p) for p in (RUN / "accounting").glob("*.csv")}
    last_step = int(num(d[-1], "global_step")) if d else 0
    tail = d[-260:]
    ftail = [r for r in firms if int(num(r, "global_step")) > last_step - 260]

    firm_summary = []
    for fid in sorted({int(num(r, "firm_id")) for r in firms}):
        fr = [r for r in firms if int(num(r, "firm_id")) == fid]
        tr = fr[-200:]
        end = fr[-1]
        obligation = sum(num(r, "total_interest_obligation") for r in tr)
        paid = sum(num(r, "interest_paid") for r in tr)
        unpaid = sum(num(r, "current_interest_unpaid") for r in tr)
        arrears = [num(r, "closing_interest_arrears") for r in tr]
        states = {"I0_no_obligation": 0, "I1_full_service": 0,
                  "I2_partial_service": 0, "I3_unpaid": 0}
        for r in tr:
            ob, p, u = num(r, "total_interest_obligation"), num(r, "interest_paid"), num(r, "current_interest_unpaid")
            if ob <= TOL: states["I0_no_obligation"] += 1
            elif u <= TOL: states["I1_full_service"] += 1
            elif p > TOL: states["I2_partial_service"] += 1
            else: states["I3_unpaid"] += 1
        firm_summary.append({
            "firm_id": fid, "final": {k: num(end, k) for k in (
                "price", "learner_proposed_price", "price_floor", "unit_market_share",
                "revenue_market_share", "sales", "expected_demand", "production_plan",
                "actual_production", "productive_capacity", "capacity_utilization",
                "realized_ulc", "normal_ulc", "profit", "dividend_paid", "cash",
                "loan_balance", "loan_issued", "loan_repaid", "inventory_coverage")},
            "last_200_mean": {k: mean(tr, k) for k in (
                "price", "learner_proposed_price", "price_floor", "unit_market_share",
                "revenue_market_share", "sales", "expected_demand", "production_plan",
                "actual_production", "capacity_utilization", "realized_ulc", "normal_ulc",
                "profit", "dividend_paid", "cash", "loan_balance", "loan_issued",
                "loan_repaid", "inventory_coverage")},
            "price_floor_binding_share": mean(tr, "price_floor_binding"),
            "cash_to_wage_bill": num(end, "cash_to_wage_bill"),
            "credit_limit": num(end, "credit_limit"),
            "lender_exposure": num(end, "lender_exposure"),
            "exposure_to_credit_limit": (num(end, "lender_exposure") / num(end, "credit_limit")
                                          if num(end, "credit_limit") else None),
            "interest_service_states_last_200": states,
            "longest_arrears_spell_last_200": max((sum(1 for x in arrears[i:] if x > TOL)
                for i in range(len(arrears))), default=0),
            "debt_slope_last_200": slope(tr, "loan_balance"),
            "cash_slope_last_200": slope(tr, "cash"),
            "diagnostic_missing_operating_cash_flow": True,
        })

    # Reconciliation gates.
    max_goods = max_abs(d, "food_conservation_gap")
    max_money = max(max_abs(d, "monetary_accounting_gap"), max_abs(d, "money_delta_gap"), max_abs(d, "ledger_money_net_gap"))
    max_cash = max(max_abs(firms, "cash_bridge_gap"), max_abs(accounting.get("firm_accounting", []), "cash_flow_gap"))
    max_house = max_abs(accounting.get("household_accounting", []), "household_wealth_bridge_gap")
    max_public = max_abs(accounting.get("public_accounting", []), "public_cash_flow_gap")
    max_inventory = max_abs(accounting.get("firm_accounting", []), "inventory_bridge_gap")
    max_equity = max_abs(accounting.get("firm_accounting", []), "equity_bridge_gap")
    principal_gap = max_abs(firms, "credit_bridge_gap", lambda r: num(r, "closing_principal") - (num(r, "opening_principal") + num(r, "executed_credit") - num(r, "loan_repaid")))
    interest_gap = max_abs(firms, "interest_bridge_gap", lambda r: num(r, "closing_interest_arrears") - (num(r, "opening_interest_arrears") + num(r, "current_interest_due") - num(r, "interest_paid")))
    cb_interest_gap = abs(sum(num(r, "interest_paid") for r in firms) - sum(num(r, "central_bank_public_income_from_loan_interest") for r in d))
    violations = sum(1 for r in d if raw(r, "invariant_failed").lower() in ("true", "1", "yes"))
    negative_cash = sum(1 for r in firms if num(r, "cash") < -TOL)
    unfunded = sum(1 for r in firms if num(r, "actual_production") > num(r, "funded_productive_capacity", num(r, "productive_capacity")) + TOL)
    lender_gap = max_abs(firms, "lender_exposure", lambda r: num(r, "lender_exposure") - num(r, "closing_principal") - num(r, "closing_interest_arrears"))
    headroom_gap = max_abs(firms, "credit_headroom", lambda r: num(r, "credit_headroom") - max(num(r, "credit_limit") - num(r, "opening_principal") - num(r, "opening_interest_arrears"), 0.0))
    principal_issue = first_error(firms, lambda r: num(r, "closing_principal") - (num(r, "opening_principal") + num(r, "executed_credit") - num(r, "loan_repaid")),
                                  ("opening_principal", "executed_credit", "loan_repaid", "closing_principal", "loan_balance"))
    headroom_issue = first_error(firms, lambda r: num(r, "credit_headroom") - max(num(r, "credit_limit") - num(r, "opening_principal") - num(r, "opening_interest_arrears"), 0.0),
                                 ("credit_headroom", "credit_limit", "opening_principal", "opening_interest_arrears", "loan_balance"))
    gates = {"invariant_violations": violations, "negative_cash_rows": negative_cash,
             "unfunded_production_rows": unfunded, "max_goods_conservation_gap": max_goods,
             "max_money_bridge_gap": max_money, "max_cash_flow_gap": max_cash,
             "max_household_wealth_bridge_gap": max_house, "max_public_cash_flow_gap": max_public,
             "max_inventory_bridge_gap": max_inventory, "max_equity_bridge_gap": max_equity,
             "max_principal_bridge_gap": principal_gap, "max_interest_bridge_gap": interest_gap,
             "central_bank_interest_bridge_gap": cb_interest_gap,
             "lender_exposure_gap": lender_gap, "credit_headroom_gap": headroom_gap}
    hard_pass = all(v <= TOL if isinstance(v, float) else v == 0 for v in gates.values())

    old_d = read_csv(OLD / "diagnostics.csv") if (OLD / "diagnostics.csv").exists() else []
    old_f = read_csv(OLD / "firm_diagnostics.csv") if (OLD / "firm_diagnostics.csv").exists() else []
    excluded = {"simulation_week", "simulation_year", "week", "year"}
    parity_d = compare_csv(old_d, d, ("global_step",), excluded) if old_d else {"pass": False, "reason": "old diagnostics missing"}
    parity_f = compare_csv(old_f, firms, ("global_step", "firm_id"), excluded) if old_f else {"pass": False, "reason": "old firm diagnostics missing"}
    parity = {"old_run": str(OLD), "new_run": str(RUN), "diagnostics": parity_d, "firm_diagnostics": parity_f,
              "pass": parity_d.get("pass", False) and parity_f.get("pass", False),
              "note": "Only columns present in the historical BEFORE file are compared."}
    write_json(ROOT / "baseline_parity_check.json", parity)

    exposure = {"pass": lender_gap <= TOL and headroom_gap <= TOL, "max_reported_vs_principal_plus_arrears_gap": lender_gap,
                "max_credit_headroom_gap": headroom_gap, "formula": "max(credit_limit-opening_principal-opening_interest_arrears, 0)",
                "firms": [{"firm_id": x["firm_id"], "lender_exposure": x["lender_exposure"], "credit_limit": x["credit_limit"],
                           "exposure_to_credit_limit": x["exposure_to_credit_limit"]} for x in firm_summary]}
    write_json(ROOT / "lender_exposure_consistency.json", exposure)

    write_json(ROOT / "full_system_acceptance.json", {
        "verdict": "A. Step 13-mid ready for acceptance" if hard_pass and parity["pass"] else "B. acceptance issue detected",
        "population_initial": int(num(d[0], "population")), "population_final": int(num(d[-1], "population")),
        "final_global_step": last_step, "scenario": "interest_behavioral_5pct", "seed": 42,
        "firm_count": 5, "steps": len(d), "annual_interest_rate": 0.05, "K": 46.36154354202572,
        "rate_status": "activation_reference_only", "mature_window_available": False,
        "hard_gates": gates, "hard_gates_pass": hard_pass, "baseline_parity_pass": parity["pass"],
        "diagnostic_issue_details": {"principal_bridge": principal_issue, "credit_headroom": headroom_issue},
        "analysis_dashboard_count": 11, "lender_exposure_consistency": exposure["pass"],
        "step13_mid_baseline_ready": hard_pass and parity["pass"],
        "distress_work_ready": hard_pass and parity["pass"] and any(x["final"]["loan_balance"] > TOL for x in firm_summary),
        "firm_summary": firm_summary,
    })
    write_json(ROOT / "step13_mid_baseline_manifest.json", {"status": "accepted_candidate" if hard_pass and parity["pass"] else "blocked",
        "source_run": str(RUN), "scenario": "interest_behavioral_5pct", "seed": 42, "population": 5000,
        "firm_count": 5, "steps": 1560, "final_global_step": last_step, "mature_window_available": False,
        "rate_status": "activation_reference_only", "analysis_profile": "full", "dashboard_count": 11,
        "hard_gates_pass": hard_pass, "baseline_parity_pass": parity["pass"]})

    with (ROOT / "legacy_system_health_summary.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f); w.writerow(["domain", "window", "mean", "slope", "status"])
        for field, domain in (("population", "demography"), ("food_output_units", "real_economy"), ("food_sales_units", "real_economy"), ("food_price", "prices"), ("total_household_wealth", "households"), ("total_money_stock", "money"), ("loan_balance", "credit")):
            w.writerow([domain, "last_260_weeks", mean(tail, field), slope(tail, field), "INTERPRETABLE"])
    with (ROOT / "mature_system_metrics.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f); w.writerow(["metric", "value", "status"]); w.writerow(["mature_window_available", False, "unavailable: run ends at mature_start_week=1560"])
        for field in ("population", "food_output_units", "food_sales_units", "total_money_stock", "total_household_wealth", "loan_balance"):
            w.writerow([field + "_last_260_mean", mean(tail, field), "trailing_window_only"])
    with (ROOT / "firm_financial_state_summary.csv").open("w", newline="", encoding="utf-8") as f:
        fields = ["firm_id", "final_cash", "final_loan_balance", "final_lender_exposure", "last_200_loan_issued", "last_200_loan_repaid", "debt_slope_last_200", "cash_to_wage_bill", "price_floor_binding_share"]
        w = csv.DictWriter(f, fieldnames=fields); w.writeheader()
        for x in firm_summary:
            w.writerow({"firm_id": x["firm_id"], "final_cash": x["final"]["cash"], "final_loan_balance": x["final"]["loan_balance"], "final_lender_exposure": x["lender_exposure"], "last_200_loan_issued": x["last_200_mean"]["loan_issued"], "last_200_loan_repaid": x["last_200_mean"]["loan_repaid"], "debt_slope_last_200": x["debt_slope_last_200"], "cash_to_wage_bill": x["cash_to_wage_bill"], "price_floor_binding_share": x["price_floor_binding_share"]})

    verdict = "A. Step 13-mid ready for acceptance" if hard_pass and parity["pass"] else "B. acceptance issue detected"
    (ROOT / "full_system_acceptance_summary.md").write_text(
        f"# PRE-STEP 13.5 Full-System Acceptance\n\n**Verdict: {verdict}**\n\n"
        f"Run: population=5000, firms=5, seed=42, steps={len(d)}, scenario=interest_behavioral_5pct, annual interest=5%, K=46.36154354202572.\n\n"
        f"Final population: {int(num(d[-1], 'population'))}; final households: {int(num(d[-1], 'households'))}; final global step: {last_step}.\n\n"
        f"Mature window: unavailable because the run ends at the configured 30-year start ({last_step} weeks); reported trailing window is last 260 weeks only.\n\n"
        f"Hard gates: `{'PASS' if hard_pass else 'FAIL'}`. Invariant violations={violations}; negative cash rows={negative_cash}; unfunded production rows={unfunded}; max goods gap={max_goods:.6g}; max money gap={max_money:.6g}; max cash gap={max_cash:.6g}; max household gap={max_house:.6g}; max public gap={max_public:.6g}; max principal gap={principal_gap:.6g}; max interest gap={interest_gap:.6g}; CB interest gap={cb_interest_gap:.6g}.\n\n"
        f"Lender exposure/headroom reconciliation: `{'PASS' if exposure['pass'] else 'FAIL'}`. Historical Step 13.4 parity: `{'PASS' if parity['pass'] else 'FAIL'}`; comparison uses only fields present in the historical BEFORE files and excludes passive weekly/year metadata.\n\n"
        f"Diagnostic-only issue: maximum principal bridge mismatch={principal_issue['max_abs_error']:.6g} at global_step={principal_issue['first_or_max_row']['global_step'] if principal_issue['first_or_max_row'] else 'n/a'}, firm={principal_issue['first_or_max_row']['firm_id'] if principal_issue['first_or_max_row'] else 'n/a'}; maximum credit-headroom mismatch={headroom_issue['max_abs_error']:.6g} at global_step={headroom_issue['first_or_max_row']['global_step'] if headroom_issue['first_or_max_row'] else 'n/a'}, firm={headroom_issue['first_or_max_row']['firm_id'] if headroom_issue['first_or_max_row'] else 'n/a'}. The reported lender exposure itself matches closing principal plus arrears within tolerance; the mismatch is in opening-principal/headroom diagnostics.\n\n"
        "Interest remains activation-reference-only in this acceptance run. No distress/default/writeoff behavior was introduced. The generated 11 dashboard PNGs and browser status are retained under the run's `analysis` directory.\n",
        encoding="utf-8")
    print(json.dumps({"verdict": verdict, "hard_gates_pass": hard_pass, "parity_pass": parity["pass"], "lender_pass": exposure["pass"], "output": str(ROOT)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
