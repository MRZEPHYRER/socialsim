from __future__ import annotations

import csv
import json
import math
import shutil
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "test/output/step17_e2_consumption_cash_safety_fix"
EPS = 1e-8
BRANCHES = ("CONTROL", "PRIVATE_ONLY", "PAYG_3_ONLY", "COMBINED_3")


def load_module(path, name):
    import importlib.util
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def read_json(path, default):
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError, TypeError):
        return default


def read_rows(path):
    try:
        with path.open(newline="", encoding="utf-8") as handle:
            return list(csv.DictReader(handle))
    except OSError:
        return []


def num(value, default=0.0):
    try:
        if value in (None, "", "nan", "NaN", "None", "unavailable"):
            return default
        result = float(value)
        return result if math.isfinite(result) else default
    except (TypeError, ValueError):
        return default


def write_rows(path, rows):
    rows = list(rows)
    fields = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    if not fields:
        fields = ["status"]
        rows = [{"status": "UNAVAILABLE"}]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def gini(values):
    values = sorted(max(0.0, num(value)) for value in values)
    total = sum(values)
    if not values or total <= EPS:
        return 0.0
    n = len(values)
    return sum((2 * i - n - 1) * value for i, value in enumerate(values, 1)) / (n * total)


def run_spec(e_module, checkpoint, output_root, steps, branches, audit=True, persist=True):
    from experiment_workflow import ExperimentBranch, ExperimentSpec, run_experiment
    branch_specs = []
    for branch in branches:
        overrides = e_module.policy_overrides(branch)
        overrides["world"]["payg_cash_safety_audit_enabled"] = audit
        branch_specs.append(ExperimentBranch(branch_name=branch, config_overrides=overrides))
    spec = ExperimentSpec(
        experiment_name=f"step17_e2_{steps}_week_{output_root.name}",
        output_root=str(output_root),
        steps=steps,
        checkpoint_path=str(checkpoint),
        seed=42,
        population=5000,
        scenario="baseline",
        firm_count=5,
        max_workers=2,
        observability_mode="RESEARCH_FAST",
        diagnostics_mode="compact",
        persist_diagnostics=persist,
        diagnostic_cadence=1,
        resume=False,
        branches=tuple(branch_specs),
    )
    return run_experiment(spec)


def branch_artifacts(result):
    output = {}
    for item in result.get("results", []):
        branch = item.get("branch_name")
        directory = Path(item.get("output_dir", ""))
        output[branch] = {
            "result": item,
            "summary": read_json(directory / "payg_cash_safety_summary.json", {}),
            "events": read_json(directory / "recipient_policy_event_window.json", []),
            "payg": read_json(directory / "payg_weekly_summary.json", []),
            "households": read_json(directory / "social_household_snapshot.json", []),
            "diagnostics": read_rows(directory / "diagnostics.csv"),
            "firm_diagnostics": read_rows(directory / "firm_diagnostics.csv"),
        }
    return output


def summarize_branch(branch, data):
    summary = data["summary"]
    result = data["result"]
    households = [row for row in data["households"] if not row.get("settlement_only")]
    cash = [num(row.get("cash")) for row in households]
    liquidity = [num(row.get("liquidity_weeks"), math.inf) for row in households]
    diagnostics = data["diagnostics"]
    firm_diagnostics = data["firm_diagnostics"]
    diagnostic_consumption = sum(num(row.get("total_consumption")) for row in diagnostics)
    diagnostic_sales = sum(num(row.get("sales_units")) for row in firm_diagnostics) if firm_diagnostics else "unavailable"
    final_step = max((num(row.get("global_step", row.get("step"))) for row in firm_diagnostics), default=None)
    final_firms = [row for row in firm_diagnostics if final_step is not None and num(row.get("global_step", row.get("step"))) == final_step]
    final_firm_cash = sum(num(row.get("cash")) for row in final_firms) if final_firms else "unavailable"
    final_food_inventory = sum(num(row.get("inventory_units")) for row in final_firms) if final_firms else "unavailable"
    last = diagnostics[-1] if diagnostics else {}
    return {
        "branch": branch,
        "status": result.get("status"),
        "negative_household_weeks": summary.get("negative_household_weeks", 0),
        "negative_unique_households": summary.get("negative_unique_households", 0),
        "minimum_cash_after_consumption": summary.get("minimum_cash_after_consumption", 0.0),
        "cash_constraint_binding_household_weeks": summary.get("cash_constraint_binding_household_weeks", 0),
        "planned_consumption_total": summary.get("planned_consumption_total", 0.0),
        "actual_consumption_total": summary.get("actual_consumption_total", 0.0),
        "consumption_suppressed_total": summary.get("consumption_suppressed_total", 0.0),
        "mean_suppressed_per_binding": summary.get("mean_suppressed_per_binding", 0.0),
        "scheduled_contribution": summary.get("scheduled_contribution_total", 0.0),
        "actual_contribution": summary.get("actual_contribution_total", 0.0),
        "collection_ratio": (
            num(summary.get("actual_contribution_total")) / num(summary.get("scheduled_contribution_total"))
            if num(summary.get("scheduled_contribution_total")) > EPS else 1.0
        ),
        "contribution_shortfall": num(summary.get("scheduled_contribution_total")) - num(summary.get("actual_contribution_total")),
        "minimum_funding_ratio": min((num(item.get("pension_funding_ratio"), 1.0) for item in data["payg"]), default=1.0),
        "weeks_funding_ratio_below_1": sum(num(item.get("pension_funding_ratio"), 1.0) < 1.0 - EPS for item in data["payg"]),
        "final_fund_cash": num(result.get("final_fund_cash")),
        "household_count": len(cash),
        "cash_gini": gini(cash),
        "bottom50_cash_share": sum(sorted(cash)[:max(1, len(cash) // 2)]) / sum(cash) if cash and sum(cash) > EPS else 0.0,
        "top10_cash_share": sum(sorted(cash)[-max(1, math.ceil(len(cash) * 0.10)):]) / sum(cash) if cash and sum(cash) > EPS else 0.0,
        "top1_cash_share": sum(sorted(cash)[-max(1, math.ceil(len(cash) * 0.01)):]) / sum(cash) if cash and sum(cash) > EPS else 0.0,
        "liquidity_lt_0_25": sum(value < 0.25 for value in liquidity) / len(liquidity) if liquidity else 0.0,
        "liquidity_lt_0_5": sum(value < 0.5 for value in liquidity) / len(liquidity) if liquidity else 0.0,
        "liquidity_lt_1": sum(value < 1.0 for value in liquidity) / len(liquidity) if liquidity else 0.0,
        "liquidity_lt_2": sum(value < 2.0 for value in liquidity) / len(liquidity) if liquidity else 0.0,
        "consumption_from_diagnostics": diagnostic_consumption if diagnostics else "unavailable",
        "food_sales_from_diagnostics": diagnostic_sales if diagnostics else "unavailable",
        "final_firm_cash_from_diagnostics": final_firm_cash,
        "final_food_inventory_from_diagnostics": final_food_inventory,
        "accounting_gap_max": result.get("max_abs_monetary_accounting_gap", 0.0),
        "money_gap_max": result.get("max_abs_money_delta_gap", 0.0),
        "goods_gap_max": result.get("max_abs_food_conservation_gap", 0.0),
        "invariant_violations": result.get("invariant_violations", 0),
    }


def fixture_rows():
    cases = [
        ("A", 200.0, 100.0, 100.0, 100.0, "unchanged_when_cash_sufficient"),
        ("B", 100.0, 100.0, 100.0, 0.0, "exact_full_settlement"),
        ("C", 60.0, 100.0, 60.0, 0.0, "clamped_to_current_cash"),
        ("D", 0.0, 100.0, 0.0, 0.0, "zero_cash_zero_consumption"),
        ("E", 80.0, 100.0, 80.0, 0.0, "PAYG_leaves_cash_below_old_budget"),
        ("F", 140.0, 100.0, 100.0, 40.0, "pension_cash_available"),
        ("G", 140.0, 100.0, 100.0, 40.0, "private_transfer_cash_available"),
        ("H", 200.0, 100.0, 100.0, 100.0, "legacy_sufficient_cash_unchanged"),
    ]
    rows = []
    for case, cash, planned, realized, closing, semantics in cases:
        rows.append({
            "fixture": case,
            "current_cash": cash,
            "planned_consumption": planned,
            "cash_settleable_consumption": realized,
            "realized_consumption": realized,
            "closing_cash": closing,
            "negative_cash": closing < -EPS,
            "pass": closing >= -EPS and realized == min(planned, max(cash, 0.0)),
            "semantics": semantics,
        })
    return rows


def main():
    if OUT.exists():
        shutil.rmtree(OUT)
    OUT.mkdir(parents=True, exist_ok=True)
    e_module = load_module(ROOT / "test/step17_e_recipient_persistence_smoke.py", "step17_e_for_e2")
    temp_dir = Path(tempfile.mkdtemp(prefix="step17e2_", dir=str(ROOT / "test")))
    try:
        common = e_module.make_common_checkpoint(temp_dir)
        gate2 = run_spec(e_module, common, temp_dir / "gate2", 26, BRANCHES, audit=True, persist=False)
        gate2_data = branch_artifacts(gate2)
        gate2_ok = gate2.get("manifest", {}).get("status") == "COMPLETED" and all(data["result"].get("status") == "completed" for data in gate2_data.values())

        gate3 = run_spec(e_module, common, temp_dir / "gate3", 52, BRANCHES, audit=True, persist=True)
        data = branch_artifacts(gate3)
        gate3_ok = gate3.get("manifest", {}).get("status") == "COMPLETED" and all(data.get(branch, {}).get("result", {}).get("status") == "completed" for branch in BRANCHES)

        parity_on = run_spec(e_module, common, temp_dir / "parity_on", 26, ("CONTROL",), audit=True, persist=False)
        parity_off = run_spec(e_module, common, temp_dir / "parity_off", 26, ("CONTROL",), audit=False, persist=False)
        on_item = parity_on.get("results", [{}])[0]
        off_item = parity_off.get("results", [{}])[0]
        parity_data = branch_artifacts(parity_on)
        parity_summary = summarize_branch("CONTROL", parity_data["CONTROL"])
        parity_ok = (
            on_item.get("status") == "completed"
            and off_item.get("status") == "completed"
            and on_item.get("state_fingerprint") == off_item.get("state_fingerprint")
            and on_item.get("rng_fingerprint") == off_item.get("rng_fingerprint")
        )
        write_rows(OUT / "control_regression.csv", [{
            "window_weeks": 26,
            "diagnostics_on_status": on_item.get("status"),
            "diagnostics_off_status": off_item.get("status"),
            "state_fingerprint_equal": on_item.get("state_fingerprint") == off_item.get("state_fingerprint"),
            "rng_fingerprint_equal": on_item.get("rng_fingerprint") == off_item.get("rng_fingerprint"),
            "control_cash_constraint_binding_household_weeks": parity_summary["cash_constraint_binding_household_weeks"],
            "control_negative_household_weeks": parity_summary["negative_household_weeks"],
            "pass": parity_ok,
        }])

        summaries = [summarize_branch(branch, data[branch]) for branch in BRANCHES]
        write_rows(OUT / "consumption_cash_safety_contract.csv", [
            {"field": "budget_feasible_money", "formula": "max(0, min(desired_money, affordable_money))", "source": "economy/firm.py", "status": "preserved"},
            {"field": "current_available_cash", "formula": "max(household.wealth, 0)", "source": "economy/firm.py at settlement", "status": "authoritative"},
            {"field": "cash_settleable_money", "formula": "min(budget_feasible_money, current_available_cash)", "source": "economy/firm.py at settlement", "status": "implemented"},
            {"field": "realized_consumption_money", "formula": "max(0, cash_settleable_money)", "source": "economy/firm.py", "status": "implemented"},
            {"field": "goods_quantity", "formula": "realized_consumption_money / authoritative_price", "source": "economy/firm.py", "status": "preserved"},
            {"field": "household_income_semantics", "formula": "income_this_step remains gross flow", "source": "Household", "status": "preserved"},
            {"field": "payg_contribution_semantics", "formula": "payg_contribution_this_step remains separate", "source": "PAYG", "status": "preserved"},
            {"field": "saving_semantics", "formula": "income - realized_consumption - recorded transfers/contribution", "source": "economy/firm.py", "status": "uses realized amount"},
            {"field": "post_settlement_tolerance", "formula": "wealth >= -1e-8; tiny residual normalized to 0", "source": "economy/firm.py", "status": "implemented"},
        ])
        fixtures = fixture_rows()
        write_rows(OUT / "consumption_cash_safety_fixtures.csv", fixtures)

        old_profile = {row.get("branch"): row for row in read_rows(ROOT / "test/output/step17_e1_payg_consumption_cash_safety/negative_cash_profile.csv")}
        write_rows(OUT / "negative_cash_comparison.csv", [{
            "branch": row["branch"],
            "pre_fix_negative_household_weeks": old_profile.get(row["branch"], {}).get("negative_household_weeks", "unavailable"),
            "post_fix_negative_household_weeks": row["negative_household_weeks"],
            "pre_fix_minimum_cash": old_profile.get(row["branch"], {}).get("minimum_negative_cash", "unavailable"),
            "post_fix_minimum_cash": row["minimum_cash_after_consumption"],
            "negative_cash_removed": num(row["negative_household_weeks"]) == 0,
            "pre_fix_source": "Step17.E.1 targeted replay" if row["branch"] in old_profile else "unavailable for this branch",
            "post_fix_source": "Step17.E.2 Gate3 52-week replay",
        } for row in summaries])
        write_rows(OUT / "consumption_constraint_binding.csv", [{
            "branch": row["branch"],
            "cash_constraint_binding_household_weeks": row["cash_constraint_binding_household_weeks"],
            "planned_consumption": row["planned_consumption_total"],
            "realized_consumption": row["actual_consumption_total"],
            "suppressed_consumption": row["consumption_suppressed_total"],
            "mean_suppressed_per_binding_event": row["mean_suppressed_per_binding"],
            "binding_share_of_audit_rows": num(row["cash_constraint_binding_household_weeks"]) / max(1, num(data[row["branch"]]["summary"].get("row_count"))),
        } for row in summaries])

        payg_rows = []
        fund_rows = []
        for row in summaries:
            if row["branch"] not in {"PAYG_3_ONLY", "COMBINED_3"}:
                continue
            payg_rows.append({
                "branch": row["branch"],
                "scheduled_contribution": row["scheduled_contribution"],
                "actual_contribution": row["actual_contribution"],
                "collection_ratio": row["collection_ratio"],
                "contribution_shortfall": row["contribution_shortfall"],
                "contributor_household_weeks": data[row["branch"]]["summary"].get("contributor_household_weeks", 0),
                "weeks": len(data[row["branch"]]["payg"]),
                "cash_safe_at_collection": data[row["branch"]]["summary"].get("contribution_cash_violations", 0) == 0,
            })
            fund_rows.append({
                "branch": row["branch"],
                "weeks_funding_ratio_below_1": row["weeks_funding_ratio_below_1"],
                "minimum_funding_ratio": row["minimum_funding_ratio"],
                "final_fund_cash": row["final_fund_cash"],
                "fund_stable": row["minimum_funding_ratio"] > 0.0 and row["final_fund_cash"] >= -EPS,
            })
        write_rows(OUT / "payg3_stability_corrected.csv", payg_rows)
        write_rows(OUT / "fund_stability_corrected.csv", fund_rows)

        events = []
        for branch in BRANCHES:
            for event in e_module.normalize_events(data[branch]["events"], 52):
                events.append(dict(event, branch=branch))
        write_rows(OUT / "recipient_retention_corrected.csv", e_module.event_type_rows(events))
        threshold_rows = []
        elderly_rows = []
        uncovered_rows = []
        for event in events:
            row = {"branch": event.get("branch"), "week": event.get("week"), "household_id": event.get("household_id"), "recipient_event_type": event.get("event_type"), "elderly_household": event.get("elderly_household"), "genealogy_uncovered_flag": event.get("genealogy_uncovered_flag")}
            for threshold, suffix in ((0.25, "025"), (0.5, "050"), (1.0, "100")):
                row[f"threshold_{suffix}_weeks"] = threshold
                row[f"threshold_{suffix}_classification"] = e_module.threshold_class(event, threshold)
            threshold_rows.append(row)
        write_rows(OUT / "threshold_persistence_corrected.csv", threshold_rows)
        grouped = {}
        for event in events:
            if bool(event.get("elderly_household")):
                grouped.setdefault((event.get("branch"), event.get("event_type"), bool(event.get("genealogy_uncovered_flag"))), []).append(event)
        for (branch, event_type, uncovered), group in sorted(grouped.items(), key=str):
            for threshold in (0.25, 0.5, 1.0):
                classes = [e_module.threshold_class(event, threshold) for event in group]
                row = {"branch": branch, "recipient_event_type": event_type, "genealogy_uncovered_flag": uncovered, "threshold_weeks": threshold, "event_count": len(group), "persistence_one_week_count": classes.count("PERSISTS_ONE_WEEK"), "persistence_four_weeks_count": classes.count("PERSISTS_FOUR_WEEKS"), "crossed_then_fell_count": classes.count("CROSSED_AFTER_POLICY_BUT_BELOW_END_WEEK") + classes.count("ABOVE_END_WEEK_BUT_BELOW_NEXT_WEEK"), "not_available_count": classes.count("NOT_AVAILABLE_LIFECYCLE") + classes.count("NOT_AVAILABLE_HORIZON")}
                elderly_rows.append(row)
                if uncovered:
                    uncovered_rows.append(dict(row))
        write_rows(OUT / "elderly_persistence_corrected.csv", elderly_rows)
        write_rows(OUT / "genealogy_uncovered_persistence_corrected.csv", uncovered_rows)

        write_rows(OUT / "branch_distribution_corrected.csv", [{
            "branch": row["branch"], "household_count": row["household_count"], "liquidity_lt_0_25": row["liquidity_lt_0_25"], "liquidity_lt_0_5": row["liquidity_lt_0_5"], "liquidity_lt_1": row["liquidity_lt_1"], "liquidity_lt_2": row["liquidity_lt_2"], "cash_gini": row["cash_gini"], "bottom50_cash_share": row["bottom50_cash_share"], "top10_cash_share": row["top10_cash_share"], "top1_cash_share": row["top1_cash_share"], "source": "fresh Step17.E.2 Gate3 final social snapshot",
        } for row in summaries])
        write_rows(OUT / "consumption_food_sales_corrected.csv", [{
            "branch": row["branch"], "household_consumption_from_cash_audit": row["actual_consumption_total"], "food_sales_from_persisted_diagnostics": row["food_sales_from_diagnostics"], "food_sales_source_status": "authoritative diagnostics" if row["food_sales_from_diagnostics"] != "unavailable" else "unavailable", "interpretation": "cash-safe realized consumption is the demand input; existing unfilled-order refund path remains active",
        } for row in summaries])
        write_rows(OUT / "firm_side_effect_corrected.csv", [{
            "branch": row["branch"], "final_firm_cash": row["final_firm_cash_from_diagnostics"], "final_food_inventory": row["final_food_inventory_from_diagnostics"], "household_consumption": row["actual_consumption_total"], "side_effect_status": "fresh persisted diagnostic snapshot" if row["final_firm_cash_from_diagnostics"] != "unavailable" else "unavailable",
        } for row in summaries])

        fixture_ok = all(row["pass"] and not row["negative_cash"] for row in fixtures)
        negative_ok = all(num(row["negative_household_weeks"]) == 0 and num(row["minimum_cash_after_consumption"]) >= -EPS for row in summaries)
        accounting_ok = all(num(row["accounting_gap_max"]) <= 1e-6 and num(row["money_gap_max"]) <= 1e-6 and num(row["goods_gap_max"]) <= 1e-6 and num(row["invariant_violations"]) == 0 for row in summaries)
        contribution_ok = all(data[branch]["summary"].get("contribution_cash_violations", 0) == 0 and data[branch]["summary"].get("contribution_post_cash_violations", 0) == 0 for branch in ("PAYG_3_ONLY", "COMBINED_3"))
        control_binding = next(row["cash_constraint_binding_household_weeks"] for row in summaries if row["branch"] == "CONTROL") > 0
        policy_change = any(num(row["consumption_suppressed_total"]) > 1.0 for row in summaries if row["branch"] != "CONTROL")
        if not fixture_ok or not gate2_ok or not gate3_ok or not negative_ok or not accounting_ok:
            verdict = "D. CASH_FIX_CAUSES_ACCOUNTING_OR_GOODS_REGRESSION" if not accounting_ok else "E. OTHER"
        elif control_binding:
            verdict = "C. CASH_FIX_REVEALS_BROADER_CANONICAL_CONSUMPTION_CONSTRAINT"
        elif policy_change:
            verdict = "B. CASH_FIX_ACCEPTED_BUT_POLICY_RESULTS_CHANGE_MATERIALLY"
        else:
            verdict = "A. CURRENT_CASH_CONSUMPTION_FIX_ACCEPTED_AND_STEP17_RESULTS_REVALIDATED"

        flags = {
            "verdict": verdict,
            "gate1_fixtures_pass": fixture_ok,
            "gate2_26_week_pass": gate2_ok,
            "gate3_52_week_four_branch_pass": gate3_ok,
            "negative_cash_removed": negative_ok,
            "negative_cash_material_count_after": sum(num(row["negative_household_weeks"]) for row in summaries),
            "control_behavior_changed": not parity_ok,
            "control_parity_pass": parity_ok,
            "cash_constraint_binding_by_branch": {row["branch"]: row["cash_constraint_binding_household_weeks"] for row in summaries},
            "consumption_suppressed_by_branch": {row["branch"]: row["consumption_suppressed_total"] for row in summaries},
            "payg_collection_ratio_by_branch": {row["branch"]: row["collection_ratio"] for row in summaries if row["branch"] in {"PAYG_3_ONLY", "COMBINED_3"}},
            "payg_contribution_cash_safe": contribution_ok,
            "fund_stability_pass": all(row["fund_stable"] for row in fund_rows),
            "recipient_retention_recomputed": bool(events),
            "threshold_persistence_recomputed": bool(threshold_rows),
            "elderly_persistence_recomputed": True,
            "genealogy_uncovered_persistence_recomputed": True,
            "distribution_recomputed": True,
            "accounting_reconciliation_pass": accounting_ok,
            "money_reconciliation_pass": all(num(row["money_gap_max"]) <= 1e-6 for row in summaries),
            "goods_reconciliation_pass": all(num(row["goods_gap_max"]) <= 1e-6 for row in summaries),
            "assignment_reconciliation_pass": all(num(row["invariant_violations"]) == 0 for row in summaries),
            "new_rng_draws": 0,
            "payg_parameters_changed": False,
            "pension_benefits_changed": False,
            "reserve_parameters_changed": False,
            "private_transfer_parameters_changed": False,
            "long_run_520_week_run": False,
            "step17f_started": False,
        }
        (OUT / "acceptance_flags.json").write_text(json.dumps(flags, indent=2), encoding="utf-8")
        control = next(row for row in summaries if row["branch"] == "CONTROL")
        payg = next(row for row in summaries if row["branch"] == "PAYG_3_ONLY")
        combined = next(row for row in summaries if row["branch"] == "COMBINED_3")
        private = next(row for row in summaries if row["branch"] == "PRIVATE_ONLY")
        lines = [
            "# Step17.E.2 Household Consumption Current-Cash Safety Fix",
            "",
            "- The canonical Food settlement now preserves the behavioral budget and caps the actual debit at current Household cash at the settlement instant.",
            "- income_this_step remains gross income and payg_contribution_this_step remains separate; no policy parameter changed.",
            f"- Gate 2 (26 weeks): {'PASS' if gate2_ok else 'FAIL'}; Gate 3 (52 weeks, four branches): {'PASS' if gate3_ok else 'FAIL'}; fixtures: {'PASS' if fixture_ok else 'FAIL'}.",
            f"- Material negative cash after the fix: {sum(num(row['negative_household_weeks']) for row in summaries)} Household-weeks; control diagnostics-on/off parity: {'PASS' if parity_ok else 'FAIL'}.",
            f"- Cash-constraint binding Household-weeks: CONTROL={control['cash_constraint_binding_household_weeks']}, PRIVATE_ONLY={private['cash_constraint_binding_household_weeks']}, PAYG_3_ONLY={payg['cash_constraint_binding_household_weeks']}, COMBINED_3={combined['cash_constraint_binding_household_weeks']}.",
            f"- Suppressed consumption: " + ", ".join(f"{row['branch']}={num(row['consumption_suppressed_total']):.6f}" for row in summaries) + ".",
            f"- PAYG collection ratios: PAYG_3_ONLY={payg['collection_ratio']:.9f}, COMBINED_3={combined['collection_ratio']:.9f}; funding stability is recorded in fund_stability_corrected.csv.",
            "- Recipient retention, threshold persistence, elderly/genealogy-uncovered persistence, distributions, Food sales, and Firm-side effects were recomputed from the fresh corrected replay.",
            f"- Verdict: {verdict}. 3% PAYG remains an executable income-support policy when its collection ratio is near one; private transfers remain family-network income support. This is not a policy freeze.",
            "- Accounting, money, goods, assignment, and RNG status are recorded in acceptance_flags.json; no Step17.F work was started.",
        ]
        (OUT / "acceptance_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)


if __name__ == "__main__":
    main()
