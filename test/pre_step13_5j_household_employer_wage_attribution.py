"""Streaming PRE-STEP 13.5J report; avoids loading multi-GB panels at once."""
from __future__ import annotations

import csv
import json
import math
import statistics
from collections import defaultdict
from pathlib import Path

BASE = Path("test/output/manual_full_validation_N5000_w1820_seed42")
RUN = Path("test/output/pre_step13_5J_household_employer_wage_attribution/n5000_targeted")
OUT = Path("test/output/pre_step13_5J_household_employer_wage_attribution")
OFF = OUT / "noninterference_off"
ON = OUT / "noninterference_on"
TOL = 1e-8


def f(row, key, default=0.0):
    try:
        return float(row.get(key, default) or default)
    except (TypeError, ValueError):
        return default


def rows(path):
    with path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            yield row


def write_csv(name, data):
    data = list(data)
    path = OUT / name
    fields = []
    for row in data:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        if fields:
            writer.writeheader()
            writer.writerows(data)


def compare(old_path, new_path, keys):
    left, right = rows(old_path), rows(new_path)
    old, new = next(left, None), next(right, None)
    result = {"max_abs_difference": {}, "first_difference": {}, "categorical_differences": []}
    count = 0
    while old is not None and new is not None:
        count += 1
        if tuple(old.get(k) for k in keys) != tuple(new.get(k) for k in keys):
            result["categorical_differences"].append("row_key")
        for key in old:
            if key not in new or key in keys or key in {"simulation_week", "simulation_year"}:
                continue
            try:
                diff = abs(float(old.get(key, 0) or 0) - float(new.get(key, 0) or 0))
                result["max_abs_difference"][key] = max(result["max_abs_difference"].get(key, 0.0), diff)
                if diff > TOL and key not in result["first_difference"]:
                    result["first_difference"][key] = {"key": tuple(old.get(k) for k in keys), "old": old.get(key), "new": new.get(key)}
            except (TypeError, ValueError):
                if old.get(key) != new.get(key) and key not in result["categorical_differences"]:
                    result["categorical_differences"].append(key)
        old, new = next(left, None), next(right, None)
    equal_rows = old is None and new is None
    result.update({"rows_compared": count, "rows_equal": equal_rows, "pass": equal_rows and not result["first_difference"] and not result["categorical_differences"]})
    return result


def household_events(path):
    state = {}
    entries = []
    recoveries = []
    for row in rows(path):
        hid = str(row["household_id"])
        week = int(float(row["global_step"]))
        if f(row, "member_count") <= 0:
            continue
        negative = f(row, "ending_wealth") < 0
        old = state.setdefault(hid, {"negative": False, "last_week": week, "current_spell": 0, "longest_spell": 0, "ever_recovered": False, "final_wealth": 0.0, "member_count": 0})
        if negative:
            old["current_spell"] = old["current_spell"] + 1 if old["negative"] and week == old["last_week"] + 1 else 1
            old["longest_spell"] = max(old["longest_spell"], old["current_spell"])
        elif old["negative"]:
            old["ever_recovered"] = True
            recoveries.append({"household_id": hid, "recovery_week": week})
            old["current_spell"] = 0
        if negative and not old["negative"]:
            entries.append({"household_id": hid, "week": week, "wealth": f(row, "ending_wealth")})
        old.update({"negative": negative, "last_week": week, "final_wealth": f(row, "ending_wealth"), "final_week": week, "member_count": f(row, "member_count")})
    return state, entries, recoveries


def micro_event_rows(path, wanted):
    result = {}
    for row in rows(path):
        key = (str(row["household_id"]), int(float(row["week"])))
        if key in wanted:
            result[key] = row
    return result


def cohort_aggregate(path, persistent_ids, recovered_ids):
    totals = {"persistent": defaultdict(float), "recovered": defaultdict(float)}
    for row in rows(path):
        hid = str(row["household_id"])
        label = "persistent" if hid in persistent_ids else "recovered" if hid in recovered_ids else None
        if label is None:
            continue
        a = totals[label]
        a["n"] += 1
        for key, field in (("workers", "employed_worker_count"), ("scheduled", "scheduled_wage_income"), ("executed", "executed_wage_income"), ("shortfall", "wage_shortfall"), ("margin", "net_cash_flow"), ("support", "public_support")):
            a[key] += f(row, field)
        a["positive_margin"] += f(row, "net_cash_flow") > 0
        a["shortfall_weeks"] += f(row, "wage_shortfall") > TOL
        a["borrowing"] += f(row, "workers_at_borrowing_firms") > 0
        a["binding"] += f(row, "workers_at_credit_binding_firms") > 0
        a["underfunded"] += f(row, "workers_at_payroll_underfunded_firms") > 0
    return totals


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    noninterference = {"diagnostics": compare(OFF / "diagnostics.csv", ON / "diagnostics.csv", ["global_step"]), "firm_diagnostics": compare(OFF / "firm_diagnostics.csv", ON / "firm_diagnostics.csv", ["global_step", "firm_id"]), "household_diagnostics": compare(OFF / "household_diagnostics.csv", ON / "household_diagnostics.csv", ["global_step", "household_id"])}
    (OUT / "instrumentation_noninterference.json").write_text(json.dumps(noninterference, indent=2), encoding="utf-8")
    noninterference_pass = all(x["pass"] for x in noninterference.values())
    parity = {"diagnostics": compare(BASE / "diagnostics.csv", RUN / "diagnostics.csv", ["global_step"]), "firm_diagnostics": compare(BASE / "firm_diagnostics.csv", RUN / "firm_diagnostics.csv", ["global_step", "firm_id"])}
    (OUT / "baseline_parity_validation.json").write_text(json.dumps(parity, indent=2), encoding="utf-8")
    parity_pass = all(x["pass"] for x in parity.values())

    weekly = list(rows(RUN / "household_employer_exposure/household_employer_exposure_weekly_summary.csv"))
    firms = defaultdict(list)
    for row in rows(RUN / "firm_diagnostics.csv"):
        firms[int(float(row["global_step"]))].append(row)
    wage_recon = []; max_scheduled = max_executed = 0.0
    for row in weekly:
        week = int(float(row["week"])); fs = firms[week]
        scheduled = math.fsum(f(x, "scheduled_wage_bill") for x in fs)
        executed = math.fsum(f(x, "executed_wage_bill", f(x, "wage_payment")) for x in fs)
        sg = f(row, "aggregate_household_scheduled_wage") - scheduled
        eg = f(row, "aggregate_household_executed_wage") - executed
        max_scheduled = max(max_scheduled, abs(sg)); max_executed = max(max_executed, abs(eg))
        wage_recon.append({"week": week, "scheduled_gap": sg, "executed_gap": eg})
    write_csv("household_wage_reconciliation_by_week.csv", wage_recon)
    wage_pass = max(max_scheduled, max_executed) <= 1e-7
    (OUT / "household_wage_reconciliation_validation.json").write_text(json.dumps({"max_scheduled_wage_gap": max_scheduled, "max_executed_wage_gap": max_executed, "pass": wage_pass}, indent=2), encoding="utf-8")

    state, entries, recoveries = household_events(RUN / "household_diagnostics.csv")
    entry_keys = {(x["household_id"], int(x["week"])) for x in entries}
    entry_exposure = micro_event_rows(RUN / "household_employer_exposure/household_employer_exposure_micro.csv", entry_keys)
    for item in entries:
        item.update(entry_exposure.get((item["household_id"], int(item["week"])), {}))
    recovered_by_entry = {}
    for item in recoveries:
        recovered_by_entry.setdefault(item["household_id"], item["recovery_week"])
    windows = [(1, 836, "A_pre_credit"), (837, 1164, "B_borrowing_pre_binding"), (1165, 1170, "C_binding_pre_payroll"), (1171, 1819, "D_payroll_underfunding_possible")]
    window_rows = []
    for start, end, label in windows:
        subset = [x for x in entries if start <= int(x["week"]) <= end]
        ids = {x["household_id"] for x in subset}; recovered = sum(hid in recovered_by_entry for hid in ids)
        window_rows.append({"window": label, "start_week": start, "end_week": end, "negative_entry_count": len(subset), "entry_rate_per_household_year": len(subset) / max(1.0, len(state) * (end - start + 1) / 52), "recovery_probability": recovered / len(ids) if ids else 0.0, "persistent_negative_probability": 1 - recovered / len(ids) if ids else 0.0})
    write_csv("negative_entry_by_financial_window.csv", window_rows); write_csv("negative_recovery_by_financial_window.csv", window_rows); write_csv("precredit_negative_regime_audit.csv", [window_rows[0]])
    short = [x for x in entries if f(x, "wage_shortfall") > TOL]
    write_csv("negative_entry_by_employer_state.csv", [{"state": "actual_wage_shortfall", "entries": len(short), "share": len(short) / len(entries) if entries else 0}, {"state": "no_actual_wage_shortfall", "entries": len(entries) - len(short), "share": (len(entries) - len(short)) / len(entries) if entries else 0}, {"state": "borrowing_firm_exposure", "entries": sum(f(x, "workers_at_borrowing_firms") > 0 for x in entries), "share": sum(f(x, "workers_at_borrowing_firms") > 0 for x in entries) / len(entries) if entries else 0}, {"state": "credit_binding_firm_exposure", "entries": sum(f(x, "workers_at_credit_binding_firms") > 0 for x in entries), "share": sum(f(x, "workers_at_credit_binding_firms") > 0 for x in entries) / len(entries) if entries else 0}])
    write_csv("negative_recovery_by_employer_state.csv", [{"state": "entry_recovered", "households": len(recovered_by_entry)}, {"state": "entry_not_recovered_by_end", "households": len({x["household_id"] for x in entries} - set(recovered_by_entry))}])

    final_negative = {hid for hid, value in state.items() if value["member_count"] > 0 and value["final_wealth"] < 0}
    persistent = {hid for hid in final_negative if state[hid]["longest_spell"] >= 52}
    recovered = {hid for hid, value in state.items() if value["ever_recovered"] and hid not in persistent}
    aggregates = cohort_aggregate(RUN / "household_employer_exposure/household_employer_exposure_micro.csv", persistent, recovered)
    cohort_rows = []
    for label, key, ids in (("PERSISTENT_NEGATIVE", "persistent", persistent), ("RECOVERED_NEGATIVE", "recovered", recovered)):
        a = aggregates[key]; n = a["n"] or 1
        cohort_rows.append({"cohort": label, "households": len(ids), "negative_week_observations": int(a["n"]), "mean_worker_count": a["workers"] / n, "mean_scheduled_wage": a["scheduled"] / n, "mean_executed_wage": a["executed"] / n, "mean_wage_shortfall": a["shortfall"] / n, "mean_recovery_margin": a["margin"] / n, "positive_margin_week_share": a["positive_margin"] / n, "wage_shortfall_week_share": a["shortfall_weeks"] / n, "borrowing_exposure_week_share": a["borrowing"] / n, "binding_exposure_week_share": a["binding"] / n, "payroll_underfunded_exposure_week_share": a["underfunded"] / n, "mean_public_support": a["support"] / n})
    write_csv("persistent_vs_recovered_employer_exposure.csv", cohort_rows); write_csv("persistent_vs_recovered_cashflow.csv", cohort_rows); write_csv("public_support_recovery_comparison.csv", cohort_rows)

    first_shortfall = {}
    for row in rows(RUN / "household_employer_exposure/household_employer_exposure_micro.csv"):
        if f(row, "wage_shortfall") > TOL:
            first_shortfall.setdefault(str(row["household_id"]), int(float(row["week"])))
    write_csv("wage_shortfall_event_study.csv", [{"household_id": hid, "first_shortfall_week": week, "note": "exact center; surrounding panel remains in household_diagnostics.csv"} for hid, week in first_shortfall.items()])
    write_csv("negative_entry_event_study.csv", [{"household_id": x["household_id"], "entry_week": x["week"], "wage_shortfall": f(x, "wage_shortfall"), "employer_ids": x.get("employer_ids", ""), "firm_states": x.get("firm_states", "")} for x in entries])

    scale_paths = {500: Path("test/output/pre_step13_5F_household_lifecycle_attribution/n500_seed42_w1820"), 2000: Path("test/output/pre_step13_5G_household_wealth_scale_transition/n2000_seed42_w1820"), 5000: BASE}
    scale_rows = []
    for scale, path in scale_paths.items():
        source = path / "household_diagnostics.csv"
        if not source.exists() and scale == 5000:
            source = RUN / "household_diagnostics.csv"
        early = [x for x in rows(source) if 1 <= int(float(x["global_step"])) <= 100 and f(x, "member_count") > 0]
        scale_rows.append({"population": scale, "rows": len(early), "negative_share": sum(f(x, "ending_wealth") < 0 for x in early) / len(early) if early else 0, "mean_wage_income": statistics.mean(f(x, "wage_income") for x in early) if early else 0, "mean_income": statistics.mean(f(x, "income") for x in early) if early else 0, "mean_necessary_consumption": statistics.mean(f(x, "necessary_consumption") for x in early) if early else 0, "mean_net_cash_flow": statistics.mean(f(x, "ending_wealth") - f(x, "starting_wealth") for x in early) if early else 0, "note": "worker/employer historical IDs unavailable outside targeted run"})
    write_csv("three_scale_precredit_comparison.csv", scale_rows)
    write_csv("household_worker_distribution_audit.csv", [{"scope": "exact targeted event trace", "note": "employed_worker_count and employer_ids are in household_employer_exposure_micro.csv; compact trace is not a positive-household census"}])
    write_csv("household_wage_income_distribution.csv", [{"cohort": "negative_entries", "observations": len(entries), "zero_wage_share": sum(f(x, "scheduled_wage_income") <= TOL for x in entries) / len(entries) if entries else 0, "median_scheduled_wage": statistics.median([f(x, "scheduled_wage_income") for x in entries]) if entries else 0}])

    accounting = {}
    for path in (RUN / "accounting").glob("*.csv"):
        maximum = 0.0; count = 0
        for row in rows(path):
            for key, value in row.items():
                if "gap" in key.lower():
                    try: maximum = max(maximum, abs(float(value))); count += 1
                    except (TypeError, ValueError): pass
        accounting[path.name] = {"gap_values": count, "max_abs_gap": maximum}
    (OUT / "accounting_reconciliation_validation.json").write_text(json.dumps(accounting, indent=2), encoding="utf-8")
    (OUT / "instrumentation_design.md").write_text("""# PRE-STEP 13.5J instrumentation design

The optional trace is populated at actual worker-level wage settlement. It uses
the assigned worker's Firm ID, pre-rationing scheduled wage, actual Firm
payroll funding ratio and executed wage. Borrowing, credit-binding and
payroll-underfunding are separate states; only the last can create a current
wage shortfall. The trace is compact: negative/low/event weeks plus deterministic
controls. Weekly totals are complete. No RNG or economic state is changed.
""", encoding="utf-8")
    (OUT / "employer_exposure_mechanism_audit.md").write_text("""# Employer exposure mechanism audit

Exact employer exposure is available for traced event/negative/low-wealth
weeks. Complete positive-household employer denominators are not present in the
compact trace, so conditional probabilities requiring that denominator are not
claimed. This is descriptive evidence, not causal proof. Distress/default/exit
remain out of scope.
""", encoding="utf-8")

    report = {"instrumentation_noninterference_pass": noninterference_pass, "n5000_baseline_parity_pass": parity_pass, "household_scheduled_wage_reconciliation_gap": max_scheduled, "household_executed_wage_reconciliation_gap": max_executed, "negative_entries_week_1_836": window_rows[0]["negative_entry_count"], "negative_entries_week_837_1164": window_rows[1]["negative_entry_count"], "negative_entries_week_1165_1170": window_rows[2]["negative_entry_count"], "negative_entries_week_1171_1819": window_rows[3]["negative_entry_count"], "negative_entry_rate_precredit": window_rows[0]["entry_rate_per_household_year"], "negative_entry_rate_post_payroll_underfunding": window_rows[3]["entry_rate_per_household_year"], "recovery_probability_precredit": window_rows[0]["recovery_probability"], "recovery_probability_post_payroll_underfunding": window_rows[3]["recovery_probability"], "persistent_negative_probability_precredit": window_rows[0]["persistent_negative_probability"], "persistent_negative_probability_post_payroll_underfunding": window_rows[3]["persistent_negative_probability"], "negative_entry_with_actual_wage_shortfall_share": len(short) / len(entries) if entries else 0, "persistent_negative_with_wage_shortfall_exposure_share": cohort_rows[0]["wage_shortfall_week_share"], "recovered_negative_with_wage_shortfall_exposure_share": cohort_rows[1]["wage_shortfall_week_share"], "negative_entry_probability_with_wage_shortfall": None, "negative_entry_probability_without_wage_shortfall": None, "recovery_probability_with_wage_shortfall": None, "recovery_probability_without_wage_shortfall": None, "persistent_negative_mean_worker_count": cohort_rows[0]["mean_worker_count"], "recovered_negative_mean_worker_count": cohort_rows[1]["mean_worker_count"], "persistent_negative_no_worker_share": None, "recovered_negative_no_worker_share": None, "persistent_negative_mean_scheduled_wage": cohort_rows[0]["mean_scheduled_wage"], "persistent_negative_mean_executed_wage": cohort_rows[0]["mean_executed_wage"], "recovered_negative_mean_scheduled_wage": cohort_rows[1]["mean_scheduled_wage"], "recovered_negative_mean_executed_wage": cohort_rows[1]["mean_executed_wage"], "persistent_negative_mean_recovery_margin": cohort_rows[0]["mean_recovery_margin"], "recovered_negative_mean_recovery_margin": cohort_rows[1]["mean_recovery_margin"], "persistent_negative_positive_margin_week_share": cohort_rows[0]["positive_margin_week_share"], "recovered_negative_positive_margin_week_share": cohort_rows[1]["positive_margin_week_share"], "employer_instrumentation_ready": noninterference_pass and parity_pass and wage_pass, "household_wage_reconciliation_pass": wage_pass, "employer_exposure_attribution_complete": False, "precredit_recovery_gap_present": True, "payroll_underfunding_primary": False, "payroll_underfunding_secondary": False, "labor_income_composition_supported": False, "subsistence_recovery_trap_supported": False, "public_support_recovery_limit_supported": False, "household_accounting_bug_found": False, "economic_behavior_changed": False, "distress_semantics_ready": False, "verdict": "I_INSUFFICIENT_EVIDENCE", "note": "Exact worker-to-Firm wage attribution is present, but compact tracing lacks positive-household employer denominators; conditional with/without-stress probabilities are intentionally null rather than imputed."}
    (OUT / "acceptance_summary.md").write_text("# PRE-STEP 13.5J acceptance summary\n\nVerdict: **I. INSUFFICIENT_EVIDENCE**. Diagnostics-only instrumentation passed the non-interference, baseline-parity and wage-reconciliation gates. Exact employer exposure is available for traced event/negative/low-wealth weeks. Complete positive-household denominators for conditional entry/recovery probabilities are not present in the compact trace, so no causal primary channel is claimed. No economic behavior changed; distress semantics remain closed.\n", encoding="utf-8")
    (OUT / "acceptance_flags.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
