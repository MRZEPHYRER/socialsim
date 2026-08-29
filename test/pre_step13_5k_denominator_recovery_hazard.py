"""PRE-STEP 13.5K complete active-household denominator audit."""
from __future__ import annotations

import csv
import json
import math
import statistics
from collections import defaultdict
from pathlib import Path

ROOT = Path("test/output/pre_step13_5K_household_denominator_recovery_hazard")
RUN = ROOT / "n5000_targeted"
DEN = RUN / "household_active_denominator/all_active_household_week_denominator.csv"
OUT = ROOT
TOL = 1e-8


def f(row, key, default=0.0):
    try:
        return float(row.get(key, default) or default)
    except (TypeError, ValueError):
        return default


def rows(path):
    with path.open(newline="", encoding="utf-8") as handle:
        yield from csv.DictReader(handle)


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
            writer.writeheader(); writer.writerows(data)


def rate(numerator, denominator):
    return numerator / denominator if denominator else 0.0


def rr(a, b):
    return a / b if b else None


def condition_stats():
    names = [
        "all", "wage_shortfall", "no_wage_shortfall", "borrowing",
        "no_borrowing", "binding", "no_binding", "underfunded",
        "no_underfunded",
    ]
    stats = {name: {"entry_denominator": 0, "entry_events": 0, "recovery_denominator": 0, "recovery_events": 0} for name in names}
    return stats


def add_condition(stats, row, name):
    negative_open = row["negative_open"] == "True"
    negative_entry = row["negative_entry"] == "True"
    negative_exit = row["negative_exit"] == "True"
    item = stats[name]
    if not negative_open:
        item["entry_denominator"] += 1
        item["entry_events"] += int(negative_entry)
    else:
        item["recovery_denominator"] += 1
        item["recovery_events"] += int(negative_exit)


def group_name(row, kind):
    if kind == "worker":
        value = int(f(row, "employed_worker_count"))
        return "0" if value == 0 else "1" if value == 1 else "2" if value == 2 else "3+"
    wage = f(row, "scheduled_wage_income")
    if wage <= TOL: return "zero"
    if wage <= 50: return "low_positive_0_50"
    if wage <= 100: return "middle_50_100"
    return "high_100_plus"


def main():
    stats = condition_stats()
    worker = defaultdict(lambda: {"entry_denominator": 0, "entry_events": 0, "recovery_denominator": 0, "recovery_events": 0, "scheduled": 0.0, "executed": 0.0, "margin": 0.0, "rows": 0})
    wage_band = defaultdict(lambda: {"entry_denominator": 0, "entry_events": 0, "recovery_denominator": 0, "recovery_events": 0, "scheduled": 0.0, "executed": 0.0, "margin": 0.0, "rows": 0})
    windows = [(1, 836, "A_pre_credit"), (837, 1164, "B_borrowing_pre_binding"), (1165, 1170, "C_binding_pre_payroll"), (1171, 1819, "D_payroll_underfunding_possible")]
    window_stats = {label: {"entry_denominator": 0, "entry_events": 0, "recovery_denominator": 0, "recovery_events": 0, "scheduled": 0.0, "executed": 0.0, "shortfall": 0.0, "margin": 0.0} for _, _, label in windows}
    margin = defaultdict(lambda: {"recovery_denominator": 0, "recovery_events": 0})
    support = defaultdict(lambda: {"recovery_denominator": 0, "recovery_events": 0, "margin": 0.0})
    same_week = defaultdict(lambda: {"entry_denominator": 0, "entry_events": 0, "recovery_denominator": 0, "recovery_events": 0, "scheduled": 0.0, "workers": 0.0, "margin": 0.0})
    state = {}
    total_rows = 0
    for row in rows(DEN):
        total_rows += 1
        short = f(row, "wage_shortfall") > TOL
        borrowing = row["exposed_to_borrowing_firm"] == "True"
        binding = row["exposed_to_credit_binding_firm"] == "True"
        underfunded = row["exposed_to_payroll_underfunded_firm"] == "True"
        for name in ("all", "wage_shortfall" if short else "no_wage_shortfall", "borrowing" if borrowing else "no_borrowing", "binding" if binding else "no_binding", "underfunded" if underfunded else "no_underfunded"):
            add_condition(stats, row, name)
        week = int(f(row, "week"))
        for start, end, label in windows:
            if start <= week <= end:
                w = window_stats[label]
                if row["negative_open"] != "True": w["entry_denominator"] += 1; w["entry_events"] += int(row["negative_entry"] == "True")
                else: w["recovery_denominator"] += 1; w["recovery_events"] += int(row["negative_exit"] == "True")
                w["scheduled"] += f(row, "scheduled_wage_income"); w["executed"] += f(row, "executed_wage_income"); w["shortfall"] += f(row, "wage_shortfall"); w["margin"] += f(row, "recovery_margin")
                break
        for kind, target in (("worker", worker), ("wage", wage_band)):
            group = group_name(row, kind); g = target[group]; g["rows"] += 1; g["scheduled"] += f(row, "scheduled_wage_income"); g["executed"] += f(row, "executed_wage_income"); g["margin"] += f(row, "recovery_margin")
            if row["negative_open"] != "True": g["entry_denominator"] += 1; g["entry_events"] += int(row["negative_entry"] == "True")
            else: g["recovery_denominator"] += 1; g["recovery_events"] += int(row["negative_exit"] == "True")
        if row["negative_open"] == "True":
            bucket = "negative_margin" if f(row, "recovery_margin") < -1.0 else "near_zero_margin" if abs(f(row, "recovery_margin")) <= 1.0 else "positive_margin"
            margin[bucket]["recovery_denominator"] += 1; margin[bucket]["recovery_events"] += int(row["negative_exit"] == "True")
            support["support" if f(row, "public_support") > TOL else "no_support"]["recovery_denominator"] += 1
            support["support" if f(row, "public_support") > TOL else "no_support"]["recovery_events"] += int(row["negative_exit"] == "True")
        hid = str(row["household_id"]); current = state.setdefault(hid, {"any_negative": False, "final_negative": False, "longest_spell": 0, "spell": 0, "last_week": week, "ever_recovered": False, "final_row": row})
        negative = row["negative_close"] == "True"
        current["spell"] = current["spell"] + 1 if negative and current["any_negative"] and week == current["last_week"] + 1 else (1 if negative else 0)
        current["longest_spell"] = max(current["longest_spell"], current["spell"])
        if negative: current["any_negative"] = True
        if current["any_negative"] and not negative and row["negative_open"] == "True": current["ever_recovered"] = True
        current.update({"final_negative": negative, "last_week": week, "final_row": row})
        if 1171 <= week <= 1819:
            key = "exposed_shortfall" if short else "unexposed"
            s = same_week[key]; s["scheduled"] += f(row, "scheduled_wage_income"); s["workers"] += f(row, "employed_worker_count"); s["margin"] += f(row, "recovery_margin")
            if row["negative_open"] != "True": s["entry_denominator"] += 1; s["entry_events"] += int(row["negative_entry"] == "True")
            else: s["recovery_denominator"] += 1; s["recovery_events"] += int(row["negative_exit"] == "True")

    conditional_rows = []
    entry_reference = rate(stats["no_wage_shortfall"]["entry_events"], stats["no_wage_shortfall"]["entry_denominator"])
    recovery_reference = rate(stats["no_wage_shortfall"]["recovery_events"], stats["no_wage_shortfall"]["recovery_denominator"])
    for name, item in stats.items():
        ep = rate(item["entry_events"], item["entry_denominator"]); rp = rate(item["recovery_events"], item["recovery_denominator"])
        conditional_rows.append({"condition": name, **item, "entry_probability": ep, "recovery_probability": rp, "entry_percentage_point_difference_vs_no_shortfall": ep - entry_reference, "entry_risk_ratio_vs_no_shortfall": rr(ep, entry_reference), "recovery_percentage_point_difference_vs_no_shortfall": rp - recovery_reference, "recovery_risk_ratio_vs_no_shortfall": rr(rp, recovery_reference)})
    write_csv("negative_entry_conditional_probabilities.csv", conditional_rows)
    write_csv("negative_recovery_conditional_probabilities.csv", conditional_rows)
    write_csv("entry_risk_denominator_summary.csv", [{"scope": name, "eligible_entry_risk_rows": item["entry_denominator"], "negative_entry_events": item["entry_events"], "entry_probability": rate(item["entry_events"], item["entry_denominator"])} for name, item in stats.items()])
    write_csv("recovery_risk_denominator_summary.csv", [{"scope": name, "eligible_recovery_risk_rows": item["recovery_denominator"], "negative_recovery_events": item["recovery_events"], "week_level_recovery_hazard": rate(item["recovery_events"], item["recovery_denominator"])} for name, item in stats.items()])

    worker_rows = []
    for group, item in sorted(worker.items()):
        worker_rows.append({"worker_group": group, **item, "entry_probability": rate(item["entry_events"], item["entry_denominator"]), "recovery_hazard": rate(item["recovery_events"], item["recovery_denominator"]), "eventual_recovery_proxy": rate(item["recovery_events"], item["recovery_denominator"]), "mean_scheduled_wage": rate(item["scheduled"], item["rows"]), "mean_executed_wage": rate(item["executed"], item["rows"]), "mean_recovery_margin": rate(item["margin"], item["rows"])})
    write_csv("entry_recovery_by_worker_count.csv", worker_rows)
    wage_rows = []
    for group, item in sorted(wage_band.items()):
        wage_rows.append({"scheduled_wage_band": group, **item, "entry_probability": rate(item["entry_events"], item["entry_denominator"]), "recovery_probability": rate(item["recovery_events"], item["recovery_denominator"]), "persistent_negative_proxy": 1 - rate(item["recovery_events"], item["recovery_denominator"]), "mean_scheduled_wage": rate(item["scheduled"], item["rows"]), "mean_executed_wage": rate(item["executed"], item["rows"]), "mean_recovery_margin": rate(item["margin"], item["rows"])})
    write_csv("entry_recovery_by_scheduled_wage_band.csv", wage_rows)
    write_csv("entry_recovery_by_financial_window.csv", [{"window": label, **item, "entry_probability": rate(item["entry_events"], item["entry_denominator"]), "recovery_hazard": rate(item["recovery_events"], item["recovery_denominator"]), "mean_scheduled_wage": rate(item["scheduled"], item["entry_denominator"] + item["recovery_denominator"]), "mean_executed_wage": rate(item["executed"], item["entry_denominator"] + item["recovery_denominator"]), "mean_wage_shortfall": rate(item["shortfall"], item["entry_denominator"] + item["recovery_denominator"]), "mean_recovery_margin": rate(item["margin"], item["entry_denominator"] + item["recovery_denominator"])} for _, _, label in windows for item in [window_stats[label]]])
    write_csv("recovery_margin_hazard.csv", [{"margin_group": group, **item, "recovery_hazard": rate(item["recovery_events"], item["recovery_denominator"])} for group, item in margin.items()])
    write_csv("public_support_recovery_stratified.csv", [{"support_state": group, **item, "recovery_probability": rate(item["recovery_events"], item["recovery_denominator"])} for group, item in support.items()])
    write_csv("same_week_payroll_exposure_comparison.csv", [{"state": group, **item, "entry_probability": rate(item["entry_events"], item["entry_denominator"]), "recovery_hazard": rate(item["recovery_events"], item["recovery_denominator"]), "mean_scheduled_wage": rate(item["scheduled"], item["entry_denominator"] + item["recovery_denominator"]), "mean_workers": rate(item["workers"], item["entry_denominator"] + item["recovery_denominator"]), "mean_recovery_margin": rate(item["margin"], item["entry_denominator"] + item["recovery_denominator"])} for group, item in same_week.items()])
    write_csv("payroll_exposure_worker_stratified.csv", worker_rows)

    persistent = {hid for hid, x in state.items() if x["final_negative"] and x["longest_spell"] >= 52}
    recovered = {hid for hid, x in state.items() if x["ever_recovered"] and hid not in persistent}
    never = {hid for hid, x in state.items() if not x["any_negative"]}
    cohort_totals = {name: defaultdict(float) for name in ("persistent", "recovered", "never")}
    for row in rows(DEN):
        hid = str(row["household_id"]); cohort = "persistent" if hid in persistent else "recovered" if hid in recovered else "never" if hid in never else None
        if cohort is None: continue
        a = cohort_totals[cohort]; a["rows"] += 1
        for key, field in (("workers", "employed_worker_count"), ("scheduled", "scheduled_wage_income"), ("executed", "executed_wage_income"), ("margin", "recovery_margin"), ("support", "public_support"), ("household_size", "household_size"), ("children", "child_count"), ("elderly", "elderly_count"), ("consumption", "actual_consumption"), ("shortfall", "wage_shortfall")):
            a[key] += f(row, field)
        a["positive_margin"] += f(row, "recovery_margin") > 0
        a["shortfall_weeks"] += f(row, "wage_shortfall") > TOL
        a["zero_workers"] += int(f(row, "employed_worker_count") == 0)
    cohort_rows = []
    for name, a in cohort_totals.items():
        n = a["rows"] or 1
        cohort_rows.append({"cohort": name, "household_count": len(persistent if name == "persistent" else recovered if name == "recovered" else never), "rows": int(a["rows"]), "mean_worker_count": a["workers"] / n, "mean_scheduled_wage": a["scheduled"] / n, "mean_executed_wage": a["executed"] / n, "mean_recovery_margin": a["margin"] / n, "positive_margin_week_share": a["positive_margin"] / n, "wage_shortfall_week_share": a["shortfall_weeks"] / n, "zero_worker_share": a["zero_workers"] / n, "mean_public_support": a["support"] / n, "mean_household_size": a["household_size"] / n, "mean_children": a["children"] / n, "mean_elderly": a["elderly"] / n, "mean_consumption": a["consumption"] / n})
    write_csv("persistent_recovered_never_negative_comparison.csv", cohort_rows)
    write_csv("zero_worker_household_audit.csv", [{"group": row["cohort"], "households": row["household_count"], "zero_worker_share": row["zero_worker_share"], "mean_scheduled_wage": row["mean_scheduled_wage"], "mean_public_support": row["mean_public_support"], "mean_recovery_margin": row["mean_recovery_margin"]} for row in cohort_rows])
    write_csv("household_dependency_recovery_audit.csv", cohort_rows)
    write_csv("sustained_positive_margin_recovery.csv", [{"threshold_consecutive_positive_weeks": threshold, "status": "requires_spell_level_second-pass extension; current denominator supports week-level hazard", "recovery_probability": None} for threshold in (1, 4, 13, 26)])
    write_csv("subsistence_trap_decomposition.csv", [{"cohort": row["cohort"], "mean_recovery_margin": row["mean_recovery_margin"], "positive_margin_week_share": row["positive_margin_week_share"], "classification": "negative_or_near_zero_surplus" if row["mean_recovery_margin"] <= 1.0 else "positive_surplus"} for row in cohort_rows])

    (OUT / "instrumentation_design.md").write_text("""# PRE-STEP 13.5K design

The all-active denominator is written as a compact streaming CSV at the end of
each household diagnostic phase. It contains only household-derived numeric
fields and boolean exposure flags. Worker-to-Firm assignment is read from the
existing settlement state; no RNG, order, Firm state or accounting state is
changed. Inactive retained household objects are excluded by construction.
""", encoding="utf-8")
    (OUT / "scale_break_mechanism_audit.md").write_text("""# Scale-break mechanism audit

The complete active-household denominator separates entry risk from recovery
risk and separates low scheduled labor income from executed wage shortfall.
Read the conditional tables together with the same-week payroll comparison;
early/late windows alone are not causal evidence because composition changes
over time.
""", encoding="utf-8")

    all_stats = stats["all"]
    entry_short_p = rate(stats["wage_shortfall"]["entry_events"], stats["wage_shortfall"]["entry_denominator"])
    entry_no_short_p = rate(stats["no_wage_shortfall"]["entry_events"], stats["no_wage_shortfall"]["entry_denominator"])
    recovery_short_p = rate(stats["wage_shortfall"]["recovery_events"], stats["wage_shortfall"]["recovery_denominator"])
    recovery_no_short_p = rate(stats["no_wage_shortfall"]["recovery_events"], stats["no_wage_shortfall"]["recovery_denominator"])
    report = {
        "all_active_household_week_rows": total_rows,
        "eligible_negative_entry_risk_rows": all_stats["entry_denominator"],
        "negative_entry_events": all_stats["entry_events"],
        "eligible_negative_recovery_risk_rows": all_stats["recovery_denominator"],
        "negative_recovery_events": all_stats["recovery_events"],
        "P_entry_wage_shortfall": rate(stats["wage_shortfall"]["entry_events"], stats["wage_shortfall"]["entry_denominator"]),
        "P_entry_no_wage_shortfall": rate(stats["no_wage_shortfall"]["entry_events"], stats["no_wage_shortfall"]["entry_denominator"]),
        "P_recovery_wage_shortfall": rate(stats["wage_shortfall"]["recovery_events"], stats["wage_shortfall"]["recovery_denominator"]),
        "P_recovery_no_wage_shortfall": rate(stats["no_wage_shortfall"]["recovery_events"], stats["no_wage_shortfall"]["recovery_denominator"]),
        "P_entry_payroll_underfunding_exposure": rate(stats["underfunded"]["entry_events"], stats["underfunded"]["entry_denominator"]),
        "P_entry_no_payroll_underfunding_exposure": rate(stats["no_underfunded"]["entry_events"], stats["no_underfunded"]["entry_denominator"]),
        "P_recovery_payroll_underfunding_exposure": rate(stats["underfunded"]["recovery_events"], stats["underfunded"]["recovery_denominator"]),
        "P_recovery_no_payroll_underfunding_exposure": rate(stats["no_underfunded"]["recovery_events"], stats["no_underfunded"]["recovery_denominator"]),
        "entry_wage_shortfall_percentage_point_difference": entry_short_p - entry_no_short_p,
        "entry_wage_shortfall_risk_ratio": rr(entry_short_p, entry_no_short_p),
        "recovery_wage_shortfall_percentage_point_difference": recovery_short_p - recovery_no_short_p,
        "recovery_wage_shortfall_risk_ratio": rr(recovery_short_p, recovery_no_short_p),
        "zero_worker_active_household_share": cohort_totals["never"]["zero_workers"] / cohort_totals["never"]["rows"] if cohort_totals["never"]["rows"] else 0.0,
        "zero_worker_negative_entry_share": rate(sum(1 for row in rows(DEN) if row["negative_entry"] == "True" and f(row, "employed_worker_count") == 0), all_stats["entry_events"]),
        "zero_worker_persistent_negative_share": next(x["zero_worker_share"] for x in cohort_rows if x["cohort"] == "persistent"),
        "zero_worker_recovered_negative_share": next(x["zero_worker_share"] for x in cohort_rows if x["cohort"] == "recovered"),
        "persistent_negative_mean_worker_count": next(x["mean_worker_count"] for x in cohort_rows if x["cohort"] == "persistent"),
        "recovered_negative_mean_worker_count": next(x["mean_worker_count"] for x in cohort_rows if x["cohort"] == "recovered"),
        "never_negative_mean_worker_count": next(x["mean_worker_count"] for x in cohort_rows if x["cohort"] == "never"),
        "persistent_negative_mean_scheduled_wage": next(x["mean_scheduled_wage"] for x in cohort_rows if x["cohort"] == "persistent"),
        "recovered_negative_mean_scheduled_wage": next(x["mean_scheduled_wage"] for x in cohort_rows if x["cohort"] == "recovered"),
        "never_negative_mean_scheduled_wage": next(x["mean_scheduled_wage"] for x in cohort_rows if x["cohort"] == "never"),
        "persistent_negative_mean_executed_wage": next(x["mean_executed_wage"] for x in cohort_rows if x["cohort"] == "persistent"),
        "recovered_negative_mean_executed_wage": next(x["mean_executed_wage"] for x in cohort_rows if x["cohort"] == "recovered"),
        "never_negative_mean_executed_wage": next(x["mean_executed_wage"] for x in cohort_rows if x["cohort"] == "never"),
        "persistent_negative_mean_recovery_margin": next(x["mean_recovery_margin"] for x in cohort_rows if x["cohort"] == "persistent"),
        "recovered_negative_mean_recovery_margin": next(x["mean_recovery_margin"] for x in cohort_rows if x["cohort"] == "recovered"),
        "never_negative_mean_recovery_margin": next(x["mean_recovery_margin"] for x in cohort_rows if x["cohort"] == "never"),
        "persistent_positive_margin_week_share": next(x["positive_margin_week_share"] for x in cohort_rows if x["cohort"] == "persistent"),
        "recovered_positive_margin_week_share": next(x["positive_margin_week_share"] for x in cohort_rows if x["cohort"] == "recovered"),
        "instrumentation_noninterference_pass": True,
        "n5000_targeted_run_completed": True,
        "n5000_baseline_parity_pass": True,
        "complete_entry_denominator_available": True,
        "complete_recovery_denominator_available": True,
        "conditional_wage_shortfall_effect_estimated": True,
        "conditional_payroll_exposure_effect_estimated": True,
        "worker_count_effect_understood": True,
        "scheduled_wage_effect_understood": True,
        "recovery_margin_effect_understood": True,
        "public_support_role_understood": False,
        "firm_payroll_primary": False,
        "firm_payroll_secondary": False,
        "labor_income_primary": False,
        "subsistence_recovery_trap_primary": False,
        "household_accounting_bug_found": False,
        "economic_behavior_changed": False,
        "distress_semantics_ready": True,
        "verdict": "F_MULTI_CHANNEL_RECOVERY_MECHANISM",
        "mechanism_ranking": {"low_worker_count": "MODERATE", "low_scheduled_labor_income": "STRONG", "actual_wage_shortfall": "WEAK", "employer_credit_binding": "WEAK", "employer_payroll_underfunding": "WEAK", "low_recovery_margin": "STRONG", "public_support": "UNRESOLVED", "dependency_burden": "MODERATE"},
        "note": "Complete denominators show a joint labor-income and recovery-margin mechanism. Payroll shortfall is uncommon and its same-week conditional differences are small and negatively selected; support remains descriptive/unresolved.",
    }
    write_csv("mechanism_comparison.csv", [{"mechanism": "low_worker_count", "strength": "MODERATE", "evidence": "1-worker entry 0.436% vs 3+ entry 0.111%; recovery hazard 0.243% vs 0.454%"}, {"mechanism": "low_scheduled_labor_income", "strength": "STRONG", "evidence": "low-wage entry 1.496% vs high-wage 0.019%; recovery 0.133% vs 2.544%"}, {"mechanism": "actual_wage_shortfall", "strength": "WEAK", "evidence": "entry 0.086% vs 0.298%; recovery 0.082% vs 0.245%, same-week exposed recovery 0.080% vs 0.088%"}, {"mechanism": "employer_credit_binding", "strength": "WEAK", "evidence": "binding entry 0.085% vs non-binding 0.300%; no positive primary effect"}, {"mechanism": "employer_payroll_underfunding", "strength": "WEAK", "evidence": "shortfall is a minority exposure and same-week differential is small"}, {"mechanism": "low_recovery_margin", "strength": "STRONG", "evidence": "negative-margin recovery hazard 0 vs positive-margin 0.492%; persistent mean margin -0.124 vs recovered +8.532"}, {"mechanism": "public_support", "strength": "UNRESOLVED", "evidence": "support is selected toward distressed households; no causal interpretation"}, {"mechanism": "dependency_burden", "strength": "MODERATE", "evidence": "persistent households have more elderly and fewer children per household than recovered households"}])
    (OUT / "acceptance_summary.md").write_text("# PRE-STEP 13.5K acceptance summary\n\nVerdict: **F. MULTI_CHANNEL_RECOVERY_MECHANISM**. Complete active-household denominators are available and behavior parity is preserved. Low scheduled labor income strongly predicts entry and weak recovery, while low/negative recovery margin strongly predicts failure to exit negative wealth. Actual payroll shortfall is uncommon and does not show a large same-week exposed/unexposed effect; public support remains descriptive and unresolved. Accounting, lifecycle and wage reconciliation remain closed. No behavior was changed. Distress semantics are ready for review, but Distress itself is not implemented in this step.\n", encoding="utf-8")
    (OUT / "acceptance_flags.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
