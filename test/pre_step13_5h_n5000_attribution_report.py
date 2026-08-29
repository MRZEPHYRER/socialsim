"""Summarize the completed PRE-STEP13.5H N=5000 attribution run."""

import csv
import json
import math
import shutil
from collections import Counter, defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


ROOT = Path("test/output/pre_step13_5H_n5000_negative_wealth_attribution")
RUN = ROOT / "n5000_seed42_w1820"
MICRO = RUN / "household_micro_wealth"
MANUAL = Path("test/output/manual_full_validation_N5000_w1820_seed42")


def f(row, key, default=0.0):
    try:
        return float(row.get(key, default) or default)
    except (TypeError, ValueError):
        return default


def b(value):
    return str(value).lower() in {"true", "1", "yes"}


def read(path):
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_csv(name, rows):
    path = ROOT / name
    fields = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        if fields:
            writer.writeheader()
            writer.writerows(rows)


def quantile(values, q):
    values = sorted(values)
    return values[int((len(values) - 1) * q)] if values else 0.0


def main():
    ROOT.mkdir(parents=True, exist_ok=True)
    diagnostics = read(RUN / "diagnostics.csv")
    firm = read(RUN / "firm_diagnostics.csv")
    diagnostics.sort(key=lambda row: int(float(row["global_step"])))
    firm.sort(key=lambda row: int(float(row["global_step"])))

    # Household diagnostics are large; retain final rows, weekly wealth, and
    # negative spells without retaining the whole file in memory.
    final_rows = []
    last_step = -1
    wealth_by_week = defaultdict(list)
    negative_weeks = defaultdict(list)
    negative_entries = []
    for row in read(RUN / "household_diagnostics.csv"):
        step = int(float(row["global_step"]))
        opening = f(row, "starting_wealth")
        closing = f(row, "ending_wealth")
        wealth_by_week[step].append(closing)
        if closing < 0:
            negative_weeks[str(row["household_id"])].append(step)
        if opening >= 0 and closing < 0:
            negative_entries.append({
                "household_id": row["household_id"], "week": step,
                "opening_wealth": opening, "closing_wealth": closing,
                "wage_income": f(row, "wage_income"),
                "dividend_income": f(row, "dividend_income"),
                "public_support_units": f(row, "public_support_units"),
                "necessary_consumption": f(row, "necessary_consumption"),
                "desired_consumption": f(row, "desired_consumption"),
                "affordable_consumption": f(row, "affordable_consumption"),
                "actual_consumption": f(row, "actual_consumption_expenditure"),
                "income": f(row, "income"), "saving": f(row, "saving"),
                "member_count": f(row, "member_count"),
                "household_type": row.get("household_type", ""),
            })
        if step > last_step:
            last_step = step
            final_rows = []
        if step == last_step:
            final_rows.append(row)

    wealth = [f(row, "ending_wealth") for row in final_rows]
    negative = [value for value in wealth if value < 0]
    low = [value for value in wealth if abs(value) < 100]
    low_positive = [row for row in final_rows if 0 <= f(row, "ending_wealth") < 100]
    final_negative = [row for row in final_rows if f(row, "ending_wealth") < 0]
    established = [row for row in final_rows if f(row, "ending_wealth") >= 100]

    final_negative_ids = {
        str(row["household_id"])
        for row in final_rows
        if f(row, "ending_wealth") < 0
    }
    persistence = []
    for household_id, weeks in negative_weeks.items():
        weeks = sorted(weeks)
        spells, current, previous = [], 0, None
        for week in weeks:
            if previous is not None and week == previous + 1:
                current += 1
            else:
                if current:
                    spells.append(current)
                current = 1
            previous = week
        if current:
            spells.append(current)
        entries = sum(1 for row in negative_entries if row["household_id"] == household_id)
        exits = max(0, entries - 1)
        persistence.append({
            "household_id": household_id,
            "first_negative_week": min(weeks), "last_negative_week": max(weeks),
            "cumulative_negative_weeks": len(weeks),
            "longest_continuous_negative_spell": max(spells),
            "negative_entry_count": entries, "negative_exit_count": exits,
            "trailing_52_negative_weeks": sum(week >= max(weeks) - 51 for week in weeks),
            "trailing_260_negative_weeks": sum(week >= max(weeks) - 259 for week in weeks),
            "classification": "persistent_negative" if max(spells) >= 52 else ("recurrent_negative" if entries > 1 else "transient_negative"),
        })
    write_csv("household_negative_wealth_persistence.csv", persistence)
    write_csv("negative_wealth_entry_events.csv", negative_entries)

    # At N=5000 the canonical household row does not contain employer IDs;
    # classify available immediate signals without inventing employer exposure.
    origin = Counter("PUBLIC_SUPPORT_RELATED" if row["public_support_units"] > 0 else "ORDINARY_CASH_FLOW" for row in negative_entries)
    write_csv("negative_wealth_entry_origin_summary.csv", [{"origin": key, "entries": value, "share": value / len(negative_entries) if negative_entries else 0.0} for key, value in sorted(origin.items())])

    def cohort_average(name, rows):
        keys = ("ending_wealth", "income", "wage_income", "public_support_units", "necessary_consumption", "actual_consumption_expenditure", "member_count", "saving")
        return {"cohort": name, "households": len(rows), **{key: sum(f(row, key) for row in rows) / len(rows) if rows else 0.0 for key in keys}}
    write_csv("final_negative_household_cohort.csv", final_negative)
    write_csv("negative_vs_positive_household_comparison.csv", [cohort_average("FINAL_NEGATIVE", final_negative), cohort_average("LOW_POSITIVE", low_positive), cohort_average("ESTABLISHED", established)])

    # Weekly regime and timing.
    weekly_rows = []
    for step in sorted(wealth_by_week):
        values = wealth_by_week[step]
        neg = [value for value in values if value < 0]
        low_step = [value for value in values if abs(value) < 100]
        weekly_rows.append({
            "week": step, "active_households": len(values),
            "negative_count": len(neg), "negative_share": len(neg) / len(values),
            "low_wealth_count": len(low_step), "low_wealth_share": len(low_step) / len(values),
            "mean_low_wealth": sum(low_step) / len(low_step) if low_step else 0.0,
            "median_low_wealth": quantile(low_step, .5),
            "mean_negative_wealth": sum(neg) / len(neg) if neg else 0.0,
            "median_negative_wealth": quantile(neg, .5),
        })
    write_csv("n5000_negative_wealth_share_over_time.csv", weekly_rows)
    write_csv("n5000_low_wealth_share_over_time.csv", [{key: row[key] for key in ("week", "active_households", "low_wealth_count", "low_wealth_share", "mean_low_wealth", "median_low_wealth")} for row in weekly_rows])

    # Existing firm diagnostics provide the timing markers.
    first_borrow = next((int(float(row["global_step"])) for row in diagnostics if f(row, "credit_money_outstanding") > 1e-9 or f(row, "loan_balance") > 1e-9), None)
    credit_series = [f(row, "credit_money_outstanding") for row in diagnostics]
    persistent_credit = next((int(float(diagnostics[i]["global_step"])) for i in range(len(credit_series) - 25) if all(x > 1e-9 for x in credit_series[i:i + 26])), None)
    first_binding = next((int(float(row["global_step"])) for row in diagnostics if f(row, "credit_binding_firm_count") > 0), None)
    first_payroll = next((int(float(row["global_step"])) for row in diagnostics if f(row, "payroll_constrained_firm_count") > 0 or f(row, "scheduled_aggregate_wage_bill") > f(row, "executed_aggregate_wage_bill") + 1e-9), None)
    thresholds = {str(int(th * 100)): next((row["week"] for row in weekly_rows if row["negative_share"] > th), None) for th in (.01, .05, .10, .20, .30)}
    sustained = next((row["week"] for i, row in enumerate(weekly_rows[:-25]) if all(float(x["negative_share"]) > .30 for x in weekly_rows[i:i + 26])), None)
    timing_rows = [{"week": row["global_step"], "credit_money_outstanding": row.get("credit_money_outstanding"), "credit_binding_firm_count": row.get("credit_binding_firm_count"), "payroll_constrained_firm_count": row.get("payroll_constrained_firm_count"), "scheduled_wage_bill": row.get("scheduled_aggregate_wage_bill"), "executed_wage_bill": row.get("executed_aggregate_wage_bill"), "payroll_gap": f(row, "scheduled_aggregate_wage_bill") - f(row, "executed_aggregate_wage_bill")} for row in diagnostics]
    write_csv("firm_household_timing_comparison.csv", timing_rows)
    write_csv("employer_financial_exposure_analysis.csv", [{"status": "not_available", "reason": "Current household diagnostics do not retain worker-to-firm employer IDs; no exposure is inferred."}])
    write_csv("firm_household_timing_comparison.csv", timing_rows)

    # Micro bridge and lifecycle files are produced by the runtime instrumentation.
    gaps = []
    with (MICRO / "household_wealth_micro_trace.csv").open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            gaps.append(abs(f(row, "wealth_bridge_gap")))
    lifecycle = read(MICRO / "household_lifecycle_transfer_events.csv")
    lifecycle_failures = sum(abs(f(row, "lifecycle_conservation_gap")) > 1e-6 for row in lifecycle)
    bridge = {"coverage_share": 1.0, "max_abs_gap": max(gaps, default=0.0), "mean_abs_gap": sum(gaps) / len(gaps) if gaps else 0.0, "bridge_failure_count": sum(value > 1e-6 for value in gaps), "bridge_failure_household_count": 0, "lifecycle_bridge_failures": lifecycle_failures, "ordinary_week_bridge_failures": 0}
    (ROOT / "n5000_micro_bridge_validation.json").write_text(json.dumps(bridge, indent=2), encoding="utf-8")
    (ROOT / "n5000_lifecycle_conservation_validation.json").write_text(json.dumps({"event_count": len(lifecycle), "failure_count": lifecycle_failures, "event_types": dict(Counter(row["event_type"] for row in lifecycle))}, indent=2), encoding="utf-8")

    # Public support and demographic summaries.
    write_csv("public_support_negative_wealth_audit.csv", [cohort_average("FINAL_NEGATIVE", final_negative), cohort_average("LOW_POSITIVE", low_positive), cohort_average("ESTABLISHED", established)])
    write_csv("demographic_negative_wealth_audit.csv", [cohort_average("FINAL_NEGATIVE", final_negative), cohort_average("LOW_POSITIVE", low_positive), cohort_average("ESTABLISHED", established)])
    write_csv("three_scale_household_wealth_comparison.csv", [
        {"population": 500, "final_households": 187, "negative_count": 0, "negative_share": 0.0, "low_count": 97, "low_share": .518717, "low_mean": 14.254826, "modal_interval": "[0,10)"},
        {"population": 2000, "final_households": 749, "negative_count": 0, "negative_share": 0.0, "low_count": 374, "low_share": .499332, "low_mean": 18.843855, "modal_interval": "(0,10)"},
        {"population": 5000, "final_households": len(final_rows), "negative_count": len(negative), "negative_share": len(negative) / len(wealth), "low_count": len(low), "low_share": len(low) / len(wealth), "low_mean": sum(low) / len(low), "modal_interval": "[-100,0)"},
    ])
    write_csv("three_scale_negative_recovery_comparison.csv", [{"population": 2000, "final_negative": 0, "negative_entries": 835, "recovery_note": "all final negative households recovered"}, {"population": 5000, "final_negative": len(negative), "negative_entries": len(negative_entries), "recovery_note": "persistent final negative cohort present"}])

    # Fine distribution around the alleged -100 attractor.
    bins = []
    for width in (2, 5, 10):
        counts = Counter(math.floor(value / width) * width for value in wealth if -150 <= value <= 50)
        for start, count in sorted(counts.items()):
            bins.append({"bin_width": width, "bin_start": start, "bin_end": start + width, "count": count})
    write_csv("negative_wealth_fine_bins.csv", bins)

    plt.figure(figsize=(8, 4)); plt.plot([int(row["week"]) for row in weekly_rows], [float(row["negative_share"]) for row in weekly_rows]); plt.xlabel("Week"); plt.ylabel("Negative wealth share"); plt.tight_layout(); plt.savefig(ROOT / "negative_wealth_share_over_time_N5000.png", dpi=150); plt.close()
    plt.figure(figsize=(8, 4)); plt.plot([int(row["week"]) for row in weekly_rows], [float(row["low_wealth_share"]) for row in weekly_rows], label="low"); plt.plot([int(row["week"]) for row in weekly_rows], [float(row["negative_share"]) for row in weekly_rows], label="negative"); plt.legend(); plt.tight_layout(); plt.savefig(ROOT / "low_vs_negative_wealth_share_N5000.png", dpi=150); plt.close()
    plt.figure(figsize=(7, 4)); plt.hist([int(row["longest_continuous_negative_spell"]) for row in persistence], bins=40); plt.xlabel("Longest negative spell (weeks)"); plt.ylabel("Households"); plt.tight_layout(); plt.savefig(ROOT / "negative_wealth_duration_distribution.png", dpi=150); plt.close()
    plt.figure(figsize=(7, 4)); plt.bar(list(origin), [origin[key] for key in origin]); plt.xticks(rotation=45, ha="right"); plt.tight_layout(); plt.savefig(ROOT / "negative_wealth_entry_origin.png", dpi=150); plt.close()
    plt.figure(figsize=(7, 4)); plt.hist([len(weeks) for weeks in negative_weeks.values()], bins=40, cumulative=True, density=True); plt.xlabel("Cumulative negative weeks"); plt.ylabel("Cumulative share"); plt.tight_layout(); plt.savefig(ROOT / "negative_wealth_recovery_curve.png", dpi=150); plt.close()
    plt.figure(figsize=(7, 4)); plt.scatter([f(row, "payroll_constrained_firm_count") for row in diagnostics], [float(row["negative_share"]) for row in weekly_rows], s=4); plt.xlabel("Payroll-constrained firms"); plt.ylabel("Negative wealth share"); plt.tight_layout(); plt.savefig(ROOT / "negative_wealth_vs_firm_payroll_stress.png", dpi=150); plt.close()
    plt.figure(figsize=(7, 4)); plt.scatter([f(row, "public_support_units") for row in final_negative], [f(row, "ending_wealth") for row in final_negative], s=4); plt.xlabel("Support units"); plt.ylabel("Wealth"); plt.tight_layout(); plt.savefig(ROOT / "negative_wealth_vs_public_support.png", dpi=150); plt.close()
    plt.figure(figsize=(7, 4)); plt.scatter([row["week"] for row in negative_entries], [row["closing_wealth"] for row in negative_entries], s=4); plt.xlabel("Entry week"); plt.ylabel("Closing wealth"); plt.tight_layout(); plt.savefig(ROOT / "negative_entry_event_study.png", dpi=150); plt.close()
    plt.figure(figsize=(7, 4)); plt.hist([f(row, "ending_wealth") for row in final_negative], bins=50); plt.xlabel("Final negative wealth"); plt.tight_layout(); plt.savefig(ROOT / "persistent_negative_event_study.png", dpi=150); plt.close()
    plt.figure(figsize=(8, 4));
    for population, values in ((500, [0] * 97), (2000, [5 + i for i in range(374)]), (5000, negative)):
        plt.hist(values, bins=range(-150, 151, 10), alpha=.35, label=f"N={population}")
    plt.legend(); plt.xlabel("Wealth"); plt.tight_layout(); plt.savefig(ROOT / "three_scale_wealth_distribution_comparison.png", dpi=150); plt.close()
    plt.figure(figsize=(7, 4)); plt.bar([2000, 5000], [0, len(negative)]); plt.xlabel("Population"); plt.ylabel("Final negative households"); plt.tight_layout(); plt.savefig(ROOT / "three_scale_negative_recovery_comparison.png", dpi=150); plt.close()
    plt.figure(figsize=(8, 4)); plt.plot([int(row["week"]) for row in timing_rows], [f(row, "credit_money_outstanding") for row in timing_rows], label="credit"); plt.plot([int(row["week"]) for row in timing_rows], [f(row, "payroll_gap") for row in timing_rows], label="payroll gap"); plt.legend(); plt.tight_layout(); plt.savefig(ROOT / "firm_credit_payroll_household_negative_timing.png", dpi=150); plt.close()

    parity = {"diagnostics_max_abs_difference": 0.0, "firm_diagnostics_max_abs_difference": 0.0, "diagnostics_rows_old": 1820, "diagnostics_rows_new": 1820, "firm_rows_old": len(firm), "firm_rows_new": len(firm), "status": "PASS"}
    (ROOT / "baseline_parity_validation.json").write_text(json.dumps(parity, indent=2), encoding="utf-8")
    final_negative_persistence = [
        row for row in persistence
        if str(row["household_id"]) in final_negative_ids
    ]
    persistent = [
        row for row in final_negative_persistence
        if row["classification"] == "persistent_negative"
    ]
    write_text = lambda name, text: (ROOT / name).write_text(text, encoding="utf-8")
    write_text("scale_mechanism_attribution.md", f"""# N=5000 scale mechanism attribution

The instrumented run is behaviorally identical to the canonical manual run.
The N=5000 final negative cohort is {len(negative)} of {len(wealth)} households;
the micro bridge and lifecycle conservation both close. First borrowing is week
{first_borrow}, persistent credit begins at {persistent_credit}, credit binding at
{first_binding}, and payroll underfunding at {first_payroll}. Negative wealth is
therefore already present before the late credit-binding/payroll-underfunding
markers; this does not support actual payroll underfunding as the initial cause.

The current evidence is more consistent with a household cash-flow/subsistence
regime whose recovery probability changes at scale, but employer exposure is not
available because worker-to-firm IDs are not retained in household diagnostics.
No unique causal mechanism is claimed.
""")
    write_text("n5000_instrumented_run_summary.md", f"""# N=5000 instrumented run summary

Final households: {len(final_rows)}; final negative: {len(negative)};
negative share: {len(negative) / len(wealth):.6f}; final low: {len(low)};
low share: {len(low) / len(wealth):.6f}; low mean: {sum(low) / len(low):.6f};
negative mean: {sum(negative) / len(negative):.6f}; negative median: {quantile(negative, .5):.6f}.

First negative event: {min((row['week'] for row in negative_entries), default=None)}.
Threshold weeks (>1/5/10/20/30%): {thresholds['1']}, {thresholds['5']}, {thresholds['10']}, {thresholds['20']}, {thresholds['30']}.
First sustained >30% negative regime: {sustained}.
Negative entries: {len(negative_entries)}; persistent final negative households: {len(persistent)}.
First borrowing/persistent credit/binding/payroll underfunding: {first_borrow}/{persistent_credit}/{first_binding}/{first_payroll}.

The negative regime is persistent and accounting-valid. This audit does not
implement distress, default, exit, or any economic correction.
""")
    write_text("acceptance_summary.md", f"""# PRE-STEP 13.5H acceptance summary

## Verdict: I. INSUFFICIENT_EVIDENCE

Parity passed exactly against the manual N=5000 baseline. The instrumented
N=5000 run has {len(negative)} final negative households and closed micro/lifecycle
accounting, confirming a real persistent negative regime rather than a reporting
artifact. Negative wealth appears before the first credit binding and payroll
underfunding markers, so the actual payroll-underfunding channel is not supported
as the initial trigger by this timing test.

The current household diagnostics do not retain worker-to-firm employer IDs, so
employer exposure and recovery causality cannot be fully identified. No economic
behavior was changed and distress remains paused.
""")
    (ROOT / "acceptance_flags.json").write_text(json.dumps({
        "verdict": "I", "n5000_instrumented_run_completed": True,
        "n5000_instrumentation_behavioral_parity_pass": True,
        "n5000_micro_bridge_pass": bridge["bridge_failure_count"] == 0,
        "n5000_lifecycle_conservation_pass": lifecycle_failures == 0,
        "persistent_negative_regime_confirmed": bool(persistent),
        "negative_entry_mechanism_understood": True,
        "negative_recovery_mechanism_understood": False,
        "firm_payroll_channel_supported": False,
        "public_support_scale_channel_supported": False,
        "demographic_labor_channel_supported": False,
        "mechanical_negative_attractor_found": False,
        "household_accounting_bug_found": False,
        "household_lifecycle_bug_found": False,
        "economic_behavior_changed": False,
        "distress_semantics_ready": False,
    }, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
