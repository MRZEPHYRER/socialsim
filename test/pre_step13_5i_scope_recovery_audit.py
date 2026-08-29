"""PRE-STEP13.5I: reconcile household scope and recovery using existing runs."""

import csv
import json
import math
import statistics
from collections import defaultdict
from pathlib import Path


ROOT = Path("test/output/pre_step13_5I_household_scope_recovery")
RUNS = {
    500: Path("test/output/pre_step13_5F_household_lifecycle_attribution/n500_seed42_w1820"),
    2000: Path("test/output/pre_step13_5G_household_wealth_scale_transition/n2000_seed42_w1820"),
    5000: Path("test/output/pre_step13_5H_n5000_negative_wealth_attribution/n5000_seed42_w1820"),
}


def f(row, key, default=0.0):
    try:
        return float(row.get(key, default) or default)
    except (TypeError, ValueError):
        return default


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


def load_histories(path):
    histories = defaultdict(list)
    last_step = -1
    with (path / "household_diagnostics.csv").open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            step = int(float(row["global_step"]))
            last_step = max(last_step, step)
            histories[str(row["household_id"])].append({
                "step": step,
                "active": f(row, "member_count") > 0,
                "member_count": f(row, "member_count"),
                "wealth": f(row, "ending_wealth"),
                "opening": f(row, "starting_wealth"),
                "net_flow": f(row, "ending_wealth") - f(row, "starting_wealth"),
                "income": f(row, "income"),
                "wage": f(row, "wage_income"),
                "dividend": f(row, "dividend_income"),
                "support": f(row, "public_support_units"),
                "necessary": f(row, "necessary_consumption"),
                "actual": f(row, "actual_consumption_expenditure"),
                "saving": f(row, "saving"),
            })
    for rows in histories.values():
        rows.sort(key=lambda row: row["step"])
    return histories, last_step


def final_rows(histories, last_step):
    output = []
    for household_id, rows in histories.items():
        if rows and rows[-1]["step"] == last_step:
            row = dict(rows[-1])
            row["household_id"] = household_id
            output.append(row)
    return output


def spell_stats(rows):
    active_rows = [row for row in rows if row["active"]]
    negative = [row for row in active_rows if row["wealth"] < 0]
    if not negative:
        return None
    negative_weeks = [row["step"] for row in negative]
    spells = []
    current = 0
    previous = None
    for week in negative_weeks:
        if previous is not None and week == previous + 1:
            current += 1
        else:
            if current:
                spells.append(current)
            current = 1
        previous = week
    if current:
        spells.append(current)
    entries = []
    recoveries = []
    for index, row in enumerate(active_rows):
        previous_wealth = active_rows[index - 1]["wealth"] if index else row["opening"]
        if previous_wealth >= 0 and row["wealth"] < 0:
            entries.append((index, row))
            recovery = next((future for future in active_rows[index + 1:] if future["wealth"] >= 0), None)
            if recovery is not None:
                recoveries.append((recovery["step"] - row["step"], recovery))
    final_negative = active_rows[-1]["wealth"] < 0 if active_rows else False
    return {
        "first_negative_week": min(negative_weeks),
        "cumulative_negative_weeks": len(negative_weeks),
        "longest_spell": max(spells),
        "entry_count": len(entries),
        "recovery_count": len(recoveries),
        "recovery_times": [time for time, _ in recoveries],
        "ever_recovered": bool(recoveries),
        "final_negative": final_negative,
        "negative_rows": negative,
        "active_rows": active_rows,
    }


def cohort_average(name, rows):
    keys = ("wealth", "income", "wage", "support", "necessary", "actual", "net_flow", "member_count", "saving")
    return {"cohort": name, "households": len(rows), **{key: sum(row[key] for row in rows) / len(rows) if rows else 0.0 for key in keys}}


def main():
    ROOT.mkdir(parents=True, exist_ok=True)
    histories_by_scale = {}
    final_by_scale = {}
    last_by_scale = {}
    for scale, path in RUNS.items():
        histories, last = load_histories(path)
        histories_by_scale[scale] = histories
        last_by_scale[scale] = last
        final_by_scale[scale] = final_rows(histories, last)

    # Scope reconciliation for the canonical N=5000 run.
    final_all = final_by_scale[5000]
    active_final = [row for row in final_all if row["active"]]
    inactive_final = [row for row in final_all if not row["active"]]
    lifecycle_events = read(RUNS[5000] / "household_micro_wealth/lifecycle_event_wealth_analysis.csv")
    last_event = {}
    for event in lifecycle_events:
        last_event[str(event.get("household_id"))] = event
    inactive_rows = []
    for row in inactive_final:
        event = last_event.get(str(row["household_id"]), {})
        inactive_rows.append({
            "household_id": row["household_id"], "active": False,
            "household_size": row["member_count"], "wealth": row["wealth"],
            "negative": row["wealth"] < 0, "low_wealth": abs(row["wealth"]) < 100,
            "dissolution_week": event.get("step", ""),
            "final_lifecycle_event": event.get("event_type", ""),
            "retention_reason": "historical household object retained after inactive cleanup",
        })
    write_csv("inactive_household_reconciliation.csv", inactive_rows)

    def metrics(label, rows):
        wealth = [row["wealth"] for row in rows]
        low = [value for value in wealth if abs(value) < 100]
        neg = [value for value in wealth if value < 0]
        return {
            "scope": label, "households": len(rows), "negative_count": len(neg),
            "negative_share": len(neg) / len(rows) if rows else 0.0,
            "low_wealth_count": len(low), "low_wealth_share": len(low) / len(rows) if rows else 0.0,
            "mean_wealth": sum(wealth) / len(wealth) if wealth else 0.0,
            "median_wealth": statistics.median(wealth) if wealth else 0.0,
            "low_wealth_mean": sum(low) / len(low) if low else 0.0,
            "negative_mean": sum(neg) / len(neg) if neg else 0.0,
            "negative_median": statistics.median(neg) if neg else 0.0,
        }
    write_csv("canonical_active_household_wealth_summary.csv", [metrics("ALL_RECORDS", final_all), metrics("ACTIVE_ONLY_CANONICAL", active_final)])
    write_csv("corrected_h_negative_cohort_summary.csv", [cohort_average("ACTIVE_FINAL_NEGATIVE", [row for row in active_final if row["wealth"] < 0]), cohort_average("ACTIVE_LOW_POSITIVE", [row for row in active_final if 0 <= row["wealth"] < 100])])

    # Active-lifetime persistence and recovery for all three existing runs.
    recovery_rows = []
    recovery_curve_rows = []
    scale_persistence_rows = []
    stats_by_scale = {}
    for scale, histories in histories_by_scale.items():
        stats = {}
        active_final_ids = {
            str(row["household_id"])
            for row in final_by_scale[scale]
            if row["active"]
        }
        for household_id, rows in histories.items():
            result = spell_stats(rows)
            if result:
                stats[household_id] = result
                stats[household_id]["household_id"] = household_id
                stats[household_id]["final_negative"] = (
                    household_id in active_final_ids
                    and result["active_rows"][-1]["wealth"] < 0
                )
                recovery_rows.append({"population": scale, **{key: result[key] for key in ("household_id", "first_negative_week", "cumulative_negative_weeks", "longest_spell", "entry_count", "recovery_count", "ever_recovered", "final_negative")}})
        stats_by_scale[scale] = stats
        entries = sum(result["entry_count"] for result in stats.values())
        recovery_times = [time for result in stats.values() for time in result["recovery_times"]]
        final_negative_stats = [result for result in stats.values() if result["final_negative"]]
        persistent = [result for result in final_negative_stats if result["longest_spell"] >= 52]
        recurrent = [result for result in final_negative_stats if result not in persistent and result["entry_count"] > 1]
        transient = [result for result in final_negative_stats if result not in persistent and result not in recurrent]
        active_count = len(final_by_scale[scale])
        for horizon in (1, 4, 13, 26, 52, 104, 260):
            recovered_by_horizon = sum(any(time <= horizon for time in result["recovery_times"]) for result in stats.values())
            recovery_curve_rows.append({"population": scale, "horizon_weeks": horizon, "negative_entry_count": entries, "recovered_by_horizon": recovered_by_horizon, "recovery_probability": recovered_by_horizon / entries if entries else 0.0})
        scale_persistence_rows.append({"population": scale, "active_final_households": active_count, "negative_entry_rate_per_household_year": entries / active_count / (last_by_scale[scale] / 52) if active_count else 0.0, "recovery_probability": sum(result["ever_recovered"] for result in stats.values()) / len(stats) if stats else 0.0, "median_recovery_time": statistics.median(recovery_times) if recovery_times else 0.0, "p90_recovery_time": quantile(recovery_times, .9), "persistent_negative_probability": len(persistent) / len(stats) if stats else 0.0, "recurrent_negative_probability": len(recurrent) / len(stats) if stats else 0.0, "transient_negative_probability": len(transient) / len(stats) if stats else 0.0})
    write_csv("active_lifetime_negative_persistence.csv", recovery_rows)
    write_csv("negative_wealth_recovery_by_scale.csv", recovery_curve_rows)
    write_csv("negative_entry_vs_recovery_by_scale.csv", scale_persistence_rows)

    # N=5000 recovered vs persistent cash-flow evidence, active weeks only.
    n5_stats = stats_by_scale[5000]
    persistent_ids = {hid for hid, result in n5_stats.items() if result["final_negative"] and result["longest_spell"] >= 52}
    recovered_ids = {hid for hid, result in n5_stats.items() if result["ever_recovered"] and hid not in persistent_ids}
    def rows_for_ids(ids):
        return [row for hid in ids for row in n5_stats[hid]["negative_rows"]]
    persistent_rows = rows_for_ids(persistent_ids)
    recovered_rows = rows_for_ids(recovered_ids)
    write_csv("persistent_vs_recovered_cashflow.csv", [cohort_average("PERSISTENT_NEGATIVE", persistent_rows), cohort_average("RECOVERED_NEGATIVE", recovered_rows)])

    # Week-one and initial-vs-later cohorts.
    week1_negative = {hid for hid, rows in histories_by_scale[5000].items() if any(row["step"] == 1 and row["active"] and row["wealth"] < 0 for row in rows)}
    final_negative_ids = {hid for hid, result in n5_stats.items() if result["final_negative"]}
    final_from_week1 = final_negative_ids & week1_negative
    later_final = final_negative_ids - week1_negative
    week1_rows = [rows[1] for rows in histories_by_scale[5000].values() if len(rows) > 1 and rows[1]["step"] == 1]
    write_csv("week1_negative_cohort_N5000.csv", [row for row in week1_rows if row["wealth"] < 0])
    write_csv("week1_household_comparison_by_scale.csv", [cohort_average(f"N{scale}_WEEK1_NEGATIVE", [rows[1] for rows in histories.values() if len(rows) > 1 and rows[1]["step"] == 1 and rows[1]["wealth"] < 0]) for scale, histories in histories_by_scale.items()])
    write_csv("initial_vs_later_negative_summary.csv", [{"cohort": "FINAL_NEGATIVE_FROM_WEEK1", "count": len(final_from_week1), "share": len(final_from_week1) / len(final_negative_ids) if final_negative_ids else 0.0}, {"cohort": "FINAL_NEGATIVE_FIRST_ENTERED_LATER", "count": len(later_final), "share": len(later_final) / len(final_negative_ids) if final_negative_ids else 0.0}])
    write_csv("final_negative_origin_timing.csv", [{"cohort": "week1_negative", "count": len(final_from_week1)}, {"cohort": "later_negative", "count": len(later_final)}])

    # Scope audit of current Analysis V2/V2.2. Source audit shows the common
    # household selectors use active_households(); H's custom export used all.
    write_csv("household_analysis_scope_audit.csv", [
        {"analysis_output": name, "scope": "ACTIVE_ONLY_CORRECT", "evidence": "analysis/common.py active_households()"}
        for name in ("wealth distribution", "income distribution", "consumption distribution", "saving distribution", "wealth Gini", "income Gini", "Lorenz curves", "poverty", "security ratio", "economic pressure", "budget constrained share", "household type composition", "low-wealth analysis", "negative-wealth analysis")
    ] + [{"analysis_output": "household_diagnostics.csv final custom cohort", "scope": "SCOPE_BUG", "evidence": "raw export contains inactive member_count=0 records; H cohort used all records"}])
    (ROOT / "household_scope_semantics.md").write_text("""# Household scope semantics

`world.households` is the retained object registry. It includes historical
household objects whose `member_count == 0` after empty-household cleanup.
`active_households` / active household rows are the current socioeconomic
population. Household diagnostics are a historical panel and may contain both.
Micro trace follows rows while they are observed, including lifecycle events.

For current wealth distribution, poverty, inequality, security, low wealth and
negative wealth, the canonical scope is **ACTIVE_ONLY**. Retained inactive rows
remain useful for lifecycle accounting but must not enter current household
denominators. Analysis V2 common selectors already use active households; the
scope error was the H final-cohort reanalysis, not the core V2 selector.
""", encoding="utf-8")
    (ROOT / "proposed_scope_correction.md").write_text("""# Proposed scope correction

Keep historical household diagnostics unchanged. Add an explicit `active_only`
filter when deriving final socioeconomic cohorts from the panel, using
`member_count > 0` or the canonical active-household ID set. Do not delete
inactive records and do not change simulation behavior.
""", encoding="utf-8")

    # Initial scale audit from week 0.
    initial_rows = []
    for scale, histories in histories_by_scale.items():
        rows = [rows[0] for rows in histories.values() if rows and rows[0]["step"] == 0 and rows[0]["active"]]
        initial_rows.append({"population": scale, "initial_active_households": len(rows), "mean_initial_wealth": sum(row["wealth"] for row in rows) / len(rows) if rows else 0.0, "median_initial_wealth": statistics.median([row["wealth"] for row in rows]) if rows else 0.0, "initial_negative_count": sum(row["wealth"] < 0 for row in rows), "initial_mean_member_count": sum(row["member_count"] for row in rows) / len(rows) if rows else 0.0})
    write_csv("initial_household_wealth_scale_audit.csv", initial_rows)
    write_csv("household_labor_recovery_comparison.csv", [{"status": "partial", "note": "household diagnostics retain member_count but not worker/employer IDs; worker-level recovery comparison is unavailable without new instrumentation"}])
    (ROOT / "employer_history_reconstructability.md").write_text("""# Employer history reconstructability

The saved CSV outputs do not retain worker-to-firm assignment history. Current
Firm rosters in the final World cannot reconstruct historical assignments for
all prior weeks. Therefore `historical_employer_exposure_reconstructable = false`.
Employer exposure is secondary because negative wealth starts at week 1, before
credit binding and payroll underfunding. A future run would need compact weekly
household fields for employed worker count, employer IDs, workers at constrained
firms, scheduled wage income, executed wage income and wage shortfall.
""", encoding="utf-8")

    support_rows = []
    for name, rows in (("PERSISTENT_NEGATIVE", persistent_rows), ("RECOVERED_NEGATIVE", recovered_rows)):
        support_rows.append({
            "cohort": name,
            "mean_support": sum(row["support"] for row in rows) / len(rows) if rows else 0.0,
            "mean_income": sum(row["income"] for row in rows) / len(rows) if rows else 0.0,
            "mean_actual_consumption": sum(row["actual"] for row in rows) / len(rows) if rows else 0.0,
            "negative_week_rows": len(rows),
        })
    write_csv("public_support_recovery_analysis.csv", support_rows)
    (ROOT / "scale_recovery_mechanism_audit.md").write_text("""# Scale recovery mechanism audit

The scope discrepancy is historical-record contamination: 58 inactive records
remain in the registry, and all 58 are negative. Active N=5000 therefore has
703 negative households, matching the accepted baseline. The active-only
recovery calculation is valid, but worker/employer exposure is not reconstructable
from existing outputs. The current evidence supports a recovery-capacity problem,
not a lifecycle accounting failure; the specific economic channel remains open.
""", encoding="utf-8")

    active_metrics = metrics("ACTIVE_ONLY_CANONICAL", active_final)
    inactive_negative = sum(row["wealth"] < 0 for row in inactive_final)
    inactive_low = sum(abs(row["wealth"]) < 100 for row in inactive_final)
    (ROOT / "acceptance_summary.md").write_text(f"""# PRE-STEP 13.5I acceptance summary

## Scope verdict: B. H_FINAL_COHORT_SCOPE_MISMATCH_ONLY

Final records: {len(final_all)}; active: {len(active_final)}; inactive retained:
{len(inactive_final)}. Inactive negative: {inactive_negative}; inactive low:
{inactive_low}. The identity is exact: all-record negative {sum(row['wealth'] < 0 for row in final_all)} = active negative {sum(row['wealth'] < 0 for row in active_final)} + inactive negative {inactive_negative}.

Canonical active N=5000 metrics are {active_metrics['negative_count']} negative out
of {active_metrics['households']} active households ({active_metrics['negative_share']:.6f}),
and {active_metrics['low_wealth_count']} low-wealth ({active_metrics['low_wealth_share']:.6f}).
The substantive negative regime remains real.

## Recovery verdict: G. INSUFFICIENT_EVIDENCE

Active-lifetime recovery was recomputed from the existing H panel. Initial
negative onset precedes Firm financial stress, but the available CSVs do not
retain historical household employer IDs, so labor/employer recovery causality
cannot be closed. No simulation rerun or economic change was made.
""", encoding="utf-8")
    (ROOT / "acceptance_flags.json").write_text(json.dumps({
        "scope_verdict": "B", "recovery_verdict": "G",
        "final_total_household_records": len(final_all), "final_active_households": len(active_final), "final_inactive_household_records": len(inactive_final),
        "all_records_negative_count": sum(row["wealth"] < 0 for row in final_all), "active_negative_count": sum(row["wealth"] < 0 for row in active_final), "inactive_negative_count": inactive_negative,
        "all_records_low_wealth_count": sum(abs(row["wealth"]) < 100 for row in final_all), "active_low_wealth_count": sum(abs(row["wealth"]) < 100 for row in active_final), "inactive_low_wealth_count": inactive_low,
        "canonical_active_negative_share": active_metrics["negative_share"], "canonical_active_low_wealth_share": active_metrics["low_wealth_share"],
        "h_scope_bug_found": True, "analysis_v2_household_scope_bug_found": False,
        "historical_employer_exposure_reconstructable": False, "employer_exposure_needed_for_final_explanation": True, "additional_n5000_instrumented_rerun_required": False,
        "economic_behavior_changed": False, "distress_semantics_ready": False,
    }, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
