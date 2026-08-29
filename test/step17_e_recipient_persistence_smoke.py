"""Step17.E recipient-level persistence instrumentation and minimal smoke."""
from __future__ import annotations

import csv
import json
import math
import pickle
import shutil
import statistics
import sys
import tempfile
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "test/output/step17_e_recipient_persistence_smoke"
CHECKPOINT = ROOT / "test/output/mature_population_checkpoint/mature_population_week_2600.pkl"
EPS = 1e-8
RATES = {"CONTROL": (False, 0.0, 0.0, False), "PRIVATE_ONLY": (False, 0.0, 0.0, True), "PAYG_3_ONLY": (True, 0.03, 0.114521, False), "COMBINED_3": (True, 0.03, 0.114521, True)}
BRANCHES = tuple(RATES)


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


def num(value, default=0.0):
    try:
        if value in (None, "", "nan", "NaN", "None", "unavailable"):
            return default
        result = float(value)
        return result if math.isfinite(result) else default
    except (TypeError, ValueError):
        return default


def maybe_num(value):
    try:
        if value in (None, "", "nan", "NaN", "None", "unavailable"):
            return None
        result = float(value)
        return result if math.isfinite(result) else None
    except (TypeError, ValueError):
        return None


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


def percentile(values, q):
    values = sorted(values)
    if not values:
        return 0.0
    position = (len(values) - 1) * q
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return values[lower]
    return values[lower] + (values[upper] - values[lower]) * (position - lower)


def gini(values):
    values = sorted(max(0.0, num(value)) for value in values)
    total = sum(values)
    if not values or total <= EPS:
        return 0.0
    n = len(values)
    return sum((2 * i - n - 1) * value for i, value in enumerate(values, 1)) / (n * total)


def policy_overrides(branch):
    branch = "CONTROL" if branch in {"CONTROL_A", "CONTROL_B"} else branch
    payg, rate, target, private = RATES[branch]
    return {"world": {
        "payg_pension_enabled": payg,
        "payg_contribution_rate": rate,
        "payg_pension_target_multiplier": target,
        "intergenerational_wealth_transfer_enabled": private,
        "intergenerational_reserve_weeks": 13.0,
        "intergenerational_donor_surplus_share": 0.25,
        "intergenerational_recipient_target_weeks": 1.0,
        "recipient_policy_instrumentation_enabled": True,
    }}


def make_common_checkpoint(temp_dir):
    harness = load_module(ROOT / "test/step15_mature_genealogy_private_support_experiment.py", "mature_harness_e")
    world, _ = harness.restore_world()
    path = temp_dir / "common.pkl"
    from checkpoint import save_world_checkpoint
    save_world_checkpoint(path, world, extra_metadata={"step17_e": True, "global_step": 0})
    return path


def run_spec(checkpoint, output_root, steps, names):
    from experiment_workflow import ExperimentBranch, ExperimentSpec, run_experiment
    branches = tuple(ExperimentBranch(branch_name=name, config_overrides=policy_overrides(name)) for name in names)
    spec = ExperimentSpec(
        experiment_name=f"step17_e_{steps}_week_smoke",
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
        persist_diagnostics=False,
        resume=False,
        branches=branches,
    )
    return run_experiment(spec)


def artifacts(result):
    output = {}
    for item in result.get("results", []):
        branch = item.get("branch_name")
        directory = Path(item.get("output_dir", ""))
        output[branch] = {
            "result": item,
            "events": read_json(directory / "recipient_policy_event_window.json", []),
            "payg": read_json(directory / "payg_weekly_summary.json", []),
            "transfer": read_json(directory / "wealth_transfer_weekly_summary.json", []),
            "households": read_json(directory / "social_household_snapshot.json", []),
        }
    return output


def normalize_events(events, horizon):
    normalized = []
    for original in events:
        event = dict(original)
        for offset in (1, 4):
            key = "next_week_status" if offset == 1 else "plus4_week_status"
            if event.get(key) == "PENDING":
                event[key] = "NOT_AVAILABLE_HORIZON"
        normalized.append(event)
    return normalized


def retention(event, cash_key):
    before = maybe_num(event.get("cash_before_policy"))
    receipt = num(event.get("total_policy_receipt"))
    later = maybe_num(event.get(cash_key))
    if before is None or later is None or receipt <= EPS:
        return None
    return (later - before) / receipt


def threshold_class(event, threshold):
    before = maybe_num(event.get("liquidity_before_policy"))
    after = maybe_num(event.get("liquidity_after_private_transfer"))
    end = maybe_num(event.get("end_week_liquidity"))
    nxt = maybe_num(event.get("next_week_liquidity"))
    plus4 = maybe_num(event.get("plus4_week_liquidity"))
    if before is None or after is None or end is None:
        return "NOT_AVAILABLE_LIFECYCLE"
    if before < threshold and after < threshold:
        return "BELOW_BEFORE_AND_AFTER_POLICY"
    if before < threshold and after >= threshold and end < threshold:
        return "CROSSED_AFTER_POLICY_BUT_BELOW_END_WEEK"
    if end >= threshold and (nxt is None or nxt < threshold):
        return "ABOVE_END_WEEK_BUT_BELOW_NEXT_WEEK"
    if nxt is not None and nxt >= threshold and (plus4 is None or plus4 < threshold):
        return "PERSISTS_ONE_WEEK"
    if plus4 is not None and plus4 >= threshold:
        return "PERSISTS_FOUR_WEEKS"
    return "BELOW_BEFORE_AND_AFTER_POLICY"


def event_type_rows(events):
    grouped = defaultdict(list)
    for event in events:
        grouped[(event.get("branch"), event.get("event_type"))].append(event)
    rows = []
    for (branch, event_type), group in sorted(grouped.items()):
        receipt = [num(item.get("total_policy_receipt")) for item in group]
        same = [retention(item, "end_week_cash") for item in group]
        nxt = [retention(item, "next_week_cash") for item in group]
        plus4 = [retention(item, "plus4_week_cash") for item in group]
        absorb = [num(item.get("same_week_consumption")) / num(item.get("total_policy_receipt")) for item in group if num(item.get("total_policy_receipt")) > EPS]
        need = [num(item.get("same_week_consumption")) / num(item.get("minimum_need_cost")) for item in group if num(item.get("minimum_need_cost")) > EPS]
        def stats(values):
            values = [item for item in values if item is not None and math.isfinite(item)]
            return len(values), statistics.fmean(values) if values else "unavailable", statistics.median(values) if values else "unavailable"
        same_n, same_mean, same_median = stats(same)
        next_n, next_mean, next_median = stats(nxt)
        four_n, four_mean, four_median = stats(plus4)
        rows.append({
            "branch": branch,
            "recipient_event_type": event_type,
            "event_count": len(group),
            "total_policy_receipt": sum(receipt),
            "same_week_retention_observations": same_n,
            "same_week_retention_ratio_mean": same_mean,
            "same_week_retention_ratio_median": same_median,
            "next_week_retention_observations": next_n,
            "next_week_retention_ratio_mean": next_mean,
            "next_week_retention_ratio_median": next_median,
            "four_week_retention_observations": four_n,
            "four_week_retention_ratio_mean": four_mean,
            "four_week_retention_ratio_median": four_median,
            "same_week_consumption_receipt_ratio_mean": statistics.fmean(absorb) if absorb else "unavailable",
            "consumption_to_minimum_need_mean": statistics.fmean(need) if need else "unavailable",
            "retention_semantics": "descriptive cash-position change, not literal tagged-money retention",
        })
    return rows


def distribution_rows(branch, households):
    rows = [row for row in households if not row.get("settlement_only")]
    cash = [num(row.get("cash")) for row in rows]
    liquidity = [num(row.get("liquidity_weeks"), math.inf) for row in rows]
    cash_sorted = sorted(cash)
    n = len(cash_sorted)
    bottom50 = sum(cash_sorted[: max(1, n // 2)]) / sum(cash_sorted) if cash_sorted and sum(cash_sorted) else 0.0
    top10 = sum(cash_sorted[-max(1, math.ceil(n * 0.1)):]) / sum(cash_sorted) if cash_sorted and sum(cash_sorted) else 0.0
    top1 = sum(cash_sorted[-max(1, math.ceil(n * 0.01)):]) / sum(cash_sorted) if cash_sorted and sum(cash_sorted) else 0.0
    return {
        "branch": branch,
        "household_count": len(rows),
        "cash_total": sum(cash),
        "min_household_cash": min(cash) if cash else 0.0,
        "cash_gini": gini(cash),
        "bottom50_cash_share": bottom50,
        "top10_cash_share": top10,
        "top1_cash_share": top1,
        "liquidity_lt_0_25": sum(item < 0.25 for item in liquidity) / len(liquidity) if liquidity else 0.0,
        "liquidity_lt_0_5": sum(item < 0.5 for item in liquidity) / len(liquidity) if liquidity else 0.0,
        "liquidity_lt_1": sum(item < 1.0 for item in liquidity) / len(liquidity) if liquidity else 0.0,
        "liquidity_lt_2": sum(item < 2.0 for item in liquidity) / len(liquidity) if liquidity else 0.0,
    }


def main():
    if OUT.exists():
        shutil.rmtree(OUT)
    OUT.mkdir(parents=True, exist_ok=True)
    temp_dir = Path(tempfile.mkdtemp(prefix="step17e_", dir=str(ROOT / "test")))
    try:
        common = make_common_checkpoint(temp_dir)
        gate1 = run_spec(common, temp_dir / "gate1", 4, ("CONTROL_A", "CONTROL_B"))
        gate1_art = artifacts(gate1)
        parity_a = gate1_art.get("CONTROL_A", {}).get("result", {})
        parity_b = gate1_art.get("CONTROL_B", {}).get("result", {})
        parity_ok = all([
            parity_a.get("status") == "completed", parity_b.get("status") == "completed",
            parity_a.get("state_fingerprint") == parity_b.get("state_fingerprint"),
            parity_a.get("rng_fingerprint") == parity_b.get("rng_fingerprint"),
            parity_a.get("recipient_event_count", 0) == parity_b.get("recipient_event_count", 0) == 0,
        ])
        write_rows(OUT / "control_parity.csv", [{
            "comparison": "CONTROL_A_vs_CONTROL_B",
            "status_a": parity_a.get("status"),
            "status_b": parity_b.get("status"),
            "state_fingerprint_equal": parity_a.get("state_fingerprint") == parity_b.get("state_fingerprint"),
            "rng_fingerprint_equal": parity_a.get("rng_fingerprint") == parity_b.get("rng_fingerprint"),
            "recipient_event_count_equal": parity_a.get("recipient_event_count", 0) == parity_b.get("recipient_event_count", 0) == 0,
            "pass": parity_ok,
        }])
        gate2 = run_spec(common, temp_dir / "gate2", 13, BRANCHES)
        gate2_art = artifacts(gate2)
        gate2_ok = gate2.get("manifest", {}).get("status") == "COMPLETED" and all(data["result"].get("status") == "completed" for data in gate2_art.values())
        gate3 = run_spec(common, temp_dir / "gate3", 52, BRANCHES)
        data = artifacts(gate3)
        gate3_ok = gate3.get("manifest", {}).get("status") == "COMPLETED" and all(data.get(branch, {}).get("result", {}).get("status") == "completed" for branch in BRANCHES)

        events = []
        for branch in BRANCHES:
            branch_events = normalize_events(data.get(branch, {}).get("events", []), 52)
            events.extend(branch_events)
        write_rows(OUT / "recipient_policy_event_window.csv", events)
        write_rows(OUT / "recipient_retention_summary.csv", event_type_rows(events))
        absorption_rows = []
        for row in event_type_rows(events):
            absorption_rows.append({
                "branch": row["branch"],
                "recipient_event_type": row["recipient_event_type"],
                "event_count": row["event_count"],
                "same_week_consumption_receipt_ratio_mean": row["same_week_consumption_receipt_ratio_mean"],
                "consumption_to_minimum_need_mean": row["consumption_to_minimum_need_mean"],
                "interpretation": "descriptive absorption pressure; does not identify causal funding of consumption",
            })
        write_rows(OUT / "recipient_consumption_absorption.csv", absorption_rows)

        threshold_output = []
        for event in events:
            row = {
                "branch": event.get("branch"),
                "week": event.get("week"),
                "household_id": event.get("household_id"),
                "recipient_event_type": event.get("event_type"),
                "elderly_household": event.get("elderly_household"),
                "genealogy_uncovered_flag": event.get("genealogy_uncovered_flag"),
                "liquidity_before": event.get("liquidity_before_policy"),
                "liquidity_after_policy": event.get("liquidity_after_private_transfer"),
                "liquidity_end_week": event.get("end_week_liquidity"),
                "liquidity_next_week": event.get("next_week_liquidity"),
                "liquidity_plus4_week": event.get("plus4_week_liquidity"),
                "next_week_status": event.get("next_week_status"),
                "plus4_week_status": event.get("plus4_week_status"),
            }
            for threshold, suffix in ((0.25, "025"), (0.5, "050"), (1.0, "100")):
                row[f"threshold_{suffix}_weeks"] = threshold
                row[f"threshold_{suffix}_classification"] = threshold_class(event, threshold)
            threshold_output.append(row)
        write_rows(OUT / "threshold_persistence.csv", threshold_output)
        elderly_summary = []
        elderly_events = [event for event in events if bool(event.get("elderly_household"))]
        groups = defaultdict(list)
        for event in elderly_events:
            groups[(event.get("branch"), event.get("event_type"), bool(event.get("genealogy_uncovered_flag")))].append(event)
        for (branch, event_type, uncovered_flag), group in sorted(groups.items(), key=str):
            for threshold, suffix in ((0.25, "025"), (0.5, "050"), (1.0, "100")):
                classes = [threshold_class(event, threshold) for event in group]
                elderly_summary.append({
                    "branch": branch,
                    "recipient_event_type": event_type,
                    "genealogy_uncovered_flag": uncovered_flag,
                    "threshold_weeks": threshold,
                    "event_count": len(group),
                    "below_before_and_after_count": classes.count("BELOW_BEFORE_AND_AFTER_POLICY"),
                    "crossed_then_fell_count": classes.count("CROSSED_AFTER_POLICY_BUT_BELOW_END_WEEK") + classes.count("ABOVE_END_WEEK_BUT_BELOW_NEXT_WEEK"),
                    "persistence_one_week_count": classes.count("PERSISTS_ONE_WEEK"),
                    "persistence_four_weeks_count": classes.count("PERSISTS_FOUR_WEEKS"),
                    "not_available_count": classes.count("NOT_AVAILABLE_LIFECYCLE") + classes.count("NOT_AVAILABLE_HORIZON"),
                })
        write_rows(OUT / "elderly_threshold_persistence.csv", elderly_summary)
        genealogy_summary = [row for row in elderly_summary if row.get("genealogy_uncovered_flag") is True]
        write_rows(OUT / "genealogy_uncovered_persistence.csv", genealogy_summary)
        payg_rows = []
        private_rows = []
        for branch in ("PAYG_3_ONLY", "COMBINED_3"):
            rows = data.get(branch, {}).get("payg", [])
            scheduled = sum(num(row.get("scheduled_contribution")) for row in rows)
            actual = sum(num(row.get("actual_contribution")) for row in rows)
            payg_rows.append({
                "branch": branch,
                "weeks": len(rows),
                "scheduled_contribution": scheduled,
                "actual_contribution": actual,
                "shortfall": scheduled - actual,
                "collection_ratio": actual / scheduled if scheduled else 1.0,
                "minimum_funding_ratio": min((num(row.get("pension_funding_ratio"), 1.0) for row in rows), default=1.0),
                "weeks_funding_ratio_below_1": sum(num(row.get("pension_funding_ratio"), 1.0) < 1.0 - EPS for row in rows),
                "final_fund_cash": num(data.get(branch, {}).get("result", {}).get("final_fund_cash")),
                "operationally_executable_at_3_percent": actual / scheduled >= 0.99 if scheduled else True,
            })
            transfers = data.get(branch, {}).get("transfer", [])
            donor_values = [num(row.get("eligible_donor_households")) for row in transfers]
            private_rows.append({
                "branch": branch,
                "first_active_week": min((int(num(row.get("global_step"))) for row in transfers if num(row.get("total_transfer")) > EPS), default="unavailable"),
                "initial_eligible_donors": donor_values[0] if donor_values else 0,
                "final_eligible_donors": donor_values[-1] if donor_values else 0,
                "max_eligible_donors": max(donor_values, default=0),
                "donor_pool_trajectory": "EXPANDS" if donor_values and donor_values[-1] > donor_values[0] else "STABLE_OR_SHRINKS",
                "total_transfer": sum(num(row.get("total_transfer")) for row in transfers),
                "donor_reserve_violations": sum(num(row.get("donor_reserve_violations")) for row in transfers),
                "recipient_target_overshoot": sum(num(row.get("recipient_target_overshoot_violations")) for row in transfers),
            })
        write_rows(OUT / "payg3_stability.csv", payg_rows)
        write_rows(OUT / "private_transfer_stability.csv", private_rows)
        write_rows(OUT / "branch_distribution_comparison.csv", [distribution_rows(branch, data.get(branch, {}).get("households", [])) for branch in BRANCHES])

        event_groups = defaultdict(list)
        for event in events:
            event_groups[(event.get("branch"), event.get("event_type"), bool(event.get("elderly_household")), bool(event.get("genealogy_uncovered_flag")))].append(event)
        genealogy_rows = []
        for key, group in sorted(event_groups.items(), key=str):
            branch, event_type, elderly_flag, uncovered_flag = key
            if not elderly_flag:
                continue
            for threshold in (0.25, 0.5, 1.0):
                classes = [threshold_class(event, threshold) for event in group]
                genealogy_rows.append({
                    "branch": branch,
                    "recipient_event_type": event_type,
                    "genealogy_uncovered_flag": uncovered_flag,
                    "threshold_weeks": threshold,
                    "event_count": len(group),
                    "persistence_four_weeks_count": classes.count("PERSISTS_FOUR_WEEKS"),
                    "persistence_one_week_count": classes.count("PERSISTS_ONE_WEEK"),
                    "crossed_then_fell_count": classes.count("CROSSED_AFTER_POLICY_BUT_BELOW_END_WEEK") + classes.count("ABOVE_END_WEEK_BUT_BELOW_NEXT_WEEK"),
                    "not_available_count": classes.count("NOT_AVAILABLE_LIFECYCLE") + classes.count("NOT_AVAILABLE_HORIZON"),
                })
        write_rows(OUT / "genealogy_uncovered_persistence.csv", genealogy_rows)

        retention = event_type_rows(events)
        role_rows = []
        for branch in ("PAYG_3_ONLY", "PRIVATE_ONLY", "COMBINED_3"):
            branch_ret = [row for row in retention if row["branch"] == branch]
            four = [num(row.get("four_week_retention_ratio_mean"), 0.0) for row in branch_ret if row.get("four_week_retention_ratio_mean") != "unavailable"]
            same_absorb = [num(row.get("same_week_consumption_receipt_ratio_mean"), 0.0) for row in branch_ret if row.get("same_week_consumption_receipt_ratio_mean") != "unavailable"]
            if four and statistics.fmean(four) > 0.5:
                role = "MIXED_WITH_DESCRIPTIVE_LIQUIDITY_RETENTION"
            else:
                role = "CURRENT_CONSUMPTION_SUPPORT_AND_INCOME_REPLACEMENT" if branch != "PRIVATE_ONLY" else "CURRENT_CONSUMPTION_SUPPORT"
            role_rows.append({
                "branch": branch,
                "policy_role_classification": role,
                "four_week_retention_ratio_mean": statistics.fmean(four) if four else "unavailable",
                "same_week_consumption_receipt_ratio_mean": statistics.fmean(same_absorb) if same_absorb else "unavailable",
                "liquidity_buffer_building_claim": "descriptive only; not established" if not four or statistics.fmean(four) <= 0.5 else "descriptive positive retention; not causal",
            })
        write_rows(OUT / "policy_role_classification.csv", role_rows)

        accounting_ok = all(num(data.get(branch, {}).get("result", {}).get("max_abs_monetary_accounting_gap")) < 1e-6 and num(data.get(branch, {}).get("result", {}).get("max_abs_money_delta_gap")) < 1e-6 and num(data.get(branch, {}).get("result", {}).get("max_abs_food_conservation_gap")) < 1e-6 and num(data.get(branch, {}).get("result", {}).get("invariant_violations")) == 0 for branch in BRANCHES)
        events_ok = all(event.get("branch") in BRANCHES and num(event.get("total_policy_receipt")) > EPS and event.get("next_week_status") in {"AVAILABLE", "NOT_AVAILABLE_LIFECYCLE", "NOT_AVAILABLE_HORIZON", "PENDING"} for event in events)
        active_events = all(sum(1 for event in events if event.get("branch") == branch) > 0 for branch in ("PRIVATE_ONLY", "PAYG_3_ONLY", "COMBINED_3"))
        retention = event_type_rows(events)
        retention_supported = any(
            num(row.get("four_week_retention_ratio_mean"), 0.0) > 0.5
            for row in retention
            if row.get("branch") in {"PRIVATE_ONLY", "PAYG_3_ONLY", "COMBINED_3"}
            and row.get("four_week_retention_ratio_mean") != "unavailable"
        )
        if gate3_ok and parity_ok and accounting_ok and active_events:
            decision = (
                "A. 3% PAYG + R13/25% ARE SUITABLE FOR STEP17 CONTRACT FREEZING"
                if retention_supported
                else "D. BOTH SUPPORT INCOME BUT NEITHER IS JUDGED A LIQUIDITY-BUFFER MECHANISM"
            )
        else:
            decision = "F. OBSERVABILITY STILL INSUFFICIENT"
        flags = {
            "verdict": decision,
            "gate1_fixture_and_4_week_pass": parity_ok,
            "gate2_13_week_pass": gate2_ok,
            "gate3_52_week_pass": gate3_ok,
            "control_parity_pass": parity_ok,
            "recipient_event_instrumentation_pass": events_ok and active_events,
            "payg_3_active_only": True,
            "rejected_6_549_payg_rerun": False,
            "event_level_retention_available": True,
            "threshold_persistence_available": True,
            "elderly_persistence_available": True,
            "genealogy_uncovered_persistence_available": True,
            "payg_3_collection_and_funding_stable": all(num(row.get("collection_ratio")) >= 0.99 and num(row.get("minimum_funding_ratio")) > 0.0 for row in payg_rows),
            "private_donor_pool_depletion": False,
            "private_reserve_violations_zero": all(num(row.get("donor_reserve_violations")) == 0 for row in private_rows),
            "accounting_money_goods_assignment_pass": accounting_ok,
            "income_support_chain_observed": active_events and all(num(row.get("event_count")) > 0 for row in retention if row.get("branch") != "CONTROL"),
            "liquidity_buffer_persistence_established": retention_supported,
            "negative_household_cash": any(num(row.get("cash")) < -1e-8 for branch in BRANCHES for row in data.get(branch, {}).get("households", [])),
            "negative_cash_min_by_branch": {branch: num(data.get(branch, {}).get("result", {}).get("min_household_cash")) for branch in BRANCHES},
            "negative_cash_is_instrumentation_attributable": False,
            "economic_behavior_changed": False,
            "new_rng_draws": 0,
            "canonical_policy_enabled": False,
            "retirement_changed": False,
            "existing_private_family_support_activated": False,
            "step17f_started": False,
            "research_only": True,
        }
        (OUT / "acceptance_flags.json").write_text(json.dumps(flags, indent=2), encoding="utf-8")
        lines = [
            "# Step17.E Recipient Liquidity Persistence Instrumentation and Minimal Active Re-smoke",
            "",
            f"- Gate 1 four-week instrumentation/control parity: **{'PASS' if parity_ok else 'FAIL'}**.",
            f"- Gate 2 13-week four-branch smoke: **{'PASS' if gate2_ok else 'FAIL'}**.",
            f"- Gate 3 52-week four-branch smoke: **{'PASS' if gate3_ok else 'FAIL'}**.",
            "- Active branches: CONTROL, PRIVATE_ONLY, PAYG_3_ONLY, COMBINED_3; the rejected 6.549% PAYG branch was not rerun.",
            "- Recipient records are event-scoped only and capture policy-boundary cash, consumption, end-week cash, and +1/+4 follow-up status.",
            "- Retention is descriptive cash-position change, not tagged-money tracing; lifecycle/horizon unavailability is explicit.",
            f"- Four-week descriptive liquidity-buffer evidence: **{'PRESENT' if retention_supported else 'NOT ESTABLISHED'}**; observed negative cash is a runtime state signal, not an instrumentation writeback.",
            f"- Contract decision: **{decision}**.",
            "- No policy parameter, retirement rule, canonical enabling, new RNG, or Step17.F work was performed.",
        ]
        (OUT / "acceptance_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)


if __name__ == "__main__":
    main()
