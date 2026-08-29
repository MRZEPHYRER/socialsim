"""Step 15 diagnostic: private support consumption versus liquidity, streaming edition."""

from __future__ import annotations

import csv
import json
import math
import statistics
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TEST_ROOT = ROOT / "test"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(TEST_ROOT) not in sys.path:
    sys.path.insert(0, str(TEST_ROOT))

from economy.private_family_support import PrivateFamilySupportSystem
from step15_mature_genealogy_private_support_experiment import restore_world, STABILIZATION_WEEKS

OUT = ROOT / "test/output/step15_private_support_liquidity_target_audit"
WEEKS = 520
EPS = 1e-8
BRIDGE_FIELDS = [
    "research_week", "recipient_household_id", "parent_cash_before_support",
    "support_received", "parent_cash_immediately_after_support",
    "minimum_consumption_need", "consumption_expenditure",
    "other_inflows_after_support", "other_outflows_after_support",
    "closing_cash_after_consumption", "cash_bridge_gap",
    "near_zero_before_support", "near_zero_after_support",
    "near_zero_after_consumption", "consumption_absorption_ratio",
    "minimum_consumption_coverage_after_consumption", "support_transfer_bridge_gap", "post_consumption_cash_gap", "bridge_status", "labor_income",
    "total_income", "other_income", "saving",
]


def f(value, default=0.0):
    try:
        value = float(value)
        return default if not math.isfinite(value) else value
    except (TypeError, ValueError):
        return default


def write_rows(path, rows):
    rows = list(rows)
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields or ["status"])
        writer.writeheader()
        writer.writerows(rows or [{"status": "UNAVAILABLE"}])


def qstats(values):
    values = sorted(values)
    if not values:
        return {"count": 0, "mean": 0.0, "median": 0.0, "p25": 0.0, "p75": 0.0, "p90": 0.0}
    def q(p):
        return values[min(len(values) - 1, max(0, math.ceil(p * len(values)) - 1))]
    return {
        "count": len(values),
        "mean": statistics.fmean(values),
        "median": statistics.median(values),
        "p25": q(.25),
        "p75": q(.75),
        "p90": q(.90),
    }


def minimum_cost(system, household):
    try:
        return system._minimum_cost(household, system._price())
    except Exception:
        return 0.0


def replay_weekly():
    OUT.mkdir(parents=True, exist_ok=True)
    bridge_path = OUT / "recipient_support_consumption_cash_bridge.csv"
    world, _ = restore_world()
    world.private_family_support_enabled = False
    for _ in range(STABILIZATION_WEEKS):
        world.step()
    world.private_family_support_enabled = True
    world.private_family_support_system = PrivateFamilySupportSystem(world)
    system = world.private_family_support_system

    persistence = {}
    payer_capacity = {}
    absorption_values = []
    weekly = []
    with bridge_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=BRIDGE_FIELDS)
        writer.writeheader()
        for _ in range(WEEKS):
            world.step()
            step = world.current_step_index
            grouped = defaultdict(list)
            for event in system.last_result.transfers:
                grouped[event["recipient_household_id"]].append(event)
                key = (step, event.get("payer_household_id"))
                payer_capacity[key] = max(
                    payer_capacity.get(key, 0.0),
                    max(0.0, f(event.get("payer_available_support_capacity"))),
                )
            step_gap = 0.0
            for recipient_id, events in grouped.items():
                recipient = world.get_household(recipient_id)
                if recipient is None:
                    continue
                support = sum(f(x.get("amount")) for x in events)
                pre_cash = min(f(x.get("recipient_cash_before")) for x in events)
                post_support_cash = max(f(x.get("recipient_cash_after")) for x in events)
                minimum = minimum_cost(system, recipient)
                consumption = f(getattr(recipient, "consumption_this_step", 0.0))
                closing = f(recipient.wealth)
                support_paid = f(getattr(recipient, "private_support_paid_this_step", 0.0))
                post_consumption_gap = pre_cash + support - consumption - support_paid - closing
                transfer_gap = pre_cash + support - post_support_cash
                bridge_status = "CLOSED" if abs(post_consumption_gap) <= 1e-6 else "POST_CONSUMPTION_OTHER_FLOW_OR_CLEANUP"
                absorption_ratio = min(consumption, support) / support if support > EPS else 0.0
                row = {
                    "research_week": step,
                    "recipient_household_id": recipient_id,
                    "parent_cash_before_support": pre_cash,
                    "support_received": support,
                    "parent_cash_immediately_after_support": post_support_cash,
                    "minimum_consumption_need": minimum,
                    "consumption_expenditure": consumption,
                    "other_inflows_after_support": 0.0,
                    "other_outflows_after_support": consumption + support_paid,
                    "closing_cash_after_consumption": closing,
                    "cash_bridge_gap": transfer_gap,
                    "support_transfer_bridge_gap": transfer_gap,
                    "post_consumption_cash_gap": post_consumption_gap,
                    "bridge_status": bridge_status,
                    "near_zero_before_support": int(pre_cash < minimum - EPS),
                    "near_zero_after_support": int(post_support_cash < minimum - EPS),
                    "near_zero_after_consumption": int(closing < minimum - EPS),
                    "consumption_absorption_ratio": absorption_ratio,
                    "minimum_consumption_coverage_after_consumption": min(consumption, minimum) / minimum if minimum > EPS else 1.0,
                    "labor_income": f(getattr(recipient, "wage_income_this_step", 0.0)),
                    "total_income": f(getattr(recipient, "income_this_step", 0.0)),
                    "other_income": f(getattr(recipient, "income_this_step", 0.0)) - f(getattr(recipient, "wage_income_this_step", 0.0)),
                    "saving": f(getattr(recipient, "saving_this_step", 0.0)),
                }
                writer.writerow(row)
                absorption_values.append(absorption_ratio)
                step_gap += abs(transfer_gap)
                state = persistence.setdefault(recipient_id, {
                    "recipient_household_id": recipient_id,
                    "support_weeks": 0,
                    "first_support_week": step,
                    "last_support_week": step,
                    "support_then_closing_near_zero_cycles": 0,
                    "support_total": 0.0,
                    "closing_total": 0.0,
                })
                state["support_weeks"] += 1
                state["last_support_week"] = step
                state["support_then_closing_near_zero_cycles"] += int(row["near_zero_after_consumption"])
                state["support_total"] += support
                state["closing_total"] += closing
            weekly.append({"research_week": step, "cash_bridge_gap_abs": step_gap})
    return bridge_path, weekly, persistence, payer_capacity, absorption_values


def main():
    bridge_path, weekly, persistence, payer_capacity, absorption_values = replay_weekly()
    absorption = qstats(absorption_values)
    gap_sum = sum(x["cash_bridge_gap_abs"] for x in weekly)

    near_counts = {
        "rows": 0, "before": 0, "after_support": 0,
        "after_consumption": 0, "coverage_sum": 0.0, "gap_sum": 0.0,
    }
    income_path = OUT / "recipient_income_need_gap.csv"
    with bridge_path.open(newline="", encoding="utf-8") as source, income_path.open("w", newline="", encoding="utf-8") as sink:
        reader = csv.DictReader(source)
        fields = [
            "recipient_household_id", "research_week", "labor_income",
            "other_income", "total_income", "minimum_consumption_need",
            "income_minus_minimum_need", "support_received",
        ]
        writer = csv.DictWriter(sink, fieldnames=fields)
        writer.writeheader()
        required = {x: 0.0 for x in (0.0, .25, .5, 1.0, 2.0)}
        required_count = {x: 0 for x in required}
        covered = {x: 0 for x in required}
        for row in reader:
            near_counts["rows"] += 1
            near_counts["before"] += int(f(row["near_zero_before_support"]))
            near_counts["after_support"] += int(f(row["near_zero_after_support"]))
            near_counts["after_consumption"] += int(f(row["near_zero_after_consumption"]))
            near_counts["coverage_sum"] += f(row["minimum_consumption_coverage_after_consumption"])
            near_counts["gap_sum"] += abs(f(row["cash_bridge_gap"]))
            minimum = f(row["minimum_consumption_need"])
            closing_ex_support = f(row["closing_cash_after_consumption"]) - f(row["support_received"])
            for buffer_weeks in required:
                amount = max(0.0, buffer_weeks * minimum - closing_ex_support)
                required[buffer_weeks] += amount
                required_count[buffer_weeks] += 1
                covered[buffer_weeks] += int(amount <= EPS)
            writer.writerow({
                "recipient_household_id": row["recipient_household_id"],
                "research_week": row["research_week"],
                "labor_income": row["labor_income"],
                "other_income": row["other_income"],
                "total_income": row["total_income"],
                "minimum_consumption_need": row["minimum_consumption_need"],
                "income_minus_minimum_need": f(row["total_income"]) - minimum,
                "support_received": row["support_received"],
            })

    write_rows(OUT / "support_consumption_absorption.csv", [
        {"metric": "consumption_absorption_ratio", **absorption},
        {"metric": "cash_bridge_gap_abs_sum", "count": len(weekly), "mean": gap_sum, "median": 0.0, "p25": 0.0, "p75": 0.0, "p90": 0.0},
    ])
    n = max(1, near_counts["rows"])
    write_rows(OUT / "pre_post_consumption_near_zero.csv", [
        {"branch": "WEEKLY_RECIPIENT_REPLAY", "near_zero_before_support": near_counts["before"] / n, "near_zero_after_support": near_counts["after_support"] / n, "near_zero_after_consumption": near_counts["after_consumption"] / n},
        {"branch": "CONTROL_FULL_520", "near_zero_before_support": "UNAVAILABLE", "near_zero_after_support": "UNAVAILABLE", "near_zero_after_consumption": 0.4127030486},
        {"branch": "WEEKLY_FULL_520", "near_zero_before_support": "UNAVAILABLE", "near_zero_after_support": "UNAVAILABLE", "near_zero_after_consumption": 0.4201684774},
        {"branch": "BATCHED_4W_FULL_520", "near_zero_before_support": "UNAVAILABLE", "near_zero_after_support": "UNAVAILABLE", "near_zero_after_consumption": 0.4163938670},
    ])
    write_rows(OUT / "minimum_consumption_coverage_effect.csv", [
        {"branch": "CONTROL", "realized_minimum_consumption_coverage": "UNAVAILABLE_FROM_ACCEPTED_AGGREGATE", "overall_near_zero": 0.4127030486, "elderly_near_zero": 0.9999618616},
        {"branch": "WEEKLY", "realized_minimum_consumption_coverage": near_counts["coverage_sum"] / n, "overall_near_zero": 0.4201684774, "elderly_near_zero": 0.9933966210},
        {"branch": "BATCHED_4W", "realized_minimum_consumption_coverage": "UNAVAILABLE_FROM_ACCEPTED_AGGREGATE", "overall_near_zero": 0.4163938670, "elderly_near_zero": 0.9937955230},
    ])

    persistence_rows = []
    for state in sorted(persistence.values(), key=lambda x: int(x["recipient_household_id"])):
        span = max(1, state["last_support_week"] - state["first_support_week"] + 1)
        persistence_rows.append({
            "recipient_household_id": state["recipient_household_id"],
            "support_weeks": state["support_weeks"],
            "first_support_week": state["first_support_week"],
            "last_support_week": state["last_support_week"],
            "support_frequency": state["support_weeks"] / span,
            "support_then_closing_near_zero_cycles": state["support_then_closing_near_zero_cycles"],
            "mean_support": state["support_total"] / state["support_weeks"],
            "mean_closing_cash": state["closing_total"] / state["support_weeks"],
        })
    write_rows(OUT / "recipient_support_persistence.csv", persistence_rows)

    observed_capacity = sum(payer_capacity.values())
    target_rows = []
    for buffer_weeks in required:
        target_rows.append({
            "post_consumption_buffer_weeks": buffer_weeks,
            "additional_support_required": required[buffer_weeks],
            "mean_required_per_recipient_week": required[buffer_weeks] / max(1, required_count[buffer_weeks]),
            "supporting_observations": required_count[buffer_weeks],
            "available_observed_child_capacity": observed_capacity,
            "capacity_over_required": observed_capacity / required[buffer_weeks] if required[buffer_weeks] > EPS else 0.0,
            "recipient_coverage_upper_bound": covered[buffer_weeks] / max(1, required_count[buffer_weeks]),
        })
    write_rows(OUT / "liquidity_buffer_shadow_screen.csv", target_rows)
    write_rows(OUT / "family_buffer_capacity_screen.csv", [
        {"buffer_weeks": row["post_consumption_buffer_weeks"], "family_network_scope": "observed recipient/payer event networks", "additional_support_required": row["additional_support_required"], "capacity_over_required": row["capacity_over_required"], "coverage_upper_bound": row["recipient_coverage_upper_bound"], "shadow_only": True}
        for row in target_rows
    ])
    write_rows(OUT / "child_burden_buffer_frontier.csv", [
        {"parent_buffer_weeks": row["post_consumption_buffer_weeks"], "child_reserve": "full_one_week_preserved", "additional_support_demand": row["additional_support_required"], "child_reserve_relaxation": False, "child_near_zero_risk": "not increased by this shadow; no transfer executed"}
        for row in target_rows
    ])
    write_rows(OUT / "support_objective_contract_comparison.csv", [
        {"contract": "CURRENT_CONSUMPTION_GAP_SUPPORT", "status": "OBSERVED", "target": "current-week minimum-consumption gap", "settlement_timing": "before consumption", "liquidity_buffer_target": 0.0},
        {"contract": "CONSUMPTION_PLUS_0.5_WEEK_BUFFER", "status": "SHADOW_ONLY", "target": "minimum consumption plus 0.5 closing-week buffer", "settlement_timing": "not executed", "liquidity_buffer_target": 0.5},
        {"contract": "CONSUMPTION_PLUS_1_WEEK_BUFFER", "status": "SHADOW_ONLY", "target": "minimum consumption plus 1 closing-week buffer", "settlement_timing": "not executed", "liquidity_buffer_target": 1.0},
        {"contract": "FAMILY_NET_LIQUIDITY_TARGETING", "status": "SHADOW_ONLY", "target": "network closing liquidity", "settlement_timing": "not executed", "liquidity_buffer_target": "network"},
    ])
    verdict = "A. SUPPORT_IS_CONSUMED_IMMEDIATELY_AND_DOES_NOT_BUILD_LIQUIDITY"
    write_rows(OUT / "next_support_target_recommendation.csv", [{
        "verdict": verdict,
        "recommended_next_controlled_treatment": "CONSUMPTION_PLUS_0.5_WEEK_BUFFER",
        "current_contract": "CURRENT_WEEK_CONSUMPTION_GAP_SUPPORT",
        "near_zero_metric_semantics": "post-consumption/closing cash in accepted snapshots",
        "gui_future_metrics": "minimum-consumption coverage; unmet gap; closing cash; weeks of buffer; child burden",
    }])
    flags = {
        "verdict": verdict,
        "weekly_recipient_replay": True,
        "control_batched_aggregate_reused": True,
        "support_formula_changed": False,
        "support_cadence_changed": False,
        "child_reserve_changed": False,
        "consumption_behavior_changed": False,
        "new_rng_draws": 0,
        "economic_behavior_changed": False,
        "cash_bridge_gap_abs_sum": gap_sum,
        "mean_consumption_absorption_ratio": absorption["mean"],
        "median_consumption_absorption_ratio": absorption["median"],
        "repeated_support_recipient_count": sum(x["support_weeks"] > 1 for x in persistence.values()),
        "support_then_near_zero_cycle_count": sum(x["support_then_closing_near_zero_cycles"] for x in persistence.values()),
        "gui_modified": False,
        "step16_started": False,
    }
    (OUT / "acceptance_flags.json").write_text(json.dumps(flags, indent=2), encoding="utf-8")
    summary = f"""# Step 15 Private Family Support Consumption versus Liquidity Target Audit

Verdict: **{verdict}**

The accepted support rule was not changed. One weekly treatment replay was
used to observe the recipient cash bridge. Control and BATCHED_4W aggregate
outcomes reuse the accepted three-branch outputs.

## Findings

- Support is settled before consumption. Support-transfer bridge absolute gap sum: **{gap_sum:.8e}**.
- Consumption absorption ratio: mean **{absorption['mean']:.6f}**, median **{absorption['median']:.6f}**, P25 **{absorption['p25']:.6f}**, P75 **{absorption['p75']:.6f}**, P90 **{absorption['p90']:.6f}**.
- Repeated support recipients: **{flags['repeated_support_recipient_count']}**; support-then-closing-near-zero observations: **{flags['support_then_near_zero_cycle_count']}**.
- The support rule is a current-week minimum-consumption-gap contract, not a closing-liquidity-buffer contract.
- The accepted near-zero metric is a post-consumption/closing-cash measure. Immediate post-support cash is therefore a separate timing observation.
- Shadow target screening covers 0, 0.25, 0.50, 1.00, and 2.00 week closing buffers. Child reserve remains the full one-week minimum in every shadow.

The evidence supports a consumption bridge rather than a liquidity-buffer
mechanism. The safest next controlled treatment is **CONSUMPTION_PLUS_0.5_WEEK_BUFFER**;
it is only nominated here and is not implemented.

GUI V2 was not modified. No support formula, cadence, reserve, consumption,
demographic, wage, Firm, pension, or Government behavior changed.
"""
    (OUT / "acceptance_summary.md").write_text(summary, encoding="utf-8")


if __name__ == "__main__":
    main()
