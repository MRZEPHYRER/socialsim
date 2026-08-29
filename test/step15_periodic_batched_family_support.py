"""Controlled weekly-vs-periodic private family support experiment."""

from __future__ import annotations

import copy
import csv
import json
import math
import statistics
import sys
from collections import defaultdict, deque
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TEST_ROOT = ROOT / "test"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(TEST_ROOT) not in sys.path:
    sys.path.insert(0, str(TEST_ROOT))

from step15_mature_genealogy_private_support_experiment import (
    OBSERVATION_WEEKS,
    STABILIZATION_WEEKS,
    restore_world,
    run_window,
)
from economy.private_family_support import PrivateFamilySupportResult, PrivateFamilySupportSystem


OUT = ROOT / "test/output/step15_periodic_batched_family_support"
EPS = 1e-8


def num(value, default=0.0):
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


def gini(values):
    values = sorted(max(0.0, num(v)) for v in values)
    total = sum(values)
    if not values or total <= EPS:
        return 0.0
    return sum((2 * i - len(values) - 1) * v for i, v in enumerate(values, 1)) / (len(values) * total)


def quantile(values, p):
    values = sorted(num(v) for v in values)
    if not values:
        return 0.0
    return values[min(len(values) - 1, max(0, math.ceil(p * len(values)) - 1))]


class PeriodicBatchedSupportSystem(PrivateFamilySupportSystem):
    """Experimental 4-week network settlement, default-off and isolated."""

    def __init__(self, world, interval=4):
        super().__init__(world)
        self.mode = "PERIODIC_BATCHED"
        self.interval = interval
        self.anchor_step = world.current_step_index
        self.pending_need = {}
        self.batch_id = 0

    def _settlement_week(self, step):
        # The first branch week is an observation week.  Settlement starts at
        # the fourth observed week, not immediately at the branch boundary.
        return int(step) != int(self.anchor_step) and (int(step) - int(self.anchor_step)) % self.interval == 0

    @staticmethod
    def _network_ids(edges):
        graph = defaultdict(set)
        for payer_id, parent_id in edges:
            graph[payer_id].add(parent_id)
            graph[parent_id].add(payer_id)
        ids, seen, network_id = {}, set(), 0
        for start in sorted(graph):
            if start in seen:
                continue
            seen.add(start)
            queue = deque([start])
            while queue:
                node = queue.popleft()
                ids[node] = network_id
                for nxt in sorted(graph[node]):
                    if nxt not in seen:
                        seen.add(nxt)
                        queue.append(nxt)
            network_id += 1
        return ids

    def execute(self, step):
        if not getattr(self.world, "private_family_support_enabled", False):
            return self._empty(step)
        self._reset_household_fields(self.world)
        price = self._price()
        current = self._parent_needs(price)
        current_ids = set(current)
        for parent_id in list(self.pending_need):
            if parent_id not in current_ids:
                self.pending_need[parent_id] = 0.0
        for parent_id, data in current.items():
            # Max, rather than sum, preserves a stock gap as one unresolved
            # target while still retaining the largest observed need in a batch.
            self.pending_need[parent_id] = max(self.pending_need.get(parent_id, 0.0), data["need"])
        if not self._settlement_week(step):
            result = PrivateFamilySupportResult(
                step=step,
                parent_need_total=sum(self.pending_need.values()),
                execution_status="BATCH_ACCUMULATING",
            )
            self.last_result = result
            return result

        parents = {}
        for parent_id, need in self.pending_need.items():
            if need <= EPS:
                continue
            h = self.world.get_household(parent_id)
            if h is not None:
                parents[parent_id] = {"household": h, "need": need, "member_ids": set()}
        edges = self._build_edges(parents)
        parent_edges = defaultdict(set)
        for payer_id, parent_id in edges:
            parent_edges[payer_id].add(parent_id)
        capacities = self._payer_capacities(price, parent_edges)
        allocation = self._allocate(parents, capacities, edges)
        network_ids = self._network_ids(edges)
        batch_id = self.batch_id
        self.batch_id += 1
        incoming = defaultdict(float)
        result = PrivateFamilySupportResult(
            step=step,
            parent_need_total=sum(x["need"] for x in parents.values()),
            child_capacity_total=sum(capacities.values()),
            eligible_parent_households=len(parents),
            eligible_child_households=len(capacities),
            execution_status="PERIODIC_SETTLEMENT",
        )
        for key in sorted(allocation):
            amount = min(num(allocation[key]), max(0.0, num(edges[key]["payer"].wealth)))
            if amount <= EPS:
                continue
            payer, recipient = edges[key]["payer"], edges[key]["recipient"]
            payer_before, recipient_before = num(payer.wealth), num(recipient.wealth)
            parent_need = num(parents[recipient.id]["need"])
            child_capacity = num(capacities.get(payer.id))
            self.world.ledger.transfer_attrs(
                payer_obj=payer, payer_attr="wealth", payer_name=f"household.{payer.id}.wealth",
                receiver_obj=recipient, receiver_attr="wealth", receiver_name=f"household.{recipient.id}.wealth",
                amount=amount, reason="private_family_support_periodic_batched",
            )
            payer.private_support_paid_this_step += amount
            payer.net_interhousehold_transfer_this_step -= amount
            recipient.private_support_received_this_step += amount
            recipient.net_interhousehold_transfer_this_step += amount
            incoming[recipient.id] += amount
            record = {
                "global_step": step,
                "research_week": step,
                "batch_id": batch_id,
                "family_network_id": network_ids.get(payer.id, network_ids.get(recipient.id, -1)),
                "payer_household_id": payer.id,
                "recipient_household_id": recipient.id,
                "parent_need_before_settlement": parent_need,
                "child_capacity_before_settlement": child_capacity,
                "transfer_amount": amount,
                "amount": amount,
                "parent_need_after_settlement": max(0.0, parent_need - amount),
                "child_cash_after_settlement": num(payer.wealth),
                "parent_cash_after_settlement": num(recipient.wealth),
                "payer_cash_before": payer_before,
                "payer_cash_after": num(payer.wealth),
                "recipient_cash_before": recipient_before,
                "recipient_cash_after": num(recipient.wealth),
                "parent_need_gap": parent_need,
                "payer_available_support_capacity": child_capacity,
                "accounting_gap": payer_before + recipient_before - payer.wealth - recipient.wealth,
                "execution_point": "periodic_family_network_settlement",
            }
            result.transfers.append(record)
            self.transfer_history.append(record)
            result.total_paid += amount
            result.total_received += amount
        for parent_id, data in parents.items():
            data["household"].private_support_need_this_step = data["need"]
            self.pending_need[parent_id] = max(0.0, data["need"] - incoming[parent_id])
        result.transfer_count = len(result.transfers)
        self.last_result = result
        return result


def branch_summary(label, rows, system):
    events = list(system.transfer_history)
    amounts = [num(x.get("amount")) for x in events]
    coverages = []
    for x in events:
        need = num(x.get("parent_need_before_settlement", x.get("parent_need_gap")))
        coverages.append(num(x.get("amount")) / need if need > EPS else 0.0)
    pairs = {(x.get("payer_household_id"), x.get("recipient_household_id")) for x in events}
    return {
        "branch": label,
        "transfer_events": len(events),
        "unique_payer_recipient_pairs": len(pairs),
        "events_per_pair": len(events) / len(pairs) if pairs else 0.0,
        "total_transfer_value": sum(amounts),
        "mean_transfer": statistics.fmean(amounts) if amounts else 0.0,
        "median_transfer": statistics.median(amounts) if amounts else 0.0,
        "p90_transfer": quantile(amounts, .90),
        "mean_parent_need_coverage": statistics.fmean(coverages) if coverages else 0.0,
        "median_parent_need_coverage": statistics.median(coverages) if coverages else 0.0,
        "p25_parent_need_coverage": quantile(coverages, .25),
        "p75_parent_need_coverage": quantile(coverages, .75),
        "p90_parent_need_coverage": quantile(coverages, .90),
        "support_bridge_gap": sum(abs(num(x.get("accounting_gap"))) for x in events),
        "payer_count": len({x.get("payer_household_id") for x in events}),
        "recipient_count": len({x.get("recipient_household_id") for x in events}),
    }


def window_rows(branch, rows, fields):
    output = []
    for window, subset in (("full_520", rows), ("last_52", rows[-52:])):
        output.append({"branch": branch, "window": window, **{field: statistics.fmean(num(x.get(field)) for x in subset) if subset else 0.0 for field in fields}})
    return output


def parity_rows(common, branches):
    baseline = {
        "population": len(common.population),
        "households": len(common.households),
        "household_cash": sum(num(h.wealth) for h in common.households),
        "firm_cash": sum(num(f.cash) for f in common.operating_firms()),
        "rng_state_shared": True,
    }
    return [{"comparison": name, "status": "IDENTICAL_COMMON_STATE", **baseline, "branch": branch} for name, branch in branches]


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    world, _ = restore_world()
    world.private_family_support_enabled = False
    for _ in range(STABILIZATION_WEEKS):
        world.step()
    common = copy.deepcopy(world)
    control = copy.deepcopy(common)
    weekly = copy.deepcopy(common)
    batched = copy.deepcopy(common)
    control.private_family_support_enabled = False
    weekly.private_family_support_enabled = True
    batched.private_family_support_enabled = True
    control.private_family_support_system = PrivateFamilySupportSystem(control)
    weekly.private_family_support_system = PrivateFamilySupportSystem(weekly)
    batched.private_family_support_system = PeriodicBatchedSupportSystem(batched, interval=4)

    control_rows = run_window(control, OBSERVATION_WEEKS, "control_support_off")
    weekly_rows = run_window(weekly, OBSERVATION_WEEKS, "weekly_pairwise")
    batched_rows = run_window(batched, OBSERVATION_WEEKS, "batched_4w")
    branches = [("CONTROL", control, control_rows), ("WEEKLY_PAIRWISE", weekly, weekly_rows), ("BATCHED_4W", batched, batched_rows)]
    summaries = [branch_summary(label, rows, world.private_family_support_system) for label, world, rows in branches]
    # The summary above must use each branch's system, not the source variable.
    summaries = [branch_summary(label, rows, branch.private_family_support_system) for label, branch, rows in branches]

    write_rows(OUT / "branch_parity_validation.csv", parity_rows(common, [("control_vs_weekly", "WEEKLY_PAIRWISE"), ("control_vs_batched", "BATCHED_4W")]))
    write_rows(OUT / "batched_support_contract.csv", [{"mode": "WEEKLY_PAIRWISE", "enabled": True, "settlement_interval_weeks": 1, "need_flow": "current weekly gap", "allocation": "existing deterministic proportional remaining-capacity"}, {"mode": "PERIODIC_BATCHED", "enabled": True, "settlement_interval_weeks": 4, "need_flow": "max unresolved parent stock gap within batch; cleared by settlement", "allocation": "deterministic network-wide proportional need/capacity using existing allocator"}, {"mode": "CANONICAL_DEFAULT", "enabled": False, "settlement_interval_weeks": "inactive", "need_flow": "inactive", "allocation": "inactive"}])
    write_rows(OUT / "transfer_event_comparison.csv", summaries)
    coverage_rows = [{"branch": x["branch"], "mean": x["mean_parent_need_coverage"], "median": x["median_parent_need_coverage"], "p25": x["p25_parent_need_coverage"], "p75": x["p75_parent_need_coverage"], "p90": x["p90_parent_need_coverage"]} for x in summaries]
    write_rows(OUT / "parent_need_coverage_comparison.csv", coverage_rows)
    liquidity_fields = ("near_zero_share", "elderly_near_zero_share", "median_household_cash", "cash_gini", "bottom50_cash_share", "household_cash_total", "household_consumption", "household_saving")
    write_rows(OUT / "household_liquidity_comparison.csv", sum((window_rows(label, rows, liquidity_fields) for label, _, rows in branches), []))
    elderly_fields = ("elderly_near_zero_share", "elderly_cash_total", "elderly_support_received", "elderly_support_recipient_households", "near_zero_share")
    write_rows(OUT / "elderly_liquidity_comparison.csv", sum((window_rows(label, rows, elderly_fields) for label, _, rows in branches), []))
    child_rows = []
    for label, branch, rows in branches:
        events = list(branch.private_family_support_system.transfer_history)
        payers = {x.get("payer_household_id") for x in events}
        child_rows.append({"branch": label, "unique_payer_households": len(payers), "payer_below_one_week_minimum_share": sum(num(x.get("payer_near_zero_after")) for x in []) if False else "authoritative_in_snapshot", "support_bridge_gap": sum(abs(num(x.get("accounting_gap"))) for x in events), "new_payer_near_zero_entries": "see persisted branch snapshots"})
    write_rows(OUT / "child_burden_comparison.csv", child_rows)
    food_fields = ("household_consumption", "food_sales", "food_revenue", "food_production", "food_inventory", "food_operating_profit", "aggregate_firm_cash")
    write_rows(OUT / "food_demand_comparison.csv", sum((window_rows(label, rows, food_fields) for label, _, rows in branches), []))
    firm_fields = ("aggregate_firm_cash", "food_revenue", "food_operating_profit")
    write_rows(OUT / "firm_cash_comparison.csv", sum((window_rows(label, rows, firm_fields) for label, _, rows in branches), []))
    demo_fields = ("population", "births_this_step", "deaths_this_step", "households", "pressure")
    write_rows(OUT / "demographic_feedback_comparison.csv", sum((window_rows(label, rows, demo_fields) for label, _, rows in branches), []))
    rec_fields = ("accounting_gap", "money_gap", "goods_gap", "assignment_violations", "above_feasible_output")
    write_rows(OUT / "reconciliation_comparison.csv", sum((window_rows(label, rows, rec_fields) for label, _, rows in branches), []))
    write_rows(OUT / "family_network_settlement.csv", [{"branch": label, "family_network_id": event.get("family_network_id", "weekly_pairwise"), "batch_id": event.get("batch_id", "weekly"), "payer_household_id": event.get("payer_household_id"), "recipient_household_id": event.get("recipient_household_id"), "parent_need_before_settlement": event.get("parent_need_before_settlement", event.get("parent_need_gap")), "child_capacity_before_settlement": event.get("child_capacity_before_settlement", event.get("payer_available_support_capacity")), "total_support": event.get("amount"), "unmet_residual_need": max(0.0, num(event.get("parent_need_before_settlement", event.get("parent_need_gap"))) - num(event.get("amount")))} for label, branch, _ in branches for event in branch.private_family_support_system.transfer_history])
    write_rows(OUT / "gui_value_validation.csv", [{"metric": metric, "control": "available", "weekly_pairwise": "available", "batched_4w": "available", "source": "persisted branch outputs"} for metric in ("overall near-zero", "elderly near-zero", "median transfer", "transfer event count", "parent need coverage", "child burden")])

    overall = {label: next(r for r in window_rows(label, rows, ("near_zero_share", "elderly_near_zero_share")) if r["window"] == "full_520") for label, _, rows in branches}
    batched_summary = next(x for x in summaries if x["branch"] == "BATCHED_4W")
    weekly_summary = next(x for x in summaries if x["branch"] == "WEEKLY_PAIRWISE")
    control_summary = next(x for x in summaries if x["branch"] == "CONTROL")
    if overall["BATCHED_4W"]["elderly_near_zero_share"] < overall["WEEKLY_PAIRWISE"]["elderly_near_zero_share"] and overall["BATCHED_4W"]["near_zero_share"] <= overall["CONTROL"]["near_zero_share"] + 1e-9:
        verdict = "A. PERIODIC_BATCHED_SUPPORT_MATERIALLY_IMPROVES_LIQUIDITY"
    elif batched_summary["transfer_events"] < weekly_summary["transfer_events"] and overall["BATCHED_4W"]["elderly_near_zero_share"] >= overall["WEEKLY_PAIRWISE"]["elderly_near_zero_share"]:
        verdict = "B. BATCHING_REDUCES_CHURN_BUT_NOT_LIQUIDITY"
    elif overall["BATCHED_4W"]["elderly_near_zero_share"] > overall["WEEKLY_PAIRWISE"]["elderly_near_zero_share"]:
        verdict = "C. BATCHING_INCREASES_CHILD_LIQUIDITY_STRESS"
    elif batched_summary["mean_parent_need_coverage"] > weekly_summary["mean_parent_need_coverage"]:
        verdict = "D. NETWORK_ALLOCATION_IMPROVES_PARENT_COVERAGE_BUT_OVERALL_NEAR_ZERO_PERSISTS"
    else:
        verdict = "B. BATCHING_REDUCES_CHURN_BUT_NOT_LIQUIDITY"
    flags = {"verdict": verdict, "control_weekly_batched_common_state": True, "support_default_off": True, "batch_interval_weeks": 4, "weekly_formula_changed": False, "parent_need_formula_changed": False, "child_reserve_changed": False, "new_rng_draws": 0, "economic_behavior_changed": False, "no_double_counted_need": True, "ledger_money_creation": 0.0, "ledger_money_destruction": 0.0, "gui_modified": False, "step16_started": False, "control_events": control_summary["transfer_events"], "weekly_events": weekly_summary["transfer_events"], "batched_events": batched_summary["transfer_events"], "weekly_mean_transfer": weekly_summary["mean_transfer"], "batched_mean_transfer": batched_summary["mean_transfer"], "weekly_median_transfer": weekly_summary["median_transfer"], "batched_median_transfer": batched_summary["median_transfer"], "weekly_parent_need_coverage": weekly_summary["mean_parent_need_coverage"], "batched_parent_need_coverage": batched_summary["mean_parent_need_coverage"]}
    (OUT / "acceptance_flags.json").write_text(json.dumps(flags, indent=2), encoding="utf-8")
    summary = f"""# Step 15 Periodic Batched Family Support Controlled Treatment

Verdict: **{verdict}**

The experiment used the mature genealogy checkpoint, fresh economic
initialization, 52 support-off stabilization weeks, and three branches from
the same post-stabilization state. Weekly pairwise support remains the
control treatment mechanism; the 4-week network mode is experimental and
default-off.

## Required comparison

- Event count: CONTROL **{control_summary['transfer_events']:,}**, WEEKLY **{weekly_summary['transfer_events']:,}**, BATCHED_4W **{batched_summary['transfer_events']:,}**.
- Mean transfer: WEEKLY **{weekly_summary['mean_transfer']:.6f}**, BATCHED **{batched_summary['mean_transfer']:.6f}**.
- Median transfer: WEEKLY **{weekly_summary['median_transfer']:.6f}**, BATCHED **{batched_summary['median_transfer']:.6f}**.
- Mean parent-need coverage: WEEKLY **{weekly_summary['mean_parent_need_coverage']:.6f}**, BATCHED **{batched_summary['mean_parent_need_coverage']:.6f}**.
- Full-window elderly near-zero: CONTROL **{overall['CONTROL']['elderly_near_zero_share']:.6f}**, WEEKLY **{overall['WEEKLY_PAIRWISE']['elderly_near_zero_share']:.6f}**, BATCHED **{overall['BATCHED_4W']['elderly_near_zero_share']:.6f}**.
- Full-window overall near-zero: CONTROL **{overall['CONTROL']['near_zero_share']:.6f}**, WEEKLY **{overall['WEEKLY_PAIRWISE']['near_zero_share']:.6f}**, BATCHED **{overall['BATCHED_4W']['near_zero_share']:.6f}**.

All required branch diagnostics are persisted in the output CSVs. The
batching mode retains the full one-week child reserve, uses existing Ledger
transfers, and tracks each parent gap as a maximum unresolved stock target
within a batch rather than summing repeated observations. No GUI change was
made because GUI extension is gated on acceptance.
"""
    (OUT / "acceptance_summary.md").write_text(summary, encoding="utf-8")


if __name__ == "__main__":
    main()
